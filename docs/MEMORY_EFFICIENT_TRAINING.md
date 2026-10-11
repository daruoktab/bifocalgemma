# Training hemat VRAM dan monitor Molab

Recipe ini dipakai kedua notebook keluarga dan jalur curated SFT repository study.
Tujuannya mengurangi retensi aktivasi tanpa mengubah cohort, context, trainable scope,
atau objective patch CE. Satu pasangan per run; tidak memindahkan aktivasi/PLE ke RAM30GB.

## Setting default

```python
RUN_NAME = "enc8k-dec2k-v2-mem1"
ENCODER_LIMIT = 8192
DECODER_LIMIT = 2048
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
RUN_TRAINING = True
HF_UPLOAD_ENABLED = True
MAX_STEPS = 0
EPOCHS = 1
MEMORY_POLICY = {
    "version": "tiled_recompute_v1",
    "attention_backend": "tiled_eager_fp32",
    "query_chunk_size": 128,
    "loss_chunk_size": 128,
    "gradient_checkpointing": True,
    "ple_offload": False,
    "kv_gradient_accumulation": "float32",
}
```

Word/PLE frozen; LoRA rank16/alpha32, LR/warmup/group sama. Simetris memakai
shared ordinary word embeddings/Identity; asimetris learned projection.
Alignment tetap dilatih pada asimetris. Kedua notebook tetap20 cell/19 identik,
dengan tiga default control center berbeda.

## Implementasi

[Attention](../src/bifocalgemma/models/memory_efficient.py):
native Q/K/V, norm, RoPE, scaling1.0 dan KV owners dipertahankan. Score/probability
diproses per128 query; satu softmax FP32 melihat seluruh keys yang diizinkan.
KV pemilik tetap compact; backward memperluas heads sementara untuk mempertahankan
reduction native. Workspace/head expansion tetap masuk pengukuran GPU. Mask merged dibuat
per query tile: decoder causal/window512, context encoder terlihat di tiap layer.
Global head512 didukung oleh jalur Torch tiled; tidak ada asumsi FlashAttention
mendukung semua head atau fallback math yang tidak dicatat. Native encoder tetap
membuat mask upstream; score/probability yang mahal dibatasi oleh tile.

Backward merecompute tiap tile dan mengakumulasi gradien KV per-head dalam FP32
sebelum satu cast ke dtype backbone dan reduction GQA native. Ini menghindari
pembulatan dari penjumlahan banyak partial gradients BF16. Forward menjaga dtype
score/probability native. Jalur ini untuk first-order SFT, bukan higher-order gradients.
Setiap tile direcompute saat backward. Layer encoder/decoder juga memakai checkpoint
non-reentrant. Mematikan gradient_checkpointing pada policy hanya mematikan checkpoint
layer; tile/loss recompute masih berlaku. Preserving RNG diaktifkan. Config official
attention dropout0; belum mengklaim parity dropout nonzero atau training multimodal.

[Loss](../src/bifocalgemma/training/objectives.py):
forward_loss memproses decoder hidden supervised saja lewat LM head→softcap30→FP32 CE
per128 posisi. Tidak mematerialisasikan logits[B,L,262144] penuh.
Chunk loss dijumlah dan dibagi seluruh supervised tokens. Input/labels telah shifted
oleh collator; loss tidak shift lagi. Prefix hidden tetap diproses decoder.
Train dan seluruh dev memakai forward_loss yang sama. forward/decode_logits untuk
diagnostik/generation tetap tersedia; decode_next_logits menghitung head token terakhir.

Tidak ada quantization, pengurangan rank, encoder detach/no_grad saat joint training,
pemotongan context, skip overflow, sampling dev atau perubahan learning rate.
Kernel/chunking dapat mengubah urutan floating-point; parity dinilai dengan toleransi,
bukan janji bitwise-identical pretrained convergence.

## Preflight sebelum epoch

Setelah model dan adapter dimuat, probe memilih:
1. Contoh dengan encoder terpanjang.
2. Contoh dengan decoder terpanjang.
3. Contoh dengan cost proxy encoder²+decoder×(encoder+decoder) terbesar.
4. Gabungan input terpanjang/output terpanjang sebagai stress shape-only.

Duplikat ID aktual hanya diprobe sekali. Contoh gabungan bukan task repair,
tidak ditambahkan ke train/dev dan tidak dipakai sebagai hasil kualitas.
Forward loss + backward memeriksa finite gradients tanpa optimizer update.
RNG/mode dipulihkan dan gradien probe dibuang. Bobot/cohort tidak berubah.

Probe memakai panjang aktual cohort; maksimum encoder v2 adalah6309, bukan otomatis
8192. Batas sintetis8192/2048 masih perlu GPU stress tambahan jika ingin menjamin
budget context penuh pada data lanjutan. Probe belum menjamin semua pasangan muat96GB.
Full-run guard weights+headroom bukan bukti aktivasi maksimum muat.

Hasil preflight disimpan pada training_report.json; kegagalan sebelum trainer disimpan
di OUTPUT_ROOT/diagnostics/EXPERIMENT/RUN_NAME bersama monitor.html.
Tidak ada retry dengan menurunkan context atau diam-diam melewati contoh yang gagal.

## Widget dan log

Training cell memiliki monitor yang di-update in-place sekitar tiap satu detik,
dengan update paksa pada batas penting. Mirip pola card/progress/loss chart V8,
tetapi memakai mo.output.replace langsung, tanpa IPython/matplotlib/dependency baru.

Ditampilkan:
- Tahap load/download/transfer, adapters, memory preflight, encoder/decoder/loss forward,
  backward, optimizer, validation, checkpoint dan completed/failed.
- Optimizer step/budget, fractional epoch, microbatch/accumulation.
- Train CE, full-dev CE, kurva CE sepanjang run dan held-out progress contoh.
- LR setiap group, norm gradient sebelum clipping, target tokens/second.
- Durasi segmen dan ETA training saja; evaluasi/save/upload belum masuk ETA.
- Panjang encoder/decoder/target dan IDs batch.
- GPU allocated/reserved/free dan cumulative peak allocated/reserved.
- RAM RSS proses dan RAM available sistem. N/A tidak diubah menjadi angka nol.
- Backend, checkpoint policy, ukuran chunk, checkpoint lengkap terakhir.
- Error/type/tahap/batch jika gagal. Tidak mengarang metrik GPU atau skor repair.

Tokenisasi memiliki progress per250 contoh dan ringkasan panjang/jumlah target token.
Unduh/verifikasi dataset memiliki spinner; upload memiliki tahap started/complete/failed
pada cell terpisah. HF password/Apply tetap terpisah; token tidak masuk event log.

Artefak:
| File | Isi |
|---|---|
| history.jsonl | Semua optimizer step, loss, phase timing, LR, source loss, resource samples |
| events.jsonl | Timestamp UTC dan transisi tahap; IDs/length sebelum eksekusi; validasi/save/upload |
| monitor.html | Snapshot dashboard terakhir, termasuk chart; bisa dibuka tanpa server |
| training_report.json | Hanya run yang selesai; full-dev, preflight, resource peaks, checkpoint paths |
| failure.json | Kegagalan/interrupt trainer, traceback, batch dan checkpoint terakhir |
| run_config.json | Data/runtime/policy/optimizer-group/memory-policy compatibility |
| checkpoint-*/READY.json | Marker checksum setelah save selesai, bukan checkpoint parsial |

Source loss diukur tepat ketika microbatch homogeneous sumber. Jika batch>1 mencampur
sumber, dilaporkan mixed_batch, bukan loss palsu per sumber. Kurva UI diringkas maksimum
sekitar240 train points; history.jsonl tetap seluruh step. Training CE bukan bukti
repair correctness. Dev CE sementara diberi label sementara, final memakai semua dev.

Resource sampling dan event file menambah overhead, bukan alokasi model/activations
besar. Tidak membaca atau mengubah live Molab dari agent saat implementasi ini.

## Checkpoint dan restart

Format checkpoint tetapv2 untuk optimizer groups; memory policy masuk compatibility.
Resume dari backend/policy lama ditolak sebelum model/run metadata dimodifikasi.
RUN_NAME baru menghindari menimpa run sebelumnya. Resume eksplisit hanya dari
checkpoint lengkap yang punya READY.json dan policy/runtime/cohort yang sama.
History resume dipulihkan ke checkpoint boundary; logs dari update yang dibatalkan
tidak muncul sebagai step sukses pada chart.

Untuk restart dari awal: unggah notebook terbaru ke runtime baru, Apply HF token,
lalu jalankan. Bundle sumber yang berubah mendeteksi module lama dan meminta restart;
tidak ada hot swap terhadap run aktif. Molab/pretrained training dijalankan pengguna.

## Validasi

Tes CPU mencakup keempat pasangan dan dua baseline: forward logits/loss dan gradient
parity, nonzero LoRA B, tying/PLE, GQA/head256/512, padding/causality/context influence,
window>512, all-masked backward, chunked CE FP32/BF16, frozen/full head gradients,
v2 uninterrupted/resume/mismatch, failure diagnostics, dashboard HTML escaping dan
preflight tidak mengubah weights/RNG/cohort. Notebook family cell parity dan
marimo check strict tetap diperlukan setelah setiap regenerate.

Type checking memakai env ai:
```powershell
ty check --python C:\Users\daru\miniconda3\envs\ai\python.exe src
python tools/generate_training_notebooks.py --check
python tools/generate_repository_notebook.py --check
python -m marimo check --strict notebooks/bifocalgemma_train_symmetric.py notebooks/bifocalgemma_train_asymmetric.py notebooks/bifocalgemma_repository_study.py
```

Angka total penghematan VRAM dan keberhasilan E4B–E4B pada96GB belum diukur lokal.
Default menjaga kualitas/objective, dengan tambahan waktu recompute; bukti repair
tetap memerlukan benchmark setelah training.
