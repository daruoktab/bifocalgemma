# Dua Notebook SFT dengan Cell yang Sama

File untuk diunggah satu per satu ke Molab:

- [bifocalgemma_train_symmetric.py](../notebooks/bifocalgemma_train_symmetric.py)
- [bifocalgemma_train_asymmetric.py](../notebooks/bifocalgemma_train_asymmetric.py)

Keduanya **17 cell**, dalam urutan yang sama. **16 isi cell identik persis**.
Hanya tiga baris pada control center berbeda:

| Setting | Simetris | Asimetris |
| --- | --- | --- |
| FAMILY | symmetric | asymmetric |
| Default EXPERIMENT | BifocalGemma-E2B-E2B | BifocalGemma-E2B-E4B |
| OUTPUT_ROOT | outputs/training_symmetric | outputs/training_asymmetric |

Pilihan keluarga simetris: E2B–E2B atau E4B–E4B. Asimetris: E2B–E4B atau E4B–E2B.
Semua model memakai encoder base / decoder IT asli. Word tying dan H projection
mengikuti spec runtime dari kode model yang **sama**, sehingga tidak memerlukan
cell arsitektur berbeda. Simetris: tied ordinary words, Identity projection.
Asimetris: untied encoder words, learned H projection. PLE tetap separate.

## Cara memulai

1. Unggah salah satu file .py ke Molab; file standalone tidak perlu checkout repo.
2. Apply HF token pada cell input/apply yang terpisah, atau gunakan akses publik
   anonim bila tersedia. Read token cukup; tidak ada upload model/checkpoint otomatis.
3. Pilih EXPERIMENT dari keluarga notebook; ubah RUN_NAME untuk eksperimen baru.
4. Periksa control center, lalu ubah **RUN_TRAINING=True**. Default False menjaga
   pembukaan notebook/checks tidak langsung mengunduh atau melatih model besar.
5. Amati preprocessing count, training logs/dev CE, lalu lokasi checkpoint/report.

Satu notebook melatih **satu pasangan pada satu run**. Untuk pasangan kedua,
ubah EXPERIMENT dan jalankan run baru. Menjalankan dua notebook di GPU yang sama
memakai resource bersama; preflight membaca free VRAM aktual, bukan mengasumsikan
kapasitas 96 GB selalu kosong. Setting training default berbeda dari micro probe
512-token terdahulu; peak di report run baru menjadi bukti yang berlaku.

## Peta cell identik

| Posisi | Peran |
| --- | --- |
| 0 | Intro |
| 1 | Control center — satu-satunya isi cell yang berbeda |
| 2 | Environment, versi, GPU dan precision |
| 3–4 | Input HF password / Apply-whoami |
| 5 | Audited config dan experiment registry |
| 6 | Batch/collator/patch CE |
| 7 | Native merged attention |
| 8 | Native model builder |
| 9 | Pretrained transfer dan attention LoRA |
| 10 | Data conversion, split dan tokenization helpers |
| 11 | Training/checkpoint/resume helpers |
| 12 | Preflight metadata/VRAM guard |
| 13 | Data preparation |
| 14 | Model load + SFT dalam satu rerun boundary |
| 15 | Cheap result summary |
| 16 | Pytest-only cell |

Perubahan UI summary tidak memerlukan rerun model/training. Model dimiliki oleh
training cell dan dilepas setelah selesai. Jangan mengubah hyperparameters saat
run aktif kecuali memang hendak memulai ulang rerun boundary.

## Data dan objective

Default **EditPackFT** revision `067b4c882f44eca19329fedd275438fc3c060f3f`, file
`data/train-00000-of-00001.parquet`. Opsi **CommitPackFT Python** revision
`fc56fe33c030c6daa414c2b112c932b8eed085e6`, file `data/python/data.jsonl`.
Pilihan local_jsonl menerima schema old_file/new_file, old_contents/new_contents,
instruction, commit/repos/license. Subject dipakai oleh pilihan CommitPackFT Python.

Input hanya instruction + kode before. Target unified diff diturunkan dari before
→ after; field content/ndiff tidak dipakai sebagai input. Existing-file modifications
saja; rename/add/delete/hidden metadata/quoted path ditolak. Diff tanpa final newline
mempunyai sentinel Git yang benar. Tidak ada execution environment, VM, Docker, RL,
reward model atau test-driven rollout pada training.

Dedup before/after pairs sebelum split. Train/dev dibagi menurut **commit group**,
bukan repo-disjoint; dataset ini tidak membuktikan generalisasi ke repository baru.
Raw/sample limits default10000/2000/200, dev fraction0.1, seed17. Token overflow
ditolak dan counted (encoder2048, decoder512); tidak ada silent truncation.
Data manifest memuat source revisions, rows/commits/repos/licenses, instruction/pair
hashes, tokenizer sources/hashes, tokenization fingerprint dan filter counts.

Token-weighted patch CE hanya pada target/closing turn, shift sekali, prefix/pad -100.
Gradient accumulation memakai proporsi supervised tokens per window supaya target
pendek tidak mendapat bobot berlebihan. Partial epoch window memakai denominator
aktual. Tidak ada loss tambahan atau knowledge distillation.

## Setting pilot dan dependency

Attention LoRA rank16 / alpha32, frozen BF16 base words/PLE, FP32 A/B, trainable
asymmetric H projection. AdamW foreach=False, weight_decay0, lr2e-5, warmup5 steps
lalu constant LR, clipping1.0, microbatch1, accumulation4, maksimum100 optimizer
steps / 1 epoch. Dev CE dan checkpoint setiap25 steps plus final checkpoint.
Ini default pilot untuk diedit setelah profiling, bukan recipe optimum yang terbukti.

PEP723 memakai runtime Molab yang berhasil: Torch2.11.0, Transformers5.18.0,
marimo0.25.1, safetensors0.8.0, pyarrow25.0.0 dan pytest8.4.2, Python3.13.
Actual runtime versions masuk report/checkpoint compatibility. Tidak ada pip/reinstall
Torch dalam cell. Model source revisions dan text mapping sama dengan
[pretrained probe](PRETRAINED_GPU_PROBE.md).

## Artifacts dan resume

Directory: `OUTPUT_ROOT/EXPERIMENT/RUN_NAME/`.

- data_manifest.json / run_config.json ditulis sebelum optimization.
- history.jsonl mencatat train/dev loss, LR, grad norm, token count dan CUDA peak.
- training_report.json menyimpan status/checkpoints/source/runtime.
- checkpoint-NNNNNN berisi adapters.safetensors, trainer_state.pt, manifest.json,
  dan READY.json dengan checksums. Adapter-only save mencakup seluruh trainable
  LoRA parameters serta learned asymmetric projection.

Checkpoint dibuat ke staging directory lalu rename setelah lengkap. Existing
checkpoint/run artifacts tidak ditimpa. Resume: isi **RESUME_FROM** dengan directory
checkpoint lengkap dan gunakan sumber/model/data/config yang sama. Pretrained base
weights direkonstruksi, ties/LoRA dipasang, lalu trainable weights + optimizer +
CPU/CUDA/Python RNG + sampler order/cursor dipulihkan. Checkpoints hanya pada
optimizer boundary; tidak ada pending gradient accumulation yang hilang.

Compatibility mencakup experiment/family/source revisions, data/tokenizer fingerprint,
rank, batch/accumulation, LR/warmup/clipping, seed dan runtime version. MAX_STEPS/EPOCHS
boleh diperpanjang. Wrong compatibility, incomplete/checksum-bad checkpoint, key/shape
mismatch atau sampler invalid ditolak. Keberhasilan resume telah diuji pada native
**tiny** simetris/asimetris terhadap run uninterrupted; big-model SFT kedua notebook
belum dijalankan dari sesi lokal. Dev CE bukan functional repair benchmark.

## Maintenance: sumber bersama, bukan edit dua file manual

[Generator](../tools/generate_training_notebooks.py) membaca
[template cell](../tools/templates/training_notebook.py.txt) serta shared helpers:
[data](../src/bifocalgemma/data/offline_sft.py) dan
[training/checkpoint](../src/bifocalgemma/training/offline_sft.py).

```powershell
python tools/generate_training_notebooks.py
python tools/generate_training_notebooks.py --check
python -m marimo check --strict notebooks/bifocalgemma_train_symmetric.py notebooks/bifocalgemma_train_asymmetric.py
python -m pytest notebooks/bifocalgemma_train_symmetric.py notebooks/bifocalgemma_train_asymmetric.py -q
```

Ubah cell umum di template atau helper, lalu regenerate keduanya. Untuk perbedaan
default keluarga, edit mapping VARIANTS pada generator/control substitution. Hasil
formatter deterministik dan --check gagal bila file generated drift. File upload
standalone tetap bisa diedit di Molab; saat pairing live, gunakan runtime cm API,
kemudian sinkronkan perubahan umum ke template sebelum regeneration lokal.

Validasi lokal: dua graph disabled, enam notebook pytest tests, tests diff/split,
token-weighted accumulation, checkpoint checksums/compatibility/run overwrite, serta
native tiny LoRA resume pada kedua keluarga. Model/data asli belum dilatih oleh
notebook baru ini; jalankan pilot di Molab dan catat hasilnya terpisah.
