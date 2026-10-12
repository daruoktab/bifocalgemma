# Notebook training BifocalGemma

Template bersama menghasilkan tiga notebook standalone: [simetris](../notebooks/bifocalgemma_train_symmetric.py),
[asimetris](../notebooks/bifocalgemma_train_asymmetric.py), dan
[baseline decoder-only](../notebooks/bifocalgemma_train_baseline.py).
Masing-masing memiliki **20 cell, 19 isi cell identik**. Perbedaan hanya tiga baris
control center: `FAMILY`, `EXPERIMENT`, dan `OUTPUT_ROOT`.

Panduan operasional dan kontrak resume lengkap ada di
[Sequential training](SEQUENTIAL_TRAINING.md). Setiap model menyelesaikan training,
upload, benchmark, dan upload hasil sebelum model berikutnya dimuat. Satu model
resident pada satu waktu; keluarga pasangan tidak melatih baseline lagi.

| Notebook | Urutan default |
| --- | --- |
| Simetris | E2B–E2B → E4B–E4B |
| Asimetris | E2B–E4B → E4B–E2B |
| Baseline | E2B-it → E4B-it |

## Peta cell

Nomor adalah posisi cell, dimulai dari nol. Label UI memakai subbagian tersendiri
untuk token input/apply dan policy.

| Cell | Isi dan tanggung jawab |
| --- | --- |
| 0 | Pengantar dan default |
| 1 | Control center: urutan model, data, limits, batch, epoch, output, upload |
| 2 | Environment, ekstraksi paket bersama, resource/runtime guard |
| 3 | Form HF token, submit eksplisit |
| 4 | Apply/verifikasi token; token tidak dicetak ke log |
| 5 | Config native Gemma dan dimensi |
| 6 | Batch, masking label, shifted token loss |
| 7 | Reference attention dan backend training bersama |
| 8 | Model builder: encoder, merged decoder, projection/tying |
| 9 | Checkpoint loader dan attention LoRA |
| 10 | Training policy: profil ketiga keluarga, parameter groups, LR/warm-up |
| 11 | Kontrak contoh, tokenisasi dan dataset helpers |
| 12 | Download/verifikasi koleksi HF beku dan tokenizer assets |
| 13 | Training/checkpoint helpers |
| 14 | Guard dan preflight plan |
| 15 | Data plan; tidak menyimpan token seluruh model dalam Python lists |
| 16 | Sequential pipeline, dashboard dan resume |
| 17 | Hasil per model |
| 18 | Ringkasan upload yang sudah diverifikasi |
| 19 | Entrypoint/test notebook |

## Policy dan objective

Encoder menggunakan base checkpoint, decoder IT checkpoint. Ordinary embeddings
bersama hanya pada simetris; decoder input/LM head tied internal, PLE antar
komponen terpisah. Asimetris memakai learned encoder→decoder alignment;
simetris memakai Identity. Baseline hanya mempunyai decoder.

| Group | Simetris | Asimetris | Baseline | LR / warm-up |
| --- | --- | --- | --- | --- |
| encoder | Attention LoRA | Attention LoRA | Tidak ada | 2e-5 / 5 steps |
| decoder | Attention LoRA | Attention LoRA | Attention LoRA | 2e-5 / 5 steps |
| alignment | Tidak ada | Learned projection | Tidak ada | 2e-5 / 5 steps |

Default rank16/alpha32, ordinary embeddings dan PLE frozen, AdamW `foreach=False`,
weight decay0, clipping1.0. Tiap trainable parameter masuk tepat satu group;
aliases tidak diduplikasi. Knob tersebut berada di cell policy.
Projection-only warm-up dan staged unfreezing belum diaktifkan. Direct SFT tetap
hipotesis adaptasi; belum terbukti setara dengan adaptasi UL2 recipe Google.

Loss adalah token-weighted CE pada target dan closing turn; prefix/pad -100,
shifting sekali. Target mengikuti task_type: diff untuk repair, daftar path JSON
untuk localization, kode hilang untuk completion. Encoder menerima context sebelum
perubahan; baseline menerima context identik di prefix causal decoder.
Tidak ada distillation/RL atau repository VM.

## Panjang, RAM dan default run

Encoder limit **32768**, paired shifted decoder input **8192** termasuk prefix;
baseline total **41472**. Overflow ditolak saat kurasi. Ini batas maksimum,
bukan semua contoh dipad ke panjang itu. QA memakai maksimal32 token jawaban;
tugas lain memakai allowance sampai budget decoder setelah prefix.

RUN_TRAINING, RUN_BENCHMARKS, FUNCTIONAL_BENCHMARKS, AUTO_RESUME,
HF_UPLOAD_ENABLED dan CHECKPOINT_UPLOAD_ENABLED default True.
Batch1, accumulation4, EPOCHS1, MAX_STEPS0 berarti satu epoch penuh atas split beku;
dev loss/checkpoint setiap500 update ditambah final. Dua checkpoint terakhir
dipertahankan per model/run. Token cache int32 mmap mempertahankan ID asli sehingga
runtime RAM sekitar30GiB tidak menampung seluruh token sebagai objek Python.

Backend tiled attention FP32, gradient recomputation dan chunked loss mempertahankan
objective. Preflight bentuk ekstrem berjalan sebelum optimizer step. Ukuran VRAM
pretrained **32K/8K belum diukur**; hasil lama8K/2K tidak menjamin batas baru muat.

## Generate dan verifikasi

Edit template/modul shared lalu regenerate ketiga file. Jangan memelihara
implementasi berbeda dengan edit manual notebook hasil generator.

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
python tools/generate_training_notebooks.py
python tools/generate_training_notebooks.py --check
python -m marimo check --strict notebooks/bifocalgemma_train_symmetric.py
python -m marimo check --strict notebooks/bifocalgemma_train_asymmetric.py
python -m marimo check --strict notebooks/bifocalgemma_train_baseline.py
python -m pytest tests/test_notebook_hf_flow.py tests/test_sequential_stages.py tests/test_sequential_hub.py
pyrefly check
ty check
ruff check --fix
```

PEP723 mengikuti ai/Molab: Python3.13, Torch2.14.1 CUDA13.0, Transformers5.19.0,
marimo0.25.1; dependency lain terpin di header. Cell tidak menginstal ulang Torch.
Notebook mengunci full commit SHA data. Perubahan cohort memerlukan RUN_NAME baru.
Resume memverifikasi optimizer groups, policy, tokenizer, data, RNG dan cursor.

Pyrefly memakai interpreter ai yang ditentukan di pyproject.toml. Pyrefly dan ty
memeriksa src, tools dan tests; reactive notebook graph diperiksa oleh marimo.
Type stubs YAML/psutil adalah dependency dev dan tidak mengubah runtime training.
