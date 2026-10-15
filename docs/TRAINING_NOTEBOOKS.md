# Notebook training BifocalGemma

Template bersama menghasilkan tiga notebook standalone: [simetris](../notebooks/bifocalgemma_train_symmetric.py),
[asimetris](../notebooks/bifocalgemma_train_asymmetric.py), dan
[baseline decoder-only](../notebooks/bifocalgemma_train_baseline.py).
Masing-masing memiliki **21 cell, 20 isi cell identik**. Perbedaan hanya tiga baris
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

Setiap model menjalankan **base AdamW → optimized GrokOrScale LoRA** dari
inisialisasi pretrained yang sama: empat hasil per notebook, 12 total. Kontrak,
angka bersama, pemilihan ide v8, dan trade-off lengkap ada di
[Training recipe comparison](TRAINING_RECIPE_COMPARISON.md).

## Peta cell

Nomor adalah posisi cell, dimulai dari nol. Label UI memakai subbagian tersendiri
untuk token input/apply dan policy.

| Cell | Isi dan tanggung jawab |
| --- | --- |
| 0 | Dependency setup: uv pip reinstall, receipt dan restart guard |
| 1 | Pengantar dan default |
| 2 | Control center: urutan model, data, limits, batch, epoch, output, upload |
| 3 | Environment, ekstraksi paket bersama, resource/runtime guard |
| 4 | Form HF token, submit eksplisit |
| 5 | Apply/verifikasi token; token tidak dicetak ke log |
| 6 | Config native Gemma dan dimensi |
| 7 | Batch, masking label, shifted token loss |
| 8 | Reference attention dan backend training bersama |
| 9 | Model builder: encoder, merged decoder, projection/tying |
| 10 | Checkpoint loader dan attention LoRA |
| 11 | Training policy: profil ketiga keluarga, parameter groups, LR/warm-up |
| 12 | Kontrak contoh, tokenisasi dan dataset helpers |
| 13 | Download/verifikasi koleksi HF beku dan tokenizer assets |
| 14 | Training/checkpoint helpers |
| 15 | Guard dan preflight plan |
| 16 | Data plan; tidak menyimpan token seluruh model dalam Python lists |
| 17 | Sequential pipeline, dashboard dan resume |
| 18 | Hasil per model |
| 19 | Ringkasan upload yang sudah diverifikasi |
| 20 | Entrypoint/test notebook |

## Policy dan objective

Encoder menggunakan base checkpoint, decoder IT checkpoint. Ordinary embeddings
bersama hanya pada simetris; decoder input/LM head tied internal, PLE antar
komponen terpisah. Asimetris memakai learned encoder→decoder alignment;
simetris memakai Identity. Baseline hanya mempunyai decoder.

| Group | Simetris | Asimetris | Baseline | LR / warm-up |
| --- | --- | --- | --- | --- |
| encoder | Attention LoRA | Attention LoRA | Tidak ada | 2e-5 / 3% updates; cosine → 2e-6 |
| decoder | Attention LoRA | Attention LoRA | Attention LoRA | 2e-5 / 3% updates; cosine → 2e-6 |
| alignment | Tidak ada | Learned projection | Tidak ada | 2e-5 / 3% updates; cosine → 2e-6 |

Default rank16/alpha32, ordinary embeddings dan PLE frozen, weight decay0,
clipping1.0. Base memakai AdamW `foreach=False`; optimized memakai EMA gradien,
Nesterov/ortogonalisasi, dan trust ratio. Seluruh trainable termasuk alignment
asimetris memakai FP32. Tiap trainable parameter masuk tepat satu group;
aliases tidak diduplikasi. Cell policy memuat `TRAINING_RECIPES` dari factory angka bersama.
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
Batch2, accumulation2, validation maksimum2, EPOCHS1, MAX_STEPS0 berarti satu epoch penuh atas split beku;
dev loss setiap25% update dan checkpoint setiap10% update ditambah final.
Dengan11.102 updates: interval validation2.776 dan checkpoint1.111. Dua checkpoint terakhir
dipertahankan per model/recipe/run. Token cache int32 mmap mempertahankan ID asli sehingga
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

Runtime notebook run baru: Python3.12/3.13, Torch2.14.1 CUDA13.0, Transformers5.19.0,
marimo0.25.1; dependency lain terpin di header. Cell setup menginstal ulang sekali
per environment receipt dan meminta restart jika native library sudah ter-load.
Notebook mengunci full commit SHA data. Perubahan cohort memerlukan RUN_NAME baru.
Resume memverifikasi optimizer groups, policy, tokenizer, data, RNG dan cursor.

Pyrefly memakai interpreter ai yang ditentukan di pyproject.toml. Pyrefly dan ty
memeriksa src, tools dan tests; reactive notebook graph diperiksa oleh marimo.
Type stubs YAML/psutil adalah dependency dev dan tidak mengubah runtime training.

## Upload ulang dan dependency reinstall

Cell pertama memakai standard library dan `uv pip --python sys.executable` agar
instalasi menargetkan kernel notebook. Default `FORCE_REINSTALL=True` memasang ulang
Torch2.14.1 dari index cu130 dan dependency lain dari PyPI, satu kali per environment
matching receipt. Versi stabil terbaru ini diverifikasi pada15 Oktober2026 melalui
[metadata PyPI](https://pypi.org/pypi/torch/json) dan
[index wheel CUDA13.0](https://download.pytorch.org/whl/cu130/torch/).
Run sekarang dimulai dari awal; pin2.11.0 untuk resume pilot lama sudah diganti.

Receipt berada di `outputs/dependency_setup/`. Jika Torch/NumPy/library native sudah
ter-load saat instalasi, cell berhenti dan meminta restart kernel. Menjalankan cell
lagi pada kernel yang sama tidak melewati guard. Setelah restart, receipt dan
versi terpasang diverifikasi sehingga tidak terjadi loop reinstall.

Conda base ditolak sebelum instalasi. Python3.12/3.13 diterima; dukungan penuh
Python3.12 belum dibuktikan melalui regresi pretrained. Cell environment memeriksa
Torch loaded/installed tepat `2.14.1+cu130` sebelum training. Run baru memakai
suffix `-torch2141`; receipt dependency dengan pin lama tidak melewati reinstall.
Untuk resume run baru, nama run, data dan policy tetap sama; checkpoint dari
runtime berbeda ditolak.

Untuk ekspor Jupyter tanpa menjalankan notebook:

```powershell
marimo export ipynb notebooks/bifocalgemma_train_symmetric.py --sort top-down -o notebooks/bifocalgemma_train_symmetric.ipynb --force
```

Cell setup berada paling awal pada ekspor. Ekspor hanya mengubah format; UI token
marimo pada Jupyter/Modal tetap memerlukan penyesuaian jika interaksinya tidak bekerja.

## Scheduler dan batch run berikutnya

Lihat [audit dan recipe baru](TRAINING_NEXT_RUN.md). Run baru memakai cosine decay,
preflight batch train/dev dan compatibility horizon; benchmark tetap single-case.
