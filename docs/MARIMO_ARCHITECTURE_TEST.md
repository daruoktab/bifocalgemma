# Marimo Architecture Test — Molab / GPU 96 GB

Notebook: [bifocalgemma_architecture_test.py](../notebooks/bifocalgemma_architecture_test.py).
Satu file standalone, tanpa import package BifocalGemma atau checkout tambahan.
Config snapshots dan inti native integration disertakan dalam cell notebook.
Pola control center, cell dependencies, penjelasan tahap, dan laporan mengikuti
contoh Molab pengguna. File referensi pengguna tidak diubah atau dimasukkan ke Git.

## Cara memakai

1. Unggah satu file notebook ke Molab dan buka dengan marimo.
2. Jalankan semua cell. Default melakukan tiny full-depth tests untuk enam model,
   inspeksi dimensi asli melalui meta, dan menulis laporan JSON lokal.
3. Lihat environment cell: nama GPU, VRAM, compute capability, Torch/CUDA dan precision
   dibaca dari runtime. Notebook tidak mengasumsikan Blackwell hanya dari nama platform.
4. Untuk GPU 96 GB, setelah tiny tests lulus, ubah `RUN_FULL_SIZE = True` pada control
   center. Full-size probe memakai satu `EXPERIMENT` yang dipilih, BF16 dan batch kecil.
   Nilai default tetap False agar notebook juga dapat diperiksa di mesin lokal.

```powershell
conda activate ai
python -m marimo check --strict notebooks/bifocalgemma_architecture_test.py
python -m pytest notebooks/bifocalgemma_architecture_test.py -q
python -m marimo edit notebooks/bifocalgemma_architecture_test.py
```

Notebook memuat PEP 723: Python >=3.13,<3.14; marimo==0.25.1,
torch==2.14.1, transformers==5.18.0, pytest==8.4.2. Pin mengikuti runtime/native integration yang
sudah diuji; compatibility Transformers 5.18.0 wajib. Sandbox marimo/Molab dapat
menyiapkan package dependencies dari metadata tersebut. Menjalankan `--no-sandbox`
memakai environment yang sudah tersedia. Tidak ada pip/upgrade/reinstall Torch di
cell, sehingga build CUDA existing tidak diganti ketika notebook dijalankan.
PEFT/Unsloth/TRL/datasets tidak diperlukan untuk tes wiring ini.

## Cell map

| Cell | Isi | Output |
| --- | --- | --- |
| 0 | Pengantar | Tujuan dan batas validasi |
| 1 | Control center | Experiment/device/precision, matrix/full flags, seed, lengths, report path |
| 2 | Environment | Versi dan GPU/VRAM/compute capability aktual |
| 3 | Config native | Dua audited config snapshots dan six-model registry |
| 4 | Meta shapes | Native hidden/depth/ordinary words/PLE dan parameter counts asli |
| 5 | Batch / loss | Synthetic IDs, left padding, satu shift, prefix/pad -100 |
| 6 | Merged attention | Q/self-K RoPE, normalized cross K, native KV owners, joint softmax |
| 7 | Model builder | Native bidirectional encoder, decoder, one H projection, symmetric aliases |
| 8 | Test helpers | Assertions perilaku wiring |
| 8a | Pytest-only cell | Local window/global context, prefix gradients, fully-masked backward |
| 9 | Tiny matrix | PASS rows, synthetic loss, parameter count dan peak allocation |
| 10 | Optional full-size | Single random forward/backward BF16 pada dimensi asli |
| 11 | Summary / JSON | Hasil aktual dan artifact path |

Cell mendapat dependencies melalui argumen dan mengekspor melalui return. Variabel
perhitungan memakai nama yang jelas dan berbeda antar-cell. Bagian tes dipisahkan dari perakitan model sehingga
parameter dapat diganti tanpa mencampur logic/data/training production.

## Yang diuji

Empat encoder/decoder pairs dan dua decoder-only baselines:

- Patch labels di-shift sekali; prefix di-ignore.
- Logits/loss/gradients finite dan gradients nonzero pada synthetic batch.
- Decoder words/head alias; encoder words hanya shared pada simetris; PLE separate.
- Optimizer-facing parameter iterator tidak menduplikasi Parameter aliases.
- Decoder tanpa encoder keys cocok dengan decoder native pada **token valid**.
- Decoder future tokens tidak memengaruhi output token sebelumnya yang valid.
- Context encoder dapat memengaruhi logits; masked encoder keys setara no-context.
- Native shared-KV layers tidak mendapat proyeksi K/V baru; struktur/depth dipertahankan.

Forward/backward memakai precision yang dipilih (auto BF16 pada CUDA). Semantic
parity/causality checks dilakukan setelah tiny model dikonversi ke FP32, dengan
atol/rtol 2e-5 dan rtol=0 pada causality. BF16 logits tidak dibandingkan seolah
bitwise/native numerical parity sudah terbukti. Report mencatat precision kedua
tahap; peak allocation meliputi keduanya.
Report menghitung kelompok check per model (11 untuk pair, 8 untuk baseline), bukan
jumlah semua assert statements. Tests arsitektur yang lebih luas tetap ada pada
[test_native_integration.py](../tests/test_native_integration.py), termasuk window
513, padding dan reload aliases. Notebook difokuskan ke preflight singkat.

## Full-size probe

Dimensi asli: E2B width1536/layers35; E4B width2560/layers42, vocab262144,
PLE width256, native local/global head dimensions256/512. Semua bobot random.
Bukan factory checkpoint pretrained M04/M05. Full-size probe tidak menjalankan
optimizer, generation loop, caching, quantization, real SFT atau benchmark repair.

Sebelum alokasi, notebook memeriksa CUDA/BF16 dan free VRAM terhadap estimasi
`4 * unique_parameter_count + 8 GiB`: BF16 weights + gradients + fixed headroom.
Estimasi ini hanya guard untuk batch pendek, bukan jaminan peak atau benchmark
kelayakan training. Actual allocated/reserved peaks dilaporkan setelah backward.
Weights diinisialisasi langsung pada CUDA/BF16; tidak dibuat sebagai full-size CPU
FP32 copy lalu dipindahkan. Satu experiment dijalankan per optional probe.

## Hasil dan reproduksi

Pada 6 Oktober 2026, graph standalone telah dijalankan lewat `app.run()` pada CPU
FP32: enam konfigurasi PASS, total 60 kelompok checks. Full-size tetap DISABLED.
CUDA/BF16 forward/backward dan FP32 semantic checks juga lulus keenam model
(total 60 kelompok checks) melalui marimo HTML export. Reference decoder
mempertahankan nonpersistent RoPE/scaling buffers sebelum attention diganti ke
native; load_state_dict saja tidak cukup untuk menyalin buffer tersebut.
Hasil dicatat pada [VALIDATION](VALIDATION.md).
GPU Molab Blackwell 96 GB belum dieksekusi dari sesi lokal ini; optional full-size
probe disediakan untuk dijalankan pengguna pada runtime tersebut.

```powershell
$env:BIFOCALGEMMA_NOTEBOOK_DEVICE = 'cuda'
$env:BIFOCALGEMMA_NOTEBOOK_PRECISION = 'bfloat16'
python -m marimo export html notebooks/bifocalgemma_architecture_test.py --no-sandbox --no-include-code -o outputs/marimo_architecture_cuda.html
```

Environment overrides ini hanya device/precision untuk QA. Control center tetap
sumber setting eksperimen, flags dan panjang. Hasil tersimpan sebagai JSON dengan
UTC timestamp di `outputs/marimo_architecture/`; tidak ditimpa pada rerun.
CUDA peaks adalah allocator counters untuk tiny batch, CPU peak null. Tidak ada
model/data download, HF token, upload artifact, RL, atau VM pada alur notebook.

## Menjaga standalone tetap konsisten

Core definitions disalin dari [native.py](../src/bifocalgemma/models/native.py),
[collator](../src/bifocalgemma/data/collator.py), contracts, visibility helper dan CE.
Perubahan core harus diselaraskan dengan notebook lalu diperiksa memakai marimo
check dan real graph execution. Pinned version/config revisions dan semantic parity checks
mencegah notebook dianggap contoh produksi lintas versi tanpa validasi.

## Probe asli pada Molab

[Pretrained GPU probe](PRETRAINED_GPU_PROBE.md) tersedia sebagai notebook standalone
terpisah. Ia memuat pinned base/IT text weights, melakukan attention-LoRA/AdamW
microsteps, dan membandingkan 1–4 independent resident models. Loader/optimizer
helpers diuji pada tiny fixtures; GPU96/pretrained measurements dijalankan pengguna.
Ini prototype profiling, bukan penyelesaian production trainer/cache/checkpoint.
