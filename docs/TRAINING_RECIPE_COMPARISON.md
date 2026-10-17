# Perbandingan base dan optimasi v8

15 Oktober 2026. Seluruh 6.742 baris `working-molab-v8-exp-temp.py` dibaca
sebagai referensi. Notebook v8 tersebut tidak dijalankan. Dataset dan benchmark
tetap memakai rilis beku yang sama; training pretrained berikutnya dijalankan pengguna.

## Matriks dan urutan

| Notebook | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Simetris | E2B–E2B base | E2B–E2B optimized | E4B–E4B base | E4B–E4B optimized |
| Asimetris | E2B–E4B base | E2B–E4B optimized | E4B–E2B base | E4B–E2B optimized |
| Baseline | E2B-it base | E2B-it optimized | E4B-it base | E4B-it optimized |

Total 12 hasil SFT. `base` berarti recipe AdamW, bukan encoder base checkpoint
dan bukan IT model tanpa tuning. Setiap run dimuat ulang dari sumber pretrained
dan seed yang sama. Optimized tidak melanjutkan hasil base. Satu model resident
pada satu waktu. Selesaikan empat SFT + upload dahulu, lalu benchmark keempat
checkpoint secara berurutan. Kegagalan training/upload menahan fase benchmark.

Factory angka bersama ada di [`comparison_recipes`](../src/bifocalgemma/training/recipes.py).
Cell **7a · Training policy** memuat `TRAINING_RECIPES`, menampilkan group dan
seluruh opsi optimizer, dan dapat dipakai mengedit policy sebelum run dimulai.
Control center tetap mengatur batch, budget, seed, data, interval, dan output.

## Default yang disamakan

| Pengaturan | Base dan optimized, seluruh model |
| --- | --- |
| Peak LR setiap group yang ada | 2e-5; tidak ada multiplier LR tersembunyi |
| Warm-up | linear 3% total optimizer updates, ceil |
| Decay | cosine ke 10% peak / 2e-6; tanpa restart |
| Epoch / budget | 1 epoch penuh, MAX_STEPS=0 |
| Train / accumulation / effective batch | 2 / 2 / 4 |
| Validation batch | maksimum 2, source/task tetap terpisah |
| LoRA | attention q/k/v/o yang tersedia; rank16, alpha32, scaling alpha/r=2 |
| Frozen | ordinary embeddings, PLE, seluruh backbone lainnya |
| Precision | backbone BF16, trainable LoRA dan alignment FP32 |
| Objective | target-only token-weighted CE, satu shifting, prefix/pad -100 |
| Regularisasi | weight decay0, global raw-gradient clipping1, smoothing0, embedding noise0 |
| Context / decoder | encoder32K, paired shifted decoder8K; baseline allowance41472 |
| Seed / cohort | 17; seluruh 44.408 train dan 4.934 dev, SHA dataset tetap |
| Benchmark | seluruh delapan protokol yang sama, greedy dan batas per-case yang sama |

Group encoder tidak ada pada baseline; alignment hanya ada pada pasangan asimetris.
Simetris tetap shared ordinary words/Identity, PLE terpisah. FP32 alignment
menghitung linear dalam FP32 lalu mengembalikan output ke dtype backbone; ini
berlaku pada **kedua recipe**. Menyamakan angka LR tidak menyamakan norm update
dua algoritme. Default adalah titik awal konservatif, bukan optimum yang telah
dibuktikan pada Gemma4. Final test tidak dipakai untuk memilih hyperparameter.

LR2e-5 mempertahankan titik awal pilot yang finite pada repo ini. Warm-up3%
dan cosine juga digunakan dalam setup LoRA pada
[Learning Rate Matters](https://arxiv.org/html/2602.04998v1); konteks, batch,
module coverage dan LoRA scaling studi itu berbeda. Angka ini bukan bukti
optimum untuk BifocalGemma. [T5Gemma2](https://arxiv.org/html/2512.14856v2)
memakai warm-up100 langkah pada adaptasi dengan batch token sangat besar dan
menyetel LR per ukuran model; menyalin100 langkah ke batch4 contoh tidak
menyamakan jumlah token adaptasinya.

## Optimizer pembanding

**Base:** AdamW, betas=(0.9,0.999), epsilon1e-8, foreach=False, weight decay0.

**Optimized:** `grok_orscale_lora_v1`, kombinasi eksperimental dari ide v8:

1. EMA gradien decay0.98 (skala waktu sekitar50 update), dimulai dari nol,
   ditambahkan ke gradien dengan amplification0.5. Nama parameter dibedakan dari
   v8 untuk menghindari tertukarnya decay dan amplification.
2. Setelah global clipping trainer, norm gradien hasil filter dibatasi1 per matriks.
   Nesterov momentum0.95, Newton–Schulz5 iterasi FP32 pada Gram yang lebih kecil.
3. Arah ortogonal dikalikan shape factor `0.2*sqrt(max(rows,cols))`, kemudian
   trust ratio dikalibrasi per matriks dan dibatasi **[0.5,2.0]**. Weight decay0
   berada di dalam arah update. Tidak ada multiplier20× dari konfigurasi v8.
4. LoRA B yang dimulai dari nol memakai trust1 sampai bobot dan arah sama-sama
   nonzero; tanpa ini norm bobot nol dapat mengunci adapter.

Ini adaptasi pada **faktor LoRA A dan B secara terpisah**, bukan ortogonalisasi
produk BA atau reproduksi eksperimen full-matrix OrScale. Bukti paper OrScale
berasal dari pretraining; hasilnya belum menetapkan LR optimum untuk attention
LoRA Gemma4. Rentang trust diperketat dan amplification diturunkan dari v8 untuk
membatasi eksperimen pertama. Norm update dan trust min/mean/max tiap group
dicatat pada setiap optimizer step; jangan menilai manfaat dari kurva yang
terlihat lebih halus saja.

Seluruh trainable saat ini matriks2D. Karena itu tidak ada cabang AdEMAMix1D
yang tidak pernah dipakai. Dua state matriks FP32 (EMA gradien dan momentum)
serta scalar calibration tersimpan di checkpoint. AdamW juga punya dua moments;
kesetaraan peak VRAM belum dijamin karena scratch Newton–Schulz berbeda.

Dasar sumber: [OrScale paper](https://arxiv.org/abs/2605.07815),
[OrScale penulis](https://github.com/NUS-HPC-AI-Lab/OrScale),
[Grokfast paper](https://arxiv.org/abs/2405.20233),
[Grokfast penulis](https://github.com/ironjr/grokfast).
Komposisi ini menguji satu bundle optimizer. Keuntungan bundle belum menjelaskan
kontribusi setiap komponen; ablation Grokfast-only/OrScale-only dan beberapa seed
diperlukan setelah pilot.

## Ide v8 yang belum diaktifkan

| Ide | Alasan |
| --- | --- |
| NEFTune alpha5 | Bukti utama instruction/chat; perlu ablation patch correctness dan penempatan noise yang tidak menggandakan shared embedding/PLE |
| Label smoothing0.05 | Mengubah objective dan interpretasi train loss; perlu bukti functional repair, validation harus tetap CE murni |
| rsLoRA / rank64 / all-linear | Mengubah kapasitas/scaling; rank16/alpha32 tetap sama agar bundle ini menguji optimizer |
| SVD IT task-vector steering | Decoder sudah IT; perubahan inisialisasi mengaburkan perbandingan |
| MTO unused-token tags | Vocabulary/token IDs v8 berbeda; tidak diperlukan oleh kontrak output dataset sekarang |
| ORPO/TLPO, vision graft | Objective/data tambahan di luar perbandingan SFT kode ini |

[NEFTune](https://arxiv.org/abs/2310.05914) dan
[rsLoRA](https://arxiv.org/abs/2312.03732) adalah kandidat ablation berikutnya,
bukan jaminan peningkatan di setting ini. v8 juga memakai custom AdEMAMix blend,
sehingga label “seluruh teknologi v8” tidak tepat untuk implementasi sekarang.

## Checkpoint dan validation lebih jarang

Control center memakai `CHECKPOINT_FRACTION=0.10` dan `EVAL_FRACTION=0.25`.
Setelah tokenisasi, horizon aktual adalah minimum budget eksplisit dan epoch
penuh. Interval = ceil(horizon × fraction), minimum1. Final validation selalu
sekali, termasuk ketika final step berimpit dengan interval; final checkpoint
selalu tersedia. Dengan 11.102 updates:

- checkpoint setiap1.111 update, ditambah final11.102: sekitar10 kali save;
- validation setiap2.776 update, ditambah final11.102: empat full dev passes;
- dua checkpoint terbaru **per model/recipe** disimpan lokal dan di HF;
- jika runtime hilang, pekerjaan sejak checkpoint terakhir (sekitar10% run)
  dapat perlu diulang. Ini trade-off pengurangan upload; log UI tetap berjalan.

Training loop tetap mencatat loss/LR/timing/resource tiap step, console setiap10
step. Interval save/validation tidak memperjarang telemetry. Tidak ada early
stopping atau pemilihan bobot berdasarkan final test; hasil akhir memakai final
step dengan budget sama. Validation bukan tambahan optimizer update.

## Identitas, resume, dan pengujian

`RUN_NAME=enc32k-dec8k-context-v3-recipes-v1-torch2141`. Torch2.14.1+cu130
dipasang ulang oleh cell setup setelah import Marimo untuk run baru; artefak pilot2.11.0 tidak
dipakai untuk auto-resume. Path lokal:
`OUTPUT_ROOT/EXPERIMENT/RUN_NAME/{base|optimized}/`.
Path HF: `{FAMILY}/{EXPERIMENT}/{RUN_NAME}/{base|optimized}/`.
Training artifacts/checkpoints, benchmark predictions, stage receipts, serta
upload receipts tidak saling menimpa. AUTO_RESUME memilih checkpoint lengkap
untuk setiap run; explicit RESUME_FROM hanya boleh jika satu model/recipe dipilih.
Perubahan recipe, LR, horizon, parameter groups atau runtime ditolak oleh
compatibility guard sebelum bobot/metadata resume dimodifikasi.

Pengujian lokal memakai native tiny, bukan bobot pretrained: update factor nol,
arah transpose/zero, state optimizer FP32, exact resume seluruh keluarga,
penolakan recipe berbeda, interval/final validation, dan empat run terpisah
per notebook. Ini validasi implementasi; keberhasilan kualitas, kestabilan,
durasi, serta peak VRAM model asli memerlukan run pengguna.
