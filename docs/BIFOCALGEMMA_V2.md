# BifocalGemma v2

Enam notebook standalone menjalankan enam eksperimen independen, satu model per
instance Molab. Ini pengujian hipotesis encoder-decoder terhadap baseline dengan
ukuran decoder sama; implementasi tidak menjamin kemenangan atau urutan skor.

| Notebook | Backbone encoder | Backbone decoder | Repo output |
|---|---|---|---|
| `notebooks/train_e2b_e2b.py` | E2B base | E2B IT | `daruokta/bifocalgemma-v2-e2b-e2b` |
| `notebooks/train_e4b_e2b.py` | E4B base | E2B IT | `daruokta/bifocalgemma-v2-e4b-e2b` |
| `notebooks/train_e2b_e4b.py` | E2B base | E4B IT | `daruokta/bifocalgemma-v2-e2b-e4b` |
| `notebooks/train_e4b_e4b.py` | E4B base | E4B IT | `daruokta/bifocalgemma-v2-e4b-e4b` |
| `notebooks/train_e2b_it.py` | - | E2B IT | `daruokta/bifocalgemma-v2-e2b-it` |
| `notebooks/train_e4b_it.py` | - | E4B IT | `daruokta/bifocalgemma-v2-e4b-it` |

## Arsitektur dan resep

Registry v1 tetap tersedia untuk membaca eksperimen lama. `get_v2_experiment`
memilih arsitektur `normalized_merged_v2`: embedding encoder base dipertahankan
terpisah, decoder input/LM head tetap tied, PLE terpisah dan word/PLE frozen.
Semua pasangan memakai learned linear FP32 tanpa bias + RMSNorm FP32 epsilon
1e-6, output kembali ke dtype backbone. Linear simetris dimulai identity;
asimetris semi-orthogonal dengan seed CPU 17; gain RMSNorm dimulai satu.

Merged attention mempertahankan proyeksi self/cross K/V bersama dan satu softmax.
Query/self-key mendapat native RoPE, cross-key tidak memakai posisi decoder.
Self causal/window mask dan padding tetap aktif; encoder keys terlihat global
pada setiap layer. Native Q/K/V norm, scaling dan KV ownership Gemma 4 dipertahankan.
Linear alignment tidak diklaim bebas kehilangan informasi.

`unified_v2` memakai AdamW (betas 0.9/0.999, eps 1e-8, weight decay 0), attention
LoRA rank16/alpha32, LR 2e-5, alignment/gain LR 1e-4. Semua grup joint, warmup 3%,
cosine menuju 10%, clip1, seed17, satu epoch, effective batch4. Backbone BF16;
trainables/optimizer FP32. Hyperparameter merupakan default eksperimen, bukan
hasil optimasi yang telah terbukti.

Dependency Molab mengikuti environment conda `ai` lokal yang diperiksa saat
implementasi: Torch2.14.1+cu130, torchvision0.29.1+cu130,
Transformers5.20.2, marimo0.25.2, huggingface-hub2.2.0, safetensors0.8.0,
tokenizers0.23.3, pyarrow26.0.0, numpy2.5.4, pytest8.4.2, psutil7.2.2,
wigglystuff0.5.36.
Kernel GPU memakai Triton3.8: `triton-windows==3.8.0.post29` di Windows dan
`triton==3.8.0` di Molab Linux. Extras project `gpu-kernels` menyediakan pin ini.
Versi dipin agar enam instance konsisten; ini bukan permintaan mengejar rilis
baru yang berbeda di setiap mesin. Setup memakai `uv pip` pada interpreter kernel.

## Menjalankan

1. Buka satu notebook untuk satu instance Molab. Sediakan `HF_TOKEN` dengan akses
   write ke repo output dan read ke Gemma gated, lalu Run notebook. Jika token
   belum tersedia di environment, isi kolom token dan tekan **Jalankan** sekali.
2. Notebook otomatis menjalankan preflight GPU, training satu epoch dari backbone
   resmi, menyimpan checkpoint, lalu seluruh benchmark pada model hasil training.
   Tidak ada dropdown tahap, pilihan recipe, atau pilihan snapshot.
3. Checkpoint dan hasil benchmark diunggah ke repo HF model tersebut. Jika jaringan
   bermasalah, hasil lokal tetap disimpan dan status menunjukkan upload tertunda.
4. Setelah sesi terputus, jalankan notebook yang sama lagi. Training yang selesai
   dilewati; benchmark melanjutkan jurnal prediksi. Jika semuanya sudah selesai,
   notebook memverifikasi receipt dan mencoba sinkronisasi tanpa memuat model lagi.

Notebook memuat runtime source bersama dari
[`daruokta/bifocalgemma-v2-assets`](https://huggingface.co/datasets/daruokta/bifocalgemma-v2-assets),
dengan commit dan checksum yang dipin di `notebooks/runtime-manifest.json`.
Unduhan runtime publik tidak memerlukan token; akses data/model dan upload hasil
tetap memakai token di atas. Tidak ada payload Base85 panjang di notebook.

Dashboard Wigglystuff menampilkan progress, loss train/dev, rincian token,
timeline tahap, log terbaru, dan waktu aktivitas terakhir. Waktu berjalan
diakumulasi lintas sesi dari jurnal lokal/checkpoint; jeda sesi ditampilkan
terpisah. Setelah crash, ekor waktu yang belum tersimpan tidak bisa dipastikan,
dan pemulihan remote hanya mencakup snapshot terakhir yang terverifikasi.
[Riset UI dan kontrak waktu](WIGGLYSTUFF_TELEMETRY.md) menjelaskan batasannya.

Notebook langsung menjalankan full training setelah preflight; tidak ada pilot
200 step yang berjalan otomatis. API riset `run_v2(..., mode="pilot")` tetap
tersedia secara terpisah untuk diagnostik 800 train/256 dev, konteks tertukar dan
attention encoder. API benchmark baseline juga tetap mendukung `baseline_snapshot="initial"`;
alur notebook standar mengevaluasi baseline sesudah SFT.

Data dipin ke `daruokta/bifocalgemma-data` commit
`7972f95d7c87f163c8b5af6a4acd308c0c1d0e23`, collection
`mixed_context_enc32k_dec8k_v3`. Konteks tidak dipotong; baseline memasukkan konteks
sama ke prompt decoder. Fingerprint token target dicatat terpisah dari prefix agar
kesetaraan target enam run dapat diaudit.

## Preflight dan waktu sesi

Preflight menguji tiled chunk128/256/512 dan kandidat FlexAttention, microbatch1/2
dengan accumulation4/2. Forward/backward diukur pada contoh pendek, median dan
panjang; stress maksimum encoder/decoder gabungan dan dev diuji. Minimal8GiB
headroom wajib terpenuhi. Loss/gradients kandidat dibandingkan tiled sebelum
backend tercepat yang lolos dipilih. Biaya compile/warmup dicatat terpisah.
Tidak ada klaim percepatan di GPU96GiB sebelum hasil preflight GPU tersebut.
Kandidat hybrid juga diuji: local head256 memakai FlexAttention, global head512
memakai tiled. Ini mempertahankan kesempatan mempercepat local layers ketika kernel
GPU tidak mendukung head512. Rute backend setiap layer dicatat dalam hasil preflight.

Training menyimpan checkpoint pada optimizer boundary setiap15menit dan final;
upload snapshot setiap30menit saat checkpoint tersedia dan final, dengan retry
serta status pending. Dua checkpoint remote terbaru dipertahankan; penghapusan
yang lebih lama dilakukan setelah replacement terunggah dan marker terverifikasi.
`last_seen.json` dikirim setiap5menit pada event boundary. Tidak ada heartbeat saat
kernel terblokir dalam satu operasi panjang; waktu kedaluwarsa bukan bukti mesin mati.

Setelah putus sesi, Run notebook yang sama untuk memulihkan checkpoint dan
cursor/RNG/optimizer/scheduler. Snapshot remote dipin selama recovery. Jika tidak
ada checkpoint pertama, restart dimulai dari backbone dan diagnostics lama diarsip.
Run training yang sudah selesai tidak memuat model atau menjalankan training lagi.
Platform Molab yang berhenti tetap perlu dinyalakan pengguna.

## Evaluasi dan kompatibilitas

Cache generasi bersifat per request: static cross KV, dynamic self KV; local self
KV dibatasi window, global KV dipertahankan. Owner KV dipakai bersama seperti model
native. Training tidak memakai inference cache. Jalur tanpa cache tetap tersedia
untuk pemeriksaan numerik. Protokol baru `greedy_cached_normalized_merged_v2`
mencegah tercampurnya prediksi lama dan baru.

Repair functional success menjadi hasil utama; apply success dan status syntax
dilaporkan terpisah. SWEQA dan completion tetap metrik sendiri.
`tools/compare_v2_results.py MODEL_DIR BASELINE_DIR` menghitung delta per contoh
dan CI bootstrap berpasangan95% dengan seed17. Cohort/order dan protokol harus sama;
tidak ada composite score lintas task. CI ini tidak mengukur variasi seed training
atau dependensi antarkasus dari repo sama.

Format checkpoint v2 baru adalah `bifocalgemma_normalized_sft_v3`. Manifest merekam
spec/revision backbone, alignment/norm, policy, data/target fingerprint dan backend.
Resume lintas arsitektur/checkpoint lama ditolak. Notebook lama ada di
`notebooks/legacy/`; repo backup HF tidak diubah oleh notebook baru.

## Pemeriksaan pengembangan

Gunakan `C:/Users/daru/miniconda3/envs/ai/python.exe`. Regenerasi keenam file dengan
`tools/generate_training_notebooks.py`, lalu jalankan mode `--check`. Keenam notebook
memakai runtime immutable yang sama. Jika source runtime berubah, siapkan `--stage`
untuk ditinjau dan terbitkan dengan `--publish` sebelum regenerasi. Jalankan Pyrefly dengan interpreter proyek,
ty dengan `--python`, Ruff, marimo `check --strict`, dan pytest terkait. Pada Windows
sandbox, marimo memerlukan local socketpair asyncio sehingga lint dapat membutuhkan
eksekusi di luar pembatasan socket sandbox; lint tidak menjalankan training.

### Validasi kernel lokal, 24 Oktober 2026

Regresi lokal selesai: 171 tes lolos, termasuk empat pasangan, baseline, cache,
resume exact, partial checkpoint, enam lifecycle notebook, status HF terminal,
retry jaringan, syntax repair dan kompatibilitas alur lama. Ruff, Pyrefly, ty,
marimo strict pada enam notebook, serta generator `--check` lolos. Pengujian
enam instance memakai model kecil dan Hub tiruan; ini bukan enam sesi Molab nyata.

Perubahan alur sekali Run diuji lagi dengan 25 tes terkait yang seluruhnya lolos:
keenam model menghasilkan checkpoint dan delapan laporan benchmark, Run ulang
melewati model yang telah selesai, interupsi benchmark melanjutkan jurnal, serta
token dari environment dapat langsung menjalankan notebook tanpa memilih tahap.
Keempat pemeriksaan statis dan generator juga lolos setelah perubahan ini.

`triton-windows==3.8.0.post29` terpasang pada conda `ai`; kernel CUDA sederhana
lolos pada RTX 3060 Laptop 6 GiB dengan Torch2.14.1+cu130. Pengujian attention
lokal menghasilkan 9 passed dan 1 expected failure: fused output/gradient lolos
untuk head256 FP32/BF16 dan head512 BF16. Backward head512 FP32 melampaui shared
memory sm86 (196608 byte diperlukan, batas101376); hybrid tetap menyediakan tiled
untuk head512. Mask causal/bidirectional, padding dan pergantian panjang7/35/17
juga lolos. Gradien BF16 fused dan tiled masing-masing dibandingkan referensi
FP32, dengan batas relative L2 error3%; preflight model memakai batas5% per tensor
parameter dan mencatat error maksimum.

Ini belum merupakan profiling GPU96GiB, pilot200step, atau hasil benchmark baru.
Preflight pada GPU target tetap wajib lolos sebelum training otomatis dimulai.
Pilot riset terpisah tetap tersedia; kemenangan terhadap baseline baru bisa
dinilai setelah benchmark pada perangkat target selesai.
