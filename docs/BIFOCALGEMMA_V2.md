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
   write ke repo output dan read ke Gemma gated, lalu Run notebook. Leader E4B→E4B
   juga memerlukan write ke dataset `daruokta/bifocalgemma-v2-assets` untuk satu
   profil bersama. Jika token
   belum tersedia di environment, isi kolom token dan tekan **Jalankan** sekali.
   Profil terukur G4 sudah tersedia untuk runtime saat ini; keenam notebook dapat
   langsung dijalankan. Jika kontrak source/data berubah dan profil belum tersedia,
   E4B→E4B mengkalibrasi dan lima notebook lain menunggu sebelum memuat backbone.
2. Notebook otomatis menjalankan pemeriksaan memori singkat, training satu epoch dari backbone
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

## Pemeriksaan memori dan waktu sesi

Semua model memakai satu profil `shared_e4b_e4b_v1`. Hanya E4B→E4B yang melakukan
kalibrasi: tiled/Flex/hybrid, microbatch1/2/4, tiled chunk512/1024/2048, serta
checkpointing nonaktif pada dua kandidat tercepat. Effective batch selalu4,
loss chunk128. Kandidat diperingkat dengan waktu update sesudah warmup pada12
contoh dari bin panjang berfrekuensi sama; maksimum absolut bukan penentu skor
throughput. Kandidat tercepat yang lolos pemeriksaan bentuk maksimum dengan
margin4GiB dipublikasikan ke HF assets `profiles/training.json`. Ini pencarian
terbatas, bukan bukti optimum global. Hasil tiap percobaan disimpan untuk resume.

Lima notebook lain memakai profil tersebut persis, tanpa tuning masing-masing.
Kalau profil gagal pada salah satu model, run berhenti dengan alasan jelas, bukan
mengganti setting diam-diam. Folder run cukup `outputs/v2/<model>/train/`; versi
teknis dan checksum profil dicatat dalam manifest. Folder eksperimen sebelumnya
tidak otomatis dipakai untuk resume.

Pemeriksaan awal hanya dua tahap pada satu bentuk gabungan maksimum train/dev:
forward/backward/clip/optimizer dan evaluasi tanpa gradien. State optimizer asli
tetap berada di perangkat saat pengukuran. Bobot trainable dan RNG dikembalikan
persis sesudah pemeriksaan; update probe tidak dihitung sebagai training. Minimal
4GiB perkiraan peak headroom wajib terpenuhi. Konteks tidak dipotong agar muat.
Ini bukan jaminan semua bentuk CUDA/fragmentasi/pekerjaan GPU lain bebas OOM.

`diagnostics/execution_preflight.json` disimpan atomik per tahap dan langsung
dicoba diunggah ke HF, bahkan sebelum checkpoint pertama. Network error sementara
meninggalkan upload pending dengan retry. Resume memverifikasi checksum, identitas
kode/data/GPU, lalu melanjutkan tahap yang belum selesai. Jika keduanya selesai,
hanya kapasitas GPU saat ini yang diperiksa; tidak mengulang forward/backward.
Allocated, reserved, peak allocated, serta driver free ditampilkan terpisah.

FlexAttention/hybrid hanya dicoba oleh kalibrasi E4B→E4B. Error Molab `Required: 114688 Hardware limit:101376`
merupakan batas shared memory kernel backward, bukan bukti VRAM96GiB habis.
Tile backward head256 kini juga dibatasi; validasi pada GPU target tetap diperlukan.
[Perbaikan eksekusi dan penjelasan kernel](EXECUTION_FIXES.md).

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

### Perbaikan eksekusi, 25 Oktober 2026

Regresi alur/profil bersama, pemulihan, Hub tiruan, data dan notebook menghasilkan
109 passed. Enam uji guard CUDA BF16 dengan model kecil juga lolos pada microbatch
1/2/4, termasuk rollback bobot/RNG dan pemakaian ulang receipt. Uji Flex terbaru
menghasilkan6 passed dan1 expected failure untuk batas shared memory head512 FP32
pada sm86. Setelah runtime baru diterbitkan,26 tes notebook/dependency/telemetri
lolos; Pyrefly, ty dengan interpreter ai, Ruff pada source/tools/tests/notebook aktif,
serta marimo strict keenam notebook aktif lolos.

Regresi akhir setelah perbaikan alignment dan helper boundary menghasilkan116
passed (enam guard CUDA terpisah tidak diulang), serta enam tes validasi rilis profil
lolos. Pyrefly, ty, Ruff seluruh proyek dan marimo strict keenam notebook bersih.

[Pengukuran penuh G4](COLAB_RESULTS.md) menggunakan
[skrip diagnostik terpisah](COLAB_GPU_TEST.md), tanpa epoch training. Keenam model
lolos batas konfigurasi penuh. Profil terpilih microbatch1/accumulation4,
tiled chunk2048, loss chunk128, checkpointing aktif sudah dipublikasikan ke HF
assets setelah verifikasi. Peak E4B→E4B pada32768/8192 ialah61,09GiB allocated;
reserved65,96GiB. Flex/hybrid berjalan tetapi ditolak oleh pemeriksaan gradien.
Pilot riset dan benchmark kualitas tetap belum dijalankan oleh pengukuran GPU ini.
