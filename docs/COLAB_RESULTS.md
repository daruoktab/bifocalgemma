# Pengukuran G4, 25 Oktober 2026

Pengukuran diagnostik pada NVIDIA RTX PRO6000 Blackwell Server Edition, compute
capability12.0, kapasitas yang dilaporkan Torch94,97GiB, driver580.82.07/CUDA13.0.
Torch2.14.1+cu130, Triton3.8.0, Transformers5.20.2, Python3.13.16. Runtime source
`runtime-2cc7b98cf02d97d2.zip`, revision assets
`521b546c0f078d2fb4a74c2fe45323b5a34d592b`.

Tidak menjalankan epoch training, pilot riset, atau benchmark kualitas. Tiap update
optimizer diagnostik dipulihkan. Arsip lokal di `outputs/colab/` berisi hardware,
kontrak source/data, receipt checksummed, log, dan hash parameter trainable.

## Kecepatan kandidat E4B→E4B

Sampel12contoh nyata dari bin panjang berfrekuensi sama, tiga optimizer window
effective batch4. Semua shape di-warmup; kompilasi dicatat terpisah. Angka ini
pengukuran sampel, bukan ETA satu epoch atau bukti optimum global.

| Query chunk | Microbatch1 × accum4 | Microbatch2 × accum2 | Microbatch4 × accum1 |
|---|---:|---:|---:|
|512|4,220s/update|6,009s/update|10,547s/update|
|1024|4,273s/update|5,693s/update|10,124s/update|
|2048|4,183s/update|5,628s/update|9,942s/update|

Seluruh baris menggunakan tiled attention, loss chunk128 dan gradient checkpointing.
Batch lebih besar meningkatkan padding dan memakai lebih banyak memori pada
sampel ini; pemakaian VRAM tinggi tidak sendiri menjamin throughput lebih baik.
Selisih microbatch1 antar-chunk kecil; pilihan2048 hanya yang tercepat terukur pada
cohort ini. Dua kandidat tercepat tanpa checkpointing OOM bahkan pada cohort
tersebut: tensor aktif mencapai sekitar90,75GiB dan free sekitar50MiB.

## Guard panjang data aktual

|Model|Encoder/input decoder|Peak tensor saat backward|
|---|---|---:|
|E2B→E2B|32767 / 6207|42,58GiB|
|E4B→E2B|32767 / 6207|51,88GiB|
|E2B→E4B|32767 / 6207|50,30GiB|
|E4B→E4B|32767 / 6207|59,60GiB|
|E2B-IT|0 / 33204|29,62GiB|
|E4B-IT|0 / 33204|39,03GiB|

Keenam model lolos backward/finite-gradients/optimizer serta evaluasi, dengan
konfigurasi yang sama. Ini menguji maksimum data yang ditemukan, bukan semua
kemungkinan input sampai cap konfigurasi.6207 berasal dari input teacher forcing
setelah shift target; cap decoder model pasangan tetap8192. Guard terpisah pada
32768/8192 (dan41472 input baseline) dicatat di `boundaries.json`.

## Guard batas konfigurasi penuh

Seluruh model lolos dua tahap, menggunakan profil yang sama dan tanpa memotong
input agar muat. Parameter trainable setiap model pulih persis menurut checksum.

|Model|Encoder/input decoder|Peak tensor backward|Peak reserved|Headroom terukur|
|---|---|---:|---:|---:|
|E2B→E2B|32768 / 8192|43,83GiB|48,15GiB|50,49GiB|
|E4B→E2B|32768 / 8192|53,12GiB|57,03GiB|41,19GiB|
|E2B→E4B|32768 / 8192|51,79GiB|56,77GiB|42,53GiB|
|E4B→E4B|32768 / 8192|61,09GiB|65,96GiB|33,23GiB|
|E2B-IT|0 / 41472|34,81GiB|38,99GiB|59,51GiB|
|E4B-IT|0 / 41472|45,23GiB|51,12GiB|49,08GiB|

Pilihan akhir: tiled attention, query chunk2048, loss chunk128, microbatch1,
accumulation4, gradient checkpointing aktif. Ini konfigurasi tercepat yang lolos
dalam pencarian terukur; tidak diklaim memenuhi VRAM sampai hampir100% atau
mengalahkan semua strategi komputasi lain yang belum diukur.

Arsip final lokal: `outputs/colab/final-results.zip` (2002956byte). Pemeriksa
`tools/release_gpu_profile.py` memverifikasi CRC arsip, source/data/versi dependency,
enam checksum receipt, batas input, kesamaan setting, margin dan rollback bobot
sebelum publikasi. Ia tidak mengunggah arsip, model atau data ke repo assets;
hanya JSON profil dan bukti agregat yang diperlukan notebook.

Profil telah diterbitkan dan diunduh ulang pada commit
[`7bc4cd74e09cb0e5f5a044886527a389097da851`](https://huggingface.co/datasets/daruokta/bifocalgemma-v2-assets/blob/7bc4cd74e09cb0e5f5a044886527a389097da851/profiles/training.json).
Keenam notebook menggunakan profil ini jika kontrak source/data/dependency cocok;
tidak perlu mengulang pencarian setting di enam Molab. Pemeriksaan kapasitas dua
tahap tetap dijalankan di setiap run dan dapat dilanjutkan dari receipt yang cocok.

Runtime Colab `bifocalgemma-test` dihentikan setelah hasil tersalin. API sesudah
penghentian melaporkan0assignment aktif dan laju0compute unit/jam. Seluruh sesi
menggunakan sekitar7,90compute unit, termasuk setup, diagnosis kegagalan,
kalibrasi dan guard. Ini bukan tarif training satu epoch.

## Penolakan fused attention

Kernel Flex dan hybrid dapat berjalan pada BF16 setelah perbaikan tile backward;
penolakan pada model penuh kali ini berasal dari pemeriksaan numerik, bukan OOM.
Pada contoh diagnostik yang sama:

|Backend BF16|Loss|Relative L2 seluruh gradien vs tiled|Maksimum per tensor|
|---|---:|---:|---:|
|Tiled|1,727224|0|0|
|Flex|1,737956|94,51%|145,00%|
|Hybrid|1,737745|74,75%|144,83%|

Pengulangan tiled identik. Pada model FP32, hybrid vs tiled menghasilkan loss
1,715817 vs1,715816, relative L2 gabungan0,071%, maksimum per tensor0,102%.
Ini menunjukkan sensitivitas besar terhadap presisi pada model penuh; bukan bukti
bahwa satu backend telah memenangkan kualitas. BF16 tiled sendiri juga berbeda
besar dari oracle model FP32 (relative L2 gabungan124,88%), sehingga perbandingan
ini tidak membenarkan klaim bahwa tiled adalah ground truth presisi penuh.
Ambang penerimaan5% terhadap referensi BF16 tidak dilonggarkan; fused belum
diaktifkan untuk training produksi. Rincian parameter ada di
`outputs/colab/numerical_details.json` dan `numerical-results.zip`.

Hash parameter trainable sebelum/sesudah kalibrasi dan diagnostik numerik identik.
Pengukuran kecepatan/kapasitas tidak menggantikan pilot, encoder-use diagnostics,
atau benchmark fungsional untuk menilai hipotesis kemenangan terhadap baseline.
