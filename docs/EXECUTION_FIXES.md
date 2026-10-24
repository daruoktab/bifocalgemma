# Perbaikan eksekusi Molab, 25 Oktober 2026

## Kalibrasi E4B→E4B, satu konfigurasi untuk enam model

E4B→E4B menjadi satu-satunya leader kalibrasi. Tujuannya waktu update terendah
di antara kandidat yang lolos batas memori, berdasarkan peak tensor allocated,
bukan membuat angka reserved setinggi mungkin. Margin awal4GiB dipertahankan.
Mengisi lebih banyak VRAM tidak sendiri menjamin komputasi lebih cepat.

Pencarian mencakup tiled/Flex/hybrid, microbatch1/2/4, dan tiled query chunk512,
1024,2048. Dua kandidat tercepat juga dicoba tanpa gradient checkpointing.
Maksimal17 percobaan kecepatan, di luar pemeriksaan numerik satu kali per backend.
Kernel yang gagal langsung dicatat dan tidak dicoba lagi pada ukuran batch lain.

Waktu diukur pada12 contoh nyata dari bin panjang berfrekuensi sama, dengan
pengelompokan effective batch4 dan pembobotan token seperti training. Semua bentuk
batch di-warmup terlebih dahulu; biaya compile/warmup dicatat terpisah. Kandidat
diperingkat dari waktu sesudah warmup, kemudian diuji pada bentuk maksimum sampai
yang tercepat dan muat ditemukan. Ini bukan bukti optimum global atau estimasi
waktu training seluruh epoch; sample throughput tetap terbatas.

Profil terpilih disimpan di dataset HF assets `profiles/training.json`, setelah
validasi lengkap. Lima notebook lain menunggu profil sebelum memuat backbone,
lalu memakai backend, chunk, microbatch, accumulation dan checkpointing yang sama.
Profil tidak boleh diganti per model. Kegagalan kapasitas follower menghentikan
run; baseline dapat memiliki konteks decoder lebih panjang daripada model pasangan,
sehingga E4B→E4B tidak diasumsikan otomatis paling berat pada setiap bentuk tensor.

Effective batch tetap empat contoh; optimizer/LoRA/data/seed/protokol benchmark
tidak diubah. Source/data contract dan checksum mengikat profil yang dipin lokal
di tiap run. Publikasi dengan parent-commit guard mencegah dua leader menimpa
pilihan valid untuk kontrak yang sama. Profil lama dari source/data lain tidak dipakai.

Versi eksekusi `shared_e4b_e4b_v1` ada dalam manifest. Nama folder cukup
`outputs/v2/e2b-it/train/` (dan model lainnya); repo output tetap satu per model.
Run lama dengan folder `normalized-merged-v2-train` tidak dimigrasikan diam-diam.
Kedua repo baseline lama dihapus atas permintaan pengguna setelah kedua notebook
dikonfirmasi berhenti. Run notebook baru akan membuat repo output baseline lagi.

## Memory guard setelah profil bersama dipilih

Pencarian terpisah per model dihapus. Probe menggabungkan
encoder terpanjang dan decoder terpanjang dari train/dev tanpa mengubah panjang
konteks, lalu menguji dua tahap saja:

1. Forward/backward, finite-gradient check, gradient clipping dan optimizer step.
2. Evaluasi tanpa gradien pada bentuk maksimum yang sama.

Optimizer AdamW asli dibuat dan state-nya dialokasikan sebelum stress pass,
sehingga memori aktivasi diuji bersama state optimizer. Parameter trainable
dicadangkan di CPU, kemudian dipulihkan persis bersama RNG dan mode model pada
keberhasilan maupun exception. Tidak ada update probe yang menjadi bobot awal
eksperimen. Model beku tidak digandakan di GPU.

Setiap tahap menyimpan receipt atomik beserta checksum dan durasi. Receipt dipin
ke sumber/data, konfigurasi, GPU, versi Torch/CUDA dan ukuran probe. Tahap selesai
dapat dipakai ulang; tahap terputus diulang. Receipt rusak atau identitas berubah
ditolak. Sebelum memakai hasil lama, free memory ditambah cache yang reclaimable
dibandingkan dengan tambahan peak terukur dan margin4GiB. GPU yang sekarang
dipakai proses lain dapat ditolak meskipun receipt lama pernah lolos.

Receipt langsung dicoba diunggah ke HF tanpa menunggu checkpoint. Upload ini
hanya memuat pipeline identity dan hasil guard, dengan receipt terpisah; tidak
mengklaim checkpoint model sudah aman. Jaringan putus mempertahankan bukti lokal
dan status pending; callback berikutnya mencoba lagi mengikuti backoff. Hanya
bukti yang sudah berhasil diunggah yang dapat dipulihkan pada instance baru.

Guard tetap mengerjakan dua pass konteks maksimum, sehingga bukan operasi instan.
Kalibrasi pertama juga memerlukan GPU target dan waktu pengukuran; pekerjaan ini
dilakukan sekali oleh leader, bukan enam pencarian independen. Kandidat dan tahap
yang selesai disimpan lokal dan dicoba disinkronkan agar dapat dilewati saat resume.
Ia tidak menjamin seluruh bentuk CUDA, fragmentasi, atau perubahan pemakaian GPU
selanjutnya akan selalu muat. Jika profil bersama gagal, training penuh berhenti dengan
alasan jelas; tidak diam-diam mengganti konteks atau membuat model memakai resep lain.

## Mengapa Flex dan hybrid gagal

FlexAttention mengompilasi attention bermask menjadi kernel fused melalui
TorchInductor/Triton. Ia memproses blok query/key/value dengan workspace dekat
compute unit, mempertahankan softmax attention yang sama. Hybrid memakai Flex
untuk layer head256 dan tiled reference untuk layer head512. Karena itu error
pada head256 juga menggagalkan hybrid.

Log kedua baseline menyebut `Required: 114688 Hardware limit:101376`: kernel
backward meminta112KiB shared memory, melebihi batas yang dilaporkan99KiB.
Nama exception mengandung `OutOfMemoryError`, tetapi ini batas sumber daya kernel,
bukan alokasi tensor global ke VRAM96GiB. Penyebab batas compiler itu sendiri
tidak disimpulkan sebagai spesifikasi seluruh GPU dari satu pesan error.

Kode lama membatasi tile backward hanya untuk head512 atau sequence<32. Head256
dengan sequence>=32 masih memakai default compiler yang dapat terlalu besar.
Sekarang keempat tile backward dibatasi16 dan `num_stages=1` untuk kedua head.
Ini perbaikan kandidat kernel, bukan janji Flex sudah cocok pada semua GPU. Hanya
backend yang lolos pemeriksaan numerik dan stress test leader boleh menjadi profil.

Tes G4 penuh juga menemukan konstruksi alignment simetris gagal sebelum kalibrasi:
`torch.nn.init.eye_` membuat operand mengikuti default device CUDA meskipun tensor
tujuannya CPU. Seluruh inisialisasi identity/semi-orthogonal sekarang dilakukan
dalam konteks device CPU eksplisit, lalu disalin ke parameter tujuan.24 tes model
lolos sesudah perbaikan, termasuk dimensi sama, ekspansi dan reduksi yang dibuat
langsung di CUDA. Ini kegagalan device placement, bukan kehabisan VRAM.

Referensi primer: [FlexKernelOptions PyTorch2.14](https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html#flexkerneloptions)
menjelaskan tile backward; [CUDA memory management PyTorch](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management)
membedakan tensor allocated, allocator reserved, dan memori yang terlihat oleh
`nvidia-smi`. Angka reserved tinggi tidak sendiri membuktikan seluruhnya tensor aktif.

## Tampilan dan runtime

Log guard menyebut tahap1/2 atau2/2, backend/chunk/batch, panjang token, peak,
dan headroom. Counter tokenisasi tidak lagi terbawa ke status training. Heartbeat
menyertakan konfigurasi batch/backend dan metrik allocated/reserved/free.

Runtime tetap arsip source immutable di HF. Pengemasan mengkanonikkan newline LF,
urutan nama file, dan metadata ZIP sehingga checkout Windows/Linux menghasilkan
arsip yang sama. Checksum penuh tetap diverifikasi; notebook menampilkan versi pendek.

## Bukti pada GPU target

[Pengukuran G4](COLAB_RESULTS.md) telah selesai dengan keenam model lolos guard
batas konfigurasi penuh. Satu profil terverifikasi dipublikasikan ke HF assets:
microbatch1/accumulation4, tiled chunk2048, loss chunk128, checkpointing aktif.
Keenam resolver diuji membaca profil publik yang sama, tanpa memulai pencarian.
Peak E4B→E4B pada32768/8192 ialah61,09GiB tensor,65,96GiB reserved. Pilihan ini
berdasarkan kecepatan sampel, bukan memaksa VRAM terisi. Full run tetap memeriksa
kapasitas saat ini pada masing-masing Molab. Pilot riset/benchmark kualitas belum
dilakukan oleh pengukuran kapasitas ini.
