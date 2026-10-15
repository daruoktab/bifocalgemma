# Rencana run berikutnya: scheduler, batching, dan kekuatan bukti

15 Oktober 2026. Pilot disimpan di `outputs/result_archives/2026-10-15/`;
angka berikut berasal dari snapshot HF `419b21e2d5ce5e4c9bf777174bf96b96cc1e972d`.
Perubahan ini tidak menjalankan training pretrained atau mengubah dataset HF.

## Default notebook baru

Ketiga notebook memakai run baru `enc32k-dec8k-context-v3-recipes-v1` dengan
base AdamW dan optimized GrokOrScale LoRA. Masing-masing menghasilkan empat
run dari dua model. [Rancangan comparison](TRAINING_RECIPE_COMPARISON.md)
merinci optimizer dan interval validation25%/checkpoint10% dari total update.
Peak LR encoder, decoder, dan alignment tetap **2e-5**. Linear warm-up adalah
**3% optimizer updates**, dibulatkan ke atas, kemudian cosine decay sampai
**10% peak LR = 2e-6**. Ini baseline eksperimen yang masuk akal, bukan optimum
yang sudah ditemukan lewat tuning. Tidak ada restart cosine atau staged unfreezing.

Untuk 44.408 contoh, satu epoch dan effective batch 4: 11.102 update,
334 warm-up update, kemudian 10.768 update decay. LR naik dari sekitar 5,99e-8
pada update pertama sampai 2e-5 pada update 334; update terakhir memakai 2e-6.
Seluruh group mempertahankan LR dan warm-up masing-masing. Scheduler dihitung
dari global optimizer step, sehingga resume tidak mengulang warm-up. Horizon,
floor, policy dan parameter groups masuk compatibility manifest; perubahan
horizon/policy ditolak pada resume. Pemanggil policy lama tetap warm-up/constant.
Sumber desain: [Transformers optimizer schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules).

| Tahap | Default | Catatan |
| --- | --- | --- |
| Training | microbatch 2 × accumulation 2 | Effective batch tetap 4; jumlah update dan cohort sama |
| Validation | maksimum 2 | Batch terpisah, hanya menggabungkan sumber/task sama untuk loss breakdown |
| Benchmark | 1 | Decoder saat ini single-case; belum ada batched generation yang tervalidasi |

Loss training tetap dibobot jumlah token target dalam seluruh optimizer window,
bukan rata-rata mean loss microbatch. Training batch dari sumber/task campuran
dilabel `mixed_batch`; jangan menafsirkannya sebagai loss satu sumber tertentu.
Validation tetap mencatat masing-masing sumber/task secara tepat. Urutan contoh
dan ID cohort tidak berubah. Floating-point batch kernels dapat memberi perbedaan
numerik kecil; kesetaraan kualitas pretrained belum diuji.

Preflight memakai batch sebenarnya, actual maxima dan gabungan panjang ekstrem.
Train dan dev diprobe sebelum optimizer update; probe dev juga backward sebagai
stress konservatif, tanpa update parameter. RNG/mode dipulihkan dan grad dibuang.
GPU harus menyisakan perkiraan 8 GiB dari peak allocated plus penggunaan di luar
allocator PyTorch untuk optimizer/runtime. Probe tidak mengalokasikan Adam states;
headroom bukan jaminan terhadap proses GPU lain atau fragmentasi berikutnya.
Jika probe gagal, pipeline berhenti: ubah training ke batch1/accumulation4 atau
turunkan eval batch, lalu gunakan identitas run sesuai konfigurasi tersebut.
Tidak ada silent truncation, pengurangan dataset, atau fallback tersembunyi.

## Apa yang diketahui tentang VRAM

| Pilot batch1 | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| E2B–E2B | 34,42 GiB | 38,34 GiB |
| E2B–E4B | 40,53 GiB | 44,73 GiB |
| E4B–E4B | Belum diukur | Belum diukur |

GPU terukur 94,97 GiB. Angka pilot tersebut terikat panjang/dtype/backend
dan versi runtime, bukan batas universal. E4B–E4B berpotensi menjadi pasangan
terberat tetapi baseline causal dengan context+target panjang juga perlu probe.
Bobot BF16 text sebelum sharing sekitar 27,80 GiB untuk E4B+E4B menurut audit
config; ordinary embedding sharing mengurangi bobot, tetapi aktivasi, gradient,
temporary attention dan loss buffers tetap harus ditambah. Jangan menyebut
27,80 GiB sebagai estimasi peak training atau menyimpulkan 2× batch berarti
2× seluruh VRAM. Padding, panjang encoder/decoder dan forward/backward menentukan
biaya variabel. Batch2 adalah kandidat yang diuji preflight, bukan kapasitas
E4B–E4B yang sudah disertifikasi. Batch4 belum diaktifkan.

Sebagai ilustrasi ketidakpastian, menambahkan selisih peak pilot sekali lagi
memberi `34,42 + 2 × (40,53 − 34,42) ≈ 46,64 GiB` untuk batch1 E4B–E4B.
Itu **ekstrapolasi linear yang belum divalidasi**, bukan prediksi kapasitas:
upgrade encoder tidak identik dengan upgrade decoder, dan word tying berbeda
pada pasangan simetris/asimetris. Karena itu angka ini tidak dipakai sebagai
syarat kelulusan atau alasan mengaktifkan batch4. Probe model asli yang menentukan.

Benchmark perlu implementasi padding/position, EOS per contoh, active sequences,
token budgets dan journal/resume per ID sebelum batch dinaikkan. QA jawaban pendek
adalah kandidat pertama. Completion/repair panjang memerlukan uji terpisah;
inference tanpa temporal KV cache juga menjadi bottleneck kecepatan sekarang.

## Apakah data dan metodologi cukup?

Lihat [spesifikasi v3](CONTEXT_DATA_RELEASE_V3.md) untuk angka terperinci.
Pool 49.342 contoh (44.408 train / 4.934 dev), 16,68 juta target token train,
cukup untuk **pilot direct SFT**, tetapi belum menjamin adaptasi encoder baru
atau kualitas repository repair. Dari seluruh pool, 44.956 contoh adalah editing
foundation, sedangkan hanya 4.386 berasal dari derivasi repository: 616 repair,
1.285 localization, 2.485 dependency completion. Sekitar 8,9% row repository
tidak otomatis berarti 8,9% gradien: objective sekarang token-weighted.
Median context encoder 343 token; kapasitas 32K bukan berarti mayoritas training
berisi pemahaman repository panjang.

Delapan protokol dari lima keluarga benchmark sudah cukup luas untuk pilot:
SWE-QA oracle/noisy oracle, CrossCodeLongEval chunk/function, LongCodeQA,
HumanEvalFix, GSM8K zero/eight-shot. Dua protokol QA memakai pertanyaan sama,
dua GSM8K memakai test sama. 30.600 kasus/protokol bukan 30.600 masalah independen.
[Repoformer](https://github.com/amazon-science/Repoformer) menguji completion;
skor ini bukan bukti test-passing repair repository. LongCodeQA 107 dan
HumanEvalFix 164 contoh juga perlu confidence interval, bukan selisih kecil saja.

Prioritas sebelum klaim paper final:

1. Selesaikan empat pasangan dan kedua baseline SFT dengan data, jadwal,
   budget update dan versi kode identik; tambahkan untouched IT sebagai kontrol.
2. Bandingkan context benar vs tanpa dependensi vs shuffled context pada cohort
   tetap; laporkan per panjang dan kebutuhan lintas file. Bekukan aturan sebelum
   melihat hasil, jangan memilih kasus yang menguntungkan model tertentu.
   Tambahkan encoder causal vs bidirectional dengan jumlah parameter sama;
   kemenangan pasangan lebih besar saja tidak mengisolasi manfaat bidirectionality.
3. Perbesar/kuatkan repository repair dengan tes fungsional native yang tervalidasi
   dan split repository; gold patch apply/compile saja belum membuktikan bug fixed.
4. Audit target-token share per objective/source. Pertimbangkan pembandingan
   mixture yang menambah exposure dependency, dengan budget token setara;
   jangan langsung menduplikasi row atau mengubah sampling tanpa eksperimen kontrol.
5. Jalankan beberapa seed untuk konfigurasi utama dan interval ketidakpastian
   berkelompok per repository, karena pertanyaan dari repository sama berkorelasi.
6. Ablasi recipe direct joint SFT vs adaptation stage terpisah hanya setelah
   baseline stabil. Scheduler dan batch tidak menggantikan adaptasi UL2 skala besar.

Dataset, benchmark cohort, objective, LoRA rank, optimizer dan source revisions
belum diubah pada langkah ini. Penambahan data atau adaptation stage perlu versi
eksperimen sendiri agar kontribusi scheduler/batching dapat dipisahkan.
