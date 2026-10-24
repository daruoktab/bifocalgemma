# Monitoring Wigglystuff, waktu lintas sesi, dan runtime ringkas

Riset dan pemeriksaan paket lokal: 24 Oktober 2026, `wigglystuff==0.5.36` pada conda
`ai`. API diverifikasi melalui source paket terpasang dan dokumentasi penulis.

## Pilihan widget

| Komponen | Penggunaan | Batas penting |
|---|---|---|
| [ProgressBar](https://koaning.github.io/wigglystuff/reference/progress-bar/) | Step optimizer dan progres tahap berjalan | Dua bar berbeda; persentase optimizer bukan persentase seluruh pekerjaan |
| [ScatterLog](https://koaning.github.io/wigglystuff/reference/scatter-log/) | Train/dev CE, tooltip presisi, zoom/pan | Memori widget bukan log tahan crash; grafik diisi kembali dari history checkpoint. Maksimal 1.000 titik ditampilkan |
| [HTMLRefreshWidget](https://koaning.github.io/wigglystuff/reference/html-refresh/) | Kartu metrik ringkas dan log terbaru | Mengganti `innerHTML`; accordion dan chart diletakkan di luar kartu agar tidak ikut reset |
| [NestedTable](https://koaning.github.io/wigglystuff/reference/nested-table/) | Distribusi token dev per sumber dan per task | Dua tabel terpisah. CE ditulis di label leaf; tidak menjumlahkan mean CE atau menghitung share-of-root untuk CE |
| TraceView, source paket `wigglystuff/trace_view.py` | Span tahap berdasarkan event | Manual span, bukan profiler baris Python atau collector jaringan. Maksimal 200 span per sesi |
| [FloatingPanel](https://koaning.github.io/wigglystuff/reference/floating-panel/) | Status terakhir tetap terlihat saat scroll | Bekerja di iframe Molab. `Pip` mempunyai batas top-level page sehingga tidak dipilih |

Widget dibuat sekali di cell tersendiri; training hanya memperbarui traitlet.
Dashboard yang sama dipakai dari training hingga benchmark. UI gagal tidak mengubah
gradien: log dan `monitor.html` tetap tersedia. Preview diuji di browser lokal dengan
data simulasi, bukan GPU96GiB. Label waktu di panel mengambang bertambah dari browser;
setelah lima menit tanpa event tertulis **tidak ada kabar**, bukan dugaan kernel mati.

Pembaruan25Oktober: dashboard membedakan allocated, reserved, peak allocated dan
driver free. Konfigurasi batch/backend terlihat di kartu dan heartbeat; counter
tokenisasi tidak terbawa ke tahap lain. Pemeriksaan awal kini dua tahap dengan
receipt per tahap, bukan sweep backend. [Detail perbaikan](EXECUTION_FIXES.md).

`LiveEdit` dan `StateViewer` tidak dipakai untuk membungkus training karena tracing
Python per baris dan penyalinan variabel besar menambah overhead yang tidak perlu.
TraceView dibangun dari callback tahap yang sudah ada.

## Arti pencatatan waktu

Implementasi sebelumnya memakai `perf_counter() - segment_started`: benar untuk
satu sesi, tetapi kembali dari nol ketika proses baru memulihkan checkpoint.

Sekarang `DurableClock` menyimpan `timing.json` secara atomik pada event, dengan
interval lima detik dan penyimpanan paksa pada batas penting. Durasi dalam checkpoint
disimpan sebagai `training_state.accumulated_seconds`. Acuan monotonic satu sesi tidak
direset saat save; elapsed = basis tersimpan + durasi sesi, sehingga tidak dihitung
dua kali. Jika jurnal lebih maju daripada checkpoint, waktu kerja yang kemudian
diulang tetap dihitung sebagai biaya komputasi teramati.

- **Aktif teramati**: akumulasi waktu proses berjalan, termasuk validasi/save/sync
  yang berlangsung di dalam panggilan training. Bukan waktu GPU-kernel murni.
- **Sesi ini**: durasi panggilan training sekarang saja; tetap tersedia terpisah.
- **Jeda tidak teramati**: selisih timestamp UTC terakhir dengan awal sesi baru.
  Ini bisa memuat downtime dan ekor aktif yang belum sempat disimpan; bukan ukuran
  downtime yang diketahui persis.
- **Wall span**: rentang kalender sejak pengamatan pertama; bukan durasi training.
- **ETA optimizer**: rata-rata 20 update terakhir × sisa update. Tidak menjanjikan
  durasi validasi, upload, gangguan jaringan, atau downtime berikutnya.

Benchmark mempunyai clock tersendiri per collection; ETA dihitung dari contoh baru
di sesi sekarang, sedangkan elapsed dapat berlanjut lintas sesi. Waktu tiap kasus
dicatat bersama prediksi. Tidak menjumlahkan durasi training dan benchmark dari field
yang mempunyai cakupan berbeda.

Hard crash sebelum event berikutnya tetap meninggalkan ekor waktu tak diketahui.
Interval lima detik bukan jaminan kehilangan maksimal lima detik: satu operasi GPU
yang lama dapat tidak menghasilkan callback. HF hanya melindungi jurnal sampai
snapshot remote terakhir yang terverifikasi, sesuai interval upload. Jam UTC mesin
bisa berubah; durasi aktif dihitung monotonic dan gap negatif dibatasi nol.

## Hash dan notebook ringkas

Blok ribuan baris sebelumnya adalah ZIP source yang diubah menjadi teks Base85.
Hash sesungguhnya adalah `bundle_sha` sepanjang 64 karakter. Hash mengikat versi
runtime dan mendeteksi korupsi; tidak mengenkripsi atau menggantikan autentikasi.

Runtime baru berupa ZIP terpisah, berisi source/config/contoh proyek, bootstrap,
launcher, serta LICENSE. Kandidat lokal dapat dibuat dengan:

```powershell
C:/Users/daru/miniconda3/envs/ai/python.exe tools/generate_training_notebooks.py --stage
```

Publikasi yang sudah diotorisasi menggunakan `--publish`. Generator mengunggah ke
`daruokta/bifocalgemma-v2-assets`, membaca kembali arsip pada commit tertentu,
memverifikasi SHA256, lalu merekam pointer pada `notebooks/runtime-manifest.json`.
Notebook hanya memuat loader kecil dan pointer itu; kode bootstrap disembunyikan
di tampilan marimo. Versi tampilan memakai 12 karakter, verifikasi tetap 64 karakter.

Loader memilih arsip lokal bila ada, atau cache, atau unduhan commit HF yang dipin.
Checksum diperiksa sebelum ekstraksi/aktivasi. Path keluar folder dan symlink ditolak.
`main/latest` tidak dipakai. `--check` menolak notebook dengan artifact yang tidak
sesuai source sekarang. Arsip bukan checkpoint model atau data training pribadi.

## Kompatibilitas dan provenance

Fingerprint penuh byte kode tetap direkam di `diagnostics/code_provenance.json`.
Identitas resume memakai fingerprint AST kode training/model dan JSON kanonik;
komentar, whitespace, docstring, serta modul tampilan/clock tidak mengubahnya.
Perubahan objective, optimizer, model, mask, data, revision dan policy tetap ditolak.
Tidak ada sakelar generik yang mengabaikan seluruh code drift.

Transisi dari identitas byte-hash lama ke `training_semantics_v1` tidak secara
diam-diam mengesahkan checkpoint eksperimen lama. Receipt yang memakai skema lama
tetap ditolak bila identitas tidak cocok. Patch tampilan berikutnya pada run baru
tidak mengubah kompatibilitas training.

## Hasil verifikasi lokal

91 tes terkait lolos; tujuh tes clock/widget/kandidat notebook diulang setelah
koreksi terakhir dan juga lolos. Pyrefly, ty, Ruff serta marimo strict keenam
kandidat bersih. Keenam kandidat berukuran 179 baris dan tersedia di
`outputs/runtime-release/preview/`. Bukti tampilan (data simulasi):
`docs/images/wiggly-dashboard-preview.png`.

Runtime pertama (24 Oktober) telah diterbitkan ke dataset publik
[`daruokta/bifocalgemma-v2-assets`](https://huggingface.co/datasets/daruokta/bifocalgemma-v2-assets/tree/39dc454c57268dd298f21e7d139f3debdd7b1041),
commit `39dc454c57268dd298f21e7d139f3debdd7b1041`. Unduhan tanpa token melalui
loader notebook berhasil dan checksum arsip 259.502 byte cocok. Keenam notebook
aktif sudah diregenerasi dengan pointer commit tersebut; masing-masing 179 baris.
Pyrefly, ty, Ruff, marimo strict keenam notebook aktif, serta generator `--check`
juga lolos setelah publikasi. Sebelas tes notebook/alur training-benchmark pada
notebook aktif lolos (79,73 detik), termasuk pemulihan dan melewati pekerjaan yang
sudah selesai. Belum ada full training atau benchmark Molab baru.

Runtime aktif 25 Oktober memakai commit
`521b546c0f078d2fb4a74c2fe45323b5a34d592b`, arsip
`runtime-2cc7b98cf02d97d2.zip` berukuran268581byte. Keenam notebook aktif menunjuk
revision ini. Ia mencakup kalibrasi E4B→E4B untuk satu profil bersama, receipt
resumable dan tampilan allocated/reserved/free yang terpisah. Setelah publikasi,
26 tes notebook/dependency/telemetri, generator `--check` dan marimo strict keenam
notebook aktif lolos. Tes G4 menemukan konflik default device CUDA dengan
inisialisasi alignment CPU; konteks inisialisasi kini eksplisit CPU.24 tes model
lolos sesudah perbaikan, termasuk konstruksi alignment langsung di CUDA.
[Rincian perubahan eksekusi](EXECUTION_FIXES.md).

Kalibrasi nyata G4 dan guard batas penuh keenam model kini selesai; hasil rinci
ada di [laporan GPU](COLAB_RESULTS.md). Satu profil HF terverifikasi telah tersedia,
sehingga keenam notebook langsung mengambil setting yang sama jika kontraknya
cocok. Regresi akhir116 tes serta enam pemeriksaan rilis profil lolos. Runtime
Colab dihentikan setelah bukti hasil disalin; belum ada epoch training/benchmark
kualitas baru yang dijalankan oleh pengukuran ini.
