# Tes GPU G4 melalui Colab

Tes ini hanya mengukur E4B base → E4B IT: pemeriksaan numerik backend, waktu update,
peak allocated/reserved, dan guard bentuk maksimum. Ia tidak memanggil training
epoch, tidak membuat checkpoint riset, dan tidak memublikasikan profil ke HF.
Update optimizer diagnostik dikembalikan; SHA256 parameter trainable sebelum dan
sesudah dibandingkan. Memakai data, revision backbone dan resep Molab yang sama.

CLI terpasang di `C:/Users/daru/.local/bin/colab.exe`. Login pertama:

```powershell
colab sessions
```

Buka tautan Google yang ditampilkan, login akun Colab, dan tempel kode otorisasi
ke terminal yang menunggu. Jangan tempel kode ke chat atau source. Token Google
disimpan oleh CLI di konfigurasi pengguna. `colab auth` bukan login CLI.

Pada akun ini, `--gpu G4 --high-mem` ditolak, sedangkan `--gpu G4` berhasil.
Sesi menghasilkan RTX PRO6000 Blackwell,97887MiB VRAM, RAM sistem sekitar177GiB,
Python3.13.16, driver580.82.07/CUDA13.0. Label GPU tetap diperiksa saat setiap tes;
skrip menolak GPU lain atau kapasitas di bawah85GiB. Ini tidak mengasumsikan
label G4 selalu memberi konfigurasi yang sama di masa depan.

Siapkan skrip menggunakan conda ai:

```powershell
& C:/Users/daru/miniconda3/envs/ai/python.exe tools/prepare_colab_probe.py
$env:PYTHONUTF8='1'
colab new --gpu G4 --session bifocalgemma-test
```

Script `outputs/colab/probe.py` mengunduh runtime publik yang revision dan checksum
nya dipin di manifest notebook. Dependency dipasang melalui `uv pip --python`
ke Python runtime Colab, dalam subprocess segar agar paket bawaan kernel tidak
tercampur. Lingkungan lokal tetap conda ai. CLI saat ini resmi mendukung Linux/
macOS; perintah noninteraktif ini diuji pada Windows, bukan jaminan seluruh CLI.

Backbone gated membutuhkan HF token yang sudah diberi akses model. Pengiriman
kredensial lokal ke runtime harus disetujui pengguna secara eksplisit; jangan
menaruh token dalam argumen command, source, atau log. Skrip dapat membaca token
di environment atau file sementara `/content/.bifocalgemma-hf-token`, lalu menghapus
file setelah dibaca. File sementara lokal juga harus segera dihapus sesudah upload.

```powershell
colab exec --session bifocalgemma-test --file outputs/colab/probe.py --timeout 21600
```

Guard pertama memakai maksimum panjang yang benar-benar terdapat pada train/dev.
Pada data ini, input decoder terpanjang model pasangan6207token (sesudah shift
input/target), walaupun cap konfigurasinya8192. Ini tidak membuktikan batas8192.
Untuk memeriksa batas konfigurasi secara terpisah:

```powershell
& C:/Users/daru/miniconda3/envs/ai/python.exe tools/prepare_colab_probe.py --boundary
colab exec --session bifocalgemma-test --file outputs/colab/boundaries.py --timeout 21600
colab download --session bifocalgemma-test /content/bifocalgemma-test.zip outputs/colab/results.zip
colab stop --session bifocalgemma-test
```

Jangan memakai `colab run` dengan penghentian otomatis sebelum arsip hasil diunduh.
Hentikan sesi setelah hasil disalin, termasuk bila tes gagal. Folder remote
`/content/bifocalgemma-test/` menyimpan status/events, hardware, manifest, receipt
kalibrasi per kandidat, profil diagnostik, dan laporan pemulihan bobot. Arsip hasil
mengecualikan cache model/data dan file token. Crash keras tetap dapat menghilangkan
hasil yang belum diunduh dari VM; script tidak mengklaim arsip sudah aman di lokal.

Boundary guard memakai32768token encoder dan8192input decoder untuk empat
model pasangan; baseline memakai cap input41472 dari manifest karena sumber dan
target berada di decoder yang sama. Token prefix/konteks diulang hanya untuk
shape stress, tanpa memotong maksimum nyata atau memasukkan contoh sintetis ke
training/cohort pengukuran. Loss tetap menilai target asli. Profil lokal hanya
diganti dengan bukti batas ini jika keenam model lolos dengan setting yang sama.
Tidak ada publikasi HF otomatis.

Profil diagnostik baru boleh dipakai bersama setelah hardware, versi dependency,
kontrak source/data, hasil numerik, margin dan pemulihan bobot ditinjau. Lima model
lain tetap wajib guard bentuk maksimum pada profil yang sama; konteks baseline
lebih panjang dapat menimbulkan kebutuhan memori berbeda.

Referensi: [CLI resmi](https://github.com/googlecolab/google-colab-cli),
[implementasi autentikasi](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/auth.py),
[Google Cloud G4](https://docs.cloud.google.com/compute/docs/gpus/).
