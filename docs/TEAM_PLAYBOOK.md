# BifocalGemma: Panduan Tim

**Tujuan:** eksperimen perbaikan bug lintas file yang dapat direproduksi, dengan
laporan kualitas dan biaya komputasi. Jadwal submission mengikuti ketentuan yang
telah diverifikasi dari [halaman kompetisi](COMPETITION_GUIDE.md).

## Pembagian kerja

| Anggota | Tanggung jawab |
| --- | --- |
| [daruoktab](https://github.com/daruoktab) | Audit bobot/memori, integrasi backbone dan bridge, konfigurasi adaptasi serta checkpoint |
| [attmhd](https://github.com/attmhd) | Kurasi data, baseline dan harness, analisis hasil, reproduksi serta penulisan paper |

## Setup dan pemeriksaan

```powershell
git clone https://github.com/daruoktab/bifocalgemma.git
cd bifocalgemma
conda create -n ai python=3.12
conda activate ai
python -m pip install -e ".[dev]"
```

Environment `ai` yang sudah ada cukup diaktifkan. Pilih build PyTorch sesuai
hardware; tes modul memakai CPU dan tensor sintetis, tanpa bobot Gemma.
Pin versi Transformers yang mendukung Gemma 4 saat tahap integrasi backbone.

```powershell
python -m unittest discover -s tests
ruff check .
ruff format --check .
ty check .
git diff --check
```

Lulus checks hanya memberi bukti pada hal yang diperiksa. Itu tidak menjamin
proyek bebas bug, agen berfungsi, ataupun kualitas patch. `ruff format .` dapat
digunakan untuk memperbaiki format sebelum menjalankan pemeriksaan ulang.

## Milestone berdasarkan bukti

| Tahap | Hasil yang harus tersedia sebelum lanjut |
| --- | --- |
| 1. Fondasi | Modul bridge, AST embedding, routing, dan laporan skor; cakupan tes sintetis dijelaskan |
| 2. Audit biaya | Parameter fisik/trainable, ukuran checkpoint, precision, offloading, peak VRAM dan latensi baseline |
| 3. Integrasi | Pilot E2B–E2B lalu empat pasangan base encoder / IT decoder; attention, forward/generasi dan checkpoint dapat direproduksi |
| 4. Adaptasi | Split data tanpa leakage (CommitPackFT, SWE-smith), konfigurasi training sesuai profiling, log dan checkpoint yang terlacak |
| 5. Evaluasi | Perbaikan lintas file pada held-out repair tasks (SWE-bench Lite/Verified) dan completion diagnostics (CrossCodeEval), baseline sebanding, cold/warm caching, ablation dan analisis kegagalan |
| 6. Paper/demo | Metode, sumber, hasil terukur dan batasan; demo issue → file terkait → patch → tes |

Tahap pertama memiliki kode dan tes yang tersedia; status lulus tes harus dicatat
dari eksekusi sebenarnya. Tahap berikutnya merupakan rencana, bukan pekerjaan yang
otomatis selesai karena unit test bridge lulus. Keempat pasangan adalah target studi;
ukuran data, precision, urutan eksekusi dan budget ditetapkan melalui profiling.

## Protokol bersama

- Gunakan dataset/split, konteks, sampling, token/tool budget, dan harness sebanding
  untuk baseline; berikan prefix caching jika backend mendukungnya.
- Pisahkan skor referensi publik dari pengukuran sendiri. Nilai belum tersedia
  memakai `N/A`; angka demonstrasi harus diberi label ilustrasi.
- Catat hardware, environment, revision model, seed, precision, peak VRAM, dan log
  kegagalan. HumanEval/MBPP bukan pengganti evaluasi repositori lintas file.
- Re-encode snapshot penuh setelah edit untuk eksperimen awal. Partial invalidation,
  multimodal merupakan tahap lanjutan. Metode konsolidasi bertahap menargetkan
  merged attention dan word tying hanya pada pasangan simetris, dengan checkpoint
  teacher/provenance dan adaptasi yang perlu diimplementasikan. Matriks 2×2 dan studi
  pengaruh encoder/decoder termasuk studi utama bersama baseline E2B-IT/E4B-IT.

## Sinkronisasi Git

```powershell
git pull --ff-only origin main
git switch -c codex/deskripsi-perubahan
git add <file-yang-diubah>
git commit -m "docs: jelaskan perubahan"
git push -u origin HEAD
```

Gunakan branch baru untuk pekerjaan terpisah dan tinjau diff sebelum commit.
Koordinasikan perubahan antarmuka paket dan config eksperimen agar baseline tetap
dapat direproduksi. Jadwal paper ditetapkan ulang setelah deadline resmi dikonfirmasi.

Gunakan [riset dataset dan benchmark](DATASETS_AND_BENCHMARKS.md) untuk manifest,
audit overlap, serta pemisahan inference model dan evaluasi patch di Docker.

Perbandingan model mengikuti [desain enam model](EXPERIMENT_DESIGN.md). T5Gemma
dan T5Gemma 2 adalah related work, bukan kandidat tambahan pada benchmark utama.
