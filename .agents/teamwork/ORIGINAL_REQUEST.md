# Original User Request

## 2026-10-07T09:43:39Z

Cari dan evaluasi dataset tambahan untuk SFT BifocalGemma (instruction-to-patch, code editing/repair Python) yang selaras dengan benchmark evaluasi repo ini (HumanEvalFix/HumanEvalPack, CanItEdit, CrossCodeEval), dengan tujuan menaikkan skor benchmark tanpa kontaminasi. Hasilnya berupa laporan riset terverifikasi; tidak ada perubahan pada kode notebook/sumber.

Working directory: d:\Codings\bifocalgemma
Integrity mode: development

## Konteks repo

- Dataset saat ini di kedua notebook training: `nuprl/EditPackFT` (default), `bigcode/commitpackft` (python), `local_jsonl`.
- Benchmark evaluasi yang tercantum di `docs/DATASETS_AND_BENCHMARKS.md`: HumanEvalFix/HumanEvalPack, CanItEdit, CrossCodeEval.
- Kontrak data saat ini: input = instruksi + kode sebelum edit; target = unified diff; split grouped per commit; tanpa RL/VM/Docker.
- Hardware target: satu GPU Molab. Data tidak boleh menuntut RL, banyak model sekaligus, teacher/reward model, atau eksekusi environment yang menambah beban VRAM.

## Requirements

### R1. Kandidat dataset
Temukan dataset publik tambahan (di luar EditPackFT dan CommitPackFT yang sudah dipakai) yang berpotensi meningkatkan kemampuan instruction-based code editing/repair Python dan relevan dengan benchmark di atas. Setiap kandidat harus bisa dipakai untuk SFT offline murni: instruksi + kode sebelum + kode sesudah/patch, tanpa eksekusi, RL, VM, atau model tambahan.

### R2. Verifikasi dan kecocokan
Untuk setiap kandidat, verifikasi dari sumber aslinya (bukan dari ingatan): lisensi, jumlah record, field yang tersedia, revision yang bisa dipin, dan apakah bisa dipetakan ke kontrak data saat ini. Nilai juga relevansinya terhadap tiap benchmark (HumanEvalFix, CanItEdit, CrossCodeEval).

### R3. Bebas kontaminasi
Periksa risiko overlap tiap kandidat dengan ketiga benchmark (sumber data, commit/repo, teks soal/solusi). Hanya dataset yang terbukti atau sangat mungkin bebas overlap yang direkomendasikan untuk training; yang berisiko harus dilaporkan terpisah beserta alasannya dan tidak masuk rekomendasi.

### R4. Laporan dan rekomendasi
Hasilkan satu laporan terstruktur yang berisi kandidat berperingkat, bukti verifikasi, hasil cek overlap, estimasi ukuran dan beban (disk, waktu, VRAM), serta rekomendasi urutan pemakaian dan cara pencampuran dengan data yang sudah ada. Tandai jelas apa yang sudah terbukti dan apa yang masih hipotesis. Jangan memodifikasi notebook, `src/`, `tools/`, atau `tests/`.

## Acceptance Criteria

### Kelengkapan
- [ ] Minimal 5 kandidat dataset baru dievaluasi (tidak termasuk EditPackFT/CommitPackFT), atau ada penjelasan konkret bila kurang dari itu.
- [ ] Setiap kandidat punya: ID Hub/URL, lisensi, jumlah record, daftar field, revision, dan pemetaan field ke kontrak data repo.

### Verifikasi independen
- [ ] Setiap angka/klaim metadata (lisensi, ukuran, field) dapat dicocokkan dengan halaman/API sumber; reviewer terpisah memeriksa sampel minimal 3 kandidat terhadap sumber asli.
- [ ] Cek overlap dilakukan secara terukur (mis. hash/near-duplicate terhadap teks benchmark, atau analisis asal repo/commit), dengan metode dan hasil dicantumkan.

### Kontaminasi dan kepatuhan batasan
- [ ] Tidak ada dataset yang direkomendasikan tanpa hasil cek overlap terhadap HumanEvalFix, CanItEdit, dan CrossCodeEval.
- [ ] Tidak ada kandidat rekomendasi yang membutuhkan RL, reward/teacher model, banyak model, atau eksekusi VM/Docker.
- [ ] `git status` menunjukkan tidak ada perubahan pada notebook, `src/`, `tools/`, `tests/`.

### Kejujuran klaim
- [ ] Laporan tidak menjanjikan kenaikan skor benchmark; klaim "meningkatkan benchmark" ditandai hipotesis kecuali ada bukti ukuran (dari literatur terkutip atau eksperimen).
