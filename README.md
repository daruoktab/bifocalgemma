# BifocalGemma: Local Code Repair Research

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![GitHub](https://img.shields.io/badge/GitHub-bifocalgemma-green)](https://github.com/daruoktab/bifocalgemma)

**Paper title:** *BifocalGemma: Evaluating Bidirectional Repository Encoding for Local Code Repair*

BifocalGemma meneliti apakah representasi repositori bidirectional dapat membantu model
lokal memperbaiki bug lintas file pada anggaran komputasi terbatas. Encoder direncanakan
untuk memahami kode, sedangkan decoder kausal menghasilkan patch melalui cross-attention.
Keunggulan kualitas, latensi, dan memori masih merupakan hipotesis yang perlu diuji.

## Status implementasi

| Bagian | Yang tersedia sekarang | Tahap berikutnya |
| --- | --- | --- |
| Cross-attention bridge | Proyeksi Q/K/V, normalisasi, gated residual | Integrasi layer Gemma dan generasi |
| Fitur AST | Embedding tipe node dan kedalaman dari ID masukan | Ekstraksi Tree-sitter dan pemetaan ke token |
| Routing | Pemisahan tensor konteks/target dan mask validitas token | Pengaturan attention encoder bidirectional |
| Estimasi self-KV | Ilustrasi layer full-attention seragam | Pengukuran memori sistem sebenarnya |
| Pelaporan skor | Tabel skor masukan dan referensi bersumber | Harness benchmark dan agent loop |

Tes memakai tensor sintetis, bridge dengan lebar hidden asli, serta kelas Gemma 4
dengan bobot acak. Struktur penuh diperiksa pada meta; runtime full-depth memakai
width dan vocabulary kecil. Tidak ada bobot pretrained dimuat. Belum ada backbone Gemma terintegrasi,
training pipeline, checkpoint BifocalGemma, repository cache, atau hasil benchmark proyek.
Mask token semua-valid tidak otomatis menghapus causal mask atau sliding window backbone.

## Fokus eksperimen

Eksperimen pertama memakai kandidat **BifocalGemma-E2B-E2B**, setelah kebutuhan memori
dan kelayakan integrasinya diukur. Fokusnya adalah pemahaman dependensi lintas file,
lokasi bug, dan patch yang lolos tes. Baseline decoder-only harus dijalankan dengan
data, konteks, sampling, token/tool budget, dan harness yang sebanding, termasuk prefix
caching jika didukung. Ukur kualitas perbaikan, latensi cold/warm cache, throughput,
dan peak VRAM total; HumanEval/MBPP hanya evaluasi tambahan, bukan bukti utama.

Konfigurasi **BifocalGemma-E4B-E2B**, **BifocalGemma-E2B-E4B**, dan
**BifocalGemma-E4B-E4B** merupakan eksperimen lanjutan. Perbandingan kapasitas encoder
versus decoder membutuhkan kedua konfigurasi asimetris. Multimodal dan pembaruan
cache satu hop juga ditunda sampai eksperimen dasar dapat direproduksi.

## Dasar ilmiah dan batas klaim

[T5Gemma](https://developers.googleblog.com/en/t5gemma/) sudah mengadaptasi model
decoder-only menjadi encoder-decoder, termasuk konfigurasi asimetris. Kontribusi
BifocalGemma harus dibuktikan melalui metode atau hasil khusus perbaikan kode.
Dalam causal attention, representasi token awal tidak melihat token sesudahnya,
tetapi token jawaban dapat mengakses konteks sebelumnya; decoder-only tetap dapat
memahami dependensi lintas file. [Prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
juga dapat menggunakan kembali komputasi prefix yang sama pada baseline.

Pengurangan decoder **self-attention KV** tidak sama dengan pengurangan total VRAM.
Encoder states, cross-attention K/V, bobot, aktivasi, dan overhead tetap harus dihitung.
Parameter efektif juga berbeda dari jumlah parameter fisik dan ukuran checkpoint;
lihat [referensi backbone](docs/GEMMA4_EDGE_ARCHITECTURE.md).

## Setup

```powershell
git clone https://github.com/daruoktab/bifocalgemma.git
cd bifocalgemma
conda create -n ai python=3.12
conda activate ai
python -m pip install -e ".[dev]"
```

Jika environment `ai` sudah ada, cukup aktifkan environment tersebut. Instalasi ini
memasang dependensi library; tidak mengunduh bobot Gemma. Versi Transformers yang
mendukung backbone sebenarnya harus diverifikasi saat tahap integrasi.

## Pemeriksaan modul

```powershell
python -m unittest discover -s tests
ruff check .
ruff format --check .
ty check .
```

Unit test tidak memerlukan GPU atau bobot model, tetapi memerlukan PyTorch.
Lulus pemeriksaan modul bukan bukti keberhasilan agen atau kualitas patch.

Contoh berikut hanya menunjukkan format pelaporan; angka **ilustratif, bukan hasil eksperimen**:

```powershell
python -c "from bifocalgemma.evaluation import print_comparison_table; print_comparison_table({'humaneval': 63.4, 'mbpp': 56.1}, illustrative=True)"
```

Skor yang belum terverifikasi atau belum diukur ditampilkan sebagai `N/A`, bukan nol.

## Dokumen dan kontributor

- [Proposal riset](docs/proposal.md)
- [Spesifikasi teknis dan batas estimasi memori](docs/RESEARCH_SPEC.md)
- [Referensi arsitektur Gemma 4](docs/GEMMA4_EDGE_ARCHITECTURE.md)
- [Riset dataset training dan benchmark repair](docs/DATASETS_AND_BENCHMARKS.md)
- [Panduan tim](docs/TEAM_PLAYBOOK.md)
- [Catatan kompetisi dan status verifikasi](docs/COMPETITION_GUIDE.md)

Kontributor: [daruoktab](https://github.com/daruoktab) dan [attmhd](https://github.com/attmhd).
Kode proyek menggunakan [Apache License 2.0](LICENSE).
