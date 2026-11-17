# Dokumentasi BifocalGemma

Implementasi aktif memakai modul tanpa direktori versi: model di
[`modeling_bifocalgemma.py`](../src/bifocalgemma/modeling_bifocalgemma.py),
bridge/cross-attention di `models/`, serta pipeline di `training/`.
Matriks aktif berisi delapan pasangan encoder–decoder dan dua baseline decoder-only.

## Panduan aktif

| Dokumen | Isi |
| --- | --- |
| [README proyek](../README.md) | Matriks sepuluh model, setup, API, dan batas klaim |
| [Peta modul](REPO_LAYOUT.md) | Lokasi implementasi dan entry point |
| [Notebook training](TRAINING_NOTEBOOKS.md) | Satu konfigurasi per notebook, schedule, benchmark, resume |
| [Rilis runtime](RUNTIME_RELEASE.md) | ZIP immutable, checksum, publikasi tanpa mengubah runtime lama |
| [Audit arsitektur Astra, 16–17 November 2026](ASTRA_BIFOCAL_AUDIT_2026-11-16.md) | Putusan perubahan, probe reproduksi, masalah yang tersisa, dan eksperimen peningkatan |
| [Audit benchmark resmi, 17 November 2026](OFFICIAL_BENCHMARK_PROTOCOL_AUDIT_2026-11-17.md) | Skor original Google yang tersedia, laporan pembuat benchmark, perbedaan protokol, dan bug diagnostik |
| [Pembanding model di bawah 7B, 18 November 2026](SUB7B_BENCHMARK_REFERENCES_2026-11-18.md) | Skor terverifikasi SWE-QA, GSM8K, HumanEvalFix dan CrossCodeLongEval, dengan status checkpoint dan protokol |
| [Paper](../paper/main.tex) | Arsitektur, rancangan eksperimen, dan placeholder hasil |
| [Dataset releases](DATASET_RELEASES.md) | Dataset privat, revision immutable, train/eval gold separation |
| [Dashboard lifecycle](DASHBOARD_WIDGET_LIFECYCLE.md) | Dependency guard dan widget host |
| [Repository study](REPOSITORY_STUDY_IMPLEMENTATION.md) | Workflow studi terpisah dan riwayat validasinya |

## Riwayat rancangan dan eksperimen

[Arsitektur lama](ARCHITECTURE.md), [build guide](BUILD_GUIDE.md),
[V2](BIFOCALGEMMA_V2.md), [sequential training](SEQUENTIAL_TRAINING.md),
[benchmark resume 2026-10-26](BENCHMARK_RESUME.md),
[perbandingan recipe](TRAINING_RECIPE_COMPARISON.md), dan
[proposal V3](BIFOCALGEMMA_V3_RESEARCH_PROPOSAL.md) menyimpan keputusan generasi
sebelumnya. Nama modul, notebook, command, dan default di dalamnya dapat sudah
dipensiunkan; gunakan panduan aktif di atas untuk checkout saat ini.

Dokumen [data](DATA_PIPELINE.md), [kandidat dataset](RESEARCH_SFT_DATASET_CANDIDATES.md),
[audit dataset](RESEARCH_SFT_DATASET_AUDIT.md),
[riset repository](RESEARCH_REPOSITORY_DATA_AND_BENCHMARKS.md),
[log riset](RESEARCH_LOG.md), [validasi](VALIDATION.md), dan
[hasil Colab](COLAB_RESULTS.md) tetap menjadi catatan sumber dan bukti bertanggal.
Hasil smoke/scorer atau benchmark generasi lama tidak menjadi hasil model untuk
implementasi baru.

## Status dan batas bukti

Tes numerik, type checking, generator, dan marimo memeriksa perilaku implementasi.
Kualitas repair, penggunaan VRAM penuh, dan klaim perbandingan memerlukan hasil
training/evaluasi yang dicatat dengan identitas runtime, dataset, dan checkpoint.
Placeholder hasil paper tetap kosong sampai bukti tersebut tersedia.

Runtime lama yang sedang berjalan di Molab harus mempertahankan notebook dan
artefak lamanya. Baseline kode bersih menolak checkpoint/schedule sebelum refactor.
Rilis baru menambahkan ZIP immutable; tidak mengganti `runtime/latest.zip`.
