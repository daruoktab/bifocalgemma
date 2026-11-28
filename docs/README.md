# Dokumentasi BifocalGemma

Implementasi aktif memakai modul tanpa direktori versi: model di
[`modeling_bifocalgemma.py`](../src/bifocalgemma/modeling_bifocalgemma.py),
bridge/cross-attention di `models/`, serta pipeline di `training/`.
Matriks aktif V4 berisi delapan pasangan encoder–decoder (`sliding-v4/`) dan dua baseline decoder-only (`latest/`).

## Panduan aktif

| Dokumen | Isi |
| --- | --- |
| [README proyek](../README.md) | Matriks sepuluh model V4, setup, API, dan batas klaim |
| [Peta modul](REPO_LAYOUT.md) | Lokasi implementasi dan entry point |
| [Notebook training](TRAINING_NOTEBOOKS.md) | Satu konfigurasi per notebook, schedule V4, 10 benchmark, resume |
| [Rilis runtime](RUNTIME_RELEASE.md) | ZIP immutable, checksum, publikasi tanpa mengubah runtime lama |
| [Data pipeline](DATA_PIPELINE.md) | Kontrak format data, collator left-padded, dan pemisahan konteks |
| [Build guide](BUILD_GUIDE.md) | Panduan komponen konstruksi dan verifikasi blok model |
| [Arsitektur awal](ARCHITECTURE.md) | Catatan desain awal embedding dan perhatian terpisah |
| [Dataset releases](DATASET_RELEASES.md) | Dataset privat, revision immutable, train/eval gold separation |
| [Dashboard lifecycle](DASHBOARD_WIDGET_LIFECYCLE.md) | Dependency guard dan widget host |
| [Repository study](REPOSITORY_STUDY_IMPLEMENTATION.md) | Workflow studi terpisah dan riwayat validasinya |
| [Audit arsitektur Astra](ASTRA_BIFOCAL_AUDIT_2026-11-16.md) | Audit arsitektur dan probe reproduksi |
| [Audit benchmark resmi](OFFICIAL_BENCHMARK_PROTOCOL_AUDIT_2026-11-17.md) | Perbandingan protokol benchmark terhadap rilis pembuat asli |
| [Pembanding model <7B](SUB7B_BENCHMARK_REFERENCES_2026-11-18.md) | Referensi skor SWE-QA, GSM8K, HumanEvalFix, dan CrossCodeLongEval |
| [Lanjutan Qwen3.5 & Gemma 4](QWEN35_GEMMA4_BENCHMARK_FOLLOWUP_2026-11-18.md) | Catatan protokol GSM8K checkpoint asli |
| [Paper](../paper/main.tex) | Manuskrip arsitektur, rancangan eksperimen V4, dan placeholder hasil |

Catatan riwayat eksperimen terdahulu yang masih dirujuk: [SEQUENTIAL_TRAINING.md](SEQUENTIAL_TRAINING.md), [TRAINING_RECIPE_COMPARISON.md](TRAINING_RECIPE_COMPARISON.md).
