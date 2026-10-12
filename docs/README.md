# Dokumentasi BifocalGemma

Dokumen di sini menjelaskan satu rancangan aktif: **langsung membangun encoder base
dan decoder IT dengan merged attention, lalu direct supervised fine-tuning**.
Word tying lintas encoder/decoder hanya dipakai pada pasangan simetris; PLE terpisah.
Distillation dan konversi dari attention terpisah adalah eksperimen opsional.

## Urutan membaca

| Dokumen | Pertanyaan yang dijawab |
| --- | --- |
| [Proposal](proposal.md) | Masalah, tiga kontribusi dan batas klaim |
| [Desain eksperimen](EXPERIMENT_DESIGN.md) | Empat pasangan, dua baseline, kontrol dan analisis |
| [Arsitektur](ARCHITECTURE.md) | Aliran tensor, merged attention, initialization dan tying |
| [Panduan pembangunan](BUILD_GUIDE.md) | Blok mana tersedia, placeholder mana dikerjakan berikutnya |
| [Pipeline data](DATA_PIPELINE.md) | Snapshot, prefix, target, offset AST, collator dan leakage |
| [Training notebooks](TRAINING_NOTEBOOKS.md) | Tiga standalone SFT notebooks dengan cell identik dan template bersama |
| [Sequential training](SEQUENTIAL_TRAINING.md) | Full SFT → upload → benchmark → upload, resume dan dua checkpoint per run |
| [Context data v3](CONTEXT_DATA_RELEASE_V3.md) | Pool gabungan, benchmark32K, provenance dan batas audit |
| [Repository study implementation](REPOSITORY_STUDY_IMPLEMENTATION.md) | Standalone prepare/SFT/generate/score notebook, native adapters dan validation |
| [Training](TRAINING.md) | Direct SFT, loss, profiling, PEFT, checkpoint dan opsi distillation |
| [Evaluasi](EVALUATION.md) | Inference, prediksi, harness, metrik dan tabel enam model |
| [Dataset dan benchmark](DATASETS_AND_BENCHMARKS.md) | Sumber primer dan alasan pemilihan data |
| [Paket data/evaluasi Molab](MOLAB_DATA_EVALUATION_PLAN.md) | Pilihan aktif, profil token, dependency benchmark dan resource single GPU |
| [Audit kandidat](RESEARCH_SFT_DATASET_AUDIT.md) | Koreksi dan batas bukti laporan riset awal |
| [Riset repository data dan benchmark](RESEARCH_REPOSITORY_DATA_AND_BENCHMARKS.md) | Katalog luas, RES-Q/RepoQA, core repo SFT, prior art dan gate resource |
| [Spesifikasi ringkas](RESEARCH_SPEC.md) | Ikhtisar kontrak, status dan batas estimasi memori |
| [Audit semua JSON](GEMMA4_JSON_AUDIT.md) | Base/IT, BOS/EOS, tokenizer, processor dan parameter fisik |
| [Referensi backbone](GEMMA4_EDGE_ARCHITECTURE.md) | Config E2B/E4B dan pengujian native tanpa bobot |
| [Log riset](RESEARCH_LOG.md) | Keputusan yang dipakai serta ide lama yang ditunda |
| [Panduan tim](TEAM_PLAYBOOK.md) | Cara mengembangkan dan meninjau tiap blok |
| [Pretrained GPU probe](PRETRAINED_GPU_PROBE.md) | Checkpoint base/IT asli, optimizer steps dan memory beberapa model resident |
| [Marimo / Molab](MARIMO_ARCHITECTURE_TEST.md) | Notebook standalone, cell map, tiny tests dan full-size probe |
| [Native lokal](NATIVE_LOCAL_VALIDATION.md) | Native tiny, tokenizer offline dan kontrak arah patch SWE-smith |
| [Validasi](VALIDATION.md) | Checks yang benar-benar dijalankan dan cakupannya |
| [Kompetisi](COMPETITION_GUIDE.md) | Ketentuan yang masih perlu dikonfirmasi |

## Cara membaca status

**Implemented** berarti fungsi tertentu bekerja sesuai kontraknya.
**Reference** berarti prototipe numerik kecil untuk memeriksa gagasan; belum pengganti
layer Gemma atau checkpoint produksi. **Placeholder** adalah slot dengan signature,
input/output dan acceptance criteria yang jelas, lalu fail dengan FeatureNotReadyError.
**Native tiny** berarti integrasi native random-weight dengan dimensi kecil;
belum pretrained. **Optional placeholder/reference** tidak diperlukan pada recipe default.

Status tersebut berlaku per blok, bukan per repo secara keseluruhan. Registry bisa
siap sementara factory bobot belum siap. Loss/collator bisa siap sementara training
belum berjalan. Serializer dan tabel skor bisa siap sementara benchmark belum dieksekusi.

## Hubungan docs dan kode

[Registry blok](../src/bifocalgemma/scaffold.py) memberi ID seperti M01 atau D03.
Masing-masing ID memiliki bagian dalam BUILD_GUIDE dan file sumber yang ada.
Perubahan satu blok harus memperbarui kontrak, acceptance criteria, tes yang relevan,
dan catatan validasi. Dokumen utama menyatakan keputusan aktif; riwayat keputusan
lama hanya ada di RESEARCH_LOG.

Tidak ada bobot pretrained, dataset besar, atau hasil code-repair BifocalGemma
yang disertakan. Config/checksum resmi dan contoh synthetic dibedakan dengan jelas.

- [Full-corpus curation and training](FULL_CORPUS_TRAINING.md): admission, benchmark quarantine, complete-epoch notebook defaults.


[GSM8K supplementary evaluation and context budgets](GSM8K_CONTEXT.md) documents the executable pinned adapter, math generation/scoring workflow, full 1,319-case tokenizer validation, and the separate larger-window GPU gate.


[Dataset and benchmark inventory from current code](DATASETS_AND_BENCHMARKS.md) distinguishes the default repository SFT pool, optional instruction runs, conditional sources, mandatory quarantine artifacts, implemented scorers and pending adapters.
