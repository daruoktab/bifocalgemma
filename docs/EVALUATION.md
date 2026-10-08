# Evaluasi: Generasi, Harness dan Analisis Enam Model

## Tujuan dan model

Model utama hanya empat BifocalGemma base/IT pairs dan E2B-IT/E4B-IT decoder-only.
[Desain eksperimen](EXPERIMENT_DESIGN.md) memberi role, tying policy, kontrol dan
RQ. T5Gemma/T5Gemma 2 adalah related work. Skor referensi publik LiveCodeBench
tidak menggantikan pengukuran repair dengan protokol yang sama.

Primary validation tahap awal adalah held-out offline editing: patch application/
reconstruction, exact edited-code match, syntax dan CE. Functional repair rate
menunggu runner Python terpisah. HumanEvalFix-docs Python (164 tasks) dipilih
sebagai benchmark repair utama. CanItEdit tambahan melalui native dependency gate
(subset 83 tasks yang screened imports-nya bukan full official benchmark). CrossCodeEval adalah completion diagnostic. SWE-bench Lite/Verified
dan Docker harness ditunda dari jalur utama. Tidak ada test reward atau RL pada
tuning; diagnostics tidak disebut resolved rate. Lihat
[riset dataset](DATASETS_AND_BENCHMARKS.md).

Rincian task/input/output, token budgets, denominator, dependency dan sequential
Molab runs ada di [paket data/evaluasi](MOLAB_DATA_EVALUATION_PLAN.md). Pilihan batas
input/output mengikuti data aktual; target 32k/8k tidak menjadi default. Baseline
memerlukan context+target dalam decoder, sehingga cap total decoder tidak disamakan
dengan pasangan. Patch-output adapter dan native CPU scorer belum terintegrasi.

## Kandidat evaluasi repository-context untuk studi akhir

[RES-Q Python](https://github.com/Qurrent-AI/RES-Q) memakai unified diff dan scorer
Conda native, sehingga lebih cocok dengan scope repository editing daripada hanya
function repair. Dataset mempunyai 95 Python / 5 JavaScript tasks; environment
3.9/3.10 dan gold tests perlu validasi terpisah dari training env. Label general
editing versus corrective repair dipisahkan. Metadata modified_files berasosiasi
dengan gold; tidak menjadi input selector main evaluation.

[RepoQA](https://github.com/evalplus/repoqa) dan CrossCodeEval melengkapi reader
evidence melalui function retrieval/completion, dengan output modes yang berbeda
dari patch. Rincian full metadata/profile dan native gates ada di
[riset repository data](RESEARCH_REPOSITORY_DATA_AND_BENCHMARKS.md). Generator/parser/
scorer tersebut belum diimplementasikan pada package hanya oleh perubahan docs ini.

## I01 — generation

[generate_patch](../src/bifocalgemma/inference/generation.py) menerima spec,
RepairExample tanpa gold context dan max_new_tokens. Short reference path sekarang menghasilkan patch bila diberi loaded native model
dan verified tokenizer cache. Tanpa model/cache tersebut ia tetap fail-fast. Temporal
KV cache/optimized long-context production belum tersedia.

Gunakan IT Jinja prefix, catat thinking mode, sampling dan token/tool budget.
generation_config IT EOS [1,106,50] berbeda dari root config EOS [1,106].
Token 50 mengisyaratkan tool-response boundary; parser tidak boleh menganggapnya
hasil test atau patch. Track actual token counts/latency/termination reason.

## A01 — agent loop opsional, ditunda

[DeveloperAgent](../src/bifocalgemma/agent/loop.py) adalah placeholder untuk
read/search/edit/test dengan max_steps. Tool grammar Gemma, allowed workspace,
timeouts, test failures, patch application dan immutable snapshot refresh harus
diimplementasikan sebelum menjalankan agent. Tidak ada shell atau repo test
yang dieksekusi oleh slot sekarang.

Semua model mendapat tools dan selection policy sebanding. Setelah edit, context
di-encode ulang penuh. Caching digunakan saat issue/snapshot/model tidak berubah
dan harus dibandingkan dengan prefix caching baseline bila didukung.

## E02 — prediction serialization

[write_swebench_predictions](../src/bifocalgemma/evaluation/predictions.py) sudah
menulis JSONL dengan instance_id, model_name_or_path dan model_patch.
Duplicate instance IDs ditolak. Tidak ada gold patch/test metadata pada output.

```json
{"instance_id":"synthetic-example","model_name_or_path":"BifocalGemma-E2B-E2B","model_patch":"diff..."}
```

Contoh itu synthetic dan bukan prediksi berhasil. Serializer tidak membuktikan
patch benar atau harness telah berjalan.

## E01 — SWE-bench harness opsional, ditunda

[SWEBenchHarness](../src/bifocalgemma/evaluation/harness.py) menerima predictions
path, dataset_revision dan run_id. Ia fail dengan FeatureNotReadyError sampai
Docker runner dan normalisasi report siap.

[SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/) menerapkan
patch di lingkungan terisolasi dan menjalankan FAIL_TO_PASS/PASS_TO_PASS. Mulai
subset hanya bila jalur optional ini kelak dipilih, di Linux/WSL2+Docker; nyatakan subset
tersebut, bukan skor Lite 300-task atau Verified 500-task penuh.

Catat attempted/submitted, resolved/unresolved, empty patches, patch-apply failures,
timeouts dan infrastructure errors. Tentukan denominator sejak awal; jangan
menghapus task gagal dari hasil tanpa penjelasan. Run_id baru dipakai untuk patch
baru agar result caching tidak memberi hasil dari percobaan sebelumnya.

## Pelaporan dan kontrol

print_repair_experiment_table menerima hasil per keenam canonical IDs dan selalu
mencetak enam baris; absent scores N/A, observed zero 0.0. illustrative=True
diperlukan untuk contoh nilai. Fungsi reporting tidak menjalankan model.

Per-run artifacts mencakup dataset/task IDs, model/checkpoint architecture version,
source revisions, tokenizer/selection policy, ordinary embedding tying, PLE policy,
trainable parameters, train token/compute budget, precision dan hardware.
Bandingkan cold/warm TTFT, total latency, throughput dan peak VRAM selain kualitas.

Duel asimetris sama-sama untied; ia menjadi perbandingan alokasi kapasitas utama.
Full 2×2 pada model default mencampur size dan symmetric-only tying. Klaim faktor
murni membutuhkan symmetric untied controls dengan training/architecture/budget
sebanding. Controls tetap dalam keluarga yang sama, bukan tambahan T5 benchmark.

## Command saat ini

```powershell
$env:PYTHONPATH = 'src'
python -c "from bifocalgemma.evaluation import print_repair_experiment_table; print_repair_experiment_table({}, illustrative=True)"
python -m bifocalgemma evaluate --predictions outputs/predictions.jsonl
```

Tabel pertama memuat N/A. Perintah kedua sengaja exit 2 pada E01; file tidak
dinilai, Docker tidak dijalankan dan tidak ada skor yang dibuat.

## Native/reference path yang tersedia — 8 Oktober 2026

[Repository CLI/notebook](REPOSITORY_STUDY_IMPLEMENTATION.md) menyiapkan source
exports, generic greedy generation dan opt-in native Python scoring. Whole
HumanEvalFix164 gold repairs lolos native ai validation; real RES-Q gold tests
belum dijalankan. RepoQA/CrossCodeEval adapter/scorer memakai explicit **adapted EM**,
bukan official syntax/parser metrics; official parity tidak diklaim. Preparation
exclusions, missing predictions, runtime/dependency errors dan static-only outcomes
tetap dilaporkan.
