# Evaluasi: Generasi, Harness dan Analisis Enam Model

## Tujuan dan model

Model utama hanya empat BifocalGemma base/IT pairs dan E2B-IT/E4B-IT decoder-only.
[Desain eksperimen](EXPERIMENT_DESIGN.md) memberi role, tying policy, kontrol dan
RQ. T5Gemma/T5Gemma 2 adalah related work. Skor referensi publik LiveCodeBench
tidak menggantikan pengukuran repair dengan protokol yang sama.

Primary metric adalah resolved rate SWE-bench Lite/Verified. Lokalisasi file
dilaporkan terpisah. CrossCodeEval adalah completion exact-match/similarity
diagnostic; HumanEvalFix tambahan unit-level repair. Dataset guidance ada pada
[riset dataset](DATASETS_AND_BENCHMARKS.md).

## I01 — generation

[generate_patch](../src/bifocalgemma/inference/generation.py) menerima spec,
RepairExample tanpa gold context dan max_new_tokens. Slot ini belum menghasilkan
patch. Native model/cache/tokenizer integration harus siap sebelumnya.

Gunakan IT Jinja prefix, catat thinking mode, sampling dan token/tool budget.
generation_config IT EOS [1,106,50] berbeda dari root config EOS [1,106].
Token 50 mengisyaratkan tool-response boundary; parser tidak boleh menganggapnya
hasil test atau patch. Track actual token counts/latency/termination reason.

## A01 — agent loop

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

## E01 — harness riil

[SWEBenchHarness](../src/bifocalgemma/evaluation/harness.py) menerima predictions
path, dataset_revision dan run_id. Ia fail dengan FeatureNotReadyError sampai
Docker runner dan normalisasi report siap.

[SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/) menerapkan
patch di lingkungan terisolasi dan menjalankan FAIL_TO_PASS/PASS_TO_PASS. Mulai
subset yang ID-nya dikunci dengan satu worker di Linux/WSL2+Docker; nyatakan subset
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
