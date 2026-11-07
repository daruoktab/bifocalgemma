# Repository study: implementasi dan cara menjalankan

Diperbarui **8 Oktober 2026**. Milestone ini menyiapkan alur data → SFT → generation → scoring. Ia tidak menjalankan training pretrained atau benchmark model di Molab secara otomatis.

## Artefak utama

- [Notebook standalone Molab](../notebooks/bifocalgemma_repository_study.py), 14 cell, source dibundel dari repo dengan SHA-256. Unggah satu file .py; tidak perlu checkout repo di Molab.
- [Konfigurasi sumber/quarantine](../configs/repository_study.json): expected source revisions, 14 verified RES-Q fork aliases, 37 quarantined repo IDs, execution/network disabled defaults.
- [Fixture lintas file sintetis](../examples/repository_study/README.md):8 train records /4 repo groups dan 1 eval record. Bukan benchmark atau training-quality evidence.
- CLI `python -m bifocalgemma.repository_study`: `fetch-assets`, `prepare`, `prepare-diagnostic`, `prepare-humanevalfix`, `generate`, `score`.

## Peta cell notebook

| Cell | Peran |
|---|---|
|0|Intro dan batas scope|
|1|Control center: source/experiment, paths, action flags dan budgets|
|2|Checksum/extract bundled source, imports dan environment|
|3|Input HF password form|
|4|Apply/verify HF token; tidak login persistent atau mencetak token|
|5|Source preparation: immutable before snapshots, public context, static patch verification|
|6|Matched six-model eligibility dan canonical repo-group split|
|7|GPU preflight dan pinned checkpoint cache; tidak ada CPU random fallback|
|8|SFT satu model dengan policy/checkpoint v2|
|9|Target-free greedy generation; encoder sekali, LM head hanya last token|
|10|Native Python scoring atau gold-scorer validation|
|11|Cheap stage summary|
|12|Pytest self-test (position 13 after corpus cell)|

Semua RUN flags default False. FETCH_SOURCE_ROWS, FETCH_TOKENIZERS, ALLOW_GIT_NETWORK dan ALLOW_CODE_EXECUTION juga default False. Tokenizer fetch hanya metadata/tokenizer, bukan weights. Model files diunduh hanya sesudah RUN_TRAINING/GENERATE diaktifkan dan guard CUDA/BF16/free-memory lulus.

Perubahan knobs/token dapat menyebabkan rerun. Matikan actions sebelum mengedit setting. Gunakan output folder/file baru setiap run; artifacts yang ada tidak ditimpa. Jika source bundle berbeda dari versi package yang sudah di-import, restart kernel; jangan mengganti live modules di tengah training.

## Alur yang disarankan di Molab

1. Buka notebook, apply HF read token bila diperlukan.
2. Mulai SOURCE=`fixture`, RUN_PREPARE=True, FETCH_TOKENIZERS=True. Ini hanya mengunduh tokenizer assets dan menyiapkan8 synthetic records. Dataset/model asli tidak dilatih.
3. Untuk evaluator smoke, pilih SOURCE=`fixture_eval` dan PREPARED_DIR baru, matikan training/generation. RUN_SCORE=True +VALIDATE_GOLD=True +ALLOW_CODE_EXECUTION=True akan memeriksa gold fixture. Itu **gold validation, bukan model score**.
4. Data riil: pilih `swe_fixer` atau `swe_rebench_v2`, set ROWS_FILE atau FETCH_SOURCE_ROWS=True. ALLOW_GIT_NETWORK=True memberi izin clone/fetch repo snapshots. Mulai limit kecil dan per-repo quota; jangan mengambil ribuan repo sekaligus.
5. Periksa preparation_report, provenance/quarantine/filter reasons. Set train input/output limits setelah melihat distribusi selected corpus dan VRAM stress.
6. RUN_TRAINING=True memakai prepared train exports; enam experiment IDs didukung, termasuk decoder-only matched-policy SFT. MAX_STEPS 100 tidak otomatis berarti satu epoch 2.000 records.
7. Untuk benchmark, prepare SOURCE=`resq` dalam folder eval yang berbeda. Jalur awal menerima Python saja; JavaScript records dicatat sebagai rejected, bukan hilang diam-diam.
8. Setelah gold validation pada interpreter/dependencies yang sesuai, set ADAPTER_CHECKPOINT untuk model pair yang dilatih, RUN_GENERATE=True pada input eval, lalu RUN_SCORE=True. Baseline original IT memakai adapter kosong; matched-SFT baseline memakai checkpoint v2 sendiri.

Checkpoint cache memerlukan puluhan GB disk, dan repo/env menambah biaya. Guard weights+headroom bukan jaminan activation memory atau long-context throughput. Runtime eval Python 3.9/3.10 dipisahkan dari training Python 3.13; scorer menolak mismatch. Dependency environment tidak dibuat otomatis oleh notebook.

## CLI smoke lokal tanpa weights/VM

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
python -m bifocalgemma.repository_study fetch-assets --tokenizer-cache outputs/repository_tokenizers
python -m bifocalgemma.repository_study prepare --source fixture --rows examples/repository_study/train_rows.jsonl --snapshots examples/repository_study/snapshots --tokenizer-cache outputs/repository_tokenizers --context-limit 1024 --output outputs/repo_train_smoke
python -m bifocalgemma.repository_study prepare --source fixture_eval --rows examples/repository_study/eval_rows.jsonl --snapshots examples/repository_study/snapshots --tokenizer-cache outputs/repository_tokenizers --context-limit 1024 --output outputs/repo_eval_smoke
python -m bifocalgemma.repository_study score --prepared outputs/repo_eval_smoke --validate-gold --allow-code-execution --output outputs/repo_gold_check.json
```

Jika memakai offline audit cache yang sudah ada, arahkan --tokenizer-cache ke root tersebut dan lewati fetch-assets. Tidak ada modulo/clipping real token IDs agar muat model tiny.

## Kontrak source/snapshot

[repository.py](../src/bifocalgemma/data/repository.py) menyediakan schema SWE-Fixer, SWE-rebenchV2, RES-Q dan fixture yang jelas diberi source sintetis. Input default hanya issue/instruction +before snapshot. Hints, PR solution prose, test_patch, evaluator tests dan gold file identities tidak dirender.

GitSnapshotStore menggunakan satu mirror per source URL; network opt-in, commit resolved, path/mode/resource checks. Default text snapshot max 5.000 files/50 MB. Non-UTF8 blobs tidak dimasukkan; symlink/submodule ditolak. **Native execution mungkin memerlukan real checkout lengkap dengan binary assets**; jangan menafsirkan text-only snapshot sebagai seluruh filesystem benchmark. Clone full history upstream dapat mahal; budget/fetch strategy tetap perlu diprofilkan pada source yang dipilih.

Static patch API mendukung UTF8 modifications, multiple files, additions/deletions. Binary patches, symlink/submodule modes, unsafe paths, renames/copies ditolak dengan alasan. Unsupported task bukan bukti task sumber invalid.

## Kontrak export dan training

Prepared train file memakai `record_type=repository_patch_v1`, role=train, instruction, public context, target_patch, source/revision, full base snapshot hash, canonical repo group dan context-selection metadata. Static patch apply diverifikasi sebelum export. Before files disimpan sebagai snapshots yang hash-nya diverifikasi.

Eval export memisahkan:

- `eval_inputs.jsonl`: public model input tanpa gold/test script.
- `eval_private/specs.jsonl`: hidden gold/reference/test/environment metadata untuk scoring.
- `preparation_report.json`: accepted/rejected IDs, source bytes digest dan scope validasi.

Kedua notebook symmetric/asymmetric tetap 19 cell/18 isi identik. Pilih DATA_SOURCE=local_jsonl dan LOCAL_DATA_FILE ke prepared train.jsonl. Helper bersama menerima format legacy dan repository exports, menolak eval/known benchmark source serta gold-context flags. Source revision/snapshot/context hashes masuk training manifest.

Notebook repository memakai [repository_sft.py](../src/bifocalgemma/training/repository_sft.py): cohort dibekukan berdasarkan intersection token eligibility **semua enam configurations**, baru split per canonical repo. Baseline memperoleh budget context+target, bukan decoder cap numerik yang identik dengan pairs. Train/dev IDs dan tokenized fingerprints dicatat.

Policy decoder_only memiliki satu optimizer group decoder; pairs mempertahankan encoder/decoder dan alignment hanya asimetris. Words/PLE frozen, rank 16/alpha 32, LR 2e-5/warmup 5/joint seperti baseline sebelumnya. v2 inference restore memeriksa checksum, format, experiment, policy, names/shapes dan finite tensors sebelum copy. Tidak membaca pickle optimizer state saat inference.

## Generation dan source identity

[Native local loader](../src/bifocalgemma/models/v2/pretrained.py) memakai explicit local files. Header seluruh decoder/encoder divalidasi pada meta sebelum real allocation. Full model memerlukan CUDA/BF16 dan memory guard. Tidak ada unduhan otomatis atau synthetic/random fallback pada loader.

Local key/shape verification **tidak membuktikan upstream weight revision identity**; provenance menyatakannya. Notebook fetch memakai pinned Hub references; pengguna tetap perlu menyimpan artifact provenance actual run.

[I01](../src/bifocalgemma/inference/generation.py) sekarang short greedy reference: target-free prompt, thinking=False, encoder-once, canonical EOS/tool boundary, last-token-only logits, explicit budgets, original model mode restored. **Decoder temporal cache belum ada**; seluruh decoder hidden states direcompute per step. Jangan menjalankan 8k output seluruh benchmark sebagai workload production sebelum memprofilkan throughput/cache/backend.

## Native scorer dan denominator

[Scorer](../src/bifocalgemma/evaluation/native_python.py) menerapkan predicted diff pada isolated text snapshot, lalu opsional menjalankan hidden Python tests pada child interpreter. HF/GitHub/API credentials dan inherited PYTHONPATH tidak diteruskan. Marker akhir, unittest result checks, snapshot hash, runtime version dan timeout mencegah beberapa false passes/infrastructure mistakes.

Conda/venv/subprocess **bukan security sandbox**. Jangan menjalankan generated code dengan credentials/akses data pribadi. Native controls tidak mengklaim proteksi setara VM atau official scorer equivalence tanpa gold parity.

Missing/invalid patch →PATCH_FAILED; tests fail →FAIL; timeout →TIMEOUT; dependencies/runtime problems →INFRA_ERROR; execution disabled →STATIC_ONLY. Functional pass rate menjadi N/A jika ada unscored/infrastructure outcomes; conservative rate memakai denominator seluruh prepared cohort. Preparation exclusions tetap dilampirkan, bukan disebut full source score. Validate-gold report tidak diberi label model benchmark.

## RepoQA dan CrossCodeEval diagnostics

```powershell
python -m bifocalgemma.repository_study prepare-diagnostic --source repoqa --rows data/repoqa-2024-06-23.json.gz --tokenizer-cache outputs/repository_tokenizers --context-limit 4096 --output outputs/repoqa_inputs
python -m bifocalgemma.repository_study prepare-diagnostic --source crosscodeeval --rows data/python/line_completion_rg1_bm25.jsonl --tokenizer-cache outputs/repository_tokenizers --context-limit 4096 --output outputs/cce_inputs
```

RepoQA window builder adalah **adaptation**: Gemma-token budget, lexicographic files, contiguous line windows. Function answer sengaja terdapat dalam context sebagaimana retrieval task; name/path metadata tidak ditambahkan sebagai query hint. CrossCodeEval menerima prompt dan crossfile_context.text dari preprocessed non-reference retrieval, menolak overflow tanpa silent truncation dan menjaga groundtruth terpisah.

CLI generation membaca task_mode dan menghasilkan function/completion text, bukan diff. Score text memakai **adapted exact-code-match**, bukan RepoQA syntax threshold 0,8, CrossCodeEval statement/comment normalization atau identifier metrics. Report official_benchmark_score=False. Gunakan upstream scorers/pinned task policy untuk published-score comparisons. Tidak mengklaim parity yang belum diuji.

## Verifikasi milestone

Pada 8 Oktober 2026: suite gabungan package dan tiga notebook 96 passed/2 skipped; suite repository terbaru 13 passed, termasuk checkpoint-loader roundtrip, diagnostic boundaries dan last-token parity. Generator, marimo strict, Ruff/type checks dan whitespace checks lulus; lihat [validation log](VALIDATION.md).

Actual smoke memakai tokenizer offline Gemma yang hash-nya diverifikasi: 8 synthetic train exports /4 repo groups, 1 eval export, dan gold native pass 1/1. **Tidak ada real model training/generation atau native RES-Q gold-test result yang disimpulkan dari smoke tersebut.**

## HumanEvalFix gold validation

```powershell
python -m bifocalgemma.repository_study prepare-humanevalfix --rows data/humanevalpack_python.parquet --tokenizer-cache outputs/repository_tokenizers --output outputs/hefix_inputs
python -m bifocalgemma.repository_study score --prepared outputs/hefix_inputs --validate-gold --allow-code-execution --output outputs/hefix_gold.json
```

Input memakai prompt+buggy_solution dengan visible docstring; canonical solution dan
hidden tests hanya dipakai untuk preparation validation/target dan scorer. Current
ai Python 3.13 menjalankan 164/164 reference repairs dengan sukses. Ini native diff-output
adapter validation, bukan published Python 3.9.13 parity atau model performance.
RepoQA Python 100 juga berhasil diprepare dengan real verified Gemma tokenizer.

## Maintenance

```powershell
python tools/generate_training_notebooks.py
python tools/generate_repository_notebook.py
python tools/generate_training_notebooks.py --check
python tools/generate_repository_notebook.py --check
python -m marimo check --strict notebooks/bifocalgemma_repository_study.py notebooks/bifocalgemma_train_symmetric.py notebooks/bifocalgemma_train_asymmetric.py
```

Edit helper source/template, lalu regenerate; jangan mengedit encoded bundle secara manual. Bundle tidak memuat pretrained weights, tokenizer payload, HF credentials atau real benchmark code/data. [Riset](RESEARCH_REPOSITORY_DATA_AND_BENCHMARKS.md) memisahkan admission/benchmark claims dari readiness engineering ini.

The added full-corpus cell prepares instruction foundation and repository snapshots before SFT. TRAINING_DATA_MODE=curated_pool uses the frozen complete cohort for all six models; MAX_STEPS=0 completes configured epochs. Legacy repository_tasks is an explicit alternative, and RESUME_FROM supports checkpoint v2. See [full-corpus guide](FULL_CORPUS_TRAINING.md).

Runtime pins now match the ai environment: Transformers 5.19.0 / Torch 2.14.1 CUDA13.0. The earlier 5.18.0/2.11.0 results are historical measurements, not re-labeled as this runtime.
