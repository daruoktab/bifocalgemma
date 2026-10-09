# Validasi Scaffold dan Native Tiny

Diperiksa **6 Oktober 2026**: Conda ai, Python 3.13.16, Torch 2.14.1+cu130,
Transformers 5.18.0, CUDA 13.0, NVIDIA GeForce RTX 3060 Laptop GPU (6144 MiB).
Tidak ada pretrained model weights yang diunduh atau dimuat.

## Hasil aktual

| Check | Hasil | Cakupan |
| --- | --- | --- |
| Unit suite | **55 tests lulus, tanpa skip** | 41 existing + 8 native integration + 6 offline data/tokenizer |
| Native integration | 8 tests lulus | Six-model backward, no-context parity, masks, KV ownership, aliases |
| Offline data/tokenizer | 6 tests lulus | Cached real tokenizers, budgets/hashes, synthetic patch roundtrip |
| CLI smoke-native CPU | Enam model lulus | Full-depth random native forward/backward |
| CLI smoke-native CUDA | Enam model lulus | Finite/nonzero gradients, aliases after GPU transfer, measured tiny memory |
| Ruff lint/format dan ty | Lulus | Source/tests dengan environment ai |
| Offline wheel | Lulus | Config/hash assets dan modul native/data terdapat dalam wheel; tanpa dependency download |
| Docs/block map | Lulus | 18 Markdown files, 23 source blocks/anchors, relative file links |
| Existing checks | Tetap dalam suite | Meta shapes, reference/bridge, data collation/loss, reporting, registry |

Dua tes tokenizer riil memakai BIFOCALGEMMA_TOKENIZER_CACHE menuju cache audit.
Tanpa environment variable ini, kedua tes tersebut di-skip; tests lainnya tetap
portable/offline. Output tabel pada test_evaluation memakai angka fixture untuk
menguji rendering, bukan hasil benchmark BifocalGemma.

## Native tiny CUDA smoke

Batch synthetic dua contoh; logits `[2,5,128]`; FP32, eager, tanpa optimizer/cache.
Parameter unik setelah alias dihitung dari model yang benar-benar dibuat.

| Model | Parameters tiny | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: | ---: |
| E2B–E2B | 1,793,224 | 31,927,808 | 37,748,736 |
| E4B–E4B | 3,129,592 | 42,767,360 | 48,234,496 |
| E2B–E4B | 2,468,064 | 37,413,376 | 48,234,496 |
| E4B–E2B | 2,468,064 | 37,413,376 | 48,234,496 |
| E2B-IT baseline | 898,660 | 24,565,760 | 48,234,496 |
| E4B-IT baseline | 1,567,868 | 30,006,272 | 48,234,496 |

Peak counters di-reset per model dalam satu process; reserved bytes dapat membawa
allocator cache model sebelumnya. Ini bukan total GPU process memory atau profile
pretrained. CPU melaporkan parameter/gradient bytes dan null untuk CUDA peaks.
Reference generic M03 tetap terpisah (sekitar 21k–45k parameter).

## Perilaku yang dibuktikan

- Keempat pasangan dan dua baseline melakukan native forward/backward dengan finite
  logits/gradients. Ties hanya pada pasangan simetris; decoder head selalu tied.
- Decoder tanpa context cocok dengan native Gemma pada token valid, termasuk left
  padding dan panjang 513 yang melampaui local window. Final softcap dipertahankan.
- Token decoder masa depan tidak memengaruhi output sebelumnya. Context encoder
  berpengaruh, bidirectional encoder membaca suffix, dan padded context setara.
- Entirely masked encoder/query rows menghasilkan finite backward. Local self
  window tidak menyembunyikan encoder keys.
- Shared layers menggunakan KV pemilik bertipe sama dan tidak memiliki k_proj/v_proj
  tambahan. Native PLE, MLP, residual dan RoPE tetap berada dalam upstream stack.
- Ties/PLE separation serta logits dipertahankan pada perubahan dtype dan state_dict
  reload (termasuk assign=True). CPU→CUDA alias diperiksa melalui smoke output.
- Tokenizer keenam konfigurasi menjaga Unicode/tab/indentasi, BOS sekali, closing
  turn token, patch-only target, zero prefix labels dan single shift. Cache hilang,
  hash/revision salah, target kosong dan overflow ditolak.
- D06 membalik bug-inducing patch SWE-smith pada synthetic single/multi-file fixtures,
  memverifikasi apply/roundtrip di temporary directory, serta menolak ref/hash/patch
  salah. Tidak ada source checkout yang dimodifikasi oleh normalizer.

## Batas bukti

M07 menguji native random-weight integration dengan width/vocabulary diperkecil.
M04 pretrained transfer dan M05 production factory masih placeholder. D03 tersedia,
D06 local normalization tersedia, tetapi D04 download/reconstruction/overlap audit
real dataset belum siap. Tidak ada official SWE-smith/SWE-bench harness execution,
repair score, real SFT, optimizer resume, autoregressive merged cache, optimized
kernel, gradient-checkpointing integration, BF16/quantized merged validation, atau
peak VRAM model penuh. Direct SFT masih hipotesis yang memerlukan pilot pretrained.

## Reproduksi

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:BIFOCALGEMMA_TOKENIZER_CACHE = 'C:\path\to\gemma4-json-audit'
python -m unittest discover -s tests
python -m bifocalgemma validate-config
python -m bifocalgemma smoke-native --experiment all --device cpu
python -m bifocalgemma smoke-native --experiment all --device cuda
ruff check .
ruff format --check .
ty check .
git diff --check
```

Isi cache root dengan path audit milik pengguna; bukan folder tokenizer HF umum.
[Native local validation](NATIVE_LOCAL_VALIDATION.md) menjelaskan API, assets,
kontrak data dan sumber riset. [Build guide](BUILD_GUIDE.md) menyatakan status per blok.

## Notebook marimo standalone — 6 Oktober 2026

[Notebook](../notebooks/bifocalgemma_architecture_test.py) berisi 13 cell dan PEP723
minimal. Marimo check --strict, Ruff lint/format dan ty lulus. Cell pytest-only
memuat tiga tes yang lulus lewat python -m pytest notebook: mask local/global,
prefix/padding loss gradients, serta finite fully-masked backward. Graph CPU FP32
berhasil, serta marimo HTML export pada CUDA dengan BF16 forward/backward dan
FP32 semantic parity/causality checks berhasil: enam model, 60 kelompok checks.
Reference decoder menyalin nonpersistent buffers sebelum mengganti attention;
semantic FP32 tolerance tetap 2e-5. Ini tidak mengklaim BF16 logits parity.
Full-size random BF16 probe untuk GPU 96 GB tersedia dengan RUN_FULL_SIZE=False
secara default; belum dieksekusi di GPU Blackwell dari sesi lokal ini.
Report JSON dan HTML berada di outputs yang ignored. Tidak ada download bobot/data.
Cara memakai dan batas probe ada pada [panduan marimo](MARIMO_ARCHITECTURE_TEST.md).

## Pretrained memory probe — notebook siap Molab, belum hasil GPU96

[Notebook standalone](../notebooks/bifocalgemma_pretrained_probe.py) dan
[panduan](PRETRAINED_GPU_PROBE.md) menyediakan pinned text checkpoint transfer,
attention LoRA/full policy, AdamW steps, serta cumulative 1–4 model residency.
CPU preflight menampilkan SKIPPED_HEADROOM sebelum network/model weights download.
Lima helper tests memakai native tiny safetensors fixture, termasuk atomic key/shape
validation, shared IT word anchor, PLE separate, LoRA scope dan actual optimizer
state formation. Tiga notebook pytest tests lulus. Marimo check --strict, Ruff dan
ty diperiksa. Pengguna mengunggah dan menjalankan probe asli di Molab; checkpoint
asli tidak dimuat lokal dan belum ada angka peak VRAM/kapasitas GPU96 yang diukur.

Header safetensors asli keempat checkpoints diperiksa via HTTP range terhadap
state_dict native meta: E2B 540 target keys / 60 ignored shared-KV, E4B 665 / 54;
zero missing, shape mismatch atau unexplained text key. Audit kecil tersimpan di
[fixture header](../tests/fixtures/gemma4_checkpoint_header_audit.json).
Payload tensor tetap belum diunduh atau diuji lokal.

HF authentication pada pretrained notebook dipisahkan ke cell2a password form dan
cell2b Apply/whoami. API mock memeriksa widget precedence, environment fallback,
anonymous False token, invalid-token failure dan safe auth report tanpa credential.
Graph CPU/anon guard serta tiga notebook pytest tests lulus setelah perubahan.

## Matched training notebooks — 6 Oktober 2026

Notebook simetris/asimetris memiliki 17 cell dalam posisi sama. Diff hanya tiga
baris FAMILY/EXPERIMENT/OUTPUT_ROOT pada control center; 16 cell lainnya identik.
Generator --check memeriksa sinkronisasi template/helpers. Kedua notebook punya
3 pytest tests (6 total) dan graph disabled tanpa data/model download. Tujuh helper
tests memeriksa cell parity/instruction identity, diff dengan missing final newline, dedup/commit split, token-weighted
accumulation versus larger batch, overwrite/compatibility/checksum failure, serta
native tiny LoRA resumed vs uninterrupted pada kedua keluarga. Big pretrained
SFT kedua notebook belum dijalankan lokal. Lihat [panduan](TRAINING_NOTEBOOKS.md).

## Training policy dan checkpoint v2 — 6 Oktober 2026

Kedua notebook kini 18 cell, 17 isi identik, dengan satu cell Training policy setelah
loader. Profil simetris encoder/decoder groups; asimetris menambah alignment. Meta
table, actual LR/warmup per group, policy/group signatures dan v2 checkpoints
tersambung. Legacy training_policy=None tetap memakai optimizer/checkpoint v1.
Tests memeriksa coverage/frozen/aliases, per-group scheduler, native tiny v2 resume
kedua keluarga, serta v1/policy/group-order rejection sebelum mutation. Big pretrained
training tetap dijalankan pengguna di Molab; ini bukan hasil eksperimen training.

Policy update checks completed: 12 helper tests (5 policy + 7 existing offline SFT)
passed, 6 notebook pytest tests passed, both disabled graphs produced the expected
meta group tables, and 18-cell/17-identical parity passed. Marimo check --strict,
generator --check, Ruff and ty passed. No pretrained training was launched.


## Repository pipeline dan Molab notebook — 8 Oktober 2026

Suite gabungan package dan tiga notebook training/repository menghasilkan **96 passed,
2 skipped** (141,29 detik). Kedua skip adalah pengujian tokenizer cache opsional.
Suite repository terbaru, termasuk tambahan roundtrip loader checkpoint native tiny,
menghasilkan **13 passed**. Semua 11 tes dalam tiga notebook juga lulus secara
terpisah. Dua notebook training kini 19 cell dengan 18 isi identik; perbedaannya
hanya tiga baris control center. Notebook repository standalone memiliki 13 cell.

Kedua generator `--check`, `marimo check --strict` untuk tiga notebook, Ruff lint
serta format, `ty check --python` memakai interpreter env ai, dan `git diff --check`
lulus. Loader diuji dengan safetensors sintetis, shared ordinary embeddings, PLE
terpisah dan checkpoint yang tidak cocok; ini belum memverifikasi transfer payload
checkpoint pretrained asli.

Smoke CLI menggunakan tokenizer Gemma offline yang hash-nya diperiksa: delapan
synthetic training exports dari empat repo groups dan satu eval export. Gold fixture
lulus 1/1. HumanEvalFix Python menghasilkan 164 prepared tasks dan **164/164 gold
repairs lulus** pada interpreter ai Python 3.13. RepoQA Python menghasilkan 100
prepared tasks menggunakan protokol adaptasi; exact-match scorer ini bukan scorer
resmi RepoQA. Ringkasan empiris tersimpan di
[validation receipt](data/repository_pipeline_validation_20261008.json).

Hasil tersebut memvalidasi persiapan data dan scorer, bukan kualitas BifocalGemma.
Training/generation checkpoint pretrained, native RES-Q gold tests, parity runtime
benchmark resmi, serta cache dan kernel long-context belum diuji pada tahap ini.
Ikuti [panduan repository study](REPOSITORY_STUDY_IMPLEMENTATION.md) untuk Molab.


## Full-pool curation — 9 Oktober 2026

Sembilan tes curation/complete-epoch runner lulus, termasuk SQLite vs memory index,
benchmark/license quarantine, cross-source dedup, connected split, prepared-pool
checksum rejection, dan complete-epoch native tiny training untuk simetris,
asimetris serta baseline decoder-only. Suite related helpers dan dua notebook
sebelumnya menghasilkan 25 passed sebelum tiga complete-epoch tests ditambahkan.
Generator parity, marimo strict, Ruff dan type checking lulus. Full-corpus job
masih menjalankan admission/profiling; manifest final belum menjadi bukti pada
catatan ini. [Pilot pretrained](data/molab_pilot_20261008.json) dicatat terpisah.


## ai runtime and standalone corpus notebooks — 9 Oktober 2026

Runtime pins now match the upgraded conda ai environment: Torch 2.14.1/cu130,
Transformers 5.19.0, PyArrow 25.0.1, tokenizers 0.23.2 and NumPy 2.5.3. The
5.18.0 → 5.19.0 Gemma4 source comparison found text RoPE changes and output/router
metadata changes; this is followed by full native/scaffold/notebook regression,
not just removal of the version guard. The official CUDA130 index contains the
Linux CPython3.13 x86_64 Torch2.14.1 wheel.

Package plus all five notebooks: **116 passed, 2 optional-cache skips**. After
moving the source bootstrap into the Environment cell, related helpers and both
training notebook checks additionally produced **20 passed**. Timing diagnostics
are excluded only from wall-clock equality; v2 resume still checks all semantic
history fields and model weights.

Both training notebooks have **20 cells /19 identical bodies**, with only the
three control-center family/experiment/output lines differing. The universal
repository notebook has 14 cells and now consumes complete frozen pools for pairs
and baselines, with explicit checkpoint-v2 resume. The Environment cell holds the
source bundle so the data-preparation cell remains readable.

The complete instruction foundation has 44,333 train and 4,926 dev records after
all-source admission/profiling. A real SWE-rebench task, wtforms__wtforms-614,
validated a 92-file pinned text snapshot, BSD source license evidence, 2,019-token
issue-only context and static application of a patch touching 10 files. This is
adapter validation, not model quality or a claim that the task fits every output
budget. Repository-corpus admission is being checked separately.

Every one of the 49,259 frozen instruction-pool records was re-tokenized on Transformers 5.19.0 for all six experiment configurations. IDs, split order, limits and closing-token policy passed with no model-specific filtering. [All-six receipt](data/all_six_pool_tokenization_20261009.json) stores per-configuration token-stream digests.

The updated repository builder completed a four-real-task end-to-end export (3 train/1 dev), with licensed immutable snapshots, issue-only context, benchmark overlap checks and static patch roundtrip. Its six tokenizer configurations are checked separately. Six archive/ledger tests cover transport retry, resume settings, nested license includes and read-only index protection. The larger 64-task adapter run is separate from production corpus preparation.

The completed 64-real-task repository adapter cohort contains 58 train/6 dev and 45 multi-file patches. All six configurations tokenized exactly those frozen IDs. This validates the repository pipeline, not the configured full production budget or functional model quality. [Receipt](data/repository_64_task_validation_20261009.json).

Universal curated-run checkpoint compatibility now records model spec, physical dimensions and runtime versions. Native tiny complete-epoch checks passed (9 tests), followed by a targeted runtime-drift resume rejection test (1 passed).

Production preparation exposed a connection-reset error separate from HTTP incomplete-read. Archive retries now cover ConnectionError; seven archive/ledger regression tests passed. Actual preparation resumed from ledger offset 1203 with all 118 accepted records preserved.


## Complete raw repository target profile — 9 Oktober 2026

The profiler scanned all 32,079 pinned SWE-rebench V2 rows and measured all
7,243 Python issue/patch targets with the hash-verified official IT tokenizer.
No rows were sampled or silently truncated. Aggregate counts, nearest-rank
quantiles, input-file SHA and ordered per-row length-stream SHA are preserved
in the [receipt](data/repository_raw_target_profile_20261009.json).
This is separate from snapshot/license/overlap admission and original-model
VRAM validation. The 2k paired-decoder gate covers only 54.45% of raw Python
rows; 8k covers 96.01%. Main-study length selection must account for this tail.
Profiler Ruff, formatting and type checks pass; its actual full-source run is
the validation evidence. Production 2k corpus preparation remains active.


Curated pool/control limit consistency: **29 tests passed**, covering curation,
complete-epoch training/resume, all three regenerated notebook test cells,
and rejection of encoder/pair/baseline manifest drift. Invalid controls and
missing limit metadata are rejected; validation does not mutate the manifest.
Both generators --check, marimo --strict, Ruff, type checking and whitespace
checks passed. AST comparison confirms 20 cells /19 identical bodies in the
matched family notebooks. The universal context default now agrees with the
bounded 2k pilot pool, and all three show frozen limits in READY.


## GSM8K executable evaluation — 9 Oktober 2026

Pinned preparation exported all 1,319 released test cases in separate zero-shot
and eight-shot directories. All six real tokenizer configurations passed
public-input and generation-budget checks for every case in both protocols.
Reference completions passed 1,319/1,319 release-style scorer checks through
the CLI; this is adapter validation, not pretrained-model accuracy. Empty
predictions retain a denominator of 1,319. The [receipt](data/gsm8k_adapter_validation_20261009.json)
stores frozen manifests and complete ordered token-stream hashes.

**22 tests passed** across the GSM8K adapter and three regenerated notebooks.
Tests cover gold isolation, train-only demonstrations, source/file hashes,
fixed denominators, duplicate/unknown predictions, mixed models/protocols,
encoder overflow, budget rejection before loading weights, and one tokenizer
load per checkpoint per generation run. Generators, strict marimo, Ruff, type
checking and whitespace checks passed. Original-model math generation and
larger-context training memory remain unmeasured.


## Original-model context sweep preparation — 9 Oktober 2026

The standalone pretrained probe now has 16 cells and an opt-in sequential
six-model length sweep. Unlike the short-target pilot, it fills synthetic
supervised decoder tensors to the requested shifted length, preserving the
prefix mask and final turn token. Baseline lengths include the same selected
context plus target budget. These are stress tensors, not real SFT examples.

**12 helper/notebook tests passed**, including exact 2k/4k/8k/16,896 collator
shapes, failure-boundary stopping, baseline budget parity, library/notebook
helper identity, and incremental per-case JSON persistence. The simulated
persistence test uses mock probe outcomes; it is not a GPU measurement.
All five notebooks pass strict marimo checking; generators, Ruff, type checking
and whitespace checks pass. [Run instructions](PRETRAINED_GPU_PROBE.md).
Original checkpoint allocation/training at these boundaries is left to the
user's Molab runtime and is not declared feasible by these CPU tests.


## Full SWE-Fixer raw target profile — 9 Oktober 2026

All 115,406 pinned raw records were measured using official patch-only IT
formatting and verified tokenizer assets, with no sampling or gold truncation.
The [aggregate receipt](data/swe_fixer_raw_target_profile_20261009.json) records
input-file and complete ordered length-stream hashes. Its missing-language
convention is explicit; it does not certify Python source validity, snapshot
licenses or functional quality. At 8k paired decoder, 113,820 raw candidates
fit (98.63%). Extreme targets exceed the original 128k position limit.
Profiler lint, formatting and type checks passed. Repository-production
preparation continues separately with its own immutable settings and ledger.


## Visible-context license coverage correction — 9 Oktober 2026

A complete point-in-time audit of 487 v2 accepted records found 483 passing
and four requiring review for visible-context source licensing. This does not
assert unlawful licensing; those files fall outside the project's currently
admitted/reliably recognized license policy. Target-file checks alone did not
cover every source window. The [receipt](data/repository_context_license_audit_20261009.json)
records cases and the captured offset.

The v3 builder gates and records licenses on every selected context path.
Consumers reject old repository pools without the marker or with incomplete
coverage before training. Settings bind study config, tokenizer assets and
Python/Transformers/tokenizer runtime. The old producer was intentionally
superseded for this confirmed coverage gap, with ledger/archives preserved;
its accepted count is not a completed approved corpus.

**39 tests passed** across archive/curation, complete-epoch training and three
regenerated notebooks. A real four-task cached v3 export (3 train/1 dev) passed
the strengthened loader with evidence for 17 visible source files. Generators,
strict marimo, Ruff, type and whitespace checks pass. The corrected production
job is active and revalidates cached source under a separate v3 ledger.
