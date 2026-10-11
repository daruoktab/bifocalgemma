# Full-corpus training: admission before SFT

Updated 9 October 2026. The full pool is being prepared; upstream row counts are
not admitted counts. The successful 100-step run was a smoke test, not the full
experiment. See the [pilot receipt](data/molab_pilot_20261008.json).

## Pipeline

1. Cache pinned EditPackFT, CommitPackFT Python and all six NextCoder shards.
2. Build a benchmark quarantine from the actual released HumanEvalFix, CanItEdit,
   RES-Q, RepoQA Python and non-reference CrossCodeEval Python artifacts.
3. Admit records with explicit source-code license policy, valid target Python,
   unambiguous adapters and complete provenance; syntax-error inputs are allowed
   when the repaired target parses. No source code is executed during curation.
4. Deduplicate across sources, profile every unique candidate, then freeze a
   repository/commit/before-seed connected-component split. Export full train/dev
   files and a checksummed manifest, without the pilot's 2,000/200 row caps.
5. Inspect token quantiles, overflow and source yields before finalizing the
   resource budget. Train each configuration on the same eligible examples.

```powershell
$env:PYTHONPATH='src'
python -m bifocalgemma.data.full_pool --fetch --audit-repository --tokenizer-cache outputs/repository_tokenizers
```

The tokenizer cache must contain the four hash-verified assets. The repository
study CLI `fetch-assets` prepares that layout. This command does not download
model weights. Raw files and disposable SQLite indexes stay under ignored
`outputs/`; the completed pool defaults to `outputs/full_training_pool_v1`.
An existing output pool is not overwritten.

## Admission and overlap scope

EditPackFT and CommitPackFT keep their per-record licenses; MIT dataset metadata
is not substituted for unknown source-code licenses. The current admitted set is
MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0 and Unlicense. Other licenses are deferred,
not declared invalid. NextCoder is synthetic instruction editing distributed under
MIT; its missing original repository lineage remains a documented limitation.
InstructCoder and GitHub Code Review remain conditional and are not silently
added to the pool.

The benchmark index checks normalized exact code and shared eight-token shingles
with Jaccard >=0.8. Short commonplace snippets need exact matches, not a fuzzy
contamination claim. Within-pool edits use normalized before/after pairs and a
Jaccard >=0.95 candidate heuristic based on 16 minimum-hash shingles. This does
not certify absence of semantic or pretraining contamination. All rejection IDs
and reasons are retained in the manifest.

RES-Q fork aliases and RepoQA origins are quarantined. CrossCodeEval provides
471 encoded repository slugs; candidate canonical identities are compared after
flattening owner/repository, rather than inventing an ambiguous slash in an
upstream hyphenated slug. The source artifacts and their SHA-256 values are
recorded. Full-source benchmark task counts are verified before admission.

The SQLite indexes avoid retaining all token-shingle postings in RAM. They are
scratch indexes and may be rebuilt; immutable raw files and pool exports remain
separate. A fresh process may reuse downloaded public raw files.

## Tokenization and split

Both base encoder tokenizers are checked on real contexts: their config files
are not identical. Decoder IT tokenizer/config/template hashes must match across
E2B/E4B before sharing decoder length measurements. No real token IDs are clipped
or remapped. Pair and baseline formats use official patch-only IT templates,
`enable_thinking=False`, closing tokens and one shifting operation.

The initial profiling caps are 2,048 encoder / 2,048 paired decoder / 4,608
baseline decoder. The report includes the distribution **before caps**, rejected
lengths and supervised token totals. These caps are a starting resource gate,
not proof that every source fits or that 32k/8k is appropriate. Final length and
training budget must follow the measured pool and GPU stress results.

Repo aliases, shared commits and normalized before-code seeds form connected
components that cannot cross train/dev. The full dev holdout is preserved. A
model-specific silent truncation, cap or re-split is rejected by prepared-pool
loading; file checksums and instance order are verified before training.

## Notebook recipe

The two matched notebooks retain 20 cells / 19 identical bodies. Their new
default is `DATA_SOURCE="curated_pool"`, with `LOCAL_DATA_FILE` pointing to the
pool directory. The preparation cell downloads one frozen collection from the private dataset repo at a full commit SHA. `PREPARE_FULL_POOL=True` is an explicit separate CPU curation option; ordinary training never implicitly starts it. See [dataset releases](DATASET_RELEASES.md). `RUN_TRAINING=False` remains the startup guard. `MAX_STEPS=0`
means all configured epochs over the frozen split; positive values explicitly
limit updates. `EPOCHS=1`, batch 1, accumulation 4 give approximately
`ceil(train_examples/4)` optimizer updates. Evaluation/checkpoint intervals default
to 500 instead of the pilot's 25. Use a new run name and keep checkpoint v2 resume.

Baseline training and the repository-context stage must consume this same frozen
cohort through their dedicated runners. Size comparisons must report examples,
input/supervised token totals and compute, not equate 100 updates to one epoch.

## Repository-context corpus

SWE-Fixer and SWE-rebench V2 tables are separately audited in full for Python,
benchmark repository intersections, missing fields and duplicate issue/patch
lineage. This table audit is **not** training admission. Before real repository
SFT, fetch immutable Git before snapshots, inspect source licenses, select context
from the public issue only, validate patch roundtrips and profile the resulting
context. Gold file identities, hints, tests and solution prose cannot become
inference input. This work remains part of the full study rather than being
replaced by the instruction-editing pool.

## Remaining experiments

Complete the pool and inspect its counts; validate native full-model memory at
selected length boundaries; run the full matrix with matched manifests and
resume support; generate/scorer-evaluate the held-out benchmarks. The 100-step
pilot only validates training mechanics. Gold-scorer checks are not model scores.

The trainer records `step_seconds` after CUDA synchronization, evaluation time, and checkpoint/segment wall time. These measurements are collected on the actual run; no GPU-hour estimate is derived from the short pilot for longer-context models.


## Verified complete instruction foundation

The full local job scanned 459,750 upstream records across all three sources.
50,461 unique candidates passed static admission; the 2,048/2,048/4,608 gate
retained **49,259 examples (97.62%)**, split into **44,333 train / 4,926 dev**.
The training split has 16,513,084 supervised target tokens. One epoch at batch 1,
accumulation 4 is **11,084 optimizer updates**. These are actual data counts,
not trained-model results. Details: [receipt](data/full_instruction_pool_20261009.json).

Before caps, P95 lengths were encoder 1,119, paired decoder 1,540 and baseline
2,385; P99 paired decoder was 2,474. The 2k gate discards a measured 2.38% tail
without truncation. Full-tail budgets would require a new GPU resource gate,
not simply setting 32k/8k.

## Manual Molab upload

Upload the regenerated symmetric/asymmetric notebook `.py` only. Apply the HF
read token, then select DATASET_REPO_ID, DATASET_REVISION and DATASET_COLLECTION.
The normal run downloads an immutable published pool and does not build snapshots.
The repository collection is not ready until a complete repository manifest
is uploaded. Do not silently substitute the instruction foundation.
Current local limits remain encoder8192 /paired decoder2048 /baseline10752.
For instruction-only editing, explicitly select its published collection,
CORPUS_KIND=instruction and the matching LOCAL_DATA_FILE.

The current notebook default selects the published instruction foundation;
repository curation and its collection are still pending.

The repository budget targets at most 10,000 accepted examples, at most 50 per
repository, and 20 GiB compressed archives. These are explicit resource controls,
not a promise of 10,000 usable tasks. Rejections, quotas and budget stops remain
in the manifest. Use CORPUS_KIND=instruction plus its foundation directory for
an instruction-only ablation. The main repository study does not silently fall
back to EditPackFT or fixtures.

The universal repository notebook accepts the same frozen pool for all six models,
including decoder-only baselines, and supports checkpoint-v2 RESUME_FROM. Model
weights load only after data preparation/tokenization and GPU preflight. No RL,
VM, Docker, teacher model or PEFT package is needed by this recipe.

The runtime now follows conda ai: Transformers 5.19.0, Torch 2.14.1 with CUDA 13.0,
PyArrow 25.0.1, tokenizers 0.23.2, huggingface-hub 1.33.0 and NumPy 2.5.3. CUDA
wheels use the explicit official cu130 uv index. No pip/reinstall cell modifies
the active runtime. Historical pilot measurements keep their original versions.

Full repository-table audit found 7,243 Python SWE-rebench V2 rows; 7,192 remained after 46 benchmark-repository and five duplicate-lineage exclusions. These are candidates before snapshot/license/context gates. The 10,000 target is therefore an upper budget, not an admitted-count promise. SWE-Fixer has 114,352 table candidates across 846 repos, but every row lacks a source-code license field; actual pinned-source license evidence is mandatory. See [table audit](data/repository_table_audit_20261009.json).

Repository preparation now retries incomplete HTTP archive transfers and checkpoints each accepted/excluded outcome in SQLite. Re-running unchanged settings resumes the source cursor and accepted records; settings fingerprints prevent mixing a new corpus policy into old state. Source archives remain cached and no repository code is executed.

The universal full-pool runner now binds checkpoint compatibility to model source references, backbone dimensions, Torch version and Transformers version as well as the frozen data/policy fingerprint. Runtime drift is rejected during resume rather than silently accepted.

## Repository target lengths: complete raw Python profile

All 32,079 pinned SWE-rebench V2 rows were scanned; all **7,243 Python rows**
had usable issue/patch fields and were measured with the audited official IT
chat template on Transformers 5.19.0. The paired decoder length includes the
issue prompt, repair patch and closing turn, with one shifting operation.
This is a raw-source profile, before snapshot, license, overlap and patch gates.
It does not measure encoder context or baseline decoder lengths.

| Paired decoder cap | Raw Python rows within cap | Coverage |
| --- | ---: | ---: |
| 2,048 | 3,944 | 54.45% |
| 4,096 | 5,868 | 81.02% |
| 8,192 | 6,954 | 96.01% |
| 16,384 | 7,241 | 99.97% |

Target-only P50/P95/P99 are 1,261 / 6,933 / 9,517 tokens; paired decoder
P50/P95/P99 are 1,848 / 7,638 / 10,427. The largest paired sequence is 20,471.
These results differ substantially from the shorter instruction-editing pool.
**The current 2k repository production pass is a bounded pilot/ablation, not
an adequate default for the full main-study population.**

An 8,192 paired-decoder budget is the next candidate for the main study: it
covers 96.01% of raw Python rows before the other admission gates. It must first
pass original-model forward/backward VRAM checks on Molab, including the E4B
pairs and decoder-only baselines. This is a total paired sequence budget,
not a guarantee of 8k generated patch tokens. Encoder retrieval length and
baseline context-plus-target limits require separate measurement. The table
cannot establish that 32k input is necessary or feasible.

Keep the running 2k pass and its immutable manifest separate. A larger-budget
pass must use a new output directory and ledger settings, reuse pinned archives,
freeze one common six-model cohort, and report overflow without truncation.
Changing a notebook knob does not enlarge an already-frozen pool; the matched
training notebooks reject a manifest with different limits.

The reproducible tool and aggregate-only [receipt](data/repository_raw_target_profile_20261009.json)
contain no upstream source code or model weights:

```powershell
$env:PYTHONPATH='src'
python tools/profile_repository_lengths.py --source swe_rebench_v2 --rows-file <pinned-parquet-path> --tokenizer-cache <verified-tokenizer-cache> --output outputs/repository-target-profile.json
```


All three curated-training notebooks now validate the requested encoder,
paired-decoder and baseline-decoder limits against the immutable pool before
reporting READY. Their bounded-pilot defaults agree at 2,048 / 2,048 / 4,608.
The universal notebook previously requested a 4k context while consuming pool
limits implicitly; this mismatch is fixed. READY displays the frozen limits.
For the next 8k-decoder study, choose a new pool directory and adjust both
encoder and decoder controls; do not reuse the 2k pilot directory or checkpoint.
No claim that the larger budget fits all original models is implied.


GSM8K is planned as supplementary reasoning-retention evaluation, with a separate math prompt and numeric scorer. Its measured eight-shot input maximum is only 1,280 tokens under the declared protocol; repository data motivates the larger window. See [GSM8K and context budgets](GSM8K_CONTEXT.md) for the 8k-encoder/8k-paired-decoder candidate and its required original-model GPU gate.


## Full secondary repository source profile: SWE-Fixer

The pinned source's **115,406 raw records** were all profiled with the same
IT issue-prefix/patch/closing-turn protocol. SWE-Fixer does not provide a
language field in the inspected schema; the task adapter defaults missing
language to Python. These are source-protocol candidate counts, not proof that
every changed file is Python or that every row is valid training data.

| Paired decoder cap | Raw rows within cap | Coverage |
| --- | ---: | ---: |
| 2,048 | 92,881 | 80.48% |
| 4,096 | 109,210 | 94.63% |
| 8,192 | 113,820 | 98.63% |
| 16,384 | 114,987 | 99.64% |

Paired decoder P50/P95/P99 are 1,009 / 4,257 / 9,714; target-only P95 is 2,788.
The maximum paired length is **1,199,365**, and the longest issue alone is
147,215 tokens. These outliers exceed the backbone position limit. Raising a
window to 32k cannot admit the whole source; overflow must remain an explicit
recorded rejection, without truncating gold patches.

This strengthens 8k paired-decoder length as a coverage candidate across both
repository sources, conditional on original-model training-memory validation.
It does not choose an encoder retrieval length. The earlier SWE-Fixer table
audit found 114,352 candidates after benchmark/lineage gates, all lacking
source-code license fields. Actual immutable snapshots and source-license
inspection are still required. SWE-Fixer is not silently added to the running
SWE-rebench production pass or called approved SFT by this profile.

The full source-file SHA, tokenizer hashes, aggregate distributions and ordered
per-row length-stream SHA are in the [receipt](data/swe_fixer_raw_target_profile_20261009.json).
Raw source text stays outside Git. Primary-source 8k coverage is 96.01%;
secondary-source 8k coverage is 98.63%, before their other admission gates.


## Corrected visible-context source licensing

A cached-snapshot audit of all 487 accepted records at a captured v2 production
checkpoint found **483 passing /4 requiring review** for visible context files.
The earlier gate verified licenses on patched files only. BM25 context can
include another file with a separate license boundary; target-file licensing
alone does not cover every piece of source shown to a training model.
These are project-policy review cases, not assertions of unlawful licensing.
[Audit receipt](data/repository_context_license_audit_20261009.json).

The v3 builder now verifies every unique selected source path independently of
gold paths, retains `context_source_code_license_evidence`, and records
`context_source_license_policy=nearest_license_for_each_visible_file_v1` in the
report. Failed context checks have separate recorded exclusion reasons. The
consumer rejects repository pools without this marker or complete per-path
license evidence before they can become training input. The instruction-only
foundation and synthetic legacy editing interfaces retain their own contracts.

The old producer was deliberately stopped for this coverage correction; its
v2 ledger and source archives remain preserved. Its accepted count is not a
completed approved corpus. The corrected production pass revalidates cached
source using a new ledger signature and produces the final pool only after
all gates pass. Settings now bind the study config, tokenizer assets and
Python/tokenizer/Transformers runtime as well as budgets and source hashes.
There is no silent migration of v2 acceptance into v3.

Historical four/64-task adapter receipts continue to describe their original
validation scope; they are not license-complete SFT approval. A separate
[four-real-task v3 check](data/repository_context_license_smoke_v3_20261009.json)
verifies the strengthened exporter and consumer using cached immutable source.


## Parser warning diagnostics (10 October 2026)

Local code now captures AST-emitted SyntaxWarning per target, retaining category,
kind, source/row or task/file identity and line number. Target bytes, patches,
normalization hashes and acceptance criteria are unchanged. SyntaxError,
ValueError and RecursionError still fail the syntax gate; other warning
categories are re-emitted after the scoped capture. No target source is executed.

Instruction progress/report summaries distinguish all parsed warning-bearing
examples from admitted examples; final train/dev summaries describe the cohort
after token caps. Records keep target_syntax_warnings and quality_flags. Repository
outcomes checkpoint these diagnostics and rebuild summaries on resume. Its ledger
version is repository_archive_v4_syntax_diagnostics so old unmeasured outcomes
cannot silently become "warning-free" v4 records.

Existing frozen pools remain immutable and show not_collected_in_this_frozen_pool
in the notebook's READY table if they predate these diagnostics. New diagnostics
require a newly prepared pool/run identity. Parser warnings are quality signals,
not functional verification. This local update does not change a running Molab
kernel, its cells or its data preparation. The current run can finish with its
existing code; upload the regenerated notebooks for a later run.


## Encoder8k default selected by the user (10 October 2026)

Local training notebooks and repository preparation defaults now use 8192
encoder/retrieval tokens. Paired decoder remains 2048 total; matched baseline
budget is 10752. New pool names distinguish enc8k_dec2k from prior 2k artifacts;
historical receipts and cohorts are not rewritten. This changes preparation
settings and requires a fresh pool/run. It does not establish training VRAM
feasibility and does not change the live Molab kernel.


## Current v2 remediation policy (11 October 2026)

The default frozen dataset is instruction_enc8k_dec2k_v2. Historical counts
above describe their named earlier builds; use [the v2 audit](DATASET_REMEDIATION_V2.md)
for current counts and measurements. Normal training consumes the pinned private
HF release, including validation of an already existing local directory.

Fresh instruction builds now apply event and AST-pair dedup, preserve discarded
repository/commit lineage, quarantine conflicting targets, exclude doc/format-only
edits from this cohort, and reject target SyntaxWarnings. Python normalization
retains INDENT/DEDENT/NEWLINE; LF-only diff generation matches Git line counting.
Old persisted overlap indexes must be rebuilt under the new normalization policy.

NextCoder retains MIT dataset metadata but marks source-code license unknown;
synthetic seed hashes are not Git commits. Private staging does not settle
underlying code attribution. All training records are auxiliary single-file
editing candidates; static checks do not prove bug repair or functional quality.
