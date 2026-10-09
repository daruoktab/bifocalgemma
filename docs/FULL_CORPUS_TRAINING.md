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

The two matched notebooks retain 19 cells / 18 identical bodies. Their new
default is `DATA_SOURCE="curated_pool"`, with `LOCAL_DATA_FILE` pointing to the
pool directory. `RUN_TRAINING=False` remains the startup guard. `MAX_STEPS=0`
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
