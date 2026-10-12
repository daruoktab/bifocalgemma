# Private curated dataset releases

The data repository is [daruokta/bifocalgemma-data](https://huggingface.co/datasets/daruokta/bifocalgemma-data),
private. Code and model artifacts remain separate.
[dataset_release.lock.json](../configs/dataset_release.lock.json) binds the immutable
Hub revision, training collection and release fingerprint.

## Context v3

The active training collection is `mixed_context_enc32k_dec8k_v3`:
**44,408 train /4,934 dev**, 49,342 records total. It combines 44,956 verified editing
foundation records with 4,386 repository repair/localization/completion records.
The training split contains16,681,401 supervised target tokens. Limits are32768
encoder,8192 prefix-inclusive shifted paired decoder and41472 baseline decoder.

The release contains one training config and eight evaluation configs. The default
suite covers30,600 case/protocol runs per model, with shared question IDs across
SWE-QA conditions and GSM8K shot protocols. Counts are not a claim of independent
tasks or model quality. See [the detailed v3 specification](CONTEXT_DATA_RELEASE_V3.md)
for sources, task counts, token distributions, exclusions and evaluation adaptations.

V2 at `d34a19b3badc0120755495c68227cba038819cd7` remains available in history:
40,706 train /4,524 dev single-file editing examples, HumanEvalFix164 and two GSM8K
protocols over1,319 test questions. V3 starts from that verified foundation rather
than the45,303-row intermediate folder. Historical v1/v2 data are not concatenated
into the current training config. A changed cohort requires a fresh RUN_NAME;
old optimizer/cursor state must not resume against v3.

## Layout and contracts

- `release.json`: roles, counts, archive/member hashes, evidence and benchmark status.
- `collections/<training-name>/pool.zip`: train/dev JSONL and the frozen manifest.
- `collections/<benchmark-name>/public.zip`: target-free model inputs and required snapshots.
- `collections/<benchmark-name>/gold.zip`: scoring references, downloaded explicitly.
- `collections/<name>/view/*.parquet`: typed fields and lossless `record_json`.
- `audits/`: frozen validation evidence; `notices/`: credited cards and exact license bytes.
- `inventory.json`: SHA256 and byte count for every packaged file.

Viewer schema v3 exposes `task_type`, `target_text`, condition, context budgets,
question, shots and protocol. Repair targets are unified diffs; localization and
completion targets must be read from target_text according to task_type. Different
objectives may share context without being duplicate input/target examples.

Gold separation is a loader contract, not access control. Repo collaborators can
read gold; generation receives public inputs only. Training configs contain targets,
evaluation views do not. Explicit HF configs prevent automatic concatenation of
different protocols, and consumers verify one named collection at one full SHA.

## Build and audit

Use env ai, without pretrained weight downloads or GPU training:

```powershell
python tools/prepare_context_data.py benchmarks
python tools/prepare_context_data.py repository
python tools/prepare_context_data.py tasks
python tools/prepare_context_data.py mix
python tools/audit_context_tokens.py
python tools/audit_context_tokens.py --legacy-only
python tools/build_context_release.py
```

Pinned source caches and the byte-verified v2 foundation are prerequisites. Outputs
are immutable; completed directories are not overwritten. Repository admission has
per-outcome SQLite recovery. `--cached-snapshots-only` resumes admission without
repeating downloads; unavailable archives remain explicit exclusions.

The mixed audit verifies every row, unique pairs, split disjointness and visible
context licenses. Foundation inputs/targets are byte-identical to the verified Git
patch/after-byte proof and compile warning-free. Repository targets apply against
the pinned buggy snapshot; completion masks exclude the target and its AST body is
warning-free. Every Parquet record is compared with its native archive row.

These checks do not establish functional SWE-rebench quality or semantic/pretraining
decontamination. NextCoder seed lineage/code licensing and benchmark snippet licenses
are not independently certified. Keep upstream notices and per-record provenance;
there is no blanket source-code license for the mixed release.

## Publication and download

```powershell
$env:PYTHONPATH='src'
python -m bifocalgemma.data.releases publish --directory outputs/dataset_release_context32k_v3 --repo daruokta/bifocalgemma-data --replace-current --expected-parent d34a19b3badc0120755495c68227cba038819cd7
python -m bifocalgemma.data.releases fetch --repo daruokta/bifocalgemma-data --revision <locked-full-sha> --collection mixed_context_enc32k_dec8k_v3 --output outputs/mixed_context_enc32k_dec8k_v3
```

Publication verifies owner/privacy and expected parent. Replacement changes the
current config tree in one versioned commit while preserving prior revisions.
Consumers reject moving revisions, wrong roles, unsafe paths and corrupt bytes.
An existing cache must match the selected collection fingerprint. Eval caches can
be reused after verifying all selected members; include_gold is opt-in.

The three training notebooks consume the curated HF release and perform full SFT,
upload, benchmark, upload and then the next model. They have no source-curation or
sampling fallback. Full-run/upload flags default True and can be disabled by the
user. See [sequential execution and checkpoint recovery](SEQUENTIAL_TRAINING.md).
