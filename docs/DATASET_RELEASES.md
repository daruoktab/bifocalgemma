
# Private curated dataset releases

The active data repository is [daruokta/bifocalgemma-data](https://huggingface.co/datasets/daruokta/bifocalgemma-data), **private**.
The code/model repository remains [daruoktab/bifocalgemma](https://github.com/daruoktab/bifocalgemma).
[dataset_release.lock.json](../configs/dataset_release.lock.json) records the exact immutable Hub commit.

## Current v2 release

| Collection | Role | Records |
| --- | --- | ---: |
| instruction_enc8k_dec2k_v2 | Auxiliary single-file editing SFT | 40,706 train +4,524 dev |
| humanevalfix_python_v2 | Native Python repair evaluation adaptation | 164 |
| gsm8k_zero_shot_v2 | Math retention evaluation, zero-shot | 1,319 |
| gsm8k_eight_shot_v2 | Same math questions with eight demonstrations | 1,319 |

GSM8K has **1,319 unique test questions across both protocols**, not 2,638 unique tasks.
The v2 viewer shows question, shots and protocol explicitly. Training remains three-source
single-file editing; repository repair is a separate unfinished corpus. See the
[detailed remediation audit](DATASET_REMEDIATION_V2.md).

V1 at commit 75ea23ff82d54d6d0f663a33c1f0108ea5b59fc2 remains available for historical
reproducibility. Its 49,296 records and earlier validation claims are superseded for
new training. Current main contains v2 collections only; history is not rewritten.
Changing dataset version requires a new run; do not resume v1 optimizer state against v2.

## Layout and contracts

- release.json: roles, counts, native archive/member hashes, evidence and benchmark status.
- collections/<training-name>/pool.zip: train.jsonl, dev.jsonl, manifest.json and rejection ledger.
- collections/<benchmark-name>/public.zip: target-free public inputs and required snapshots.
- collections/<benchmark-name>/gold.zip: scoring references, opt-in only.
- collections/<name>/view/*.parquet: explicit typed columns plus lossless record_json.
- audits/: frozen CPU validation reports; notices/: exact credited source cards/license bytes.
- inventory.json: SHA256 and byte count for every packaged file.

Native container format names retain _v1 for reader compatibility; v2 collection names,
dataset_schema_version=2 on training rows, viewer_schema_version=2 and explicit policies
identify the corrected cohort. Source text is preserved; corrected patches use LF-only
Git line counting. Archive hashes bind the exact payloads, not merely row counts.

Gold separation is a loader contract, **not access control**: private collaborators
can access gold. Generation uses public inputs only. Training views contain targets;
evaluation views never contain answer, gold_patch, target_patch or test code.
Explicit HF configs prevent automatic concatenation of archives and different protocols.

NextCoder has MIT dataset metadata, unresolved original repository lineage and unknown
source-code license. Its legacy synthetic grouping hash is not a Git commit or a
verified original-code revision. Per-record upstream code licenses for Git exports are
declarations, not certification of the original checkout or complete notices.

## Build and audit

Use conda env ai; no model weights, GPU training, RL or VM is required.

```powershell
$env:PYTHONPATH = 'src'
python -m bifocalgemma.data.remediation --source <frozen-v1-pool> --output <new-pool> --tokenizer-cache <verified-tokenizer-cache>
python -m bifocalgemma.data.releases build --spec <filled-release-spec.json> --output <new-release-directory>
python tools/audit_dataset_release.py --pool <new-pool> --release <new-release-directory> --tokenizer-cache <verified-tokenizer-cache> --output <audit-report.json>
```

The remediation CLI rejects reuse of an output directory. Changed targets require
fresh verified tokenizer profiling. A fresh full-corpus build applies the same event,
structure, ambiguity, warning and compile gates. Older overlap indexes must be rebuilt
under python_structure_preserving_v2.

The full audit checks canonical patch hashes and applies **every** target with Git,
compares byte-exact after contents, verifies tokenizer assets/equivalence, rejects
overflow and compares every Parquet row against the native archive. An explicit
--reuse-validated-pool-audits mode binds a later viewer audit to completed frozen
patch/token evidence and unchanged subset identities; it does not invent fresh checks.
See the receipt for the actual mode used for this release.

## Publish and download

```powershell
python -m bifocalgemma.data.releases publish --directory <new-release-directory> --repo daruokta/bifocalgemma-data --replace-current --expected-parent <previous-full-sha>
python -m bifocalgemma.data.releases fetch --repo daruokta/bifocalgemma-data --revision <locked-full-sha> --collection instruction_enc8k_dec2k_v2 --output outputs/full_instruction_pool_enc8k_dec2k_v2
python -m bifocalgemma.data.releases fetch --repo daruokta/bifocalgemma-data --revision <locked-full-sha> --collection gsm8k_zero_shot_v2 --role eval --output outputs/gsm8k_v2_public
```

Publishing refuses a public destination and can enforce the expected parent commit.
Replacement changes current collection files in one versioned commit; old revisions
remain retrievable. The consumer rejects moving revisions such as main, wrong roles,
unsafe archive paths, corrupt bytes and a local training cache with another fingerprint.
Even an existing local directory is verified against the selected HF collection.

Evaluation gold requires --include-gold with --role eval. Eval directories must be new,
so protocols cannot be mixed silently. Normal notebook training downloads one frozen
named collection. Curation is available through the separate local CLI only;
training notebooks have no curation toggle or raw-source fallback. Full-epoch
training and family-notebook upload default to enabled and can be set to False.

## Not yet ready

repository_enc8k_dec2k_v1 has no complete production manifest. CanItEdit, RES-Q,
RepoQA and CrossCodeEval are quarantine/catalog references with their documented
scorer, environment or licensing work still pending. They are not silently counted
as published benchmarks. Native HumanEvalFix scoring is an adaptation; Python runtime
parity with the original release and model quality are not established by gold checks.

The earlier repository KeyError: pair_hash fix remains in place: editing_manifest
normalizes native repository exports in a temporary copy, preserving source bytes,
IDs and snapshot/patch identity. That schema fix alone does not finish the repository corpus.
