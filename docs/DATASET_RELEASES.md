# Private curated dataset repository

Data releases belong in **daruokta/bifocalgemma-data** on Hugging Face (dataset,
private), separately from this GitHub code/model repository. The logged-in HF
namespace is daruokta, which differs from the GitHub account daruoktab.

## Why freeze before Molab

Curate and validate once on CPU, then upload completed pools and evaluation
artifacts. Molab downloads one named collection at one full Hub commit SHA.
It does not repeat upstream downloads, snapshot materialization or curation when
RUN_TRAINING is enabled. Tokenization remains necessary for the selected model.

The uploaded October 10 log proves completion of an 8k instruction pool
(fingerprint 2d3a50c12d0521875b073b926771dbaa0327df758dfb6b13bb745aded88ea67d).
Its final repository progress event is 1,597 accepted /16,808 examined, with no
final repository manifest. Those are intermediate counts, not a frozen release.
A log cannot reconstruct the missing repository-task payloads or train/dev IDs.

The local 8k instruction rebuild completed: **44,366 train +4,930 dev**. Its
fingerprint exactly matches the logged completed Molab instruction pool. Release
revision: 75ea23ff82d54d6d0f663a33c1f0108ea5b59fc2. See
[verified receipt](data/private_dataset_release_20261010.json).
The local 8k build is independently curated from cached upstream artifacts;
its own hashes are authoritative, even if they differ from the Molab log.

## Layout and boundaries

- release.json: roles, collection kinds, native archive hashes and member hashes,
  counts, cohort token limits, provenance and benchmark status.
- collections/<training-name>/pool.zip: exact train.jsonl, dev.jsonl and manifest.json.
- collections/<benchmark-name>/public.zip: scorer-ready public inputs and required snapshots.
- collections/<benchmark-name>/gold.zip: scoring references, explicitly opted in.
- collections/<name>/view/*.parquet: a small stable viewer schema. Training targets
  appear only in training views; benchmark views contain public inputs only.
- notices/: pinned upstream cards and license texts. No blanket relicense of code.
- inventory.json: SHA256 and byte sizes for every release file.

Gold separation is a loader contract, **not access control** inside a private
repo: collaborators can download gold. Generation must use public inputs only.
The dev split is for tuning; evaluation benchmarks are never converted to train.
The six configurations must use the same frozen cohort.

## Build, publish and consume

Use env ai. No GPU, model weights, RL or VM is needed for release packaging.
Copy configs/dataset_release.example.json to outputs/dataset_release_spec.json
and fill its paths/notices after CPU curation. Production source cards and license
texts must be retained; the example contains no dataset payloads.

To curate a new 8k instruction pool (pinned public downloads are explicit):

```powershell
python -m bifocalgemma.data.full_pool --fetch --tokenizer-cache <verified-cache-root> --encoder-limit 8192 --decoder-limit 2048 --output outputs/full_instruction_pool_enc8k_dec2k_release_v1
```

The tokenizer root must hold the audited four Gemma tokenizer assets. This CLI
never downloads model weights. Use a new output directory for each build.

```powershell
$env:PYTHONPATH = 'src'
python -m bifocalgemma.data.releases build --spec outputs/dataset_release_spec.json --output outputs/dataset_release_v1
python -m bifocalgemma.data.releases publish --directory outputs/dataset_release_v1 --repo daruokta/bifocalgemma-data
python -m bifocalgemma.data.releases fetch --repo daruokta/bifocalgemma-data --revision <full-commit-sha> --collection instruction_enc8k_dec2k_v1 --output outputs/full_instruction_pool_enc8k_dec2k_v1
```

Publishing refuses a public destination. Revisions such as main are rejected by
the consumer. Fetch verifies archive SHA256, explicit member names, sizes,
member hashes and the original curated-pool fingerprint, then installs atomically.
An existing training pool must match the selected collection fingerprint.

Benchmark fetch is separate:

```powershell
python -m bifocalgemma.data.releases fetch --repo daruokta/bifocalgemma-data --revision <full-commit-sha> --collection gsm8k_zero_shot_v1 --role eval --output outputs/gsm8k_inputs
python -m bifocalgemma.data.releases fetch --repo daruokta/bifocalgemma-data --revision <full-commit-sha> --collection gsm8k_zero_shot_v1 --role eval --include-gold --output outputs/gsm8k_scoring
```

Default fetch cannot select an eval collection as train. Training fetch has no
gold option. Eval directories with existing contents are rejected rather than
silently mixing protocols. The Hub viewer's explicit configs prevent a recursive
glob from concatenating training, benchmark inputs and scoring references.
See [HF configuration documentation](https://huggingface.co/docs/hub/datasets-data-files-configuration).

## Training notebooks

The matched preparation cell loads a local frozen pool or downloads a named
collection from DATASET_REPO_ID at DATASET_REVISION. Curation is a separate,
explicit PREPARE_FULL_POOL=True operation. A normal training run never silently
builds a missing foundation or starts repository snapshot downloads.

Only select a collection with matching CORPUS_KIND and token limits. Selecting
instruction foundation is a separate editing experiment, not completion of the
repository study. The default current stage explicitly selects instruction_enc8k_dec2k_v1,
CORPUS_KIND=instruction, and its matching local path. The repository collection remains pending until a
complete licensed snapshot/context/patch cohort is produced and uploaded.
No incomplete ledger or synthetic smoke pool should be substituted for it.

## Benchmark release status

HumanEvalFix Python and GSM8K have local prepared artifacts and scorer checks.
These are adapted protocols, not proof of model quality or official runtime parity.
RepoQA is currently an answer-centered oracle-window diagnostic. CrossCodeEval
uses precomputed non-reference retrieval and adapted exact match. Their official
protocols and source-code notice coverage still need separate verification.
CanItEdit has quarantine artifacts but no production scorer here; RES-Q needs
native task dependency/harness validation. Catalog entries without admitted
payloads are references only, never reported as ready evaluation.

Freeze eval IDs and reference hashes before model experiments. Preserve upstream
benchmarks rather than claiming newly authored benchmarks. Native execution is
opt-in; static AST/patch checks do not prove functional repair.

## Repository manifest compatibility fix (11 October 2026)

Native repository_patch_v1 exports do not store the paired-file convenience
fields pair_hash and instruction_hash. The notebook previously indexed those
fields directly after tokenization and could fail with KeyError: pair_hash.
The shared editing_manifest now normalizes a temporary repository record using
its existing snapshot/patch contract. Hashes bind snapshot_sha256 plus the exact
target patch, and the normalized instruction. Dataset bytes, IDs, splits and
frozen-pool fingerprints are preserved. Already stored conflicting repository
hashes are rejected; eval records remain prohibited from training manifests.

All 49,296 published paired records produce the same run manifest as before.
The actual generated preparation helper also passes on a four-record native
repository export with real audited tokenizers for E2B-E2B, E4B-E4B, E2B-E4B
and E4B-E2B, without loading pretrained model weights. This validates the data
contract, not model training or repair quality. See
[schema audit](data/editing_manifest_schema_audit_20261011.json).

Normal notebook training still downloads the named instruction collection from
our private HF dataset repository at the locked commit. The repository corpus
and the other pending benchmark payloads remain separate unfinished stages;
this client fix does not certify them as ready datasets.
