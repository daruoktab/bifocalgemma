# BifocalGemma: Bidirectional Repository Encoding for Local Code Repair

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](pyproject.toml)

BifocalGemma is a research framework for repository-level code repair using
Gemma-4 text backbones. A bidirectional repository encoder feeds a SwiGLU
dual-gate bridge and decoupled full-rank cross-attention in a causal decoder.
The training data contract supports 32K encoder context and an 8K decoder budget.

The active implementation uses canonical modules under `src/bifocalgemma/`.
The former `v3/` and `models/v2/` trees have been unified and removed.
See [the module map](docs/REPO_LAYOUT.md), [notebook guide](docs/TRAINING_NOTEBOOKS.md),
and [paper](paper/main.tex). Paper result placeholders remain pending measured
training and evaluation; passing engineering checks does not establish model quality.

## Ten-model experiment matrix

Each notebook runs one configuration under the active `v4` profile, then its ten configured benchmarks.

| Encoder | E2B-IT decoder | E4B-IT decoder |
| --- | --- | --- |
| E2B base | [2base–2it](notebooks/train_2base_2it.py) | [2base–4it](notebooks/train_2base_4it.py) |
| E2B IT | [2it–2it](notebooks/train_2it_2it.py) | [2it–4it](notebooks/train_2it_4it.py) |
| E4B base | [4base–2it](notebooks/train_4base_2it.py) | [4base–4it](notebooks/train_4base_4it.py) |
| E4B IT | [4it–2it](notebooks/train_4it_2it.py) | [4it–4it](notebooks/train_4it_4it.py) |
| Decoder-only | [E2B-IT baseline](notebooks/train_e2b_it.py) | [E4B-IT baseline](notebooks/train_e4b_it.py) |

The active `v4` paired configurations use 270 Stage-1 alignment steps followed by
4,224 Stage-2 joint SFT steps (4,494 total steps, 1.5 epochs). Decoder-only
baselines use one continuous 4,494-step SFT stage. Micro-batch size 2 and
gradient accumulation 8 give an effective batch of 16 over 47,932 curated
training examples (`mixed_context_enc32k_dec8k_v4`, revision `4b8d6b36…`). The
immutable schedule and checkpoint identity govern resume behavior. The
six-model construction registry used by `inspect` and `validate-config` remains
separate from this ten-model training matrix.

The active `v4` benchmark suite covers ten tasks: LongCodeQA, HumanEvalFix
(unified-diff and full-function modes), SWE-QA oracle/noisy, GSM8K 0/8-shot,
CrossCodeLong chunk/function completion, and RepoNIAH retrieval. Heavy
completion cohorts sample 500 eligible cases deterministically; the remaining
cohorts use their configured full sets.

## Run a standalone notebook

Upload one matrix notebook from the table to Molab. Its setup verifies the
pinned runtime ZIP and prepares dependencies before displaying the HF token form.
Provide `HF_TOKEN` or submit the form, then run training and evaluation. Each
configuration writes to its configured model repository; inspect the notebook
and [config registry](src/bifocalgemma/config.py) for the exact destination.

The notebook includes a SHA-256 and immutable Hub revision. A local
`LOCAL-PREVIEW` requires its matching ZIP and cannot be shipped alone.
[Runtime publication](docs/RUNTIME_RELEASE.md) describes generation and release checks.

**Preserve existing Molab runs:** keep their original notebook, pinned runtime,
and checkpoint artifacts. The clean unified implementation intentionally rejects
pre-refactor checkpoints and schedules. Publishing a new runtime adds an immutable
ZIP and leaves `runtime/latest.zip` and old remote artifacts unchanged. Adopting
the new notebook is an explicit new-run decision; do not replace the source of
an active old kernel.

## Local setup and checks

Use the Python environment selected for this project. The established development
environment is conda `ai`; package installation uses `uv pip`.

```powershell
conda activate ai
uv pip install --python (Get-Command python).Source -e ".[dev,notebook]"
$env:PYTHONPATH = 'src'

python -m bifocalgemma inspect --experiment BifocalGemma-E4B-E2B
python -m bifocalgemma validate-config --config configs/experiments.json
python -m bifocalgemma smoke-native --experiment all --device cpu
python -m pytest -q
pyrefly check
ty check --python (Get-Command python).Source src tools tests
ruff check --fix src tools tests tests_e2e notebooks
Get-ChildItem notebooks/*.py | ForEach-Object { python -m marimo check --strict $_.FullName }
python tools/generate_full_matrix_notebooks.py --check --require-published
python tools/generate_repository_notebook.py --check
```

Inspect, config validation, and native tiny smoke do not download model weights.
Tiny random-weight checks validate implementation behavior, not pretrained repair
performance. Configure Pyrefly's interpreter in `pyproject.toml` for your selected
environment. The generic `train` and Docker-harness `evaluate` CLI commands remain
explicit placeholders; use the matrix notebooks for the active experiment flow.

## Model API and other entry points

`BifocalGemmaConfig` and `BifocalGemmaForCausalLM` provide the Hugging Face model
interface, standalone save/load, and generation with encoder context.

```python
from bifocalgemma import BifocalGemmaForCausalLM

model = BifocalGemmaForCausalLM.from_pretrained("path/to/exported/model")
tokens = model.generate(
    input_ids=decoder_prefix_ids,
    encoder_input_ids=repo_context_ids,
    max_new_tokens=256,
)
```

The [fast-training notebook](notebooks/train_bifocalgemma_fast.py) and
`python -m bifocalgemma train-fast --help` expose a separate training workflow.
The [repository-study notebook](notebooks/bifocalgemma_repository_study.py) provides
offline preparation, generation, and native scoring. These entry points do not
replace the ten-model matrix protocol.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/bifocalgemma/modeling_bifocalgemma.py` | Native model and Hugging Face API |
| `src/bifocalgemma/models/` | Bridge, cross-attention, embeddings, memory helpers |
| `src/bifocalgemma/training/` | Matrix, schedule, optimizer, checkpoint, resume, telemetry |
| `src/bifocalgemma/data/`, `evaluation/`, `inference/` | Data contracts and evaluation |
| `notebooks/`, `tools/` | Standalone entry points and deterministic generators |
| `tests/`, `tests_e2e/` | Numerical, protocol, recovery, and integration checks |
| `paper/`, `docs/` | Manuscript, active guides, and historical research records |
| `outputs/` | Local caches/results; ignored by Git |

Frozen data releases live in the separate private
[HF dataset repository](https://huggingface.co/datasets/daruokta/bifocalgemma-data).
See [dataset releases](docs/DATASET_RELEASES.md) for revision and gold-separation
contracts. Model weights, large datasets, credentials, and local run results are
not bundled in this Git repository.

Start with [the documentation index](docs/README.md). Older V2 and sequential
guides are retained as historical records and do not specify the active matrix.

Contributors: [daruoktab](https://github.com/daruoktab) and
[attmhd](https://github.com/attmhd). Code is licensed under [Apache 2.0](LICENSE).
