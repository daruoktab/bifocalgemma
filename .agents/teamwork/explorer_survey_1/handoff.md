# Handoff Report: Specification Mining on Data Contracts and Evaluation Benchmarks

**Agent:** `teamwork_preview_spec_miner` (`explorer_survey_1`)  
**Date:** 2026-10-07  
**Recipient:** `orchestrator_1` (`25871991-f75d-4c31-a84f-84a5e07b84ad`)  
**Status:** Complete  

---

## 1. Observation

Authoritative findings gathered directly from codebase documentation, source modules, training notebooks, configurations, and test suites.

### 1.1 Specification Mining: Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Dataset | `nuprl/EditPackFT` | Primary pilot dataset for Python instruction editing | HuggingFace dataset `nuprl/EditPackFT`, rev `067b4c882f44eca19329fedd275438fc3c060f3f`, file `data/train-00000-of-00001.parquet` | ~22,602 raw records; fields: `instruction`, `old_contents`, `new_contents`, `old_file`, `new_file`, `commit`, `repos`, `license` | Rejects missing instruction, unchanged code, rename/add/delete, bad paths | `docs/DATASETS_AND_BENCHMARKS.md:12`, `notebooks/bifocalgemma_train_symmetric.py:1611` |
| 2 | Dataset | `bigcode/commitpackft` (Python) | Optional supplemental dataset | HuggingFace dataset `bigcode/commitpackft`, rev `fc56fe33c030c6daa414c2b112c932b8eed085e6`, file `data/python/data.jsonl` | 56,025 raw records; fields: `subject`, `old_contents`, `new_contents`, `old_file`, `new_file`, `commit`, `repos`, `license` | Same record validator; requires commit/pair dedup with EditPackFT | `docs/DATASETS_AND_BENCHMARKS.md:13`, `notebooks/bifocalgemma_train_symmetric.py:1616` |
| 3 | Dataset | `local_jsonl` | Local dataset source | Local file path (`data/editing_records.jsonl`) | JSONL records matching standard schema | Fails if file missing or lines fail schema | `notebooks/bifocalgemma_train_symmetric.py:55, 1624` |
| 4 | Data Contract | `editing_record` | Validates raw record and generates deterministic unified diff | Raw dict containing `instruction`/`subject`, `old_contents`, `new_contents`, `old_file`, `new_file`, `commit`, `repos`, `license` | Normalized dict: `instance_id`, `group`, `pair_hash`, `instruction_hash`, `path`, `before`, `after`, `instruction`, `target_patch`, `source`, `commit`, `repos`, `license` | Raises `ValueError` for missing instruction, empty/unchanged code, renames/adds/deletes, unsupported paths | `src/bifocalgemma/data/offline_sft.py:11-64` |
| 5 | Data Pipeline | Deduplication | Removes exact before/after code duplicates | List of validated records within `raw_limit` | Filtered records list; counts logged to `filters["duplicate_before_after"]` | Silently counts and drops duplicates | `src/bifocalgemma/data/offline_sft.py:79-82` |
| 6 | Data Pipeline | `split_editing_records` | Commit-group held-out train/dev splitting | Deduplicated records, `seed=17`, `dev_fraction=0.1`, `train_limit=2000`, `dev_limit=200` | `(train, dev)` tuple where commit groups are strictly disjoint (`set(train_groups) & set(dev_groups) == empty`) | Raises `ValueError` if `< 2` distinct commit groups or split becomes empty | `src/bifocalgemma/data/offline_sft.py:87-101` |
| 7 | Tokenization | `tokenize_editing_row` | Offline tokenization for paired encoder-decoder model | Validated record, `ExperimentSpec`, encoder tokenizer, decoder tokenizer, sequence limits | `TokenizedRepairExample`: `instance_id`, `encoder_ids`, `prefix_ids`, `target_ids` | Raises `ValueError` on token overflow, boundary mismatch, or special token violations | `notebooks/bifocalgemma_train_symmetric.py:1650-1688`, `src/bifocalgemma/data/tokenization.py:47-120` |
| 8 | Batching | `RepairCollator` | Left-padded batch assembly with teacher forcing labels | List of `TokenizedRepairExample` | `Seq2SeqBatch`: `encoder_input_ids`, `encoder_attention_mask`, `decoder_input_ids`, `decoder_attention_mask`, `labels` (labels `-100` on prefix/padding) | Raises `ValueError` on empty batch, mixed baseline/pair, or negative token IDs | `src/bifocalgemma/data/collator.py:8-46` |
| 9 | Optimization | `patch_cross_entropy` | Aligned patch-only cross-entropy loss | Logits `[B, L, V]`, labels `[B, L]`, `ignore_index=-100` | Scalar float32 loss | Returns differentiable 0 if all tokens are ignored; fails on shape mismatch | `src/bifocalgemma/training/objectives.py:7-16` |
| 10 | Training Policy | Attention LoRA | PEFT policy for encoder/decoder attention projections | Models with Q/K/V/O projections; `rank=16`, `alpha=32`, `lr=2e-5`, `warmup_steps=5` | Trainable LoRA parameters FP32; frozen BF16 words and PLE; learned Linear H on asymmetric | Raises `ValueError` on unrecognized parameter or misconfigured policy | `src/bifocalgemma/training/policy.py:11-100` |
| 11 | Hardware Guard | `gpu_guard` | Preflight memory guard preventing OOM before download/training | `probe_estimate` dict, `device` | Returns `None` if safe; error string if device not CUDA, BF16 unsupported, or free VRAM < guard bytes | Halts training pipeline cleanly before any download | `tools/templates/training_notebook.py.txt:1122-1130`, `notebooks/bifocalgemma_train_symmetric.py:2177` |
| 12 | Benchmark Target | Held-out offline editing | Primary early validation metric | Local test split | Reconstruction match, exact edited-code match, syntax validity, held-out CE | Evaluates without code execution or VM | `docs/DATASETS_AND_BENCHMARKS.md:48` |
| 13 | Benchmark Target | HumanEvalFix / HumanEvalPack | Functional unit-level repair benchmark | Python unit repair tasks (`bigcode-project/octopack`) | Pass@1 functional execution rate | Execution separated from training; requires Python runner audit | `docs/DATASETS_AND_BENCHMARKS.md:49`, `docs/EVALUATION.md:11-16` |
| 14 | Benchmark Target | CanItEdit | Instruction-based Python code editing benchmark | 105 Python editing problems (`nuprl/CanItEdit`) | Functional pass rate per instruction edit | Scorer must be pinned; test execution in separate evaluation | `docs/DATASETS_AND_BENCHMARKS.md:50` |
| 15 | Benchmark Target | CrossCodeEval | Cross-file context understanding diagnostic | Preprocessed repository completion contexts (`amazon-science/cceval`) | Completion Exact Match (EM) % / similarity | Labeled as completion diagnostic, not repair success rate | `docs/DATASETS_AND_BENCHMARKS.md:51`, `src/bifocalgemma/evaluation/eval_code.py:105-156` |

---

### 1.2 Specification Mining: Edge Cases

| # | Feature | Input / Condition | Observed Behavior | Source Citation |
|---|---------|-------------------|-------------------|-----------------|
| 1 | `editing_record` | `instruction` is `""` or whitespace only | Raises `ValueError("missing_instruction")` | `src/bifocalgemma/data/offline_sft.py:15` |
| 2 | `editing_record` | `before == after` or either is empty | Raises `ValueError("empty_or_unchanged_code")` | `src/bifocalgemma/data/offline_sft.py:17` |
| 3 | `editing_record` | `old_file != new_file` (file rename, add, delete) | Raises `ValueError("unsupported_rename_add_delete")` | `src/bifocalgemma/data/offline_sft.py:20` |
| 4 | `editing_record` | Path contains `\`, `:`, `..`, whitespace, leading `/`, or hidden folder (e.g., `.git/foo.py`) | Raises `ValueError("unsupported_path")` | `src/bifocalgemma/data/offline_sft.py:30` |
| 5 | `editing_record` | File without trailing newline | Unified diff automatically appends `\ No newline at end of file\n` sentinel line | `src/bifocalgemma/data/offline_sft.py:43`, `tests/test_offline_sft_training.py:59` |
| 6 | `split_editing_records` | Fewer than 2 distinct commit groups | Raises `ValueError("Need at least two distinct commit groups for a held-out split.")` | `src/bifocalgemma/data/offline_sft.py:92` |
| 7 | `tokenize_editing_row` | Context length > `ENCODER_LIMIT` (2048) or full decoder length - 1 > `DECODER_LIMIT` (512) | Raises `ValueError("token_overflow")`; no silent truncation performed | `notebooks/bifocalgemma_train_symmetric.py:1687` |
| 8 | `tokenize_editing_row` | IT template BPE token merges across the prefix/target boundary | Raises `ValueError("prefix_boundary")` | `notebooks/bifocalgemma_train_symmetric.py:1682` |
| 9 | `tokenize_editing_row` | First token not BOS (2) or last target token not `<turn|>` (106) | Raises `ValueError("special_token_policy")` | `notebooks/bifocalgemma_train_symmetric.py:1685` |
| 10 | `RepairCollator` | Mixing decoder-only (encoder_ids=()) and pair examples in one batch | Raises `ValueError("Do not mix decoder-only and encoder-decoder examples in one batch.")` | `src/bifocalgemma/data/collator.py:18` |
| 11 | `gpu_guard` | `device != "cuda"` | Returns `"CUDA is required; no checkpoint downloaded."` | `tools/templates/training_notebook.py.txt:1124` |
| 12 | `gpu_guard` | `torch.cuda.is_bf16_supported() == False` | Returns `"BF16 CUDA support is required; no checkpoint downloaded."` | `tools/templates/training_notebook.py.txt:1126` |
| 13 | `gpu_guard` | Free VRAM < `guard_bytes` (weights + trainable*3 + headroom) | Returns `"Free VRAM X.XX GiB < guard Y.YY GiB."` and halts | `tools/templates/training_notebook.py.txt:1129` |
| 14 | Training Checkpoint | Run directory already contains previous artifacts (and `RESUME_FROM` is empty) | Raises `FileExistsError("Run directory already contains artifacts; choose new RUN_NAME or RESUME_FROM.")` | `src/bifocalgemma/training/offline_sft.py:1988` |
| 15 | Optimizer Accumulation | Accumulation window has 0 supervised tokens (all target tokens ignored) | Raises `ValueError("No supervised tokens in optimizer window.")` | `src/bifocalgemma/training/offline_sft.py:2050` |

---

### 1.3 Detailed Evidence for Key Question 1: Documented Datasets & Benchmark Targets

#### A. Documented Datasets and Formats (`docs/DATASETS_AND_BENCHMARKS.md`, `docs/DATA_PIPELINE.md`)
1. **Primary Pilot:**
   - **Name:** `nuprl/EditPackFT`
   - **HF Revision:** `067b4c882f44eca19329fedd275438fc3c060f3f`
   - **File:** `data/train-00000-of-00001.parquet`
   - **Record Count & Size:** ~22,602 rows, 48.1 MB Parquet.
   - **Role:** Primary instruction-to-patch Python training data.
   - **Constraint:** Not all records are verified bug fixes (mix of general editing and bug fixes).
2. **Optional Addition:**
   - **Name:** `bigcode/commitpackft` (Python subset)
   - **HF Revision:** `fc56fe33c030c6daa414c2b112c932b8eed085e6`
   - **File:** `data/python/data.jsonl`
   - **Record Count & Size:** 56,025 rows, ~136 MB JSONL.
   - **Role:** Supplemental candidate; requires strict dedup with EditPackFT (both source from GitHub commits and may share identical commits/diffs).
   - **Constraint:** Commit messages are often noisy descriptions, not clean bug reports.
3. **Local JSONL:**
   - **Path:** `data/editing_records.jsonl` (configurable via `LOCAL_DATA_FILE`).
4. **Deferred Datasets:**
   - `SWE-bench/SWE-smith-py`, `SWE-Gym/SWE-Gym`, `SWE-bench/SWE-smith-trajectories`, `SWE-bench Lite` (300 tasks), `SWE-bench Verified` (500 tasks).
   - Deferred because they require execution environments, Docker harnesses, or VM infrastructure contrary to the pure offline SFT design. Furthermore, SWE-smith records store bug-inducing patches and require reversing to obtain repair patches against specific verified snapshots.

#### B. Benchmark Targets & Exact Requirements
1. **Primary Validation: Held-out Offline Editing**
   - **Metric:** Held-out cross-entropy loss, patch application and reconstruction (`git apply` or string reconstruction check), exact edited code match, format and syntax validity.
   - **Requirements:** Purely offline on local files; zero execution of repository code or VM usage.
   - **Caveat:** Exact match is an editing diagnostic, not a functional pass rate (it can reject semantically correct alternative patches).
2. **Functional Unit Repair: HumanEvalFix / HumanEvalPack**
   - **Source:** `https://github.com/bigcode-project/octopack`
   - **Metric:** `pass@1` functional execution success rate.
   - **Requirements:**
     - Inference (generation) and test execution must be strictly separated.
     - BigCode evaluation harness separates generation and CPU evaluation (Docker available).
     - Test execution is NOT part of the training loop (no test-driven reward, no RL).
     - Python native scorer compatibility on Windows/Conda must be explicitly verified before running.
     - Strict contamination check: candidate training datasets must have zero overlap with HumanEvalFix problem statements, buggy implementations, or reference solutions.
3. **Instruction Editing: CanItEdit**
   - **Source:** `https://github.com/nuprl/CanItEdit`
   - **Metric:** Functional test execution pass rate on 105 Python editing tasks.
   - **Requirements:**
     - Official setup and test runner must be pinned and audited prior to use.
     - Purely an evaluation benchmark; test execution is strictly decoupled from SFT training.
     - Contamination check: candidate datasets must be audited against CanItEdit's 105 programs and instructions.
4. **Cross-file Completion Diagnostic: CrossCodeEval**
   - **Source:** `https://github.com/amazon-science/cceval`
   - **Metric:** Completion Exact Match (EM) percentage and code similarity.
   - **Requirements:**
     - Must be explicitly labeled and reported as a **"completion diagnostic"**, NEVER as a "repair resolved rate" or "bug fix success rate" (`docs/EVALUATION.md:13-16`, `src/bifocalgemma/evaluation/eval_code.py:93-97`).
     - Preprocessed context and official scorer must be pinned.
     - Contamination check: candidate datasets must be checked against CrossCodeEval prompt files and repositories.

---

### 1.4 Detailed Evidence for Key Question 2: Training Notebooks & Pipelines

Inspected `notebooks/bifocalgemma_train_symmetric.py` and `notebooks/bifocalgemma_train_asymmetric.py` (both 19 marimo cells, generated via `tools/generate_training_notebooks.py` from `tools/templates/training_notebook.py.txt`):

#### A. Expected Data Format / Schema
`dataset_rows` (`notebooks/bifocalgemma_train_symmetric.py:1623-1649`) and `editing_record` (`src/bifocalgemma/data/offline_sft.py:11-64`):
- Format supported: Parquet (via `pyarrow.parquet.ParquetFile.iter_batches(batch_size=256)`) or JSONL (line-by-line `json.loads`).
- Required Schema fields:
  ```python
  {
      "instruction": str,  # or "subject" if source == "commitpackft_python"
      "old_contents": str, # original source code before edit
      "new_contents": str, # modified source code after edit
      "old_file": str,     # relative POSIX path
      "new_file": str,     # must equal old_file (single-file modification)
      "commit": str,       # commit SHA / identifier (optional, fallback to pair_hash)
      "repos": str,        # repository name / aliases (optional)
      "license": str       # open-source license string (optional)
  }
  ```
- Field Constraints:
  - `old_file == new_file` strictly required (`ValueError("unsupported_rename_add_delete")`).
  - Path must be relative POSIX without `..`, `\`, `:`, whitespace, or hidden folder parts (`ValueError("unsupported_path")`).
  - `before != after` and both non-empty (`ValueError("empty_or_unchanged_code")`).
  - `instruction` non-empty after stripping (`ValueError("missing_instruction")`).
  - Never use precomputed `content` or `ndiff` as inputs, as they leak target solutions.

#### B. Input Formatting
In `tokenize_editing_row` (`notebooks/bifocalgemma_train_symmetric.py:1651-1666`):
- **Encoder Context String:**
  ```python
  context = f"### Instruction\n{row['instruction']}\n\n### File: {row['path']}\n{row['before']}"
  ```
- **Encoder Tokenization:**
  `encoder_ids = tuple(encoder.encode(context, add_special_tokens=True))`  
  Asserted: `encoder_ids[0] == 2` (Gemma BOS token ID).
- **Decoder User Prompt String:**
  ```python
  prompt = (
      "Return only a unified diff fulfilling the instruction using the encoder context.\n"
      + row["instruction"]
  )
  ```
- **Decoder Prefix:**
  Rendered using official IT chat template with `add_generation_prompt=True` and `enable_thinking=False`:
  ```python
  prefix = decoder.apply_chat_template(
      [{"role": "user", "content": prompt}],
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
  )
  prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
  ```
  Asserted: `prefix_ids[0] == 2` (BOS token supplied by template).

#### C. Target Formatting (Unified Diff / Patch)
In `editing_record` (`src/bifocalgemma/data/offline_sft.py:32-43`):
- Generated using `difflib.unified_diff`:
  ```python
  lines = list(
      difflib.unified_diff(
          before.splitlines(keepends=True),
          after.splitlines(keepends=True),
          fromfile=f"a/{path}",
          tofile=f"b/{path}",
          lineterm="\n",
      )
  )
  patch = f"diff --git a/{path} b/{path}\n"
  for line in lines:
      patch += line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
  ```
- In `tokenize_editing_row`:
  ```python
  messages = [
      {"role": "user", "content": prompt},
      {"role": "assistant", "content": row["target_patch"]},
  ]
  complete = decoder.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
  )
  full_ids = tuple(decoder.encode(complete.rstrip(), add_special_tokens=False))
  ```
  - Boundary check: `assert full_ids[:len(prefix_ids)] == prefix_ids` (no cross-boundary token merging).
  - Target tokens: `target_ids = full_ids[len(prefix_ids):]`.
  - Asserted closing turn token: `target_ids[-1] == 106` (`<turn|>`).
- Supervision in `RepairCollator` (`src/bifocalgemma/data/collator.py:37-45`):
  - Prefix tokens and padding get label `-100`.
  - Target tokens get teacher-forcing labels (aligned for next-token prediction).
  - Cross-entropy loss computed only on target tokens + `<turn|>`.

#### D. Commit-Grouped Splits and Deduplication
In `prepare_editing_records` and `split_editing_records` (`src/bifocalgemma/data/offline_sft.py:67-101`):
- **Pair Deduplication:** Exact `(before, after)` pairs deduplicated via `pair_hash = sha256(before + "\0" + after)`.
- **Commit Grouping:** `group = str(raw.get("commit") or pair_hash)`.
- **Split Logic:**
  - Unique groups shuffled with `random.Random(seed).shuffle(groups)` (default `seed = 17`).
  - Held-out count: `round(len(groups) * dev_fraction)` (default `dev_fraction = 0.1`).
  - Limits: `train_limit = 2000`, `dev_limit = 200`.
  - Strict assertion: `assert not ({r["group"] for r in train} & {r["group"] for r in dev})`.
  - Manifest label: `"split_policy": "commit_group_holdout_not_repo_disjoint"`.

#### E. Tokenizers, Sequence Lengths, and Special Tokens
- **Tokenizers:**
  - Pinned Hugging Face checkpoints:
    - Base Encoder: `google/gemma-4-E2B` (rev `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`) or `google/gemma-4-E4B` (rev `411aa17b749aa952df1359d2dcea73917a544d9a`).
    - IT Decoder: `google/gemma-4-E2B-it` (rev `3e22461f65e89153144f8adb70e3b8c2cc9845a7`) or `google/gemma-4-E4B-it` (rev `ee0ef6023621cff504d758262d4e04895a5af4a2`).
    - Decoder uses official `chat_template.jinja` (SHA256 `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`).
  - Vocabulary: BPE, 262,144 tokens, 514,906 merges.
- **Sequence Length Limits:**
  - `ENCODER_LIMIT = 2048`
  - `DECODER_LIMIT = 512`
  - Strict rejection: `if len(encoder_ids) > encoder_limit or len(full_ids) - 1 > decoder_limit: raise ValueError("token_overflow")`.
  - Explicit counting: No silent truncation.
- **Special Tokens:**
  - `<pad>`: ID `0` (used for left padding).
  - `<eos>`: ID `1`.
  - `<bos>`: ID `2` (asserted at `encoder_ids[0]` and `prefix_ids[0]`).
  - `<|tool_call>` / `<tool_call|>`: IDs `48` / `49`.
  - `<|tool_response>` / `<tool_response|>`: IDs `50` / `51`.
  - `<|channel>` / `<channel|>`: IDs `100` / `101`.
  - `<|turn>` / `<turn|>`: IDs `105` / `106` (`target_ids[-1] == 106` asserted).

#### F. Hardware Constraints & Training Configuration
- **Hardware Target:** Single GPU Molab (e.g., NVIDIA Blackwell 96 GB or similar single GPU).
- **Precision:** BF16 base weights (`torch.cuda.is_bf16_supported()` required), FP32 LoRA adapters and gradients.
- **Memory Footprint & Guard (`gpu_guard`):**
  - Text parameter counts:
    - E2B–E2B (tied ordinary words): ~8.85B parameters (~17.71 GiB BF16 weights; PLE + tied words alone ~9.5 GiB).
    - E2B–E4B / E4B–E2B (untied): ~12.09B parameters (~24.18 GiB BF16 weights).
    - E4B–E4B (tied ordinary words): ~14.25B parameters (~28.51 GiB BF16 weights; PLE + tied words alone ~11.75 GiB).
  - Guard calculation: `guard_bytes = weight_bytes + trainable_bytes * 3 + int(HEADROOM_GIB * 2**30)` where `HEADROOM_GIB = 16`.
  - Fails preflight on local laptop GPUs (e.g., 6 GiB VRAM).
- **Training Hyperparameters:**
  - `BATCH_SIZE = 1` (microbatch)
  - `GRAD_ACCUMULATION = 4` (effective batch size = 4)
  - `MAX_STEPS = 100`, `EPOCHS = 1`
  - `MAX_GRAD_NORM = 1.0`
  - Optimizer: `torch.optim.AdamW`, `weight_decay = 0.0`, `foreach = False`
  - Learning rate: `2e-5` per group with `warmup_steps = 5`
  - LoRA PEFT: attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), `rank = 16`, `alpha = 32`. Word embeddings and PLE tables frozen. Asymmetric H projection trainable `nn.Linear`.
  - Accumulation weighting: Token-weighted loss across the accumulation window (`output.loss * (count / token_count)`).
- **Prohibited Frameworks/Techniques:**
  - No Reinforcement Learning (RL / PPO / DPO).
  - No Reward models or Teacher distillation models.
  - No Agent rollouts or test-driven rollouts in the training loop.
  - No VM or Docker container execution during data prep or training.

---

## 2. Logic Chain

1. **Premise on Data Scope & Infrastructure:**
   - From `ORIGINAL_REQUEST.md:14-15`, `docs/DATASETS_AND_BENCHMARKS.md:3-6`, and `configs/train.example.json:16-19`, the repo recipe deliberately transitioned away from RL, agent rollouts, and SWE-bench task execution to pure offline SFT (`instruksi + kode sebelum edit → patch`).
   - Hardware is explicitly constrained to a single GPU on Molab without containerization or VM overhead.
   - Therefore, any candidate dataset that requires code execution, docker images, or test evaluation during data preparation or training is fundamentally incompatible with the repository pipeline.

2. **Premise on Data Loader & Contract Compatibility:**
   - From `src/bifocalgemma/data/offline_sft.py:11-31` and `notebooks/bifocalgemma_train_symmetric.py:1650-1688`, the data ingestion code enforces strict structural invariants:
     - Field `old_file` must equal `new_file`, meaning modifications must occur on an existing single file.
     - POSIX relative paths without `..`, `\`, `:`, whitespace, or hidden directories.
     - `before` and `after` must be different strings (non-empty).
     - Target diff is constructed using `difflib.unified_diff` with git patch headers and newline sentinels.
   - Any candidate dataset that features multi-file diffs, file creations/deletions, rename operations, or non-POSIX paths will be rejected by the validation assertions.

3. **Premise on Sequence Lengths & Filtering:**
   - From `notebooks/bifocalgemma_train_symmetric.py:1686-1687` and `src/bifocalgemma/data/tokenization.py:99-100`, records are strictly filtered if `len(encoder_ids) > 2048` or `len(full_ids) - 1 > 512`.
   - Silent truncation is forbidden to prevent corrupting patch syntax or partial prompt semantics.
   - Therefore, candidate datasets consisting of massive repository files or large diffs will suffer massive record dropouts during token filtering. Datasets with focused function/class-level edits will retain a much higher yield.

4. **Premise on Benchmark Alignment & Contamination:**
   - The three evaluation benchmarks established in `docs/DATASETS_AND_BENCHMARKS.md:46-52` are:
     1. HumanEvalFix / HumanEvalPack (Python unit-level repair)
     2. CanItEdit (Python instruction-based editing, 105 programs)
     3. CrossCodeEval (Cross-file context completion diagnostic)
   - To genuinely evaluate and improve model capability without artificial test leakage (per `ORIGINAL_REQUEST.md:25-27`), candidate training datasets must undergo rigorous contamination checks (repository origins, commit hashes, problem text matching) against these three benchmarks. Datasets with known overlap cannot be recommended.

---

## 3. Caveats

1. **Absence of Pretrained Big-Model SFT Logs:**
   - While data helpers and native tiny LoRA models have been thoroughly verified with unit tests (`tests/test_offline_data.py`, `tests/test_offline_sft_training.py`), full-scale SFT runs on the 96 GB Molab GPU have not yet been executed in this local session. Actual peak VRAM during long-sequence training remains implementation-derived.
2. **Commit Message vs Bug Fix Quality:**
   - Datasets sourced from git commits (like `commitpackft`) frequently contain noisy commit messages or refactorings that are not verified bug fixes. Filtering for high-quality instruction-editing semantics remains a challenge.
3. **Multi-file Context Gap:**
   - The current data pipeline only supports single-file paired modifications (`old_file == new_file`). Cross-file context understanding (relevant to CrossCodeEval) is planned via static multi-file records in future milestones, but cannot be processed by the current v1 data loader without code changes.

---

## 4. Conclusion

### Summary of Authoritative Data Contracts & Pipeline Requirements
1. **Data Schema:** Ingestion requires JSONL/Parquet with fields: `instruction` (or `subject`), `old_contents`, `new_contents`, `old_file`, `new_file`, `commit`, `repos`, `license`.
2. **Format Constraints:** Strictly single-file modifications (`old_file == new_file`), valid POSIX relative paths, non-empty and non-identical `before`/`after`. Diff is dynamically generated via `difflib.unified_diff`.
3. **Token Budgets:** Maximum 2048 encoder tokens and 512 decoder tokens under Gemma 4 BPE tokenizers; strictly rejects overflows.
4. **Hardware & Training:** Single GPU Molab (CUDA + BF16 required; >34 GiB free VRAM required by `gpu_guard` for E2B pairs with 16 GiB headroom), microbatch 1, accumulation 4, AdamW lr 2e-5, attention LoRA rank 16 / alpha 32. Zero RL, VM, Docker, or teacher networks.
5. **Benchmark Targets:**
   - Primary: Held-out offline editing (CE, patch reconstruction, exact match).
   - Evaluation: HumanEvalFix / HumanEvalPack (functional unit repair), CanItEdit (105-task instruction editing), CrossCodeEval (cross-file completion diagnostic only).

### Strict Compatibility Criteria for Candidate Datasets
To be used in the current offline SFT pipeline without requiring modifications to `src/` or `notebooks/`, any candidate dataset MUST:
1. Provide pure text/code pairs (instruction + before code + after code) requiring zero execution environments, compilers, or VMs.
2. Map directly into `instruction`, `old_contents`, `new_contents`, `old_file`, `new_file`, `commit`, `repos`, `license`.
3. Feature single-file edits on existing files (`old_file == new_file`).
4. Use standard relative POSIX file paths (no `..`, `\`, `:`, whitespace, or hidden parts).
5. Have code changes where `before != after` that cleanly reconstruct via unified diff.
6. Fit predominantly within 2048 encoder tokens and 512 decoder tokens to avoid excessive overflow rejection.
7. Provide distinct commit/group IDs to enable commit-grouped train/dev splitting without cross-split leakage.
8. Be provably free from contamination/overlap against HumanEvalFix, CanItEdit, and CrossCodeEval.
9. Possess a permissive open-source license.

---

## 5. Verification Method

To independently verify all findings and test suites reported here, run the following commands:

```powershell
# 1. Activate environment and set PYTHONPATH
$env:PYTHONPATH = "src"

# 2. Run unit tests for data contracts, tokenization, and offline SFT training
python -m unittest tests/test_offline_data.py tests/test_offline_sft_training.py tests/test_evaluation.py

# 3. Verify training notebooks syntax and formatting
python -m pytest notebooks/bifocalgemma_train_symmetric.py notebooks/bifocalgemma_train_asymmetric.py -q

# 4. Verify notebook generation tool consistency
python tools/generate_training_notebooks.py --check
```
