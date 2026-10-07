# Benchmark Provenance, Signatures, and Contamination Detection Protocol

**Author**: `teamwork_preview_explorer` (explorer_survey_3)  
**Parent Orchestrator**: `orchestrator_1` (conversation ID: `25871991-f75d-4c31-a84f-84a5e07b84ad`)  
**Scope**: HumanEvalFix (`bigcode/humanevalpack`), CanItEdit (`nuprl/CanItEdit`), CrossCodeEval (`amazon-science/cceval`).  
**Milestone**: Phase 0 — Survey & Scoping (Step 0)

---

## 1. Observation

### 1.1 HumanEvalFix (`bigcode/humanevalpack` / OpenAI HumanEval)
* **Primary Sources Observed**:
  - Hugging Face Dataset Card: `https://huggingface.co/datasets/bigcode/humanevalpack` (`README.md` inspected directly).
  - Foundation Paper: *"OctoPack: Instruction Tuning Code Large Language Models"* (Muennighoff et al., August 2023, arXiv:2308.07124).
  - Base Dataset: OpenAI HumanEval (*"Evaluating Large Language Models Trained on Code"*, Chen et al., July 2021, arXiv:2107.03374).
  - Evaluation Harness: `https://github.com/bigcode-project/bigcode-evaluation-harness` (task: `humanevalpack`).
* **License**: MIT (`license: mit`).
* **Volume**: 164 Python examples in the `test` split (config: `python`), corresponding to `task_id` values `"Python/0"` through `"Python/163"`.
* **Problem Construction**:
  - The Python split reuses the exact 164 problems from OpenAI HumanEval.
  - The canonical solution of each problem was corrupted with human/expert-written subtle bugs (`buggy_solution`).
  - Bug taxonomy is annotated via:
    - `bug_type`: `['missing logic', 'excess logic', 'value misuse', 'operator misuse', 'variable misuse', 'function misuse']`
    - `failure_symptoms`: `['incorrect output', 'stackoverflow', 'infinite loop']`
* **Exact Schema & Data Signatures**:
  | Field Name | Type | Description / Verbatim Content Signature |
  |---|---|---|
  | `task_id` | `string` | `"Python/0"` to `"Python/163"` |
  | `prompt` | `string` | Imports + function declaration + docstring with doctests |
  | `declaration` | `string` | Function signature without docstring |
  | `canonical_solution` | `string` | Reference bug-free Python solution passing all unit tests |
  | `buggy_solution` | `string` | Corrupted Python implementation with injected bug |
  | `bug_type` | `string` | One of the 6 fault categories |
  | `failure_symptoms` | `string` | One of the 3 runtime symptom categories |
  | `entry_point` | `string` | Exact function identifier (e.g. `has_close_elements`, `separate_paren_groups`, `truncate_number`, `below_zero`, `mean_absolute_deviation`, `intersperse`, `filter_by_prefix`, `factorize`, `concatenate`, `make_palindrome`, `string_xor`, ..., `generate_integers`) |
  | `test` | `string` | Deterministic unit test suite starting with `def check(entry_point):` containing assertions and ending with `check(entry_point)` |
  | `example_test` | `string` | Doctests parsed from problem docstring |
  | `signature` | `string` | Function header (e.g. `has_close_elements(numbers: List[float], threshold: float) -> bool`) |
  | `docstring` | `string` | Natural language description with example inputs/outputs |
  | `instruction` | `string` | Prompt template: `Write a Python function {signature} to solve the following problem:\n{docstring}` |

---

### 1.2 CanItEdit (`nuprl/CanItEdit`)
* **Primary Sources Observed**:
  - Hugging Face Dataset Card: `https://huggingface.co/datasets/nuprl/CanItEdit` (`README.md` inspected directly).
  - Foundation Paper: *"Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions"* (Cassano et al., COLM 2024 / arXiv:2312.12450).
  - GitHub Repository: `https://github.com/nuprl/CanItEdit`.
* **License**: MIT (`license: mit`).
* **Volume**: 105 hand-crafted Python programs in the `test` split (`id`: `0` to `104`).
* **Problem Construction & Provenance**:
  - Authored by a team of 8 experienced programmers to evaluate instructional code editing on self-contained programs.
  - Unlike CommitPackFT/EditPackFT (crawled from git commits), CanItEdit problems are **not** extracted from GitHub commit history; they are standalone Python files paired with a hidden test suite and human-validated edits.
  - Each challenge provides two distinct instruction variants:
    1. `instruction_descriptive`: Highly detailed, step-by-step specifications (simulating architectural plans, PR descriptions, or planning agents).
    2. `instruction_lazy`: Terse, informal, colloquial user prompts (simulating developer chat queries like "add caesar_cipher method to ciphers").
  - Context Type: **Full-file / Whole-file**. Each problem represents an entire Python script or module (often a complete class with multiple methods and imports). Both `before` and `after` are full-file source representations.
* **Exact Schema & Data Signatures**:
  | Field Name | Type | Description / Verbatim Content Signature |
  |---|---|---|
  | `id` | `int64` | Integer index `0` through `104` |
  | `name` | `string` | Short problem slug / identifier |
  | `full_name` | `string` | Descriptive title of the challenge |
  | `before` | `string` | Full Python source file before modification |
  | `after` | `string` | Target Python source file after modification |
  | `tests` | `string` | Test suite with unittests/assertions verifying both old functionality (regression test) and newly edited behavior |
  | `instruction_descriptive` | `string` | Verbose, explicit instruction specifying all technical requirements |
  | `instruction_lazy` | `string` | Concise, high-level developer instruction |
  | `taxonomy` | `struct` | Metadata object containing: <br>• `change_kind` (`string`): e.g. `adaptive`, `additive`, `refactoring`, `corrective`<br>• `libraries` (`sequence[string]`): external packages used (e.g. standard library, `numpy`, `math`)<br>• `topic` (`string`): problem domain (e.g. `Language`, `Algorithms`, `Data Processing`, `Games`, `Math`) |

---

### 1.3 CrossCodeEval (`amazon-science/cceval`)
* **Primary Sources Observed**:
  - GitHub Repository: `https://github.com/amazon-science/cceval`.
  - Foundation Paper: *"CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion"* (Ding et al., Columbia University & AWS AI Labs / Amazon Science, NeurIPS 2023 Datasets and Benchmarks Track, arXiv:2310.11248).
  - Role in BifocalGemma (`docs/DATASETS_AND_BENCHMARKS.md`, line 51): Designated as a "completion diagnostic" for multi-file context understanding (Exact Match / Edit Similarity), not an execution-based patch repair benchmark.
* **License**: Apache-2.0 (`Apache License 2.0`).
* **Volume (Python Subset)**:
  - 471 open-source GitHub repositories.
  - 1,368 Python source files.
  - 2,665 code completion samples.
  - Average prompt context length: ~3,390 tokens.
* **Problem Construction & Provenance**:
  - Sourced from real-world open-source GitHub repositories active up to mid-2023 (prior to NeurIPS June 2023 track deadline; v1 published October 2023).
  - Repositories were selected strictly with permissive licenses (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause) and filtered to exclude low-star / abandoned repositories.
  - Static analysis with Tree-sitter and symbol resolution identified cursor positions that **strictly require cross-file context** (dependencies spanning classes, helper functions, variables, or types across multiple files within the same repository).
  - Fixed endpoint commits: For each repository, a pinned commit SHA was frozen as the repository snapshot.
* **Exact Schema & Data Signatures**:
  | Component / Field | Type | Description |
  |---|---|---|
  | `repository` / `repo_name` | `string` | GitHub repository identifier (`owner/repo`) from the 471 Python repos |
  | `file_path` | `string` | Relative path of the target file within the repository |
  | `commit` | `string` | Pinned commit hash of the repository snapshot |
  | `prompt` | `string` | File prefix up to cursor position where cross-file dependency is needed |
  | `groundtruth` / `target` | `string` | Ground-truth code tokens immediately following the cursor |
  | `context` / `retrieval` | `list[dict]` | Cross-file snippets retrieved via BM25, UniXCoder, or dense embeddings |
  | Evaluation modes | N/A | `baseline` (in-file only), `retrieval` (in-file + retrieved cross-file snippets), `retrieval w/ ref` (ideal context) |
  | Metrics | N/A | Exact Match (EM) % and Edit Similarity (ES) |

---

## 2. Logic Chain

```
[Observation 1.1: HumanEvalFix uses 164 exact HumanEval function names and docstrings]
       │
       ▼
[Inference 1: Any dataset containing HumanEval function names (e.g. `has_close_elements`)
              or OpenAI HumanEval docstrings is directly contaminated]
       │
[Observation 1.2: CanItEdit is hand-crafted by 8 programmers with 105 full-file programs]
       │
       ▼
[Inference 2: CanItEdit code does NOT come from standard git commit dumps; any training sample
              matching CanItEdit `before`, `after`, or dual instructions is test-set leakage]
       │
[Observation 1.3: CrossCodeEval draws from 471 specific permissively licensed Python GitHub repos
                  frozen at mid-2023 pinned commits]
       │
       ▼
[Inference 3: Training sets drawn from GitHub commits (like CommitPackFT, EditPackFT, or raw git diffs)
              have a high risk of sharing commits or files with these 471 CrossCodeEval repos]
       │
       ▼
[Synthesis: A multi-tiered protocol combining (1) cryptographic exact hashing, (2) repository & commit
            intersection, and (3) token n-gram MinHash overlap is necessary and sufficient to prevent leakage]
```

1. **HumanEvalFix Contamination Vector**:
   - Because HumanEval was published in 2021, thousands of GitHub repositories, blogs, and synthetic instruction datasets (e.g., Code Alpaca, Evol-Instruct-Code, Magicoder) include solutions or variations of HumanEval problems.
   - Any candidate dataset generated from seeds that include HumanEval or containing the 164 canonical function signatures/docstrings must be detected and removed.
2. **CanItEdit Contamination Vector**:
   - CanItEdit was created by NUPRL (who also released EditPackFT).
   - If EditPackFT or candidate instruction datasets were synthesized using CanItEdit prompts or programs, models would trivially memorize the 105 test challenges.
   - Hash checking of the 105 `before`/`after` code blocks and `instruction_descriptive` / `instruction_lazy` strings provides a definitive filter.
3. **CrossCodeEval Contamination Vector**:
   - Because CrossCodeEval uses 471 real Python GitHub repositories, any commit-based dataset (such as `nuprl/EditPackFT`, `bigcode/commitpackft`, or other git diff datasets) might contain commits from the exact same repositories.
   - A candidate dataset containing commits from these 471 repositories risks memorizing the repository context, artificially inflating CrossCodeEval completion EM scores without genuinely improving multi-file reasoning.
   - Therefore, a repository-level and commit-level blacklist audit is required.

---

## 3. Measurable Contamination & Overlap Detection Protocol

The protocol defines concrete, computable rules to evaluate any candidate training dataset against the three target benchmarks.

```
Candidate Training Dataset
           │
           ▼
[Stage 1: Cryptographic Hash & Exact String Matching]
 ├── Check against 164 HumanEval function names, docstrings, unit tests
 ├── Check against 105 CanItEdit before/after hashes & dual instructions
 └── Check against CrossCodeEval ground-truth target tokens
           │
           ├── Match Found? ──► [CONTAMINATED (EXCLUDE)]
           ▼ No
[Stage 2: Repository & Commit ID Intersection]
 ├── Cross-reference `repo_name` against CrossCodeEval 471 Python repos
 └── Cross-reference `commit_id` against CrossCodeEval pinned commits (±30 days)
           │
           ├── Match Found on Benchmark Evaluated Files/Commits? ──► [CONTAMINATED (EXCLUDE)]
           ├── Match Found on Repo Only (different files/history)? ──► [RISKY (AUDIT / PURGE REPO)]
           ▼ No
[Stage 3: Token N-Gram & MinHash Near-Duplicate Analysis]
 ├── Extract token 13-grams for prompt, old_contents, new_contents
 └── Compute MinHash Jaccard similarity (128 hash permutations)
           │
           ├── Jaccard Similarity ≥ 0.80 ──► [CONTAMINATED (EXCLUDE)]
           ├── 0.50 ≤ Jaccard Similarity < 0.80 ──► [RISKY (REQUIRES DEDUP FILTERING)]
           └── Jaccard Similarity < 0.50 ──► [CLEAN (APPROVED FOR SFT)]
```

### 3.1 Stage 1: Exact String & Cryptographic Hash Matching
* **Signature Database to Build**:
  1. `HE_SIGNATURES`:
     - 164 entry point function names (`has_close_elements`, `separate_paren_groups`, ...).
     - Normalized docstring SHA-256 hashes (stripping whitespace, lowercasing, normalizing quotes).
     - Verbatim test assertions (e.g., `assert has_close_elements([1.0, 2.0, 3.9, ...])`).
  2. `CANIT_SIGNATURES`:
     - 105 problem names/full names.
     - SHA-256 hashes of normalized `before` and `after` code blocks.
     - SHA-256 hashes of `instruction_descriptive` and `instruction_lazy`.
     - Verbatim test assertions from `tests`.
  3. `CCEVAL_SIGNATURES`:
     - SHA-256 hashes of 2,665 Python ground-truth completion targets.
* **Execution**:
  - Compute SHA-256 on candidate sample fields: `instruction`, `old_contents`, `new_contents`.
  - Perform substring search for the 164 HumanEval function declarations (`def {entry_point}(`).

### 3.2 Stage 2: Repository and Commit Overlap
* **Repository Matching**:
  - Extract candidate metadata fields: `repo_name`, `repository`, `repo_url`, `commit_id`, `commit_hash`, `file_name`, `file_path`.
  - Compare candidate repository identifiers against:
    - 471 CrossCodeEval Python repositories (`amazon-science/cceval`).
    - NUPRL benchmark repositories (`nuprl/CanItEdit`, `nuprl/EditPackFT`).
    - OpenAI / BigCode evaluation repositories (`openai/human-eval`, `bigcode-project/octopack`).
* **Commit & File Path Matching**:
  - Check if candidate commit hash matches any pinned commit SHA in CrossCodeEval.
  - If repo matches: verify if the edited file matches any of the 1,368 evaluated files in CrossCodeEval.

### 3.3 Stage 3: Near-Duplicate / MinHash / N-Gram Overlap
* **Token Shingling (Standard BigCode / StarCoder Decontamination)**:
  - Tokenize Python code using standard Python AST tokenizer (stripping comments and docstrings for code-to-code comparison).
  - Construct token 13-grams ($N=13$).
  - A candidate sample is flagged if it shares an exact token 13-gram substring of length $\ge 50$ characters with any benchmark reference code.
* **MinHash LSH Index**:
  - Use 128 hash permutations (`datasketch.MinHash`).
  - Index all benchmark prompts, reference solutions, and test cases.
  - Query each candidate record against the MinHash index.

---

### 3.4 Classification Rules & Decision Matrix

| Category | Definition & Criteria | Action & Policy |
|---|---|---|
| **CONTAMINATED** | • Exact match with any of the 164 HumanEval entry points, docstrings, or unit tests.<br>• Exact match with any of the 105 CanItEdit `before`/`after` hashes, instructions, or tests.<br>• Exact match with CrossCodeEval evaluated file in a matching repository, OR matching pinned commit hash.<br>• Token 13-gram Jaccard similarity $\ge 0.80$ against any benchmark solution or prompt. | **MUST BE EXCLUDED**.<br>If contaminated sample rate $> 0.5\%$ or dataset derives from benchmark seed problems, **REJECT ENTIRE DATASET**. If isolated ($\le 0.5\%$), execute strict purge before training. |
| **RISKY** | • Candidate originates from one of CrossCodeEval's 471 Python repos, but commit is older/different and file does not match evaluated files.<br>• MinHash Jaccard similarity between $0.50$ and $0.80$ against benchmark items.<br>• Synthetic dataset generated from unspecified coding seeds (unverifiable prompt heritage).<br>• Dataset lacks repository, commit, or file path provenance metadata. | **QUARANTINE / CONDITIONAL AUDIT**.<br>Cannot be recommended for primary training without automated deduplication filtering. Unverifiable metadata candidates cannot be used for final benchmarks. |
| **CLEAN** | • 0% repository overlap with CrossCodeEval 471 repos, OpenAI HumanEval, and CanItEdit.<br>• 0% commit hash overlap.<br>• Max token 13-gram Jaccard similarity $< 0.50$ across all benchmark items.<br>• Verifiable permissive license (MIT, Apache-2.0, BSD).<br>• Transparent provenance and fully compliant with BifocalGemma data contract. | **APPROVED FOR SFT**.<br>Eligible for ranking, token profiling, and data mixing. |

---

## 4. Caveats

1. **Pretraining Contamination vs SFT Contamination**:
   - This protocol decontaminates the **SFT fine-tuning data** for BifocalGemma.
   - It cannot retroactively remove data seen by base models (e.g., Gemma 2) during their original pretraining. However, rigorous SFT decontamination ensures that no post-pretraining gains on HumanEvalFix, CanItEdit, or CrossCodeEval are artifacts of fine-tuning memorization.
2. **Synthetic Data Lineage**:
   - Many synthetic datasets (e.g., Code Alpaca, Evol-Instruct) used HumanEval docstrings as seed prompts. If a dataset does not provide seed lineage, Stage 1 (exact string matching) and Stage 3 (MinHash) must be relied upon to catch leaked descendants.
3. **CrossCodeEval Historical Commits**:
   - Commits made to CrossCodeEval repositories *years prior* to the benchmark snapshot may contain related utility functions. While not strictly evaluated in the benchmark, classifying them as "Risky" prevents any indirect repository memorization.

---

## 5. Conclusion

1. **HumanEvalFix** is a 164-problem functional repair benchmark derived 1-to-1 from OpenAI HumanEval, with subtle human-injected bugs across 6 taxonomies. Its signature is unambiguous: 164 specific function identifiers, OpenAI docstrings, and `check(entry_point)` test cases.
2. **CanItEdit** is a 105-problem hand-crafted Python editing benchmark with full-file context (`before` / `after`) and dual instruction modes (`instruction_descriptive`, `instruction_lazy`). It was authored by 8 programmers and is independent of general GitHub commit crawls.
3. **CrossCodeEval** is a 471-repository, 1,368-file, 2,665-sample repository-level completion diagnostic (under Apache-2.0, NeurIPS 2023) focused on cross-file dependencies.
4. **Contamination Protocol**: The three-stage verification pipeline (Exact Hash & String $\rightarrow$ Repo/Commit Intersection $\rightarrow$ 13-gram / MinHash LSH) provides a deterministic, automated mechanism to classify candidates as **Contaminated** (must exclude), **Risky** (requires purge/audit), or **Clean** (approved for SFT).

---

## 6. Verification Method

To independently verify the benchmark findings and execute the contamination protocol:

1. **Inspect HumanEvalPack Metadata**:
   ```python
   from datasets import load_dataset
   ds = load_dataset("bigcode/humanevalpack", "python", split="test")
   print(len(ds))  # Expect: 164
   print(ds[0]["task_id"], ds[0]["entry_point"], ds[0]["bug_type"])
   # Expect: Python/0, has_close_elements, missing logic
   ```
2. **Inspect CanItEdit Metadata**:
   ```python
   from datasets import load_dataset
   ds = load_dataset("nuprl/CanItEdit", split="test")
   print(len(ds))  # Expect: 105
   print(ds[0].keys())
   # Expect: ['id', 'name', 'full_name', 'before', 'after', 'tests', 'instruction_descriptive', 'instruction_lazy', 'taxonomy']
   ```
3. **Inspect CrossCodeEval Repository List**:
   - Clone or inspect `amazon-science/cceval`:
   ```bash
   git clone --depth 1 https://github.com/amazon-science/cceval.git
   tar -tvf cceval/data/crosscodeeval_data.tar.xz | grep python
   ```
   - Verify 471 unique repositories in Python completion samples.
4. **Protocol Verification Test**:
   - Run candidate dataset audit script taking `candidate.jsonl` and outputting classification report:
   ```bash
   python scripts/audit_contamination.py --candidate <path_to_jsonl> --benchmarks he_fix,canitedit,cceval
   ```
   - Invalidation Condition: Any candidate labeled "Clean" that produces a 13-gram match $\ge 0.80$ or shares a repository name with CrossCodeEval's 471 repos immediately fails verification.
