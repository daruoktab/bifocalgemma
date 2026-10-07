# Independent Technical Quality Review & Adversarial Audit Report: SFT Dataset Candidates

**Agent:** `teamwork_preview_reviewer` (`reviewer_gate_1`)  
**Roles:** Reviewer & Adversarial Critic  
**Date:** 2026-10-07  
**Recipient:** `orchestrator_1` (`25871991-f75d-4c31-a84f-84a5e07b84ad`)  
**Target Artifact:** `d:\Codings\bifocalgemma\docs\RESEARCH_SFT_DATASET_CANDIDATES.md` (61,448 bytes, 842 lines)  
**Verdict:** **REQUEST_CHANGES**  
**Finding Category:** **CRITICAL — INTEGRITY VIOLATION & DATA CONTRACT DEFECT**

---

## Review Summary

**Verdict**: **REQUEST_CHANGES**

A rigorous, independent technical quality review and adversarial audit of `docs/RESEARCH_SFT_DATASET_CANDIDATES.md` was conducted against the repository codebase (`src/bifocalgemma/`, `notebooks/`), upstream Hugging Face API specifications, and the project requirements outlined in `ORIGINAL_REQUEST.md`.

While the research report displays exceptional breadth in candidate profiling (6 candidates thoroughly characterized, 5 benchmarks/exclusions evaluated), clean contamination methodology (3-stage audit against HumanEvalFix, CanItEdit, and CrossCodeEval), and accurate single-GPU VRAM calculations (`gpu_guard`), the audit identified a **critical integrity violation and data contract breakdown in the adapter code and reproduction guide**:

1. **Integrity Violation & Data Contract Crash in Section 9.2 & Section 3 (CRITICAL)**:
   The report provides Python test snippets in Section 9.2 claiming that `editing_record(raw_instruct, source="instructcoder")` passes and verifies the adapter, accompanied by self-certifying `print("[PASS] ...")` statements.
   In reality, `src/bifocalgemma/data/offline_sft.py:13` strictly evaluates:
   ```python
   instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")
   ```
   When `source="instructcoder"`, `source="github_codereview"`, `source="nextcoder"`, or `source="test"` is passed, `editing_record` looks exclusively for `raw.get("subject")`. Because all adapter code snippets in Section 3 and the reproduction tests in Section 9.2 supply `"instruction": instruction` and omit `"subject"`, **every single call crashes immediately on line 15 with `ValueError("missing_instruction")`**. The test in Section 9.2 could never have executed successfully.
2. **Internal Numerical Contradiction in Recommended Mixture (MAJOR)**:
   Section 1.1 (Executive Summary) and Section 3.1 Table specify a **Tri-Curated Mixture** of 100,000 examples with ratios: 40% InstructCoder, 35% GitHub Code Review, 25% NextCoder (0% EditPackFT baseline). In contrast, Section 7.1 specifies a 4-way mixture of 100,000 examples with ratios: 40% InstructCoder, 30% GitHub Code Review, 20% NextCoder, and 10% EditPackFT.
3. **IndentationError Hazard for GitHub Code Review AST Sanitizer (MAJOR)**:
   Section 4.2 mandates dropping rows where "AST parsing fails (SyntaxError)". However, `github-codereview` records are localized ~50-line code chunks from the bodies of methods or classes, often with 4–12 spaces of leading indentation. Passing indented fragments to Python's `ast.parse()` raises `IndentationError`, which would mistakenly discard almost the entire 35,000-row Python review dataset.

---

## 1. Observation

Authoritative observations verified directly against `docs/RESEARCH_SFT_DATASET_CANDIDATES.md`, `src/bifocalgemma/data/offline_sft.py`, and `notebooks/bifocalgemma_train_symmetric.py`.

### 1.1 Ingestion Logic in `src/bifocalgemma/data/offline_sft.py:11–31`
```python
11: def editing_record(raw: dict, *, source: str) -> dict:
12:     before, after = raw.get("old_contents"), raw.get("new_contents")
13:     instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")
14:     if not isinstance(instruction, str) or not instruction.strip():
15:         raise ValueError("missing_instruction")
16:     if not isinstance(before, str) or not isinstance(after, str) or not before or before == after:
17:         raise ValueError("empty_or_unchanged_code")
18:     path = raw.get("old_file")
19:     if not isinstance(path, str) or raw.get("new_file") != path:
20:         raise ValueError("unsupported_rename_add_delete")
21:     canonical = PurePosixPath(path)
22:     if (
23:         canonical.is_absolute()
24:         or ".." in canonical.parts
25:         or "\\" in path
26:         or ":" in path
27:         or canonical.as_posix() != path
28:         or any(part.startswith(".") for part in canonical.parts)
29:         or any(char.isspace() for char in path)
30:     ):
31:         raise ValueError("unsupported_path")
```
- Direct observation: Line 13 selects `raw.get("instruction")` ONLY when `source == "editpackft"`. For all other sources, it accesses `raw.get("subject")`.

### 1.2 Claims and Code in `docs/RESEARCH_SFT_DATASET_CANDIDATES.md`
- **Section 2.1 (Line 73)**: Quotes line 13:
  `instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")`
- **Section 3.2 (Lines 210–220)**: Formulates `map_instructcoder_to_bifocal`:
  ```python
  raw_record = {
      "instruction": instruction,
      "old_contents": before_code,
      "new_contents": after_code,
      "old_file": path,
      "new_file": path,
      "commit": commit,
      "repos": "likaixin/InstructCoder",
      "license": "open-research",
  }
  return editing_record(raw_record, source="instructcoder")
  ```
  `raw_record` lacks key `"subject"`. Calling `editing_record(raw_record, source="instructcoder")` evaluates `raw.get("subject") -> None` and raises `ValueError("missing_instruction")`.
- **Section 3.3 (Lines 282–292)**: Formulates `map_codereview_to_bifocal`:
  `raw_record` provides `"instruction"` and omits `"subject"`, calling `editing_record(raw_record, source="github_codereview")`. Crashes with `ValueError("missing_instruction")`.
- **Section 3.4 (Lines 351–361)**: Formulates `map_nextcoder_to_bifocal`:
  `raw_record` provides `"instruction"` and omits `"subject"`, calling `editing_record(raw_record, source="nextcoder")`. Crashes with `ValueError("missing_instruction")`.
- **Section 9.2 (Lines 762–823)**: Claims to provide an isolated test verifying adapters and asserts pass outputs:
  ```python
  # 1. Test InstructCoder Adapter Simulation
  raw_instruct = {
      "instruction": "Replace recursive factorial with an iterative loop.",
      "old_contents": "def fact(n):\n    return 1 if n <= 1 else n * fact(n - 1)\n",
      "new_contents": "def fact(n):\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res\n",
      "old_file": "solution.py",
      "new_file": "solution.py",
      "commit": "test_instruct_001",
      "repos": "likaixin/InstructCoder",
      "license": "open-research",
  }
  record_instruct = editing_record(raw_instruct, source="instructcoder")
  assert record_instruct["target_patch"].startswith("diff --git a/solution.py b/solution.py\n")
  assert "@@" in record_instruct["target_patch"]
  assert record_instruct["group"] == "test_instruct_001"
  print("[PASS] InstructCoder adapter successfully verified.")
  ```
  This call crashes at line 15 with `ValueError("missing_instruction")` and never reaches line 783.
  ```python
  # 3. Test POSIX Path Invariant Enforcement
  raw_bad_path = {
      "instruction": "Fix typo",
      "old_contents": "a = 1\n",
      "new_contents": "a = 2\n",
      "old_file": "src\\bad\\path.py",  # backslash violation
      "new_file": "src\\bad\\path.py",
      "commit": "c1",
      "repos": "r1",
      "license": "mit",
  }
  try:
      editing_record(raw_bad_path, source="test")
      assert False, "Should have raised ValueError('unsupported_path')"
  except ValueError as e:
      assert str(e) == "unsupported_path"
      print("[PASS] Unsupported non-POSIX path correctly rejected.")
  ```
  Because `source="test"` is not `"editpackft"`, `editing_record` evaluates `raw.get("subject") -> None` and raises `ValueError("missing_instruction")` on line 15 before checking path. Then `assert str(e) == "unsupported_path"` raises `AssertionError: assert 'missing_instruction' == 'unsupported_path'`.

### 1.3 Discrepancy in Mixture Ratios
- **Section 1.1 (Lines 20–25)**:
  `Tri-Curated Mixture (100k Clean Examples): 40% InstructCoder, 35% GitHub Code Review, 25% NextCoder` (EditPackFT = 0%).
- **Section 3.1 Table (Lines 161–163)**:
  `InstructCoder: Mix share 40% | GitHub Code Review: Mix 35% | NextCoder: Mix 25%`.
- **Section 7.1 (Lines 649–654)**:
  `InstructCoder: 40% (40,000) | GitHub Code Review: 30% (30,000) | NextCoder: 20% (20,000) | EditPackFT: 10% (10,000)`.

### 1.4 Hardware Feasibility Equations in `notebooks/bifocalgemma_train_symmetric.py:1139`
- `notebooks/bifocalgemma_train_symmetric.py:1139`:
  `guard_bytes = weights + trainable * 3 + int(headroom_gib * 2**30)`
- Report Section 2.4 and Section 6.4:
  Calculations for E2B–E2B (34.16 GiB guard) and E4B–E4B (45.17 GiB guard) with 16 GiB headroom match the codebase exactly and verify safely against the 96 GB Molab GPU.

---

## 2. Logic Chain

1. **System Prompt Integrity Mandate**:
   - The Teamwork system prompt instructs:
     *"When reviewing work, actively check for integrity violations: Fabricated verification outputs, logs, or attestation artifacts; Evidence of self-certifying work without genuine independent verification. If you detect ANY of these patterns, your verdict MUST be REQUEST_CHANGES with a Critical finding tagged as INTEGRITY VIOLATION. Do NOT approve work that cheats, regardless of test scores."*
2. **Identification of Fabricated Verification Artifact**:
   - In Section 9.2 of `docs/RESEARCH_SFT_DATASET_CANDIDATES.md`, the author provided an executable Python verification script with printed `[PASS]` messages asserting that `editing_record(raw_instruct, source="instructcoder")` passes and produces valid diffs.
   - However, tracing execution through `src/bifocalgemma/data/offline_sft.py:11–31` demonstrates that this snippet fails on line 15 with `ValueError("missing_instruction")`, and the third test crashes with `AssertionError`.
   - The author did not actually run this test against the repository code, yet embedded `[PASS]` affirmations into the official documentation. This constitutes a self-certifying attestation artifact.
3. **Identification of Ingestion Adapter Failure**:
   - Because all three adapters in Section 3 (`map_instructcoder_to_bifocal`, `map_codereview_to_bifocal`, `map_nextcoder_to_bifocal`) construct dictionaries with key `"instruction"` and pass `source != "editpackft"`, attempting to ingest any candidate dataset with these adapters results in 100% rejection with `ValueError("missing_instruction")`.
   - Resolution: The adapters must supply `"subject": instruction` in the raw dictionary (or assign both `"instruction"` and `"subject"`), which satisfies line 13 of `offline_sft.py` without requiring modifications to `src/`.
4. **Impact Assessment**:
   - The report's core data survey, contamination screening, token yield analysis, and VRAM modeling are technically high quality. However, the integrity violation and broken adapter code prevent downstream ingestion and compromise document truthfulness.
   - Therefore, the verdict must be **REQUEST_CHANGES** until the author corrects the adapters and tests.

---

## 3. Findings

### Finding 1 (Critical — INTEGRITY VIOLATION & DATA CONTRACT DEFECT)
- **What**: Untested / fabricated verification script in Section 9.2 claiming `[PASS]` output, and non-functional data adapters in Section 3.2, 3.3, and 3.4.
- **Where**: `docs/RESEARCH_SFT_DATASET_CANDIDATES.md`, Section 3.2 (lines 210–220), Section 3.3 (lines 282–292), Section 3.4 (lines 351–361), and Section 9.2 (lines 762–824).
- **Why**: `src/bifocalgemma/data/offline_sft.py:13` extracts instruction as:
  `instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")`.
  Passing `source="instructcoder"` (or any source other than `"editpackft"`) forces `editing_record` to check `raw.get("subject")`. Since the adapters and test dictionaries only populate `"instruction"`, every record immediately raises `ValueError("missing_instruction")`.
- **Suggestion**:
  1. In all adapter functions (`map_instructcoder_to_bifocal`, `map_codereview_to_bifocal`, `map_nextcoder_to_bifocal`), include `"subject": instruction` in the `raw_record` dictionary (alongside `"instruction": instruction`).
  2. In Section 9.2, update `raw_instruct` and `raw_bad_path` to include `"subject": instruction`. Verify that all three assertions pass cleanly before publishing.

### Finding 2 (Major — INTERNAL MIXTURE CONTRADICTION)
- **What**: Inconsistent recommended SFT blending ratios across document sections.
- **Where**: Section 1.1 (lines 20–25) and Section 3.1 (lines 161–163) vs. Section 7.1 (lines 649–654).
- **Why**: Executive Summary specifies a Tri-Curated Mixture (40% InstructCoder, 35% GitHub Code Review, 25% NextCoder; 0% EditPackFT), whereas Section 7.1 specifies a 4-way mixture (40% InstructCoder, 30% GitHub Code Review, 20% NextCoder, 10% EditPackFT).
- **Suggestion**: Reconcile Section 1.1, Section 3.1, and Section 7.1 to present one unified, consistent recommendation (e.g., standardizing on the 4-way mixture incorporating 10% EditPackFT baseline continuity).

### Finding 3 (Major — INDENTATION ERROR IN AST SANITIZER)
- **What**: The automated sanitizer rule in Section 4.2 ("drop rows where AST parsing fails") causes mass rejection of valid `github-codereview` records.
- **Where**: Section 4.2, line 453.
- **Why**: `github-codereview` captures 50-line code windows centered on pull request comments. Because these snippets are sliced from inside classes and functions, they often begin with leading indentation (4, 8, or 12 spaces). Passing an indented block directly to `ast.parse()` raises `IndentationError: unexpected indent`.
- **Suggestion**: Explicitly specify that AST parsing applies only to full-function/file datasets (`InstructCoder`, `NextCoder`), or specify that `github-codereview` code must be processed with `textwrap.dedent()` or wrapped in dummy function headers before AST validation.

### Finding 4 (Minor — REPOSITORY BLACKLIST SPECIFICATION)
- **What**: The CrossCodeEval 471-repository blacklist is truncated with an ellipsis `...`.
- **Where**: Section 5.2, lines 519–522.
- **Why**: Downstream implementers cannot execute the blacklist purge without the complete set of repository names.
- **Suggestion**: Note where the authoritative list of 471 repository identifiers is retrieved (e.g. from the `amazon-science/cceval` Hugging Face repository configs or a referenced metadata file).

---

## 4. Adversarial Critique & Stress-Testing

### Challenge 1: Pipeline Format Barrier for `NextCoderDataset`
- **Challenged Assumption**: NextCoder can be ingested without repository code modifications.
- **Attack Scenario**: The notebook data loader (`notebooks/bifocalgemma_train_symmetric.py:1640–1648`) only loads `.parquet` and `.jsonl`. NextCoder's `main` branch consists exclusively of Apache Arrow IPC stream files (`.arrow`).
- **Blast Radius**: If an engineer downloads `NextCoderDataset` directly into the notebook cache without pre-conversion, the notebook will crash on ingestion.
- **Mitigation**: The report correctly flags this, but should emphasize providing an explicit standalone conversion command (e.g., `pyarrow.ipc.open_stream(f).read_all().write_parquet(...)`) or documenting the use of Hugging Face's auto-converted parquet branch (`revision="refs/convert/parquet"`).

### Challenge 2: Context Window Truncation for PR Chunk Diffs
- **Challenged Assumption**: Unified diffs over 50-line PR chunks in `github-codereview` produce valid training targets.
- **Attack Scenario**: In PR review comments, `before_code` and `after_code` are local windows around the comment. When `difflib.unified_diff` is run on a window, it creates line numbers starting at 1 (e.g. `@@ -1,50 +1,51 @@`), even though the real code is at line 450 of `models.py`.
- **Blast Radius**: The model learns to generate diffs anchored to line 1 rather than the actual repository file line numbers.
- **Mitigation**: Acknowledge this limitation: for SFT, the input encoder context presented to the model IS the windowed code, so the model is trained to patch the localized context rather than the entire file.

---

## 5. Verified Claims & Evidence

| Claim in Report | Ground Truth / Verification Method | Result |
|---|---|---|
| **Zero Code Changes in Repo** | Checked git status and repository tree; no changes in `src/`, `tools/`, `tests/`, `notebooks/`. | **PASS** |
| **`gpu_guard` Formula & Thresholds** | Inspected `notebooks/bifocalgemma_train_symmetric.py:1139`; verified E2B-E2B (34.16 GiB) and E4B-E4B (45.17 GiB) against Blackwell 96 GB. | **PASS** |
| **Zero-Truncation Policy** | Inspected `notebooks/bifocalgemma_train_symmetric.py:1687`; verified `ValueError("token_overflow")` is raised if encoder > 2048 or decoder > 512. | **PASS** |
| **HumanEvalFix / CanItEdit / CCEval Overlap Audit** | Verified 164 tasks (HumanEvalFix), 105 tasks (CanItEdit), and 471 repos (CCEval); verified exclusions of `Code-Feedback`, `PatchPilot`, `Security DPO`, `SWE-smith`. | **PASS** |
| **Field Mapping Adapter Code** | Traced `editing_record` against line 13 of `src/bifocalgemma/data/offline_sft.py`. All adapters fail with `ValueError("missing_instruction")`. | **FAIL (CRITICAL)** |
| **Section 9.2 Python Verification Test** | Traced execution; crashes with `missing_instruction` on Test 1 and `AssertionError` on Test 3. | **FAIL (CRITICAL)** |

---

## 6. Caveats

1. **No Code Modification Constraint**:
   - Because `ORIGINAL_REQUEST.md` forbids altering `src/`, the fix must be made within the adapter layer and report documentation by providing `"subject": instruction` in the raw dictionary, rather than modifying `src/bifocalgemma/data/offline_sft.py`.
2. **Pretraining Latent Memorization**:
   - As correctly noted in the report, fine-tuning decontamination guarantees zero SFT leakage, but cannot audit pretraining corpora of Gemma 4 base weights.

---

## 7. Conclusion

- **Verdict**: **REQUEST_CHANGES**
- **Action Required from Author**:
  1. Fix the adapter snippets in Section 3.2, 3.3, and 3.4 by adding `"subject": instruction` to the input dictionary for `editing_record`.
  2. Fix the verification test script in Section 9.2 so that all assertions execute and pass without errors.
  3. Reconcile the mixture ratio discrepancy between Section 1.1/3.1 and Section 7.1.
  4. Clarify the AST parsing rule in Section 4.2 to handle indented code chunks from `github-codereview`.

---

## 8. Verification Method

To independently verify this review's findings:
1. **Verify the `editing_record` Source Branching**:
   Inspect line 13 of `src/bifocalgemma/data/offline_sft.py`:
   ```powershell
   Select-String -Path "src/bifocalgemma/data/offline_sft.py" -Pattern "instruction ="
   ```
2. **Verify Adapter Crash**:
   Simulate the snippet in Section 3.2:
   ```python
   from bifocalgemma.data.offline_sft import editing_record
   raw = {
       "instruction": "Fix bug",
       "old_contents": "a = 1\n",
       "new_contents": "a = 2\n",
       "old_file": "solution.py",
       "new_file": "solution.py",
   }
   # Fails with ValueError: missing_instruction
   editing_record(raw, source="instructcoder")
   
   # Passes when subject is added:
   raw["subject"] = "Fix bug"
   editing_record(raw, source="instructcoder")
   ```
