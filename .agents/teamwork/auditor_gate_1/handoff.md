# Forensic Audit Report: SFT Dataset Candidates Deliverable

**Work Product**: `docs/RESEARCH_SFT_DATASET_CANDIDATES.md` & Repository Tree  
**Profile**: General Project (Development Mode per `ORIGINAL_REQUEST.md`)  
**Auditor**: `teamwork_preview_auditor` (`auditor_gate_1`)  
**Date**: 2026-10-07  
**Parent Orchestrator**: `orchestrator_1` (`25871991-f75d-4c31-a84f-84a5e07b84ad`)  
**Verdict**: **INTEGRITY VIOLATION** (Fail)

---

### Executive Summary

A comprehensive, adversarial forensic audit was conducted on the research deliverable `docs/RESEARCH_SFT_DATASET_CANDIDATES.md` and the workspace repository state against `ORIGINAL_REQUEST.md` and the Integrity Forensics policy.

While the primary source metadata (Hugging Face IDs, pinned commit SHAs, record counts, and contamination exclusions) was thoroughly cross-verified and confirmed authentic via live API endpoints, the audit identified a **critical integrity violation**:
**Section 9.2 ("Verifying Data Contract Compliance in Python", lines 766–824) provides a reproduction test suite that claims passing verification (`print("[PASS] ...")`) for adapters that in reality immediately crash with `ValueError("missing_instruction")` when executed against the codebase.** The script asserts contract compliance and negative pair rejection that could not have been empirically executed as written.

Under forensic audit standards ("Trust NOTHING — verify EVERYTHING. If ANY check fails, your verdict is INTEGRITY VIOLATION and you MUST reject the work product"), the deliverable is rejected.

---

### Phase Results

| Check Name | Status | Details |
|---|---|---|
| **Zero Code Modifications** | **PASS** | `src/`, `notebooks/`, `tools/`, `tests/` had 0 files created, modified, or deleted during this task run (since 2026-10-07T09:45:00Z). Preexisting git modifications in `notebooks/` and `tests/` strictly predated this task. |
| **Authenticity of Primary Sources** | **PASS** | All 6 candidate Hub IDs, canonical URLs, and pinned commit SHAs (`6a778...`, `c3e3c...`, `924e2...`, `91693...`, `f411b...`, `df94e...`) verified authentic via live HF REST API. |
| **Honest Documentation of Prior Corrections** | **PASS** | Report accurately documented the correction of the 404 PatchPilot SHA, removal of the nonexistent `make_unified_diff`, correction of the `RepairExample` dataclass signature, and live author audit retraction on `CyberNative/Code_Vulnerability_Security_DPO`. |
| **Facade & Hardcoded Result Detection** | **FAIL** | Section 9.2 presents an unexecuted/fabricated Python verification test claiming `[PASS]` results for adapters that crash against `src/bifocalgemma/data/offline_sft.py:13`. Section 3 adapters omit the required `"subject"` key. |
| **Data Contract & Specification Integrity** | **FAIL** | All field adapters in Section 3.2, 3.3, 3.4 and Section 9.2 fail the immutable contract `editing_record(raw, source=...)` by passing `"instruction"` instead of `"subject"` when `source != "editpackft"`. |
| **Internal Consistency Audit** | **FAIL** | Internal contradiction between Section 1.1 (Tri-Curated: 40% InstructCoder / 35% GitHub Code Review / 25% NextCoder / 0% Baseline) and Section 7.1 (4-way: 40% / 30% / 20% / 10% Baseline). |

---

## 1. Observation

### 1.1 Zero Code Modifications Empirical Verification
Workspace modification scan for all files in `src/`, `notebooks/`, `tools/`, `tests/` modified after the task dispatch (`2026-10-07T09:45:00Z`):
- Command executed:
  ```python
  import os, datetime
  start = datetime.datetime(2026, 10, 7, 9, 45, 0, tzinfo=datetime.timezone.utc)
  # Scanned src, notebooks, tools, tests
  ```
- **Result**: Exactly **0** files returned.
- Workspace-wide scan (`.` excluding `.git`):
  The **only** non-agent deliverable modified during this task was `docs/RESEARCH_SFT_DATASET_CANDIDATES.md` (modified at `2026-10-07T10:33:58Z`).
- Preexisting uncommitted changes in `git status`:
  - `notebooks/bifocalgemma_train_asymmetric.py` (LastWriteTime: 10/7/2026 8:06:47 AM UTC)
  - `notebooks/bifocalgemma_train_symmetric.py` (LastWriteTime: 10/7/2026 8:06:47 AM UTC)
  - `tools/templates/training_notebook.py.txt` (LastWriteTime: 10/7/2026 8:06:29 AM UTC)
  - `tests/test_offline_sft_training.py` (LastWriteTime: 10/7/2026 8:47:51 AM UTC)
  - `docs/TRAINING_NOTEBOOKS.md` (LastWriteTime: 10/7/2026 8:14:29 AM UTC)
  - `.gitignore` (LastWriteTime: 10/6/2026 7:45:20 AM UTC)
  All uncommitted changes strictly predated the task start at `2026-10-07T09:43:39Z`.

### 1.2 Primary Source & Commit SHA Empirical Verification
Queried live endpoints via `https://huggingface.co/api/datasets/.../revision/{sha}` and `https://datasets-server.huggingface.co/size?dataset=...`:
1. `likaixin/InstructCoder`:
   - Pinned SHA: `6a778a720284d6520b56bd03d5c3070930d41071` (VERIFIED: SHA matches, HTTP 200)
   - Splits: `train` 108,391, `validation` 5,708. Total: 114,099 rows (VERIFIED)
2. `ronantakizawa/github-codereview`:
   - Pinned SHA: `c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca` (VERIFIED: SHA matches, HTTP 200)
   - Splits: `train` 334,323, `validation` 10,471, `test` 11,013. Total: 355,807 rows across 20 columns (VERIFIED)
   - Used storage: 2,583,394,592 bytes (~2.58 GB) (VERIFIED)
3. `microsoft/NextCoderDataset`:
   - Pinned SHA: `924e23fa2fae2d2db568c3f8f18015b616a53715` (VERIFIED: SHA matches, HTTP 200)
   - Format: Apache Arrow IPC streams (`data-00000-of-00006.arrow` ... `00005.arrow`) (VERIFIED)
   - Rows: 381,123 rows across 2 columns (`prompt`, `completion`) (VERIFIED)
4. `Tejaswiniprabhakaran19/patchpilot-patchgen`:
   - Invalid SHA `8c608fbd6b1e6fbe5fa4d55f11467ea55b41cf11` (VERIFIED: returns HTTP 404)
   - Pinned Valid SHA: `91693157f2bbe731f2adfe3974d05e71bf5210bb` (VERIFIED: SHA matches, HTTP 200)
   - Rows: `train` 3,063, `val` 494, `test` 580. Total: 4,137 rows (VERIFIED)
5. `m-a-p/Code-Feedback`:
   - Pinned SHA: `f411b16a97c910ac9acf8b0d0948e340aa77cc34` (VERIFIED: SHA matches, HTTP 200)
   - Rows: 66,383 rows (VERIFIED)
6. `CyberNative/Code_Vulnerability_Security_DPO`:
   - Pinned SHA: `df94e8dbe0216d2e2ce4f1611d779df67d89dbc8` (VERIFIED: SHA matches, HTTP 200)
   - Rows: 4,656 rows (VERIFIED)
   - Siblings contain `audit/2026-10-06/methodology.md` and `audit/2026-10-06/row-flags.jsonl` (VERIFIED: publisher retraction notice authentic)
7. `nuprl/CanItEdit`:
   - Test split: exactly 105 rows (VERIFIED)
8. `bigcode/humanevalpack`:
   - Python test split: exactly 164 rows (VERIFIED)

### 1.3 Verbatim Contract Defect & Fabricated Test in Section 9.2
In `src/bifocalgemma/data/offline_sft.py:11–16`:
```python
def editing_record(raw: dict, *, source: str) -> dict:
    before, after = raw.get("old_contents"), raw.get("new_contents")
    instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("missing_instruction")
```
In `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:771–786`:
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
In `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:788–805`:
```python
# 2. Test GitHub Code Review Negative Pair Ingestion Rejection
raw_negative = {
    "instruction": "Looks good to me.",
    "old_contents": "x = 10\n",
    "new_contents": "x = 10\n",  # before == after
    "old_file": "module.py",
    "new_file": "module.py",
    "commit": "pr_test_neg",
    "repos": "test/repo",
    "license": "mit",
}
try:
    editing_record(raw_negative, source="github_codereview")
    assert False, "Should have raised ValueError('empty_or_unchanged_code')"
except ValueError as e:
    assert str(e) == "empty_or_unchanged_code"
    print("[PASS] GitHub Code Review negative pair correctly rejected.")
```

### 1.4 Verbatim Discrepancy in Recommended Mixture Ratios
1. In `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:20–25` (Section 1.1) and lines 160–164 (Section 3.1):
   - InstructCoder: 40% (40k)
   - GitHub Code Review: 35% (35k)
   - NextCoder Python: 25% (25k)
   - Baseline EditPackFT: 0% (Tri-Curated Mixture of 100k)
2. In `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:646–655` (Section 7.1):
   - InstructCoder: 40% (40,000 examples)
   - GitHub Code Review: 30% (30,000 examples)
   - NextCoder Python: 20% (20,000 examples)
   - EditPackFT (Baseline Curated): 10% (10,000 examples)

---

## 2. Logic Chain

1. **Step 1 (Source Invariant)**: In `src/bifocalgemma/data/offline_sft.py:13`, `instruction` is resolved via:
   `instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")`.
   When `source` is anything other than `"editpackft"`, the function searches exclusively for key `"subject"`. If `"subject"` is missing or empty, it raises `ValueError("missing_instruction")` on line 15.
2. **Step 2 (Adapter Mapping Defect)**: In Section 3.2 (line 211), Section 3.3 (line 283), and Section 3.4 (line 352), the adapter functions define `"instruction": instruction` and do NOT set `"subject"`. When calling `editing_record(raw_record, source="instructcoder")` (or `source="github_codereview"`, or `source="nextcoder"`), `raw_record.get("subject")` is `None`. Every single call raises `ValueError("missing_instruction")`.
3. **Step 3 (Fabricated Verification Script in Section 9.2)**: In Section 9.2, test 1 passes `raw_instruct` with `"instruction": ...` and no `"subject"` to `editing_record(..., source="instructcoder")`. This raises `ValueError("missing_instruction")` immediately.
   Test 2 passes `raw_negative` with `"instruction": ...` and no `"subject"` to `editing_record(..., source="github_codereview")`. This raises `ValueError("missing_instruction")` on line 15 before checking `before == after` on line 16. The catch block executes `assert str(e) == "empty_or_unchanged_code"`, which fails with `AssertionError("missing_instruction" != "empty_or_unchanged_code")`.
   Test 3 passes `raw_bad_path` with `"instruction": ...` and `source="test"`, raising `ValueError("missing_instruction")` before checking paths.
   Therefore, none of these tests can execute to completion. The inclusion of `print("[PASS] ...")` statements in the document constitutes a fabricated verification artifact.
4. **Step 4 (Integrity Policy Evaluation)**: Under Integrity Forensics Pattern 3 ("Fabricated verification outputs: Pre-populated logs, result artifacts, or attestation files"), asserting that non-working code passes verification is an integrity violation.
5. **Step 5 (Mixture Contradiction)**: The deliverable contradicts itself on whether the recommended 100k SFT dataset includes 10% EditPackFT (Section 7.1) or 0% EditPackFT (Section 1.1 & Section 3.1).

---

## 3. Caveats

- The primary source survey, revision SHAs, row counts, license documentation, token budget calculations, and contamination protocols are technically comprehensive and accurate.
- The defect in the adapters is easily rectifiable by populating both keys (`"subject": instruction, "instruction": instruction`) in all adapter dictionaries, or by specifying `source="editpackft"` when feeding pre-adapted dictionaries.
- No source code in `src/`, `notebooks/`, `tools/`, or `tests/` was altered during this task; the integrity violation is confined entirely to the delivered research document (`docs/RESEARCH_SFT_DATASET_CANDIDATES.md`).

---

## 4. Conclusion

**Verdict**: **INTEGRITY VIOLATION** (Fail).  
The deliverable `docs/RESEARCH_SFT_DATASET_CANDIDATES.md` cannot be certified as CLEAN in its current state.

### Actionable Remediation Required:
1. **Fix Field Adapters in Section 3 and Section 9.2**:
   Update all adapter snippets (`map_instructcoder_to_bifocal`, `map_codereview_to_bifocal`, `map_nextcoder_to_bifocal`) and the Section 9.2 verification script to include `"subject": instruction` alongside `"instruction": instruction`:
   ```python
   raw_record = {
       "subject": instruction,
       "instruction": instruction,
       "old_contents": before_code,
       ...
   }
   ```
2. **Reconcile Recommended SFT Mixture Ratios**:
   Align Section 1.1 / Section 3.1 with Section 7.1: either adopt the 4-way mix (40% InstructCoder / 30% GitHub Code Review / 20% NextCoder / 10% EditPackFT) consistently throughout the document, or update Section 7.1 to reflect the Tri-Curated 40/35/25 mixture.
3. **Guard AST Sanitizer for GitHub Code Review**:
   Note in Section 4.2 that indented code fragments from `github-codereview` require dedenting (`textwrap.dedent`) before AST parsing to prevent spurious `IndentationError` drops.

---

## 5. Verification Method

To independently verify this audit finding:
1. Run the Python verification snippet from Section 9.2 lines 771–786 against the project root:
   ```python
   from bifocalgemma.data.offline_sft import editing_record
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
   editing_record(raw_instruct, source="instructcoder")
   ```
   **Observation**: Raises `ValueError: missing_instruction`.
2. Inspect `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:20–25` vs `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:646–655` to confirm the mixture percentage mismatch.
