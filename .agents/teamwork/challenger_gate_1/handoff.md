# Adversarial Challenge & Empirical Verification Report: SFT Dataset Candidates (Gate 1)

**Agent:** `teamwork_preview_challenger` (`challenger_gate_1`)  
**Roles:** Critic, Specialist (Empirical Challenger)  
**Date:** 2026-10-07  
**Recipient:** `orchestrator_1` (`25871991-f75d-4c31-a84f-84a5e07b84ad`)  
**Target Document:** `docs/RESEARCH_SFT_DATASET_CANDIDATES.md` (Version 1.0.0, 842 lines)  
**Verdict:** **REQUEST_CHANGES**  
**Overall Risk Assessment:** **CRITICAL**

---

## 1. Observation

All observations are derived from direct inspection of the codebase (`src/bifocalgemma/data/offline_sft.py`, `notebooks/bifocalgemma_train_symmetric.py`), and the candidate research document (`docs/RESEARCH_SFT_DATASET_CANDIDATES.md`).

### 1.1 Observation O1: Fatal `editing_record` Contract Mismatch in All Field Mappings
In `src/bifocalgemma/data/offline_sft.py:11–16`:
```python
def editing_record(raw: dict, *, source: str) -> dict:
    before, after = raw.get("old_contents"), raw.get("new_contents")
    instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("missing_instruction")
```
However, in `docs/RESEARCH_SFT_DATASET_CANDIDATES.md`:
1. **InstructCoder Adapter** (lines 210–220):
   ```python
   raw_record = {
       "instruction": instruction,
       ...
   }
   return editing_record(raw_record, source="instructcoder")
   ```
2. **GitHub Code Review Adapter** (lines 282–292):
   ```python
   raw_record = {
       "instruction": instruction,
       ...
   }
   return editing_record(raw_record, source="github_codereview")
   ```
3. **NextCoder Adapter** (lines 351–361):
   ```python
   raw_record = {
       "instruction": instruction,
       ...
   }
   return editing_record(raw_record, source="nextcoder")
   ```
4. **Verification Test Simulation** (Section 9.2, lines 772–786):
   ```python
   raw_instruct = {
       "instruction": "Replace recursive factorial with an iterative loop.",
       ...
   }
   record_instruct = editing_record(raw_instruct, source="instructcoder")
   ```
When `source != "editpackft"`, `editing_record` evaluates `instruction = raw.get("subject")`. Because all proposed adapters provide `"instruction"` and omit `"subject"`, `instruction` evaluates to `None`, which triggers line 15: `raise ValueError("missing_instruction")`. Every single proposed adapter and the Section 9.2 test assertion will crash with `ValueError("missing_instruction")`.

### 1.2 Observation O2: NextCoder Train/Dev Data Leakage via Row-Indexed Commits
In `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:350`:
```python
commit = f"nextcoder_py_{index}"
```
In `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:311`:
> "Python subset: 45,837 rows (15,279 unique Python programs × 3 instruction variants)."

In `src/bifocalgemma/data/offline_sft.py:90–100`:
```python
groups = sorted({record["group"] for record in records})
random.Random(seed).shuffle(groups)
dev_groups = set(groups[:count])
train = [r for r in records if r["group"] not in dev_groups][:train_limit]
dev = [r for r in records if r["group"] in dev_groups][:dev_limit]
assert not ({r["group"] for r in train} & {r["group"] for r in dev})
```
By assigning `commit = f"nextcoder_py_{index}"`, each of the 3 prompt variants of the same base Python program receives a different `group` (`nextcoder_py_0`, `nextcoder_py_1`, etc.). Consequently, variants of the identical program seed are distributed across both `train` and `dev` splits, directly violating the repository's held-out split invariant.

### 1.3 Observation O3: Internal Mixture Contradiction & Arithmetic Discrepancies
1. **Mixture Share Contradiction**:
   - `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:21–24` (Section 1.1) and Table 3.1 specify a **Tri-Curated Mixture**:
     - 40% InstructCoder
     - 35% GitHub Code Review
     - 25% NextCoder
     - Total = 100% (0% EditPackFT).
   - `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:649–654` (Section 7.1) specifies a **Quad-Curated Mixture**:
     - 40% InstructCoder (40,000 examples)
     - 30% GitHub Code Review (30,000 examples)
     - 20% NextCoder Python (20,000 examples)
     - 10% EditPackFT (10,000 examples)
     - Total = 100%.
   - `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:670–684` (Section 7.2) specifies a two-stage curriculum averaging:
     - 42.5% InstructCoder, 25% GitHub Code Review, 22.5% NextCoder, 10% EditPackFT.
2. **Token Volume Arithmetic**:
   - Section 6.2 lists per-dataset mean tokens: InstructCoder (~325), GitHub Code Review (~385), NextCoder (~510), EditPackFT (~385).
   - Under Section 1.1 ratios (40/35/25): weighted mean is $0.40(325) + 0.35(385) + 0.25(510) = 392.25$ tokens/ex ($39.23\text{M}$ total).
   - Under Section 7.1 ratios (40/30/20/10): weighted mean is $0.40(325) + 0.30(385) + 0.20(510) + 0.10(385) = 386.0$ tokens/ex ($38.60\text{M}$ total).
   - Section 6.3 line 592 asserts: "$\text{Volume} = 100{,}000 \times \approx 384\text{ tokens/example} \approx 38.4\text{M tokens}$". The number 384 matches neither mixture formula.
3. **Training Duration Under-Estimation for E4B–E4B**:
   - Section 6.3 asserts 155 ms per microbatch and 4.3 hours across the board.
   - For E4B–E4B (14.25B parameters, 28.51 GiB base weights), forward+backward on a single GPU requires ~350–450 ms per microbatch, resulting in $100{,}000 \times 0.400 = 40{,}000\text{ s} \approx \mathbf{11.1\text{ hours}}$, not 4.3 hours.

### 1.4 Observation O4: VRAM Profile Inconsistencies & Headroom Double-Counting
1. In Section 2.4, the preflight guard threshold is:
   $$\text{guard\_bytes} = \text{weights} + (\text{trainable} \times 3) + 16\text{ GiB}$$
   For E2B–E2B: $17.71 + (0.15 \times 3) + 16.00 = 34.16\text{ GiB}$.
   For E4B–E4B: $28.51 + (0.22 \times 3) + 16.00 = 45.17\text{ GiB}$.
2. In Section 6.4 (lines 614–631), the memory breakdown lists:
   - E2B–E2B: Base (17.71) + LoRA (0.15) + AdamW (0.30) + Activations (1.85) + Preflight Guard Buffer (16.00) = **36.01 GiB**.
   - E4B–E4B: Base (28.51) + LoRA (0.22) + AdamW (0.44) + Activations (2.65) + Preflight Guard Buffer (16.00) = **47.82 GiB**.
   Here, activation memory (1.85 / 2.65 GiB) is double-counted on top of the 16 GiB safety headroom buffer (which was specifically sized to cover activations and workspace memory). Furthermore, the breakdown omits gradient memory ($1 \times \text{trainable}$).
3. Section 6.4 treats 96 GB as $96.00\text{ GiB}$. If an NVIDIA GPU has 96 GB (decimal, $96 \times 10^9$ bytes), its capacity in binary GiB is $96 \times 10^9 / 2^{30} \approx \mathbf{89.41\text{ GiB}}$.

### 1.5 Observation O5: Data Ingestion Failure Modes
1. **`sanitize_posix_path` Crash on Nullable Path**:
   In Section 3.3 (line 278): `path = sanitize_posix_path(raw.get("file_path", "module.py"))`. If a record has key `"file_path": None` (standard in parquet tables with null values), `raw.get(...)` returns `None`. `sanitize_posix_path` immediately crashes with `AttributeError: 'NoneType' object has no attribute 'replace'`.
2. **Missing AST Syntax Check**:
   Section 4.2 line 453 mandates: *"Ingestion must drop rows containing 'Copy code', rows where before == after, and rows where AST parsing fails (SyntaxError)."* However, the implementation snippet in Section 3.2 (lines 201–205) omits `ast.parse` entirely.
3. **Regex Truncation of Internal Code Fences in NextCoder**:
   In Section 3.4 line 328: `re.search(r"```python\s*(.*?)\s*```", prompt, re.DOTALL)` uses a non-greedy match. If Python code contains docstrings with internal markdown code fences, `.*?` terminates prematurely at the first inner fence, silently truncating the source code. Additionally, line 344 falls back to `completion.strip()`, retaining backtick fences if ````python` is omitted, causing diff corruption.

---

## 2. Logic Chain

1. **Premise 1 (Contract Execution)**: The research report explicitly presents field mapping adapters in Sections 3.2, 3.3, and 3.4 as production-ready code that satisfies `src/bifocalgemma/data/offline_sft.py:editing_record`.
2. **Premise 2 (Source Inspection)**: `src/bifocalgemma/data/offline_sft.py:13` branches strictly on `source == "editpackft"`. Any other source string expects the field `"subject"`.
3. **Deduction 1**: Passing `source="instructcoder"`, `source="github_codereview"`, or `source="nextcoder"` to `editing_record` with a dict containing `"instruction"` (and not `"subject"`) triggers an immediate unhandled `ValueError("missing_instruction")`.
4. **Impact 1**: Ingestion cannot run as written. The isolated verification test in Section 9.2 fails on its first assertion.
5. **Premise 3 (Held-out Integrity)**: `split_editing_records` groups by commit to prevent leakage across splits.
6. **Premise 4 (Seed Multiplicity)**: NextCoder provides 3 instruction variants per Python seed.
7. **Deduction 2**: Setting `commit = f"nextcoder_py_{index}"` places different variants of the same problem into different commit groups, permitting simultaneous presence in train and dev splits. This breaches the project's zero-leakage invariant.
8. **Premise 5 (Internal Consistency)**: A publication-grade research report must have coherent ratios, reproducible token arithmetic, and accurate workload models.
9. **Deduction 3**: The discrepancy between Section 1.1 (Tri-curated 40/35/25) and Section 7.1 (Quad-curated 40/30/20/10), the uncalibrated 4.3-hour estimate for E4B, and the double-counting of activations in VRAM profile indicate incomplete coordination between report sections.
10. **Conclusion**: The report cannot be approved in its current state. Concrete revisions are required.

---

## 3. Caveats

1. **Candidate Quality**: The identification of `likaixin/InstructCoder`, `ronantakizawa/github-codereview`, and `microsoft/NextCoderDataset` remains sound. Their licensing, record counts, and benchmark alignment were accurately identified.
2. **Single-GPU Feasibility**: The conclusion that training fits on a 96 GB / 89.4 GiB GPU remains valid. The VRAM double-counting is conservative (over-estimates memory rather than under-estimates), so no OOM risk is introduced.
3. **Upstream Exclusion Accuracy**: The exclusion of `CanItEdit`, `HumanEvalPack`, `Code-Feedback`, `PatchPilot`, `Security DPO`, and `SWE-smith` is rigorously justified and fully upheld.

---

## 4. Conclusion & Actionable Recommendations

**Verdict:** **REQUEST_CHANGES**

The report must be updated with the following corrections:

1. **Fix Field Mapping Snippets & Verification Test**:
   In Sections 3.2, 3.3, 3.4, and 9.2, ensure all raw record dictionaries include `"subject": instruction` (or set `source="editpackft"` when calling `editing_record`), or update the mapping to:
   ```python
   raw_record = {
       "instruction": instruction,
       "subject": instruction,  # Satisfies offline_sft.py:13 contract
       "old_contents": before_code,
       "new_contents": after_code,
       "old_file": path,
       "new_file": path,
       "commit": commit,
       "repos": ...,
       "license": ...,
   }
   ```
2. **Eliminate Train/Dev Leakage in NextCoder**:
   Update `map_nextcoder_to_bifocal` to derive `commit` from the base code hash or seed index:
   ```python
   seed_hash = hashlib.sha256(before_code.encode("utf-8")).hexdigest()[:16]
   commit = f"nextcoder_seed_{seed_hash}"
   ```
3. **Reconcile SFT Mixture Ratios & Token Arithmetic**:
   - Align Section 1.1, Table 3.1, and Section 7.1 onto a single authoritative distribution (e.g. 40% InstructCoder, 30% GitHub Code Review, 20% NextCoder, 10% EditPackFT).
   - Recalculate total tokens using the exact weighted average ($38.60\text{M}$ tokens).
   - Differentiate training duration: ~4.3 hours for E2B–E2B vs ~11.1 hours for E4B–E4B.
4. **Correct Section 6.4 VRAM Breakdown**:
   - Reconcile `Total Allocated / Guard` to match Section 2.4 ($34.16\text{ GiB}$ for E2B, $45.17\text{ GiB}$ for E4B). Explain that the 16 GiB headroom buffer subsumes the 1.85 / 2.65 GiB activation requirements.
   - Note decimal 96 GB $\approx$ 89.41 GiB binary VRAM.
5. **Harden Edge Cases**:
   - Update `sanitize_posix_path` call to handle `None`: `path = sanitize_posix_path(raw.get("file_path") or "module.py")`.
   - Add `ast.parse` validation to `map_instructcoder_to_bifocal`.
   - Use greedy matching or delimiter tracking for NextCoder markdown extraction.

---

## 5. Verification Method

To verify these findings:
1. **Contract Crash Reproduction**:
   Inspect `src/bifocalgemma/data/offline_sft.py:11–16`. Trace a call to `editing_record({"instruction": "foo", "old_contents": "a", "new_contents": "b", "old_file": "x.py", "new_file": "x.py"}, source="instructcoder")`. Observe that `raw.get("subject")` returns `None`, raising `ValueError("missing_instruction")`.
2. **NextCoder Leakage Audit**:
   Trace `split_editing_records` with 3 items sharing identical `before_code` but different `commit = f"nextcoder_py_{i}"`. Observe that `dev_groups` splits them across `train` and `dev`.
3. **Ratio Comparison**:
   Compare `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:21–24` against `docs/RESEARCH_SFT_DATASET_CANDIDATES.md:649–654`.

### Invalidation Conditions
This critique will be invalidated if:
- `src/bifocalgemma/data/offline_sft.py` accepts `"instruction"` for any `source` value.
- NextCoder records have unique base programs with no shared seeds across rows.
- Section 1.1 and Section 7.1 present mathematically identical mixture proportions.
