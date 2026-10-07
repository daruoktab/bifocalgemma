# Dispatch Log - worker_remediation_1

## 2026-10-07T10:50:35Z
You are teamwork_preview_worker in .agents/teamwork/worker_remediation_1.
Your parent orchestrator is orchestrator_1 (conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY INSTRUCTION:
First, read d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md.
Also read:
- d:\Codings\bifocalgemma\docs\RESEARCH_SFT_DATASET_CANDIDATES.md
- d:\Codings\bifocalgemma\.agents\teamwork\auditor_gate_1\handoff.md (FULL AUDIT REPORT — ADDRESS EVERY VIOLATION)
- d:\Codings\bifocalgemma\.agents\teamwork\reviewer_gate_1\handoff.md
- d:\Codings\bifocalgemma\.agents\teamwork\challenger_gate_1\handoff.md

FULL AUDITOR EVIDENCE TO REMEDIATE:
1. In `src/bifocalgemma/data/offline_sft.py:11–16`:
   ```python
   def editing_record(raw: dict, *, source: str) -> dict:
       before, after = raw.get("old_contents"), raw.get("new_contents")
       instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")
       if not isinstance(instruction, str) or not instruction.strip():
           raise ValueError("missing_instruction")
   ```
   All adapters in Section 3 and the reproduction tests in Section 9.2 only supplied `"instruction"` and omitted `"subject"`, causing `ValueError("missing_instruction")` on 100% of records when `source != "editpackft"`.
   REMEDIATION:
   In all adapter functions (Section 3.2, 3.3, 3.4, 3.5, 3.6, 3.7) and in Section 9.2 verification scripts, populate BOTH `"subject": instruction` and `"instruction": instruction` in the raw dictionary.
2. In Section 9.2, ensure the reproduction test suite actually runs and genuinely passes against the repository codebase (`python -c "..."` or unittest). Verify that `assert str(e) == "empty_or_unchanged_code"` succeeds because line 13 receives `"subject"` properly!
3. Reconcile the mixture percentage numbers across the document:
   Adopt a consistent, unified 4-way recommended mixture throughout (Section 1.1, Section 3.1, Section 7.1):
   - 40% `likaixin/InstructCoder` (40,000 examples)
   - 30% `ronantakizawa/github-codereview` (30,000 CCEval-purged examples)
   - 20% `microsoft/NextCoderDataset` (20,000 Python examples)
   - 10% `nuprl/EditPackFT` (10,000 baseline examples)
   Total: 100,000 examples.
4. NextCoder commit grouping: Update `commit = f"nextcoder_seed_{seed_id}"` so that variants of the same seed program share the same commit group and are never split across train and dev splits.
5. In Section 4.2 / Section 3.3, document that indented code chunks from `github-codereview` must be dedented using `textwrap.dedent()` before `ast.parse()` to prevent spurious `IndentationError` discards.
6. In `sanitize_posix_path`, handle `None` safely (`if not raw_path or not isinstance(raw_path, str): return "module.py"`).
7. Recompute token counts and training durations accurately:
   - Total tokens for 100k: calculate weighted average token length and total token count precisely.
   - Training duration: explicitly state duration for E2B–E2B (~4.3 hrs) and E4B–E4B (~11.1 hrs).
   - VRAM calculation: note decimal 96 GB = 89.41 GiB binary VRAM, and note that `HEADROOM_GIB = 16` subsumes activation requirements under microbatch 1.
8. ZERO CODE MODIFICATIONS:
   Only modify `docs/RESEARCH_SFT_DATASET_CANDIDATES.md`. DO NOT touch `src/`, `notebooks/`, `tools/`, or `tests/`.

TASK:
Update `docs/RESEARCH_SFT_DATASET_CANDIDATES.md` with all the above fixes.
Write your completion report to `d:\Codings\bifocalgemma\.agents\teamwork\worker_remediation_1\handoff.md`.
When done, send a message to orchestrator_1 (Recipient="25871991-f75d-4c31-a84f-84a5e07b84ad").
