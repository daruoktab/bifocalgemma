## 2026-10-07T10:11:41Z
You are teamwork_preview_challenger in .agents/teamwork/challenger_contamination_1.
Your parent orchestrator is orchestrator_1 (conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad).

MANDATORY INSTRUCTION:
First, read d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md.
Also read:
- d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_3\handoff.md (benchmark provenance, signatures, and 3-stage protocol)
- d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_2\handoff.md (candidate datasets)

TASK:
Execute a measurable contamination and benchmark overlap audit on the candidate datasets against all 3 benchmarks:
1. HumanEvalFix (`bigcode/humanevalpack` / OpenAI HumanEval, 164 tasks)
2. CanItEdit (`nuprl/CanItEdit`, 105 full-file challenges)
3. CrossCodeEval (`amazon-science/cceval`, 471 Python repositories)

Carry out the 3-stage audit protocol:
- Stage 1: Exact Hash & String Matching:
  * Check candidate prompts, before/after code, and functions for occurrences of the 164 HumanEval function names (`has_close_elements`, `separate_paren_groups`, etc.) and canonical docstrings.
  * Check candidate prompts/code for occurrences of the 105 CanItEdit problems, titles, or exact hashes.
  * Check for occurrences of CrossCodeEval groundtruth targets.
- Stage 2: Repository & Commit Intersection:
  * Cross-reference candidate repository origins (e.g. from `github-codereview`, `patchpilot-patchgen`, `EditPackFT`, `CommitPackFT`) against the 471 Python repositories in CrossCodeEval.
  * Are any candidate commits drawn from the same repositories or within the benchmark snapshot windows?
- Stage 3: Near-duplicate / MinHash / Synthetic Lineage:
  * Analyze the risk of indirect contamination from synthetic datasets (e.g. InstructCoder seeds, Evol-Instruct variations).
- Categorization:
  * Classify each candidate into: CONTAMINATED (must exclude), RISKY (separate and exclude from primary recommendation), or CLEAN (approved for SFT).

OUTPUT:
Write your comprehensive contamination audit report to `d:\Codings\bifocalgemma\.agents\teamwork\challenger_contamination_1\handoff.md` following the Handoff Protocol.
DO NOT MODIFY ANY FILES IN src/, tools/, tests/, or notebooks/.
When done, send a message to orchestrator_1 (Recipient="25871991-f75d-4c31-a84f-84a5e07b84ad").
