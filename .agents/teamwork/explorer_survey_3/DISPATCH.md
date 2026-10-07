## 2026-10-07T09:48:52Z
You are teamwork_preview_explorer in .agents/teamwork/explorer_survey_3.
Your parent orchestrator is orchestrator_1 (conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad).

MANDATORY INSTRUCTION:
First, read d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md.

TASK:
Investigate and document the exact provenance, source repositories, problem construction, and data signatures of the three target benchmarks:
1. HumanEvalFix (from `bigcode/humanevalpack` / OpenAI HumanEval):
   - Provenance: 164 hand-crafted Python problems originally from HumanEval, corrupted with synthetic/manual bugs.
   - Exact signatures: function names, docstrings, unit tests, problem structure.
2. CanItEdit (from NUPRL / `nuprl/canitedit`):
   - Provenance: instruction-based code editing benchmark. Where did the code originate? (Python packages/repos, commit history, synthetic instructions?).
   - Exact signatures: repository origins, instruction styles, context types (full file vs function level).
3. CrossCodeEval (from `amazon-science/crosscodeeval` or similar):
   - Provenance: multi-file cross-repo code completion / editing benchmark.
   - What repositories and commits are used? What date cutoffs? What project licenses?

Define a concrete, measurable contamination & overlap detection protocol for candidate training datasets:
- Exact string/hash matching on docstrings/prompts/function signatures
- Repository and commit overlap (checking if candidate datasets contain commits from the benchmark repos)
- Near-duplicate / MinHash / n-gram overlap considerations
- Clear classification rules: when is a candidate "Contaminated" (MUST BE EXCLUDED) vs "Risky" vs "Clean"?

OUTPUT:
Write your findings to `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_3\handoff.md` following the Handoff Protocol.
When done, send a message to orchestrator_1 (Recipient="25871991-f75d-4c31-a84f-84a5e07b84ad") confirming completion.
