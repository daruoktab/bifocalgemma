# Dispatch History

## 2026-10-07T10:11:41Z
You are teamwork_preview_worker in .agents/teamwork/worker_verifier_1.
Your parent orchestrator is orchestrator_1 (conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY INSTRUCTION:
First, read d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md.
Also read:
- d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1\handoff.md (data contracts, constraints, sequence limits)
- d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_2\handoff.md (candidate dataset landscape)

TASK:
Perform deep verification and schema mapping for at least 6 candidate datasets:
1. `likaixin/InstructCoder`
2. `ronantakizawa/github-codereview`
3. `microsoft/NextCoderDataset`
4. `Tejaswiniprabhakaran19/patchpilot-patchgen`
5. `m-a-p/Code-Feedback`
6. `CyberNative/Code_Vulnerability_Security_DPO`
And document why the following are excluded: `nuprl/CanItEdit`, `bigcode/humanevalpack`, `SWE-bench/SWE-smith-py`.

For EACH candidate:
1. Primary Source Verification: Query Hugging Face API / datasets-server to verify:
   - Hub ID & canonical URL
   - Exact pinned commit revision SHA
   - Primary license & terms of use
   - Exact row count (overall and Python subset)
   - Schema field names and data types
2. Concrete Field Mapping to BifocalGemma Data Contract:
   - Provide concrete, working Python snippet mapping the raw record dict to `RepairExample` / `editing_record` fields (`instruction`, `path`, `before_code`, `after_code`, `target_patch`, `commit`/`group`, `repos`, `license`).
   - Verify that target patch can be produced deterministically via `difflib.unified_diff`.
   - Verify single-file constraint (`old_file == new_file`) and relative POSIX path validity.
3. Offline SFT Suitability:
   - Confirm whether it requires any runtime, environment execution, Docker, VM, or reward/teacher models (MUST BE PURE OFFLINE SFT).
4. Relevance to Target Benchmarks:
   - Rate and explain relevance to HumanEvalFix (Python unit bug repair), CanItEdit (Python instruction code editing), and CrossCodeEval (multi-file completion context).
5. Sequence Length & Token Distribution:
   - Estimate context token lengths against the 2048 encoder / 512 decoder limits.

OUTPUT:
Write your full verification report to `d:\Codings\bifocalgemma\.agents\teamwork\worker_verifier_1\handoff.md` following the Handoff Protocol.
DO NOT MODIFY ANY FILES IN src/, tools/, tests/, or notebooks/.
When done, send a message to orchestrator_1 (Recipient="25871991-f75d-4c31-a84f-84a5e07b84ad").
