## 2026-10-07T10:11:41Z
You are teamwork_preview_reviewer in .agents/teamwork/cross_verifier_1.
Your parent orchestrator is orchestrator_1 (conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad).

MANDATORY INSTRUCTION:
First, read d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md.
Also read:
- d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1\handoff.md
- d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_2\handoff.md

TASK:
Perform INDEPENDENT CROSS-VERIFICATION against primary sources for at least 3 top candidates (e.g. `likaixin/InstructCoder`, `ronantakizawa/github-codereview`, `microsoft/NextCoderDataset`).
This fulfills Acceptance Criterion R2 ("reviewer terpisah memeriksa sampel minimal 3 kandidat terhadap sumber asli").

For each of the 3 candidates:
1. Independently query and check the source (Hugging Face API, dataset card, paper citation, git repo).
2. Verify:
   - Does the license match? (Check for research-only, commercial, or OpenAI ToS constraints).
   - Do the split row counts and Python row counts match?
   - Do the available fields match the claimed schema?
   - Is the pinned revision SHA valid on the Hugging Face Hub?
   - Is there any hidden dependency on external compilers, Docker, or execution environments?
3. Compare with claims made in `explorer_survey_2/handoff.md`:
   - Identify any discrepancies, omissions, or unverified assumptions.
   - Specifically assess the risk of synthetic data artifacts (e.g. GPT-4 hallucinations in `InstructCoder`, negative pairs filtering in `github-codereview`, markdown fences parsing in `NextCoderDataset`).

OUTPUT:
Write your independent cross-verification audit report to `d:\Codings\bifocalgemma\.agents\teamwork\cross_verifier_1\handoff.md` following the Handoff Protocol.
DO NOT MODIFY ANY FILES IN src/, tools/, tests/, or notebooks/.
When done, send a message to orchestrator_1 (Recipient="25871991-f75d-4c31-a84f-84a5e07b84ad").
