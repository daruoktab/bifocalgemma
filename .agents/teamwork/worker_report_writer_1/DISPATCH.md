## 2026-10-07T10:29:38Z
You are teamwork_preview_worker in .agents/teamwork/worker_report_writer_1.
Your parent orchestrator is orchestrator_1 (conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY INSTRUCTION:
First, read d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md.
Also read all evidence reports produced by our specialist team:
- d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1\handoff.md (repo contracts, limits, GPU guard, benchmarks)
- d:\Codings\bifocalgemma\.agents\teamwork\worker_verifier_1\handoff.md (deep candidate verification, schema mappings, token distributions)
- d:\Codings\bifocalgemma\.agents\teamwork\cross_verifier_1\handoff.md (independent cross-verification, adversarial critique, negative filtering, arrow format caveats, invalid SHA fixes)
- d:\Codings\bifocalgemma\.agents\teamwork\challenger_contamination_1\handoff.md (3-stage contamination audit, repo blacklist, final classification)

TASK:
Author the comprehensive, publication-grade research report in:
`d:\Codings\bifocalgemma\docs\RESEARCH_SFT_DATASET_CANDIDATES.md`.

STRICT CONSTRAINTS:
- ZERO MODIFICATIONS to src/, tools/, tests/, or notebooks/. Only create docs/RESEARCH_SFT_DATASET_CANDIDATES.md and write metadata in your working directory.
- All claims must be honest: benchmark score increases must be explicitly framed as testable hypotheses, not promised facts.

The report MUST include:
1. Executive Summary & Key Recommendations (Pyramid principle).
2. Repository Context & SFT Data Contracts (input formatting, unified diff generation via difflib.unified_diff, sequence limits 2048/512, commit-group splitting, single Molab GPU VRAM guard).
3. Candidate Evaluation Matrix & Deep Profiles (evaluate >= 6 candidates: InstructCoder, github-codereview, NextCoderDataset, patchpilot-patchgen, Code-Feedback, Code_Vulnerability_Security_DPO):
   - Hub ID, URL, primary license, total & Python rows, schema fields, pinned revision SHA.
   - Working Python field-mapping snippet to `editing_record`.
   - Relevance to HumanEvalFix, CanItEdit, CrossCodeEval.
4. Independent Cross-Verification Record (documenting independent verification of at least 3 candidates against primary sources, discrepancy analysis, corrected SHAs, negative pair filtering, and format caveats).
5. Measurable Contamination & Benchmark Overlap Audit:
   - 3-stage audit protocol against HumanEvalFix (164 tasks), CanItEdit (105 challenges), CrossCodeEval (471 Python repos).
   - Clean / Filtered Approved Candidates vs. Quarantined / Excluded Datasets (with clear technical justifications).
6. Resource Footprint & Hardware Workload Estimation:
   - Disk storage, token volume, training duration estimates.
   - VRAM memory profile on single Molab GPU under BF16 + FP32 LoRA + gpu_guard.
7. Recommended Data Mixing Strategy:
   - Blending ratios with existing EditPackFT and CommitPackFT.
   - Curriculum stages / recipe.
8. Empirical Hypotheses:
   - Testable hypotheses for expected effects on HumanEvalFix, CanItEdit, and CrossCodeEval, grounded in cited literature.
9. Verification & Audit Reproduction Guide.

Write your completion handoff report to `d:\Codings\bifocalgemma\.agents\teamwork\worker_report_writer_1\handoff.md`.
When done, send a message to orchestrator_1 (Recipient="25871991-f75d-4c31-a84f-84a5e07b84ad").
