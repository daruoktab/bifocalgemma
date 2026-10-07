# BRIEFING — 2026-10-07T10:27:00Z

## Mission
Execute a measurable contamination and benchmark overlap audit on candidate datasets against HumanEvalFix, CanItEdit, and CrossCodeEval using a 3-stage empirical audit protocol.

## 🔒 My Identity
- Archetype: challenger
- Roles: critic, specialist
- Working directory: d:\Codings\bifocalgemma\.agents\teamwork\challenger_contamination_1
- Original parent: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Milestone: candidate_dataset_contamination_audit
- Instance: 1 of 1

## 🔒 Key Constraints
- Review-only — do NOT modify implementation code in src/, tools/, tests/, or notebooks/
- Write metadata only to d:\Codings\bifocalgemma\.agents\teamwork\challenger_contamination_1/
- Empirically verify claims — run code yourself, do not trust claims blindly
- Follow 5-component handoff protocol (Observation, Logic Chain, Caveats, Conclusion, Verification Method)

## Current Parent
- Conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Updated: not yet

## Review Scope
- **Files to review**: d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md, explorer_survey_3\handoff.md, explorer_survey_2\handoff.md
- **Interface contracts**: Candidate SFT dataset selection criteria for BifocalGemma (Gemma 2 2B/9B diff editing)
- **Review criteria**: Stage 1 (exact hash/string matching), Stage 2 (repo/commit intersection), Stage 3 (MinHash / synthetic lineage)

## Attack Surface
- **Hypotheses tested**: 
  1. Hypothesis: `ronantakizawa/github-codereview` can be trained directly without contamination risk. -> REJECTED: It contains real `repo_name` fields that intersect with CrossCodeEval's 471 Python repos. Must be filtered with CCEval repo blacklist.
  2. Hypothesis: `Tejaswiniprabhakaran19/patchpilot-patchgen` is a clean repair dataset. -> REJECTED: Highly risky, overlaps with SWE-bench and CrossCodeEval repos (`DataDog/integrations-core`, `django`, etc.), and uses SEARCH/REPLACE format.
  3. Hypothesis: `m-a-p/Code-Feedback` is clean for HumanEvalFix. -> REJECTED: Academic literature specifically documented HumanEval data leakage in Code-Feedback.
  4. Hypothesis: `CyberNative/Code_Vulnerability_Security_DPO` provides high-quality security repair. -> REJECTED: Official publisher retraction/audit notice (2026-10-06) confirms labels are unverified, inverted, and only ~450 rows are Python.
  5. Hypothesis: `microsoft/NextCoderDataset` is clean. -> CONFIRMED: Authors ran empirical decontamination proving 0% leakage against CanItEdit and HumanEvalFix; MIT license; single-file focus.
  6. Hypothesis: `likaixin/InstructCoder` is clean. -> CONFIRMED: Synthesized May 2023 (predates CanItEdit Dec 2023), 0 matches on HumanEval doctests/signatures, no repo overlap.
- **Vulnerabilities found**: 
  - CrossCodeEval repository collisions in GitHub-scraped datasets (`github-codereview`, `patchpilot-patchgen`).
  - Synthetic lineage leakage in `Code-Feedback`.
  - Quality and correctness invalidation in `CyberNative/Code_Vulnerability_Security_DPO`.
- **Untested angles**:
  - Full automated token 13-gram matrix across all 114k InstructCoder rows against The Stack v2 (computationally prohibitive offline, but bounded by architectural prompt difference).

## Loaded Skills
- None specified in dispatch

## Key Decisions Made
- Completed live API inspection of CanItEdit (105 tasks) and HumanEvalPack (164 tasks).
- Analyzed all 8 candidates + 2 existing datasets + 3 benchmarks across all 3 audit stages.
- Classified datasets into CONTAMINATED, RISKY, and CLEAN tiers.

## Artifact Index
- handoff.md — Final comprehensive contamination audit report
- progress.md — Heartbeat and step tracking
- DISPATCH.md — Timestamped dispatch log
