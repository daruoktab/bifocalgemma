# BRIEFING — 2026-10-07T10:00:00Z

## Mission
Investigate and document the exact provenance, source repositories, problem construction, and data signatures of target benchmarks (HumanEvalFix, CanItEdit, CrossCodeEval), and define a concrete, measurable contamination & overlap detection protocol for candidate training datasets.

## 🔒 My Identity
- Archetype: teamwork_preview_explorer
- Roles: Benchmark Contamination & Provenance Analyst
- Working directory: d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_3
- Original parent: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Milestone: Phase 0 - Survey & Scoping (Step 0)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Zero code modifications in src/, notebooks/, tools/, tests/
- Investigate from primary sources (papers, official code repositories, dataset cards, commit history)
- Ground all claims in direct observations and primary sources
- Strict classification rules: Contaminated vs Risky vs Clean

## Current Parent
- Conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `docs/DATASETS_AND_BENCHMARKS.md`, `.agents/teamwork/ORIGINAL_REQUEST.md`, `.agents/teamwork/orchestrator_1/plan.md`
  - Primary source dataset card `bigcode/humanevalpack` and paper arXiv:2308.07124 (OctoPack)
  - Primary source dataset card `nuprl/CanItEdit` and paper arXiv:2312.12450 (CanItEdit)
  - Primary source repository `amazon-science/cceval` and paper arXiv:2310.11248 (CrossCodeEval, NeurIPS 2023)
- **Key findings**:
  - HumanEvalFix: 164 Python problems from OpenAI HumanEval, corrupted with subtle human-written bugs across 6 taxonomies and 3 failure symptoms; signature consists of 164 exact entry points, docstrings, and `check(entry_point)` tests.
  - CanItEdit: 105 hand-crafted Python programs authored by 8 programmers; full-file context with dual instructions (`instruction_descriptive`, `instruction_lazy`), hidden test suite; MIT license.
  - CrossCodeEval: 471 open-source GitHub repos, 1,368 files, 2,665 Python completion samples (Apache-2.0); static analysis ensures cross-file context requirement; pinned commits mid-2023.
  - Protocol formulated: 3-stage detection pipeline (Exact Hash & String $\rightarrow$ Repo & Commit Intersection $\rightarrow$ 13-gram / MinHash LSH) with rigorous decision matrix (Contaminated vs Risky vs Clean).
- **Unexplored areas**: None for Phase 0 survey.

## Key Decisions Made
- Established a 3-tier contamination taxonomy: Contaminated (MUST EXCLUDE / PURGE), Risky (QUARANTINE / CONDITIONAL AUDIT), Clean (APPROVED FOR SFT).
- Used token 13-gram overlap threshold (0.80) and MinHash Jaccard threshold (0.50 / 0.80) aligning with BigCode/StarCoder decontamination standard.

## Artifact Index
- `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_3\DISPATCH.md` — Inbound task dispatch
- `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_3\progress.md` — Liveness & progress tracking
- `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_3\BRIEFING.md` — Situational awareness
- `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_3\handoff.md` — 5-component handoff report
