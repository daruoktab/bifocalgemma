# BRIEFING — 2026-10-07T10:10:00Z

## Mission
Survey, search, and verify 6-8 promising public dataset candidates for offline Python instruction-to-patch code editing/repair SFT, analyzing metadata, schemas, compatibility, and benchmark alignment.

## 🔒 My Identity
- Archetype: explorer
- Roles: dataset candidate discovery, schema verification, SFT alignment analysis
- Working directory: d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_2
- Original parent: 25871991-f75d-4c31-a84f-84a5e07b84ad (orchestrator_1)
- Milestone: Phase 0 - Candidate Pool Survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Do NOT modify any files in src/, tools/, tests/, or notebooks/
- Only write metadata/handoff files in d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_2
- Pure offline SFT: no RL, no VM/Docker execution, no teacher/reward model
- Schema must support: instruction + before_code -> after_code or unified diff

## Current Parent
- Conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Updated: not yet

## Investigation State
- **Explored paths**: `docs/DATASETS_AND_BENCHMARKS.md`, `.agents/teamwork/ORIGINAL_REQUEST.md`, `src/bifocalgemma/contracts.py`, `src/bifocalgemma/data/offline_sft.py`, Hugging Face Hub APIs for 10 candidates.
- **Key findings**:
  1. Identified 10 candidates with exact SHAs, schemas, licenses, and row counts.
  2. Ranked top candidates: `likaixin/InstructCoder` (114k Python triplets, 1:1 drop-in), `ronantakizawa/github-codereview` (~35k+ Python PR review edits, 1:1 drop-in), `microsoft/NextCoderDataset` (~47k Python edits, minimal markdown regex).
  3. Identified critical contamination risks: `nuprl/CanItEdit` and `bigcode/humanevalpack` are pure evaluation benchmarks and must be strictly excluded from training.
- **Unexplored areas**: None for Phase 0 survey. Investigation complete.

## Key Decisions Made
- Prioritize candidates directly providing before/after pairs or diffs with natural language instructions.
- Exclude SWE-bench variants from offline SFT due to lack of pre-edit file content in parquet tables.
- Strict blacklist on `nuprl/CanItEdit` and `bigcode/humanevalpack` to protect evaluation integrity.

## Artifact Index
- `DISPATCH.md` — Record of dispatch task
- `BRIEFING.md` — Situational awareness
- `progress.md` — Liveness heartbeat
- `handoff.md` — Final 5-component report
