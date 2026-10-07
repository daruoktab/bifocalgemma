# BRIEFING — 2026-10-07T09:58:00Z

## Mission
Probe and extract exact technical specifications and requirements regarding data contracts and evaluation benchmarks from codebase, docs, and notebooks.

## 🔒 My Identity
- Archetype: teamwork_preview_spec_miner
- Roles: Specification Miner
- Working directory: d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1
- Original parent: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Milestone: Milestone 1 - Specification Mining (Data Contracts & Benchmarks)

## 🔒 Key Constraints
- DO NOT MODIFY ANY FILES IN src/, tools/, tests/, or notebooks/.
- Only write metadata/handoff files in assigned directory: d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1.
- Follow Handoff Protocol (Observation, Logic Chain, Caveats, Conclusion, Verification Method).
- Communicate via send_message to orchestrator_1 (25871991-f75d-4c31-a84f-84a5e07b84ad).

## Current Parent
- Conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Updated: 2026-10-07T09:58:00Z

## Task Summary
- **What to build**: Extract exact technical specifications and requirements from codebase and docs regarding data contracts and evaluation benchmarks.
- **Success criteria**: Full authoritative extraction answering datasets, formats, benchmark targets (HumanEvalFix/HumanEvalPack, CanItEdit, CrossCodeEval), training notebook schemas, formatting, tokenization, hardware constraints, and dataset compatibility criteria.
- **Interface contracts**: handoff.md in explorer_survey_1
- **Code layout**: Read-only codebase, metadata in .agents/teamwork/explorer_survey_1

## Key Decisions Made
- Analyzed `docs/DATASETS_AND_BENCHMARKS.md`, `docs/DATA_PIPELINE.md`, `docs/EVALUATION.md`, `docs/TRAINING.md`, `docs/TRAINING_NOTEBOOKS.md`, and `docs/PRETRAINED_GPU_PROBE.md`.
- Analyzed `src/bifocalgemma/data/offline_sft.py`, `tokenization.py`, `collator.py`, `contracts.py`, `policy.py`, and `training/offline_sft.py`.
- Analyzed `notebooks/bifocalgemma_train_symmetric.py` and `notebooks/bifocalgemma_train_asymmetric.py` (marimo notebooks).
- Produced comprehensive `handoff.md` strictly covering the 3 questions and following Handoff Protocol.

## Artifact Index
- DISPATCH.md — Recorded dispatch instructions
- BRIEFING.md — Situational awareness
- progress.md — Liveness heartbeat
- handoff.md — Comprehensive findings, features discovered, edge cases, and evidence chains
