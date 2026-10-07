# BRIEFING — 2026-10-07T09:46:25Z

## Mission
Cari, verifikasi, dan evaluasi >= 5 dataset tambahan untuk offline SFT BifocalGemma (instruction-to-patch) yang selaras dengan HumanEvalFix, CanItEdit, dan CrossCodeEval tanpa kontaminasi, menghasilkan laporan riset terverifikasi tanpa menyentuh kode repo.

## 🔒 My Identity
- Archetype: teamwork_preview_orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: d:\Codings\bifocalgemma\.agents\teamwork\orchestrator_1
- Original parent: parent (sentinel)
- Original parent conversation ID: 1e01ff46-5059-4db2-a4e4-af2cd4263387

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: d:\Codings\bifocalgemma\.agents\teamwork\orchestrator_1\PROJECT.md
1. **Decompose**: Survey and evaluate dataset candidates for BifocalGemma SFT against benchmarks without contamination; deliver verified research report.
2. **Dispatch & Execute** (pick ONE):
   - **Direct (iteration loop)**: Explorer survey -> Candidate verification & overlap audit -> Synthesis & Report drafting -> Reviewer & Auditor gate.
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (sub-orchestrators only, last resort)
4. **Succession**: At 16 spawns, write handoff.md, spawn successor
- **Work items**:
  1. Survey repo contracts, benchmarks, and candidate landscape [done]
  2. Dataset discovery, schema verification, and revision pinning (>=5 candidates) [in-progress]
  3. Contamination & benchmark overlap check (HumanEvalFix, CanItEdit, CrossCodeEval) [pending]
  4. Workload estimation and mixing strategy synthesis [pending]
  5. Final verified research report and review gate [pending]
- **Current phase**: 2
- **Current focus**: Milestone 1 (Deep Verification & Schema Mapping)

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly.
- ZERO CODE MODIFICATIONS in src/, tools/, tests/, notebooks.
- Pure offline SFT only: instruction + pre-code + post-code/patch. No RL, VM, Docker, teacher/reward models.
- Hardware target: 1 Molab GPU.
- Verified from primary sources (Hugging Face / APIs / repos). Independent cross-verification for at least 3 candidates.
- Contamination check against HumanEvalFix, CanItEdit, CrossCodeEval must be measurable. Risky datasets separated and excluded from primary recommendations.
- Claims must be honest: mark expected gains as hypotheses unless backed by cited empirical evidence.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.

## Current Parent
- Conversation ID: 1e01ff46-5059-4db2-a4e4-af2cd4263387
- Updated: 2026-10-07T09:45:47Z

## Key Decisions Made
- Research & evaluation task with strict zero-code modification constraint.
- Dispatch 3 parallel Explorers for survey per Project Pattern Step 0.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_survey_1 | teamwork_preview_spec_miner | Repo Contract & Benchmark Spec Mining | completed | 3a273948-cae2-4ca5-b2f6-9b0bd788f539 |
| explorer_survey_2 | teamwork_preview_explorer | SFT Dataset Candidate Discovery | completed | e0b5c213-0ecd-4a47-89d0-8f1bf8aaa407 |
| explorer_survey_3 | teamwork_preview_explorer | Benchmark Provenance & Contamination | completed | e11bb1a7-aa01-4eff-a578-c8719cb0f183 |
| worker_verifier_1 | teamwork_preview_worker | Primary Candidate Verifier & Schema Mapper | completed | 8d843e05-5fe9-46ad-8fe2-b25da6edc5f0 |
| cross_verifier_1 | teamwork_preview_reviewer | Independent Source Cross-Verifier (>=3 sets) | completed | 80038afd-4553-490a-9d46-94c4f367af51 |
| challenger_contamination_1 | teamwork_preview_challenger | Measurable Contamination & Overlap Auditor | completed | 29c4d201-bb1d-4a96-a8de-74534b4af24c |
| worker_report_writer_1 | teamwork_preview_worker | Final Research Report Author (docs/) | completed | 3aa127dc-eb7e-4a50-a2d2-6dddb0e0f9eb |
| reviewer_gate_1 | teamwork_preview_reviewer | Technical Quality Reviewer | in-progress | 89357c21-e700-4ed3-bc53-394d26c8cef3 |
| reviewer_gate_2 | teamwork_preview_reviewer | Benchmark & Contamination Reviewer | in-progress | d47d0a74-5bb4-4297-bd6d-a36322989ac2 |
| challenger_gate_1 | teamwork_preview_challenger | Empirical & Stress Test Challenger | in-progress | 22bdb4f1-9ff8-465f-a5c6-0b2a5a34d819 |
| auditor_gate_1 | teamwork_preview_auditor | Forensic Integrity Auditor | completed | 5163f66d-2ae8-4f25-a25e-fe557c89b97b |
| worker_remediation_1 | teamwork_preview_worker | Remediation Worker (docs/) | completed | 424435c2-5822-4046-9392-7ba51e1f9259 |
| auditor_gate_2 | teamwork_preview_auditor | Forensic Integrity Re-Auditor | in-progress | 9b36fe6f-8c09-456a-82ff-5baa293532e7 |
| reviewer_gate_3 | teamwork_preview_reviewer | Technical Quality Re-Reviewer | in-progress | 6575951c-fa96-49eb-ad25-857933f8c119 |

## Succession Status
- Succession required: no
- Spawn count: 14 / 16
- Pending subagents: 9b36fe6f-8c09-456a-82ff-5baa293532e7, 6575951c-fa96-49eb-ad25-857933f8c119
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: 25871991-f75d-4c31-a84f-84a5e07b84ad/task-12
- Safety timer: none
- On succession: kill all timers before spawning successor
- On context truncation: run `manage_task(Action="list")` — re-create if missing

## Artifact Index
- d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md — Original user request
- d:\Codings\bifocalgemma\.agents\teamwork\orchestrator_1\DISPATCH.md — Dispatch log
- d:\Codings\bifocalgemma\.agents\teamwork\orchestrator_1\BRIEFING.md — Working memory
- d:\Codings\bifocalgemma\.agents\teamwork\orchestrator_1\progress.md — Liveness & progress tracker
- d:\Codings\bifocalgemma\.agents\teamwork\orchestrator_1\plan.md — Orchestration plan
