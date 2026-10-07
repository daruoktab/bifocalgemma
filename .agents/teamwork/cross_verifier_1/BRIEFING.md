# BRIEFING — 2026-10-07T10:22:00Z

## Mission
Perform independent cross-verification against primary sources for at least 3 dataset candidates (likaixin/InstructCoder, ronantakizawa/github-codereview, microsoft/NextCoderDataset) to satisfy Acceptance Criterion R2 and stress-test upstream claims.

## 🔒 My Identity
- Archetype: teamwork_preview_reviewer
- Roles: reviewer, critic
- Working directory: d:\Codings\bifocalgemma\.agents\teamwork\cross_verifier_1
- Original parent: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Milestone: survey cross-verification
- Instance: 1 of 1

## 🔒 Key Constraints
- Review-only — do NOT modify implementation code (DO NOT MODIFY ANY FILES IN src/, tools/, tests/, or notebooks/)
- Write only to .agents/teamwork/cross_verifier_1
- Independent verification against primary sources (HF Hub API, dataset card, paper citation, git repo)
- Actively check for integrity violations (hardcoded test results, facade implementations, shortcuts, fabricated verification outputs, self-certification)
- Adhere to Handoff Protocol with 5 mandatory components (Observation, Logic Chain, Caveats, Conclusion, Verification Method)

## Current Parent
- Conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad
- Updated: not yet

## Review Scope
- **Files to review**:
  - `d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md`
  - `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1\handoff.md`
  - `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_2\handoff.md`
- **Interface contracts**: Acceptance Criterion R2 ("reviewer terpisah memeriksa sampel minimal 3 kandidat terhadap sumber asli")
- **Review criteria**: License verification, row counts / split counts, schema field matching, pinned SHA validity, runtime/compiler dependencies, synthetic data artifact risks, adversarial failure modes.

## Key Decisions Made
- Executed direct primary-source verification against HF Hub API endpoints for `likaixin/InstructCoder`, `ronantakizawa/github-codereview`, `microsoft/NextCoderDataset`, `Tejaswiniprabhakaran19/patchpilot-patchgen`, and `CyberNative/Code_Vulnerability_Security_DPO`.
- Issued verdict `REQUEST_CHANGES` due to Critical Integrity Violations in `explorer_survey_2/handoff.md` (fabricated dataclass signatures and non-existent `make_unified_diff` function in verification code, 404 SHA on PatchPilot) and Major Technical Inaccuracies (missing schema fields on InstructCoder, negative example crashes on github-codereview, unhandled arrow format on NextCoderDataset).

## Artifact Index
- `d:\Codings\bifocalgemma\.agents\teamwork\cross_verifier_1\handoff.md` — Final cross-verification audit report

## Review Checklist
- **Items reviewed**: `explorer_survey_1/handoff.md`, `explorer_survey_2/handoff.md`, HF Hub API endpoints, raw dataset cards, ICML 2025 / ACL 2024 papers.
- **Verdict**: REQUEST_CHANGES
- **Unverified claims**: Addressed and debunked fabricated verification code in survey 2.

## Attack Surface
- **Hypotheses tested**: Direct drop-in claim for InstructCoder (failed due to missing path/commit fields); valid revision SHA for PatchPilot (failed with 404); runnable verification script in survey 2 (failed due to non-existent import).
- **Vulnerabilities found**: Integrity violation in survey 2; synthetic hallucinations in InstructCoder; negative pair unhandled crash in github-codereview; Arrow binary format incompatibility in NextCoderDataset.
- **Untested angles**: Full corpus token-level n-gram overlap between all 355k github-codereview rows and 105 CanItEdit tasks (deferred to offline data prep step).
