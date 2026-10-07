# Project: SFT Dataset Research & Evaluation for BifocalGemma

## Architecture
- **Objective**: Identify, evaluate, and verify candidate datasets for offline SFT code repair/editing in Python aligned with HumanEvalFix, CanItEdit, and CrossCodeEval without data contamination.
- **Scope**: Pure offline research, verification, and evaluation report. Zero modifications to `src/`, `notebooks/`, `tools/`, `tests/`.
- **Target Environment**: Single Molab GPU (CUDA, BF16, >=34 GiB VRAM guard), microbatch=1, grad_accum=4.
- **Data Contract**:
  - Input: `### Instruction\n{instruction}\n\n### File: {path}\n{before_code}`
  - Target: Deterministic unified diff generated via `difflib.unified_diff` with git headers.
  - Limits: 2048 encoder tokens, 512 decoder tokens (strict overflow rejection, no truncation).
  - Splitting: Commit-group disjoint held-out splits.

## Feature Inventory
| # | Feature / Work Item | Description | Milestone | Source |
|---|---------------------|-------------|-----------|--------|
| 1 | Contract Specification | Ingestion schema, tokenization limits, and hardware guard rules | M0 | `docs/DATASETS_AND_BENCHMARKS.md`, notebooks |
| 2 | Benchmark Signatures | Exact schemas and provenances for HumanEvalFix, CanItEdit, CrossCodeEval | M0 | HF Hub datasets, benchmark papers |
| 3 | Candidate Verification | Verify >=5 SFT dataset candidates from primary sources (licensing, size, schema, pinned rev) | M1 | HF APIs / datasets-server |
| 4 | Independent Cross-Verification | Independent verification of at least 3 candidates against primary sources | M1 | Secondary verification track |
| 5 | Data Contract Mapping | Field-to-contract mapping for each candidate to BifocalGemma schema | M1 | `contracts.py`, `offline_sft.py` |
| 6 | Contamination Audit | 3-stage contamination check (exact string/hash, repo/commit overlap, MinHash) | M2 | Benchmark signatures vs candidate corpora |
| 7 | Contaminated Set Quarantine | Exclude and quarantine any benchmark-contaminated or high-risk datasets | M2 | Contamination decision matrix |
| 8 | Workload & VRAM Estimation | Disk size, training token volume, training duration, and single Molab GPU VRAM profile | M3 | Tokenizer & `gpu_guard` profiling |
| 9 | Data Mixing Strategy | Recommended blending ratios with EditPackFT and CommitPackFT | M3 | Domain balance & diversity analysis |
| 10 | Research Report Drafting | Comprehensive structured report in `docs/` with evidence and testable hypotheses | M4 | Synthesis of M1-M3 findings |
| 11 | Review & Audit Gate | Independent Reviewers, Challenger, and Forensic Auditor verification | M4 | Gate verification protocol |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | Survey & Specifications | Repo data contracts, benchmark signatures, initial candidate discovery | none | DONE |
| 1 | Candidate Deep Verification | Verify >=5 candidates from primary sources, pinned revisions, schema mapping, 3 independent checks | M0 | DONE |
| 2 | Contamination & Overlap Audit | 3-stage contamination audit against HumanEvalFix, CanItEdit, CrossCodeEval; quarantine list | M1 | DONE |
| 3 | Workload & Mixing Strategy | Resource footprint (disk/time/VRAM) on single Molab GPU; blending ratios and hypotheses | M1, M2 | IN_PROGRESS |
| 4 | Research Report & Review Gate | Author final research report in `docs/`, pass Reviewer, Challenger, and Forensic Auditor gates | M1, M2, M3 | PLANNED |

## Interface Contracts
### Candidate Ingestion Mapping ↔ Data Contract
```python
# Raw candidate dict -> BifocalGemma normalized record
{
    "instruction": str,  # mapped from prompt / instruction / reviewer_comment
    "before_code": str,  # mapped from input / before_code / old_contents
    "after_code": str,   # mapped from output / after_code / new_contents
    "path": str,         # relative POSIX path (native or synthetic e.g. "solution.py")
    "repo": str,         # repo identifier or synthetic source
    "commit": str,       # commit SHA or deterministic pair_hash
    "license": str       # verified open-source license
}
```

## Code Layout
- Agent state & metadata: `.agents/teamwork/`
- Documentation & research reports: `docs/`
- Source code (FROZEN - NO MODIFICATIONS): `src/`, `notebooks/`, `tools/`, `tests/`
