# Orchestration Plan: BifocalGemma SFT Dataset Research & Evaluation

## Objective
Evaluate and discover additional public datasets for BifocalGemma offline SFT (instruction-to-patch, code editing/repair in Python) aligned with HumanEvalFix, CanItEdit, and CrossCodeEval benchmarks without contamination. Produce a rigorous, verified research report meeting all criteria (R1-R4) with ZERO code modifications.

## Phases and Steps

### Phase 0: Survey & Scoping (Step 0)
- Dispatch 3 Explorers in parallel:
  - **Explorer 1 (Repo & Benchmark Spec)**: Inspect `docs/DATASETS_AND_BENCHMARKS.md`, existing training notebooks, data contracts (input/target, tokenizer, split, formatting), and benchmark characteristics.
  - **Explorer 2 (Dataset Candidates Pool & SFT Alignment)**: Search and identify public Python instruction-to-patch / code editing / bug repair datasets (e.g. Magicoder / Evol-Instruct-Code editing subsets, CoNaLa / commit datasets, PyScent/CodeRefactoring, SWE-bench / SWE-In-The-Wild / SWE-fix subsets, CodeXGlue repair, GitHub diff datasets) suited for offline single-GPU SFT.
  - **Explorer 3 (Benchmark Contamination & Provenance Analysis)**: Analyze the exact origins of HumanEvalFix/HumanEvalPack (HumanEval origins), CanItEdit (Python repos & scenarios), and CrossCodeEval (repositories, commits, time cutoffs) to establish strict contamination exclusion criteria.
- Synthesize survey findings into `PROJECT.md` with Feature/Dataset Inventory and interface/data contract specifications.

### Phase 1: Deep Verification & Extraction (>= 5 Candidates)
- Dispatch Worker / Explorers to verify candidates against primary sources (Hugging Face API / repos / papers):
  - Hub ID / URL, exact license, record count, available fields, pinned commit revision (SHA).
  - Exact mapping from candidate fields to repo data contract: `instruction`, `before_code`, `after_code`/`patch`.
  - Offline SFT compatibility (verifying NO RL, NO teacher model, NO VM/docker execution required).
  - Benchmark alignment rating (HumanEvalFix, CanItEdit, CrossCodeEval).
  - Independent cross-verification for at least 3 candidates.

### Phase 2: Measurable Contamination & Overlap Audit
- Verify overlap against benchmark datasets:
  - Data sources, repositories, commits, problem descriptions, and code solution hashes/signatures.
  - Separate risky / contaminated datasets and exclude them from primary recommendations.

### Phase 3: Workload Estimation & Mixing Strategy
- Compute storage size, training token/time estimates, single Molab GPU VRAM feasibility.
- Define data blending/curation ratios with existing `EditPackFT` and `CommitPackFT`.
- Rigorous claim framing: mark potential benchmark gains strictly as hypotheses unless backed by literature citations.

### Phase 4: Final Research Report & Verification Gate
- Compile final report (e.g., `docs/RESEARCH_SFT_DATASET_CANDIDATES.md`).
- Multi-perspective verification gate:
  - Reviewer 1 & Reviewer 2: Verify accuracy, adherence to constraints, citations, zero code modifications.
  - Challenger: Empirical / metadata stress check.
  - Auditor (`teamwork_preview_auditor`): Forensic integrity check (zero code modifications in src/notebooks/tools/tests, authentic data sources).
- Final synthesis and reporting to sentinel/parent.
