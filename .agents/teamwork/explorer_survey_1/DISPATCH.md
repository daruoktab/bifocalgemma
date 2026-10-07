## 2026-10-07T09:48:51Z
You are teamwork_preview_spec_miner in .agents/teamwork/explorer_survey_1.
Your parent orchestrator is orchestrator_1 (conversation ID: 25871991-f75d-4c31-a84f-84a5e07b84ad).

MANDATORY INSTRUCTION:
First, read d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md.

TASK:
Proactively probe and extract the exact technical specifications and requirements from the codebase and documentation regarding data contracts and evaluation benchmarks.
DO NOT MODIFY ANY FILES IN src/, tools/, tests/, or notebooks/. Only write metadata/handoff files in your assigned directory: d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1.

Key questions to investigate and answer with authoritative evidence:
1. Examine `d:\Codings\bifocalgemma\docs\DATASETS_AND_BENCHMARKS.md` and any other docs in `docs/`:
   - What are the documented datasets, formats, and benchmark targets?
   - What are the exact requirements for HumanEvalFix/HumanEvalPack, CanItEdit, and CrossCodeEval?
2. Inspect the training notebooks (e.g. `d:\Codings\bifocalgemma\notebooks\bifocalgemma_train_colab.ipynb` and any other notebook in `notebooks/`):
   - What is the exact data format/schema expected by the dataset loader and tokenization pipeline?
   - How are inputs formatted (instruction + code before edit)?
   - How are targets formatted (unified diff / patch / code after edit)?
   - How are commit-grouped splits handled?
   - What tokenizers, sequence lengths, and special tokens are configured?
   - What hardware constraints are configured (e.g., single GPU Molab, VRAM, batch size, gradient accumulation)?
3. Summarize the strict compatibility criteria that any new candidate dataset MUST meet to be used in offline SFT without code/pipeline changes.

OUTPUT:
Write your comprehensive findings and evidence chains to `d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_1\handoff.md` following the Handoff Protocol (Observation, Logic Chain, Caveats, Conclusion, Verification Method).
When done, send a message to orchestrator_1 (Recipient="25871991-f75d-4c31-a84f-84a5e07b84ad") confirming completion and pointing to your handoff.md.
