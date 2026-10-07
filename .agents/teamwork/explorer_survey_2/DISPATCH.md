## 2026-10-07T09:48:52Z

[Message] timestamp=2026-10-07T09:48:52Z sender=25871991-f75d-4c31-a84f-84a5e07b84ad priority=MESSAGE_PRIORITY_HIGH
Recipient: explorer_survey_2

Task:
Search and identify at least 6-8 promising public dataset candidates for offline instruction-based Python code editing/repair SFT (beyond `nuprl/EditPackFT` and `bigcode/commitpackft`).
DO NOT MODIFY ANY FILES IN src/, tools/, tests/, or notebooks/. Only write metadata/handoff files in your assigned directory: d:\Codings\bifocalgemma\.agents\teamwork\explorer_survey_2.

Investigate candidates from Hugging Face / research publications:
- Candidates that fit the schema: natural language instruction + pre-edit code (buggy/original) + post-edit code or unified diff (fixed/modified).
- Pure offline SFT: no RL, no VM/Docker execution, no teacher/reward model.
- High relevance to Python code editing and the 3 benchmarks: HumanEvalFix, CanItEdit, CrossCodeEval.
Examples of candidate directions to explore and verify:
  1. `m-a-p/Code-Feedback` (or its code-editing/refactoring subsets)
  2. `NTU-NLP-sg/InstructCoder`
  3. `ise-uiuc/Magicoder-Evol-Instruct-110K` / `ise-uiuc/Magicoder-OSS-Instruct-75K` (analyze if there are editing pairs or pure generation)
  4. SWE-bench derivative datasets (e.g., `princeton-nlp/SWE-bench_Lite`, `SWE-fix`, `Salesforce/SWE-bench-train` or similar patch datasets)
  5. `CyberNative/Code_Vulnerability_Security_DPO` (if SFT pairs exist) or security patch datasets
  6. `nuprl/CanItEdit` (does it have training data or is it pure evaluation?)
  7. GitHub commit diff datasets like `codeparrot/github-code` diffs, `google/code_contests`, or `PyScent`/refactoring datasets
  8. Any other prominent instruction-to-patch / diff datasets (e.g. `vhellendoorn/codereview`, `deepseek-ai/DeepSeek-Coder` instruction subsets, `bigcode/humanevalpack` - caution: contamination risk!)

For EACH candidate found, gather:
- Hugging Face Hub repo ID / URL
- Primary license
- Approximate record count (overall & Python subset)
- Available columns/fields
- Compatibility with offline instruction-to-patch SFT
