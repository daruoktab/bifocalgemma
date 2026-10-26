# Resume after the 26 October 2026 benchmark fix

Use the latest six `notebooks/train_*.py` files and restart the Molab kernel before
Run. The notebook keeps its existing model repository and `train` run directory.
There is no stage selector or configuration change to make.
Benchmark output also reuses the original collection folders, including their
existing short suffixes. The new protocol is recorded in metadata and does not
create another version folder on Hugging Face.

- A completed, verified training stage skips training and loads its final adapter
  checkpoint for evaluation.
- An incomplete training stage resumes the latest complete local checkpoint, or
  the latest uploaded checkpoint when recovering a new Molab instance. Work after
  the last remote checkpoint cannot be recovered from Hugging Face.
- The benchmark reset `2026-10-26-exact-prompt-budget-v3` archives stale local
  evaluation artifacts once. Further interruptions resume the new prediction
  journal rather than resetting it again.
- Remote benchmark artifacts are removed in the existing repositories. Training
  reports, validation reports, data identities, checkpoint weights, optimizer/RNG
  states and training cursors remain intact.

## Cause and correction

Repair generation prepends `Return only a unified diff fulfilling the instruction.`
to its decoder prompt. The previous shared output-budget calculation omitted that
instruction, reserving too many output tokens. This caused paired models to fail
the generation limit check before producing a prediction.

Both actual generation and output-budget calculation now use the same prompt
renderer. The output allowance remains shared by paired and decoder-only models;
source context is not truncated to fit. New reports use
`greedy_cached_normalized_merged_v3_exact_prompt_budget`, and the budget policy is
part of the journal signature.

The original pinned E2B/E4B tokenizers were checked on all 164 prepared HumanEvalFix
Python examples for each of the six architectures: zero overflow after the fix,
with an identical output allowance of 8,156 tokens. No GPU generation or new
quality benchmark was performed in this local audit.

## Checkpoint compatibility

The old source identity included evaluation implementation. A reviewed exact-source
entry in `configs/training_resume_compatibility.json` retains the prior training
identity for this benchmark-only update, allowing the same checkpoints and GPU
profile to resume. It records the original immutable runtime and before/after AST
digests of the three changed source files. Any further semantic source change
breaks that exact match and restores normal strict identity checking.

The model, data processing, loss, optimizer, scheduler, batch settings, and training
loop were not changed by this release. Actual source fingerprints are still
recorded separately in code provenance.

## Remote reset tooling

`tools/reset_hf_benchmarks.py` defaults to a read-only plan. An explicit `--apply`
commits only benchmark deletions and replacement pipeline/upload/status metadata,
with a parent-commit check. It verifies replacement bytes and compares all training
Git blob identities before and after the commit, including checkpoint files.
It supports completed and interrupted training runs. Run it only while the
affected notebooks are stopped, to prevent an old writer from uploading stale
results again. Prior results remain recoverable from the archived immutable Hub
revision recorded under `outputs/benchmark_reset/`.
