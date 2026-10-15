# Sequential full SFT, benchmarks and upload

The common template generates three standalone Molab notebooks. Each has 21 cells;
20 cells are identical. Only family, first experiment and output directory differ.

| Notebook | Default sequence |
|---|---|
| [Symmetric](../notebooks/bifocalgemma_train_symmetric.py) | E2B–E2B → E4B–E4B |
| [Asymmetric](../notebooks/bifocalgemma_train_asymmetric.py) | E2B–E4B → E4B–E2B |
| [Baseline](../notebooks/bifocalgemma_train_baseline.py) | E2B-it → E4B-it |

One model is resident at a time. All six configurations receive the same frozen
training IDs and evaluation cohorts. Baselines place the same context in their
causal decoder prefix. The family notebooks do not train the baselines again.
These are SFT-matched baselines initialized from the IT checkpoints. Their scores
must not be labeled as untouched pretrained baseline results.

Each model now runs **base AdamW → optimized GrokOrScale LoRA**, independently
initialized from the same pretrained sources/seed. Four runs per notebook,
twelve total. See [recipe comparison](TRAINING_RECIPE_COMPARISON.md) for shared
numbers, exact optimizer semantics, and evidence limits.

## Budgets and defaults

The user-selected training limits are **32,768 encoder tokens** and **8,192 shifted
decoder input tokens**, retaining the project's existing prefix-inclusive decoder
contract. The baseline total allowance is **41,472** (32,768 +8,192 +512).
These are limits, not padding targets; short examples keep their actual lengths.
Input and target overflow is rejected during curation. Training never silently
truncates or resamples the published pool.

`RUN_TRAINING`, `RUN_BENCHMARKS`, `FUNCTIONAL_BENCHMARKS`, `AUTO_RESUME`,
`HF_UPLOAD_ENABLED` and `CHECKPOINT_UPLOAD_ENABLED` default to `True`.
`MAX_STEPS=0` means all frozen training IDs for `EPOCHS=1`. Explicit positive
MAX_STEPS is a user-selected partial budget. Ordinary embeddings and PLE remain
frozen; full-corpus training still uses attention LoRA, not full parameter tuning.

QA requires a letter and allows 32 generated tokens. Code completion, repair and
math allow up to the decoder budget after reserving the paired generation prefix.
Pairs and baselines use the same per-case output allowance. Completion cohort
admission also rejects references that cannot fit that allowance and long exact
answer copies already visible in the supplied context. These are documented
adaptations, not claims of official benchmark protocol parity.

The larger GPU memory requirement has **not** been measured on pretrained
32K/8K models. Before optimization, the notebook checks actual maximum encoder,
decoder and joint example lengths plus a combined shape stress. This preflight
does not update the optimizer or add synthetic rows to the training pool.

## Durable stages

Each model/recipe owns `OUTPUT_ROOT/EXPERIMENT/RUN_NAME/{base|optimized}/`:

1. `training`: full SFT and full held-out loss; optimizer-boundary checkpoints.
2. `training_upload`: verified training artifacts on the Hub.
3. `benchmark:<collection>`: predictions and scoring for each selected cohort.
4. `final_upload`: verified benchmark results and the finished pipeline receipt.

`pipeline_state.json` binds the model, data fingerprint, tokenizer provenance,
policy, budgets and runtime to hash-verified evidence. A file merely existing does
not mark a stage complete. Failed/running stages are retried; intact completed
stages are skipped. Changing the training cohort or policy requires a fresh run
name. Benchmark protocol settings have their own receipts and output directories.

Training resumes from the latest complete v2 checkpoint, including optimizer,
sampler order/cursor, Python and Torch CPU/CUDA RNG. A partial directory is ignored;
a corrupt completed checkpoint is rejected. If a recognized interrupted run has
no complete checkpoint yet, its diagnostics are archived and training restarts
from zero. Incorrect compatibility never silently initializes a different run.

Validation interval is ceil(25% of actual total optimizer steps); checkpoint
interval is ceil(10%). Final validation runs once and a final checkpoint is
always available. At11,102 updates these intervals are2,776 and1,111. Exactly
the latest two checkpoints are retained per model/recipe. Fewer uploads mean
up to about10% of a run may need repeating after a lost runtime. Telemetry
continues at every step; neither periodic validation nor upload is a training
update. Optimizer EMA/momentum/calibration states are included in v2 resume.

Every benchmark prediction is journaled and flushed before advancing to the next
case. A final partial journal append is discarded after a process crash. Completed
IDs must exactly match the expected cohort prefix; duplicates and reordered IDs
are errors. Failed upload can resume without generating committed predictions
again. A different model/checkpoint/cohort cannot reuse that prediction journal.

## Two-checkpoint retention and Hub recovery

Only the **two latest complete checkpoints per model/recipe/run** remain locally and in
the current Hub tree. Newly saved files are validated before pruning. Remote
deletions target only obsolete numeric checkpoint folders inside this exact run;
other experiments and historical runs are outside that scope. Hub history is not
squashed or erased.

Checkpoint uploads contain a complete optimizer state. Final uploads include a
receipt in the same Hub commit as the artifacts. A fresh runtime recovers the
selected run at one pinned Hub revision and validates hashes before resuming.
Benchmark progress is uploaded every 100 cases by default. Local progress is
committed after every case; losing the runtime can require repeating cases after
the most recent successful remote synchronization.

## RAM and monitoring

Token IDs are stored without modification in an int32 disk cache with checksums.
Only current microbatches become Python tuples/tensors. Values are not clipped or
reduced modulo a tiny vocabulary. Cache identity binds the pool, actual tokenizer
asset bytes, limits and prompt recipe. Code/model uploads do not include this
private token cache.

The monitor shows the active model and stage, optimizer/epoch progress, train/dev
CE, LR by group, gradient norm, throughput, GPU allocated/reserved/peak/free and RAM.
Linux RAM readings honor container limits instead of presenting host RAM as the
Molab allowance. Benchmark progress includes counts, timing, ETA and the latest
prediction. Durable files include history, events, monitor HTML, stage receipts,
predictions and benchmark reports. Source and objective loss breakdowns are stored
separately; validation CE remains distinct from functional repair success.

The v8 reference was read in full (6,742 lines). Staged recovery, complete
checkpoint selection, scoped retention and verified uploads are retained.
The optimized comparison adds its gradient-filter/orthogonal/trust-ratio ideas;
steering, vision grafting and preference/RL stages remain outside this recipe.

## Validation boundary

CPU tests use native tiny Gemma topology for all six configurations. They validate
sequencing, model release, parameter policies and checkpoint behavior; they do not
establish pretrained model quality. Real tokenizer audits are separate from tiny
model tests. Original pretrained training/benchmark execution remains a Molab run
performed by the user after applying the HF write token.

## Dependency bootstrap for fresh runs

The first cell explicitly reinstalls pinned packages with `uv pip` into the current
kernel interpreter. Notebook Torch is pinned to **2.14.1+cu130**, the latest stable
version verified on15 October2026; both Python3.12 and3.13 are accepted. These runs
start from scratch under a new `-torch2141` run suffix. The old2.11.0 resume pin
has been replaced. A dependency receipt includes the new pins, so an old receipt
cannot skip reinstall. A matching receipt avoids repeated reinstall after
restart, while preloaded native modules force a kernel restart before proceeding.
No packages are installed into conda base. A runtime mismatch still blocks resume.

Torchvision is pinned to0.29.1+cu130 and reinstalled in the same CUDA-index
transaction as Torch. Inherited torchvision0.26.0 in the Molab image caused
`torchvision::nms` registration to fail during Gemma4 config imports. Both build
tags are verified; loaded torchvision requires a restart after reinstall.
A tiny CPU NMS invocation checks the compiled extension before native imports.

## Next-run scheduler and batching

See [the next-run audit](TRAINING_NEXT_RUN.md). Defaults now use microbatch2,
accumulation2, validation batch up to2, 3% warm-up and cosine decay to10% peak LR.
The new run name prevents accidental resume of constant-LR pilots. Train/dev
shape preflight uses the configured batch size. Benchmark generation stays single-case.
