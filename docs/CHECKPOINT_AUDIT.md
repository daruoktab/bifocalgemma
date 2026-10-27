# Trained-checkpoint audit on Colab G4, 27 October 2026

## Finding

The observed repair failure is not explained by missing adapters or by temporal
KV caching alone. On the inspected HumanEvalFix examples, trained Bifocal models
use encoder information to reduce teacher-forced loss but fail to reproduce source
code accurately enough to generate applicable patches. This is evidence of weak
source grounding in the current trained configuration, not proof that the general
encoder–decoder approach cannot work.

Evaluation also imposes a strict diff contract: git headers and valid hunk counts.
That is a separate source of failures. Correcting hunk counts alone did not rescue
the inspected encoder-only predictions.

## Procedure and scope

- Dedicated Colab CLI session `bifocalgemma-audit`, G4 RTX PRO 6000, stopped after
  results were downloaded and checked. The observed rate was 8.90 compute units
  per hour; the final check confirmed zero active assignments and zero usage rate.
- Official pinned base/IT backbones, published notebook runtime, and final step
  11,102 adapters from E2B→E2B, E4B→E4B, E2B-IT and E4B-IT.
- Checkpoint metadata/checksums validated by the production loader; every loaded
  adapter/alignment tensor compared exactly against the safetensors file.
- HumanEvalFix `Python/0`, `Python/1`, `Python/39`, selected for inspection rather
  than as a representative statistical sample. Asymmetric models were not tested.
- Cached versus uncached generation, and teacher-forced logit comparison at
  prefill and decoder positions 510, 511, 512, 513 and 540.
- Correct, empty and swapped encoder context, holding decoder prompt and gold
  target identical for the loss comparisons.
- Generation with the exact training prompt, duplicated source in the decoder,
  and decoder source with an empty encoder.
- Temporary LoRA scaling zeroed for a pretrained decoder control, then restored.
  No optimizer updates, checkpoint changes, training runs, or HF writes.
- Generation capped at 384 tokens for these diagnostics, not the full benchmark
  allowance. Length-limited diagnostic outputs must not be treated as complete
  benchmark predictions. Gold targets are used for teacher-forced loss and hidden
  test scoring, not as input to free generation.

## Loss with controlled encoder context

Arithmetic mean over the three inspected cases; these are diagnostic NLLs, not
the token-weighted full-validation losses.

| Model | Correct context | Empty context | Swapped context |
|---|---:|---:|---:|
| E2B→E2B | 0.4815 | 0.6451 | 0.7647 |
| E4B→E4B | 0.3930 | 0.6155 | 0.8084 |
| E2B-IT, source in decoder | 0.1153 | — | — |
| E4B-IT, source in decoder | 0.1269 | — | — |

Correct context improves all six paired cases. Thus these measurements contradict
the claim that the encoder contributes no information whatsoever. They do not
demonstrate accurate copying, functional repair, or benchmark superiority.

## Cache comparison

All twelve prefill comparisons have the same next-token argmax. All sixty selected
boundary comparisons have the same argmax. Relative logit L2 differences are
nonzero: up to 3.91% for paired models and 4.47% for baseline models at the selected
boundary positions. The tiled FP32-softmax reference and BF16 SDPA cache are not
bit-identical; complete generations sometimes diverge in both model families.

Disabling the cache did not rescue any of the six paired predictions. On
`Python/0`, for example, E2B→E2B invents `are_close`/`is_close` functions while the
source defines `has_close_elements`, in both cache and reference generation.
This does not certify all cache behavior or long-context workloads; it rules out
the cache as the sole explanation for these failures.

## Prompt, decoder-source, and patch controls

Strict functional test passes out of three selected cases:

| Model | Benchmark, cached | Benchmark, uncached | Training prompt | Source in both paths | Source only in decoder |
|---|---:|---:|---:|---:|---:|
| E2B→E2B | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 |
| E4B→E4B | 0/3 | 0/3 | 0/3 | 1/3 | 0/3 |

Recomputing hunk counts with `git apply --recount` is a diagnostic counterfactual,
not a change to published benchmark scoring:

- Encoder-only benchmark and training-prompt predictions still fail to apply in
  all six cases, including uncached predictions. Hunk counts cannot repair invented
  source lines or function names.
- With source duplicated into the decoder, E2B `Python/0` becomes a functional pass
  after recounting. E4B `Python/0` already passes strictly; E4B `Python/1` can apply
  after recounting but fails its hidden tests.
- With source only in the decoder, E2B reaches 2/3 passes when the recount-only
  correction is included. E4B reaches 2/3; its third applicable patch times out.
  These are altered input architectures and are not scores for encoder-only
  BifocalGemma.
- Without LoRA, the original decoder generally emits plain fenced unified diffs
  without the required `diff --git` header. The strict parser rejects these as
  `unexpected_patch_prose`. Therefore this control cannot establish catastrophic
  forgetting from its strict score alone. The prompt says “unified diff” while
  the parser accepts a narrower git-diff format; this contract deserves explicit
  alignment and separate measurement for every model.

## Interpretation and next work

The evidence points toward insufficient source fidelity through the learned
encoder interface in these checkpoints, alongside diff-format sensitivity. It
does not identify norm, LoRA rank, training duration, or bidirectional conversion
as a uniquely proven cause. A useful encoder signal and low teacher-forced loss
are not enough to guarantee source-grounded autoregressive generation.

Before another full training run, evaluate a small held-out copying/localization
cohort, measure source fidelity separately from diff formatting, and align the
prompt/scorer contract consistently for baselines and paired models. Any revised
recipe should pass a short pilot on actual source copying and patch applicability.
Simply disabling the cache, replacing normalization again, or recounting diffs is
not supported here as a complete fix.

## Artifacts and reproduction

Local evidence, intentionally outside Git:

- `outputs/colab/audit-final.zip`: verified archive containing all three phases,
  hardware, detailed generations, scores, and worker logs. No token or weight files.
- `outputs/colab/audit.json`: primary loss/cache/adapter evidence. Its initial
  static patch scores included EOS; use the corrected EOS-stripped scores in
  `audit-controls.json`, which match the production benchmark's stripping policy.
- `outputs/colab/audit-controls.json`: prompt/context controls and native tests.
- `outputs/colab/audit-decoder-control.json`: temporary no-LoRA control and recount
  counterfactuals, including proof that LoRA scaling was restored.

Export scripts with conda `ai` using `tools/prepare_colab_audit.py`, then
`--controls`, then `--decoder-control`, in the same dedicated G4 runtime. These
flags belong only to the developer audit tool; the six Molab notebooks remain
one-Run workflows. The final control uses cached local backbone files and requires
no additional token. Download evidence before stopping the runtime.
