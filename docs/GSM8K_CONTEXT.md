# GSM8K supplementary evaluation and context budgets

Updated 9 October 2026. This defines the next evaluation protocol and length
candidates; it is not a completed model evaluation or a GPU-fit guarantee.

## Role in the study

GSM8K evaluates arithmetic reasoning retention before/after code-repair SFT,
for the same four pairs and two IT decoder-only baselines. It is supplementary
to repository repair, not evidence of repository understanding or long-context
retrieval. Keep its released test questions/answers outside the SFT pool.
Eight-shot demonstrations, when used, come from the released train split.
No RL, calculator, tool execution, VM or external verifier is required for the
proposed numeric-answer accuracy protocol. A math prompt must be used; the
repair runner's patch-only prompt/scorer cannot be reused as a GSM8K evaluator.

## Actual tokenizer audit

All 1,319 released test questions were measured with hash-verified Gemma IT
assets and the official chat template, with enable_thinking=False. Zero-shot
prompts request written solution steps and a final `####` numeric answer.
Eight-shot uses the first eight upstream train rows in original order; inline
calculator annotations are removed. Test solutions never enter the prompt.

| Token stream | P50 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot prompt | 90 | 136 | 160 | 222 |
| Eight-shot prompt | 1,148 | 1,194 | 1,218 | 1,280 |
| Reference answer, annotations removed | 87 | 171 | 220 | 301 |

These are actual lengths under this declared prompt protocol, not the length
of future generated reasoning. A 2,048-token input cap and up to 2,048 newly
generated tokens are ample candidates for this GSM8K protocol. Overflow,
missing final answer and generation-limit termination must remain visible.
For fair comparisons use identical prompts/demonstrations, sample order,
greedy decoding, generation budget and a final numeric-answer scorer; report
zero-shot and eight-shot separately. Do not compare these settings to scores
from another protocol as if they were identical.

The [audit receipt](data/gsm8k_context_profile_20261009.json) binds source commit,
raw train/test SHA-256, tokenizer assets, demonstration-prefix SHA and the full
ordered per-row length-stream hash. Raw dataset files remain outside Git.

## Longer repository context

Official [E2B IT config](https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json)
and [E4B IT config](https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json)
set text max_position_embeddings=131072. This is an architectural position
limit, not a measured feasible training length for native BifocalGemma.
The current merged attention is eager, creates dense attention tensors and
has no temporal generation cache. Memory grows with query/key lengths; a 96 GB
GPU does not automatically make 32k-input/8k-output training feasible.

The next larger repository budget candidate is **8,192 encoder tokens and
8,192 total paired-decoder training tokens**. For matched decoder-only training,
the corpus gate reserves 16,896 total tokens (8,192 + 8,192 + 512). Paired
training length includes the issue prompt, target and closing turn; it must
not be presented as 8,192 freely generated output tokens. GSM8K generation can
use its own shorter budget even when a model has a larger position limit.

Before freezing this larger corpus as the main study, measure original-model
forward/backward memory and runtime on Molab, starting at smaller boundaries
and covering both symmetric/asymmetric E4B pairs and the longer-prefix
baselines. Run one configuration at a time. If eager attention cannot meet
this target, attention/checkpoint optimization must be validated; merely
raising the position-limit setting does not solve activation memory.
Keep the live 2k pilot corpus, new larger corpus and checkpoints independent.
The raw repository profile shows 8k paired-decoder coverage of 96.01%, before
license/snapshot/context gates; encoder retrieval length still needs profiling.

Source: [official GSM8K release](https://github.com/openai/grade-school-math).
