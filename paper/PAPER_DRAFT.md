# BifocalGemma: Bidirectional Repository Encoding and Decoupled Cross-Attention for Gemma 4

**Buana Daru Okta — November 2026**

## Abstract

BifocalGemma investigates a bidirectional repository encoder paired with a causal Gemma 4 decoder for long-context code tasks. The current matrix contains eight encoder–decoder pairings and two decoder-only baselines. Paired models use a trainable SwiGLU bridge and gated, separately normalized cross-attention while preserving the backbone's layer-wise key/value sharing. A staged supervised fine-tuning recipe and common evaluation schedule support comparison, subject to differences in parameter count and training dynamics. **[PLACEHOLDER: Final verified benchmark, uncertainty, and hardware results.]**

## 1. Introduction

Repository tasks require evidence scattered across files and a precise generated answer or patch. A causal prefix cannot revise an earlier token's representation using later tokens, although later decoder positions can attend to the preceding prefix. We study whether a bidirectional encoder and causal decoder provide a useful alternative. Benefits in accuracy, memory, or throughput remain empirical questions.

## 2. Architecture and Methodology

### 2.1 Backbones and bridge

Each paired model has a Gemma 4 encoder and decoder; encoder and decoder word embeddings and per-layer embeddings are separate in the matrix. Bidirectional encoding removes the causal direction restriction while retaining the backbone's sliding-window or global attention structure; a sliding layer is not globally connected. An optional query prefix can condition the encoder. Decoder generation remains causal.

For each encoder token $h$, the trainable bridge maps $d_{enc}$ to $d_{dec}$ with $d_m=\max(d_{enc},d_{dec})$:

$$x=\operatorname{RMSNorm}_{pre}(h),\qquad z=\gamma\odot\operatorname{RMSNorm}_{post}(W_{res}x+W_{down}[\operatorname{SiLU}(W_{gate}x)\odot W_{up}x]).$$

The residual is always a learned linear map: initialized to identity for equal widths and orthogonally otherwise. The down projection is sampled with standard deviation $0.05/\sqrt{d_m}$, and $\gamma$ starts at one. Pre/post RMSNorm means this initialization is not an exact identity function. The bridge is tokenwise and does not compress the encoder sequence.

### 2.2 Decoupled cross-attention

The decoder retains native causal self-attention and adds a separate encoder cross-attention softmax; this is distinct from merged attention, which uses a joint normalization over encoder and decoder keys. Cross queries reuse the decoder query projection with a separate normalization, without cross-stream rotary positions. Owner layers have independent cross key/value projections initialized from corresponding self projections; shared layers reuse their designated owner's states. The cross key/value cache can be reused during decoding, but cross-attention still reads the encoder-length cache.

For valid encoder length $N$, the cross query is multiplied by $\beta(N)=\max\{1,1+0.1\ln(N/2048)\}$ and a learned per-head scale clamped to $[0.1,5]$. Multiplication sharpens logits relative to unit scale; an effect on task accuracy is not established. For head $h$ and unscaled normalized cross query $q$, outputs are combined before the shared output projection as $O(A_{self}+\tanh(a_h)\,2\sigma(4q^\top w_h/\sqrt{d_h})\,A_{cross})$. Here $a_h$ starts at $\operatorname{atanh}(0.05)$ and $w_h$ at zero, giving an initial cross multiplier of 0.05. This retains the self branch but does not make the complete model identical to its decoder initialization.

### 2.3 Training

The paired matrix recipe defaults to a frozen-decoder first stage followed by joint adaptation; encoder LoRA and bridge/cross parameters are trainable in the first stage, and decoder LoRA joins in the second. The default split is approximately half of the chosen one-pass step horizon, subject to configuration. Decoder-only baselines use their decoder adaptation throughout that horizon. The common schedule matches examples and effective batch across models; this does not equalize trainable parameters, FLOPs, tokenization, or time spent adapting the decoder. LoRA rank and alpha are 16 and 32. Default peak learning rates are $2\times10^{-5}$ for encoder/decoder adapters and $10^{-4}$ for alignment, with warmup and cosine scheduling. These are configuration choices rather than established optima.

Supervision uses causally aligned target labels: prompt positions are masked and each prediction is aligned with the next target token, retaining target whitespace. The active matrix objective and any marker weighting must be reported from the run manifest; the implementation supports both uniform target loss and differential marker-token weighting. Marker weighting applies to token IDs, not whole diff lines. **[PLACEHOLDER: Dataset revision, sample counts, effective batch, actual steps, epochs, objective, and run-specific settings.]**

## 3. Experimental Setup

The matrix includes `2base-2it`, `2it-2it`, `4base-4it`, `4it-4it`, `4base-2it`, `4it-2it`, `2base-4it`, and `2it-4it`, plus `e2b-it` and `e4b-it` causal baselines. The first component denotes the encoder and the second the instruction-tuned decoder. Context budgets, cohorts, model revisions, decoding settings, and denominators must be taken from each completed run; nominal maxima do not prove every example uses the same number of tokens.

The active suite contains eight locally adapted cases: [LongCodeQA from LongCodeBench](https://arxiv.org/html/2505.07897v3); a Python [HumanEvalFix](https://arxiv.org/html/2308.07124v2) docstring-to-unified-diff task based on HumanEvalPack; [SWE-QA](https://arxiv.org/html/2604.24814v1) oracle and noisy-oracle conditions; [GSM8K](https://github.com/openai/grade-school-math) zero-shot and eight-shot tasks; and CrossCodeLongEval chunk and function completion from [Repoformer](https://arxiv.org/html/2403.10059v2). LongCodeQA is not LongCodeArena, and CrossCodeLongEval is distinct from CrossCodeEval. The prepared LongCodeQA cohort has 107 of 113 source cases; each completion task samples 100 eligible cases by default. These protocols are not reproductions of published creator scores: HumanEvalFix requests a patch rather than a full function, the GSM8K eight-shot exemplars differ from `lm-eval gsm8k_cot`, and completion formatting, budgets, postprocessing and metrics differ from Repoformer. The post-training causal baselines must also be distinguished from the original Google instruction-tuned checkpoints.

For multiple-choice tasks, the primary strict score accepts an isolated option with allowed delimiters; a separately reported resilient parser may extract an answer from more elaborate text. This strict policy differs from SWE-QA's published explanation-tolerant extraction. GSM8K strict extraction follows the released `####` numeric-string convention. For code completion, raw exact match removes all whitespace before comparison, and normalized metrics additionally remove supported fences, fill-in-the-middle markers, and echoed suffixes. Whitespace removal can erase differences in string literals and indentation. SequenceMatcher similarity is not the normalized Levenshtein edit similarity described by Repoformer, and regex identifier-set F1 is an auxiliary local metric; none of these establishes execution correctness. HumanEvalFix functional scoring applies the patch and runs hidden tests when execution is enabled. Report invalid outputs, failure counts and denominators, and never substitute normalized or resilient values for the primary metric. Repair syntax and localization metrics do not by themselves establish test-passing issue resolution.

## 4. Empirical Results

**[PLACEHOLDER: Final verified per-model, per-benchmark primary and secondary metrics, sample counts, uncertainty, and failures.]**

**[PLACEHOLDER: Matched hardware, peak memory, prefill latency, decode throughput, and training dynamics.]**

**[PLACEHOLDER: Predefined ablations, including architecture and training-schedule comparisons.]**

## 5. Limitations

The paired model has an additional encoder and cross-attention parameters, so matching data and steps alone cannot isolate architecture from compute and adaptation differences. Bidirectional context, stationary cross-attention, gating, and staged training are design hypotheses; their effects require controlled ablations. A finite benchmark cohort, parser sensitivity, length truncation, and checkpoint differences can affect comparisons. **[PLACEHOLDER: Observed failure modes and limitations from final runs.]**

## 6. Conclusion

We describe a versionless BifocalGemma implementation with a tokenwise bridge, separately normalized gated cross-attention, and a staged training recipe. **[PLACEHOLDER: Empirical conclusions and deployment recommendations after final validation.]**
