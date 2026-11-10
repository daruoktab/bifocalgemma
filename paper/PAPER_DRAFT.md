# BifocalGemma: Asymmetric Bidirectional Repository Encoding and Stationary Cross-Attention for Gemma 4 Developer Agents

**Track**: Google — The Gemma 4 Developer Agent Paper Track (Kaggle Competition Writeup, $\le 3,000$ words)  
**Repository**: [https://github.com/daruokta/bifocalgemma](https://github.com/daruokta/bifocalgemma) | **Hugging Face**: [https://huggingface.co/daruokta/bifocalgemma-2026-11-02](https://huggingface.co/daruokta/bifocalgemma-2026-11-02)

---

## Abstract

Autoregressive decoder-only language models process software repositories under a structural bottleneck: causal masks prevent earlier repository tokens from attending to symbols, type definitions, or call sites appearing later in the linearized file stream, while prefix-concatenated prompts force the decoder to maintain massive key-value (KV) caches across tens of thousands of tokens during patch generation. We introduce **BifocalGemma**, an asymmetric encoder-decoder developer agent architecture built natively on the **Gemma 4** family (`Gemma 4 E2B` and `Gemma 4 E4B`). BifocalGemma decouples repository comprehension from autoregressive patch synthesis by pairing a bidirectional Gemma 4 repository encoder ($N_{\text{enc}} \le 32{,}768$ tokens) with a causal Gemma 4 patch decoder ($N_{\text{dec}} \le 8{,}192$ tokens). To bridge these streams without destabilizing pretrained causal representations or breaking Gemma 4's cross-layer KV-sharing topology, we introduce three innovations: (1) a **SwiGLU Dual-Gate Coordinate Bridge** (`SwiGLUDualGateBridge`) with orthogonal/identity residual pathways, zero-initialized down-projection, and per-channel post-normalization calibration; (2) **Decoupled Stationary Full-Rank Cross-Attention** (`BifocalAttentionBlock`) enforcing rotational position invariance ($\Theta_{\text{cross}} = 0$), logarithmic **Critical Attention Length Scaling** $\beta(N_{\text{enc}})$, learnable per-head sharpness $\sigma_{\text{cross}}$, and dual-gated head- and token-level fusion; and (3) a **Two-Stage Paired Training Curriculum** driven by **Chunked Differential Diff Cross-Entropy** ($\mathcal{L}_{\text{diff}}$). Across the 6-model controlled ablation matrix evaluated on `LongCodeArena` (`SWE-QA`), `RepoBench-C-8k`, and `SWE-bench Lite` under a strict Neutral Prompt Contract, **[PLACEHOLDER: Quantitative summary of primary benchmark outcomes across SWE-QA, RepoBench-C-8k, and SWE-bench Lite, alongside 32K memory and decoding throughput gains, to be populated upon completion of the active runs]**. All models are exported as standalone Hugging Face `PreTrainedModel` artifacts (`BifocalGemmaForCausalLM`) supporting native `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

---

## 1. Introduction

Repository-level software engineering requires two distinct computational modes: **global bidirectional comprehension** over multi-file dependency graphs and **precise local autoregressive synthesis** of unified diffs. Standard causal language models conflate both into a single left-to-right token stream. When a repository snapshot is linearized into a causal prefix $x_1, \dots, x_N$, a declaration at token $x_i$ cannot attend to downstream callers or implementations at token $x_j$ ($j > i$). Moreover, during iterative patch generation, every generated decoder token must attend over the entire $N$-token causal KV cache, incurring $\mathcal{O}(N_{\text{enc}} + N_{\text{dec}})$ memory bandwidth cost at every decoding step.

While recent work adapts causal backbones into bidirectional encoders (e.g., LLM2Vec, T5Gemma), grafting a bidirectional repository encoder onto a **Gemma 4** decoder presents three architectural challenges:
1. **Subspace and Norm Mismatch Under Unmasked Self-Attention**: Removing the causal mask alters the spectral distribution and activation norms of final hidden states, causing variance explosion if injected directly into a causal decoder.
2. **RoPE Coordinate Collision and Long-Context Entropy Dilution**: Applying Rotary Position Embeddings (RoPE) across decoupled encoder and decoder streams creates artificial distance decay between a decoder token at step $t_{\text{dec}}$ and repository definitions at arbitrary file offsets $t_{\text{enc}} \gg t_{\text{dec}}$. Meanwhile, attending over $N_{\text{enc}} \in [8\text{K}, 32\text{K}]$ encoder tokens increases softmax entropy as $\mathcal{O}(\ln N_{\text{enc}})$, washing out sharp symbol lookups.
3. **Catastrophic Forgetting During Adapter Warmup**: Updating decoder weights alongside unaligned cross-attention projections corrupts fluent code generation before the cross-attention bridge aligns its coordinate system.

### Contributions
- **Asymmetric Bidirectional-to-Causal Gemma 4 Topology**: We pair unmasked bidirectional Gemma 4 encoders (`E2B` at $d=1536$, 35L; `E4B` at $d=2560$, 42L) with causal Gemma 4 decoders while isolating Per-Layer Embeddings (PLE, $d_{\text{ple}}=256$) per stream and preserving Gemma 4's cross-layer KV-sharing hierarchy (15/35 KV-owner layers in E2B; 24/42 in E4B).
- **SwiGLU Dual-Gate Coordinate Bridge**: A residual manifold projector with width $d_{\text{mid}} = \max(d_{\text{enc}}, d_{\text{dec}})$, identity/orthogonal shortcuts, zero-initialized down-projection, and learnable per-channel calibration $\gamma_{\text{calib}}$ supporting symmetric (`E2B`$\to$`E2B`, `E4B`$\to$`E4B`) and asymmetric (`E2B`$\to$`E4B`, `E4B`$\to$`E2B`) pairings.
- **Stationary Full-Rank Cross-Attention with Critical Length Scaling**: A RoPE-free ($\Theta_{\text{cross}}=0$) cross-attention module equipped with logarithmic length scaling $\beta(N_{\text{enc}})$, learnable per-head sharpness $\sigma_{\text{cross}} \in [0.1, 5.0]$, and dual-gated fusion initialized at $5\%$ amplitude ($\tanh(g_{\text{head}}^{(0)}) = 0.05$).
- **Two-Stage Paired Curriculum & Differential Diff Objective**: Combines frozen-decoder bridge alignment (Stage 1) and conservative joint LoRA SFT with $20\%$ comprehension replay (Stage 2) under Chunked Differential Diff Cross-Entropy ($\mathcal{L}_{\text{diff}}$).
- **Standalone Hugging Face Integration**: Self-contained `BifocalGemmaForCausalLM` artifacts with bundled modeling code and self-healing config assets.

---

## 2. Architecture & Methodology

### 2.1 Asymmetric Gemma 4 Backbone & Bidirectional Repository Encoder

Given a repository context $X_{\text{enc}}$ ($N_{\text{enc}} \le 32{,}768$) and task instruction $q$, we prepend `[QUERY] <instruction>\n` to $X_{\text{enc}}$ so encoder self-attention conditions repository representations on the target issue from layer 0. To convert the Gemma 4 backbone into a bidirectional encoder:
1. **Unmasked Bidirectional Attention**: We override causal masking across all sliding-window and global attention layers (`use_bidirectional_attention="all"`), constructing a non-causal 4D padding mask $M_{\text{enc}} \in \{0, -\infty\}^{B \times 1 \times 1 \times N_{\text{enc}}}$.
2. **Stream-Isolated PLE**: Gemma 4 Per-Layer Embeddings ($d_{\text{ple}} = 256$) are computed strictly per stream ($E_{\text{ple}}^{\text{enc}}$ from $X_{\text{enc}}$; $E_{\text{ple}}^{\text{dec}}$ from $X_{\text{dec}}$), preventing token identity leakage.
3. **Post-Encoding Padding Zeroing**: Padded encoder states are explicitly zeroed ($H_{\text{enc}} \leftarrow H_{\text{enc}} \odot \mathbb{I}[\text{mask}_{\text{enc}} = 1]$) prior to bridging.

### 2.2 SwiGLU Dual-Gate Coordinate Bridge (`SwiGLUDualGateBridge`)

To project $H_{\text{enc}} \in \mathbb{R}^{B \times N_{\text{enc}} \times d_{\text{enc}}}$ into decoder coordinate space $Z_{\text{enc}} \in \mathbb{R}^{B \times N_{\text{enc}} \times d_{\text{dec}}}$ with intermediate width $d_{\text{mid}} = \max(d_{\text{enc}}, d_{\text{dec}})$:

$$\tilde{H} = \text{RMSNorm}_{\text{pre}}(H_{\text{enc}}), \qquad G = \text{SiLU}(\tilde{H} W_{\text{gate}}) \odot (\tilde{H} W_{\text{up}})$$

$$Z = W_{\text{res}}(\tilde{H}) + G W_{\text{down}}, \qquad Z_{\text{enc}} = \left(\gamma_{\text{calib}} \odot \text{RMSNorm}_{\text{post}}(Z)\right) \odot \mathbb{I}[\text{mask}_{\text{enc}} = 1]$$

Here $W_{\text{res}}$ is an identity mapping when $d_{\text{enc}} = d_{\text{dec}}$ and an orthogonally initialized projection when $d_{\text{enc}} \neq d_{\text{dec}}$. $W_{\text{down}}$ is initialized near zero ($\mathcal{N}(0, (0.01 / \sqrt{d_{\text{mid}}})^2)$) so the bridge begins as a clean linear transport, while $\gamma_{\text{calib}} \in \mathbb{R}^{d_{\text{dec}}}$ (initialized to $\mathbf{1}$) provides per-channel scale calibration.

### 2.3 Decoupled Stationary Full-Rank Cross-Attention (`BifocalAttentionBlock`)

Every decoder layer wraps the native Gemma 4 attention module inside `BifocalAttentionBlock`:
- **Gemma 4 Cross-Layer KV-Sharing**: Only KV-owner layers (15/35 in `E2B`, 24/42 in `E4B`) instantiate dedicated BF16 full-rank projections $W_{K,\text{cross}}, W_{V,\text{cross}}$ and RMSNorms, warm-started from self-attention $W_K, W_V$. During decoding, $(K_{\text{cross}}, V_{\text{cross}})$ are computed once during prefill, cached in `BifocalDynamicCache`, and reused across KV-shared layers and subsequent steps with $\mathcal{O}(1)$ recomputation cost.
- **Stationary Coordinate Invariance ($\Theta_{\text{cross}} = 0$)**: Because $Z_{\text{enc}}$ already encodes structural topology from 35–42 layers of RoPE-equipped bidirectional self-attention, cross-attention omits RoPE rotations ($\Theta_{\text{cross}} = 0$):
  $$Q_{\text{cross}} = \text{RMSNorm}_{Q,\text{cross}}(H_{\text{dec}} W_Q), \quad K_{\text{cross}} = \text{RMSNorm}_{K,\text{cross}}(Z_{\text{enc}} W_{K,\text{cross}}), \quad V_{\text{cross}} = \text{RMSNorm}_{V,\text{cross}}(Z_{\text{enc}} W_{V,\text{cross}})$$
- **Critical Attention Length Scaling & Learnable Sharpness**: To counteract softmax dilution up to $N_{\text{enc}} = 32{,}768$, we scale $Q_{\text{cross}}$ by a logarithmic normalizer $\beta(N_{\text{enc}})$ and learnable per-head sharpness $\sigma_{\text{cross}} \in [0.1, 5.0]$:
  $$\beta(N_{\text{enc}}) = \max\left(1.0,\; 1.0 + 0.1 \ln\!\left(\frac{\max(N_{\text{enc}}, 1)}{2048}\right)\right), \qquad \hat{Q}_{\text{cross}} = \frac{\text{clamp}(\sigma_{\text{cross}}, 0.1, 5.0)}{\beta(N_{\text{enc}})} Q_{\text{cross}}$$
- **Dual-Gated Head & Token Fusion**: Combining a per-head gate $g_{\text{head}}$ ($\tanh(g_{\text{head}}^{(0)}) = 0.05$) with a query-conditional token gate $w_{\text{gate}}$ ($w_{\text{gate}}^{(0)} = \mathbf{0} \Rightarrow g_{\text{query}}^{(0)} = 1.0$):
  $$g_{\text{query}} = 2\,\sigma\!\left(\frac{4}{\sqrt{d_h}} \langle Q_{\text{cross}}, w_{\text{gate}} \rangle\right), \quad g_{\text{eff}} = \tanh(g_{\text{head}}) \cdot g_{\text{query}}, \quad O_{\text{total}} = W_O\left(O_{\text{self}} + g_{\text{eff}} \odot O_{\text{cross}}\right)$$

### 2.4 Chunked Differential Diff Objective & Two-Stage Curriculum

- **Chunked Differential Diff Cross-Entropy ($\mathcal{L}_{\text{diff}}$)**: Assigns syntax-aware token weights to unified diff lines ($w_{\text{add}} = 1.4$ for `+`, $w_{\text{del}} = 1.2$ for `-`, $w_{\text{hdr}} = 1.1$ for `@@` headers, $w_{\text{ctx}} = 0.8$ for context), normalized per sequence to unit mean. Logits are evaluated in 128-token chunks with Gemma 4 softcapping ($30.0$). Pointer copy loss is disabled ($\lambda_{\text{ptr}} = 0.0$) to avoid verbatim copy bias.
- **Two-Stage Paired Curriculum (BifocalGemma) vs. 1-Stage SFT (Causal Baselines)**:
  - *Stage 1 (Frozen Decoder Bridge Alignment, first 50% steps)*: Encoder and decoder backbones are frozen; only `SwiGLUDualGateBridge` and KV-owner `BifocalAttentionBlock` parameters train at $\text{LR} = 10^{-4}$.
  - *Stage 2 (Joint LoRA SFT, final 50% steps)*: LoRA ($r=16, \alpha=32$) is enabled on encoder and decoder at conservative $\text{LR}_{\text{lora}} = 2 \times 10^{-5}$ ($\text{LR}_{\text{bridge}}=\text{LR}_{\text{cross}}=5 \times 10^{-5}$) with $80\%$ patch synthesis and $20\%$ comprehension replay.
  - *Causal Baselines*: Trained with continuous Single-Stage Full Decoder LoRA SFT across $100\%$ of steps on the identical dataset and step budget.
  - *Partitioned Weight Decay*: $\lambda_{\text{wd}} = 10^{-4}$ strictly on 2D linear bridge weights ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}, W_{\text{res}}$); $0.0$ on norms, $\gamma_{\text{calib}}$, gates, scales, and LoRA weights.

---

## 3. Experimental Setup

### 3.1 Controlled 6-Model Ablation Matrix

| Model ID | Topology | Encoder Backbone | Decoder Backbone | Context ($N_{\text{enc}} / N_{\text{dec}}$) | Curriculum |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `Baseline-E2B-IT` | Causal Only | None | `Gemma 4 E2B-IT` ($d=1536$, 35L) | $0 / 40{,}960$ (32K+8K) | 1-Stage SFT |
| `Baseline-E4B-IT` | Causal Only | None | `Gemma 4 E4B-IT` ($d=2560$, 42L) | $0 / 40{,}960$ (32K+8K) | 1-Stage SFT |
| `Bifocal-2B` | Symmetric | `Gemma 4 E2B` ($d=1536$, 35L) | `Gemma 4 E2B-IT` ($d=1536$, 35L) | $32{,}768 / 8{,}192$ | 2-Stage Paired |
| `Bifocal-4B-E2B` | Dec-Heavy | `Gemma 4 E2B` ($d=1536$, 35L) | `Gemma 4 E4B-IT` ($d=2560$, 42L) | $32{,}768 / 8{,}192$ | 2-Stage Paired |
| `Bifocal-2B-E4B` | Enc-Heavy | `Gemma 4 E4B` ($d=2560$, 42L) | `Gemma 4 E2B-IT` ($d=1536$, 35L) | $32{,}768 / 8{,}192$ | 2-Stage Paired |
| `Bifocal-4B` | Symmetric | `Gemma 4 E4B` ($d=2560$, 42L) | `Gemma 4 E4B-IT` ($d=2560$, 42L) | $32{,}768 / 8{,}192$ | 2-Stage Paired |

### 3.2 Benchmarks & Evaluation Integrity Safeguards

Models are trained on a deterministic repository mixture (`SWE-smith`, `LongCodeArena` `SWE-QA`, `RepoBench-C-8k`) and evaluated on: (1) `SWE-QA` across $8\text{K}$, $16\text{K}$, and $32\text{K}$ context buckets; (2) `RepoBench-C-8k` cross-file retrieval; (3) `SWE-bench Lite` unified diff validity, localization, and patch resolution; and (4) 32K hardware telemetry. Our harness enforces a **Neutral Prompt Contract** (identical task instructions across causal and bifocal models) and a **256-Token Generation Floor** (`max_new_tokens >= 256`) on all comprehension tasks.

---

## 4. Empirical Results

> **[PLACEHOLDER — ACTIVE EXPERIMENTAL RUNS]**: *All empirical metrics from earlier exploratory runs have been omitted. The tables and analyses below will be populated with final verified metrics upon completion of the active training and evaluation runs.*

### 4.1 Long-Context Repository Comprehension (`SWE-QA` & `RepoBench-C-8k`)

| Model | `SWE-QA` @ 8K (%) | `SWE-QA` @ 16K (%) | `SWE-QA` @ 32K (%) | `SWE-QA` Overall (%) | `RepoBench-C-8k` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Baseline-E2B-IT` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Baseline-E4B-IT` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-2B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-4B-E2B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-2B-E4B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-4B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

- `[PLACEHOLDER: Comparative analysis of SWE-QA (8K/16K/32K) and RepoBench-C-8k accuracy between Causal Baselines and BifocalGemma, plus encoder-vs-decoder scaling trade-offs.]`

### 4.2 Repository-Level Code Repair (`SWE-bench Lite`)

| Model | Valid Unified Diff (%) | File / Hunk Localization (%) | Exact Patch Resolution (%) | Validation Loss ($\mathcal{L}_{\text{diff}}$) |
| :--- | :---: | :---: | :---: | :---: |
| `Baseline-E2B-IT` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Baseline-E4B-IT` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-2B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-4B-E2B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-2B-E4B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-4B` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

- `[PLACEHOLDER: Evaluation of unified diff synthesis quality, git patch syntactic validity, and bug localization accuracy on SWE-bench Lite.]`

### 4.3 Hardware Telemetry & Training Dynamics

| Model | Active Params | KV-Owner Cross Layers | Peak VRAM @ 32K (GB) | Prefill @ 32K (ms) | Decode (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Baseline-E2B-IT` | $2.0\text{B}$ | $0$ | `[TBD]` | `[TBD]` | `[TBD]` |
| `Baseline-E4B-IT` | $4.0\text{B}$ | $0$ | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-2B` | $2.0\text{B} + 2.0\text{B}$ | $15 / 35$ | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-4B-E2B` | $2.0\text{B} + 4.0\text{B}$ | $24 / 42$ | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-2B-E4B` | $4.0\text{B} + 2.0\text{B}$ | $15 / 35$ | `[TBD]` | `[TBD]` | `[TBD]` |
| `Bifocal-4B` | $4.0\text{B} + 4.0\text{B}$ | $24 / 42$ | `[TBD]` | `[TBD]` | `[TBD]` |

- `[PLACEHOLDER: Analysis of VRAM efficiency, decoding throughput (tok/s), Stage 1 -> Stage 2 convergence curves, and learned distributions of tanh(g_head) and cross_scale.]`

---

## 5. Limitations & Negative Findings

1. **Dual-Backbone Static Parameter Footprint**: Instantiating both a bidirectional encoder and causal decoder doubles static weight memory ($\sim 4\text{B}$ total parameters for `Bifocal-2B`), although decoding steps only attend over the compact decoder KV cache and the 15/24 KV-owner cross-attention states.
2. **Sensitivity to Single-Stage Warmup**: Training decoder LoRA adapters jointly from step 0 alongside unaligned cross-attention projections degrades unified diff formatting; Stage 1 frozen-decoder alignment and $\tanh(g_{\text{head}}^{(0)}) = 0.05$ initialization are required for stability.
3. **Pointer-Generator Copy Bias**: Enabling explicit pointer copy loss ($\lambda_{\text{ptr}} > 0$) biases the decoder toward copying unchanged context lines rather than synthesizing novel bug fixes; setting $\lambda_{\text{ptr}} = 0.0$ with $\mathcal{L}_{\text{diff}}$ resolves this.
4. **`[PLACEHOLDER: Additional negative findings or boundary conditions observed in the final evaluation runs.]`**

---

## 6. Conclusion

We presented **BifocalGemma**, an asymmetric encoder-decoder developer agent architecture built natively on **Gemma 4** (`E2B` and `E4B`). By combining unmasked bidirectional repository encoding ($N_{\text{enc}} \le 32{,}768$), a norm-calibrated **SwiGLU Dual-Gate Coordinate Bridge**, **Decoupled Stationary Full-Rank Cross-Attention** ($\Theta_{\text{cross}} = 0$) with **Critical Attention Length Scaling** $\beta(N_{\text{enc}})$ and native Gemma 4 KV-sharing, and a **Two-Stage Paired Curriculum** with **Chunked Differential Diff Cross-Entropy**, BifocalGemma bridges global repository comprehension and local patch synthesis.

**[PLACEHOLDER — FINAL CONCLUSION SUMMARY]**: *`[Empirical conclusions and practical deployment recommendations across symmetric and asymmetric Gemma 4 configurations will be inserted here upon completion of the active training and evaluation runs.]`*

---

## Appendix: Standalone Hugging Face Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "daruokta/bifocalgemma-2026-11-02"
subfolder = "fast-bifocalgemma-2b"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder=subfolder,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```
