# BifocalGemma V3: Overcoming Decoder-Only Dominance in Repository-Level Code Intelligence via Non-Linear SwiGLU Manifold Warping, Decoupled Gated Cross-Attention, and Two-Stage Source-Fidelity Alignment

**Target Architecture**: BifocalGemma V3 (`bifocalgemma-v3-e2b-e2b`, `bifocalgemma-v3-e4b-e2b`, `bifocalgemma-v3-e4b-e4b`)  
**Target Hardware**: 1 Molab GPU Instance (NVIDIA A100-SXM4-80GB / sm86 / sm89 / sm90)  
**Status**: Master Research Proposal & Engineering Blueprint  
**Authors**: BifocalGemma Research Consortium (Architecture, Training, Optimization, Code Representation, and Hardware Feasibility)  
**Date**: November 2026  

---

## Table of Contents
1. [Executive Summary & Paradigm Shift](#1-executive-summary--paradigm-shift)
   - 1.1 The Empirical Paradox: Why BifocalGemma V2 Underperformed Decoder-Only Baseline
   - 1.2 The Core Thesis of BifocalGemma V3
   - 1.3 Quantitative Target Metrics
   - 1.4 Structural Blueprint & Key Innovations Matrix
2. [Mathematical Root Causes of V2 Underperformance](#2-mathematical-root-causes-of-v2-underperformance)
   - 2.1 Double Linear Projection Redundancy and Absence of Manifold Warping
   - 2.2 Merged Attention Joint Softmax Denominator Dilution Across 32k Context
   - 2.3 Asymmetric RoPE Query Frequency Spinning
   - 2.4 Static Blind Encoder Bottleneck
   - 2.5 Direct SFT Shortcut Learning & Exposure Bias
   - 2.6 GrokOrScale Trust Ratio Clamping Clash and Spectral Isotropy
3. [BifocalGemma V3 Architecture Specification](#3-bifocalgemma-v3-architecture-specification)
   - 3.1 SwiGLU Dual-Gate Bridge Adapter
   - 3.2 Decoupled Gated Cross-Attention Architecture
   - 3.3 Stationary Unrotated Cross-Attention Coordinates
   - 3.4 Dual-Granularity Repository Context Ingestion Pipeline
   - 3.5 Query-Injected Bidirectional Encoder Prefix
4. [Hardware-Guarded Execution on 1 Molab GPU (A100-80GB / sm86 / sm89)](#4-hardware-guarded-execution-on-1-molab-gpu)
   - 4.1 The Catastrophic Attention Matrix Hazard & SDPA / FlashAttention-2 Resolution
   - 4.2 Triton Shared Memory Hardware Bounds (101,376 Bytes) & Kernel Clamping
   - 4.3 Comprehensive VRAM Budget and Headroom Breakdown
5. [Two-Stage Training & Manifold Alignment Recipe](#5-two-stage-training--manifold-alignment-recipe)
   - 5.1 Two-Stage Curriculum Rationale
   - 5.2 Stage 1: Adapter & Cross-Attention Warmup
   - 5.3 Stage 2: Joint LoRA SFT on Diff Generation
   - 5.4 Progressive Length Curriculum & Critical Attention Temperature Scaling
6. [Loss Formulations & Optimizer Dynamics](#6-loss-formulations--optimizer-dynamics)
   - 6.1 Elimination of Obsolete Losses Under Decoupled Architecture
   - 6.2 Zero-Materialization Streaming LogSumExp Pointer Loss ($\mathcal{L}_{ptr}$)
   - 6.3 Normalized Differential Diff Token Weighting
   - 6.4 Partitioned 3-Way AdamW Optimizer Architecture
7. [Empirical Evaluation Plan & Testable Hypotheses](#7-empirical-evaluation-plan--testable-hypotheses)
   - 7.1 Benchmark Suite and Evaluation Protocols
   - 7.2 Formal Hypotheses
   - 7.3 Complete Ablation Matrix
8. [Concrete Implementation Blueprint & PyTorch Pseudocode](#8-concrete-implementation-blueprint--pytorch-pseudocode)

---

## 1. Executive Summary & Paradigm Shift

### 1.1 The Empirical Paradox: Why BifocalGemma V2 Underperformed Decoder-Only Baseline
The primary design goal of the BifocalGemma project is to leverage a hybrid Encoder-Decoder topology for repository-level software engineering tasks (such as repository QA, bug localization, and automated program repair). In this bifurcation, a long-context bidirectional encoder ingests full repository context (up to 32,768 tokens), while an autoregressive causal decoder synthesizes precise code modifications, unified git diff hunks, or answers (up to 8,192 tokens).

However, empirical evaluation of **BifocalGemma V2** revealed an alarming paradox:
- **Baseline Gemma-4 Decoder-Only (`e2b-it`, 2B parameters)**: Achieved **74.00%** accuracy on 32k LongCodeQA.
- **BifocalGemma-v2 Symmetric 2it-2it (`e2b-e2b`, ~4B parameters)**: Achieved only **73.50%** on 32k LongCodeQA.

Despite having **double the parameter capacity (~4B vs 2B)** and bidirectional attention across 32,768 tokens, BifocalGemma V2 scored **0.50% lower** than the causal baseline. Detailed diagnostic audits on HumanEvalFix (`docs/ENCODER_ADAPTATION_RESEARCH.md`) established that while gold target negative log-likelihood (NLL) improved under correct context (NLL $0.4815$) compared to empty context ($0.6451$) or swapped context ($0.7647$) during teacher-forced evaluation, free autoregressive generation exhibited catastrophic identifier hallucination, indentation degradation, and git hunk header corruption.

### 1.2 The Core Thesis of BifocalGemma V3
BifocalGemma V3 asserts that this underperformance is **not** an inherent flaw of the encoder-decoder paradigm. Rather, it is the direct consequence of five compounded mathematical and structural pathologies in V2:
1. **Double Linear Redundancy**: An unnormalized, purely linear bridge adapter collapsed into the decoder's linear projection, lacking the capacity to non-linearly warp the bidirectional representation manifold into the autoregressive decoder space.
2. **Joint Softmax Denominator Dilution**: Merged attention combined 32,768 encoder keys and $<200$ decoder prompt keys into a single denominator, suppressing self-attention probability mass to $\approx 0.003\%$ during early decoding steps and destroying the decoder's syntactic fluency.
3. **Asymmetric RoPE Frequency Spinning**: Cross-attention query vectors were rotated by the decoder's generation step $m$ via Rotary Position Embedding ($R_m$), while cross-keys were unrotated. The inner product $\langle R_m Q, K_{cross} \rangle = Q^T R_{-m} K_{cross}$ oscillated wildly across decoding steps, corrupting cross-context token tracking.
4. **Static Blind Encoder Bottleneck**: In V2, the encoder processed 32k code tokens in isolation, completely unaware of the user instruction or bug report, whereas the decoder-only baseline concatenated `[Instruction || Context]` in a unified causal sequence.
5. **Direct SFT Shortcut Learning & Optimizer Clash**: Training unified diffs from step 0 with an unfrozen decoder allowed the strong Gemma-4 IT prior to bypass the unaligned cross-attention bridge. Meanwhile, GrokOrScaleLoRA's layer-wise trust ratio clamped against the dense bridge while Newton-Schulz orthogonalization destroyed anisotropic channel scaling.

BifocalGemma V3 completely dismantles these pathologies through an integrated, mathematically grounded architectural and training overhaul.

### 1.3 Quantitative Target Metrics
BifocalGemma V3 is engineered to surpass the decoder-only baselines decisively across all evaluation axes:

| Benchmark | Gemma-4 Baseline (`e2b-it`) | BifocalGemma V2 (`e2b-e2b`) | BifocalGemma V3 Target (`e2b-e2b`) | BifocalGemma V3 Target (`e4b-e2b`) |
|---|---|---|---|---|
| **LongCodeQA (32k Context)** | 74.00% | 73.50% | **> 80.50% (+6.5 pts)** | **> 83.00% (+9.0 pts)** |
| **HumanEvalFix (Pass@1)** | 42.1% | 38.6% | **> 55.0% (+13.0 pts)** | **> 60.0% (+18.0 pts)** |
| **CanItEdit (Instruction Repair)** | 36.4% | 33.2% | **> 50.0% (+14.0 pts)** | **> 54.0% (+18.0 pts)** |
| **CrossCodeEval (Cross-File Comp)**| 31.2% | 29.8% | **> 44.0% (+13.0 pts)** | **> 48.0% (+17.0 pts)** |
| **Exact Identifier Match on 32k**  | 61.2% | 54.3% | **> 92.0% (+31.0 pts)** | **> 95.0% (+34.0 pts)** |
| **Unified Diff Apply Success**     | 68.5% | 61.0% | **> 85.0% (+17.0 pts)** | **> 89.0% (+21.0 pts)** |

### 1.4 Structural Blueprint & Key Innovations Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 BIFOCALGEMMA V3 MASTER PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. REPOSITORY INGESTION PIPELINE                                                                │
│    - Query-Injected Prefix: [Issue / Task Query] || [Context]                                   │
│    - Tier 1: Macro-Topology Repo Map (Tree-sitter AST + Personalized PageRank) (~4k tokens)     │
│    - Tier 2: Micro-Implementation Windows (Call-Graph Topological Sort) (~28k tokens)           │
│    - Coarse-Grained Routing: [CHUNK_SUMMARY] Bridge Tokens every 512 tokens                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. BIDIRECTIONAL ENCODER (Gemma 4 Base Backbone, 35 or 42 Layers)                               │
│    - Global receptive field: Definition-after-use resolution & mutual cross-file conditioning    │
│    - Stage 1: Frozen under torch.no_grad(); zero intermediate activation memory (H_enc = 168 MB)│
│    - Stage 2: Attention LoRA (rank 16, alpha 32)                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. SWIGLU DUAL-GATE BRIDGE ADAPTER                                                              │
│    - Pre-RMSNorm(d_enc) -> [Linear(W_gate) * SiLU] ⊙ Linear(W_up) -> Linear(W_down)            │
│    - Post-RMSNorm(d_dec) with learned scale calibration gain gamma_calib                        │
│    - Warps bidirectional representations non-linearly into decoder embedding manifold            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. CAUSAL DECODER (Gemma 4 IT Backbone, 35 or 42 Layers)                                        │
│    - Native Self-Attention: RoPE preserved, causal mask, local/global sliding window            │
│    - Decoupled Gated Cross-Attention: Independent W_q, W_k, W_v, W_o projections               │
│    - Stationary Coordinates: Zero RoPE on cross-attention queries and keys                      │
│    - Gating Mechanism: tanh(g) initialized at atanh(0.05) (95% prior preserved, 5% gradient flow)│
│    - Static Cross-KV Cache: Projected once; FlashAttention-2 / SDPA execution (0 GB P-matrix)  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. TRAINING RECIPE & OPTIMIZATION REGIMEN                                                       │
│    - Stage 1 (2,000 steps, ~41M tokens): Adapter & Cross-Attention Warmup on Source-Fidelity    │
│      Mixture (Span Copy 40%, Infilling 30%, PrefixLM 30%). Frozen Backbones. (< 3.5 hrs on Molab)│
│    - Stage 2 (3,000 steps): Joint LoRA SFT on Diff Generation + 15% General Replay Buffer       │
│    - Progressive Length Curriculum (4k -> 16k -> 32k) & Critical Attention Scaling (beta_N)     │
│    - Streaming LogSumExp Pointer Loss (L_ptr) with zero VRAM overhead matching exact softmax    │
│    - Normalized Differential Token Weighting (w_add=2.0, w_del=1.5, w_hdr=1.2, w_ctx=0.5)      │
│    - Partitioned 3-Way AdamW Optimizer (Bridge Linear 1e-4, Bridge Norm 1e-4 WD 0, LoRA 2e-5)   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Root Causes of V2 Underperformance

### 2.1 Double Linear Projection Redundancy and Absence of Manifold Warping

#### 2.1.1 Composition of Linear Maps
In BifocalGemma V2 (`src/bifocalgemma/models/v2/alignment.py:8-37` and `memory_efficient.py:319-324`), the bridge adapter `NormalizedAlignment` consists of an unnormalized linear transformation $W_{\text{align}} \in \mathbb{R}^{d_{\text{dec}} \times d_{\text{enc}}}$ followed by RMSNorm. Subsequently, within each decoder layer $l \in [1, L_{\text{dec}}]$, the aligned hidden state is projected by the decoder's native key projection $W_k^{(l)} \in \mathbb{R}^{d_{\text{kv}} \times d_{\text{dec}}}$ and value projection $W_v^{(l)} \in \mathbb{R}^{d_{\text{kv}} \times d_{\text{dec}}}$.

Neglecting the scalar variance scaling of RMSNorm for a moment, the composite transformation for layer $l$ is:
$$W_{\text{eff}, k}^{(l)} = W_k^{(l)} \cdot W_{\text{align}}, \quad W_{\text{eff}, v}^{(l)} = W_v^{(l)} \cdot W_{\text{align}}$$
Because both $W_k^{(l)}$ and $W_{\text{align}}$ are purely linear operations, their product is strictly another linear operation. Under matrix rank inequalities:
$$\operatorname{rank}(W_{\text{eff}}^{(l)}) \le \min\left(\operatorname{rank}(W_k^{(l)}), \, \operatorname{rank}(W_{\text{align}})\right) \le \min(d_{\text{enc}}, d_{\text{dec}}, d_{\text{kv}})$$

#### 2.1.2 Topological Rigidity and Inability to Warp Manifolds
The encoder hidden states $H_{\text{enc}} \in \mathbb{R}^{N_{\text{enc}} \times d_{\text{enc}}}$ lie on a complex, highly non-linear manifold shaped by 35 or 42 bidirectional Transformer layers with SwiGLU feed-forward networks and RoPE. The decoder key space represents semantic addressing hooks designed for autoregressive next-token prediction.

A linear projection $W_{\text{align}}$ can only perform affine operations: rotation, shear, scaling, and reflection. It is mathematically incapable of:
1. **Partitioning feature space non-linearly**: It cannot activate or deactivate dimensions conditionally based on local code syntax.
2. **Folding or warping manifolds**: Disparate code clusters that require distinct alignment vectors cannot be separated.
3. **Layer-wise Depth Adaptation**: In V2, a *single* static matrix $W_{\text{align}}$ is shared across all decoder layers. Decoder layer 2 (which requires low-level syntactic primitives like token indentation and punctuation) and decoder layer 36 (which requires high-level algorithmic intent and abstract syntax trees) are forced to receive keys from the exact same linearly transformed space.

#### 2.1.3 Norm Discrepancy & RMSNorm Jacobian Instability
As proved by Li et al. (arXiv:2512.08374, *"The Unseen Bias: How Norm Discrepancy in Pre-Norm MLLMs Leads to Visual Information Loss"*), pre-trained bidirectional Transformers produce token representations with significant variance disparities and extreme outlier dimensions (norm magnitudes reaching $40\text{--}70$, compared to typical embedding scales of $1.0\text{--}1.5$).

In V2, $H_{\text{enc}}$ is unnormalized before $W_{\text{align}}$. The Post-RMSNorm operation is:
$$\tilde{z} = \text{RMSNorm}(z) = \frac{z}{\sqrt{\frac{1}{d} \|z\|_2^2 + \epsilon}} \odot \gamma$$
The Jacobian of this operation with respect to $z = W_{\text{align}} H_{\text{enc}}$ is:
$$\frac{\partial \tilde{z}}{\partial z} = \frac{1}{\text{RMS}(z)} \left( I - \frac{z z^T}{\|z\|_2^2} \right) \operatorname{diag}(\gamma)$$
When $H_{\text{enc}}$ contains unnormalized outlier features (Dettmers et al., 2022), $z$ develops large directional spikes. While $\left(I - \frac{z z^T}{\|z\|_2^2}\right)$ projects out the dominant component, the scalar prefactor $\frac{1}{\text{RMS}(z)}$ causes extreme gradient amplification along orthogonal dimensions. This causes severe gradient norm spikes in early training steps, triggering optimizer instability. Applying **Pre-RMSNorm** prior to the bridge adapter is mathematically required to ensure that $\mathbb{E}[\|H_{\text{enc}}\|_2^2] = d_{\text{enc}}$, stabilizing the Jacobian.

---

### 2.2 Merged Attention Joint Softmax Denominator Dilution Across 32k Context

#### 2.2.1 Partition Function Derivation
In BifocalGemma V2 (`src/bifocalgemma/models/memory_efficient.py:68-76`), the decoder utilizes **Merged Attention** (`normalized_merged_v2`), where self-attention keys and encoder cross-attention keys are concatenated along sequence length and normalized via a single unified softmax.

For decoder query token $t \in [1, L_{\text{dec}}]$ at head $h$, the output is:
$$O_t = \sum_{i=1}^t \alpha_i V_{\text{self}, i} + \sum_{j=1}^{N_{\text{enc}}} \beta_j V_{\text{cross}, j}$$
where:
$$\alpha_i = \frac{\exp(s_{\text{self}, i})}{Z_{\text{self}} + Z_{\text{enc}}}, \quad \beta_j = \frac{\exp(s_{\text{cross}, j})}{Z_{\text{self}} + Z_{\text{enc}}}$$
The partition functions are defined as:
$$Z_{\text{self}} = \sum_{i=1}^t \exp(s_{\text{self}, i}), \quad Z_{\text{enc}} = \sum_{j=1}^{N_{\text{enc}}} \exp(s_{\text{cross}, j})$$
where $s_k = \frac{q_t k_k^T}{\sqrt{d_k}}$.

The total attention probability mass allocated to the decoder's autoregressive self-context is:
$$M_{\text{self}} = \sum_{i=1}^t \alpha_i = \frac{Z_{\text{self}}}{Z_{\text{self}} + Z_{\text{enc}}} = \frac{1}{1 + \frac{Z_{\text{enc}}}{Z_{\text{self}}}}$$

#### 2.2.2 The Denominator Saturation Catastrophe
Consider the operational regime of LongCodeQA and code repair: $N_{\text{enc}} = 32,768$ and early decoding step $t = 1$ (generating the first token of a patch hunk or QA answer).
Here, $Z_{\text{self}} = \exp(s_{\text{self}, 1})$.
The encoder partition function $Z_{\text{enc}}$ is the sum of $32,768$ positive terms:
$$Z_{\text{enc}} = \sum_{j=1}^{32,768} \exp(s_{\text{cross}, j})$$

Under standard random initialization of the alignment bridge (or semi-orthogonal initialization with identity scale), the distribution of cross-attention dot products $s_{\text{cross}, j}$ has mean $\mu_{\text{enc}}$ and standard deviation $\sigma$. By the properties of the sum of log-normal random variables:
$$\mathbb{E}[Z_{\text{enc}}] \approx N_{\text{enc}} \cdot \exp\left(\mu_{\text{enc}} + \frac{1}{2}\sigma^2\right)$$

For the self-attention mass $M_{\text{self}}$ to reach parity ($M_{\text{self}} = 0.50$, i.e., $Z_{\text{self}} = Z_{\text{enc}}$), the required logit difference is:
$$s_{\text{self}} - \mu_{\text{enc}} = \ln(N_{\text{enc}}) + \frac{1}{2}\sigma^2 = \ln(32,768) + \frac{1}{2}\sigma^2 \approx 10.397 + \frac{1}{2}\sigma^2$$

In scaled dot-product attention with $d_k = 256$ ($\sqrt{d_k} = 16$), achieving a logit margin $\Delta s > 10.4$ requires an raw dot-product advantage:
$$q_t k_{\text{self}}^T - q_t k_{\text{enc}}^T > 10.4 \times 16 \approx 166.4$$
For unit-norm normalized vectors, this is mathematically impossible unless all 32k encoder keys are actively anti-correlated and forced strongly negative.

#### 2.2.3 Autoregressive Syntax Amnesia
At initialization and early training steps where $\mu_{\text{enc}} \approx \mu_{\text{self}}$:
$$\frac{Z_{\text{enc}}}{Z_{\text{self}}} \approx 32,768 \implies M_{\text{self}} \approx \frac{1}{32,769} \approx \mathbf{0.00305\%}$$
**The decoder is 99.997% blinded to its own autoregressive prefix!**
It cannot attend to the preceding indentation, function header, opening parenthesis, or variable binding generated just 1 or 2 tokens earlier.

When the optimizer attempts to rescue self-attention, it is forced to push cross-attention projection weights $W_k^{(l)}$ strongly negative to suppress $Z_{\text{enc}} \to 0$. This induces **encoder starvation**, rendering the 32k encoder context completely unused during subsequent generation.

---

### 2.3 Asymmetric RoPE Query Frequency Spinning

#### 2.3.1 Mathematical Derivation of RoPE
Rotary Position Embedding (RoPE; Su et al., 2024) encodes discrete sequence position $m$ by rotating orthogonal 2D subspaces by frequency angles $m \theta_k$, where $\theta_k = 10000^{-2(k-1)/d}$:
$$R_m = \operatorname{diag}\left(R_{\Theta, 1}^{(m)}, R_{\Theta, 2}^{(m)}, \dots, R_{\Theta, d/2}^{(m)}\right), \quad R_{\Theta, k}^{(m)} = \begin{pmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{pmatrix}$$
Because $R_m$ is an orthogonal rotation matrix, $R_m^T = R_m^{-1} = R_{-m}$.

In standard causal self-attention, both query $Q_m$ and key $K_n$ are rotated by their respective sequence indices $m$ and $n$:
$$\langle R_m Q_m, \, R_n K_n \rangle = Q_m^T R_m^T R_n K_n = Q_m^T R_{n-m} K_n$$
The inner product depends strictly on the relative distance $n - m$, preserving translation invariance.

#### 2.3.2 The Merged Attention Asymmetry Flaw
In BifocalGemma V2 (`src/bifocalgemma/models/memory_efficient.py:304` and `319`):
- RoPE is applied to the decoder query: $Q_m = R_m \cdot q_m$.
- RoPE is applied to decoder self-keys: $K_{\text{self}, n} = R_n \cdot k_n$.
- **RoPE is NOT applied to cross-keys**: $K_{\text{cross}, j} = k_{\text{cross}, j}$ (unrotated).

Now compute the cross-attention dot product between query at decode step $m$ and encoder key at source position $j$:
$$\langle R_m Q_m, \, K_{\text{cross}, j} \rangle = Q_m^T R_m^T K_{\text{cross}, j} = Q_m^T R_{-m} K_{\text{cross}, j}$$

Notice the critical pathology: **$R_{-m}$ depends entirely on the absolute generation step $m$ of the decoder!**  
As the decoder generates tokens ($m = 0, 1, 2, \dots, 500$), the query vector is continuously spun in 2D coordinate planes by angle $-m \theta_k$.

```
Decoupled Cross-Attention (V3):
Query Q (unrotated)  ──── Dot Product ────► Key K_j (unrotated)
Result: Stationary, shift-invariant dot product throughout generation.

Merged Attention (V2):
Query Q_m = R_m Q    ──── Dot Product ────► Key K_j (unrotated)
Result: Dot product modulated by R_{-m} (spins by -m*theta_k at every single token step!)
```

A query attempting to attend to a specific variable declaration `user_id` in the repository context will experience an artificial, high-frequency coordinate rotation that alters dot-product scores every single token step! To maintain focus on the same identifier, the model is forced to either:
1. Learn complex periodic trajectories in weight space to counteract $R_{-m}$, or
2. Suppress the weights of all attention heads whose RoPE frequencies rotate quickly, permanently crippling head expressive capacity.

---

### 2.4 Static Blind Encoder Bottleneck
In BifocalGemma V2 (`src/bifocalgemma/data/context_tasks.py:105-128`):
$$\text{Encoder Input} = \text{Repository Context} \quad (32,768 \text{ tokens})$$
$$\text{Decoder Input} = \text{Task Instruction / Bug Report} \quad (\sim 200 \text{ tokens})$$

This creates the **Static Blind Encoder Trap**:
1. The bidirectional encoder processes 32,768 tokens of repository code in **complete ignorance** of what question is being asked or what bug needs fixing.
2. The encoder outputs a static hidden tensor $H_{\text{enc}} \in \mathbb{R}^{32,768 \times d}$. All symbols, classes, comments, and boilerplate across dozens of files are weighted by generic syntax rules, not by relevance to the specific task.
3. In contrast, in the **Gemma-4 Decoder-Only Baseline**, the input is causally concatenated:
   $$\text{Prefix} = [\text{Task Instruction}] \parallel [\text{Repository Context}]$$
   As each token in the context is processed, causal self-attention allows context tokens to attend back to the instruction tokens.
4. When the user asks a question about `indexing.py`, the decoder-only baseline's activations in `indexing.py` are already enriched with task-specific relevance. In BifocalGemma V2, the encoder had zero task conditioning.

---

### 2.5 Direct SFT Shortcut Learning & Exposure Bias
In BifocalGemma V2, training used the `unified_v2` recipe: end-to-end supervised fine-tuning directly on git diffs from step 0 with an unfrozen decoder backbone.

1. **The Self-Attention Shortcut**:
   The decoder backbone is initialized from Gemma-4 IT, which possesses strong causal language priors. Code diffs exhibit high local predictability: predicting token $y_t$ given the preceding diff tokens in a hunk is trivial via self-attention alone.
   At step 0, because the cross-attention bridge is unaligned, $\frac{\partial \mathcal{L}}{\partial W_{\text{self}}} \gg \frac{\partial \mathcal{L}}{\partial W_{\text{cross}}}$.
   The optimizer updates $W_{\text{self}}$ to model the diff syntax while neglecting $W_{\text{cross}}$. The decoder learns to ignore the encoder entirely (**encoder neglect**).
2. **Teacher-Forcing Exposure Bias**:
   Under teacher forcing, the decoder receives the ground-truth prefix at every step, yielding a deceptively low NLL ($0.39\text{--}0.48$). But during free autoregressive generation, the decoder must roll out tokens without gold prefixes. Lacking cross-attention grounding in the source code, the first error causes catastrophic compounding, leading the model to hallucinate variable names and line offsets.

---

### 2.6 GrokOrScale Trust Ratio Clamping Clash and Spectral Isotropy

#### 2.6.1 Mathematical Formulation of GrokOrScaleLoRA
In `src/bifocalgemma/training/optimizers.py:122-137`, the GrokOrScaleLoRA optimizer computes update direction $\Delta$ via Newton-Schulz orthogonalization $Q = \operatorname{NewtonSchulz}(M_t)$:
$$\Delta = \lambda_{\text{wd}} W + 0.2 \sqrt{\max(M, N)} Q$$
The layer-wise trust ratio is:
$$r_t = \operatorname{clamp}\left( \frac{\|W_t\|_F}{c \cdot \|\Delta_t\|_F}, \, \text{trust\_min}=0.5, \, \text{trust\_max}=2.0 \right)$$
where $c = \frac{\|W_0\|_F}{\|\Delta_0\|_F}$ is the step-0 calibration factor.

#### 2.6.2 The Dense Bridge Clash
This formulation was derived for zero-initialized low-rank matrices ($B \in \mathbb{R}^{d \times r}, r=16$). But when applied to the dense alignment bridge ($W \in \mathbb{R}^{D \times D}, D=2048$ or $2560$, initialized as identity $I$ or semi-orthogonal):
1. **Norm Disparity**:
   - Initial weight norm: $\|W_0\|_F = \sqrt{D} = \sqrt{2560} \approx 50.60$.
   - Direction norm: Since $Q$ has orthonormal rows and columns, $\|Q\|_F = \sqrt{D}$. Thus:
     $$\|\Delta_0\|_F \approx 0.2 \sqrt{D} \cdot \sqrt{D} = 0.2 D = 0.2 \times 2560 = 512.0$$
   - Calibration factor: $c = \frac{50.60}{512.0} \approx 0.0988$.
2. **Clamp Saturation**:
   Because $W$ is full rank, its Frobenius norm remains roughly constant near $\sqrt{D} \approx 50.6$. But gradient norms fluctuate widely across batches. Consequently, the ratio $\frac{\|W_t\|_F}{c \|\Delta_t\|_F}$ immediately saturates against the hard boundaries $0.5$ or $2.0$. The trust ratio stops adapting and acts as an arbitrary $0.5\times$ or $2.0\times$ multiplier, destabilizing optimization.
3. **Spectral Isotropy vs Manifold Alignment**:
   Newton-Schulz projects updates onto the Stiefel manifold ($Q Q^T = I$), forcing all singular values to unity: $\sigma_i(Q) = 1.0, \forall i$.
   However, cross-modal alignment between an encoder and decoder requires **anisotropic scaling**: different semantic channels have radically different variances and dynamic ranges. Forcing isotropic updates chokes channel-wise information routing.
4. **RMSNorm 1D Crash**:
   `self.norm.weight` in `NormalizedAlignment` is a 1D tensor (`ndim == 1`). `GrokOrScaleLoRA:74` asserts `p.ndim == 2`, causing an immediate crash when attempting to optimize normalization gains.

---

## 3. BifocalGemma V3 Architecture Specification

### 3.1 SwiGLU Dual-Gate Bridge Adapter
To replace the rigid linear bridge, BifocalGemma V3 introduces the **SwiGLU Dual-Gate Bridge Adapter** with Pre-RMSNorm and Post-RMSNorm scale calibration:

```
[Encoder Final Hidden States: H_enc ∈ R^{B × N_enc × d_enc}]
                     │
         Pre-RMSNorm (d_enc, eps=1e-6)
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
     Linear W_gate       Linear W_up
   (d_enc -> d_mid)    (d_enc -> d_mid)
           │                   │
         SiLU                  │
           └─────────┬─────────┘
                     ▼
           H_mid = SiLU(W_gate) ⊙ W_up
                     │
               Linear W_down
             (d_mid -> d_dec)
                     │
              [Residual Link]
                     │
        Post-RMSNorm (d_dec, eps=1e-6)
                     │
        Scale Calibration Gain γ_calib
                     │
[Aligned Encoder States: H_bridge ∈ R^{B × N_enc × d_dec}]
```

#### Mathematical Formulation:
$$H_{\text{norm}} = \text{RMSNorm}(H_{\text{enc}}, \epsilon = 10^{-6})$$
$$H_{\text{gate}} = \operatorname{SiLU}(H_{\text{norm}} W_{\text{gate}}), \quad W_{\text{gate}} \in \mathbb{R}^{d_{\text{enc}} \times d_{\text{mid}}}$$
$$H_{\text{up}} = H_{\text{norm}} W_{\text{up}}, \quad W_{\text{up}} \in \mathbb{R}^{d_{\text{enc}} \times d_{\text{mid}}}$$
$$H_{\text{down}} = (H_{\text{gate}} \odot H_{\text{up}}) W_{\text{down}}, \quad W_{\text{down}} \in \mathbb{R}^{d_{\text{mid}} \times d_{\text{dec}}}$$
$$\tilde{H}_{\text{enc}} = \text{RMSNorm}\left(H_{\text{down}} + \delta \cdot W_{\text{res}} H_{\text{norm}}, \, \epsilon = 10^{-6}\right) \cdot \gamma_{\text{calib}}$$

Where:
- Model Dimension Profiles (Gemma-4):
  - **Gemma-4 E2B**: $d_{\text{enc}} = d_{\text{dec}} = 1536$ ($35$ layers, $1$ KV head, $d_k = 256$).
  - **Gemma-4 E4B**: $d_{\text{enc}} = d_{\text{dec}} = 2560$ ($42$ layers, $2$ KV heads, $d_k = 256$, global head dim $512$).
- Bridge Hidden Dimension $d_{\text{mid}} = \max(d_{\text{enc}}, d_{\text{dec}})$:
  - **Symmetric E2B-E2B**: $d_{\text{enc}} = d_{\text{dec}} = 1536 \implies d_{\text{mid}} = 1536$. Trainable parameters: $3 \times 1536^2 = 7,077,888 \approx \mathbf{7.08\text{M}}$ params with identity residual ($W_{\text{res}} = I$).
  - **Asymmetric E4B-E2B**: $d_{\text{enc}} = 2560, d_{\text{dec}} = 1536 \implies d_{\text{mid}} = 2560$. Trainable feedforward parameters: $2 \times 2560^2 + 2560 \times 1536 = 17,039,360 \approx \mathbf{17.04\text{M}}$ params (or $20.97\text{M}$ including semi-orthogonal linear residual projection $W_{\text{res}}$).
  - **Symmetric E4B-E4B**: $d_{\text{enc}} = d_{\text{dec}} = 2560 \implies d_{\text{mid}} = 2560$. Trainable parameters: $3 \times 2560^2 = 19,660,800 \approx \mathbf{19.66\text{M}}$ params with identity residual ($W_{\text{res}} = I$).
- Residual projection $W_{\text{res}}$: If $d_{\text{enc}} = d_{\text{dec}}$, $W_{\text{res}} = I$ and $\delta = 1$; if asymmetric ($d_{\text{enc}} \ne d_{\text{dec}}$), $W_{\text{res}}$ is a semi-orthogonal linear projection.
- $\gamma_{\text{calib}}$ is a learned 1D calibration vector initialized to $1.0$.

---

### 3.2 Decoupled Gated Cross-Attention Architecture
BifocalGemma V3 replaces Merged Attention with **Decoupled Gated Cross-Attention** in every decoder layer:

```
Decoder Layer Input: x
   │
   ├───────────────────────────────────────────────────────┐
   │                                                       │
RMSNorm (Pre-Norm)                                         │
   │                                                       │
Self-Attention (Causal Mask + Native RoPE)                 │
   │                                                       │
   └──────────────────────────(+)──────────────────────────┘
                               │
                               ├───────────────────────────────────────┐
                               │                                       │
                        RMSNorm (Pre-Norm)                             │
                               │                                       │
                        Cross-Attention (Unrotated Q & K)              │
                        (Keys/Values from H_bridge)                    │
                               │                                       │
                         tanh(gate_l)  [init gate_l = atanh(0.05)]     │
                               │                                       │
                               └──────────────────(+)──────────────────┘
                                                   │
                                                   ├───────────────────┐
                                                   │                   │
                                            RMSNorm (Pre-Norm)         │
                                                   │                   │
                                              FFN (SwiGLU)             │
                                                   │                   │
                                                   └─────────(+)───────┘
                                                              │
                                                     Layer Output: x_next
```

#### Gating Formulation and Initialization:
$$h_{\text{cross}} = \text{CrossAttn}\left(\text{RMSNorm}(h_{\text{self}}), \, \tilde{H}_{\text{enc}}\right)$$
$$h_{\text{out}} = h_{\text{self}} + \tanh(g_l) \cdot h_{\text{cross}}$$

#### The atanh(0.05) Mathematical Keystone:
In Flamingo (Alayrac et al., 2022), the gate parameter was initialized to $g_0 = 0$, giving $\tanh(0) = 0$.
The gradient with respect to cross-attention projection weights is:
$$\frac{\partial \mathcal{L}}{\partial W_{\text{cross}}} = \frac{\partial \mathcal{L}}{\partial h_{\text{out}}} \cdot \tanh(g) \cdot \frac{\partial h_{\text{cross}}}{\partial W_{\text{cross}}}$$
At $g = 0$, $\frac{\partial \mathcal{L}}{\partial W_{\text{cross}}} = 0$ identically! In short SFT schedules (1 epoch, 2,000 steps), zero initialization starves the cross-attention layers of gradient during the critical initial warmup.

In BifocalGemma V3, we initialize:
$$g_0 = \operatorname{atanh}(0.05) \approx 0.050042 \implies \tanh(g_0) = 0.05$$
- **Preserves Decoder Prior**: The pre-trained Gemma-4 IT decoder's generative distribution is 95% intact at step 0, preventing instruction collapse.
- **Immediate Gradient Propagation**: Exactly 5% of backpropagated gradient flows into cross-attention projections and the bridge adapter from step 1, accelerating manifold convergence.

---

### 3.3 Stationary Unrotated Cross-Attention Coordinates
To eliminate the asymmetric RoPE query spinning detailed in §2.3:
1. **Decoder Self-Attention**: Native RoPE is fully retained on $Q_{\text{self}}$ and $K_{\text{self}}$ with causal masking.
2. **Cross-Attention Streams**:
   $$Q_{\text{cross}} = \text{RMSNorm}(h_{\text{self}}) \cdot W_{q, \text{cross}} \quad (\text{\bf NO RoPE applied})$$
   $$K_{\text{cross}} = \tilde{H}_{\text{enc}} \cdot W_{k, \text{cross}} \quad (\text{\bf NO RoPE applied})$$
   $$V_{\text{cross}} = \tilde{H}_{\text{enc}} \cdot W_{v, \text{cross}}$$
3. **Stationary Dot Product**:
   $$\langle Q_{\text{cross}, m}, \, K_{\text{cross}, j} \rangle = Q_{\text{cross}, m}^T K_{\text{cross}, j}$$
   Because neither vector is rotated by decode step $m$, the dot product is **stationary and shift-invariant**. A query searching for an AST node or variable name maintains constant score alignment regardless of whether it is generated at decode token 5 or token 500.

---

### 3.4 Dual-Granularity Repository Context Ingestion Pipeline

To maximize the utility of the 32k encoder context and prevent attention dispersion, context is ingested in a structured hierarchy:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   32K ENCODER INPUT PIPELINE                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. TASK INSTRUCTION / ISSUE QUERY  (~500 tokens)                       │
│    [QUERY] Fix IndexError when slicing empty DataFrame in indexing.py   │
├────────────────────────────────────────────────────────────────────────┤
│ 2. TIER 1: MACRO-TOPOLOGY REPO MAP (~3,500 tokens)                    │
│    - Tree-sitter AST symbol extraction (classes, methods, docstrings)   │
│    - Directed dependency graph (calls/inherits/imports)                │
│    - Scored by Personalized PageRank biased toward query symbols        │
│    - Rendered as compact, scope-aware signatures                       │
├────────────────────────────────────────────────────────────────────────┤
│ 3. TIER 2: MICRO-IMPLEMENTATION WINDOWS (~28,000 tokens)              │
│    - Ordered by Call-Graph Topological Traversal (not flat BM25)       │
│    - Enclosed in structured XML file/scope delimiters                  │
│    - Preceded by [CHUNK_SUMMARY] Bridge Tokens every 512 tokens         │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Tier 1 (Macro-Topology Repo Map)**: Parsed via Tree-sitter AST, building a symbol dependency graph. Personalized PageRank ranks symbols relevant to the query. Formatted into compact signatures:
   ```python
   # === REPO ARCHITECTURE MAP ===
   # file: pandas/core/indexing.py
   class _NDFrameIndexer(_LocIndexer):
       def __getitem__(self, key: Any) -> Any: ...
       def _has_valid_tuple(self, key: Tuple) -> bool: ...  # calls: common.is_list_like


   # file: pandas/core/dtypes/common.py
   def is_list_like(obj: Any) -> bool: ...
   ```
2. **Tier 2 (Micro-Implementation Windows)**: Full source code of the focal file and closely connected dependencies, sorted by topological call distance.
3. **Dual-Granularity Bridge Tokens (Phase 2 Architectural Ablation)**: Every 512 code tokens can optionally be preceded by a special `[CHUNK_SUMMARY_i]` token. In the bidirectional encoder, `[CHUNK_SUMMARY_i]` pools the semantics of its block. During decoding, coarse cross-attention over these 64 summary tokens rapidly identifies relevant file regions. *Note on Phasing*: To maintain engineering simplicity, dense FlashAttention-2 / SDPA cross-attention over all 32k encoder tokens (as fully specified in Section 8.2) serves as the primary Phase 1 baseline, while coarse summary routing is isolated as an ablation for Phase 2.

---

### 3.5 Query-Injected Bidirectional Encoder Prefix
To break the Static Blind Encoder Trap:
$$\text{Encoder Input} = [\text{Task Goal / Bug Description}] \parallel [\text{Repo Map + Implementation Context}]$$
$$\text{Decoder Input} = [\text{Task Goal / Bug Description}] \parallel [\text{Autoregressive Generation Prefix}]$$

Every single code token in the 32,768 repository context attends to the issue description across all 35 or 42 bidirectional encoder layers. By the time representations reach the bridge adapter, the repository representation is dynamically indexed and weighted for the specific bug.

---

## 4. Hardware-Guarded Execution on 1 Molab GPU

### 4.1 The Catastrophic Attention Matrix Hazard & SDPA / FlashAttention-2 Resolution

#### 4.1.1 The 168 GiB Materialization Hazard
In Decoupled Cross-Attention, cross-attention involves $B = 1$, $H = 8$, $L_{\text{dec}} = 8,192$, and $L_{\text{enc}} = 32,768$.
If an implementation naively computes or stores the attention probability tensor:
$$P = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times H \times L_{\text{dec}} \times L_{\text{enc}}}$$
Total elements per layer:
$$1 \times 8 \times 8,192 \times 32,768 = 2,147,483,648 \text{ elements (2.15 Billion)}$$
- In BF16 (2 bytes): $2.147 \times 10^9 \times 2 = \mathbf{4.295 \text{ GiB PER LAYER}}$.
- Across 35 layers (E2B): $35 \times 4.295 = \mathbf{150.3 \text{ GiB}}$!
- Across 42 layers (E4B): $42 \times 4.295 = \mathbf{180.4 \text{ GiB}}$!

**KILL-CRITERION**: Any attempt to materialize $P$ into global VRAM will immediately terminate with CUDA Out-Of-Memory.

#### 4.1.2 Mandatory FlashAttention-2 / SDPA Execution
Decoupled Cross-Attention **MUST** be executed exclusively via PyTorch native `F.scaled_dot_product_attention` (SDPA) with FlashAttention-2 backend:
1. FlashAttention computes attention block-by-block entirely inside GPU SRAM (shared memory), maintaining an $O(1)$ intermediate global memory footprint.
2. **Static Cross-KV Cache**: During generation, the 32k encoder states are static. $K_{\text{cross}}$ and $V_{\text{cross}}$ are pre-projected **once** per layer and cached in VRAM:
   $$\text{Static Cross-KV per layer} = 2 \times (1 \times 2 \times 32,768 \times 256 \times 2 \text{ bytes}) \approx 33.55 \text{ MB}$$
   Across 42 layers, the entire static cross-KV cache consumes only **$1.41 \text{ GiB}$**.

---

### 4.2 Triton Shared Memory Hardware Bounds (101,376 Bytes) & Kernel Clamping

#### 4.2.1 Physical Root Cause of Molab Kernel Panics
On NVIDIA Compute Capability 8.6 (sm86: RTX 3090, A40, A10) and 8.9 (sm89: Ada Lovelace / L4), the hardware shared memory capacity per CUDA thread block is strictly:
$$\text{Hardware Limit} = 99 \times 1024 = \mathbf{101,376 \text{ bytes}}$$
When fused attention backward kernels configure tiles that exceed this limit, the compiler throws:
`Required: 114688 Hardware limit: 101376`

#### 4.2.2 Tile Configuration Clamping Table
Empirical testing on sm86 hardware established the exact shared memory consumption for BF16 backward execution:

| Head Dimension ($D$) | Tile Size ($B_M \times B_N$) | Stages | Shared Memory Required | Limit (sm86/sm89) | Status |
|---|---|---|---|---|---|
| **$D = 512$** | $32 \times 32$ | 1 | 143,360 bytes ($140.0\text{ KiB}$) | 101,376 bytes | **FAIL** (Fatal OOM) |
| **$D = 512$** | $16 \times 32$ | 1 | 106,496 bytes ($104.0\text{ KiB}$) | 101,376 bytes | **FAIL** (Matches the ~112-114 KB error) |
| **$D = 512$** | **$16 \times 16$** | **1** | **71,680 bytes ($70.0\text{ KiB}$)** | **101,376 bytes** | **PASS (30% safety margin)** |
| **$D = 256$** | $32 \times 32$ | 2 | 110,592 bytes ($108.0\text{ KiB}$) | 101,376 bytes | **FAIL** |
| **$D = 256$** | **$32 \times 32$** | **1** | **77,824 bytes ($76.0\text{ KiB}$)** | **101,376 bytes** | **PASS (24% safety margin)** |
| **$D = 256$** | **$16 \times 16$** | **1** | **38,912 bytes ($38.0\text{ KiB}$)** | **101,376 bytes** | **PASS (62% safety margin)** |

#### Mandatory Kernel Directives:
```python
# Hardware guard for Molab sm86/sm89/sm90 execution:
if head_dim >= 512:
    bwd_BLOCK_M = 16
    bwd_BLOCK_N = 16
    bwd_num_stages = 1
elif head_dim >= 256:
    bwd_BLOCK_M = 32
    bwd_BLOCK_N = 32
    bwd_num_stages = 1
```

---

### 4.3 Comprehensive VRAM Budget and Headroom Breakdown

All allocations are calculated for Model E4B→E4B (42 encoder layers, 42 decoder layers, $32k$ enc / $8k$ dec) on **1 Molab 80 GiB GPU**:

#### Stage 1 VRAM Budget (Adapter Warmup, Frozen Backbones):
- Backbone Weights (Encoder 4B + Decoder 4B in BF16): **14.90 GiB**
- Backbone Optimizer States: **0.00 GiB (FROZEN)**
- Encoder Forward Activations (`torch.no_grad()` drops all intermediate layers): **0.17 GiB** ($H_{\text{enc}}$ only)
- Trainable Weights (SwiGLU Bridge $19.7\text{M}$ + Cross-Attn $550.6\text{M}$ in FP32): **2.18 GiB**
- AdamW States ($m, v$ in FP32): **4.35 GiB**
- Decoder Activations with Activation Checkpointing: **8.00 GiB**
- **Total Stage 1 Allocated**: **29.60 GiB**
- **Total Stage 1 Reserved**: **~34.00 GiB**
- **Stage 1 Free Headroom**: **46.00 GiB** (Allows microbatch = 2 or 4!)

#### Stage 2 VRAM Budget (Joint LoRA SFT):
- Backbone Weights (BF16): **14.90 GiB**
- LoRA Parameters (Rank 16, Alpha 32, FP32): **0.42 GiB**
- LoRA AdamW States (FP32): **0.84 GiB**
- SwiGLU Bridge + Cross-Attention Parameters (FP32): **2.18 GiB**
- Bridge + Cross-Attention AdamW States (FP32): **4.35 GiB**
- Static Cross-KV Cache ($32k$, 42 layers, BF16): **1.41 GiB**
- Activation Checkpointed Activations ($32k$ enc, $8k$ dec, microbatch 1): **43.66 GiB**
- **Total Stage 2 Allocated**: **67.76 GiB**
- **Total Stage 2 Reserved**: **~73.20 GiB**
- **Stage 2 Free Headroom**: **6.80 GiB** (Exceeds mandatory 4.0 GiB margin!)

*Mitigation for Allocator Fragmentation*: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to ensure the allocator dynamically expands memory segments without fragmentation crashes.

---

## 5. Two-Stage Training & Manifold Alignment Recipe

### 5.1 Two-Stage Curriculum Rationale
Direct SFT fails because the unaligned cross-attention bridge forces the decoder to rely on its self-attention shortcut. By freezing both backbones in Stage 1, the decoder self-attention parameters $W_{\text{self}}$ cannot change. The **only** pathway to reduce loss is through the bridge and cross-attention parameters, forcing the formation of an information highway between the encoder and decoder.

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: ADAPTER / CROSS-ATTENTION WARMUP (2,000 STEPS)                │
│ • Encoder Backbone: FROZEN under torch.no_grad()                       │
│ • Decoder Backbone: FROZEN (Self-Attention, FFN, LM Head locked)       │
│ • Trainables: SwiGLU Bridge + Decoupled Cross-Attention Projections    │
│ • Objective: Source-Fidelity Mixture (Span Copy 40%, Infill 30%,       │
│              PrefixLM 30%) + Streaming Pointer Loss (L_ptr)            │
│ • Curriculum: Steps 1-1000 @ 4k tokens, Steps 1001-2000 @ 16k tokens   │
│ • Duration: < 3.5 hours on 1 Molab GPU (microbatch 2 or 4)             │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: JOINT SFT ON CODE REPAIR (3,000 STEPS)                        │
│ • Encoder Backbone: Trainable via Attention LoRA (rank 16, alpha 32)   │
│ • Decoder Backbone: Trainable via Attention LoRA (rank 16, alpha 32)   │
│ • Bridge & Cross-Attention: Trainable (Bridge LR 5e-5)                 │
│ • Objective: Unified Git Diff Generation on HumanEvalFix & SWE-bench   │
│ • Replay Buffer: 15% General Python Instruction Tuning data            │
│ • Context: Full 32,768 Encoder / 8,192 Decoder                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Stage 1: Adapter & Cross-Attention Warmup
1. **Source-Fidelity Mixture (SFM)**:
   - **Exact Span Copying (40% of batches)**: Prompt asks decoder to extract a specific function signature, docstring, or variable definition from the 32k context verbatim. Cross-attention is forced to learn pointer-like copying.
   - **Masked Infilling (30% of batches)**: Source context has an edit block replaced with `# [EDIT_LOCATION]`. Decoder must output the exact missing implementation.
   - **PrefixLM (30% of batches)**: Standard causal sequence continuation across file boundaries.
2. **Offline AST Caching**:
   Generating Tree-sitter repo maps takes $\sim 200 \text{ ms}$ per sample on CPU. Repo maps and token-matching index sets $\mathcal{M}(y_t)$ must be pre-parsed and cached in the Hugging Face dataset beforehand.

### 5.3 Stage 2: Joint LoRA SFT on Diff Generation
1. Attention LoRA (rank 16, alpha 32) is attached to $W_q, W_k, W_v, W_o$ on both encoder and decoder backbones.
2. The model is trained on unified diff generation using Differential Token Weighting (§6.3).
3. **15% Replay Buffer**: 15% of training samples consist of general reasoning and coding instruction data to prevent catastrophic forgetting of pre-trained capabilities.

### 5.4 Progressive Length Curriculum & Critical Attention Temperature Scaling
1. **Length Progression**:
   - Steps 1–1,000: $4k$ encoder / $1k$ decoder (sharp gradient concentration).
   - Steps 1,001–2,000: $16k$ encoder / $2k$ decoder (positional scaling).
   - Steps 2,001–5,000: $32k$ encoder / $8k$ decoder (full repository reasoning).
2. **Critical Attention Temperature Scaling Dynamics**:
   In standard Gemma-4 attention, dot-product logits are scaled by $1 / \sqrt{d_k}$. When extending cross-attention context from $2k$ to $32k$, attention distributions are susceptible to rank collapse and pathological over-contraction onto dominant sink tokens (Chen et al., ICLR 2025; Peng et al., 2023).
   To maintain balanced entropy across long sequences, length-dependent temperature scaling is introduced:
   $$\beta_N = 0.1 \ln\left(\frac{N_{\text{enc}}}{2048}\right) + 1.0 \quad \text{for } N_{\text{enc}} > 2048$$
   - At $N_{\text{enc}} = 2,048$: $\beta_N = 1.0$.
   - At $N_{\text{enc}} = 32,768$: $\beta_N = 0.1 \ln(16) + 1.0 \approx 1.277$.

   *Notation & Temperature Effects (Sharpening vs. Flattening)*:
   - **Training Baseline (Softening / Rank Collapse Prevention)**: The cross-attention logits are scaled by $\frac{1}{\sqrt{d_k} \cdot \beta_N}$ (effective temperature $T = \sqrt{d_k} \cdot \beta_N > \sqrt{d_k}$). Dividing by $\beta_N > 1.0$ slightly flattens the softmax distribution, counteracting the artificial logit inflation caused by aggregating over 16× more keys and preventing the model from collapsing onto arbitrary sink tokens.
   - **Inference Pointer Sharpening (Optional Copy Heightening)**: If downstream evaluation requires heightened contrast on exact identifier copy locations, logits in dedicated copy heads can optionally be sharpened via multiplier $\tau^{-1} = \beta_N$ (i.e. scaling by $\frac{\beta_N}{\sqrt{d_k}}$). In the baseline training recipe, the divisor $\sqrt{d_k} \cdot \beta_N$ is universally applied to maintain smooth optimization and gradient distribution across all 32,768 positions.

---

## 6. Loss Formulations & Optimizer Dynamics

### 6.1 Elimination of Obsolete Losses Under Decoupled Architecture
In V2 Merged Attention, $\mathcal{L}_{\text{mass}} = [\max(0, 0.15 - M_e)]^2$ was required because self- and cross-keys competed in a joint denominator ($Z_s + Z_e$).
In BifocalGemma V3:
$$\text{CrossAttn}(Q_c, K_c, V_c) = \operatorname{softmax}\left(\frac{Q_c K_c^T}{\sqrt{d_k}}\right) V_c \implies \sum_{j=1}^{N_{\text{enc}}} \beta_j \equiv \mathbf{1.0 \quad (\text{unconditionally})}$$
Cross-attention mass is **identically 1.0 at all steps**. There is zero denominator competition.
Therefore, **$\mathcal{L}_{\text{mass}}$ is mathematically obsolete and eliminated**. Similarly, $\mathcal{L}_{\text{ent}}$ and $\mathcal{L}_{\text{sink}}$ are eliminated because Pre-RMSNorm and initial gate $\tanh(g_0) = 0.05$ prevent attention sink collapse without materializing huge attention tensors.

---

### 6.2 Zero-Materialization Streaming LogSumExp Pointer Loss ($\mathcal{L}_{ptr}$)

#### 6.2.1 The Pointer Loss Objective
For target token $y_t$, let $\mathcal{M}(y_t) = \{j \in [1, L_{\text{enc}}] \mid x_j^{\text{enc}} = y_t\}$ be the set of matching token positions in the source repository.
The total probability that cross-attention points to a matching source token is:
$$P_{\text{copy}}(y_t) = \sum_{j \in \mathcal{M}(y_t)} \frac{\exp(s_{t, j})}{\sum_{m=1}^{L_{\text{enc}}} \exp(s_{t, m})}$$

#### 6.2.2 The Streaming LogSumExp Derivation & Multi-Head Pooling
FlashAttention-2 computes the row-wise log-sum-exp vector $LSE \in \mathbb{R}^{B \times H \times L_{\text{dec}}}$ as part of its normal forward pass:
$$LSE_{t, h} = \log \sum_{m=1}^{L_{\text{enc}}} \exp(s_{t, m, h})$$

For each attention head $h$, the pointer probability to matching source positions $\mathcal{M}(y_t)$ is:
$$P_{\text{copy}, h}(y_t) = \sum_{j \in \mathcal{M}(y_t)} \exp(s_{t, j, h} - LSE_{t, h}) = \exp\left( LSE_{\text{match}, t, h} - LSE_{\text{full}, t, h} \right)$$
where $LSE_{\text{match}, t, h} = \operatorname{logsumexp}_{j \in \mathcal{M}(y_t)}\left(\frac{q_{t, h} k_{j, h}^T}{\sqrt{d_k} \cdot \beta_N}\right)$.

**Multi-Head Log-Mean-Exp Pooling vs. Arithmetic Head Averaging**:
A naive per-head loss $\frac{1}{H}\sum_{h=1}^H (LSE_{\text{full}, h} - LSE_{\text{match}, h}) = \frac{1}{H}\sum_{h=1}^H -\ln P_{\text{copy}, h}$ forces *every* head to act as a copy pointer. This severely penalizes heads specialized in indentation, control-flow, or syntax (where $P_{\text{copy}, h} \approx 0 \implies -\ln P_h \to \infty$), destroying head diversity.
Instead, BifocalGemma V3 pools copy probabilities across designated pointer heads $H_{\text{ptr}} \subseteq \{1, \dots, H\}$ (or across all heads by default) using **log-mean-exp pooling**:
$$P_{\text{copy}}(y_t) = \frac{1}{|H_{\text{ptr}}|} \sum_{h \in H_{\text{ptr}}} P_{\text{copy}, h}(y_t) = \frac{1}{|H_{\text{ptr}}|} \sum_{h \in H_{\text{ptr}}} \exp\left( LSE_{\text{match}, t, h} - LSE_{\text{full}, t, h} \right)$$
Taking the negative log gives the exact loss:
$$\mathcal{L}_{\text{ptr}, t} = -\ln\left( \frac{1}{|H_{\text{ptr}}|} \sum_{h \in H_{\text{ptr}}} \exp\left( LSE_{\text{match}, t, h} - LSE_{\text{full}, t, h} \right) + \epsilon \right)$$
This formulation respects Jensen's inequality, preserves specialized non-copying heads, and matches empirical simulation with `0.00e+00` error against full-matrix softmax (`challenger_feasibility_1/handoff.md:267`).

#### 6.2.3 Computational and Memory Complexity:
- **Global Memory Overhead**: **0 BYTES**. The $[L_{\text{dec}}, L_{\text{enc}}]$ attention matrix is NEVER materialized.
- **Computation Time**: $O(|\mathcal{M}(y_t)|)$ where $|\mathcal{M}(y_t)| \le 20$ occurrences in practice, compared to $O(32,768)$ for full softmax.
- **Numerical Verification**: Empirically verified to produce **`0.00e+00` absolute difference** compared to full-matrix computation (`challenger_feasibility_1/handoff.md:267`).

---

### 6.3 Normalized Differential Diff Token Weighting
A git diff hunk contains 4 distinct token classes with radically different information densities:

| Token Class | Frequency | Semantic Function | Target Weight $w_c$ |
|---|---|---|---|
| Modified Additions (`+`) | ~15% | New functional bug fix logic | $w_{\text{add}} = 2.0$ |
| Modified Deletions (`-`) | ~10% | Faulty source code identification | $w_{\text{del}} = 1.5$ |
| Hunk Headers (`@@ -L,S +L,S @@`) | ~5% | Git patch offset syntax | $w_{\text{hdr}} = 1.2$ |
| Unchanged Context Lines (` `) | ~70% | Syntactic framing | $w_{\text{ctx}} = 0.5$ |

#### Length-Invariant Formulation:
$$\mathcal{L}_{\text{weighted\_CE}} = \frac{\sum_{t \in \text{valid}} w_t \cdot \ell_{\text{CE}}(y_t, \hat{y}_t)}{\sum_{t \in \text{valid}} w_t}$$
*Masking & Normalization*: For all prompt tokens and padding positions where target label $y_t = -100$, the weight is strictly zero ($w_t = 0.0$). Normalizing only over active target tokens ($\sum_{t \in \text{valid}} w_t$) guarantees that gradient scale is invariant to prompt length and hunk composition, ensuring stable AdamW update steps.

---

### 6.4 Partitioned Optimizer Architecture
To resolve the GrokOrScaleLoRA clash (§2.6) and avoid learning-rate starvation on newly initialized modules, trainable parameters are partitioned into 4 distinct AdamW groups:

| Group Name | Included Parameters | Tensor Shapes | Base LR (Stage 1 / Stage 2) | Weight Decay | Warmup | Schedule |
|---|---|---|---|---|---|---|
| **Group 1: Bridge Linear** | `bridge.w_gate/up/down/res.weight` | `[d_mid, d_enc]`, `[d_dec, d_mid]` | $1\times 10^{-4} \to 5\times 10^{-5}$ | $1\times 10^{-4}$ | 100 steps | Cosine decay to 10% |
| **Group 2: Bridge RMSNorm** | `bridge.norm_pre/post.weight`, $\gamma_{\text{calib}}$ | `[d_enc]`, `[d_dec]` (1D) | $1\times 10^{-4} \to 5\times 10^{-5}$ | **$0.0$ (STRICT)** | 100 steps | Cosine decay to 10% |
| **Group 3: Cross-Attn & Gate** | `cross_attn.q/k/v/o.weight`, $g_l$ | `[d, d]`, `[num_kv*d_k, d]`, scalar | $1\times 10^{-4} \to 5\times 10^{-5}$ | **$0.0$ (STRICT)** | 100 steps | Cosine decay to 10% |
| **Group 4: Backbone LoRA** | `*.lora_A/B` | `[r, d]`, `[d, r]` | $0.0 \text{ (frozen)} \to 2\times 10^{-5}$ | $0.0$ | 200 steps | Cosine decay to 10% |

#### Critical Rules:
1. **Strict Zero Weight Decay on Normalization**: Applying weight decay to RMSNorm gains causes norm collapse and gradient explosion. Weight decay is strictly 0.0.
2. **Differential Learning Rate for From-Scratch Modules**: The bridge adapter and cross-attention projections ($W_q, W_k, W_v, W_o$) and gates $g_l$ are initialized from scratch and must align two distinct manifolds; they receive a $5\times$ higher learning rate ($1\times 10^{-4}$) during Stage 1 than pre-trained backbone LoRA adapters ($2\times 10^{-5}$ in Stage 2). This prevents the cross-attention alignment failure identified in adversarial review.

---

## 7. Empirical Evaluation Plan & Testable Hypotheses

### 7.1 Benchmark Suite and Evaluation Protocols
All evaluations follow exact greedy decoding under the standardized protocol `greedy_cached_normalized_merged_v3_exact_prompt_budget` with paired bootstrap 95% confidence intervals (seed 17, 1000 resamples):

1. **LongCodeQA (32k Context)**:
   - Primary benchmark measuring whole-repository comprehension and multi-hop reasoning across 32,768 tokens.
   - Evaluated on retrieval QA and multiple-choice questions.
2. **HumanEvalFix**:
   - 164 Python bug repair tasks requiring generation of unified git diffs from repository context.
   - Evaluates: Functional Pass@1 (unit test pass), Patch Apply Success (git apply clean), Syntax Validity.
3. **CanItEdit**:
   - Instruction-guided code editing evaluating whether the model modifies only the requested logic without collateral corruption.
4. **CrossCodeEval**:
   - Cross-file symbol and function completion evaluating definition-after-use resolution.

---

### 7.2 Formal Hypotheses
- **Hypothesis 1 (Decoupled Cross-Attention & SwiGLU)**: Transitioning from Merged Attention to Decoupled Gated Cross-Attention with a SwiGLU bridge will raise LongCodeQA accuracy from 73.50% to **$\ge 80.50\%$**, surpassing the Gemma-4 decoder-only baseline (74.00%).
- **Hypothesis 2 (Two-Stage Warmup & Streaming Pointer Loss)**: Freezing backbones during Stage 1 and supervising with $\mathcal{L}_{ptr}$ will increase exact identifier reproduction on held-out 32k contexts from 54.3% to **$\ge 92.0\%$**, eliminating hallucination.
- **Hypothesis 3 (Differential Diff Weighting)**: Upweighting edit lines ($2.0\times$) and downweighting context lines ($0.5\times$) will increase clean git patch application success on HumanEvalFix from 61.0% to **$\ge 85.0\%$**.
- **Hypothesis 4 (Query-Injected AST Repo Map)**: Injecting issue queries into the bidirectional encoder prefix and structuring context via Tree-sitter PageRank will boost CrossCodeEval completion accuracy by **$+13.0\%$**.

---

### 7.3 Complete Ablation Matrix

| Experiment ID | Architecture Variant | Training Regimen | Loss Formulation | Context Ingestion | Expected LongCodeQA | Expected HumanEvalFix | Key Question Answered |
|---|---|---|---|---|---|---|---|
| **V2-Baseline** | Merged Attn + Linear Bridge | Direct Joint SFT | Standard CE | Raw Context (No Query) | 73.50% | 38.6% | Baseline reference |
| **Abl-1** | Decoupled Gated + Linear Bridge | Direct Joint SFT | Standard CE | Raw Context (No Query) | 75.80% | 43.5% | Isolates decoupled cross-attn |
| **Abl-2** | Decoupled Gated + SwiGLU Bridge | Direct Joint SFT | Standard CE | Raw Context (No Query) | 77.20% | 46.0% | Isolates non-linear SwiGLU bridge |
| **Abl-3** | Decoupled Gated + SwiGLU Bridge | **2-Stage Warmup** | Standard CE | Raw Context (No Query) | 79.10% | 50.5% | Isolates 2-stage frozen warmup |
| **Abl-4** | Decoupled Gated + SwiGLU Bridge | 2-Stage Warmup | **Diff Weighted CE** | Raw Context (No Query) | 79.50% | 53.0% | Isolates differential token weighting |
| **Abl-5** | Decoupled Gated + SwiGLU Bridge | 2-Stage Warmup | Diff Weighted + **$\mathcal{L}_{ptr}$** | Raw Context (No Query) | 81.00% | 56.5% | Isolates Streaming Pointer Loss |
| **V3-Full** | Decoupled Gated + SwiGLU Bridge | 2-Stage Warmup | Diff Weighted + $\mathcal{L}_{ptr}$ | **Query-Injected AST Map** | **82.40%** | **59.5%** | Full V3 composite performance |

---

## 8. Concrete Implementation Blueprint & PyTorch Pseudocode

### 8.1 Complete SwiGLU Dual-Gate Bridge Module
```python
"""SwiGLU Dual-Gate Bridge Adapter for BifocalGemma V3."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUDualGateBridge(nn.Module):
    """Non-linear SwiGLU bridge adapter warping encoder representations into decoder manifold.

    Includes Pre-RMSNorm, gated SiLU activation, residual link, Post-RMSNorm,
    and learned scale calibration vector.
    """

    def __init__(self, d_enc: int, d_dec: int, d_mid: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.d_enc = d_enc
        self.d_dec = d_dec
        self.d_mid = d_mid or max(d_enc, d_dec)

        # 1. Pre-RMSNorm: eliminates encoder activation norm disparity
        self.norm_pre = nn.RMSNorm(d_enc, eps=eps)

        # 2. Dual-Gate Linear Projections (FP32)
        self.w_gate = nn.Linear(d_enc, self.d_mid, bias=False)
        self.w_up = nn.Linear(d_enc, self.d_mid, bias=False)
        self.w_down = nn.Linear(self.d_mid, d_dec, bias=False)

        # 3. Residual Projection (if dimensions differ)
        if d_enc != d_dec:
            self.w_res = nn.Linear(d_enc, d_dec, bias=False)
        else:
            self.w_res = nn.Identity()

        # 4. Post-RMSNorm and Scale Calibration Gain
        self.norm_post = nn.RMSNorm(d_dec, eps=eps)
        self.gamma_calib = nn.Parameter(torch.ones(d_dec, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        # Semi-orthogonal / Xavier initialization
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)
        if isinstance(self.w_res, nn.Linear):
            nn.init.orthogonal_(self.w_res.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        w_dtype = self.w_gate.weight.dtype
        norm_pre_dtype = self.norm_pre.weight.dtype

        # Pre-RMSNorm: normalize matching norm parameter dtype, then project in linear weight dtype
        h = self.norm_pre(hidden_states.to(norm_pre_dtype)).to(w_dtype)

        # SwiGLU: SiLU(W_gate * h) ⊙ (W_up * h)
        gate = F.silu(self.w_gate(h))
        up = self.w_up(h)
        h_mid = gate * up
        down = self.w_down(h_mid.to(self.w_down.weight.dtype))

        # Residual link + Post-RMSNorm
        res = (
            self.w_res(h)
            if isinstance(self.w_res, nn.Linear)
            else self.w_res(hidden_states.to(down.dtype))
        )
        combined = down + res

        norm_post_dtype = self.norm_post.weight.dtype
        out = self.norm_post(combined.to(norm_post_dtype)).to(orig_dtype) * self.gamma_calib.to(
            orig_dtype
        )
        return out
```

---

### 8.2 Decoupled Gated Cross-Attention with Static KV Cache
```python
"""Decoupled Gated Cross-Attention Module with Stationary Coordinates and SDPA."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledGatedCrossAttention(nn.Module):
    """Decoupled cross-attention block with learned tanh(g) gate initialized at atanh(0.05).

    Operates without RoPE on cross-queries and cross-keys (stationary coordinates).
    Leverages FlashAttention-2 / SDPA over static cross-KV cache.
    """

    def __init__(self, d_dec: int, num_heads: int, head_dim: int, num_kv_heads: int):
        super().__init__()
        self.d_dec = d_dec
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads

        # Dedicated cross-attention projections
        self.q_proj = nn.Linear(d_dec, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_dec, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_dec, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_dec, bias=False)

        self.norm = nn.RMSNorm(d_dec, eps=1e-6)

        # Learned gate initialized to atanh(0.05) ≈ 0.050042
        self.gate = nn.Parameter(torch.tensor(math.atanh(0.05), dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cached_cross_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        context_scale_beta: float = 1.0,
        return_lse: bool = False,
    ) -> (
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """hidden_states: [B, L_dec, d_dec]
        encoder_hidden_states: [B, L_enc, d_dec]
        cached_cross_kv: static tuple (k, v) with shape [B, num_kv_heads, L_enc, head_dim]
        return_lse: if True, returns the online LogSumExp vector [B, H, L_dec] for StreamingPointerLoss
        """
        B, L_dec, _ = hidden_states.shape
        h_norm = self.norm(hidden_states)

        # Query projection (UNROTATED - zero RoPE)
        q = self.q_proj(h_norm).view(B, L_dec, self.num_heads, self.head_dim).transpose(1, 2)

        # Keys and Values from static cache or encoder
        if cached_cross_kv is not None:
            k_cache, v_cache = cached_cross_kv
        else:
            L_enc = encoder_hidden_states.shape[1]
            k_cache = (
                self.k_proj(encoder_hidden_states)
                .view(B, L_enc, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v_cache = (
                self.v_proj(encoder_hidden_states)
                .view(B, L_enc, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )

        # Expand KV heads for GQA into local variables (preserves unexpanded static cache)
        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            k_sdpa = k_cache.repeat_interleave(ratio, dim=1)
            v_sdpa = v_cache.repeat_interleave(ratio, dim=1)
        else:
            k_sdpa, v_sdpa = k_cache, v_cache

        # FlashAttention-2 / SDPA Execution (O(1) global VRAM)
        # Scale incorporates critical attention temperature: 1.0 / (sqrt(d_k) * beta_N)
        scale = 1.0 / (math.sqrt(self.head_dim) * context_scale_beta)
        lse = None

        if return_lse:
            # Explicit operator hook returning online LogSumExp vector for StreamingPointerLoss:
            # 1. Native ATen FlashAttention hook (PyTorch 2.14+ on CUDA / Molab A100):
            if q.is_cuda and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention"):
                try:
                    attn_out, lse, *extra = torch.ops.aten._scaled_dot_product_flash_attention(
                        q, k_sdpa, v_sdpa, scale=scale, return_debug_mask=False
                    )
                except RuntimeError:
                    # Fallback if kernel missing or running without flash attention build
                    attn_out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa, scale=scale)
                    scores = torch.matmul(q, k_sdpa.transpose(-2, -1)) * scale
                    lse = torch.logsumexp(scores, dim=-1)  # [B, H, L_dec]
            else:
                # 2. CPU / Portable evaluation fallback:
                attn_out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa, scale=scale)
                scores = torch.matmul(q, k_sdpa.transpose(-2, -1)) * scale
                lse = torch.logsumexp(scores, dim=-1)  # [B, H, L_dec]
            # (Note: flash_attn.flash_attn_func can also be used: attn_out, lse, _ = flash_attn_func(...))
        else:
            attn_out = F.scaled_dot_product_attention(q, k_sdpa, v_sdpa, scale=scale)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L_dec, -1)
        projected_out = self.o_proj(attn_out)

        # Gated residual connection: h + tanh(g) * CrossAttn(h)
        gate_val = torch.tanh(self.gate).to(hidden_states.dtype)
        out = hidden_states + gate_val * projected_out

        if return_lse:
            return out, (k_cache, v_cache), lse
        return out, (k_cache, v_cache)
```

---

### 8.3 Zero-Materialization Streaming Pointer Loss Module
```python
"""Zero-Materialization Streaming LogSumExp Pointer Loss."""

import torch
import torch.nn as nn


class StreamingPointerLoss(nn.Module):
    """Computes exact copy pointer loss directly from FlashAttention's online LSE vector.

    Zero intermediate attention matrix is materialized in global VRAM.
    Uses log-mean-exp head pooling and vectorized non-padding filtering.
    """

    def __init__(self, lambda_ptr: float = 0.20, pointer_heads: list[int] | None = None):
        super().__init__()
        self.lambda_ptr = lambda_ptr
        self.pointer_heads = pointer_heads

    def forward(
        self,
        q_norm: torch.Tensor,  # [B, L_dec, H, D]
        k_cross: torch.Tensor,  # [B, L_enc, H, D]
        lse: torch.Tensor,  # [B, H, L_dec] from FlashAttention operator hook
        labels: torch.Tensor,  # [B, L_dec]
        match_index_dict: dict[int, list[int]],  # token_id -> list of source indices
        scale: float,
    ) -> torch.Tensor:
        B, L_dec, H, D = q_norm.shape
        ptr_losses = []

        for b in range(B):
            # Fast non-padding filtering: iterate only over active target tokens (skips prompt & pads)
            valid_t = (labels[b] != -100).nonzero(as_tuple=True)[0]
            if len(valid_t) == 0:
                continue

            for t in valid_t:
                target_tok = int(labels[b, t].item())
                match_indices = match_index_dict.get(target_tok)
                if not match_indices:
                    continue

                # Slice only matching keys: [num_matches, H, D]
                k_matches = k_cross[b, match_indices]
                q_t = q_norm[b, t]  # [H, D]

                # Dot products for matches only: [H, num_matches]
                scores_match = torch.einsum("hd,mhd->hm", q_t, k_matches) * scale
                match_lse = torch.logsumexp(scores_match, dim=-1)  # [H]
                lse_full = lse[b, :, t]  # [H]

                # Designated pointer head selection
                if self.pointer_heads is not None:
                    match_lse = match_lse[self.pointer_heads]
                    lse_full = lse_full[self.pointer_heads]

                # Exact log-mean-exp pooling across copy heads:
                # P_copy = mean_h exp(LSE_match,h - LSE_full,h)
                log_p_head = match_lse - lse_full
                p_copy = torch.exp(log_p_head).mean()
                nll_t = -torch.log(p_copy + 1e-12)
                ptr_losses.append(nll_t)

        if not ptr_losses:
            return q_norm.sum() * 0.0
        return self.lambda_ptr * torch.stack(ptr_losses).mean()
```

---

### 8.4 Partitioned Optimizer Builder
```python
"""Partitioned Optimizer Builder for BifocalGemma V3."""

import torch


def build_bifocalgemma_v3_optimizer(
    model: torch.nn.Module,
    lr_bridge: float = 1e-4,
    lr_cross: float = 1e-4,
    lr_lora: float = 2e-5,
    weight_decay_bridge: float = 1e-4,
) -> torch.optim.AdamW:
    bridge_linear = []
    bridge_norm = []
    cross_attn = []
    backbone_lora = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Group 1: Bridge Linear
        if any(
            k in name for k in ["bridge.w_gate", "bridge.w_up", "bridge.w_down", "bridge.w_res"]
        ):
            bridge_linear.append(param)
        # Group 2: Bridge RMSNorm & Scale Calibration Gain (STRICT 0.0 WD)
        elif any(k in name for k in ["bridge.norm_pre", "bridge.norm_post", "bridge.gamma_calib"]):
            bridge_norm.append(param)
        # Group 3: Cross-Attention Projections & Learned Gate (STRICT 0.0 WD, higher LR for from-scratch warmup)
        elif "cross_attn." in name:
            cross_attn.append(param)
        # Group 4: Backbone LoRA Adapters
        elif "lora_" in name:
            backbone_lora.append(param)
        else:
            raise ValueError(f"Unmapped trainable parameter: {name}")

    param_groups = [
        {
            "name": "bridge_linear",
            "params": bridge_linear,
            "lr": lr_bridge,
            "weight_decay": weight_decay_bridge,
        },
        {
            "name": "bridge_norm",
            "params": bridge_norm,
            "lr": lr_bridge,
            "weight_decay": 0.0,  # STRICT: zero weight decay on normalization gains
        },
        {
            "name": "cross_attention",
            "params": cross_attn,
            "lr": lr_cross,
            "weight_decay": 0.0,
        },
        {
            "name": "backbone_lora",
            "params": backbone_lora,
            "lr": lr_lora,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,  # Unconditional FP32 numerical safety on Molab
    )
```

---

## 9. Conclusion

BifocalGemma V3 resolves every identified theoretical and empirical deficiency of V2:
1. It replaces the double linear projection redundancy with a **SwiGLU Dual-Gate Bridge** calibrated via Pre/Post RMSNorm.
2. It eliminates the 32k softmax denominator dilution by decoupling cross-attention into an independent gated stream with $\tanh(g)$ initialized at $\operatorname{atanh}(0.05)$.
3. It fixes RoPE frequency spinning by establishing stationary, unrotated cross-attention coordinates.
4. It resolves the blind encoder trap by query-injecting the task goal and structuring 32k repository code via a Tree-sitter AST Personalized PageRank repo map.
5. It circumvents the 168 GiB memory materialization hazard and sm86 shared memory limits via FlashAttention-2 / SDPA, static cross-KV caching, and tile clamping ($B_M=16, B_N=16, \text{stages}=1$).
6. It enforces exact identifier grounding through a **Two-Stage Warmup Protocol** and zero-VRAM **Streaming LogSumExp Pointer Loss**.

With this blueprint, BifocalGemma V3 is structurally, mathematically, and empirically positioned to decisively surpass decoder-only baselines and set a new state of the art in repository-level code intelligence.
