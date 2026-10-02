# 🏆 JANUS-4: Master Research Proposal & Technical Specification
**Judul Paper:** *JANUS-4: Looking Both Ways—Asymmetric Bidirectional Encoding and Causal Generation with Encoder-Free Gemma 4 for Multimodal Developer Agents*  
**Target Kompetisi:** [Google - The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper) ($35,000 Prize Pool)  
**Timeline Resmi:** Early Submission: 12 Nov 2026 | Final Deadline: 2 Des 2026  

---

## 1. Executive Summary & Validasi Gemma 4 12B

Menggunakan **Gemma 4 12B** sebagai backbone membawa keunggulan sekaligus tantangan unik:
1. **Sifat Encoder-Free (Revolusioner)**:
   - Tidak seperti model lama (SigLIP 550M / Conformer 305M), Gemma 4 12B menggunakan **Linear Patch Projection 35M** (memproyeksikan patch RGB $48 \times 48 \times 3$ langsung ke embedding space $d=3840$) dan audio chunk 40ms ($640 \to 3840$).
   - Menghilangkan latensi terpisah dari encoder vision/audio dan menyatukan seluruh representasi ke dalam manifold representasi LLM.
2. **Kesesuaian dengan Jalur Paper Track**:
   - Menjadi paper pertama di komunitas yang membedah dan mengadaptasi model *encoder-free multimodal LLM* menjadi arsitektur *bidirectional encoder-decoder* untuk developer agent.
   - Mengatasi tantangan software engineering modern yang tidak hanya berbasis teks, melainkan sarat multimodal: **UI screenshot bug, sketsa wireframe, diagram arsitektur UML/Mermaid, dan plot log eksekusi pytest**.

---

## 2. Analisis Keamanan & Arsitektur Gemma 4 12B Bidirectional

Apakah aman membangun bidirectional encoder langsung dari Gemma 4 12B? **Sangat aman dan feasible, asalkan 4 jebakan teoritis berikut dimitigasi**:

```
                  ARSITEKTUR ENCODER-DECODER JANUS-4 (GEMMA 4 12B)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BIDIRECTIONAL ENCODER (12B)                           │
│  [Code AST Tokens]      [System Prompt]      [UI Screenshot (576 Patches)]      │
│         │                      │                          │                     │
│         └──────────────────────┼──────────────────────────┘                     │
│                                ▼                                                │
│              Bidirectional Self-Attention (QK-Norm + p-RoPE)                    │
│                 (Dihitung HANYA SEKALI di awal turn - O(1))                     │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼ H_enc (Hidden States d=3840)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AUTOREGRESSIVE DECODER (E4B atau 12B)                    │
│                                                                                 │
│   [Issue Query] ──► Self-Attention (Causal Gemma 4 IT)                          │
│                            │                                                    │
│                            ▼                                                    │
│                    Gated Cross-Attention Bridge ◄─── (K, V dari H_enc)          │
│                    z = h + tanh(g) * CrossAttn (g_0 = 0.05)                     │
│                            │                                                    │
│                            ▼                                                    │
│                    Feed-Forward Network (SwiGLU)                                │
│                            │                                                    │
│                            ▼                                                    │
│         Generasi: <|channel>thought ... <|tool_call> ... git diff               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Mitigasi 4 Jebakan Fisika & Matematika Transformer:
1. **RoPE Phase Inversion ($\Delta > 0$)**:
   - RoPE memiliki suku sinus anti-simetris ($\sin(-\Delta\theta) = -\sin(\Delta\theta)$). Model causal hanya pernah melihat $\Delta \le 0$. Saat bidirectional diaktifkan, query ke masa depan ($\Delta > 0$) membalik fase rotasi.
   - **Penyelamat Gemma 4**: Gemma 4 menggunakan **p-RoPE ($p=0.25$)** pada layer global. Artinya, **75% dimensi kepala (192 dari 256 dim) tidak diputar sama sekali**, sehingga kebal terhadap inversi fase! Hanya 64 dimensi posisi yang terpengaruh, sehingga proses adaptasi jauh lebih cepat stabil.
2. **The Omnipresent Sink Problem**:
   - Bobot token 0 (`<bos>`) yang awalnya menjadi penampung residual attention pada causal decoding berpotensi menyerap perhatian berlebih saat causal mask dicabut.
   - **Penyelamat Gemma 4**: Mekanisme **QK-Norm** bawaan Gemma 4 menormalisasi $\mathbf{q}$ dan $\mathbf{k}$ sebelum dot-product, mencegah logit meledak dan menjaga stabilitas gradien.
3. **Norm Drift pada Patch Matmul 35M**:
   - Memasukkan 576 token patch mentah ke Layer 0 bidirectional encoder dapat memicu lonjakan entropi perhatian pada teks.
   - **Solusi**: Terapkan laju pembelajaran $0.2\times$ pada matriks patch $W_{\text{patch}}$, atau bekukan (*freeze*) selama 500 step pertama pemanasan adapter LoRA.
4. **Gradien Nol IPEG (Deadlock Mitigation)**:
   - Jika gate cross-attention diinisialisasi $g_i = 0$, gradien $\frac{\partial \mathcal{L}}{\partial H_{\text{enc}}} \propto \tanh(0) = 0$ (encoder mengalami *gradient starvation*).
   - **Solusi**: Inisialisasi $g_i = \text{atanh}(0.05) \approx 0.05$. Nilai ini mempertahankan 95% kemampuan bawaan decoder Gemma 4 IT sejak step awal, sekaligus mengalirkan 5% gradien aktif ke encoder dan modul patch vision.

---

## 3. Aturan Emas: Token Routing Multimodal

> [!IMPORTANT]
> **Token visual (UI/diagram/patch) WAJIB masuk 100% ke Bidirectional Encoder dan DILARANG masuk ke Decoder!**

### Bukti Matematis Efisiensi KV-Cache:
- Gambar $1152 \times 1152$ menghasilkan 576 patch token.
- Jika dimasukkan ke Decoder:
  576 token $\times$ 48 layer $\times$ 8 KV head $\times$ 256 head dim $\times$ 2 byte $\approx$ **226,5 MB per gambar**.
  Tiga gambar dalam diskusi isu (screenshot bug, expected design, console log) akan membakar **~680 MB KV-cache statis**.
- Jika dimasukkan ke Encoder:
  KV-cache decoder untuk gambar adalah **0 MB**! Cross-attention Key-Value ($K_{\text{cross}}, V_{\text{cross}}$) dihitung **hanya sekali** di awal turn, dan generasi token berikutnya berjalan pada kecepatan $O(1)$.

---

## 4. Ketersediaan Dataset Multimodal Code

| Nama Dataset | Repositori Hugging Face / Sumber | Volume / Sampel | Lisensi | Peran dalam Kurikulum |
| :--- | :--- | :--- | :--- | :--- |
| **SWE-bench Multimodal** | [`SWE-bench/SWE-bench_Multimodal`](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal) | **612 task terverifikasi** (102 dev, 510 test) | MIT | Benchmark utama & trajectory SFT bug visual |
| **MMCode** | [`likaixin/MMCode`](https://huggingface.co/datasets/likaixin/MMCode) | **3.548 soal, 6.620 gambar** + Python unit tests | Apache-2.0 | Penalaran algoritma visual & grafik koordinat |
| **WebSight** | [`HuggingFaceM4/WebSight`](https://huggingface.co/datasets/HuggingFaceM4/WebSight) | **2.745.658 screenshot $\to$ HTML/Tailwind** | CC-BY-4.0 | Penyelarasan tata letak visual UI ke kode |
| **OmniDiagram** | GitHub `Haoyue-Yang/OmniDiagram` (*ACL 2026*) | **196.000 diagram $\to$ Mermaid/PlantUML/TikZ** | CC-BY-4.0 | Pemahaman diagram arsitektur sistem software |
| **Design2Code** | [`SALT-NLP/Design2Code`](https://huggingface.co/datasets/SALT-NLP/Design2Code) | **484 baris utama + 80 HARD** | ODC-By | Evaluasi presisi CSS dan komponen frontend |
| **Web2Code** | [`MBZUAI-LLM/web2code`](https://huggingface.co/datasets/MBZUAI-LLM/web2code) | **100.000+ dialog instruksi halaman web** | Apache-2.0 | Percakapan multi-turn instruksi visual |
| **ChartQA** | [`ahmed-masry/ChartQA`](https://huggingface.co/datasets/ahmed-masry/ChartQA) | **32.719 plot visual $\to$ QA code grounding** | GPL-3.0 | Debugging grafik visual data science |

---

## 5. Komputasi & VRAM Budget (Asimetris 12B $\to$ E4B vs Simetris 12B+12B)

| Parameter / Beban VRAM | Simetris (12B Enc + 12B Dec) | Asimetris (12B Enc + E4B Dec) *(Direkomendasikan)* |
| :--- | :--- | :--- |
| **Parameter Total** | ~25.15B (+2.26B cross-attn) | **~16.48B (+0.73B cross-attn)** |
| **Base Weights 4-bit (NF4)** | ~12.2 GB | **~8.2 GB** |
| **Cross-Attention Sublayer** | 48 layer (1.2 GB di 4-bit / 4.5 GB BF16) | **32 layer (0.4 GB di 4-bit / 1.5 GB BF16)** |
| **LoRA ($r=16$) + 8-bit AdamW** | ~1.8 GB | **~1.2 GB** |
| **Static VRAM Training** | ~15.2 GB – 18.5 GB | **~9.8 GB – 10.9 GB** |
| **Sisa Headroom pada GPU 24GB** | 5.5 – 8.8 GB (Rentan OOM jika context panjang) | **13.1 – 14.2 GB (Sangat Lega & Aman)** |

### Kelayakan Hardware Kaggle:
- **Kaggle 2x NVIDIA T4 (16GB + 16GB = 32GB Total)**:
  - Model Parallelism: Letakkan **12B Encoder** di `cuda:0` (~6.6 GB) $\to$ sisa headroom **~8.4 GB** untuk 4.096 token gambar/kode.
  - Letakkan **E4B Decoder + Cross-Attention** di `cuda:1` (~4.3 GB) $\to$ sisa headroom **~10.7 GB** untuk autoregressive decoding.
  - **100% Feasible di environment gratis Kaggle!**
- **GPU Lokal 24GB (RTX 3090 / 4090)**:
  - Varian asimetris berjalan mulus dengan batch size 2, gradient accumulation, dan sequence context hingga 4.096 token.
