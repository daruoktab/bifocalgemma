# 🏆 JANUS-Edge: Master Research Proposal & Technical Specification
**Judul Paper:** *JANUS-Edge: Looking Both Ways—Complete 2×2 Factorial Analysis of Symmetric and Asymmetric Encoder-Decoder Gemma 4 for Multimodal Developer Agents*  
**Target Kompetisi:** [Google - The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper) ($35,000 Prize Pool)  
**Timeline Resmi:** Deadline: 12 Nov 2026 | NeurIPS 2026 Google Expo  

---

## 1. Executive Summary & Validasi Gemma 4 Edge (E2B & E4B)

Fokus proyek ini adalah mengadaptasi **Gemma 4 E2B** dan **Gemma 4 E4B** menjadi keluarga model **Encoder-Decoder** pertama untuk developer agent multimodal.

### Karakteristik Arsitektur Gemma 4 Edge:
1. **Vision Modality Terintegrasi**: Menggunakan Vision Transformer (ViT) 150M terdedikasi (patch size 16, 2D axial RoPE) dengan resolusi dinamis ($N_{\max} \in \{70, 140, 280, 560, 1120\}$).
2. **Audio Modality**: Menggunakan Conformer USM 305M (40ms chunks).
3. **Per-Layer Embeddings (PLE)**: E2B memiliki 2.34B PLE (efektif 2.3B), dan E4B memiliki 2.82B PLE (efektif 4.5B).
4. **Proyeksi K-V Terpisah di Seluruh Layer**: Mempertahankan Key dan Value independen di layer lokal dan global, sangat ideal untuk integrasi Cross-Attention Bridge.

---

## 2. Matriks Eksperimen Faktorial 2×2

Kami merancang dan menguji matriks faktorial $2 \times 2$ lengkap yang mencakup seluruh pasangan simetris dan asimetris dari keluarga Edge:

$$\begin{pmatrix} \text{JANUS-E2B-E2B} & \text{JANUS-E2B-E4B} \\ \text{JANUS-E4B-E2B} & \text{JANUS-E4B-E4B} \end{pmatrix}$$

| Konfigurasi | Encoder | Decoder | Parameter Efektif | Bobot 16-bit | Peran Ilmiah & Pertanyaan Riset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **JANUS-E2B-E2B** | E2B | E2B IT | **~4.6B** | **~9.2 GB** | **RQ1 (Iso-Parameter Test)**: Apakah arsitektur Encoder-Decoder 4.6B mengalahkan model Decoder-Only seukuran (Base E4B 4.5B)? |
| **JANUS-E4B-E2B** | E4B | E2B IT | **~6.8B** | **~13.6 GB** | **RQ2a (Heavy Reader)**: Apakah mengalokasikan kapasitas lebih besar ke Encoder (E4B) memberikan grounding AST kode dan visual UI yang lebih unggul? |
| **JANUS-E2B-E4B** | E2B | E4B IT | **~6.8B** | **~13.6 GB** | **RQ2b (Deep Reasoner)**: Apakah mengalokasikan kapasitas lebih besar ke Decoder (E4B) dengan Thinking Mode lebih penting untuk perencanaan agent? |
| **JANUS-E4B-E4B** | E4B | E4B IT | **~9.0B** | **~18.0 GB** | **RQ3 (Upper Bound Scaling)**: Seberapa dekat performa varian flagship 9B ini menyaingi model raksasa 27B–31B pada SWE-bench Multimodal? |

### Pertarungan Asimetris Iso-Parameter: E4B-E2B vs E2B-E4B (~6.8B)
Pada anggaran parameter yang persis sama (~6.8B parameter efektif), paper ini menjawab dilema fundamental rekayasa perangkat lunak AI:
> *"Lebih krusial mana bagi autonomous developer agent: Kapasitas pemahaman konteks repositori yang dalam (Heavy Encoder), ataukah kapasitas penalaran sintaksis generasi patch yang besar (Heavy Decoder)?"*

---

## 3. Modul Gated Cross-Attention Bridge

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BIDIRECTIONAL ENCODER (E2B / E4B)                      │
│   [Tree-sitter Code AST]     [Issue Query]     [150M ViT Vision Tokens]     │
│                              │                                              │
│                              ▼                                              │
│         H_enc = Bidirectional_Self_Attention(QK-Norm, RoPE)                 │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼ (K, V dari H_enc d_enc)
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTOREGRESSIVE DECODER (E2B IT / E4B IT)                 │
│                                                                             │
│   X ──► Causal Self-Attention (QK-Norm, RoPE)                               │
│            │                                                                │
│            ▼                                                                │
│   z = X + tanh(g) * CrossAttention(Q(X), K(H_enc), V(H_enc))                │
│       (Inisialisasi: g = atanh(0.05) ~ 0.05004, menjaga 95% decoder identity)│
│            │                                                                │
│            ▼                                                                │
│   Output Generasi: <|channel>thought ... <|tool_call> ... git patch         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 5 Bumbu Inovasi Arsitektur

1. **Dual-Use Self-Indexing Encoder**: Encoder menghasilkan representasi vektor padat untuk pencarian MIPS repositori sekaligus menyediakan *hidden states* untuk cross-attention decoder tanpa perlu proses re-encoding (memotong latensi RAG repositori hingga 50%).
2. **AST-Guided Infilling (Graph-UL2)**: Masking pre-training berbasis simpul AST (Tree-sitter) untuk *Function Body Infilling* dan *Call-Site Resolution*.
3. **Multi-Positive Contrastive Localization Loss**: Metrik penalti yang mampu mengenali perubahan di beberapa file/fungsi sekaligus dalam satu PR tanpa mendiskriminasi file terkait lainnya.
4. **Latent Repository Memory dengan 1-Hop Invalidation**: Cache representasi repositori di VRAM; saat developer agent menjalankan `edit_file`, hanya file target dan modul dependensi 1-hop yang di-encode ulang (latensi multi-turn turun hingga 8x).
5. **Asymmetric Model Scaling**: Memfasilitasi fleksibilitas pemasangan pasangan E2B dan E4B beda ukuran.

---

## 5. Anggaran VRAM Pelatihan Presisi Penuh 16-Bit (BF16)

| Konfigurasi Model | Bobot 16-bit | LoRA (r=16) + 8-bit AdamW | Bridge/Adapter Fine-Tuning | Kelayakan Hardware |
| :--- | :--- | :--- | :--- | :--- |
| **E2B-E2B (~4.6B)** | **~9.2 GB** | **~11.5 – 13.0 GB** | **~14.5 – 16.5 GB** | Single GPU 24GB & Kaggle Dual T4 |
| **E4B-E2B (~6.8B)** | **~13.6 GB** | **~15.5 – 17.5 GB** | **~18.5 – 20.0 GB** | Single GPU 24GB (Headroom Aman) & Kaggle Dual T4 |
| **E2B-E4B (~6.8B)** | **~13.6 GB** | **~15.5 – 17.5 GB** | **~18.5 – 20.0 GB** | Single GPU 24GB (Headroom Aman) & Kaggle Dual T4 |
| **E4B-E4B (~9.0B)** | **~18.0 GB** | **~20.5 – 22.5 GB** | **~23.5 – 25.5 GB** | Kaggle Dual T4 (16GB+16GB via Pipeline Parallel) |

---

## 6. Ketersediaan Dataset Multimodal Code

* **SWE-bench Multimodal**: 612 task terverifikasi (102 dev, 510 test) — Benchmark utama & trajectory SFT bug visual.
* **MMCode**: 3.548 soal, 6.620 gambar + Python unit tests — Penalaran algoritma visual & koordinat grafik.
* **WebSight**: 2.7M screenshot $\to$ HTML/Tailwind — Penyelarasan tata letak visual UI ke kode frontend.
* **OmniDiagram**: 196.000 diagram $\to$ Mermaid/PlantUML/TikZ — Pemahaman diagram arsitektur sistem software.
