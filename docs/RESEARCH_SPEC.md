# 🔬 Spesifikasi Teknis Riset: JANUS-4

> Dokumen spesifikasi teknis arsitektur model, formulasi matematika efisiensi konteks panjang, dan anggaran komputasi 16-bit untuk tim pengembang.

---

## 🏛️ 1. Matriks Faktorial 2×2 Arsitektur Gemma 4 Edge

Kami mengeksplorasi seluruh kombinasi pasangan Encoder (Bidirectional) dan Decoder (Causal IT) dari keluarga Gemma 4 Edge:

$$\begin{pmatrix} \text{JANUS-E2B-E2B} & \text{JANUS-E2B-E4B} \\ \text{JANUS-E4B-E2B} & \text{JANUS-E4B-E4B} \end{pmatrix}$$

```
                                  KAPASITAS DECODER (Causal IT)
                         E2B IT (~2.3B)                    E4B IT (~4.5B)
                  ┌───────────────────────────────┬───────────────────────────────┐
   E2B (~2.3B)    │        JANUS-E2B-E2B          │        JANUS-E2B-E4B          │
                  │   • Parameter: ~4.6B          │   • Parameter: ~6.8B          │
KAPASITAS         │   • Bobot 16-bit: ~9.2 GB     │   • Bobot 16-bit: ~13.6 GB    │
ENCODER           │   • Headline: Iso-Parameter   │   • Vibe: "Agile Reader,      │
(Bidirectional)   │     Head-to-Head vs Base E4B  │            Deep Reasoner"     │
                  ├───────────────────────────────┼───────────────────────────────┤
   E4B (~4.5B)    │        JANUS-E4B-E2B          │        JANUS-E4B-E4B          │
                  │   • Parameter: ~6.8B          │   • Parameter: ~9.0B          │
                  │   • Bobot 16-bit: ~13.6 GB    │   • Bobot 16-bit: ~18.0 GB    │
                  │   • Vibe: "Heavy Reader,      │   • Headline: Flagship SOTA   │
                  │            Agile Coder"       │     Agent Rivaling 27B-31B    │
                  └───────────────────────────────┴───────────────────────────────┘
```

### Hipotesis Ilmiah & Pertanyaan Riset Utama:
1. **RQ1 (Uji Iso-Parameter 4.6B vs 4.5B)**:
   * *Hipotesis*: `JANUS-E2B-E2B` (~4.6B) akan mengungguli model decoder-only `Gemma 4 E4B` (4.5B) pada tugas rekayasa perangkat lunak berkonten panjang karena encoder bidirectional membaca relasi dan dependensi repositori secara utuh tanpa terhambat batasan sliding-window atau causal masking.
2. **RQ2 (Duel Asimetris 6.8B: Heavy Reader vs Deep Reasoner)**:
   * *Hipotesis*: Pada anggaran parameter identik (~6.8B), kita menguji apakah developer agent lebih diuntungkan oleh pemahaman repositori yang dalam (`E4B-E2B`) atau kapasitas penalaran deliberatif (*Thinking Mode*) saat menghasilkan patch perbaikan (`E2B-E4B`).
3. **RQ3 (Batas Skalabilitas Model 9B)**:
   * *Hipotesis*: `JANUS-E4B-E4B` (~9.0B) mampu menyaingi performa model raksasa 27B–31B pada SWE-bench Lite dengan efisiensi memori inferensi yang jauh lebih unggul di hardware konsumen.

---

## ⚡ 2. Efisiensi Arsitektur pada Konteks Repositori Panjang (Long Context)

Pada tugas rekayasa perangkat lunak nyata (SWE-bench, navigasi repositori, refactoring), konteks masukan model terdiri atas pohon berkas, dependensi lintas modul, definisi kelas, dan log pengujian yang mencapai **16.000 hingga 32.000+ token**.

### Masalah Mendasar Model Decoder-Only pada Konteks Panjang:
1. **Kebutaan Kausal (Causal Blindness)**: Model decoder-only memproses token secara sekuensial dari kiri ke kanan. Definisi fungsi atau import di baris 50 tidak dapat melihat pemanggil (*caller*) di baris 2.500 saat token awal diproses.
2. **Ledakan Memori Dynamic KV-Cache saat Generasi**:
   Selama proses pembuatan patch (misalnya $L_{\text{gen}} = 512$ token), model decoder-only harus menyimpan seluruh representasi Key-Value dari *prompt* ($L_{\text{prompt}} = 16.384$ atau $32.768$ token) di VRAM:
   $$\text{Mem}_{\text{dec-only}} = (L_{\text{prompt}} + L_{\text{gen}}) \times 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times \text{bytes}$$
   Pada model 4B dengan $L_{\text{prompt}} = 32\text{k}$, alokasi KV-cache dinamis ini menghabiskan lebih dari 4–8 GB VRAM tambahan dan memperlambat throughput inferensi karena *memory bandwidth starvation*.

### Solusi JANUS-4 (Asymmetric Encoder-Decoder):
1. **Representasi Konteks Utuh Sekali Hitung**:
   Encoder bidirectional memproses seluruh konteks repositori $L_{\text{prompt}}$ secara paralel dalam satu kali forward pass. Hasil representasi $H_{\text{enc}} \in \mathbb{R}^{B \times L_{\text{prompt}} \times D_{\text{enc}}}$ bersifat statis.
2. **Pengurangan KV-Cache Dinamis hingga >95%**:
   Decoder kausal **hanya** menghasilkan kode patch perbaikan. Dynamic autoregressive KV-cache decoder **hanya dialokasikan untuk token patch yang digenerate**:
   $$\text{Mem}_{\text{janus-dec}} = L_{\text{gen}} \times 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times \text{bytes}$$
   $$\text{Penghematan} = 1 - \frac{L_{\text{gen}}}{L_{\text{prompt}} + L_{\text{gen}}} = \frac{16.384}{16.896} \approx \mathbf{96.97\%}$$

| Panjang Prompt ($L_{\text{prompt}}$) | Panjang Patch ($L_{\text{gen}}$) | KV-Cache Decoder-Only | KV-Cache Decoder JANUS | Penghematan Memori Dinamis |
| :--- | :--- | :--- | :--- | :--- |
| **8.192 token** | 512 token | ~1.14 GB | ~0.07 GB | **94.1%** |
| **16.384 token** | 512 token | ~2.21 GB | ~0.07 GB | **96.9%** |
| **32.768 token** | 512 token | ~4.36 GB | ~0.07 GB | **98.4%** |

---

## 🔬 3. Modul Gated Cross-Attention Bridge

Terletak di [`src/janus/models/bridge.py`](file:///c:/Users/HYPE%20AMD/Documents/Coding/project-hackhaton-gemma/src/janus/models/bridge.py):

$$\mathbf{z} = \mathbf{X} + \tanh(g) \cdot \text{CrossAttn}(\mathbf{X}, \mathbf{H}_{\text{enc}})$$

* **Mitigasi Gradient Deadlock**: Parameter gate $g$ diinisialisasi pada $g_0 = \text{atanh}(0.05) \approx 0.05004$.
* **Keunggulan Matematis**:
  * Mempertahankan **95% identitas bobot asli Gemma 4 IT Decoder** pada langkah awal training ($t=0$).
  * Mengalirkan **5% gradien aktif** langsung ke modul encoder sejak langkah pertama untuk mencegah fenomena *gradient starvation*.
  * Dilengkapi **QK-Norm (RMSNorm)** sebelum perkalian skalar untuk menstabilkan logit attention sink saat konteks repositori sangat panjang.

---

## 💻 4. Anggaran Memori Pelatihan Full Presisi 16-Bit (BF16)

Semua konfigurasi dirancang agar dapat dilatih di hardware konsumen (GPU 24GB lokal atau Kaggle 2x T4 32GB total):

| Konfigurasi Model | Bobot Model (16-bit) | VRAM Training: LoRA (r=16) + 8-bit AdamW | VRAM Training: Bridge/Adapter Tuning | Kelayakan Eksekusi |
| :--- | :--- | :--- | :--- | :--- |
| **E2B-E2B (~4.6B)** | **~9.2 GB** | **~11.5 – 13.0 GB** | **~14.5 – 16.5 GB** | Sangat lega di GPU 24GB Lokal & Kaggle Dual T4 |
| **E4B-E2B (~6.8B)** | **~13.6 GB** | **~15.5 – 17.5 GB** | **~18.5 – 20.0 GB** | GPU 24GB Lokal (Headroom ~5 GB) & Kaggle Dual T4 |
| **E2B-E4B (~6.8B)** | **~13.6 GB** | **~15.5 – 17.5 GB** | **~18.5 – 20.0 GB** | GPU 24GB Lokal (Headroom ~5 GB) & Kaggle Dual T4 |
| **E4B-E4B (~9.0B)** | **~18.0 GB** | **~20.5 – 22.5 GB** | **~23.5 – 25.5 GB** | Kaggle Dual T4 (16GB+16GB via Pipeline Parallel) |

---

## 🌳 5. Modul Pemahaman Struktur AST & Graf Kode

Terletak di [`src/janus/models/code_graph_projector.py`](file:///c:/Users/HYPE%20AMD/Documents/Coding/project-hackhaton-gemma/src/janus/models/code_graph_projector.py):

* **AST Node Categorical Embedding**: Memetakan tipe simpul AST (misalnya `function_definition`, `call_expression`, `class_definition` dari Tree-sitter) ke ruang embedding vektor.
* **Hierarchical Scope Depth**: Menginjeksi kedalaman nesting sintaksis dan blok indentasi secara eksplisit.
* **Graph-UL2 Denoising**: Memadukan infilling badan fungsi (Function Body Infilling) dan pemecahan titik pemanggil lintas berkas (Cross-File Call-Site Resolution) pada pra-pelatihan encoder-decoder.
