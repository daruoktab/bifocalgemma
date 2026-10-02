# 🏆 JANUS-4: Long-Context Developer Agent Architecture

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Target Competition](https://img.shields.io/badge/Kaggle-Gemma_4_Developer_Agent_Paper_Track-orange)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
[![GitHub Repo](https://img.shields.io/badge/GitHub-janus--4-green)](https://github.com/daruoktab/janus-4)

**Paper Title:**  
*JANUS-4: Looking Both Ways—Complete 2×2 Factorial Analysis of Symmetric and Asymmetric Encoder-Decoder Gemma 4 for Long-Context Developer Agents*

> **Misi Utama Google DeepMind:**  
> *"Today’s best coding agents rely on massive API models running in the cloud. What if every developer could rely on an autonomous agent equally capable offline, on consumer hardware? Post-train an open model into a reliable agent that navigates complex codebases and drafts fixes for real software issues, accelerating developer workflows on everyday hardware."*

---

## 📚 Panduan Kolaborasi Tim (Quick Navigation)

Seluruh dokumen teknis dan panduan pengerjaan telah disusun rapi di folder `docs/`:
* 🏆 [**`docs/COMPETITION_GUIDE.md`**](docs/COMPETITION_GUIDE.md) — Aturan resmi hackathon, rubrik 5 kriteria juri Kaggle, hadiah $35.000, batas 3.000 kata, dan topik resmi (Code Comprehension, Graph Reasoning, Tuning & Optimization).
* 🔬 [**`docs/RESEARCH_SPEC.md`**](docs/RESEARCH_SPEC.md) — Spesifikasi teknis arsitektur model, formula matematika bridge $\tanh(g)$, bukti penghematan KV-cache konteks panjang (>95%), dan tabel VRAM 16-bit.
* 🏛️ [**`docs/GEMMA4_EDGE_ARCHITECTURE.md`**](docs/GEMMA4_EDGE_ARCHITECTURE.md) — Analisis arsitektur lengkap Gemma 4 E2B & E4B hasil verifikasi langsung dari Hugging Face Hub dan Transformers config.
* 👥 [**`docs/TEAM_PLAYBOOK.md`**](docs/TEAM_PLAYBOOK.md) — Pembagian peran 2 anggota tim, roadmap mingguan menuju deadline 12 Nov 2026, serta SOP pengodingan bersama menggunakan `ty` dan `ruff`.

---

## 🏛️ Matriks Faktorial 2×2 Model Gemma 4 Edge

Kami mengeksplorasi seluruh pasangan Encoder (Bidirectional) dan Decoder (Causal IT) dari keluarga Gemma 4 Edge untuk tugas rekayasa perangkat lunak berkonten panjang:

$$\begin{pmatrix} \text{\textbf{JANUS-E2B-E2B}} & \text{\textbf{JANUS-E2B-E4B}} \\ \text{\textbf{JANUS-E4B-E2B}} & \text{\textbf{JANUS-E4B-E4B}} \end{pmatrix}$$

| Konfigurasi Model | Encoder | Decoder (IT) | Parameter Efektif | Bobot 16-bit | Peran & Nilai Ilmiah Utama |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **JANUS-E2B-E2B** | Gemma 4 E2B | Gemma 4 E2B IT | **~4.6B** | **~9.2 GB** | **Core Iso-Parameter Test**: Perbandingan langsung *head-to-head* melawan **Base E4B (4.5B Decoder-Only)** pada pemahaman repositori. |
| **JANUS-E4B-E2B** | Gemma 4 E4B | Gemma 4 E2B IT | **~6.8B** | **~13.6 GB** | **"Heavy Reader, Agile Coder"**: Encoder 4.5B bidirectional menyerap 16k–32k repositori/AST; Decoder 2.3B menghasilkan patch kode dengan kecepatan inferensi maksimal. |
| **JANUS-E2B-E4B** | Gemma 4 E2B | Gemma 4 E4B IT | **~6.8B** | **~13.6 GB** | **"Agile Reader, Deep Reasoner"**: Prefill repositori lincah; memanfaatkan *Thinking Mode* E4B untuk penalaran perbaikan bug yang rumit. |
| **JANUS-E4B-E4B** | Gemma 4 E4B | Gemma 4 E4B IT | **~9.0B** | **~18.0 GB** | **Flagship SOTA Agent**: Menyaingi model raksasa 27B–31B pada SWE-bench Lite dengan jejak memori yang jauh lebih hemat. |

---

## ⚡ 5 Inovasi Inti Rekayasa Perangkat Lunak Konteks Panjang

1. **Bidirectional Long-Context Repository Encoding**: Ingesti repositori multi-file (16k–32k token) secara utuh. Tidak ada batasan *left-to-right causal blindness* maupun *sliding window* 1024 token—setiap fungsi dapat melihat pemanggil (*callers*) dan definisi (*callees*) di masa depan secara simultan.
2. **Asymmetric KV-Cache Decoupling (>95% Memory Reduction)**: Representasi konteks repositori di-encode **sekali** oleh encoder ($H_{\text{enc}}$). Selama generasi kode autoregresif, KV-cache dinamis decoder hanya dialokasikan untuk token patch yang dihasilkan ($L_{\text{gen}} \ll L_{\text{prompt}}$), mengeliminasi lonjakan VRAM saat konteks sangat panjang.
3. **AST-Guided Infilling (Graph-UL2)**: Denoising pre-training berbasis pohon sintaksis Tree-sitter (Function Body Infilling, Cross-File Dependency Infilling, dan Call-Site Resolution).
4. **Multi-Positive Contrastive Localization Loss**: Mengidentifikasi titik perbaikan bug di beberapa file sekaligus dalam satu pull request tanpa mendiskriminasi file yang saling terkait.
5. **Latent Repository Memory dengan 1-Hop Invalidation**: State representasi repositori di-cache di VRAM; saat model menjalankan tool edit file, hanya file yang diedit dan dependensi 1-hop yang di-encode ulang (latensi multi-turn turun hingga 8x).

---

## 📊 Suite Evaluasi Coding (Ringan, Cepat & Tanpa Docker Berat)

Kami menyediakan evaluasi berjenjang yang dapat dijalankan secara instan:
1. **Tier 1 (Instant Algorithmic Synthesis)**: **HumanEval / HumanEval+** & **MBPP / MBPP+** via `evalplus` (selesai dalam ~2 menit di GPU/CPU lokal).
2. **Tier 2 (Long-Context Repository Infilling)**: **HumanEval-Infilling** & **CrossCodeEval** (membuktikan keunggulan natif arsitektur Encoder-Decoder dibanding trik FIM pada decoder-only).
3. **Tier 3 (Real-World Software Engineering)**: **SWE-bench Lite** (subset 300 tugas perbaikan bug nyata dari repositori open-source GitHub).

---

## 🚀 Setup & Standar Pengodingan Tim (Universal & Portabel)

Kedua anggota tim menggunakan alur kerja yang seragam agar hasil linter, formatter, dan type-checker selalu sama:

### 1. Setup Environment Awal
```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

### 2. Standar Pemeriksaan Wajib Sebelum Commit
```powershell
# Jalankan seluruh unit test matriks 2x2
$env:PYTHONPATH = "src"
python -m unittest discover tests

# Format & linter otomatis (Ruff)
ruff format .
ruff check --fix .

# Cek type-safety LSP (Ty — otomatis mendeteksi venv aktif)
ty check .
```

### 3. Eksekusi Evaluasi Cepat
```powershell
# Cetak tabel perbandingan resmi vs baseline Google
python -c "from janus.evaluation import print_comparison_table; print_comparison_table({'humaneval': 63.4, 'mbpp': 56.1}, 'JANUS-E2B-E2B')"
```

---

## 📄 Lisensi Resmi

Seluruh kode dan aset pada repositori ini dilisensikan di bawah [Apache License 2.0](LICENSE) sesuai kewajiban aturan resmi Google Gemma 4 Developer Agent Paper Track.
