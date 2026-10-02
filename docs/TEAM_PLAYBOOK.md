# 👥 Team Playbook & Panduan Standar Kolaborasi (2 Anggota)

> Panduan kolaborasi taktis untuk tim beranggotakan 2 orang agar standar koding, linter, type-safety, dan eksekusi riset **100% identik di laptop masing-masing**, siap sebelum deadline **12 November 2026**.

---

## 🎯 1. Pembagian Peran & Tanggung Jawab

Agar pengerjaan efisien dan tidak tumpang tindih:

### 👤 Anggota A: [@daruoktab](https://github.com/daruoktab) — *Lead Model Architect & Training Engineer*
* Integrasi arsitektur `GatedCrossAttentionBridge` dan `ASTGraphProjector` dengan backbone Hugging Face `google/gemma-4-E2B-it` dan `google/gemma-4-E4B-it`.
* Implementasi pipeline pelatihan (LoRA / adapter tuning) dengan presisi 16-bit (BF16) dan optimizer 8-bit AdamW.
* Manajemen komputasi GPU (GPU 24GB lokal dan Kaggle 2x T4).
* Penyimpanan checkpoint model untuk evaluasi matriks $2 \times 2$.

### 👤 Anggota B: [@attmhd](https://github.com/attmhd) — *Lead Evaluation, Data Curator & Paper Writer*
* Kurasi dataset niche software engineering berkonten panjang (CommitPackFT, repositori open-source Python, SWE-bench Lite) menggunakan `tree-sitter` untuk ekstraksi AST.
* Eksekusi benchmark instan (`evalplus` HumanEval/MBPP, HumanEval-Infilling, SWE-bench Lite) dan pembuatan tabel/grafik perbandingan vs baseline Google.
* Penyusunan draf naskah riset (Kaggle Writeup & LaTeX preprint berstandar 3.000 kata).
* Pembuatan Kaggle Public Demo Notebook untuk lampiran submission.

---

## ⚙️ 2. Standar Setup Environment (Bebas Masalah Folder Berbeda)

Kedua anggota tim menggunakan standar tool modern yang sama (`uv`, `ruff`, dan `ty`) sehingga konfigurasi project sepenuhnya portabel:

```powershell
# 1. Clone repositori
git clone https://github.com/daruoktab/janus-4.git
cd janus-4

# 2. Buat virtual environment lokal masing-masing
uv venv
.venv\Scripts\Activate.ps1

# 3. Install seluruh dependensi dan modul lokal dalam mode editable
uv pip install -e ".[dev]"
```

---

## ⚡ 3. SOP Pengodingan Bersama (Cheat Sheet Wajib Sebelum Commit)

Sebelum melakukan `git push`, kedua anggota **wajib menjalankan 3 perintah inti ini** agar hasil koding, formatting, dan type-checking selalu seragam:

```powershell
# 1. Pastikan modul src terbaca dan jalankan seluruh Unit Test Matriks 2x2
$env:PYTHONPATH = "src"
python -m unittest discover tests

# 2. Format kode & cek linter otomatis (Ruff)
ruff format .
ruff check --fix .

# 3. Validasi Type-Safety LSP (Ty) — otomatis mendeteksi venv aktif
ty check .
```

*Jika ketiga perintah di atas menampilkan **"All checks passed!"**, maka kode dijamin bebas bug sintaksis dan aman di-push ke GitHub.*

---

## 📅 4. Jadwal & Milestone Taktis Menuju Deadline (12 Nov 2026)

```
TIMELINE PENGERJAAN:
┌─────────────────────┬────────────────────────────────────────────────────────┐
│ Periode             │ Target & Deliverable                                   │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ Minggu 1 (Sekarang) │ • Setup repo & modul dasar (SELESAI ✅)                │
│                     │ • Uji forward/backward pass matriks 2x2 (SELESAI ✅)   │
│                     │ • Persiapan dataset kurikulum infilling AST (<350M tok)│
├─────────────────────┼────────────────────────────────────────────────────────┤
│ Minggu 2 (Okt 2026) │ • Pelatihan varian utama: JANUS-E2B-E2B (~4.6B)        │
│                     │ • Evaluasi head-to-head vs Base E4B (RQ1 selesai)      │
│                     │ • Kerangka draf naskah paper (Introduction & Methods)  │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ Minggu 3 (Okt 2026) │ • Pelatihan varian asimetris: E4B-E2B & E2B-E4B (RQ2)  │
│                     │ • Analisis trade-off kapasitas pemahaman vs generasi   │
│                     │ • Pelatihan varian flagship: JANUS-E4B-E4B (RQ3)       │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ Minggu 4 (Awal Nov) │ • Pengujian lengkap HumanEval-Infilling & SWE-bench    │
│                     │ • Finalisasi naskah paper 3.000 kata (Kaggle Writeup)  │
│                     │ • Pembuatan Kaggle Public Demo Notebook                │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ 8 - 10 Nov 2026     │ • Internal Review & Freeze Submission                  │
│                     │ • Submit lebih awal ke Kaggle (Tie-breaker advantage!) │
└─────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 🔄 5. Sinkronisasi Git Harian

```powershell
git pull origin main
git add .
git commit -m "feat/eval/docs: deskripsi perubahan"
git push origin main
```
