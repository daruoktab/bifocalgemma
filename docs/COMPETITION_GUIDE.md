# 🏆 Panduan Resmi Kompetisi: The Gemma 4 Developer Agent Paper Track

> Dokumen acuan resmi untuk tim pengembang **JANUS-4**. Disarikan langsung dari panduan Kaggle & Google DeepMind.

---

## 📌 1. Esensi & Misi Kompetisi

> *"Today’s best coding agents rely on massive API models running in the cloud. What if every developer could rely on an autonomous agent equally capable offline, on consumer hardware? Post-train an open model into a reliable agent that navigates complex codebases and drafts fixes for real software issues, accelerating developer workflows on everyday hardware."*

Google DeepMind menyelenggarakan **Paper Track** ini untuk mendobrak ketergantungan developer pada model API raksasa berbasis cloud yang mahal dan membutuhkan koneksi internet konstan. Tujuannya adalah menciptakan agen rekayasa perangkat lunak otonom (*autonomous software engineering agent*) yang:
1. Mampu membaca dan memahami repositori kode besar (*large complex codebases*) dengan konteks panjang (16k–32k+ token).
2. Mampu merancang dan menghasilkan perbaikan bug (*draft real fixes*) dan penambahan fitur secara mandiri.
3. Dapat dilatih dan dijalankan pada **hardware konsumen / lokal (single accelerator / everyday hardware)** tanpa membutuhkan infrastruktur server komersial raksasa.

---

## 📅 2. Timeline & Ketentuan Pengumpulan

* **Mulai Kompetisi**: 22 September 2026
* **Deadline Akhir Pengumpulan**: **12 November 2026 (23:59 UTC)**
* **Maksimum Panjang Naskah**: **3.000 kata** (*concise, dense, and impactful*).
* **Format Pengumpulan**:
  * **Kaggle Writeup**: Laporan riset langsung di platform Kaggle (berisi: Judul, Subjudul, Abstrak, Pendahuluan, Deskripsi Riset/Metode/Eksperimen, Related Works & Sitasi).
  * **(Opsional & Sangat Disarankan) Public Project Link**: Lampiran paper PDF format preprint (LaTeX/arXiv template) siap publikasi.
  * **(Opsional) Public Kaggle Notebook**: Notebook demonstrasi inferensi publik agar juri dapat menguji model kita dengan 1 klik.
* **Sifat Publikasi (Non-Archival)**: Pengumpulan ke Kaggle Paper Track **tidak menggugurkan hak kita** untuk mengirimkan paper ini ke konferensi AI bergengsi lainnya (seperti ICLR / ACL / NeurIPS workshop).
* **Lisensi Pemenang**: Wajib open-source di bawah lisensi **Apache 2.0**.

---

## 💰 3. Total Hadiah ($35.000) & Kategori Juara

Pemenang akan disorot pada acara resmi yang diselenggarakan Google (seperti **NeurIPS Expo Workshop 2026**):
1. **Overall Best Paper ($15.000)**: Diberikan kepada submission dengan nilai rata-rata tertinggi dari 5 kriteria penjurian.
2. **Best New Resource ($10.000)**: Diberikan untuk penciptaan dataset baru, pustaka (*library*), atau perkakas rekayasa agen baru.
3. **Best New Application ($10.000)**: Untuk inovasi atau aplikasi kreatif terkait representasi *code-graphs* dan embeddings.

---

## ⚖️ 4. Rubrik Penjurian Resmi (Skala 0 – 5 per Kriteria)

Penilaian dilakukan secara setara (bobot 20% per kategori) dengan nilai akhir adalah rata-rata dari kelimanya:

| Kategori | Pertanyaan Kunci Dewan Juri | Strategi Kita di JANUS-4 |
| :--- | :--- | :--- |
| **Novelty** | Apakah karya memberikan wawasan baru (*new insights*) atau menonjolkan sifat penting dari metode yang ada? | Paper pertama yang mengadaptasi arsitektur **Gemma 4 Edge (E2B & E4B)** menjadi **Encoder-Decoder Asimetris Konteks Panjang** dengan *Gated Cross-Attention Bridge* dan *AST Graph Projector*. |
| **Quality** | Seberapa umum dan universal pendekatan yang diajukan di luar kompetisi? Apakah tergeneralisasi dengan baik? | Menyajikan **Matriks Faktorial 2×2 Lengkap** (E2B-E2B, E4B-E2B, E2B-E4B, E4B-E4B) yang menguji trade-off kapasitas pemahaman repositori vs generasi patch secara universal. |
| **Relevance** | Seberapa signifikan dampak riset ini bagi bidang *software engineering* dan pembelajaran berbasis agen (*agentic learning*)? | Menyelesaikan dua hambatan terbesar SWE agent lokal: (1) lonjakan memori KV-cache pada konteks repositori panjang (dipangkas >95% saat generasi patch), dan (2) keterbatasan causal masking dalam melihat dependensi dua arah pada basis kode. |
| **Verifiability** | Apakah paper memuat informasi yang cukup untuk memahami cara kerja inovasi, perolehan data, dan analisisnya? | Seluruh kode di-open-source (Apache 2.0), dataset publik terverifikasi (SWE-bench Lite, CommitPackFT, HumanEval-Infilling), unit tests mandiri, dan lingkungan komputasi transparan (16-bit BF16 pada GPU 24GB / Kaggle Dual T4). |
| **Clarity** | Apakah naskah paper ditulis dan disajikan dengan sangat jelas dan terstruktur? | Dilengkapi formulasi matematika efisiensi KV-cache presisi, diagram alur arsitektur 2x2 standar publikasi, tabel ablasi rapi, dan bahasa akademik yang padat. |

---

## 🎯 5. Keselarasan Proyek Kita dengan 4 Topik Resmi Google

Kaggle menyarankan 4 topik utama, dan proyek **JANUS-4** kita mencakup **keempat-empatnya sekaligus**:

1. **Tuning & Optimization**: Mengadaptasi model causal Gemma 4 Edge menjadi encoder-decoder dengan kurikulum niche infilling (<350M token) dan PEFT/LoRA presisi 16-bit.
2. **Code Comprehension**: Menggunakan Bidirectional Encoder untuk membaca pohon sintaksis AST Tree-sitter dan dependensi repositori secara holistik tanpa blind spot kausalitas.
3. **Tasks & Benchmarks**: Menghadirkan suite evaluasi bertingkat (EvalPlus HumanEval/MBPP, HumanEval-Infilling, dan SWE-bench Lite).
4. **Graph Reasoning**: Memetakan struktur AST dan relasi pemanggilan lintas berkas melalui `ASTGraphProjector` dan *Latent Repository Memory dengan 1-Hop Invalidation*.
