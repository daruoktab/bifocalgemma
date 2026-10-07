# BifocalGemma: Kecocokan Kompetisi dan Status Verifikasi

Diperiksa 7 Oktober 2026. Target proyek adalah [Google — The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper), terpisah dari [prediction competition](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview).

## Informasi overview yang terkonfirmasi

Halaman overview resmi dapat dibaca melalui hasil indeks pencarian Kaggle pada pemeriksaan ini. Halaman Rules masih tidak menghasilkan teks melalui pembaca web. Informasi overview berikut tidak menggantikan pemeriksaan Rules dan persetujuan peserta di Kaggle.

| Informasi | Bukti overview resmi |
|---|---|
| Paper Track menerima riset terkait yang belum dipublikasikan | Partisipasi pada main competition tidak diwajibkan untuk Paper Track. |
| Deadline paper | 12 November 2026, 23:59 UTC; 13 November 2026, 06:59 WIB. Timeline dapat diperbarui organizer. |
| Penilaian paper | Novelty, Quality, Relevance, Verifiability, Clarity; masing-masing 0–5, hasil akhir rata-rata lima kriteria. |
| Bentuk submission | Kaggle Writeup; draft atau writeup yang belum disubmit tidak dinilai pada deadline. |
| Naskah dan aset | Submission requirements mengonfirmasi maksimal 3.000 kata, non-archival; public notebook dan PDF/project link opsional. |
| Hadiah Paper Track | Total $35.000: Overall Best Paper $15.000, Best New Resource $10.000, Best New Application $10.000. |
| Graph/embedding data yang disediakan | Overview main competition menganjurkan penggunaannya; tidak menyatakannya sebagai prasyarat paper submission. |
| Tie-breaker pada overview paper | Paper yang dimasukkan lebih dahulu menang bila skor seri; jangan menukar kualitas/reproduksibilitas demi mengoptimalkan ini. |
| Model untuk leaderboard prediksi | Overview main competition saat ini hanya mendukung `gemma-4-31b-it-qat-w4a16-ct` untuk semua agents/subagents. |

Matriks BifocalGemma E2B/E4B dengan custom encoder/merged decoder **tidak merupakan submission yang sesuai model selection leaderboard prediksi saat ini**. Arah proyek adalah empirical research pada Paper Track; kelayakan final tetap mengikuti Rules, unpublished status, model/data licenses dan submission requirements yang berlaku.

## Detail Rules yang belum terverifikasi

Batas 3.000 kata, status non-archival dan aset opsional telah dikonfirmasi lewat submission requirements. Ketentuan Apache 2.0 pemenang, eligibility/tim dan syarat distribusi model/data pada Rules Paper Track masih perlu diperiksa langsung sebelum submission. Jangan menyimpulkan seluruhnya sudah benar dari overview atau catatan tim. Isi Rules belum berhasil dibaca dalam pemeriksaan ini.

## Kecocokan scope dengan identitas BifocalGemma

Judul target tetap *BifocalGemma: Evaluating Bidirectional Repository Encoding for Local Code Repair*. Satu-file editing dan function repair adalah **pilot adaptasi**, bukan keseluruhan bukti untuk judul ini.

Studi akhir perlu tiga lapisan:

1. Adaptation checks: patch SFT, valid generation, convergence dan functional unit-level repair; EditPackFT/HumanEvalFix dapat memenuhi bagian awal.
2. Repository-context evidence: context beberapa file dari snapshot sebelum fix, perbaikan yang diterapkan ke snapshot, task yang memang memerlukan dependensi lintas file, serta held-out repository/task splits. CrossCodeEval berguna sebagai completion diagnostic, belum pengganti repair evidence.
3. Architecture evidence: empat pasangan dan dua baseline, data/budget sebanding, context-present/context-removed/context-shuffled ablation, asymmetric capacity allocation, serta latency/VRAM/physical parameters. Klaim tentang reader/generator perlu didukung eksperimen ini.

Tidak ada minimum jumlah dataset/benchmark yang ditemukan pada overview yang dibaca. Mengisi corpus dengan banyak sumber tanpa provenance tidak memperkuat bukti. Gunakan data cukup untuk adaptasi, evaluasi yang mencakup klaim, dan analisis yang dapat direproduksi. Jumlah final ditentukan kualitas kurasi dan budget Molab yang diukur.

Resource Molab mendukung studi bertahap dengan satu model per run, SFT tanpa RL, dan materialisasi static Git snapshots tanpa VM. Native CPU tests untuk repo Python sederhana dapat dirancang, tetapi corpus/scorer belum tersedia dan kompatibilitasnya perlu dibuktikan. Jangan menyebut hasil native subset sebagai skor official SWE-bench atau full CanItEdit.

[Analisis Molab](MOLAB_DATA_EVALUATION_PLAN.md) memberi paket pilot dan gate implementasi. Pilihan primary pada dokumen tersebut berarti primary untuk **pilot S1**, bukan pembatasan scope paper menjadi single-file study. Corpus repository-context dan evaluasi repair menjadi kebutuhan studi akhir bila judul dipertahankan.

Riset yang lebih luas kini menempatkan RES-Q Python sebagai kandidat main repository
editing evaluation dan RepoQA sebagai reader diagnostic, dengan corpus SFT berbasis
issue/patch/Git snapshots. Lihat [katalog dan rekomendasi lengkap](RESEARCH_REPOSITORY_DATA_AND_BENCHMARKS.md).

## Sumber

- [Overview Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper): scope, rubric, timeline, writeup dan awards.
- [Overview prediction competition](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview): model selection dan topik paper yang dianjurkan.
- [Submission requirements](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview/submission-requirements): batas kata, struktur writeup, optional public assets dan non-archival.
- [Rules Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/rules): belum berhasil dibaca melalui tool; pemeriksaan pengguna masih diperlukan sebelum submission final.
