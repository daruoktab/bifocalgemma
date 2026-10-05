# BifocalGemma: Catatan Kompetisi dan Status Verifikasi

**Target proyek:** [Google — The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper).

Dokumen ini adalah catatan kerja tim, bukan aturan resmi kompetisi. Pada pemeriksaan
**5 Oktober 2026**, halaman Kaggle tidak menampilkan isi aturan melalui pembaca web
yang digunakan. Karena itu detail di bawah berasal dari catatan proyek sebelumnya
dan **belum terverifikasi**; tim harus memeriksanya langsung di halaman resmi sebelum
memakai informasi tersebut untuk timeline, publikasi, atau submission.

## Catatan yang harus dikonfirmasi

| Detail dari catatan sebelumnya | Status |
| --- | --- |
| Mulai 22 September 2026 | Belum terverifikasi |
| Deadline 12 November 2026, 23:59 UTC | Belum terverifikasi; bila benar, 13 November 2026, 06:59 WIB |
| Batas naskah 3.000 kata | Belum terverifikasi |
| Hadiah total $35.000 dan kategori $15.000/$10.000/$10.000 | Belum terverifikasi |
| Rubrik Novelty, Quality, Relevance, Verifiability, Clarity dengan bobot sama | Belum terverifikasi |
| Writeup Kaggle, lampiran PDF dan notebook publik | Belum terverifikasi |
| Publikasi non-archival dan hak submission konferensi | Belum terverifikasi |
| Kewajiban lisensi pemenang Apache 2.0 | Belum terverifikasi; kode proyek sendiri sudah memakai Apache 2.0 |
| Presentasi atau sorotan NeurIPS 2026 Google Expo | Belum terverifikasi |
| Topik resmi serta ketentuan model, data, hardware dan tim | Belum terverifikasi |

Setelah membaca halaman resmi, tambahkan URL bagian aturan, tanggal pemeriksaan,
dan ringkasan yang terverifikasi. Tidak ada asumsi keuntungan tie-breaker untuk
submission lebih awal tanpa aturan resmi yang mendukungnya.

## Strategi submission proyek

BifocalGemma berfokus pada riset representasi repositori bidirectional untuk local
code repair. Proposal tidak mengklaim otomatis mencakup semua kategori kompetisi,
menjadi karya pertama, atau menjamin kemenangan. Fokus eksperimen awal adalah satu
konfigurasi kecil dan bukti pada tugas lintas file.

- **Kontribusi:** jelaskan hubungan dengan T5Gemma dan kontribusi spesifik untuk kode.
- **Kualitas:** baseline sebanding, ablation yang menjawab pertanyaan riset, dan analisis kegagalan.
- **Relevansi:** patch yang benar pada tugas nyata serta biaya memori/latensi terukur.
- **Reproduksi:** pin data, config, model revision, environment, harness, serta hardware.
- **Kejelasan:** pisahkan implementasi, rencana, hipotesis, contoh angka, dan hasil terukur.

Daftar ini merupakan standar kerja tim; penggunaannya tidak menyatakan rubrik
penjurian resmi. Lihat [proposal](proposal.md), [spesifikasi](RESEARCH_SPEC.md), dan
[playbook](TEAM_PLAYBOOK.md) untuk eksperimen serta bukti yang diperlukan.
