# BifocalGemma: Studi Empat Pasangan Base–IT dan Dua Baseline

Notasi `E2B–E4B IT` berarti **encoder Gemma 4 E2B base** dan
**decoder Gemma 4 E4B instruction-tuned**. Suffix IT selalu menjelaskan decoder.
Rancangan utama mencakup empat pasangan; semua merupakan target eksperimen paper.
E2B–E2B boleh menjadi pilot integrasi pertama, tetapi tiga pasangan lain tetap
bagian studi utama. Tidak ada training model penuh dijalankan di GPU lokal saat ini.

## Enam model dalam perbandingan

| ID pelaporan | Encoder checkpoint | Decoder checkpoint | Peran |
| --- | --- | --- | --- |
| BifocalGemma-E2B-E2B | google/gemma-4-E2B | google/gemma-4-E2B-it | Pasangan simetris kecil |
| BifocalGemma-E4B-E4B | google/gemma-4-E4B | google/gemma-4-E4B-it | Pasangan simetris besar |
| BifocalGemma-E2B-E4B | google/gemma-4-E2B | google/gemma-4-E4B-it | Encoder kecil, decoder besar |
| BifocalGemma-E4B-E2B | google/gemma-4-E4B | google/gemma-4-E2B-it | Encoder besar, decoder kecil |
| Gemma-4-E2B-IT | Tidak ada encoder repositori terpisah | google/gemma-4-E2B-it | Baseline decoder-only |
| Gemma-4-E4B-IT | Tidak ada encoder repositori terpisah | google/gemma-4-E4B-it | Baseline decoder-only |

T5Gemma, T5Gemma 2, Gemma 3, dan keluarga lain bukan pembanding benchmark utama.
T5Gemma tetap dibahas dalam related work. Model yang sama diuji pada task/context
yang sama; model registries dan tabel pelaporan tidak dianggap harness benchmark.

## Tiga kontribusi yang ditargetkan

1. **Adaptasi encoder-decoder kelas Edge untuk codebase understanding dan repair.**
   Evaluasi berfokus pada dependensi lintas file, lokalisasi bug, dan patch yang lolos
   tes. Ini kontribusi domain/metode yang harus dibuktikan, bukan klaim bahwa
   encoder-decoder coding baru muncul pertama kali; [CodeT5](https://arxiv.org/abs/2109.00859)
   sudah meneliti code understanding dan generation.
2. **Inisialisasi campuran: base encoder + IT decoder.** Encoder dilatih untuk
   representasi bidirectional; decoder dimulai dari checkpoint yang sudah
   instruction-tuned. Kandidat tujuan adalah menjaga kemampuan mengikuti instruksi
   saat menghubungkan konteks repositori. Gate kecil tidak menjamin retention.
   Tanpa kontrol init alternatif, enam model ini tidak mengisolasi keuntungan
   initialization campuran; recipe merupakan desain metode, bukan klaim sebab-akibat.
3. **Studi alokasi kapasitas encoder versus decoder.** Matriks penuh memungkinkan
   pengukuran efek kedua faktor dan interaksinya. Duel E4B–E2B IT versus E2B–E4B IT
   adalah perbandingan alokasi kapasitas dengan jumlah parameter text awal sama.

[T5Gemma](https://arxiv.org/abs/2504.06225) menyediakan dasar adaptation dan
konfigurasi asimetris. [T5Gemma 2](https://arxiv.org/abs/2512.14856) menambah
long-context/multimodal, tied embeddings dan merged decoder attention, dengan
inisialisasi dari pretrained checkpoint diikuti UL2 dan post-training.
BifocalGemma saat ini memakai gated cross-attention terpisah. Menggabungkan inspirasi
kedua paper tidak berarti mekanisme T5Gemma 2 tersebut sudah diimplementasikan.
Nama atau recipe ini juga tidak otomatis membuktikan novelty terhadap seluruh literatur.

## Pertanyaan dan perbandingan utama

- **RQ1 — Manfaat encoder repositori:** bandingkan setiap pasangan dengan baseline
  IT yang ukuran decodernya sama. E2B–E2B IT versus E4B-IT menjadi perbandingan
  khusus yang menarik: apakah reader kecil plus generator kecil memberi kualitas
  repair lebih baik daripada decoder-only berkapasitas efektif lebih besar?
- **RQ2 — Alokasi kapasitas:** bandingkan E4B–E2B IT dengan E2B–E4B IT. Ukur
  kualitas repair/lokalisasi sekaligus latensi, throughput dan peak VRAM.
- **RQ3 — Efek faktor dan interaksi:** ukur perubahan saat ukuran encoder naik
  dengan decoder tetap, dan saat decoder naik dengan encoder tetap. Jangan memakai
  satu duel saja untuk menyimpulkan encoder selalu lebih penting.

Jika S22, S24, S42 dan S44 adalah skor task yang sama dari empat pasangan,
efek ukuran encoder diringkas sebagai `((S42−S22)+(S44−S24))/2`;
efek ukuran decoder sebagai `((S24−S22)+(S44−S42))/2`;
interaksi sebagai `S44−S42−S24+S22`. Laporkan juga setiap sel dan per-task outcomes,
bukan hanya rata-rata faktor. Gunakan paired uncertainty estimates dan seed yang
tercatat; kesimpulan dibatasi keluarga model serta tugas yang benar-benar diuji.
E2B/E4B berbeda pada depth, width, MLP dan konfigurasi KV; efek faktor
menjelaskan pilihan varian dalam keluarga ini, bukan hukum universal jumlah parameter.

## Kontrol yang diperlukan

- Pin checkpoint revisions, data training/split dan held-out task IDs. Untuk
  atribusi efek arsitektur, baseline menerima data/kurikulum repair dan budget
  token adaptasi yang sebanding; skor checkpoint IT bawaan harus diberi label
  zero-shot jika dilaporkan. Membandingkan model teradaptasi dengan baseline
  zero-shot mengukur peningkatan sistem secara keseluruhan.
- Gunakan context selection/retrieval, sampling, token/tool budget, tool suite dan
  test harness yang sama. Baseline mendapat prefix caching bila didukung. Ikuti
  formatting BOS dan stopping policy base/IT yang benar.
- Tetapkan bridge layout yang sama secara metodologis. Khusus duel asimetris,
  kontrol inner projection width, jumlah bridge, trainable parameters dan biaya
  cross-KV; jumlah decoder layers 35 versus 42 bisa menambah confound.
  Penempatan di global layers saja bukan keputusan default yang diasumsikan
  bebas trade-off; [T5Gemma 2](https://arxiv.org/html/2512.14856v2#S2)
  melaporkan penurunan kualitas dalam ablation tersebut.
- Laporkan effective, physical dan trainable parameters, precision/quantization,
  training tokens/FLOPs/wall time, latensi cold/warm, dan peak VRAM. Jika budget
  atau precision berbeda, nyatakan perbandingan sebagai trade-off yang terukur.

Sebelum bridge, parameter fisik text pasangan E2B–E2B sekitar **9,26B**, sedangkan
E4B-IT tunggal sekitar **7,46B**. Jadi kemenangan E2B–E2B IT tetap menarik, tetapi
bukan perbandingan iso-parameter fisik atau iso-memory. Kedua pasangan asimetris
memiliki sekitar **12,09B** parameter text sebelum bridge. Biaya dan retention
instruction/tool behavior perlu diukur, bukan disimpulkan dari nama ukuran.

## Data, pelaporan, dan status

[Riset dataset](DATASETS_AND_BENCHMARKS.md) tetap menjadi sumber protokol data.
SWE-bench Lite/Verified mengukur issue resolution; CrossCodeEval merupakan
completion diagnostic, dan lokalisasi dilaporkan terpisah. Skor kosong adalah N/A.

`print_repair_experiment_table` menerima dictionary hasil yang di-key oleh
keenam ID di atas dan mencetak enam baris tetap. Nilai dummy harus menggunakan
`illustrative=True`; fungsi ini tidak mengeksekusi model atau benchmark.

Urutan kerja: pilot integrasi E2B–E2B, bekukan protokol, jalankan empat pasangan
secara bertahap beserta dua baseline, lalu analisis faktor. Full physical configs
baru diuji pada meta, versi kecil full-depth, serta bridge CUDA. Hasil code repair
keenam model dan keuntungan recipe initialization belum tersedia. Multimodal,
partial cache invalidation dan perubahan merged attention menjadi riset terpisah.
