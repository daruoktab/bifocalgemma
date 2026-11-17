# Qwen3.5 dan seluruh seri Gemma 4: pencarian lanjutan benchmark

Tanggal pemeriksaan: 18 November 2026. Ini melengkapi [audit protokol](OFFICIAL_BENCHMARK_PROTOCOL_AUDIT_2026-11-17.md) dan [pembanding sub-7B](SUB7B_BENCHMARK_REFERENCES_2026-11-18.md). Riset awal terlalu membatasi pencarian Gemma 4 pada model card dan laporan rilis. Pencarian lanjutan menemukan skor **original checkpoint sebelum tambahan training peneliti**, terutama pada GSM8K, dalam paper riset lain dan laporan evaluator. Temuan tersebut tidak mengubah hasil run proyek.

Delapan kasus kita berasal dari lima keluarga: LongCodeQA, HumanEvalFix, SWE-QA (dua kondisi), GSM8K (dua shot setting), dan CrossCodeLongEval (dua jenis completion). **Hasil full benchmark tetap merupakan referensi yang relevan**, walaupun dua completion lokal hanya disampel 100. Kesamaan task dan kesetaraan protokol adalah dua pemeriksaan berbeda; pencarian ini tidak mensyaratkan bahwa sumber harus memakai sampel 100 lokal.

## 1. Hasil yang terverifikasi

### Gemma 4 E2B/E4B: paper riset dengan afiliasi Google/Ulm

[On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models, v1](https://arxiv.org/html/2610.32770v1#S4.T1), diajukan 23 Oktober 2026, memiliki penulis berafiliasi Google dan Ulm University. Ini **paper riset berafiliasi Google**, bukan tabel rilis/model card Gemma atau skor benchmark creator GSM8K. Karena itu, menyebut semua bukti baru sebagai sepenuhnya di luar Google juga tidak tepat.

| Original checkpoint pada row `Base` | GSM8K accuracy | Setting |
| --- | ---:| --- |
| Gemma 4 E2B instruction-tuned | **72.7%** | Eight-shot CoT, thinking |
| Gemma 4 E4B instruction-tuned | **84.4%** | Eight-shot CoT, thinking |

Paper menjelaskan bahwa `Base` berarti **instruction-tuned model sebelum self-training mereka**, bukan pretrained base checkpoint dan bukan model OPST/SFT/GRPO hasil eksperimen. GSM8K hanya diukur pada thinking regime. [Table 1](https://arxiv.org/html/2610.32770v1#S4.T1) dan [Appendix A.6](https://arxiv.org/html/2610.32770v1#A1.SS6) menyebut eight-shot CoT, adaptasi lm-evaluation-harness ke Gemma chat template, dan exact match jawaban numerik akhir. Jumlah contoh GSM8K tidak dinyatakan secara eksplisit dalam bagian yang diperiksa; jangan menambahkan klaim full 1319 hanya dari nama dataset. Temperature, generation cap, pinned checkpoint revision dan kesamaan dengan delapan exemplar lokal juga belum terkonfirmasi.

### Gemma 4 31B: full-test eight-shot, laporan evaluator quantization

Pembuat artefak quantization melaporkan baseline **`google/gemma-4-31B-it` BF16 96.74% (1276/1319)** pada seluruh test GSM8K, eight-shot, vLLM, temperature 0, maximum 512 token, chat template dan format `####`. Ini baseline original BF16; skor INT8 mereka yang terpisah adalah 96.66% (1275/1319). Sumber ini merupakan **laporan evaluator langsung pada model card pihak ketiga**, bukan angka resmi Google atau paper peer-reviewed. [Card dan metode](https://huggingface.co/nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8).

### Gemma 4 zero-shot: subset paper, versi harus dikunci

[Unified Deployment-Aware Evaluation of Open Reasoning Language Models, v3 Table 1](https://arxiv.org/html/2604.07035v3#S4.T1) melaporkan zero-shot **26B-A4B 79.4%** dan **E4B 79.0%**, masing-masing pada **238 matched examples**, bukan full-test. Setup memakai BF16, greedy temperature 0, batch 1 dan cap GSM8K 512 token. Ini penelitian pihak lain dengan prompt/extractor tersendiri. Versi lama v1 melaporkan angka lain pada 100 kasus dan salah memberi label arsitektur E2B/E4B sebagai MoE; hasil v1 tidak boleh dipakai seolah hasil v3. V3 juga memuat robustness matrix 500 soal; jangan menukar denominator antar-tabel. Sumber ini lebih berguna sebagai contoh sensitivitas protokol daripada ambang akurasi original yang universal.

### Qwen3.5-4B: full-test eight-shot CoT

[Beyond K/V Caches: Cross-Model State Handoff in Hybrid Linear-Attention LLMs, v1](https://arxiv.org/html/2611.08133v1#A1.T3), diajukan 5 November 2026, melaporkan row **4B Native: GSM8K 89.39%**. Setup mengidentifikasi model sebagai **Qwen3.5-4B**, memakai seluruh **1319 test examples** dan **eight-shot CoT**, dengan BF16 model weights. Angka `Native` adalah model asli, bukan konfigurasi state handoff terlatih atau model 4B→27B. [Setup §5.1](https://arxiv.org/html/2611.08133v1#S5.SS1), [Appendix A Table 3](https://arxiv.org/html/2611.08133v1#A1.T3). Ini pengukuran penulis paper, bukan publikasi skor Qwen Team. Exact decoding cap, thinking mode, exemplar dan parser parity dengan lokal masih perlu dibuktikan.

Paper lain, [Beyond the Assistant Turn, v2 Table 3](https://arxiv.org/html/2604.02315v2#A1.T3), mengukur GSM8K secara langsung pada 1319 kasus dengan generasi terutama temperature 0 dan ekstraksi final numeric answer:

| Original Qwen3.5 | GSM8K answer accuracy |
| --- | ---:|
| 0.8B | 41.6% |
| 2B | 62.2% |
| 4B | 86.5% |

Kolom yang dipakai adalah **GSM8K Acc.**, bukan genuine-followup rate. Jumlah shot dan cap tidak ditentukan secara eksplisit dalam setup yang diperiksa; karena itu nilai-nilai ini tidak diberi label zero-shot/eight-shot di sini. Paper yang sama memiliki eksperimen post-training lain dengan angka 2B berbeda; row original scaling Table 3 jangan diganti dengan hasil trained intervention Table 2.

### Qwen3.5: ada angka resmi GSM8K, tetapi untuk model besar pretrained

[Blog resmi Qwen3.5, bagian Pretraining](https://qwen.ai/blog?id=qwen3.5#pretraining) melaporkan **Qwen3.5-397B-A17B Base 93.71% GSM8K**. Itu **397B total / 17B active, pretrained**, bukan Qwen3.5-4B atau instruction-tuned. Shot count tidak ditentukan pada tabel tersebut. Angka ini membuktikan adanya laporan resmi pada keluarga GSM8K, tetapi tidak boleh dimasukkan ke kolom kecil sub-7B atau setting 0/8-shot yang belum terverifikasi.

## 2. Status delapan kasus

“Belum ditemukan” di sini dibatasi sumber dan pencarian yang diperiksa, bukan klaim bahwa hasil tidak ada di mana pun. Kolom mencakup paper riset selain laporan rilis, serta seluruh seri Gemma 4, bukan hanya E2B/E4B.

| Kasus pilihan kita | Qwen3.5 | Gemma 4 series |
| --- | --- | --- |
| LongCodeQA 32K | Belum ditemukan skor exact-task terverifikasi | Belum ditemukan skor exact-task terverifikasi |
| HumanEvalFix Python | Ada baseline **9B 80.5%** pada varian **Tests**, paper SciWalker; bukan sub-7B atau adapter diff lokal | Belum ditemukan skor exact-task terverifikasi |
| SWE-QA oracle, 9072 MCQ | Belum ditemukan | Belum ditemukan |
| SWE-QA noisy oracle, 9072 MCQ | Belum ditemukan | Belum ditemukan |
| GSM8K zero-shot | Ada hasil GSM8K, tetapi shot count sumber small-model Table 3 belum terkonfirmasi; jangan dipaksa menjadi zero-shot | E4B **79.0%**, 26B-A4B **79.4%**, subset 238, paper v3 |
| GSM8K eight-shot | **4B Native 89.39%**, seluruh 1319 soal, paper handoff | **E2B 72.7%, E4B 84.4%**, thinking, paper Google/Ulm; **31B-it BF16 96.74%**, seluruh 1319 soal, laporan evaluator |
| CrossCodeLongEval chunk | Belum ditemukan skor exact-task terverifikasi | Belum ditemukan skor exact-task terverifikasi |
| CrossCodeLongEval function | Belum ditemukan skor exact-task terverifikasi | Belum ditemukan skor exact-task terverifikasi |

Qwen3.5-9B HumanEvalFix-Python **80.5%** berasal dari row baseline, [SciWalker v1 Table 2](https://arxiv.org/html/2609.30054v1#S5.T2). Footnote menyebut **test version**. Evaluasi menggunakan thinking, cap 32768, temperature .6/top-p .95/top-k 20; tidak setara dengan adapter docstring-to-diff greedy lokal. Paper juga melaporkan GSM8K baseline 9B 91.1%, tetapi shot count belum terverifikasi sehingga tidak dipindahkan ke dua setting kita.

## 3. Salah identitas yang disaring

- Search result Qwen3.5 pada **SWE-QA** dapat mengarah ke benchmark repository agent dengan **720 pertanyaan dan judge score**, dari Peng dkk., bukan dataset **9072 MCQ oracle/noisy** Elkoussy/Perez yang dipakai proyek. Angka [Harness Annealing](https://arxiv.org/html/2610.01235#S4) tidak dipakai untuk mengisi dua baris SWE-QA kita.
- Card [Med-Qwen](https://huggingface.co/adi274903/Med-Qwen-3.5-4B-it) memberi tabel GSM8K eight-shot pada 150 contoh, termasuk E4B 80.7 dan Qwen3.5-4B 85.3, tetapi melabeli metode **log-likelihood tanpa reasoning**. Itu tidak dianggap bukti generated-final-answer accuracy yang sebanding tanpa pemeriksaan harness lebih jauh.
- GSM8K-Platinum, LongBench, LongCodeArena, HumanEval biasa, dan LiveCodeBench tetap tidak menggantikan benchmark kita yang berbeda nama/protokol.

## 4. Batas klaim dan tindak lanjut

Model card dan technical report utama Google yang diperiksa belum memuat delapan kasus ini sebagai tabel hasil, termasuk GSM8K. Namun, ini **tidak berarti tidak ada hasil Gemma 4 di paper lain**; temuan Google/Ulm di atas memperbaiki cakupan jawaban sebelumnya. [Model card seri](https://ai.google.dev/gemma/docs/core/model_card_4), [technical report v1](https://arxiv.org/html/2607.02770v1).

Untuk perbandingan langsung, catat original checkpoint, task variant, jumlah shot/exemplars, thinking mode, output cap, parser, sample IDs dan denominator. Proyek meminta non-thinking dan mempunyai extractor tersendiri; skor thinking pada paper bukan target reproduksi otomatis. Full-test dapat dipakai sebagai referensi eksternal, sedangkan pengujian pada 100 ID lokal yang sama diperlukan bila ingin mengisolasi perbedaan model pada cohort lokal.

Perubahan riset ini hanya dokumentasi. Source runtime, evaluator, checkpoint, notebook, runtime ZIP, proses Molab dan placeholder hasil paper tidak diubah. Angka eksternal tidak diisikan sebagai hasil BifocalGemma.
