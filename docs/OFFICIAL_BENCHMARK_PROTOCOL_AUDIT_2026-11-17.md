# Audit sumber resmi dan protokol benchmark — 17 November 2026

Audit ini memeriksa delapan kasus aktif pada source `80522ad`, dengan HEAD awal `0ff9943`; riset dimulai 17 November dan dilanjutkan 18 November 2026. Sol memeriksa evaluator, Terra menelusuri sumber pembuat benchmark, dan Luna memverifikasi tabel/model serta varian HumanEvalFix. Tidak ada training, generasi model, unduhan bobot, perubahan evaluator, atau perubahan pada baseline Molab yang sedang berjalan. Probe memakai interpreter proyek `C:/Users/daru/miniconda3/envs/ai/python.exe` dan tokenizer lokal secara offline. Permintaan lanjutan tentang model seukuran dibahas pada [pembanding di bawah 7B](SUB7B_BENCHMARK_REFERENCES_2026-11-18.md).

## Kesimpulan

Kecurigaan tentang ketidaksetaraan benchmark beralasan: evaluasi lokal adalah protokol adaptasi, sehingga hasilnya tidak otomatis setara dengan tabel pembuat benchmark. Perbedaan paling jelas adalah prompt/perintah unified diff pada HumanEvalFix, subset dan metrik CrossCodeLongEval, serta extractor dan contoh few-shot pada tugas QA/GSM8K. Ini belum membuktikan penyebab selisih skor model tertentu; audit ini tidak menghasilkan skor model baru.

Dalam **model card dan laporan rilis utama Google yang diperiksa**, tidak ditemukan skor original Gemma 4 E2B-it/E4B-it untuk delapan varian lokal persis ini. Cakupan itu tidak mencakup semua paper peneliti Google: **pencarian lanjutan menemukan baseline GSM8K eight-shot thinking E2B 72.7% dan E4B 84.4% pada paper Google/Ulm**, serta pengukuran eksternal lain. Lihat [laporan lanjutan Qwen3.5/Gemma 4](QWEN35_GEMMA4_BENCHMARK_FOLLOWUP_2026-11-18.md). Angka tersebut adalah referensi task yang relevan, tetapi belum menjadi target reproduksi langsung untuk protokol lokal.

Ada satu bug yang direproduksi pada **parser MCQ resilient/diagnostik**, bukan pada headline strict. Metrik completion yang menghapus semua whitespace juga menyamakan kode yang berbeda semantik. Sebaliknya, tes HumanEvalFix benar-benar dieksekusi ketika execution diaktifkan; dugaan bahwa scorer hanya memeriksa syntax atau tidak memanggil fungsi `check` tidak terbukti.

## 1. Model original dan skor resmi Google

Yang dimaksud original di sini adalah checkpoint instruction-tuned Google **sebelum fine-tuning proyek**, bukan model tanpa training, checkpoint base, atau baseline decoder-only proyek yang sudah diadaptasi:

- [Gemma 4 E2B-it, revision 3e22461](https://huggingface.co/google/gemma-4-E2B-it/resolve/3e22461f65e89153144f8adb70e3b8c2cc9845a7/README.md).
- [Gemma 4 E4B-it, revision ee0ef60](https://huggingface.co/google/gemma-4-E4B-it/resolve/ee0ef6023621cff504d758262d4e04895a5af4a2/README.md).

| Kasus aktif lokal | Original E2B-it | Original E4B-it |
| --- | --- | --- |
| LongCodeQA, 32K, cohort lokal | Tidak ditemukan | Tidak ditemukan |
| HumanEvalFix Python, adapter unified diff | Tidak ditemukan | Tidak ditemukan |
| SWE-QA oracle | Tidak ditemukan | Tidak ditemukan |
| SWE-QA noisy oracle | Tidak ditemukan | Tidak ditemukan |
| GSM8K zero-shot lokal | Tidak ditemukan | Tidak ditemukan |
| GSM8K eight-shot lokal | Tidak ditemukan | Tidak ditemukan |
| CrossCodeLongEval chunk lokal | Tidak ditemukan | Tidak ditemukan |
| CrossCodeLongEval function lokal | Tidak ditemukan | Tidak ditemukan |

Pencarian dibatasi pada dua model card pinned di atas, [Gemma 4 Technical Report v1](https://arxiv.org/html/2607.02770v1), [model card Google AI](https://ai.google.dev/gemma/docs/core/model_card_4), dan [dokumentasi evaluation suites gbench](https://github.com/google-deepmind/gbench/blob/main/docs/eval-suites.md). “Tidak ditemukan” bukan bukti bahwa hasil tersebut tidak pernah ada di tempat lain.

Table 5 laporan Google memuat **LiveCodeBench v6: E2B 44.0%, E4B 52.0%**, dalam **thinking mode**. LiveCodeBench berbeda dari LongCodeBench/LongCodeQA, HumanEvalFix, dan CrossCodeLongEval. Nilai itu tidak bisa dipindahkan ke salah satu tugas kita; tabel laporan juga tidak mengikat skor ke hash revision model card proyek. Prompt lokal meminta `enable_thinking=False`, dengan fallback tokenizer yang tidak menerima argumen tersebut. [Sumber Google](https://arxiv.org/html/2607.02770v1).

## 2. Angka dari pembuat benchmark: model dan kondisi harus ikut disebut

Semua nilai pada tabel berikut adalah persen dari laporan pembuat benchmark, **bukan hasil Gemma 4 proyek**.

| Benchmark/varian resmi | Contoh angka laporan pembuat | Batas perbandingan |
| --- | --- | --- |
| LongCodeQA 32K | Qwen2.5-14B-Instruct 61.9; Gemini 2 Flash 66.4; Gemini 2.5 Pro 75.2; GPT-4o 65.5 | 113 kasus; model berbeda. [LongCodeBench v3, Table 3](https://arxiv.org/html/2505.07897v3#S4.T3) |
| HumanEvalFix **Tests**, Python | StarCoder 8.7; OctoCoder 30.4; GPT-4 47.0, pass@1 | Buggy function + unit tests; berbeda dari adapter lokal tanpa tests di prompt. [OctoPack v2, Table 2](https://arxiv.org/html/2308.07124v2#S4.T2) |
| HumanEvalFix **Docs**, Python | StarCoder Commit Format 58.8; StarCoder Instruct Format 41.7; OctoCoder 53.8; GPT-4 88.4, pass@1 | Docstring menjadi spesifikasi; bukan Table 2, bukan rata-rata lintas bahasa. Dua format StarCoder adalah prompt untuk checkpoint yang sama. [Table 14](https://arxiv.org/html/2308.07124v2#A11.T14) |
| SWE-QA oracle / noisy oracle | **Gemma 3 4B IT 72.00 / 70.63**; Llama 3.3 70B Instruct 74.41 / 74.31; Qwen3 4B Instruct 70.05 / 68.79 | Gemma **3**, bukan Gemma **4**. [Dataset card pembuat](https://huggingface.co/datasets/lailaelkoussy/swe-qa) |
| CrossCodeLongEval, chunk EM / chunk ES / function ES | StarCoderBase-1B tanpa retrieval 31.08 / 60.09 / 47.49; dengan always retrieval 37.22 / 63.73 / 50.50; StarCoderBase-3B always retrieval 42.26 / 67.74 / 53.39 | Retrieval, checkpoint, task formatting dan ES harus cocok. [Repoformer v2, Table 2](https://arxiv.org/html/2403.10059v2#S5.T2) |
| GSM8K zero/eight-shot | Tidak ada satu angka pembuat yang mendefinisikan dua protokol lokal ini | Paper OpenAI mempelajari fine-tuning/verifier; `gsm8k`, `gsm8k_cot`, dan prompt lokal memiliki setting berbeda. [OpenAI release](https://github.com/openai/grade-school-math), [gbench](https://github.com/google-deepmind/gbench/blob/main/docs/eval-suites.md) |

HumanEvalFixDocs pada paper memakai docstring yang benar sebagai sumber spesifikasi. Adapter lokal mempertahankan **docstring tugas asli**, lalu meminta patch unified diff. `repository_study.py:172–173` memakai `prompt` yang sama sebelum buggy dan canonical solution. Tidak ada bukti bahwa docstring lokal rusak: inspeksi statis menemukan docstring fungsi pada 163/164 snapshot dan tidak ada gold patch yang mengganti baris docstring tersebut. Varian lokal lebih dekat ke FixDocs daripada FixTests, tetapi tetap tidak identik dengan generasi full-function pada paper.

## 3. Protokol aktual untuk delapan kasus

Semua batas token berikut adalah ceiling. Allowance aktual adalah minimum antara batas suite, batas row, dan `8192 - panjang paired decoder prefix`, diterapkan sama pada baseline dan pasangan encoder–decoder. Budget nominal bukan bukti bahwa setiap jawaban memperoleh 8192 token baru. Source: `evaluation/protocol.py:27–52`, `evaluation/benchmark.py:604–635`.

| Kasus | Denominator lokal | Generasi / scoring lokal | Perbedaan penting dari pembuat |
| --- | ---:| --- | --- |
| LongCodeQA | 107 dari 113 | Greedy; ceiling 256; strict A–D sebagai headline | Cohort disaring oleh eligibility bersama; bukan seluruh 113 kasus 32K. Exact extractor/cap pembuat belum terverifikasi. |
| HumanEvalFix Python | 164 | Greedy unified diff untuk `solution.py`; ceiling 8192; patch + hidden tests ketika execution aktif | Prompt/output berbeda; default FixTests menyediakan tests, FixDocs memberi docstring; paper sampling menggunakan temperature .2/top-p .95 dan umumnya 20 sampel, GPT-4 satu. |
| SWE-QA oracle | 9072 | Greedy; ceiling 256; strict letter | Paper memakai normalized extraction yang menerima jawaban berpenjelasan; label revision dan exact parser belum terbukti sama. |
| SWE-QA noisy oracle | 9072 | Setting sama; identitas pertanyaan dipasangkan dengan oracle | Perbedaan extractor/label yang sama; distractor context adalah kondisi tersendiri. |
| GSM8K zero-shot | 1319 | Step-by-step + `####`; greedy; ceiling 8192; release regex/string match | Protokol lokal, bukan eksperimen supervised/verifier OpenAI atau gbench eight-shot CoT. |
| GSM8K eight-shot | 1319 | Delapan row training pertama, calculator annotation dibuang; extractor sama | Bukan delapan fixed exemplars `lm-eval gsm8k_cot`; prompt, stop strings dan extractor berbeda. |
| CrossCodeLongEval chunk | 100 sampel deterministik dari 4713 eligible / 5000 sumber | Chat FIM, sparse retrieval lokal; ceiling 2048; whitespace EM, SequenceMatcher, regex identifier-set F1 | Paper memakai native StarCoder FIM, budget input 1024/512/512 dan output 50, postprocessing chunk, EM dan Levenshtein ES. |
| CrossCodeLongEval function | 100 dari 4834 eligible / 5000 sumber | Adaptasi dan ceiling 2048 yang sama | Paper memakai output 256, ekstraksi function dengan tree-sitter, dan ES sebagai metrik utama. |

Sampling completion memakai seed 17 serta nama collection; semua model dijadwalkan memakai cohort yang sama. Ini mendukung perbandingan internal, tetapi 100 contoh tetap bukan hasil full-cohort paper. Cohort preparation juga memeriksa eligibility terhadap enam konfigurasi representatif, bukan bukti bahwa setiap setting inference masa depan akan setara. [Bukti konfigurasi lokal](audits/2026-11-17-benchmarks/active-protocols.json).

### Target peningkatan dan pemeriksaan hasil anomali

Target utama adalah kenaikan pada enam kasus yang dijalankan tanpa subsampling tambahan: LongCodeQA (seluruh 107 kasus eligible dari 113 sumber), HumanEvalFix Python (164), kedua kondisi SWE-QA (masing-masing 9072), dan kedua setting GSM8K (masing-masing 1319). Bandingkan baseline fine-tuned dengan checkpoint Google original yang memakai decoder sama, lalu Bifocal dengan baseline fine-tuned yang sepadan. Gunakan ID soal, instruksi tugas, batas keluaran, dan scorer yang konsisten. Catat delta per kasus benchmark serta jumlah kasus yang berubah dari salah ke benar dan sebaliknya untuk metrik biner. Untuk CrossCodeLongEval chunk dan function, audit output pada 100 ID tetap per jenis untuk melihat kesesuaian dengan konteks, format, dan kemungkinan kesalahan scorer. Simpan hasil sampel sebagai diagnostik.

Skor 0% memicu triase sebelum hasil digunakan untuk menilai training. Periksa status run, denominator, ketersediaan skor, dan jumlah prediksi valid. Baca keluaran mentah bersama stop reason, token count, hasil parsing, serta outcome patch dan tes untuk HumanEvalFix. Bila prediksi kosong atau berhenti sejak token awal, periksa template, BOS/EOS, dan budget decode. Bila prediksi masuk akal tetapi ditolak, periksa parser, format keluaran, dan pasangan ID dengan gold. Bila prediksi valid tetapi semuanya salah, bandingkan checkpoint original dan fine-tuned pada soal yang sama untuk menelusuri konfigurasi, pemuatan adapter, atau regresi training. Gunakan contoh dengan jawaban benar yang sudah diketahui untuk memeriksa jalur scoring secara terpisah. Skor yang tidak tersedia, invalid output, dan jawaban valid yang salah perlu dilaporkan sesuai status masing-masing.

### LongCodeQA: identitas dan cohort

Dataset aktif adalah `Steefano/LCB`, file `LQA/32K.json`, dari **LongCodeBench**, bukan LongCodeArena. Pembuat memakai zero-shot dengan repository Python tersusun alfabetis dan pertanyaan empat pilihan; bracket 32K berisi 113 kasus dari 27 repository. Lokal memakai 107 eligible tanpa silent cropping. Karena enam kasus dikeluarkan, denominator berbeda sekalipun file sumber sama. Tidak ditemukan dasar untuk menyebut pilihan generated-letter versus logprob sebagai mismatch yang telah terbukti; detail extractor dan cap creator masih terbuka. [Metode paper](https://arxiv.org/html/2505.07897v3#S3.SS2), [dataset](https://huggingface.co/datasets/Steefano/LCB), [creator eval](https://github.com/Zteefano/long-code-bench/blob/master/eval.py).

### HumanEvalFix: eksekusi benar, definisi tugas berbeda

Lokal mengubah buggy solution dan spesifikasi menjadi snapshot `solution.py`, meminta unified diff, menerapkan patch, kemudian menjalankan script test pada snapshot terpisah dengan timeout. Syntax score merupakan diagnostik tambahan. Jika execution disabled, functional score null; kegagalan infrastruktur tidak boleh dianggap kegagalan model. Inspeksi AST seluruh 164 test script menemukan definisi `check` **dan pemanggilan top-level `check`** pada semuanya; audit tidak mengeksekusi hidden tests.

Jadi skor functional lokal tetap bermakna sebagai solve rate satu generasi greedy pada tugas patch ini. Ia bukan reproduksi estimator sampled pass@1 pembuat. Contoh command creator memakai `max_length_generation=2048` sebagai panjang total, bukan otomatis 2048 token baru. [Harness pembuat](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/humanevalpack.py), [command OctoPack](https://github.com/bigcode-project/octopack#evaluation), [paper](https://arxiv.org/html/2308.07124v2).

### SWE-QA: parser dan versi label

9072 pertanyaan terdiri dari 4584 Declaration-and-Call dan 4488 Interacting-Entity, dari 12 repository Python. Oracle dan noisy oracle harus dibandingkan dengan identitas pertanyaan yang sama, seperti pada preparation lokal. Namun, headline strict-letter dapat menolak jawaban benar dengan penjelasan yang diterima normalized extraction paper. Parser resilient kita juga belum dibuktikan identik dengan parser pembuat.

Paper melaporkan koreksi konsensus pada 66 label. Audit belum memastikan apakah revision pinned proyek memuat koreksi itu; ini **gap verifikasi**, bukan temuan bahwa label lokal salah. Temperature, output cap, dan implementasi extractor creator yang persis juga belum terverifikasi. Pseudocode equality singkat dalam dataset card tidak cukup untuk merekonstruksi seluruh parser paper. [Paper setup](https://arxiv.org/html/2604.24814v1#S4), [dataset card](https://huggingface.co/datasets/lailaelkoussy/swe-qa), [repository](https://github.com/lailanelkoussy/swe-qa).

### GSM8K: release extraction benar, eight-shot berbeda

Release OpenAI memakai marker `#### ` diikuti numeric string bertanda opsional, menghapus koma, lalu membandingkan string. Implementasi strict lokal mengikuti ini. `The answer is 42.` dan `\\boxed{42}` tidak diterima strict tetapi dapat diterima diagnostik resilient; itu bukan bukti parser release salah. Contoh `#### 42.` menghasilkan strict `42.` dan resilient `42`, sehingga strict bukan general numerical equivalence. [Parser OpenAI](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py), [1319 test rows](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl).

Gbench mendokumentasikan `lm-eval gsm8k_cot`, 1319 soal, eight-shot CoT, dengan jumlah shot configurable. Ini dokumentasi protokol, bukan publikasi skor Gemma 4. YAML CoT memakai delapan contoh tetap, `Q:/A:`, stop strings, dan strict `The answer is <number>.`; YAML standard `gsm8k` adalah tugas berbeda dengan default five-shot. Lokal memakai delapan row pertama training dan marker `####`. Tidak boleh menyamakan seluruh setting itu hanya karena sama-sama bernama GSM8K. Contoh cap 65536 pada dokumentasi gbench juga bukan cap universal. [Gbench](https://github.com/google-deepmind/gbench/blob/main/docs/eval-suites.md), [CoT YAML](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml), [standard YAML](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml).

### CrossCodeLongEval: metrik dan postprocessing tidak setara

CrossCodeLongEval diperkenalkan **Repoformer**, terpisah dari CrossCodeEval walaupun menggunakan repository mentah terkait. Paper mendefinisikan chunk EM dan normalized Levenshtein edit similarity (ES), serta function ES. SequenceMatcher lokal bukan Levenshtein; identifier-set F1 lokal juga bukan headline metric paper. [Paper dan Appendix A](https://arxiv.org/html/2403.10059v2#A1).

Penelusuran snapshot creator `5b0571318e9918fd2af132c4b35077f9ea331133` menemukan dua jalur scorer. `eval_metric_cceval.py` membandingkan nonempty lines setelah comment removal dan strip per baris; `eval_metric.py` memakai tokenized-code EM untuk chunk. Keduanya berbeda dari menghapus seluruh whitespace. Keduanya menyediakan normalized Levenshtein `es_repoeval` dan fuzzy ratio terpisah. Default loop shell yang diperiksa menjalankan **line completion**, sehingga tidak cukup membuktikan scorer invocation yang menghasilkan setiap baris tabel CrossCodeLong. Definisi ES paper jelas berbeda dari metrik lokal, sementara reproduksi EM persis masih harus mengikat entry point dan versi. [Penelusuran kode pinned](audits/2026-11-17-benchmarks/repoformer-scorer-verification.md).

Probe tokenizer offline juga menutup dugaan bahwa semua reference completion terlalu panjang untuk cap lokal 2048:

| Cohort | Reference terpanjang, seluruh eligible | Reference terpanjang, sampel 100 | Reference >2048 |
| --- | ---:| ---:| ---:|
| Chunk | 410 token | 104 token | 0 |
| Function | 935 token | 516 token | 0 |

Penghitungan memakai tokenizer E2B-it lokal tanpa special tokens dan sampling yang sama dengan kode. Ini membuktikan panjang reference yang diperiksa muat pada cap 2048; bukan jaminan generasi model tidak habis budget, dan bukan penyetaraan cap dengan paper. [Bukti hash dan penghitungan](audits/2026-11-17-benchmarks/reference-cap-probe.json).

## 4. Bug yang direproduksi dan batas metrik

1. **Parser MCQ resilient memilih kandidat yang salah.** Input `A is correct? No, the final answer is B.` menghasilkan `A`, karena prioritas regex mengambil pola awal sebelum resolusi akhir. Parser strict menghasilkan null; bug ini mengenai diagnostik resilient, bukan otomatis headline strict. Perbaikan perlu scorer identity baru dan dapat diterapkan pada saved predictions yang sesuai tanpa generasi ulang.
2. **Whitespace EM menyamakan kode yang semantiknya berbeda.** `return "ab"` dan `return "a b"` memperoleh EM true dan similarity 1.0. Memindahkan `cleanup()` ke dalam blok `if` juga hilang setelah normalisasi. Metrik ini harus disebut adapted whitespace-normalized EM; jangan menyebutnya execution correctness atau creator EM.
3. **Tidak semua invalid adalah jawaban salah secara isi.** Jawaban MCQ berpenjelasan dan jawaban GSM tanpa marker dapat gagal strict sesuai kontrak. Laporkan strict, resilient, invalid rate dan denominator secara terpisah; jangan diam-diam mengganti headline dengan diagnostik.

[Probe fungsi evaluator](audits/2026-11-17-benchmarks/offline-scorer-probes.json) hanya memakai fungsi lokal; tidak memuat bobot, menjalankan model, atau menilai kemampuan checkpoint.

## 5. Urutan perbaikan dan dampak yang diharapkan

| Prioritas | Tindakan berikut | Peningkatan yang masuk akal | Batas klaim |
| --- | --- | --- | --- |
| 1 | Labeli hasil yang ada sebagai protokol adaptasi; pisahkan original Google, baseline fine-tuned, dan paired | Perbandingan dan interpretasi lebih jujur | Tidak mengubah kemampuan model |
| 2 | Versikan perbaikan parser resilient dan rescore saved predictions yang kompatibel | Diagnostik format jawaban lebih benar | Headline strict tidak otomatis naik |
| 3 | Tambahkan mode creator-compatible terpisah: HumanEvalFixDocs/Tests, SWE-QA extractor/release, pinned GSM8K CoT, Repoformer budgets/postprocessing/ES | Mengurangi confound evaluator dan memungkinkan pemeriksaan lintas laporan | Harus mengunci detail sumber yang masih terbuka; skor lama tidak boleh dilabel ulang |
| 4 | Jalankan original checkpoint terpisah pada protokol lokal yang sama, setelah resource tersedia | Memisahkan pengaruh fine-tuning dari arsitektur dan evaluator | Tidak dapat digantikan angka benchmark lain; Molab lama tetap berjalan |
| 5 | Perluas completion ke full eligible cohort dan laporkan uncertainty | Estimasi lebih stabil daripada sampel 100 | Masih perlu cohort resmi untuk klaim reproduksi paper |

Perbaikan evaluator terutama meningkatkan **validitas pengukuran**. Dampaknya pada skor bisa naik atau turun; audit ini tidak membuktikan BifocalGemma lebih kuat atau lebih lemah dari original Google. Potensi manfaat arsitektur dan ablation tetap mengikuti [audit arsitektur sebelumnya](ASTRA_BIFOCAL_AUDIT_2026-11-16.md).

## 6. Bukti lokal dan perubahan yang dilakukan

Source anchors pada revision `80522ad`:

- `data/context_benchmarks.py:82–399`: sumber, retrieval, pemisahan gold dan eligibility.
- `evaluation/context.py:112–181`: routing dan chat template; `:242–257` prioritas regex; `:406–435` completion metrics; `:473–560` strict/resilient dan functional score.
- `evaluation/benchmark.py:121–181`: sampling; `:487–559` input/gold/checkpoint/protocol signatures; `:604–714` executed budget dan scoring.
- `evaluation/protocol.py:27–52`: allowance bersama; `:61–114` fingerprint parser/scorer; `:133–247` identitas run dan pemisahan headline/diagnostik.
- `repository_study.py:162–235`, `evaluation/native_python.py:31–199`: adapter HumanEvalFix, patch dan test execution.
- `evaluation/gsm8k.py:35–191`: extractor, contoh, source identity; `:309–345` routing.

Direktori [bukti audit](audits/2026-11-17-benchmarks/) menyimpan konfigurasi ringkas, probe scorer/cap, dan penelusuran scorer creator. Tidak ada private reference code atau hidden tests yang disalin ke dokumentasi publik.

Perubahan audit hanya dokumentasi, identitas benchmark dan batas perbandingan di kedua format paper, serta bibliography sumber aktif. Semua placeholder hasil paper dipertahankan. Evaluator, checkpoint, runtime ZIP, `runtime/latest.zip`, dan proses Molab tidak diubah.

Validasi penutupan, 18 November: probe scorer dan reference-cap tersimpan sebagai JSON valid; tautan lokal laporan serta seluruh citation key LaTeX diperiksa; tujuh placeholder pada masing-masing draft tidak berubah. Tes `test_f8_scientific_paper_structure` lulus (1 passed). Kompilasi LaTeX–BibTeX–LaTeX selesai tanpa undefined citation/reference. Masih ada warning overfull equation pada baris 23, rumus bridge yang tidak diubah dalam audit ini. Tidak dilakukan pengujian ulang seluruh training/evaluator karena tidak ada source runtime yang berubah.
