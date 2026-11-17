# Pembanding model di bawah 7B — 18 November 2026

Riset lanjutan untuk [audit protokol](OFFICIAL_BENCHMARK_PROTOCOL_AUDIT_2026-11-17.md). Tujuannya mencari hasil publik model kecil yang lebih relevan daripada GPT-4 atau model 14B–70B, pada keluarga benchmark yang benar-benar dipakai proyek. Tabel mengutamakan generasi 2024–2025 dan laporan sampai 2026; bukan klaim bahwa setiap model adalah rilis paling baru. Semua angka adalah hasil pihak sumber, **bukan hasil evaluasi BifocalGemma**.

Kriteria ukuran memakai **total parameter di bawah 7B**, bukan active parameter MoE. Model 7B dan Qwen3-30B-A3B tidak masuk. Status checkpoint base, instruction-tuned, atau tambahan fine-tuning dicantumkan. Nilai antartabel tidak membentuk satu leaderboard karena prompt, decoder, parser dan cohort berbeda. Tidak ada model yang dijalankan atau diunduh untuk riset ini.

## Ringkasan cakupan

| Kasus pilihan kita | Pembanding kecil yang terverifikasi | Kecocokan dengan protokol lokal |
| --- | --- | --- |
| LongCodeQA 32K | Belum ditemukan angka sub-7B yang cukup terverifikasi | Jangan mengganti dengan LongBench, LongCodeArena atau LiveCodeBench |
| HumanEvalFix Python | Qwen2.5-Coder 3B Instruct, NextCoder 3B, Qwen3 1.7B Base | Nama benchmark sama; edit format, sampling dan varian input belum identik |
| SWE-QA oracle / noisy oracle | Gemma 3 4B IT, Qwen3 4B Instruct | Kondisi dan count sangat relevan; extractor dan label revision perlu disamakan |
| GSM8K zero-shot | Gemma 3 1B/4B IT, Nemotron-H 4B Instruct | Sama-sama zero-shot, tetapi sampling, prompt dan format jawaban berbeda |
| GSM8K eight-shot | Phi-4 mini, Qwen2.5 3B Instruct, Llama 3.2 3B Instruct | Jumlah shot sama; bukan bukti contoh, chat template atau extractor sama |
| CrossCodeLongEval chunk / function | Qwen2.5-Coder 0.5B/1.5B/3B Base | Varian task benar; native FIM, retrieval, budget, postprocessing dan ES berbeda dari lokal |

## SWE-QA: pembanding paling dekat

Sumber adalah **pembuat benchmark**, bukan angka yang diklaim pembuat masing-masing model. Kedua kondisi memuat 9072 pertanyaan; paper menggunakan zero-shot dan normalized extraction. Exact checkpoint revision, generation cap dan temperature belum ditentukan dari sumber yang diperiksa.

| Model sesuai nama sumber | Ukuran | Oracle accuracy | Noisy-oracle accuracy |
| --- | ---:| ---:| ---:|
| Gemma-3-4B-it | 4B | **72.00%** | **70.63%** |
| Qwen3-4B-Instruct | 4B | **70.05%** | **68.79%** |

[Dataset card pembuat, baseline results](https://huggingface.co/datasets/lailaelkoussy/swe-qa), [paper SWE-QA v1](https://arxiv.org/html/2604.24814v1#S5.T1). Nama `Qwen3-4B-Instruct` tidak cukup untuk menetapkan revision atau suffix `2507`; jangan menambah identitas itu tanpa bukti.

Paper juga memuat **accuracy per kategori** oracle, yang tidak boleh disajikan sebagai overall accuracy:

| Model | Declaration-and-Call | Interacting-Entity |
| --- | ---:| ---:|
| Llama-3.2-3B-Instruct | 66.79% | 65.50% |
| Qwen3-4B-Thinking | 69.85% | 54.81% |
| Phi-4-Mini-Instruct | 58.09% | 47.90% |
| Phi-4-Mini-Reasoning | 59.77% | 50.35% |
| Qwen3-1.7B | 48.32% | 43.18% |

[Creator Table 1](https://arxiv.org/html/2604.24814v1#S5.T1). Overall/noisy counterparts untuk baris ini belum terverifikasi di sumber yang diperiksa; tidak dihitung sendiri dari rata-rata dua kategori. Untuk membandingkan hasil proyek, gunakan metrik dan pertanyaan yang sama, serta verifikasi apakah revision dataset sudah memasukkan koreksi 66 label yang dilaporkan paper.

## GSM8K: pisahkan zero-shot dan eight-shot

| Model | Ukuran | Hasil | Setting menurut sumber | Siapa yang melaporkan |
| --- | ---:| ---:| --- | --- |
| Gemma-3-4B-IT | 4B | **89.2%** | 0-shot CoT, sampling | Google |
| Gemma-3-1B-IT | 1B | **62.8%** | 0-shot CoT, sampling | Google |
| Nemotron-H-4B-Instruct-128K | 4B | **88.93%** | 0-shot; reasoning lalu jawaban `\\boxed{}` | NVIDIA |
| Phi-4-mini-instruct | 3.8B | **88.6%** | 8-shot CoT | Microsoft |
| Qwen2.5-3B-Instruct | 3B | **80.6%** | 8-shot CoT | Microsoft, evaluasi model pembanding |
| Llama-3.2-3B-Instruct | 3B | **75.6%** | 8-shot CoT | Microsoft, evaluasi model pembanding |
| Phi-3.5-mini-instruct | 3.8B | **76.9%** | 8-shot CoT | Microsoft |

Sumber: [Google Gemma 3 Table 18, scores](https://arxiv.org/html/2503.19786v1#Sx1.T18) dan [Table 21, IT methods](https://arxiv.org/html/2503.19786v1#Sx1.T21); [NVIDIA instruct card](https://huggingface.co/nvidia/Nemotron-H-4B-Instruct-128K); [Microsoft Phi-4 mini card, Model Quality](https://huggingface.co/microsoft/Phi-4-mini-instruct). Microsoft memakai platform evaluasi internal; angka competitor di tabel tersebut bukan publikasi dari Meta/Qwen sendiri.

Khusus Nemotron, jawaban boxed tidak akan diterima parser strict lokal yang meminta `####`, jika output dipakai tanpa menyesuaikan protokol. Angka Google memakai sampling, sedangkan run lokal greedy. Exact temperature, batas output dan exemplar/parser parity belum terverifikasi untuk seluruh baris ini. Karena itu, skor tersebut adalah referensi kemampuan pada GSM8K, bukan ambang reproduksi langsung untuk run kita.

Perbedaan checkpoint/shot dapat menghasilkan angka yang sangat berbeda:

- Google melaporkan **Gemma-3-4B pretrained 38.4%**, **8-shot CoT**, sedangkan **89.2%** di atas adalah **instruction-tuned, zero-shot CoT**. Jangan menukar status PT/IT. [PT scores dan metode](https://arxiv.org/html/2503.19786v1#Sx1.T10), [PT method table](https://arxiv.org/html/2503.19786v1#Sx1.T19).
- NVIDIA melaporkan **Nemotron-H-4B-Base-8K 69.6%**, **8-shot**, terpisah dari instruct-128K 88.93%. [Base card](https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K).
- Qwen melaporkan **Qwen3-4B-Base 87.79%**, tetapi **4-shot CoT**, sehingga tidak dimasukkan ke kelompok zero/eight-shot kita. [Qwen3 report, Table 7](https://arxiv.org/html/2505.09388v1#S3.T7).

Tidak ditemukan angka 0/8-shot original Qwen3.5-4B yang cukup terverifikasi dalam sumber yang diperiksa. GSM8K-Platinum atau GSM8K Hard bukan pengganti test GSM8K 1319 kasus yang kita pilih.

## CrossCodeLongEval: Qwen2.5-Coder kecil

Ini hasil **checkpoint base**, dari technical report pembuat Qwen. Bukan Qwen2.5-Coder-Instruct, bukan HumanEval-FIM, bukan CrossCodeEval.

| Model | Chunk EM | Chunk ES | Function EM | Function ES |
| --- | ---:| ---:| ---:| ---:|
| Qwen2.5-Coder-0.5B | 29.8% | 64.2% | 9.5% | 38.0% |
| Qwen2.5-Coder-1.5B | **44.2%** | **73.9%** | **12.4%** | **44.4%** |
| Qwen2.5-Coder-3B | **46.6%** | **76.1%** | **13.5%** | **46.4%** |
| StarCoder2-3B | 18.5% | 62.0% | 10.2% | 39.2% |

[Qwen2.5-Coder report v3, Table 9](https://arxiv.org/html/2409.12186v3#S6.T9). Baris StarCoder2 adalah pengukuran yang dilaporkan Qwen, bukan laporan original BigCode.

Protocol report: maximum sequence 8192 token, cross-file context hingga 2048, retrieval BM25 resmi, output 50 token untuk chunk dan 256 untuk function. Metrik ES berbeda dari SequenceMatcher lokal. Konfigurasi ini juga berbeda dari default pembuat Repoformer yang dibahas audit awal, jadi “official” tetap harus menyebut **laporan siapa** dan setting yang dipakai. Tidak boleh memasukkan skor ini langsung ke kolom protokol chat lokal 32K/2048-output/sampel-100.

Sebagai riset 2026 yang lebih baru, paper SRI melaporkan **SRI-Coder-3B aggregate CrossCodeLongEval EM 30.9 / ES 61.9**, versus **Qwen2.5-Coder-3B FIM aggregate EM 30.1 / ES 61.3**. Itu model **tambahan fine-tuning** dan output search-and-replace, serta angka **aggregate**, bukan chunk atau function individual. Selisih aggregate baseline 30.1 di sini versus 30.0 pada laporan Qwen harus dipertahankan menurut sumber; jangan disatukan sebagai satu run. [SRI paper v1, Table 8](https://arxiv.org/html/2601.13384v1#A2.T8).

## HumanEvalFix Python: model instruction dan base harus terpisah

| Model / status | Ukuran | Skor yang dilaporkan | Sumber |
| --- | ---:| ---:| --- |
| Qwen2.5-Coder-3B-Instruct, sebelum tambahan fine-tuning NextCoder | 3B | **73.2%** | Paper NextCoder, nama tabel `QwenCoder-2.5-3B` |
| NextCoder-3B, tambahan adaptation SeleKT | 3B | **75.6%** | Paper dan card Microsoft |
| Qwen2.5-Coder-3B + SFT pada eksperimen tersebut | 3B | 76.2% | Tambahan fine-tuning, bukan original Qwen |
| Qwen3-1.7B-Base, tanpa tambahan fine-tuning AgentPack | 1.7B | **0.34 pass@1, setara 34%** | AgentPack, HumanEvalFix-Py |

[NextCoder/Robust Learning v2, Table 5](https://arxiv.org/html/2503.03656v2#S5.T5), [Microsoft card](https://huggingface.co/microsoft/NextCoder-14B), [AgentPack v3, Table 4](https://arxiv.org/html/2509.21891v3#S5.T4). Card NextCoder-14B memuat tabel lintas ukuran; angka 3B bukan skor model 14B. Bagian experimental setup paper menyebut inisialisasi dari **instruct versions**, sehingga label “base model” pada pembahasan NextCoder berarti checkpoint sebelum adaptation mereka, bukan pretrained base Qwen.

NextCoder menggunakan 164 contoh HumanEvalFix, temperature .2/top-p .95 dan metrik `pass@1,1`. AgentPack melaporkan pass@1 dari 20 sampled completions dengan temperature .2/top-p .95 dan prompt benchmark tersendiri. Varian Docs versus Tests serta format yang identik dengan adapter unified-diff lokal belum terverifikasi. Oleh sebab itu, **73.2%, 75.6%, dan 34% bukan ranking yang dikontrol lintas dua paper**; pretrained Base, IT, adaptation, dan prompt memengaruhi hasil.

## LongCodeQA: gap bukti tetap terbuka

Belum ditemukan hasil sub-7B pada **LongCodeQA 32K** yang cukup jelas untuk dipakai. LongCodeBench original mengevaluasi model lebih besar; hasil Llama 3.2 3B pada LongBench-v2 dalam paper ATFlash bukan hasil LongCodeQA. Setup LongCodeQA ATFlash sebenarnya memakai **Qwen2.5-7B-Instruct-1M**, sehingga keluar dari kriteria ketat di bawah 7B. [ATFlash v1, setup](https://arxiv.org/html/2608.02947v1#S6.T2). Gap ini tidak diisi dengan angka tugas lain.

## Implikasi untuk eksperimen kita

Pembanding yang paling praktis menurut cakupan sumber adalah Gemma 3 4B IT/Qwen3 4B untuk SWE-QA, Qwen2.5-Coder 3B untuk completion dan repair, serta Phi-4 mini/Gemma 3 4B IT untuk matematika. Ini usulan berdasarkan kedekatan ukuran dan tersedia-tidaknya laporan, bukan rekomendasi pemenang semua tugas.

Prioritas validasi tetap: (1) simpan hasil lama dengan label protokol lokal; (2) cocokkan scorer, input condition dan template sebelum mengejar angka publik; (3) evaluasi original checkpoint terpilih dengan protokol lokal yang sama pada run terpisah setelah resource tersedia. Riset ini tidak mengubah evaluator, menjalankan benchmark, atau mengganggu Molab. Placeholder hasil paper tetap dipertahankan.
