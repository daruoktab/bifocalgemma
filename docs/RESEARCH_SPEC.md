# BifocalGemma: Spesifikasi Teknis Riset

**Judul:** *BifocalGemma: Evaluating Bidirectional Repository Encoding for Local Code Repair*

Dokumen ini memisahkan utilitas yang sudah tersedia dari arsitektur dan eksperimen
yang direncanakan. Tidak ada hasil training atau benchmark BifocalGemma saat ini.

## Modul yang tersedia

- [Bridge](../src/bifocalgemma/models/bridge.py): Q/K/V projection, QK normalization,
  attention, dan gated residual dengan lebar encoder/decoder yang dapat berbeda.
- [AST projector](../src/bifocalgemma/models/code_graph_projector.py): embedding dari
  ID tipe node dan kedalaman yang disuplai. Belum mengekstrak AST atau memproses edge graf.
- [Routing](../src/bifocalgemma/models/routing.py): tensor split, mask validitas token,
  dan ilustrasi decoder self-attention KV.
- [Pelaporan](../src/bifocalgemma/evaluation/eval_code.py): tabel skor masukan dan
  referensi. Modul ini belum menjalankan benchmark.

`create_bidirectional_repo_mask` mengembalikan mask boolean `(1, seq_len)` semua-valid.
Nama helper menunjukkan tujuan pemakaian encoder; mask itu belum mengubah causal
mask atau sliding-window behavior pada Gemma. Integrasi backbone harus mengatur
attention yang benar dan menguji dependensi token di kedua arah.

## Bridge dan batas interpretasi gate

$$z = X + \tanh(g)\,\mathrm{CrossAttention}(X, H_{enc})$$

Default `g = atanh(0.05)` memberi koefisien residual tambahan `0.05`.
Jalur residual `X` tetap berkoefisien satu. Tidak ada jaminan persentase identitas
perilaku decoder atau persentase gradien; besarnya bergantung pada aktivasi,
bobot proyeksi, dan loss. Bridge belum diverifikasi sebagai pengganti layer Gemma.

## Ilustrasi decoder self-attention KV

Helper `RepoContextRouter.calculate_decoder_self_kv_savings` mengasumsikan semua
layer memakai full attention, jumlah KV head seragam, serta K dan V terpisah:

$$C = 2N_{layers}N_{kv\_heads}D_{head}b$$
$$M_{baseline,self} = (L_{prompt}+L_{gen})C,\qquad M_{split,self}=L_{gen}C$$

Konversi ke MiB membagi byte dengan `1024**2`. Key API memakai suffix `_mb`:

| Key hasil | Arti |
| --- | --- |
| `decoder_only_self_kv_mb` | Self-KV prompt dan generasi pada ilustrasi baseline |
| `bifocalgemma_decoder_self_kv_mb` | Self-KV generasi pada ilustrasi arsitektur split |
| `self_kv_saved_mb` | Selisih dua ukuran self-KV tersebut |
| `self_kv_reduction_percentage` | Persentase selisih terhadap self-KV baseline |

Dengan default `Lprompt=16384`, `Lgen=512`, 32 layer, 8 KV head, dimensi 256,
dan 2 byte/elemen, ukuran ilustratifnya 4224 MiB versus 128 MiB. Angka ini
**bukan konfigurasi Gemma** dan **bukan penghematan total VRAM**. Tanpa prompt,
selisihnya nol; tanpa generasi, self-KV split nol; dua panjang nol menghasilkan
semua nilai nol. Panjang negatif dan ukuran layer/head/elemen nonpositif ditolak.

Total biaya sistem mencakup bobot yang resident, encoder states, cross-attention
K/V yang disimpan, decoder self-KV, aktivasi/workspace, dan overhead allocator.
Jika proyeksi K/V cross-attention dihitung ulang, biaya komputasi itu juga dilaporkan.
Gemma memakai hybrid attention dan KV sharing; estimasi seragam ini tidak memodelkannya.

## Protokol eksperimen awal

Kandidat pertama **BifocalGemma-E2B-E2B**, dengan kelayakan yang ditentukan melalui
profiling. Perbandingan dengan E4B tidak boleh disebut iso-parameter/iso-memory
hanya karena jumlah parameter efektif berdekatan; bobot fisik, bridge, precision,
offloading, parameter trainable, dan budget training harus dilaporkan.

Baseline decoder-only memakai data, konteks, sampling, serta token/tool budget
sebanding dan prefix caching bila didukung. Encoder bidirectional dapat mengubah
representasi token konteks; itu bukan bukti bahwa decoder-only tidak mampu membaca
relasi lintas file atau harus menghitung seluruh prefix pada setiap giliran.

Ukur task repair success, ketepatan lokalisasi, latensi encoding/TTFT/end-to-end,
throughput, dan peak VRAM total pada cold/warm cache. Benchmark tidak boleh
menganggap skor dari protokol publik yang berbeda sebagai kemenangan head-to-head.

Snapshot penuh di-encode ulang setelah edit. Partial re-encoding satu hop ditunda
karena full bidirectional attention dapat menyebarkan perubahan ke seluruh snapshot.
Tunda matriks empat model dan multimodal sampai eksperimen dasar dapat direproduksi.

## Dataset Training untuk Code Repair & Agent

Untuk melatih kemampuan repository encoding dan code repair lintas file, kandidat data training mencakup:

1. **SWE-smith** (Princeton / Cognition):
   - Pipeline otomatisasi sintesis bug pada repositori riil (>50.000 task instances pada 128+ repositori open-source).
   - Menghasilkan bug melalui mutasi kode prosedural dan pembalikan pull request (PR mirroring) yang diuji merusak unit test yang sebelumnya lulus.
   - Dilengkapi lingkungan Docker terisolasi dan issue description yang disintesis, ideal untuk Supervised Fine-Tuning (SFT) dan Reinforcement Learning (RL) agent software engineering.
2. **CommitPackFT** (BigCode / OctoPack):
   - Subset terkurasi 2 GB dari 4 TB koleksi commit GitHub pada 350+ bahasa pemrograman.
   - Difilter khusus untuk pesan commit berbentuk instruksi imperatif yang dipasangkan dengan patch diff.
   - Ablasi riset OctoPack menunjukkan bahwa fine-tuning pada CommitPackFT merupakan faktor penentu keberhasilan model dalam tugas perbaikan bug nyata (bug repair).
3. **CanItEdit / Refactoring Instruction Datasets**:
   - Dataset instruksi multi-turn untuk refactoring kode repositori dan perbaikan kode terarah instruksi pengguna.

## Benchmark Evaluasi untuk Repository-Level Code Repair

Evaluasi difokuskan pada pemahaman dependensi lintas file dan kebenaran patch:

1. **SWE-bench (Verified & Lite)**:
   - Standar utama evaluasi perbaikan bug level repositori end-to-end berbasis issue GitHub riil.
   - **SWE-bench Verified** (500 instance): Divalidasi oleh engineer manusia untuk menjamin deskripsi tidak ambigu dan unit test dapat diselesaikan.
   - **SWE-bench Lite** (300 instance): Subset terpilih dari 12 repositori Python populer untuk evaluasi yang lebih cepat dan efisien biaya.
   - Metrik utama: Tingkat resolusi patch (seluruh test `FAIL_TO_PASS` menjadi lulus dan tidak ada regresi pada test `PASS_TO_PASS`).
2. **CrossCodeEval** (Amazon Science):
   - Benchmark multi-bahasa (Python, Java, TypeScript, C#) yang dirancang khusus untuk mengukur kemampuan model dalam memanfaatkan dependensi lintas file (cross-file context).
   - Menggunakan analisis statis untuk memastikan bahwa tugas secara ketat membutuhkan konteks di luar file yang sedang diedit.
3. **RepoBench** (RepoBench-R, RepoBench-C, RepoBench-P):
   - Mengevaluasi pipeline repositori multi-file: retrieval snippet lintas file (RepoBench-R), code completion dengan in-file dan cross-file context (RepoBench-C), dan pipeline end-to-end (RepoBench-P).
4. **HumanEval-Fix & HumanEvalPack** (BigCode):
   - Ekstensi HumanEval untuk tugas perbaikan bug tingkat fungsi terisolasi (single-file / unit-level) pada 6 bahasa pemrograman dengan eksekusi unit test.
5. **Defects4J**:
   - Benchmark klasik berbasis commit bug riil pada proyek open-source Java dengan test suite regresi lengkap.

## Referensi

- [Spesifikasi backbone dan parameter](GEMMA4_EDGE_ARCHITECTURE.md)
- [T5Gemma: adaptation dan encoder-decoder asimetris](https://developers.googleblog.com/en/t5gemma/)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [SWE-smith: Automated SWE Agent Data Synthesis](https://github.com/SWE-bench/SWE-smith)
- [SWE-bench: Resolving Real-World GitHub Issues](https://www.swebench.com/)
- [OctoPack & CommitPackFT (BigCode)](https://github.com/bigcode-project/octopack)
- [CrossCodeEval: Cross-File Code Completion Benchmark](https://github.com/amazon-science/cceval)
- [RepoBench: Repository-Level Code Completion](https://github.com/Leolty/repobench)
