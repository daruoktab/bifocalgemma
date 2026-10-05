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

Matriks utama memakai E2B–E2B IT, E4B–E4B IT, E2B–E4B IT dan E4B–E2B IT,
dengan encoder base / decoder IT. E2B-IT dan E4B-IT menjadi baseline decoder-only.
[Desain eksperimen](EXPERIMENT_DESIGN.md) menjadi acuan perbandingan enam model.
Pilot integrasi memakai E2B–E2B; eksekusi empat pasangan dilakukan bertahap setelah profiling. Perbandingan dengan E4B tidak boleh disebut iso-parameter/iso-memory
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
Matriks empat model tetap menjadi target utama. Multimodal ditunda sampai eksperimen
dasar dapat direproduksi; T5Gemma/T5Gemma 2 dibahas sebagai prior art.

## Dataset training dan benchmark

[Riset dataset dan protokol](DATASETS_AND_BENCHMARKS.md) memuat sumber primer,
peran tiap dataset, pemisahan train/evaluation, kebutuhan eksekusi, dan urutan eksperimen.
SWE-smith task/trajectory menjadi kandidat utama repair; CommitPackFT Python
merupakan warm-up editing opsional, sedangkan SWE-Gym menyediakan alternatif real issue.

SWE-bench Lite (300 task test) dan Verified (500 task test) mengukur resolusi issue
melalui tes. CrossCodeEval/RepoBench adalah diagnostik completion/retrieval lintas file,
bukan keberhasilan patch. CanItEdit merupakan benchmark editing 105 program Python;
EditPackFT adalah sumber training terpisah. HumanEvalFix memberi evaluasi repair fungsi.
Gold patch dan test artifacts hanya digunakan untuk supervisi training atau evaluator,
bukan masukan model saat inference evaluation. Sumber training harus diaudit terhadap
instance/commit/patch overlap; prioritas pada konfigurasi dan subset yang dapat direproduksi.

## Referensi

- [Spesifikasi backbone dan parameter](GEMMA4_EDGE_ARCHITECTURE.md)
- [T5Gemma: adaptation dan encoder-decoder asimetris](https://developers.googleblog.com/en/t5gemma/)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [SWE-smith: Automated SWE Agent Data Synthesis](https://github.com/SWE-bench/SWE-smith)
- [SWE-bench: Resolving Real-World GitHub Issues](https://www.swebench.com/)
- [OctoPack & CommitPackFT (BigCode)](https://github.com/bigcode-project/octopack)
- [CrossCodeEval: Cross-File Code Completion Benchmark](https://github.com/amazon-science/cceval)
- [RepoBench: Repository-Level Code Completion](https://github.com/Leolty/repobench)

## Implikasi audit JSON base dan IT

[Aset base dan IT](GEMMA4_JSON_AUDIT.md) memiliki text config dan vocabulary
yang kompatibel, tetapi post-processing BOS dan generation EOS berbeda. Kandidat
awal memakai encoder base E2B dan decoder E2B IT dengan revision masing-masing
dipin. Full physical text parameters sekitar 4,63B/7,46B per backbone; pasangan
E2B/E2B BF16 membutuhkan sekitar 18,514 GB bobot sebelum bridge dan state.
MQA/GQA serta KV sharing baseline harus dibandingkan dengan full-MHA cross-KV
bridge; keuntungan total memori tidak mengikuti ilustrasi decoder self-KV saja.


## Target word-embedding sharing

Pasangan E2B–E2B IT/E4B–E4B IT memakai word embeddings tied pada tahap konversi;
pasangan E2B–E4B IT/E4B–E2B IT mempertahankan embedding terpisah. PLE tidak di-tie.
Checkpoint sebelum konversi menyimpan keadaan encoder base / decoder IT yang
diadaptasi dengan attention terpisah. Tabel bersama simetris memakai embedding
decoder teacher sebagai default anchor, dengan perubahan provenance encoder
dicatat. Konversi merged attention memerlukan adaptasi/distillation; pasangan
asimetris tetap perlu penyelarasan hidden states untuk shared K/V projections.

Hitungan BF16 18,514 GB E2B–E2B di atas adalah kondisi untied sebelum bridge.
Word tying saja menurunkannya ke sekitar 17,709 GB; itu belum peak VRAM maupun
ukuran model akhir setelah perubahan attention. Protokol mengikuti
[desain eksperimen](EXPERIMENT_DESIGN.md); runtime tying/konversi belum terintegrasi.
