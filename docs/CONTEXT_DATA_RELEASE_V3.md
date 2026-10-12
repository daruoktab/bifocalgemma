# Context data v3: training dan evaluasi 32K/8K

Tanggal kurasi akhir: **13 Oktober 2026**. Rilis masuk repo dataset private
[daruokta/bifocalgemma-data](https://huggingface.co/datasets/daruokta/bifocalgemma-data).
Revision terverifikasi: [`7972f95d7c87`](https://huggingface.co/datasets/daruokta/bifocalgemma-data/commit/7972f95d7c87f163c8b5af6a4acd308c0c1d0e23).
Revision aktif dan fingerprint dicatat di [lock](../configs/dataset_release.lock.json);
kode dan bobot tetap berada di repositori yang berbeda.

## Koleksi training

`mixed_context_enc32k_dec8k_v3` berisi **49.342 contoh: 44.408 train /4.934 dev**.
Satu epoch, batch1 dan accumulation4 menghasilkan **11.102 optimizer updates**;
train split mempunyai **16.681.401 token target**, termasuk closing turn template.
Angka ini bukan jumlah token context maupun total token komputasi encoder/decoder.
Train split memuat **47.758.770 encoder tokens** dan **20.405.380 shifted paired
decoder tokens**; baseline memakai **65.162.741 shifted decoder tokens**. Penjumlahan
stream bukan estimasi FLOPs atau durasi GPU.

| Sumber | Contoh di seluruh pool | Peran |
| --- | ---: | --- |
| nuprl/EditPackFT | 17.233 | Instruksi editing dengan kode sebelum perubahan |
| bigcode/commitpackft, Python | 15.237 | Editing dari riwayat commit |
| microsoft/NextCoderDataset, Python | 12.486 | Synthetic instruction editing; lineage/lisensi seed belum terverifikasi |
| nebius/SWE-rebench-V2 | 4.386 | Repair, localization dan completion dari snapshot pre-fix |

| Objective | Contoh | Target |
| --- | ---: | --- |
| Editing foundation (`task_type=repair`) | 44.956 | Unified diff; tidak seluruhnya bug fix |
| Repository repair | 616 | Unified diff; seluruh preimage hunk terlihat di context terpilih |
| Localization | 1.285 | Daftar path JSON yang terlihat di context |
| Dependency completion | 2.485 | Body fungsi Python yang dimask, mempertahankan indentasi |

Keempat pasangan dan dua baseline menerima ID serta isi contoh yang sama.
Encoder menerima context sebelum perubahan; baseline menempatkan context yang sama
di prefix decoder. Target dan gold tests tidak dimasukkan ke input model.
Nama baseline pada notebook merujuk **IT initialization yang dituning dengan SFT
yang sama**, bukan hasil checkpoint IT untouched.

## Dari kandidat menuju pool

SWE-rebench-V2 dikunci pada revision
`10483de0f50fe5da545942705a76c6150171af7f`. Dari 32.079 row sumber, adapter menghasilkan
6.733 kandidat setelah pemeriksaan schema, perubahan Python non-test dan karantina
repository benchmark. Tidak ada kuota jumlah task atau jumlah contoh per repository.

5.084 kandidat lolos audit archive, patch, syntax, lisensi dan token awal. Itu bukan
jumlah task yang langsung digunakan training. Derivasi objective menghasilkan
4.444 contoh setelah pemeriksaan visibilitas preimage repair, masking, dedup dan
budget token. Karantina kode benchmark mengeluarkan 56 contoh tambahan; dua body
completion berwarning dikeluarkan. Pool akhir memakai 4.386 contoh repository.

Fondasi dimulai dari **45.230 contoh rilis v2 yang sudah terverifikasi**, bukan
folder kandidat tahap antara45.303. Input/target fondasi harus identik dengan bukti
Git patch/after-byte rilis tersebut; target diperiksa ulang secara compile-only.
Karantina benchmark baru mengeluarkan 201 contoh karena repository dan73 karena
fragmen kode ternormalisasi, menyisakan44.956.

Repair context memakai seleksi window dari issue publik, tanpa gold patch/hints
untuk memilih window. Pemeriksaan eligibility sesudah seleksi memastikan preimage
seluruh hunk terlihat; window tidak diisi ulang memakai lokasi gold. Karena itu,
cohort repair merupakan subset bersyarat pada coverage selector ini, bukan seluruh
SWE-rebench atau skor agent dengan akses repository penuh.

Completion menghubungkan fungsi tak berubah dengan simbol yang diimpor dari file
lain di snapshot yang sama. Ini bukti hubungan struktural; belum membuktikan bahwa
setiap jawaban memerlukan encoder atau dependency context secara kausal. Context
ablation tetap diperlukan untuk kesimpulan paper.

## Panjang dan resource

Encoder maksimum **32.768 token**. Budget paired decoder **8.192 shifted input
tokens**, termasuk prefix; baseline total **41.472**. Ini batas maksimum per row,
bukan padding wajib. Curation/tokenization menolak overflow secara eksplisit.
Penentuan file/window repository adalah langkah context selection yang tercatat;
berbeda dari pemotongan token diam-diam saat training.

| Stream train | P50 | P95 | P99 | Maksimum |
| --- | ---: | ---: | ---: | ---: |
| Encoder | 343 | 7.367 | 12.480 | 32.767 |
| Paired decoder, shifted | 279 | 1.354 | 1.879 | 5.804 |
| Target, termasuk closing turn | 232 | 1.163 | 1.710 | 5.757 |
| Baseline decoder, shifted | 597 | 7.802 | 12.770 | 33.204 |

Dev mempunyai4.934 contoh dan **1.773.593 target tokens**. Maksimum encoder dev25.122,
paired decoder6.207, target2.679 dan baseline decoder25.230. Budget8K tidak berarti
semua target training panjang8K; encoder panjang dan output pendek tetap relevan.

QA memakai maksimum32 token jawaban. Completion, repair dan GSM8K mempunyai
allowance hingga budget decoder setelah prefix paired disisihkan; allowance sama
untuk baseline dan pasangan. Referensi completion juga harus muat pada allowance
tersebut. Ukuran GPU pretrained32K/8K belum merupakan hasil pengukuran.

Source archive cache lokal sekitar puluhan GiB diperlukan untuk kurasi; cache itu
tidak diunggah dan tidak diperlukan notebook konsumen. Notebook mengambil release
ZIP/Parquet yang sudah beku, memakai token cache int32 mmap dan satu model resident.
Tidak ada RL, Docker/VM repository harness atau distillation pada recipe ini.

## Benchmark default

| Koleksi | Cases per model | Peran dan scorer |
| --- | ---: | --- |
| sweqa_oracle_gemma32k_dec8k_v1 | 9.072 | QA code comprehension; strict letter accuracy |
| sweqa_noisy_oracle_gemma32k_dec8k_v1 | 9.072 | Pertanyaan sama, context dengan distraktor |
| crosscodelong_chunk_gemma32k_dec8k_v1 | 4.713 | Chunk completion; adapted EM/similarity/identifier F1 |
| crosscodelong_function_gemma32k_dec8k_v1 | 4.834 | Function completion; scorer yang sama |
| longcodeqa_gemma32k_dec8k_v1 | 107 | QA atas released32K prompts yang muat pada tokenizer Gemma |
| humanevalfix_python_v2 | 164 | Kontrol repair Python; native tests adaptation |
| gsm8k_zero_shot_v2 | 1.319 | Kontrol math retention |
| gsm8k_eight_shot_v2 | 1.319 | Pertanyaan sama, delapan demonstrasi train upstream |

Total **30.600 case/protocol runs per model**, bukan30.600 masalah independen.
SWE-QA mempunyai9.072 question IDs bersama; context oracle/noisy berbeda pada
seluruh pasangan. GSM8K mempunyai1.319 question IDs bersama. CrossCodeLongEval
chunk/function memakai9.547 upstream ID strings berbeda, tetapi tidak otomatis
berarti repository atau code family independen.

QA/code completion menguji pembacaan dan penggunaan context; skor tersebut tidak
menggantikan functional repository repair. HumanEvalFix merupakan kontrol repair
kecil. CanItEdit, RES-Q, RepoQA dan CrossCodeEval tetap katalog/karantina dengan
pekerjaan scorer, environment atau notices yang tercatat sebelumnya; tidak ikut
benchmark otomatis ini.

CrossCodeLongEval memakai released sparse rg1/FIM context, bukan retrieval query
yang memasukkan target. Dari masing-masing5.000 source cases, chunk mengecualikan
99 overflow dan188 literal references yang sudah terlihat; function mengecualikan
111 overflow dan55 reference terlihat. LongCodeQA mengecualikan6 dari113 source32K
cases karena tokenizer Gemma melewati32K; context tidak dicrop.

Completion scorer bernama **whitespace exact match, SequenceMatcher similarity,
regex identifier-set F1**. Ini bukan klaim parity terhadap scorer resmi.
QA meminta satu letter; repair meminta unified diff secara eksplisit bagi semua
model. Seluruh164 gold HumanEvalFix patches lolos native tests; itu validasi scorer,
bukan skor model atau bukti official runtime parity.

## Provenance, audit dan batas bukti

- Snapshot commit, hash archive/snapshot, source revision, original upstream row,
  candidate row, context-selection policy dan target identity disimpan.
- Setiap file context repository mempunyai nearest-license evidence; teks lisensi
  yang dirujuk dipertahankan di notices. Dataset metadata license tidak menggantikan
  lisensi kode. NextCoder seed lineage/code license dan lisensi tiap snippet benchmark
  tidak disertifikasi ulang.
- Train/dev memakai komponen repository, commit, before-seed dan source-task yang
  tak dapat dibagi. Varian dari issue yang sama tetap berada pada split sama.
- Karantina mencakup repository benchmark dan exact normalized code/subtree checks
  pada foundation before/after, context targets serta visible code sections.
  Semantic contamination dan seluruh subtree window yang tidak parsable tidak
  diklaim telah dihilangkan; paparan pretraining juga tidak disertifikasi.
- Public eval inputs terpisah dari gold ZIP. Semua collaborator private repo tetap
  dapat mengakses gold; pemisahan ini kontrak loader, bukan access-control isolation.
- Audit membandingkan setiap Parquet record_json dengan row arsip native. Target
  completion bebas syntax warning pada body wrapper AST; repair diuji patch apply,
  bukan harness SWE-rebench asli. Static checks tidak membuktikan functional quality.

Sumber dikunci: [SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2/tree/10483de0f50fe5da545942705a76c6150171af7f),
[SWE-QA](https://huggingface.co/datasets/lailaelkoussy/swe-qa/tree/579a8e1220c46742213998a36ccbb26c8bd26468),
[Repoformer/CrossCodeLongEval](https://github.com/amazon-science/Repoformer/tree/5b0571318e9918fd2af132c4b35077f9ea331133),
[LongCodeQA](https://huggingface.co/datasets/Steefano/LCB/tree/989d5eff750d65a72c522e47f8f745ef2e22906b).

## Reproduksi

```powershell
conda activate ai
python tools/prepare_context_data.py benchmarks
python tools/prepare_context_data.py repository
python tools/prepare_context_data.py tasks
python tools/prepare_context_data.py mix
python tools/audit_context_tokens.py
python tools/audit_context_tokens.py --legacy-only
python tools/build_context_release.py
```

Source caches/fondasi v2 yang beku diperlukan; proses tidak menimpa output selesai.
Pipeline consumer, resume, upload dan dua-checkpoint retention dijelaskan di
[Sequential training](SEQUENTIAL_TRAINING.md).

## Ukuran dan validasi implementasi

Rilis berisi843 file dengan total608.317.371 bytes (sekitar580MiB), termasuk
Parquet viewer, arsip native, manifest dan atribusi. Arsip training native berukuran
146.903.865 bytes (sekitar140MiB); cache archive repository sumber tidak ikut diunggah.

Regresi lokal:229 tes lulus,2 tes opt-in dilewati. Native tiny mencakup keenam
konfigurasi; tokenizer asli diaudit terpisah. Generator, kesamaan20cell/19cellidentik,
marimo strict, Pyrefly, ty, Ruff dan whitespace checks lulus. Ini tidak membuktikan
kelayakan VRAM32K/8K atau kualitas pretrained; training Molab tetap perlu diukur.
