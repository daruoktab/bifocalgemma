# Paket Dataset dan Evaluasi yang Layak untuk Molab

Diperbarui 7 Oktober 2026. Dokumen ini menjadi rekomendasi aktif setelah [audit laporan kandidat](RESEARCH_SFT_DATASET_AUDIT.md). Rekomendasi resource didasarkan pada schema, profiling CPU/tokenizer, dan implementasi sekarang; bukan hasil training atau benchmark BifocalGemma.

**Pilihan awal: EditPackFT untuk direct patch SFT, HumanEvalFix-docs Python untuk functional repair, serta held-out editing untuk diagnosis.** CanItEdit merupakan evaluasi tambahan dengan batas dependency yang eksplisit. CrossCodeEval merupakan diagnostik lintas file. NextCoder Python ditambahkan setelah pilot pendek berhasil. Tidak ada RL, teacher, VM, atau Docker pada jalur SFT yang dipilih.

**Batas scope:** paket awal di atas merupakan pilot adaptasi S1. Studi akhir untuk
judul repository encoding tetap memerlukan corpus/evaluasi repair dengan context
lintas file dan architecture ablations. [Kecocokan Paper Track](COMPETITION_GUIDE.md)
menjelaskan perbedaan pilot, kontribusi riset, dan aturan leaderboard prediksi.

Pencarian lanjutan menemukan RES-Q dan RepoQA serta core repo SFT sources yang
lebih dekat ke judul. Lihat [riset repository lengkap](RESEARCH_REPOSITORY_DATA_AND_BENCHMARKS.md).
Tabel di bawah tetap paket pilot S1; tidak membatasi keseluruhan paper.

## 1. Keputusan per dataset

Semua ukuran di bawah adalah decimal MB/GB file upstream, bukan ukuran GPU, RAM, atau cache hasil konversi. Jumlah upstream bukan jumlah data training setelah filtering/split.

| Dataset | Bukti dan kecocokan | Resource / pekerjaan tambahan | Keputusan |
|---|---|---|---|
| **EditPackFT** | 22.602 records Python; instruksi dan before/after sudah tersedia; provenance commit/path/repos/license; sesuai tujuan instructional editing. Tidak seluruhnya verified bug repair. | 48.124.127 byte Parquet; sudah dapat dibaca notebook. Profil seluruh corpus tersedia di section 3. MIT dataset card, pertahankan lisensi tiap source record. | **Data pilot utama.** Mulai 2.000 train / 200 dev dari manifest beku. |
| **CommitPackFT Python** | 56.025 records upstream; before/after dan subject tersedia. Sumber asal EditPackFT sehingga bukan tambahan independen. | 135.858.935 byte JSONL. Tambahan pembersihan instruksi, dedup pair/commit/near-duplicate lintas sumber dan benchmark diperlukan. | **Opsional, bukan campuran otomatis.** Tambah hanya bila memberikan contoh unik/berguna. |
| **NextCoder Python** | Card: 15.279 seed Python × 3 varian instruksi = nominal 45.837 rows. MIT. Prompt/code fences harus dipisahkan, lalu full-code target diubah menjadi diff. | Seluruh enam Arrow shards 2.627.226.320 byte. Sampel Python menunjukkan output patch jauh lebih panjang; reader notebook belum menerima Arrow. Sampling/konversi dilakukan sekali secara CPU, kemudian gunakan local JSONL. | **Tambahan tahap kedua.** Batas pasangan 2.048/2.048 dan baseline total 4.096 merupakan titik awal yang perlu diperiksa pada seluruh subset. |
| **InstructCoder** | 114.099 rows menurut viewer, input/output/instruction; Python function editing. Sampel mengandung artefak/ketidaksesuaian instruksi yang tidak bisa diselesaikan AST saja. | 146.961.219 byte train+valid JSON; card yang diperiksa tidak mendeklarasikan lisensi dataset. Sampel 100 bukan audit seluruh corpus. | **Kandidat bersyarat.** Jangan otomatis masuk pilot atau mengisi license `open-research` seolah lisensi terverifikasi. |
| **GitHub Code Review** | 355.807 rows lintas bahasa; Python positif perlu dihitung/filter. Review window bukan full file; komentar tidak otomatis hanya menjelaskan perubahan target. Card menyatakan split repo-disjoint dan source permissive, metadata license `other`. | 652.886.604 byte Parquet semua bahasa. Butuh source/offset provenance, quality filter, duplicate PR grouping, benchmark intersections, dan uji adapter Python upstream. Yield 30 ribu bersih belum terbukti. | **Tunda dari pilot utama.** Berguna untuk studi review-driven editing, belum corpus repository repair siap pakai. |
| **PatchPilot Patchgen** | Format SEARCH/REPLACE dan lineage SWE-bench tidak sejalan langsung dengan pipeline diff kita. | Konversi dan pemeriksaan leakage/format diperlukan; klaim overflow >40% belum diukur ulang. | **Tidak digunakan pada studi awal.** Alasan cukup: objective/provenance/adapter tambahan, bukan persentase tak terukur. |
| **Code-Feedback** | Multi-turn execution feedback; penelitian melaporkan risiko leakage HumanEval. Bukan before/after editing drop-in. | Ekstraksi percakapan/fix pairs, lineage audit dan decontamination baru diperlukan. Tidak intrinsik memerlukan RL, tetapi menambah scope. | **Tidak digunakan pada studi awal.** |
| **CyberNative Security DPO** | Publisher correction notice menyatakan chosen/rejected bukan ground truth security terverifikasi. Preference data bisa diubah menjadi SFT, tetapi validitas label tetap masalah. | Perlu contextual review; audit publisher hanya statis terbatas. | **Tidak digunakan.** Bukan kebutuhan tujuan repair umum kita. |
| **SWE-smith Python** | Bisa dipakai untuk SFT. Patch upstream menyisipkan bug; repair menggunakan arah sebaliknya dari buggy snapshot. | Snapshot materialization, tests provenance dan repo-specific adapter; menjalankan harness resmi biasanya memakai container. Materialisasi statis sendiri tidak wajib VM. | **Lanjutan, bukan prasyarat pilot.** |
| **SWE-Gym / SWE-smith trajectories** | Lingkungan repo/agent trajectories melampaui one-shot offline patch SFT. | Banyak teks/tool traces dan executable environments, tidak memberi solusi murah otomatis. | **Ditunda.** |
| **Static multi-file records** | Sangat sesuai klaim codebase/context understanding jika snapshot sebelum edit dan target dipisahkan. Target tetap boleh satu file meskipun input lintas file. | Corpus riil belum dikurasi. Snapshot Git, retrieval dan filtering bisa dilakukan tanpa VM. | **Tahap khusus setelah pilot.** Jangan menyebut single-file pilot sebagai bukti cross-file repair. |

Sumber metadata dataset baru dan kekeliruan laporan awal dijelaskan dalam [audit independen](RESEARCH_SFT_DATASET_AUDIT.md). Label “kandidat” tidak berarti bebas kontaminasi atau siap ingestion.

## 2. Pilihan benchmark dan batas protokol

| Evaluasi | Tugas / jumlah | Kecocokan dan resource | Status penggunaan |
|---|---|---|---|
| **Held-out EditPackFT** | Prediction diff → apply ke before → reconstructed code; dev CE, apply rate, exact edited-code match | Tidak perlu menjalankan repository. Exact match dapat menolak solusi alternatif benar. Split saat ini commit-group, bukan repo-disjoint. | Diagnosis pilot; CE tersedia, prediction diagnostics menunggu generator/parser. |
| **HumanEvalFix-docs Python** | 164 buggy functions; input function specification + buggy implementation; hidden tests untuk scoring | Full dataset Python hanya 191.279 byte. Pemeriksaan AST seluruh prompt/buggy/canonical/test menemukan import standar Python. Patch apply + native CPU tests dapat dipisahkan dari GPU generation. | **Benchmark functional repair utama.** Native scorer perlu dirakit dan diuji; tidak perlu setup repo/VM. |
| **CanItEdit** | 105 program, descriptive dan lazy: 210 prompts yang dilaporkan terpisah | Data 250.477 byte. 22 tasks memiliki import pihak ketiga yang terdeteksi, 83 tidak. Official scorer memakai Docker/Podman dan dependency lama, termasuk vllm. | **Tambahan bertahap.** Mulai native subset 83 tasks setelah gold-test validation; label subset, jangan laporkan sebagai full official score. |
| **CrossCodeEval Python** | 2.665 line-completion examples dari 471 repos; cross-file context | Archive resmi semua bahasa 41.997.264 byte. Code EM/edit similarity tidak perlu menjalankan repo. Retrieval context dan split harus berasal dari data asli. | Diagnostik lintas file, mulai 200 tasks tetap setelah adapter generation tersedia. Bukan repair pass rate. |
| **SWE-bench Lite / Verified** | Repo-level repair dengan executable test environments | Harness resmi memakai Docker dan repo dependencies. Tidak cocok sebagai syarat pilot tanpa VM/container. | Tidak digunakan pada evaluasi awal. |

### HumanEvalFix: dokumen dan tests bukan satu setting

Pilih **humanevalfixdocs-python**, bukan mencampur setting docs dan tests. Bentuk virtual before-file dari `prompt + buggy_solution`; specification berasal dari docstring yang memang terlihat pada setting docs. Patch-only output diterapkan ke before; program hasil dipakai dalam scorer yang menambahkan setup/test upstream. Canonical solution, hidden tests, bug labels dan hasil scoring tidak menjadi context model. Doctest examples dalam prompt tetap bagian specification resmi yang terlihat.

Setting humanevalfixtests memang memberikan tests tertentu sebagai input; itu eksperimen berbeda. Jangan melabeli penggunaan tests input pada setting tersebut sebagai leakage, atau menggunakannya diam-diam pada docs setting.

Kita menggunakan output **unified diff**, sedangkan banyak hasil paper memakai repaired code. Sebut “HumanEvalFix-docs, diff-output adapter” dan gunakan task/scorer yang sama untuk semua enam model. Jangan mengklaim reproduksi skor paper tanpa mencocokkan prompt, decoding dan runtime. Native CPU runner wajib memverifikasi gold solutions terlebih dahulu, mencatat timeout dan memakai workspace evaluasi terpisah tanpa HF credentials. Batas process/time bukan sandbox keamanan setara VM.

### CanItEdit: dependency gate nyata

Official benchmark requirements berisi pandas2.0.2, torch2.1.2, numpy1.23.5, sklearn1.2.2, scipy1.10.1, vllm0.2.7 dan lain-lain. Jangan memasangnya ke notebook training Python3.13/Torch2.11/Transformers5.18. Official README secara eksplisit memakai container untuk dependency.

Audit statis seluruh 105 tasks menemukan **83 tanpa import non-stdlib yang terdeteksi** dan **22 dengan dependency pihak ketiga**, termasuk satu task `llm_inference` dengan flask/vllm. Ini klasifikasi import, belum bukti tests bisa dijalankan: import dinamis atau asumsi runtime belum diuji. Subset native harus melewati gold `after + tests` dan kesesuaian scorer sebelum dipakai. Full CanItEdit memerlukan CPU interpreter/environment terpisah yang cocok atau port dengan validasi semua gold tests; bila task dihapus, tampilkan subset IDs/coverage dan jangan menyebut skor full official.

Laporkan descriptive dan lazy terpisah, satu deterministic attempt per prompt untuk pilot. 83 tasks berarti 166 prompt outcomes, bukan 166 program independen. ExcessCode memerlukan coverage sesuai scorer; tidak otomatis tersedia dari pass/fail saja.

### CrossCodeEval: completion diagnostic

Gunakan preprocessed retrieval **tanpa reference/gold-assisted retrieval**. Daftar repo blacklist diturunkan dari artefak benchmark, bukan nama repo populer. Metode asli memilih repo yang baru dibuat Maret–Juni 2023; Django/Flask bukan blacklist yang disimpulkan dari paper tersebut.

Model harus menerima prompt completion yang sama dan menghasilkan continuation, bukan unified diff. Ini memerlukan generation mode tambahan. Patch-SFT dapat memengaruhi completion behavior; perubahan skor tidak otomatis mengukur repair atau membuktikan tidak ada kontaminasi. Mulai greedy, satu completion, `max_new_tokens=50` mengikuti protokol paper awal. Batas context ditentukan setelah format retrieval ditokenisasi dengan Gemma; EM/edit similarity lebih sederhana daripada identifier metrics yang memerlukan parser. 200 tasks merupakan subset diagnostik yang harus dibekukan, bukan skor full 2.665.

## 3. Profil token: hasil terukur, bukan perkiraan

Semua pengukuran memakai tokenizer Gemma base/IT offline dengan hash/revision diperiksa, `enable_thinking=False`, chat template resmi dan patch yang dibuat `editing_record`. **Decoder length** berarti prefix + patch + closing turn setelah satu shifting; berbeda dari panjang patch saja. Baseline total menaruh context yang sama dalam prefix decoder.

### Seluruh EditPackFT

Sumber dipin `067b4c882f44eca19329fedd275438fc3c060f3f`; Parquet SHA-256 `22999d95436bb32dc7a8eb0924551e1c49c80ebad677f6d9cfebcd9def8a10a3`.

22.602 raw records → **22.583 diterima kontrak/dedup**, 19 ditolak `unsupported_path`. Ini sebelum split, semantic quality audit dan near-duplicate audit.

| Token field | P50 | P95 | P99 | Maksimum |
|---|---:|---:|---:|---:|
| Encoder context | 242 | 404 | 440 | 749 |
| Decoder pasangan | 242 | 557 | 768 | 1.113 |
| Target patch + closing turn | 211 | 526 | 738 | 1.085 |
| Baseline decoder total | 493 | 844 | 1.055 | 1.682 |

| Batas encoder / decoder pasangan | Diterima dari 22.583 |
|---|---:|
| 2.048 / 512 | 21.011 |
| 2.048 / 1.024 | 22.580 |
| 2.048 / 2.048 | 22.583 |

Karena encoder maksimum 749, **1.024 / 1.536** juga menampung seluruh 22.583 records yang lolos kontrak. Baseline total **2.048** cukup untuk format yang diprofilkan. Tidak ada alasan data ini memerlukan input 32k/output 8k. Panjang adalah batas, bukan padding wajib; batch 1 tetap mengikuti panjang record aktual. Split/filter final dapat mengubah counts, bukan mengubah fakta raw profile ini.

### Kandidat tambahan dan benchmark

| Sumber dan cakupan | Encoder P50 / P95 / max | Decoder pasangan P50 / P95 / max | Target P50 / P95 / max | Catatan |
|---|---|---|---|---|
| InstructCoder first100, 96 diterima adapter | 147 / 310 / 535 | 248 / 519 / 987 | 208 / 478 / 945 | 89 dari 100 muat 2k/512; 4 adapter rejects, 7 overflow. |
| NextCoder 100 Python rows pada offset150000 | 576 / 798 / 968 | 927 / 1.360 / 1.705 | 849 / 1.282 / 1.584 | Hanya 3/100 muat 2k/512. Ada **34 unique before/after pairs dan 34 seed groups**; 100 rows bukan 100 edit independen. |
| CanItEdit descriptive, semua 105 | 371 / 1.130 / 1.949 | 380 / 751 / 1.421 | 266 / 638 / 1.105 | Inference input2k dan max_new 1536 merupakan budget awal yang cukup untuk reference sizes yang diperiksa. |
| CanItEdit lazy, semua 105 | 330 / 1.106 / 1.889 | 315 / 708 / 1.282 | 266 / 638 / 1.105 | Target sama; instruction style berbeda. |
| HumanEvalFix-docs, semua 164 | 209 / 418 / 638 | 145 / 208 / 262 | 112 / 175 / 229 | Seluruh references muat pasangan2k/512; baseline total max883. |

Sampel InstructCoder dan NextCoder tidak acak, berdekatan, dan **tidak digunakan untuk menyatakan yield populasi**. Profil full CanItEdit/HumanEvalFix tidak merupakan skor model. Reference lengths dipakai untuk diagnosis resource; gold/reference tidak menjadi context atau alat memilih hanya benchmark tasks yang mudah. Format/task adapter final harus diprofilkan ulang bila prompt/path/closing policy berubah.

Kuantil menggunakan sorted array dengan indeks `floor((n−1)*q)`. Hasil lengkap, fingerprints dan dependency task IDs: [JSON profil](data/molab_data_profile_20261007.json).

## 4. Panjang dan recipe yang direkomendasikan

Preferensi terbaru: **sesuaikan input/output dengan dataset**, bukan mewajibkan32k/8k.

| Tahap | Data | Batas pasangan saat SFT | Baseline total saat SFT | Tujuan |
|---|---|---|---|---|
| S0 | 16–32 examples pendek dan contoh dekat batas | Mulai512/512 lalu stress shapes kandidat | Context + target lengkap, bukan512 total | Cek finite loss/gradient, checkpoint dan peak GPU nyata. |
| S1 | EditPackFT, 2.000 train / 200 dev terkurasi | **1.024 encoder / 1.536 decoder total** | **2.048 total** | Menghindari pembuangan target panjang pada corpus yang sudah diprofilkan. |
| S2 | EditPackFT + NextCoder Python yang diaudit | **2.048 / 2.048** kandidat awal | **4.096 total** kandidat awal | Longer diffs dan variasi instruksi; pilih limit setelah profil seluruh subset, bukan dari 100 rows saja. |
| S3 | Static multi-file corpus riil | Bucket context berdasarkan distribusi nyata, naik4k → 8k → 16k → 32k hanya jika diperlukan | Same visible context, prefix + output allowance | Membuktikan kontribusi context lintas file; butuh kernel/inference work tambahan. |

Knob notebook saat audit masih **2.048/512**. Tabel di atas merupakan konfigurasi yang direkomendasikan sesudah memory stress S0; dokumentasi ini tidak mengklaim kernel/generator sudah mendukung seluruh tahap. Tidak ada perubahan otomatis ke training aktif.

Untuk S1: attention LoRA rank 16/alpha 32, frozen ordinary words/PLE, alignment trainable hanya asimetris; microbatch 1, accumulation 4, policy LR 2e-5/warmup 5 yang sudah tersedia. Pertama100 optimizer updates maksimal (~400 example visits), bukan satu epoch 2.000 examples. Satu epoch penuh2.000 examples dengan accumulation 4 sekitar500 updates. Semua pasangan memakai manifest/train order/budget yang sama; gunakan token totals untuk menjelaskan variasi lengths.

Mixture S2 adalah eksperimen terpisah, misalnya80% unique EditPackFT dan20% unique NextCoder, **bukan janji kuota bersih100k**. Dedup before/after saat ini menyatukan tiga varian instruksi NextCoder pada pasangan sama. Jika ingin mempertahankan instruction variants, tentukan policy explicit dan seed-group splits, serta hindari memberi seed tersebut bobot 3× tanpa sengaja.

## 5. Resource GPU, disk, dan durasi

### Satu model setiap run

GPU disebut96GB Blackwell; total/free bytes, RAM dan disk aktual harus dicatat pada sesi pengguna. Jalankan satu pasangan training atau satu baseline inference pada satu waktu. Probe residency terdahulu bukan bukti beberapa full-length training muat bersama. Hapus model/optimizer lama dan periksa free VRAM sebelum load berikutnya.

Empat source files `model.safetensors` upstream memerlukan:

| Source | File bytes |
|---|---:|
| E2B base | 10.246.621.918 |
| E2B IT | 10.246.621.918 |
| E4B base | 15.992.595.884 |
| E4B IT | 15.992.595.884 |
| **Keempat sumber** | **52.478.435.604 (~52,48GB)** |

File monolithic mencakup media yang tetap diunduh walaupun GPU memakai text-only. S1 hanya memerlukan E2B base/IT (~20,49 GB) untuk run pertama; cache bisa digunakan kembali. Full four-pair study dapat menjaga empat sumber bila disk cukup. **80 GB free disk** adalah budget perencanaan untuk full cache + data + conversion + reports, bukan kapasitas Molab yang telah diukur. Bila disk kurang, download dua sumber yang dibutuhkan per run secara bertahap dan bersihkan hanya cache yang sudah terverifikasi aman untuk dibuang. Dataset bukan biaya disk terbesar.

NextCoder menambah 2,63 GB raw jika seluruh Arrow diambil. Jangan unduh semua training candidates sekaligus atau menyimpan semua padded tensors GPU. Baca Parquet/Arrow/JSONL dengan batch CPU, tulis subset unik sekali, tokenisasi/cache offline, dan transfer microbatch saja ke GPU.

Durasi belum dapat dinyatakan dalam jam. Ukur sesudah warm-up kernel pada actual records:

`estimated_training_time = measured_median_seconds_per_optimizer_update × planned_updates`

Laporkan interval/p95, token totals, eval/checkpoint overhead, compile warm-up dan download terpisah.100 updates × 4 models =400 updates untuk smoke study; full one-epoch2k × 4 sekitar2.000 updates. Jendela Molab/timeouts belum diketahui; checkpoint/resume v2 wajib dipakai untuk run panjang.

### Mengapa32k/8k bukan knob aman pada eager implementation

Dengan batch 1 dan8 query heads, tabel berikut menghitung **satu tensor dense**, bukan total VRAM:

| Tensor | Elemen / precision | Ukuran |
|---|---|---:|
| Encoder scores pada 32.768 tokens | 8 ×32768×32768, BF16 / FP32 | 16 /32GiB |
| Decoder merged scores,8.192 queries + 32.768 encoder keys | 8×8192×40960, BF16 /FP32 | 5 /10GiB |
| Logits8.192×262.144 vocabulary | BF16 /FP32 | 4 /8GiB |

Kode eager mempertahankan dense mask/attention intermediates; CE mengonversi logits keFP32. Multiple layers, gradients, activations, softmax tensors dan base weights menambah biaya. Window mask tidak dengan sendirinya menghilangkan tensor dense yang sudah dibuat. Ini alasan tidak menganggap32k/8k training layak pada kode sekarang; bukan peak model yang telah diukur.

S3 memerlukan attention backend yang benar-benar menjaga joint softmax, causal/local decoder masks, encoder visibility, GQA, KV ownership/sharing, dan native scaling/norm. Flash/SDPA/Flex bukan toggle yang otomatis memperbaiki custom merged block; verifikasi fused kernel yang dipilih dan cegah fallback math yang membentuk dense tensors. Activation checkpointing memerlukan pengujian khusus karena shared-KV state saat ini dibawa melalui same-forward mutable state. Loss/logits chunking, encoder reuse, decoder cache dan last-token logits juga belum terintegrasi. Quantization linear saja tidak menghapus biaya PLE/ordinary embeddings atau activations.

## 6. Protokol enam model yang sebanding

Empat pasangan tetap main target; baselines E2B-IT danE4B-IT, tanpa T5Gemma tambahan.

- Berikan instruction dan before-code/context yang sama. Pasangan menaruh context diencoder; baseline menaruhnya di decoder prefix. **Samakan isi yang terlihat, bukan angka total decoder length.**
- Patch format/path, thinking=False, no tools, greedy one attempt, stop policy dan max_new_tokens sama untuk task yang sama. Target gold tidak dipakai untuk retrieval, context selection, early stopping, atau repair retry.
- HumanEvalFix-docs:164 tasks, max_new 512 sebagai budget awal. CanItEdit:83 dependency-screened tasks ×2 styles, max_new 1536; native runner dan gold validation masih diperlukan. Overflow/incomplete patches merupakan outcomes yang dicatat, bukan task dibuang diam-diam.
- Six-model pilot tersebut menghasilkan6 × (164 + 166)=**1.980 generations**, bukan training6 models bersamaan. Full CanItEdit akan menjadi6 × (164 + 210)=2.244 generations jika dependency gate lulus.
- Original IT baseline dilabeli zero-shot. Empat trained pairs vs dua untuned baselines menjawab peningkatan sistem, **belum mengisolasi efek arsitektur dari SFT**. Tambahkan matched-budget LoRA SFT pada dua baseline untuk klaim arsitektur; itu dua run tambahan dalam keluarga yang sama, bukan model comparator baru.
- Save generations lebih dulu, unload GPU model, lalu CPU scoring. Laporkan patch syntax/apply failures, functional pass, timeout, infrastructure errors dan denominator yang dibekukan. Runtime/model failures tidak boleh membuat task hilang dari denominator; infrastructure recovery harus dicatat.
- RQ encoder-versus-decoder memakai asymmetric duel dan controls yang telah dijelaskan di [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md). Single-file benchmark tidak membuktikan repository-level understanding.

## 7. Leakage dan provenance: hasil terbatas yang benar-benar ada

Untuk **seluruh 22.583 EditPackFT records yang lolos kontrak**, exact whole-program hashes before/after dibandingkan dengan 537 unique benchmark-code hashes dari CanItEdit before/after dan HumanEvalFix prompt+buggy/canonical. Normalisasi hanya CRLF→LF dan strip outer whitespace. Hasil **0 matching rows**.

Ini tidak memeriksa fragments, docstring similarity, near-duplicates, repo intersections, semantic equivalents, generator lineage, maupun pretraining contamination Gemma. Jangan melabelinya “zero contamination verified”. NextCoder/InstructCoder/GitHub review tetap memerlukan audit sendiri sebelum admission. Pemilihan source lebih lama dari benchmark saja tidak membuktikan tidak ada overlap.

Manifest wajib menyimpan revisions/file SHA, IDs/seed/commit/repo aliases, license, code/instruction/patch hashes, filter counts, tokenizer version/hash dan actual lengths. Train/dev grouping tidak menggantikan quarantine benchmark. Simpan benchmark IDs dan tests terpisah dari training files. Full corpus dan evaluation native belum dijalankan modelnya dalam riset ini.

## 8. Pekerjaan implementasi agar paket benar-benar bisa dijalankan

| Urutan | Komponen | Status / acceptance |
|---|---|---|
| 1 | S1 preparation dan frozen split | Converter/tokenizer notebook ada. Terapkan quarantine checks dan simpan final corpus manifest sebelum klaim hasil. |
| 2 | S0 memory stress pada real model | Jalankan pengguna di Molab, satu model, true optimizer steps pada longest actual records; jangan menyimpulkan full-shape dari small probe. |
| 3 | S1 SFT | Dua notebook families sudah menyediakan loader/LoRA/policy/v2 resume. GPU pilot/data-quality results belum merupakan output riset ini. |
| 4 | I01 autoregressive patch generation | **Placeholder** sekarang. Perlu encoder-once, target-free decoder prefix, stop/length handling; short no-cache reference bisa dipakai untuk correctness, bukan long-context production. |
| 5 | Diff extraction/apply dan native HumanEvalFix scorer | Belum backend production; verify gold patches/reference programs, exact upstream tests, subprocess results dan denominator. |
| 6 | CanItEdit native subset + CrossCodeEval mode | Adapt scorer/prompt mode dan bekukan subset; validasi gold/metric parity. Jangan memasang official old requirements ke training kernel. |
| 7 | S2 NextCoder serta matched SFT baselines | Adapter + Arrow conversion + complete Python profiling/dedup/lineage audit; no automatic 100k pool. |
| 8 | Multi-file/long-context study | Curate static snapshots dan validate optimized attention/cache; panjang ditentukan data. |

**Paket ini layak sebagai jalur bertahap pada single GPU Molab, dengan S0 sebagai pemeriksaan resource aktual. Repo sekarang belum memiliki generator dan scorer benchmark production, sehingga “layak dirancang” tidak sama dengan “sudah menghasilkan score”.**

## 9. Sumber primer dan bukti

- [EditPackFT card](https://huggingface.co/datasets/nuprl/EditPackFT), [CommitPackFT card](https://huggingface.co/datasets/bigcode/commitpackft) dan [OctoPack licensing/protocol](https://github.com/bigcode-project/octopack).
- [NextCoder card](https://huggingface.co/datasets/microsoft/NextCoderDataset), [API files](https://huggingface.co/api/datasets/microsoft/NextCoderDataset?blobs=true) dan [ICML paper](https://proceedings.mlr.press/v267/aggarwal25b.html).
- [HumanEvalPack Python pinned revision](https://huggingface.co/datasets/bigcode/humanevalpack/tree/9a41762f73a8cb23bb5811b73d5aab164efcf378/python), [OctoPack evaluation protocol](https://github.com/bigcode-project/octopack#evaluation).
- [CanItEdit pinned dataset](https://huggingface.co/datasets/nuprl/CanItEdit/tree/3c07f38b1f9385f3214fcea94d4664c79df0d36a), [official benchmark README](https://github.com/nuprl/CanItEdit/blob/main/benchmark/README.md), [requirements](https://github.com/nuprl/CanItEdit/blob/main/benchmark/requirements.txt).
- [CrossCodeEval paper](https://arxiv.org/html/2310.11248v1), [official data archive](https://github.com/amazon-science/cceval/blob/main/data/crosscodeeval_data.tar.xz).
- [PyTorch2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html), [FlexAttention](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html). Availability/suitability of particular kernels on Blackwell and custom merged masks still needs testing.
- Source file sizes Gemma: [E2Bbase](https://huggingface.co/api/models/google/gemma-4-E2B?blobs=true), [E2BIT](https://huggingface.co/api/models/google/gemma-4-E2B-it?blobs=true), [E4Bbase](https://huggingface.co/api/models/google/gemma-4-E4B?blobs=true), [E4BIT](https://huggingface.co/api/models/google/gemma-4-E4B-it?blobs=true).
- Profiling memuat no GPU execution. [JSON hasil](data/molab_data_profile_20261007.json) menyimpan full EditPackFT, full small benchmarks, sample-only candidates dan exact-overlap result; counts tidak merupakan quality score.
