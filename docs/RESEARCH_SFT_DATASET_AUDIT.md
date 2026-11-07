# Audit independen laporan kandidat dataset SFT

Tanggal: 7 Oktober 2026. Dokumen yang diperiksa: [RESEARCH_SFT_DATASET_CANDIDATES.md](RESEARCH_SFT_DATASET_CANDIDATES.md), versi 1.0.0, SHA-256 teks UTF-8 `d2f53bf59e4e86a4e305b801a3170bc7dd014e869cdc8977d31ed04d9d4244c8`.

**Kesimpulan: kandidatnya layak diteliti, tetapi status “Approved for Training Ingestion”, “Zero Contamination Verified”, dan campuran 100 ribu data bersih belum didukung bukti yang dapat direproduksi. Laporan belum dapat menjadi dasar ingestion produksi atau klaim paper.**

Audit ini memeriksa lampiran percakapan, laporan, working tree, sumber primer upstream, tiga smoke test dokumentasi, serta adapter dan tokenisasi pada 100 contoh awal InstructCoder. Tidak dilakukan audit kontaminasi seluruh dataset, training pretrained, atau eksekusi kode dalam dataset. Ketidaktersediaan bukti dalam bahan yang diperiksa tidak membuktikan bahwa agent tidak pernah melakukan pemeriksaan lain.

## Temuan yang harus diperbaiki

| Prioritas | Lokasi laporan awal | Temuan dan koreksi |
|---|---|---|
| P1 | Baris 6, 38, 507–601 | Klaim audit kontaminasi sudah selesai dan hasil nol belum memiliki manifest blacklist lengkap, benchmark fingerprints, skrip pemindaian, log, atau daftar baris yang dibuang dalam bahan yang diperiksa. Section 9 tidak mereproduksi pemindaian tersebut. Turunkan status menjadi kandidat dengan kontaminasi belum terverifikasi. |
| P1 | Baris 559–575 | Angka 471 repo Python benar, tetapi deskripsi “popular top repositories” dan contoh Django/Flask/scikit-learn tidak sesuai metode benchmark asli: repo dibuat 2023-03-05 sampai 2023-06-15, diambil 2023-09-01, bukan fork, ukuran zip di bawah 1 MB, lalu difilter lagi. Angka 42 repo overlap dan 5.000 baris terbuang belum terverifikasi. Daftar blacklist harus diturunkan dari artefak benchmark sebenarnya, bukan daftar repo terkenal. |
| P1 | Baris 668 | Laporan mengikat word embeddings DAN PLE antar encoder/decoder. Implementasi hanya mengikat ordinary word embeddings pada pasangan simetris; PLE terpisah. PLE adalah per-layer embeddings, bukan “Pleasant Language Embeddings”. Perhitungan penghematan yang mengikuti klaim tying PLE harus dibuang. |
| P1 | Baris 56, 137–154, 635–696 | GB dan GiB tercampur; estimasi guard bukan peak VRAM terukur. 17,71 GB sekitar 16,49 GiB; 28,51 GB sekitar 26,55 GiB. Angka 9,5/11,75 GiB dari riset sebelumnya adalah biaya dua PLE ditambah satu ordinary word table, bukan penghematan tying PLE. Headroom 16 GiB adalah knob cadangan, bukan jaminan aktivasi/workspace selalu muat atau bebas OOM. Kapasitas GPU harus dibaca dari runtime. |
| P1 | Baris 622–663 | Median token, yield, overflow, 155/400 ms, dan durasi 4,3/11,1 jam tidak mempunyai hasil profiling dataset yang disertakan. 38,60 juta token adalah hasil perkalian asumsi, bukan korpus yang sudah ditokenisasi; encoder/prefix tokens juga bukan seluruhnya token supervised pada loss. Gunakan label ilustrasi atau hapus sampai pilot mengukur konfigurasi aktual. |
| P1 | Baris 637, 729–764 | 100 ribu contoh, batch 1, accumulation 4, satu epoch sekitar 25 ribu optimizer updates. Curriculum 50 ribu + 50 ribu optimizer steps mengonsumsi sekitar 400 ribu contoh, bukan satu epoch campuran 100 ribu. Warm-up 500, LR tahap kedua, dan cosine decay juga bukan recipe default helper saat ini. Pisahkan rencana eksperimen dari implementasi yang tersedia. |
| P2 | Baris 609–620 | Ukuran unduhan beberapa dataset salah. NextCoder enam Arrow shards berjumlah 2.627.226.320 byte, bukan 6,15 GB. GitHub Code Review berjumlah 652.886.604 byte, bukan 2,58 GB. Biaya cache/materialisasi terpisah dari bytes file asli. Total pool 9,06 GB dan ukuran subset/preprocessed perlu dihitung ulang. |
| P2 | Baris 164, 224–236, 249–251, 319 | InstructCoder tidak mendeklarasikan lisensi dataset pada metadata card yang diperiksa. “Open Research” dan `open-research` bukan lisensi upstream yang terverifikasi. GitHub Code Review mendeklarasikan `other`; card mengklaim sumber permissive, tetapi adapter tidak boleh mengganti provenance menjadi lisensi generik `permissive-oss`. NextCoder memang mendeklarasikan MIT. |
| P2 | Baris 344, 546, 553, 581 | Sumber NextCoder yang diperiksa tidak membuktikan klaim spesifik “author audit 0% CanItEdit/HumanEvalFix leakage”. Card menyebut tidak memasukkan benchmark/test splits; itu tidak sama dengan pengukuran near-duplicate nol. Pada preprint v1 tidak ditemukan pembahasan contamination/overlap. Jangan mengatribusikan hasil audit nol kepada penulis tanpa halaman/tabel pendukung. |
| P2 | Baris 195–409, 828–892 | Tiga PASS di section 9.2 adalah tes kontrak atas dict buatan yang sudah dinormalisasi. Tidak memanggil ketiga fungsi `map_*` pada data upstream. Tidak memverifikasi adapter NextCoder, patch apply ke snapshot asli, token overflow seluruh korpus, atau decontamination. |
| P2 | Baris 53, 445–459, 595 | Notice CyberNative 6 Oktober 2026 memang ada pada revision yang dicantumkan. Namun istilah tepatnya correction/audit notice, bukan penarikan dataset. Label preference tidak merupakan ground truth keamanan terverifikasi; audit statis terbatas dan kesimpulan coverage inconclusive. Mengecualikannya dari pilot masuk akal, tetapi jangan mengklaim seluruh dataset terbukti cacat atau sudah withdrawn. |
| P2 | Baris 598 | SWE-smith tidak merupakan tabel before/after siap pakai. Namun materialisasi snapshot Git dan pembalikan patch dapat dirancang secara statis tanpa menjalankan VM/Docker. Harness resmi untuk membuktikan task berhasil merupakan kebutuhan terpisah. Pengecualian pilot boleh berdasarkan kompleksitas adapter, bukan “SFT wajib VM”. |
| P2 | Baris 550–554, 583–584, 785 | Temporal precedence saja tidak membuktikan tidak ada shared code/template. Nama fungsi umum tidak cukup untuk memutuskan leakage. Skor CrossCodeEval stabil dalam ±1,5% juga tidak membuktikan tidak ada kontaminasi. Perlukan pemeriksaan data dan provenance, bukan inferensi dari skor. |

## Metadata yang berhasil dikonfirmasi

Data berikut berasal dari API resmi saat audit, bukan penghitungan penuh oleh pipeline kurasi kita.

| Dataset | Revision head sesuai laporan | Metadata yang dikonfirmasi | Batas bukti |
|---|---|---|---|
| InstructCoder | `6a778a720284d6520b56bd03d5c3070930d41071` | 108.391 train + 5.708 validation = 114.099; fields input/output/instruction; file train+valid 146.961.219 byte | README menyebut 114.239; angka rilis berbeda. Seluruh baris belum divalidasi kualitas dan kontaminasinya. |
| GitHub Code Review | `c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca` | 334.323 train + 10.471 validation + 11.013 test = 355.807; 20 fields; 652.886.604 byte file Parquet | Jumlah Python positif 35 ribu, overlap 42 repo, serta yield 30 ribu belum dihitung ulang. Card menyatakan split repository-disjoint; pertahankan atau dokumentasikan perubahan split. |
| NextCoder | `924e23fa2fae2d2db568c3f8f18015b616a53715` | 381.123 train menurut dataset_info; dua fields prompt/completion; enam Arrow shards; MIT | 15.279 unique Python berasal dari card; 45.837 adalah 3 × jumlah tersebut, bukan hitungan language filter yang dijalankan audit ini. File dataset_info masih memuat download checksums/path revision lama dengan checksum null. Revision Git bukan checksum konten seluruh corpus. |
| CyberNative | `df94e8dbe0216d2e2ce4f1611d779df67d89dbc8` | Apache-2.0 pada card; correction notice dan audit 2026-10-06 tersedia; 4.656 pasangan menurut audit publisher | Peninjauan statis publisher mencakup 66 pasangan; tidak merupakan sertifikasi keamanan seluruh data. |

## Pengujian yang benar-benar dijalankan

1. Section 9.2 diekstrak sebagai fenced Python block dan dijalankan terhadap `editing_record` saat ini: ketiga assertion PASS. Ini tes sintetis kontrak.
2. Fungsi `map_instructcoder_to_bifocal` pada section 3.2 dijalankan pada **100 baris awal train** dari API viewer: **96 diterima, 4 ditolak**, tanpa error adapter.
3. Sebanyak 96 hasil tersebut ditokenisasi menggunakan fungsi `tokenize_editing_row` dari template notebook aktual, pasangan E2B–E2B, limit encoder 2048/decoder 512. Tokenizer base dan IT dimuat offline melalui `load_verified_tokenizer`, termasuk pemeriksaan hash dan revision. Hasil keseluruhan: **89 diterima, 7 token_overflow, 4 adapter_rejected**.
4. Median pada **89 contoh yang diterima saja**: encoder 144 token, decoder prefix 41 token, target 196 token. Ini bukan median korpus. Klaim median target 85 dan overflow <3,5% dalam laporan tidak bisa dilabeli hasil terukur berdasarkan bukti ini.
5. Sampel pertama GitHub Code Review yang dikembalikan viewer berisi 43 baris non-Python, sehingga tidak dipakai untuk mengklaim verifikasi adapter Python. Sampel pertama NextCoder juga C. Permintaan filter Python untuk keduanya timeout; adapter Python kedua dataset ini tetap belum diverifikasi pada sampel upstream oleh audit ini.

**Batas sampling:** 100 contoh awal bukan random sample. Angka 89% bukan estimasi yield populasi. Penerimaan AST/format tidak membuktikan instruksi dipenuhi, bug benar-benar diperbaiki, atau tests lulus. Kode contoh dataset tidak dijalankan atau di-import.

SHA-256 payload sampel first-rows InstructCoder yang diperiksa: `f1ccb4fbfdb3f93e476a7a9bbc43c03ec6349a400d2ee1588333b46cc52c12dc`. Endpoint viewer tidak dipin ke revision dataset; hash payload dicatat sebagai identitas sampel yang benar-benar diperiksa, terpisah dari revision head API.

## Pemeriksaan kode dan working tree

`src/bifocalgemma/config.py` menetapkan `share_ple=False`. `_tie_words` pada `src/bifocalgemma/models/v2/native.py` hanya menyamakan decoder/head dan ordinary words encoder simetris. Template notebook juga mengeluarkan `PLE_shared=False`.

`editing_record` menghasilkan diff dan metadata; validasi fungsi ini tidak menerapkan patch ke repo upstream. Adapter code review memperlakukan potongan window sebagai virtual-file source: validitas patch atas window berbeda dari validitas patch atas file asli pada commit upstream. Simpan offset/snapshot/provenance apabila ingin menggunakan output sebagai patch repo nyata. Parsing dedented chunk juga belum membuktikan potongan itu complete module; `return` di luar fungsi dapat lolos `ast.parse`.

Saat audit dimulai, working tree sudah berisi perubahan enam file tracked: `.gitignore`, `docs/TRAINING_NOTEBOOKS.md`, dua notebook training, `tests/test_offline_sft_training.py`, dan template bersama. Perubahan antara lain menambah cell upload HF dan mengubah parity menjadi 19 cell/18 identik. Laporan riset, `.agents/`, dan `skills-lock.json` masih untracked. Klaim laporan bahwa tidak ada file kode diubah tidak menjelaskan seluruh keadaan working tree; **diff tidak membuktikan agent riset tertentu yang mengubahnya**.

Audit ini menambah dokumen ini saja di repo. Perubahan notebook yang sudah ada tidak direvert, tidak dijalankan untuk upload, dan tidak dimasukkan ke commit oleh audit ini. Tidak ada training GPU atau publikasi eksternal.

## Syarat sebelum menaikkan status menjadi dataset siap ingestion

- Pin revisions dan hash artefak; rekam jumlah baris setiap split serta pemisahan bytes unduhan/cache/output.
- Buat adapter implementasi, uji sampel Python asli semua kandidat, pertahankan before-code/indentasi/provenance, dan lakukan patch roundtrip pada snapshot yang dideklarasikan.
- Bangun blacklist dari benchmark sebenarnya. Rekam exact/near-duplicate checks, repo intersections, metode/threshold, dan before/after counts. Status terbatas pada pemeriksaan yang dijalankan, bukan “zero contamination” universal.
- Ukur distribusi encoder/prefix/target dan overflow dengan tokenizer Gemma serta format unified diff aktual; kuota 40/30/20/10 adalah target, bukan hasil kurasi.
- Pastikan source/seed/PR groups tidak bocor antar split; pertahankan split repository-disjoint upstream bila dipakai untuk evaluasi held-out repository.
- Mulai pilot kecil dengan recipe joint direct SFT yang ada; ukur peak VRAM dan throughput pada panjang aktual sebelum memperkirakan 100 ribu contoh. Curriculum dua tahap adalah eksperimen terpisah setelah pilot.

## Sumber primer

- [InstructCoder card pada revision yang diperiksa](https://huggingface.co/datasets/likaixin/InstructCoder/blob/6a778a720284d6520b56bd03d5c3070930d41071/README.md).
- [InstructCoder sizes API](https://datasets-server.huggingface.co/size?dataset=likaixin/InstructCoder) dan [first-rows API](https://datasets-server.huggingface.co/first-rows?dataset=likaixin/InstructCoder&config=default&split=train).
- [GitHub Code Review card](https://huggingface.co/datasets/ronantakizawa/github-codereview) dan [sizes API](https://datasets-server.huggingface.co/size?dataset=ronantakizawa/github-codereview).
- [GitHub Code Review file metadata API](https://huggingface.co/api/datasets/ronantakizawa/github-codereview?blobs=true).
- [NextCoder file metadata API](https://huggingface.co/api/datasets/microsoft/NextCoderDataset?blobs=true), [dataset_info pada pinned revision](https://huggingface.co/datasets/microsoft/NextCoderDataset/blob/924e23fa2fae2d2db568c3f8f18015b616a53715/dataset_info.json), dan [dataset card](https://huggingface.co/datasets/microsoft/NextCoderDataset).
- [NextCoder paper ICML landing page](https://proceedings.mlr.press/v267/aggarwal25b.html) dan [preprint v1 yang diperiksa](https://arxiv.org/html/2503.03656v1). Pemeriksaan preprint v1 tidak menggantikan verifikasi seluruh teks versi ICML final.
- [CrossCodeEval paper, metode collection dan Table 1](https://arxiv.org/html/2310.11248v1) dan [repo resmi](https://github.com/amazon-science/cceval).
- [CyberNative correction notice pada pinned revision](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO/blob/df94e8dbe0216d2e2ce4f1611d779df67d89dbc8/README.md) dan [bounded static audit](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO/blob/df94e8dbe0216d2e2ce4f1611d779df67d89dbc8/audit/2026-10-06/methodology.md).
- [Paper tentang risiko kontaminasi Code-Feedback](https://aclanthology.org/2024.emnlp-main.777.pdf). Risiko HumanEval didukung penelitian; jangan mengubahnya menjadi klaim semua rows terkontaminasi atau semua seed berasal dari benchmark tanpa bukti tambahan.
- [SWE-smith harness resmi](https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/harness/utils.py); bandingkan kontrak static fixture yang sudah ada di [NATIVE_LOCAL_VALIDATION.md](NATIVE_LOCAL_VALIDATION.md), serta implementasi `src/bifocalgemma/data/swesmith.py`.
