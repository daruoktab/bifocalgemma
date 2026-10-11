
# Audit dan spesifikasi dataset v2 — 11 Oktober 2026

## Status yang telah terbukti

Seluruh **empat koleksi yang sebelumnya diterbitkan** telah diperbaiki dan diterbitkan
ulang pada repo dataset private
[daruokta/bifocalgemma-data](https://huggingface.co/datasets/daruokta/bifocalgemma-data/tree/d34a19b3badc0120755495c68227cba038819cd7).

- Commit HF: d34a19b3badc0120755495c68227cba038819cd7.
- Fingerprint rilis: 8ebbd10a5ce83d58fb2e4cc4ae9f989a57c22011b441ab7df821cd1e43b04400.
- Fingerprint pool: 6d4aaaa7688af41559de679562a209c5a0faf38eeac0ef5aaa817a9908c4d769.
- Training: **45.230 contoh =40.706 train +4.524 dev**.
- Main memiliki lima Parquet: dua split training dan tiga tampilan evaluasi.
- V1 tetap tersedia pada commit 75ea23ff82d54d6d0f663a33c1f0108ea5b59fc2; tidak ada rewrite history.

Ini remediasi seluruh 49.296 baris v1 yang sudah dirilis, bukan klaim mengulang admission
459.750 baris upstream dari awal. Setiap ID v1 masuk tepat satu kategori: 45.230
dipertahankan atau 4.066 dikeluarkan, tanpa ID hilang/terhitung dua kali. Ledger disimpan
di pool.zip; rujukan retained_id berarti pemenang pada tahap tersebut, dengan flag
retained_in_final_release untuk membedakan pemenang yang kemudian ikut dikarantina.

## Perbaikan yang dilakukan

| Alasan pengeluaran | Baris |
| --- | ---: |
| Edit hanya dokumentasi/format pada AST tanpa docstring | 1.617 |
| SyntaxWarning saat parse target | 308 |
| Ekspor berulang dari event Git/seed yang sama | 1.900 |
| Varian event sama tetapi before/after bertentangan | 104 |
| Input terlihat identik tetapi target kode bertentangan | 12 |
| Pasangan struktur before/after berulang | 52 |
| SyntaxWarning tambahan saat kompilasi | 56 |
| Target lolos parse AST tetapi gagal kompilasi | 17 |
| Overflow setelah pembentukan ulang patch | 0 |
| **Total dikeluarkan** | **4.066** |

Angka merupakan kategori eksklusif menurut urutan tahap remediasi, bukan jumlah kandidat
masalah yang dihitung sebelum penyaringan. Contoh kegagalan kompilasi mencakup await di
luar async, return di luar fungsi, dan nonlocal tanpa binding.

**Delapan patch diperbaiki**, bukan dibuang. Generator lama menggunakan pemisahan baris
Python yang juga memisahkan CR tunggal/formfeed/pemisah Unicode; Git menghitung delimiter
LF. Generator baru menjaga karakter tersebut di dalam isi baris, sehingga patch
menerapkan before menjadi after secara byte-exact. Teks before/after sumber tidak
ditulis ulang dan tidak dipotong.

Normalisasi Python kini mempertahankan NEWLINE, INDENT dan DEDENT sambil mengabaikan
lebar indentasi/comment yang tidak relevan. Fallback kode yang gagal tokenize menjaga
teks utuh. Indeks overlap lama ditolak sampai dibangun ulang dengan kebijakan yang cocok.

Dedup event menyimpan alias sumber serta repository/commit yang dibuang untuk menjaga
lineage split. CommitPack diprioritaskan pada varian event yang struktur kodenya sama
karena ekspornya mempertahankan header/modul yang hilang pada varian lain; ini tetap
ekspor upstream, bukan checkout Git yang sudah diverifikasi.

Penghapusan docstring adalah **heuristik struktur**, bukan pembuktian kesetaraan semantik:
perubahan __doc__ memang dapat mengubah perilaku. Edit dokumentasi saja sengaja dikeluarkan
dari cohort ini untuk fokus pada perubahan kode. Target bertentangan dikarantina tanpa
memilih jawaban yang dianggap benar hanya berdasarkan panjang atau sumber.

## Komposisi training yang benar-benar dipakai

| Sumber | Train | Dev | Total | Token target train |
| --- | ---: | ---: | ---: | ---: |
| nuprl/EditPackFT | 15.613 | 1.748 | 17.361 | 3.928.528 |
| bigcode/commitpackft, Python | 13.767 | 1.565 | 15.332 | 2.703.449 |
| microsoft/NextCoderDataset, Python | 11.326 | 1.211 | 12.537 | 9.041.449 |
| **Total** | **40.706** | **4.524** | **45.230** | **15.673.426** |

Revision upstream lengkap:

| Sumber | Revision | Dataset metadata |
| --- | --- | --- |
| [EditPackFT](https://huggingface.co/datasets/nuprl/EditPackFT/tree/067b4c882f44eca19329fedd275438fc3c060f3f) | 067b4c882f44eca19329fedd275438fc3c060f3f | MIT |
| [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft/tree/fc56fe33c030c6daa414c2b112c932b8eed085e6) | fc56fe33c030c6daa414c2b112c932b8eed085e6 | MIT |
| [NextCoder](https://huggingface.co/datasets/microsoft/NextCoderDataset/tree/924e23fa2fae2d2db568c3f8f18015b616a53715) | 924e23fa2fae2d2db568c3f8f18015b616a53715 | MIT |

NextCoder mempunyai **57,69% token target train**, walaupun sekitar 27,82% contoh train.
Tidak ada pembobotan sumber tersembunyi; per-source loss/hasil perlu dilaporkan saat pilot
agar dominasi token sintetis terlihat. InstructCoder dan github-codereview tidak masuk
rilis ini; laporan riset kandidat tidak sama dengan daftar sumber yang dikonsumsi kode.

Seluruh contoh memiliki satu file context dan target unified diff. Tidak ada
repository_patch_v1 dalam pool ini. Taxonomy statis: 44.294 substantive_edit_candidate
dan 936 syntax_adaptation_candidate. Ini mencakup editing/refactoring/penambahan fitur;
label tersebut bukan bukti seluruh contoh adalah bug fix.

Input paired: instruction + path + before di encoder, instruction/prompt di decoder.
Baseline menerima informasi context yang sama di prefix decoder. Target patch menjadi
supervision; after/alias mentah dalam artefak tidak dipakai sebagai input context.
Keenam konfigurasi memakai cohort dan split yang sama.

## Lisensi dan provenance

| Deklarasi lisensi kode | Contoh |
| --- | ---: |
| MIT | 17.561 |
| Apache-2.0 | 5.850 |
| BSD-3-Clause | 6.558 |
| BSD-2-Clause | 1.992 |
| ISC | 390 |
| Unlicense | 248 |
| CC0-1.0 | 94 |
| Unknown — seluruh NextCoder | 12.537 |

MIT metadata dataset NextCoder tidak membuktikan lisensi setiap seed kode asal.
git_commit=null untuk NextCoder, provenance_type=synthetic_edit_from_unresolved_seed,
code_license_verification=unverified_original_seed. Field commit/seed_program_hash
mempertahankan **signature grouping sintetis legacy**, bukan raw-file SHA atau commit
Git asli; commit_kind menjelaskan perbedaannya. before_signature baru menjaga struktur
Python. pair_hash mengikat teks before/after, sedangkan SHA file/arsip mengikat byte rilis.

Metadata commit Git pada semua ekspor Git yang dipertahankan mempunyai bentuk SHA40
yang valid. Keberadaan checkout, kebenaran instruksi/repair, seluruh hak kode/notice,
dan provenance seed asli belum dibuktikan. Repo tetap private dan tidak memakai satu
lisensi blanket untuk seluruh source code.

## Batas token dan distribusi nyata

Semua retained rows tercakup pengukuran ulang tokenizer Gemma asli yang diverifikasi
hash/revision, tanpa bobot model. E2B/E4B behavior assets diperiksa sebelum hasil profil
dipakai pada empat pasangan dan dua baseline.

| Panjang | Batas | Median | P95 | P99 | Maksimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Encoder context, termasuk BOS | 8.192 | 323 | 1.056 | 1.540 | 6.309 |
| Paired decoder sequence setelah shifting | 2.048 | 276 | 1.334 | 1.844 | 2.048 |
| Target supervision, termasuk penutup turn | — | 238 | 1.170 | 1.692 | 1.935 |
| Baseline decoder context + target | 10.752 | 564 | 2.129 | 2.741 | 7.400 |

Batas 2k adalah sequence decoder paired termasuk prefix, bukan jaminan 2k token patch
bebas di luar prompt. Tidak ada silent truncation. Batas input8k sesuai cohort ini;
GSM8K yang disiapkan juga tidak membutuhkan32k input.

Satu epoch: batch1, accumulation4, **10.177 optimizer updates**; 15.673.426 token
target train dan 1.709.122 dev. MAX_STEPS=0 tetap berarti seluruh epoch, bukan pilot
kecil terselubung. Durasi dan peak VRAM pretrained di Blackwell96GB tidak diukur dalam
audit CPU ini. Jalankan satu konfigurasi per run dan ukur pilot terlebih dahulu jika
ingin menentukan anggaran waktu; file data kecil tidak membuktikan activations/model fit.

## Benchmark yang diterbitkan

| Koleksi | Task unik dalam koleksi | Protokol/metric | Pemeriksaan yang dilakukan |
| --- | ---: | --- | --- |
| humanevalfix_python_v2 | 164 | Patch Python, native test pass adaptation | 164/164 gold patch applied dan lulus unit test |
| gsm8k_zero_shot_v2 | 1.319 | Zero-shot, greedy, first #### regex/string accuracy | 1.319/1.319 gold numeric reference diterima scorer |
| gsm8k_eight_shot_v2 | 1.319 | First8 upstream train demonstrations, metric sama | Task IDs/questions sama dengan zero-shot; scorer reference lulus |

Gabungan unik: **164 code tasks +1.319 math questions =1.483**; dua protokol GSM8K
menghasilkan dua pengukuran pada pertanyaan yang sama. Demonstrasi eight-shot berulang
di prefix memang disengaja. Instruksi HumanEvalFix yang sama juga bukan duplicate task:
context/snapshot/ID setiap soal berbeda.

HumanEvalFix bersumber dari bigcode/humanevalpack revision
9a41762f73a8cb23bb5811b73d5aab164efcf378, file python/test-00000-of-00001.parquet.
GSM8K memakai openai/grade-school-math revision 3101c7d5072418e28b9008a6636bde82a006892c:
7.473 train rows upstream; hanya first8 dipakai sebagai demonstrations, bukan coding SFT.
Keseluruhan1.319 test IDs dipertahankan. Tidak ada tools/calculator/RL/VM dan
enable_thinking=False. Scorer memakai marker pertama, menghapus koma dan membandingkan
string; tidak mengubahnya menjadi numeric equivalence.

| Profil benchmark, tokenizer asli | Median | P95 | Maksimum |
| --- | ---: | ---: | ---: |
| HumanEvalFix encoder context | 216 | 425 | 645 |
| HumanEvalFix raw gold diff | 112 | 175 | 231 |
| GSM8K0-shot encoder context | 63 | 109 | 195 |
| GSM8K8-shot encoder context | 1.121 | 1.167 | 1.253 |
| GSM8K0-shot baseline decoder prefix | 90 | 136 | 222 |
| GSM8K8-shot baseline decoder prefix | 1.148 | 1.194 | 1.280 |

Paired GSM8K decoder prefix34 token. Gold numeric answer maksimum7 token bukan panjang
chain-of-thought generation. Output budget tetap dipilih/frozen sebelum model evaluation.
Angka oracle/gold validation di atas **bukan skor BifocalGemma**.

Public dan gold ZIP ketiga benchmark byte-identical terhadap v1. Pembaruan benchmark
ada pada tampilan/schema/protokol yang diperjelas; soal/jawaban asli tidak diubah untuk
meningkatkan skor. Runtime lokal Python3.13.16 bukan klaim parity skor official runtime.

## Audit duplikasi, split dan benchmark quarantine

- Nol duplicate instance_id, full row atau model input-target pair pada setiap Parquet.
- Nol event Git/seed atau structural edit pair berulang dalam cohort final.
- Nol input terlihat identik dengan target kode berbeda setelah quarantine.
- Nol train/dev intersections untuk repo, commit/seed, signature before, event,
  structural pair, visible input, input-target, dan struktur kode before/after.
- Split memakai26.200 connected components, seed17, target dev sekitar10%.
- Seluruh49.296 candidate rows diperiksa ulang terhadap kelima benchmark dengan
  exact/token-shingle overlap dan blocked repository families: nol temuan.
- Pemeriksaan tambahan45.303 candidate rows terhadap3.846 AST benchmark cores
  tanpa docstring: nol temuan. Final45.230 adalah subset tanpa perubahan source bytes.
- AST primary task code memakai seluruh body nonempty; snippet benchmark lain minimal32
  token. Tidak ada sertifikasi semantic/fragment/pretraining contamination.

Kelima artefak quarantine: HumanEvalFix164, CanItEdit105, RepoQA Python100,
CrossCodeEval Python2.665 dari471 repository slugs, dan RES-Q100. Quarantine payload
tidak berarti kelimanya evaluator produksi. CanItEdit/RES-Q/RepoQA/CrossCodeEval dan
corpus repository tetap memiliki pekerjaan scorer/harness/notices yang belum selesai.

## Ukuran, schema dan Molab

Total rilis: **206.399.063 byte ≈196,8MiB**, meliputi native ZIP, viewer dan bukti audit.
Training download normal hanya **pool.zip 61.950.538 byte ≈59,1MiB**, kemudian diekstrak.
Raw upstream Arrow2,6GB, source archives dan indeks curation tidak perlu diunduh Molab
untuk training normal.

| Parquet | Rows | Bytes |
| --- | ---: | ---: |
| Training train | 40.706 | 128.515.468 |
| Training dev | 4.524 | 14.021.758 |
| HumanEvalFix public | 164 | 115.552 |
| GSM8K zero-shot public | 1.319 | 483.283 |
| GSM8K eight-shot public | 1.319 | 613.387 |

Rilis lebih besar daripada v1 walaupun contoh lebih sedikit karena viewer sekarang
lossless dan menyediakan explicit before/after/record_json. Native ZIP dan Parquet
adalah dua representasi untuk tujuan berbeda; keduanya tidak dikonkatenasi untuk training.
Duplikasi representasi/kolom di dalam record bukan contoh training tambahan.

Viewer:16 kolom UTF-8 string untuk eval,17 untuk train. Kolom eksplisit instance_id,
role, source, instruction, context, provenance_json, question, shots, protocol_json,
task_scope, research_role, edit_kind, path, before, after, record_json; train menambah
target_patch. shots tampil sebagai string "0"/"8". record_json mempertahankan tepat
record native, termasuk struktur provenance/token_lengths/alias; protocol dan shots
benchmark berasal dari manifest koleksi.

Native training mencatat source/revision/upstream_row, original before/after/path,
patch, physical pair_hash/instruction_hash, normalization/structural hashes, split keys,
discarded-source aliases, lisensi dan kualitas statis. Old/new file-content aliases
dipertahankan untuk adapter legacy. Container format _v1 tetap compatible, dengan row
schema_version2 dan collection/viewer version2.

Setiap record Parquet dibandingkan dengan native ZIP, semua SHA inventory diverifikasi
pada download ulang27 file (26 hashed entries +inventory), lalu empat koleksi dipulihkan
melalui fetch_collection. Fetch eval public-only terbukti tidak mengunduh gold.
HF repo tetap private; main tepat lima Parquet v2; immutable v1 masih dapat diakses.

## Notebook dan batas hasil

Ketiga notebook lokal mengunci SHA baru. Dua notebook family tetap20 cell dengan19
body identik; perbedaan hanya tiga control defaults. Cache yang sudah ada tidak lagi
melewati validasi fingerprint. Default terbaru RUN_TRAINING=True dan HF_UPLOAD_ENABLED=True. Kurasi tidak tersedia
di notebook; run name v2 serta verifikasi fingerprint memisahkan cohort/checkpoint lama.

Audit berjalan memakai env ai: Python3.13.16, Transformers5.19.0, tokenizers0.23.3,
huggingface_hub2.2.0, PyArrow25.0.1. Validasi statis tidak mengeksekusi source training.
Gold HumanEvalFix memakai subprocess scorer native yang telah diotorisasi.
Training pretrained/Molab tidak dijalankan ulang pada pekerjaan ini.

Bukti lengkap: [release receipt](data/private_dataset_release_20261011_v2.json),
[specs](data/dataset_remediation_specs_20261011.json),
[viewer/asset audit](data/dataset_remediation_audit_20261011.json),
[HF round-trip](data/dataset_remote_verification_20261011.json),
[benchmark token profiles](data/dataset_benchmark_specs_20261011.json),
[compile exclusions](data/dataset_compile_audit_20261011.json), serta
[Git coverage](data/dataset_git_patch_audit_20261011.json).

Seluruh45.303 patch kandidat round-tripped dengan Git check/apply dan byte-exact after.
Final45.230 merupakan unchanged subset yang dibuktikan oleh identity hash; viewer audit
menggunakan flag reuse-validated-pool-audits sehingga tidak mengklaim menjalankan
round-trip baru dua kali. Profil tokenizer mencakup seluruh kandidat tersebut sebelum
73 compile exclusions, dengan semua retained token lengths tetap cocok.

Validasi ini memperbaiki integritas/kontrak dataset, bukan membuktikan seluruh training
edit benar secara fungsional atau keberhasilan direct SFT. Kualitas repair, generalisasi
repository, dominasi encoder/decoder dan peak memory tetap memerlukan pilot model nyata.

Regression akhir: **114 passed,0 failed,0 skipped** (180,67s), termasuk tokenizer riil. Generator --check, marimo check --strict pada ketiga notebook, Ruff, type checking dan git diff --check lulus. [Receipt pemeriksaan kode](data/dataset_code_validation_20261011.json).
