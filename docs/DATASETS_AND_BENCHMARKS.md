# Dataset dan Benchmark: Offline Code Editing / Repair SFT

Recipe aktif diperbarui **6 Oktober 2026**: SFT `instruksi + kode sebelum edit → patch`.
Data preparation dan tuning berjalan dari file lokal, tanpa RL, reward model,
agent rollouts, VM, atau Docker sebagai prasyarat. Environment `ai` tetap dipakai;
kebutuhan GPU pretrained ditentukan oleh profiling, terpisah dari kebutuhan VM.

## Dataset untuk tahap awal

| Sumber | Peran aktif | Field / batas |
| --- | --- | --- |
| [EditPackFT](https://huggingface.co/datasets/nuprl/EditPackFT) | Primary pilot instruction editing Python | `instruction`, `old_contents`, `new_contents`, file paths, commit/repo/license tersedia; sekitar 22.602 records dan 48,1 MB Parquet pada pemeriksaan metadata 6 Oktober. Tidak semua edit adalah verified bug fix. |
| [CommitPackFT Python](https://huggingface.co/datasets/bigcode/commitpackft) | Tambahan opsional setelah dedup/quality audit | `subject`, `old_contents`, `new_contents`, paths, commit/repo/license; 56.025 records, sekitar 136 MB JSONL. Commit message tidak otomatis merupakan issue yang baik atau bug fix yang teruji. |
| Static multi-file records lokal | Tahap lanjutan untuk repository understanding | Semua file sebelum edit diberikan sebagai snapshot; instruksi + target patch. Records dengan full snapshot dapat dipakai tanpa menjalankan environment repository. Corpus riil belum tersedia. |

Mulai dengan EditPackFT saja. CommitPackFT bukan campuran otomatis: keduanya memakai
sumber commit GitHub yang dapat overlap. Audit commit IDs, before/after pairs dan
near-duplicates sebelum menambah data. Angka rows/bytes adalah upstream metadata,
bukan ukuran train split setelah filtering; revision dipin saat data diambil.

Adapter [EditPackFTAdapter / CommitPackFTAdapter](../src/bifocalgemma/data/adapters.py)
masih slot D04, belum downloader/converter production. Input hanya old_contents
serta instruction/subject; target unified diff diturunkan dari before → after.
Jangan memakai field `content` siap pakai atau `ndiff` sebagai input karena dapat
memuat kode setelah edit/jawaban. Untuk pilot, gunakan modification pada path sama;
rename/add/delete memerlukan kontrak adapter tambahan dan tidak dipaksakan menjadi
contoh modification biasa.

Cross entropy hanya pada patch dan closing token. Patch application/rekonstruksi
before→after dapat diverifikasi pada data preparation tanpa mengeksekusi kode
repository. Syntax check adalah diagnostik; keberhasilan apply/syntax bukan bukti
perbaikan fungsional. Tidak ada reward dari test execution pada training loop.

## Sasaran tuning dan batas klaim

- Mengikuti instruksi edit, menjaga format unified diff, dan mempertahankan bagian
  kode yang tidak perlu diubah.
- Memanfaatkan context encoder dan membandingkan empat pasangan dengan E2B-IT/E4B-IT.
- Awal studi memakai single-file editing; klaim lintas file memerlukan static
  multi-file records dan matched held-out evaluation tersendiri.
- EditPackFT/CommitPackFT tidak boleh dilabeli corpus verified repair seluruhnya.
  Laporkan bug-fix subset dan general editing secara terpisah bila dikurasi.

## Evaluasi yang sesuai dengan tahap ini

| Evaluasi | Penggunaan | Kebutuhan |
| --- | --- | --- |
| Held-out offline editing | Primary validation awal: patch apply/reconstruction, exact edited-code match, format/syntax, held-out CE | File lokal; tidak mengeksekusi repository atau memakai VM |
| [HumanEvalFix Python / HumanEvalPack](https://github.com/bigcode-project/octopack) | Functional unit-level repair setelah generation/scorer siap | Inference dan test execution terpisah. Python native scorer dapat digunakan bila kompatibel; runner masih perlu diintegrasikan, tidak menjalankan tes dalam training |
| [CanItEdit](https://github.com/nuprl/CanItEdit) | Instruction-editing benchmark tambahan, 105 program Python | Official setup/scorer dipin dan dicek sebelum dipakai; test execution tahap evaluasi terpisah, bukan prasyarat SFT |
| [CrossCodeEval](https://github.com/amazon-science/cceval) | Diagnostik context lintas file melalui completion EM/similarity | Pinned preprocessed context/scorer; tidak disebut repair resolved rate |

Held-out CE dan exact match membantu diagnosis, tetapi bukan pengganti functional
pass rate. Jika functional benchmark belum dijalankan, laporkan hasil sebagai
editing diagnostics saja. Exact match dapat menolak patch alternatif yang benar.
[BigCode harness](https://github.com/bigcode-project/bigcode-evaluation-harness)
memisahkan generation dan CPU evaluation; Docker tersedia sebagai opsi eksekusi.
Kompatibilitas scorer Python di Windows/Conda tetap perlu diperiksa, tidak dijamin
hanya karena data berupa Python.

## Sumber yang ditunda dari jalur utama

[SWE-smith Python](https://huggingface.co/datasets/SWE-bench/SWE-smith-py),
[SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym),
[SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories),
serta SWE-bench Lite/Verified tetap referensi/opsi lanjutan. Mereka tidak menjadi
prasyarat pilot yang dipilih. SWE-smith dapat dipakai untuk SFT; tidak otomatis
memerlukan RL. Penyiapan executable task environments dan benchmark Docker adalah
bagian yang tidak sesuai dengan preferensi pipeline awal tanpa VM.

D06 [kontrak arah patch SWE-smith](NATIVE_LOCAL_VALIDATION.md#d06--kontrak-lokal-swe-smith)
tetap utility lokal yang sudah diuji. Itu tidak mengaktifkan environment task.
A01 agent loop dan E01 SWE-bench Docker harness menjadi optional placeholders;
CLI evaluate lama tetap fail-fast pada E01. Tidak ada backend evaluator offline
production yang diam-diam disimpulkan tersedia dari perubahan recipe ini.

## Filtering dan reproducibility

1. Catat source revision, commit, repo set, file paths, license, before/after hashes,
   target patch hash, token lengths, split dan filter reasons.
2. Dedup lintas sumber; grouped train/dev/test berdasarkan repo/commit, termasuk
   seluruh aliases repo dalam field `repos`. Jangan pisahkan record satu commit
   ke split berbeda. Audit benchmark overlap sebelum evaluation final.
3. Pertahankan old code sebagai input dan after/patch sebagai target. Tolak records
   kosong/tidak berubah, paths tidak didukung, atau patch yang tidak merekonstruksi after.
4. Gunakan context selection/token limits yang sama secara metodologis pada enam
   model. Overflow saat ini ditolak oleh D03; tidak ada gold-based selection.
5. Simpan attribution/license; dataset card tidak menggantikan lisensi kode asal.
   Filtering adaptasi sendiri tidak menghapus kemungkinan pretraining contamination.

## Urutan pengerjaan

Native tiny M07 dan offline tokenizer D03 → D04 paired-file adapter/manifest →
M04/M05 pretrained transfer dan T03 profiling → direct patch SFT / S01 checkpoint →
I01 patch generation → held-out editing diagnostics → functional Python evaluation →
static multi-file study. Tidak ada tahap RL atau VM pada jalur tuning.

Direct SFT tetap hipotesis: [T5Gemma 2](https://arxiv.org/html/2512.14856v2) memakai
adaptasi UL2 sebelumnya. Data yang lebih sederhana mengurangi kebutuhan environment,
tidak otomatis menjawab apakah adaptasi arsitektur berhasil. PLE + tied words BF16
sendiri 9,5/11,75 GiB pada pasangan simetris; profiling GPU tetap diperlukan.
