# BifocalGemma: Dataset Training dan Benchmark Code Repair

Riset sumber primer diperiksa **5 Oktober 2026**. Dokumen ini adalah rekomendasi
eksperimen; belum ada dataset besar diunduh, model dilatih, atau skor repair dihasilkan.
Pin revision dataset dan catat jumlah baris setelah filtering; card dan viewer dapat berbeda.

## Pilihan training

| Sumber | Peran yang disarankan | Batas dan kebutuhan |
| --- | --- | --- |
| [SWE-smith tasks](https://huggingface.co/datasets/SWE-bench/SWE-smith) dan [toolkit](https://github.com/SWE-bench/SWE-smith) | Sumber utama pasangan issue/snapshot → patch dan label lokasi file | Task sintetis dengan executable environment; perlu rekonstruksi snapshot dan verifikasi patch. Card menyebut 50.137 task/128 repo, sedangkan viewer terkini lebih besar; jangan menganggap angka historis sebagai jumlah split yang dipakai. |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | SFT aksi/tool-use setelah patch-only training bekerja | Ada messages, instance_id, resolved, model, dan patch. Pilih trajectory resolved, cocokkan schema split dan task; task dataset tidak otomatis merupakan trajectory agen. |
| [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) / [OctoPack](https://github.com/bigcode-project/octopack) | Warm-up instruction editing dari subset Python terkurasi | Card mencatat 702.062 contoh total dan 56.025 Python. Commit bukan selalu bug fix, tidak selalu multi-file, dan tidak otomatis disertai reproduksi test; metadata commit-message bukan jaminan issue berkualitas. |
| [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) / [toolkit](https://github.com/SWE-Gym/SWE-Gym) | Sumber alternatif real GitHub issue untuk tahap repair | Card menampilkan sekitar 2,44k task dengan repo, base_commit, patch, test_patch, dan kelompok tes. Audit overlap dengan evaluation terlebih dahulu; lingkungan eksekusi tetap perlu disiapkan. |
| [EditPackFT](https://huggingface.co/datasets/nuprl/EditPackFT) | Opsional untuk instruction editing | Dataset training berbeda dari benchmark CanItEdit. Tidak diperlakukan sebagai bukti kemampuan agent multi-turn lintas repositori. |

**Rekomendasi awal:** patch-only SFT pada subset SWE-smith Python yang terverifikasi,
dengan CommitPackFT Python sebagai warm-up opsional. Tambahkan trajectories setelah
format tool-call dan agent loop benar-benar terintegrasi. SWE-Gym menjadi pembanding
sumber real issue, bukan tambahan otomatis sebelum audit overlap dan compute.

Input training berasal dari snapshot sebelum perbaikan ditambah issue. Patch merupakan
target, bukan bagian konteks encoder. Label file dapat diambil dari diff hanya untuk
supervisi lokalisasi. Data edit satu file tidak membuktikan kompetensi lintas file.

## Benchmark dan apa yang diukurnya

| Benchmark | Tugas dan metrik | Posisi dalam eksperimen |
| --- | --- | --- |
| [SWE-bench Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) | Test split 300 issue–PR pairs; resolved rate lewat test suite | Iterasi awal end-to-end; subset yang diumumkan eksplisit untuk debugging |
| [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | 500 test samples yang divalidasi manusia; resolved rate | Evaluasi akhir repair; tidak digunakan untuk tuning atau training |
| [CrossCodeEval](https://github.com/amazon-science/cceval) | Completion lintas file; exact match dan metrik similarity dari scorer resmi | Diagnostik pemanfaatan dependensi, bukan resolved rate atau bukti bug repair |
| [HumanEvalFix / HumanEvalPack](https://github.com/bigcode-project/octopack) | Bug repair level fungsi dalam enam bahasa; pass@k dari tes | Pemeriksaan tambahan repair skala kecil, mulai Python |
| [CanItEdit](https://github.com/nuprl/CanItEdit) | 105 program Python dengan before/after, dua jenis instruksi, dan hidden tests | Benchmark instruction editing; bukan dataset training multi-turn |
| [RepoBench](https://github.com/Leolty/RepoBench) | Retrieval, completion, dan pipeline repository-level | Opsional untuk ablation retrieval; bukan issue-resolution benchmark |
| [Defects4J](https://github.com/rjust/defects4j) | Bug Java riil dengan lingkungan dan regression tests | Tahap lanjutan lintas bahasa; tidak dicampur dengan benchmark Python |

Card Lite menyebut **11 repo Python** dan 300 baris pada split test; ukuran agregat
atau tabel dokumentasi yang berbeda tidak dijadikan denominator test. Validasi manusia
pada Verified memperbaiki kualitas benchmark, bukan jaminan bahwa setiap model mampu
menyelesaikannya. HumanEval/MBPP standar mengukur synthesis dan bersifat tambahan.

Pelaporan kode mempertahankan key `crosscodeeval` sebagai **exact-match percentage**
untuk kompatibilitas; edit similarity dicatat terpisah dalam laporan eksperimen.
Jangan menggabungkan completion score dengan SWE-bench resolved rate.

## Filtering dan pemisahan data

1. Pin revision data; simpan manifest instance_id, repo, base_commit, sumber, split,
   jumlah file yang berubah, panjang token, serta alasan filtering.
2. Buang task ID, issue/PR, commit, patch identik dan near-duplicate yang overlap
   dengan Lite/Verified evaluation, termasuk dari trajectory dan commit datasets.
   Audit repository overlap; repo-disjoint holdout dipakai untuk klaim generalisasi
   ke repo baru dan jumlah sisa data harus dilaporkan.
3. Rekonstruksi snapshot sebelum edit. Filter patch yang tidak berlaku atau gagal
   verifikasi tes. Pisahkan tugas satu file dan lintas file untuk analisis.
4. Bagi train/dev secara berkelompok menurut repo atau task; trajectory dari task yang
   sama tidak boleh tersebar antar split. Jangan memakai gold patch, test_patch,
   label FAIL_TO_PASS/PASS_TO_PASS, atau hint yang membocorkan solusi sebagai input model.
5. Saat truncation diperlukan, catat strategi context selection yang sama untuk baseline
   dan BifocalGemma; subset 16k-token dilaporkan sebagai subset, bukan seluruh benchmark.
6. Simpan attribution dan license per sumber serta lisensi repo asal. Lisensi kode
   BifocalGemma tidak otomatis mencakup semua data atau kode pihak ketiga.

Filter overlap mengurangi kontaminasi dari adaptasi sendiri; pretraining contamination
pada checkpoint publik tidak dapat dihapus hanya dengan filter ini.

## Eksekusi pada hardware terbatas

[Harness SWE-bench](https://www.swebench.com/SWE-bench/guides/evaluation/) menerapkan
patch lalu mengeksekusi tes dalam Docker. Evaluasi patch tidak membutuhkan bobot
Gemma atau GPU; beban utamanya environment, CPU, RAM dan disk. Inference model dan
evaluasi patch dapat dijalankan terpisah. Toolkit SWE-smith didokumentasikan untuk
Linux/Docker, sehingga pembuatan environment tidak diasumsikan berjalan native Windows.

Mulai dengan 10–20 task Lite dalam manifest tetap dan satu worker pada Linux/WSL2
dengan Docker yang telah disiapkan. Itu smoke subset untuk iterasi, bukan skor Lite
300-task penuh. Setelah pipeline stabil, jalankan seluruh split Lite dan gunakan
Verified untuk hasil akhir; laporkan overlap antar kedua benchmark.

Catat %resolved, patch-apply failures, waktu inference/tes, timeouts dan infrastructure
failures secara terpisah. Gunakan run_id baru untuk patch baru agar cache harness
tidak menyamarkan pengujian ulang. Gold test artifacts tersedia hanya untuk evaluator.
Baseline memakai issue/context, retrieval, training budget, tool budget, serta caching
yang sebanding; simpan config/manifest dan rerun semua metode dengan harness yang sama.

## Urutan eksperimen yang dipilih

1. Uji modul dan topology Gemma tanpa bobot pretrained.
2. Bentuk manifest subset training SWE-smith terverifikasi dan subset Lite untuk smoke.
3. Integrasikan pilot E2B–E2B lalu empat pasangan base encoder / IT decoder; pilih training budget setelah profiling.
4. Adaptasi patch-only, bandingkan localization/repair dan ablate bidirectionality/AST.
5. Bandingkan empat pasangan dan baseline E2B-IT/E4B-IT dengan harness Lite; lanjut Verified setelah tuning dikunci.
6. Tambahkan trajectory SFT dan completion diagnostics bila diperlukan oleh pertanyaan riset.

Ini menetapkan sumber dan protokol awal, tanpa menjanjikan keberhasilan training
atau menetapkan ukuran model yang belum diprofilkan.

Pembanding benchmark utama mengikuti [desain enam model](EXPERIMENT_DESIGN.md).
T5Gemma dan T5Gemma 2 dibahas dalam related work tanpa menambah kandidat benchmark.
