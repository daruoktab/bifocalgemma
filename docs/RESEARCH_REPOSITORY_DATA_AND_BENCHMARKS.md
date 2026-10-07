# Riset Menyeluruh: Repository Data, Benchmark, dan Scope BifocalGemma

Diperiksa **7 Oktober 2026**. Target: empirical paper pada Gemma 4 Developer Agent Paper Track, satu GPU Molab, SFT tanpa RL/VM. Batas input/output mengikuti data; 32k/8k bukan target tetap.

**Kesimpulan:** studi akhir dapat diperkuat dengan issue-to-patch Git snapshots, **RES-Q Python sebagai kandidat main repository-editing evaluation**, RepoQA/CrossCodeEval untuk reader evidence, dan HumanEvalFix sebagai kontrol repair kecil. Admission source dan native scorer tetap memerlukan provenance, compatibility dan gold validation.

Ini perluasan [paket pilot Molab](MOLAB_DATA_EVALUATION_PLAN.md), bukan penghapusan temuan profiling EditPackFT. [Audit laporan lama](RESEARCH_SFT_DATASET_AUDIT.md) tetap berlaku. Tidak ada training GPU, model benchmark, clone semua repo, atau upstream test execution dalam riset ini.

## 1. Cakupan dan batas bukti

Pencarian mencakup official competition, instruction editing, issue/PR repair, synthetic repo repair, edit-history prediction, repo completion, understanding dan functional evaluation. Katalog luas ini bukan klaim seluruh dataset dunia sudah ditemukan.

| Bukti | Yang diperiksa | Batas |
|---|---|---|
| Sumber primer | Official README/card/paper/scorer source | Klaim publisher tetap harus diuji. |
| Metadata | Revision, file bytes, fields dan splits | Bukan unique/admitted examples setelah filtering. |
| Sampel | Viewer rows dan tokenization | Nonrandom rows tidak menentukan yield populasi. |
| Artefak kecil penuh | RES-Q100 metadata, RepoQA release; EditPackFT22.602 dan benchmark kecil pada riset sebelumnya | Bukan hasil model. |
| Static patch check | Gold diff terhadap virtual before snapshots | Applicability saja, bukan functional success. |
| Execution | **Belum dijalankan pada benchmark repos dalam riset ini** | Molab native/gold acceptance masih pekerjaan berikutnya. |

Revision dan angka: [repository_candidate_audit_20261007.json](data/repository_candidate_audit_20261007.json). Profil pilot: [molab_data_profile_20261007.json](data/molab_data_profile_20261007.json).

## 2. Ketentuan dan kecocokan kompetisi

[Submission requirements resmi](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview/submission-requirements) mengonfirmasi **3.000 kata**, original/unpublished research dan non-archival submission. Writeup berisi title/subtitle, abstract, introduction, methods/experiments, related works/citations. Public notebook dan PDF/project link opsional.

Paper Track tidak mewajibkan leaderboard participation. Rubrik novelty, quality/generalization, relevance, verifiability dan clarity menuntut bukti yang sesuai klaim. Graph/embeddings dianjurkan pada overview main competition, bukan paper prerequisite. Tidak ditemukan minimum jumlah dataset/tasks pada overview/requirements yang dibaca.

[Prediction track](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview) sekarang membatasi model ke `gemma-4-31b-it-qat-w4a16-ct`. Matriks E2B/E4B custom encoder–decoder diarahkan ke Paper Track. Rules Paper Track belum seluruhnya dapat dibaca; jangan memindahkan main-track winner-license/team/private-sharing clauses otomatis. [Panduan kompetisi](COMPETITION_GUIDE.md) mencatat batas verifikasi.

## 3. Katalog dataset training

Decimal MB/GB di bawah adalah data files, bukan Git clones, expanded cache, environments atau GPU memory.

| Sumber | Tujuan dan resource yang diperiksa | Keputusan |
|---|---|---|
| [EditPackFT](https://huggingface.co/datasets/nuprl/EditPackFT) | Instruction before/after Python;22.602 raw,48,12MB; full profile tersedia | Pilot/fondasi. Belum cukup sendiri untuk repository repair. |
| [CommitPackFT Python](https://huggingface.co/datasets/bigcode/commitpackft) | Subject/file edit;56.025 upstream,135,86MB JSONL | Tambahan setelah dedup dengan EditPackFT. |
| [NextCoder](https://huggingface.co/datasets/microsoft/NextCoderDataset) | Synthetic edits/instruction variants;2,627GB Arrow all languages; card 15.279 Python seeds | Pelengkap; full-code-to-diff conversion. Bukan repo snapshots. |
| [InstructCoder](https://huggingface.co/datasets/likaixin/InstructCoder) | Function edit triples;114.099 viewer rows,146,96MB JSON | Bersyarat kualitas/license; bukan core repo corpus. |
| [GitHub Code Review](https://huggingface.co/datasets/ronantakizawa/github-codereview) | Review/window pairs;355.807 all-language rows,652,89MB | Bersyarat provenance/offset/source license; window bukan full file. |
| **[SWE-Fixer Train](https://huggingface.co/datasets/internlm/SWE-Fixer-Train-110K)** | **115.406 train rows**, real issue/patch/base commit;1,471GB JSONL; converted Parquet534,06MB | **Shortlist core repo SFT.** Before snapshots belum ada dalam table. |
| **[SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2)** | **32.079 multilingual train rows**, actual Parquet428,84MB; expanded2,566GB | **Shortlist alternatif core repo SFT.** Python yield belum dihitung; static snapshots bisa tanpa gym/Docker. |
| [SWE-rebench V1](https://huggingface.co/datasets/nebius/SWE-rebench) | test 21.336 + filtered 6.542 rows;245,17MB files | Jangan jumlahkan sebagai independent cases tanpa ID intersection. Bekukan peran split. |
| [SWE-rebench V2 PRs](https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs) | Tiga Parquet shards2,682GB; PR/task annotations | Lebih luas; perlu roundtrip/base-commit/lineage audit. Jangan campur otomatis dengan V2. |
| [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) |2.438 train tasks upstream,43,64MB table | Bisa static SFT; repo scope terbatas dan overlap penting. Gym execution ditunda. |
| [SWE-smith Python](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) |50.908 upstream tasks,220,65MB Parquet | Candidate scale; patch adalah **bug insertion**, repair reverse dari buggy snapshot. Missing issue text harus ditangani. |
| [R2E-Gym Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) |943,93MB tables; [official workflows](https://github.com/R2E-Gym/R2E-Gym) memakai environments | Static extraction optional; container/trajectory collection bukan default. |
| [SWE-bench train](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | Train file106,61MB | Optional; quarantine test/dev dan benchmark repository families. |
| [Clean-PR](https://arxiv.org/html/2602.07457v1) | Paper melaporkan2M PRs/12 languages, verified Search/Replace | Prior art/pipeline inspiration. Exact official download/revision belum dicocokkan. |
| [Coeditor/PyCommits](https://github.com/MrVPlusOne/Coeditor) | Edit history/static repo context;paper 1.650 Python projects | Objective berbeda: subsequent edits. Future edits tidak boleh menjadi pre-fix context task kita. |
| [RepoExec-Instruct](https://huggingface.co/datasets/Fsoft-AIC/RepoExec-Instruct) | Actual files756,59MB; metadata155.938 rows, tetapi shard metadata tidak konsisten | Auxiliary generation candidate; belum diff repair atau fully audited training source. |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | Multi-turn feedback; [HumanEval leakage risk](https://aclanthology.org/2024.emnlp-main.777.pdf) | Ditunda: extraction/objective/contamination work tambahan. |
| [SWE-RL](https://github.com/facebookresearch/swe-rl) | Related method; code majority CC-BY-NC4.0; ready data release belum diperiksa | Related work; label RL tidak memaksa patch data memakai RL, tetapi akses/license tidak diasumsikan. |

PatchPilot, CyberNative DPO dan agent trajectories tetap deferred/excluded sebagaimana [paket pilot](MOLAB_DATA_EVALUATION_PLAN.md). Prioritaskan **satu core repository source + satu fondasi instructional editing**; tambah sumber berdasarkan ablation, bukan total rows.

### SWE-Fixer: shortlist core

Schema langsung: repo, pull_number, instance_id, issue_numbers, base_commit, patch, test_patch, problem_statement, hints_text, created_at. Tidak ada full before/after code siap pakai. Fetch repo@base_commit, construct allowed before context, validate patch application, kemudian train statis.

[Paper SWE-Fixer](https://arxiv.org/html/2501.05040v1) mempunyai retrieve/editor pipeline dan teacher rationalization. Kita memakai **raw issue/patch labels** tanpa wajib menjalankan teacher 72B, RL, atau reproduce rationalizations. Corpus raw tidak otomatis menjadi verified functional repair.

25 initial complete viewer rows: raw patch P50=502,max7.544 Gemma tokens, tanpa chat prefix. Bukan population yield; sample lintas repo/period dan full selected-snapshot profiling masih diperlukan.

### SWE-rebench V2: shortlist alternatif

Schema lebih eksplisit: language/license, base_commit, problem_statement, patch/test_patch, install_config dan test outcome lists. Snapshot tetap perlu diambil; static SFT berbeda dari official Docker scoring.

Actual API/file bytes428,84MB berbeda dari download_size2,263GB pada sebagian card rendering. Gunakan pinned files. Input default tidak memakai pr_description, generated interface, hints/LLM suggested fixes, test_patch atau hidden outcomes yang bisa diketahui sesudah solusi.

6 complete rows dari10 initial viewer rows: raw patch P50=1.121,max9.708 tokens;4 truncated rows tidak diprofilkan. Bukan Python population profile. Output8k juga bukan jaminan seluruh source muat.

## 4. Katalog benchmark

| Benchmark | Tugas/resource | Status native dan peran |
|---|---|---|
| **[RES-Q](https://github.com/Qurrent-AI/RES-Q)** |100 repo-editing tasks, output unified diff;raw JSON 4,653 MB | **Kandidat main evaluation.** Scorer Conda native; clone/dependency/gold checks belum dijalankan. General editing, bukan semua bugs. |
| **[RepoQA](https://github.com/evalplus/repoqa)** | Searching Needle Function, NL description → function;release12,526MB gzip | **Main reader diagnostic.** Parser/similarity scoring tanpa repo execution. Python100 tasks. |
| **[CrossCodeEval](https://github.com/amazon-science/cceval)** | Python2.665 completions/471 repos;archive41,997MB all languages | Supporting context diagnostic; preprocessed retrieval tanpa reference-assisted context. EM/edit similarity tidak mengeksekusi repo. |
| [RepoBench v1.1](https://github.com/Leolty/repobench) | Retrieval/completion/pipeline;Python Parquet473,0MB | Supporting alternative; license metadata`cc` belum cukup specificity. Bukan repair. |
| [HumanEvalFix-docs](https://github.com/bigcode-project/octopack) |164 Python repair tasks,191KB;stdlib imports diperiksa | Small-repair sanity/retention control, bukan main repo evidence. |
| [CanItEdit](https://github.com/nuprl/CanItEdit) |105 programs;descriptive/lazy;250KB | Official container/old dependencies. Native83-task import-screened cohort belum execution-certified. |
| [RepoExec](https://github.com/FSoft-AI4Code/RepoExec) |355 tasks ×3 context variants,12,47MB tables | Official runner Docker. Bukan1.065 independent tasks. Native port membutuhkan gold/metric parity. |
| [DevEval](https://github.com/seketeam/DevEval) |1.825 samples/115 repos;source gzip916,75MB + dependency708,16MB | Official Conda setup. Native candidate, tetapi broad deps; generation, bukan repair. |
| [BugsInPy](https://github.com/soarsmu/BugsInPy) | Paper awal493 real bugs/17 projects | Historical runtimes/dependencies; native possibility, bukan guaranteed reproducible. |
| [BugsInPy++](https://github.com/UNIST-LOFT/BugsInPyPP) |501 bug.info entries/17 projects pada tree diperiksa | **Backup native repair candidate**;501 metadata tidak berarti501 runnable bugs. |
| [RunBugRun](https://github.com/giganticode/run_bug_run) | Current README>700k buggy/fixed short-program pairs/9 languages | SQLite/lrzip/Linux Bubblewrap, bukan VM. Setup perlu audit; bukan repo context. Deferred. |
| [CodeEditorBench](https://github.com/CodeEditorBench/CodeEditorBench) | Debug/translate/polish/switch competitive-programming tasks | HUSTOJ evaluator mempunyaiGPL2 portion; task/runtime mismatch untuk core study. |
| [Aider Polyglot](https://aider.chat/docs/leaderboards/) |225 exercises/6 languages, edit/feedback protocol | Exercise-solving/attempt budget berbeda; native subset tidak disebut full leaderboard score. |
| [BigCodeBench](https://github.com/bigcode-project/bigcodebench) | Library/API-heavy function generation | Optional retention check; dependency closure besar, bukan repository repair. |
| [SWE-bench Lite/Verified](https://github.com/SWE-bench/SWE-bench) |300/500 real issue tasks + regression tests | Strong reference, official Docker. Static patch metrics bukan resolved rate. |
| [R2E-Gym/rebench/smith environments](https://github.com/R2E-Gym/R2E-Gym) | Executable repo task environments | Table bytes kecil tidak mencakup images/install costs. Official execution ditunda. |

### RES-Q: metadata dan static checks penuh

Dataset revision `c0e850b1c90d1c0eb7d3b2f3c9aa6ec09c2c04fa`; raw JSON SHA-256 `43bc9f0644e6fa0347ac58f9d9eb3f7b7235255ca7be4dd27c62d55ecbf342d2`.

- **100 tasks /14 fork URLs: 95 Python, 5 JavaScript.**
- 48 tasks mempunyai>1 gold-associated modified file;14 mempunyai add/delete markers. Multi-file edit bukan otomatis dependency-required reasoning.
- Testbed97 `python3.9`,3 `python3.10`;16 distinct version/requirements signatures. Environment reuse perlu install/state identity checks.
- [Scorer task source](https://github.com/Qurrent-AI/RES-Q/blob/main/resq/task.py) mereset base_commit, apply patch, lalu menjalankan test_script lewat [Conda child process](https://github.com/Qurrent-AI/RES-Q/blob/main/resq/testbeds/base.py), tanpa Docker. Conda availability di Molab belum diperiksa.
- `modified_files` adalah gold-associated metadata. Contoh pertama content-nya before-code, tetapi file identity memberi oracle localization; bukan main input selector.

Audited Gemma tokenizer/template, seluruh95 Python tasks:

| Field |P50|P95|P99|Max|
|---|---:|---:|---:|---:|
|Gold patch+closing turn|655|3.251|4.809|6.535|
|Teacher-forced decoder total, after shift|750|3.371|4.919|6.652|
|Gold-associated file context, **oracle diagnostic only**|4.348|42.010|52.508|65.247|

Context terakhir bukan full repo dan bukan deployable context policy; missing dependencies dapat membuat actual context lebih besar. Ia menunjukkan pilot1k/1,5k tidak mewakili benchmark akhir.

**Static gold apply:**99/100 patches lolos `git apply --check` pada virtual before snapshots dari modified_files. Task3_2 gagal karena missing paths pada snapshot virtual; belum bukti task invalid pada repo asli. Tidak ada native functional test atau source-code execution dalam check ini.

Dataset card tidak mendeklarasikan license; code repoMIT tidak otomatis menyelesaikan dataset/source redistribution rights. Official runner normal-clone per-task dan native env per-task bisa melipatgandakan disk. Perlu real-checkout/gold validation, sequential scoring, safe cache dan requirements preflight.

### RepoQA: actual release berbeda dari paper lama

Release2024-06-23:12.526.636 bytes gzip,71.510.991 bytes JSON; SHA-256 `c050a2ad90a7df89d9dc1f1c3b3b20683edd20a56293b35fcaae43dec115d681`.

Paper/README lama500 tasks/5 languages; **release yang diperiksa600 needles/60 repos/6 languages**, termasukGo. Python tetap100 needles/10 repos. Pin release dan jangan menyebut current file persis original paper set.

Stored content/dependency/functions/commit SHAs memungkinkan text-only scoring. Requested function memang ada dalam input: legitimate retrieval/understanding task, bukan patch-answer leakage. Gold needle path/name tidak diberi sebagai extra hint. Scorer mencari syntactically closest function dengan threshold default 0,8.

Full stored Python source47.488–994.787 Gemma tokens, **bukan actual task context**. [Upstream task builder](https://github.com/evalplus/repoqa/blob/main/repoqa/search_needle_function.py) memilih token windows dan needle position. Pertahankan policy/task identities;4k/8k study dilabeli bila berbeda dari published setting. Function-output mode berbeda dari unified-diff mode.

### BugsInPy++: native possibility dengan gate ketat

Tree `bbb7d38a99446af8471ae205db4aecb4a1e2bdb4`:501 bug.info entries/17 projects.169 dari pandas;PySnooper 3,cookiecutter 4,tqdm 9,httpie 5. Ini metadata count, bukan reproduced bugs.

PySnooper bug 1 menyebut Python 3.8.1, buggy/fixed commits, dependencies`py`/`python_toolbox`, targeted pytest. [Current compile script](https://github.com/UNIST-LOFT/BugsInPyPP/blob/main/framework/bin/bugsinpy-compile) memakai Conda; sebagian create-env code commented meskipun README menyatakan automatic creation. Jangan menjalankan opaque helper atau menganggap Python 3.13 sesuai semua bugs.

Issue NL tidak otomatis tersedia. Task dari public failing-test report perlu disebut diagnostic-driven repair; sanitized logs tidak memuat gold. Pilih cohort berdasarkan compatibility sebelum model outcomes, reproduce buggy-fail/fixed-pass, dan laporkan exclusions. Native subset tidak disebut full 493/501 benchmark.

## 5. Prior art dan novelty

| Sumber primer | Yang sudah diteliti | Implikasi |
|---|---|---|
|[SequenceR](https://arxiv.org/abs/1901.01808)|Seq2seq program repair|Kategori encoder–decoder repair sudah ada.|
|[CodeT5](https://arxiv.org/abs/2109.00859)|Unified encoder–decoder coding|Model kecil encoder–decoder untuk kode memerlukan novelty lebih spesifik.|
|[CodeT5+](https://arxiv.org/abs/2305.07922)|Modular/scalable coding components|Component adaptation mempunyai prior art.|
|[Coeditor](https://arxiv.org/html/2305.18584v2)|CodeT5-base 220M, repo/edit-history context, block-sparse encoder|Closest repo-editing prior art; context scaling juga bukan baru.|
|[T5Gemma](https://arxiv.org/html/2504.06225v1)|Decoder-to-encoder–decoder adaptation, asymmetric sizes|Capacity allocation umum sudah diteliti; fokuskan pada native Gemma 4/code/directSFT.|
|[T5Gemma2](https://arxiv.org/html/2512.14856v2)|Merged attention/tied words;adaptation pretraining~2T tokens|Belum bukti direct patch SFT kecil dengan base-reader/IT-generator cukup.|
|[Gemma Encoder](https://arxiv.org/abs/2503.02656)|Decoder-to-encoder non-generative adaptation|Bidirectional attention sendiri bukan novelty otomatis.|
|[RepoCoder](https://arxiv.org/abs/2303.12570)|Retrieval+generation untuk repository context|Decoder-only dapat memanfaatkan dependencies; samakan visible context.|
|[SWE-Fixer](https://arxiv.org/html/2501.05040v1)|Retrieve-then-edit issue resolution|Editor pipeline tidak wajib complex agent loop.|
|[Agentless](https://github.com/OpenAutoCoder/Agentless)|Structured localization/edit/validation|RL atau autonomous loop bukan syarat kontribusi SWE.|
|[Clean-PR](https://arxiv.org/html/2602.07457v1)|PR edits, roundtrip verification, simplified SFT|Data/context controls penting untuk claim repository editing.|

Related work tersebut **tidak ditambahkan ke comparator utama**. Empirical matrix tetap empat pairs dan dua Gemma 4 IT baselines.

Novelty yang diuji: direct patch SFT untuk native Gemma 4 base-reader/IT-generator dengan joint merged attention; capacity allocation pada context-dependent edits; quality/VRAM/latency setelah controls. Tidak ada klaim first paper/pasti unggul.

Symmetric ordinary words shared ber-anchorIT;PLE separate. Full 2×2 mencampur capacity dan diagonal-only tying. Asymmetric duel lebih bersih; untied symmetric controls diperlukan untuk strong causal attribution. E2B–E2B physical parameters lebih banyak daripada E4B-IT tunggal; menang tidak otomatis berarti model keseluruhan lebih kecil.

## 6. Rekomendasi studi akhir

### Corpus

1. EditPackFT untuk instruction/patch adaptation, memakai full eligible corpus atau subset yang dibekukan sebelum benchmark.
2. **Pilih satu core source dulu:** SWE-Fixer raw issues atau SWE-rebench V2 Python, reconstruct static before snapshots dan audit source aliases/benchmark overlap.
3. Size ladder:pilot 2k →5k/10k unique repo examples →hingga 20k jika convergence/compute mendukung. Ini target eksperimen, bukan admitted count yang sudah diperoleh.
4. NextCoder/Coeditor optional auxiliary ablation. Tidak ada arbitrary 100k pool atau teacher generation wajib.

### Main evaluation

- **RES-Q Python 95:** goal full eligible cohort setelah native gold/scorer gate. Full100 hanya bila JavaScript handled dan dinyatakan multilingual. General editing dipisahkan dari corrective repair.
- **RepoQA Python 100** untuk reader evidence; **CrossCodeEval Python** fixed cohort/full2.665 setelah context/output backend siap.
- **HumanEvalFix-docs 164** untuk small-repair sanity dan retention.
- BugsInPy compatible cohort optional untuk real-bug repair tambahan, bukan dipaksakan full501.

RES-Q 95 + RepoQA 100 + HumanEvalFix 164 =**359 generations/model**, **2.154 untuk 6 models**, sebelum ablations. CrossCodeEval 200 menambah 1.200 menjadi3.354. Ini workload plan, bukan independent-sample count atau hasil model.

### Ablations

Context selected set vs target-file-only vs shuffled irrelevant dependencies; encoder-size/decoder-fixed dan decoder-size/encoder-fixed; asymmetric duel; context-length 4k/8k dan 16k hanya setelah kernel gate; matched-budget SFT baselines selain original IT zero-shot.

Task strata, contexts dan exclusions dibekukan tanpa prediction results. Same visible context/tools/lexical-BM25 retrieval untuk semua models. Gold-file selection hanya **oracle upper-bound ablation**. Multi-file edit tidak otomatis dependency-required; annotate/review distinction. Budget meliputi unique tasks, supervised tokens, actual updates; encoder input tokens bukan supervised decoder tokens.

## 7. Resource dan panjang context

### CPU/data

Metadata table bukan biaya total. Git fetch hanya selected repos/commits, cache menurut canonical repo+commit+hash, batasi history dan disk, serta catat inaccessible snapshots. Jangan clone 115k tasks/ribuan repos secara naif.

RES-Q normal clone/env per-task dapat menggandakan resource. Sequential scoring membatasi peak; reuse env hanya setelah Python/install identity dan state restoration valid.100 environments dapat menghabiskan disk jauh lebih besar daripada metadata. RAM/disk/session lifetime/Conda availability Molab belum diukur ulang.

Training tetap Python 3.13/Torch2.11/Transformers5.18 seperti notebook. Eval interpreter3.9/3.10/3.8 terpisah; jangan downgrade training dependencies. Conda/venv bukan sandbox keamanan. Scorer model-generated code membutuhkan bounded workspaces/time dan credentials yang tidak tersedia di child process.

### GPU

Satu model per run. Empat original model files~52,48GB; repo clones, expanded data dan envs menambah disk. Budget80GB sebelumnya belum mengukur full repo benchmark environments. Durasi training perlu actual warm-up/record timing; tidak ada GPU-hours pasti.

Native eager attention/masks dan FP32 logits sangat mahal pada 32k/8k. [Paket pilot](MOLAB_DATA_EVALUATION_PLAN.md) mencatat satu encoder FP32 score tensor 32 GiB, merged decoder 10 GiB dan logits 8 GiB pada shapes itu, belum seluruh layers/gradients.

**Kernel-specific gate:** configs memakai local head_dim 256/global 512. [FA issue 2427](https://github.com/Dao-AILab/flash-attention/issues/2427) masih open saat API diperiksa. [FA4 RFC 2456](https://github.com/Dao-AILab/flash-attention/issues/2456), updated 29 September 2026, menargetkan SM100/SM110 dan masih mencatat symmetric 512 planned. Jangan menyamakan B200/SM100 dengan RTX PRO6000/SM120 yang teramati di sesi sebelumnya. Ini bukan bukti semua backend 512 gagal; **actual GPU/backend/mask/head_dim forward/backward** perlu diuji tanpa silent dense fallback.

Backend harus mempertahankan joint softmax, local/causal decoder visibility, encoder context, norm/scaling dan shared KV owners. Dua independent softmax bukan replacement equivalent. Checkpointing perlu mutable shared-KV recomputation tests. Encoder-once reuse, temporal cache, last-token logits dan chunked/target-aware CE belum terintegrasi pada production path.

### Panjang mengikuti data

EditPackFT memungkinkan 1k encoder/1,5k decoder. Core repo corpus mungkin memerlukan 4k/8k input dan 1k/2k output buckets, tetapi complete reconstructed-snapshot distribution belum tersedia. RES-Q references hingga6.535 tokens memberi alasan output budget lebih besar sesudah generator/cache gate; tidak berarti semua training examples harus dipad 8k.

Pilih P95/P99/max menurut unique training tasks/categories, laporkan overflow. Benchmark target panjang tidak dihapus berdasar gold untuk menaikkan score. Baseline total memuat visible context+prefix+output; decoder cap tidak disamakan numeriknya dengan pair yang memisahkan encoder context.

## 8. Leakage dan licenses

Quarantine RES-Q/RepoQA/CrossCodeEval/HumanEvalFix/CanItEdit serta native bug cohort **sebelum training**. Resolve fork aliases: teamqurrent/requests tidak independen dari origin psf/requests. Compare Git objects/code hashes, bukan literal repo strings saja.

Input main tidak memakai solution_commit, solution_patch, gold modified-files identity, hidden tests, test_patch, post-fix hints, generated suggested fixes/solution reasoning, atau evaluator outcomes. Training boleh memakai patch sebagai target; target-conditioned training context perlu label oracle dan measure inference-selector gap.

SWE-Fixer/rebench/Gym/PR collections dapat share issue/commit lineage. Dedup lintas sumber sebelum menggabungkan counts. Context variants bukan independent tasks. Pretraining contamination yang tidak diketahui tetap limitation.

Card declarations:MIT pada SWE-Fixer/Gym/RepoExec;CC-BY4 pada rebench/DevEval;Apache2 pada R2E-GymSubset/Coeditor code. Per-repo/source-code licenses tetap disimpan. RES-Q data card tidak menyatakan license; RepoBench`cc` specificity belum jelas; BugsInPy API tidak memberikan code license. Metadata release/software license bukan clearance otomatis semua source-code redistribution.

## 9. Implementasi berikutnya dan acceptance

| Urutan | Work item | Gate |
|---|---|---|
|1|RES-Q snapshot/dependency preflight|Canonical aliases, real pinned checkout, install/disk costs, gold patch/reference tests.|
|2|Multi-file SFT adapter|BeforeSnapshot+issue+unified diff, correct direction/roundtrip, preserve target/test separation.|
|3|Patch generation/parser|Target-free decoding; multi-file paths/add/delete/rename policy; stop/length/failure handling.|
|4|Native scorer|HumanEvalFix sanity lalu RES-Q; upstream tests/termination, fixed denominator.|
|5|Attention/cache compatibility|Local 256/global 512, SM120, masks/KV sharing, eager parity, causal correctness.|
|6|Core source profile/pilot|Complete selected snapshots/split/lengths/quality; true optimizer peak/convergence; direct SFT remains hypothesis.|
|7|Four-pair matrix/baselines|Frozen manifest/context/compute, sequential GPU, paired outcomes and uncertainty.|
|8|RepoQA/CrossCodeEval/ablations|Separate function/completion/diff output modes dan scorer semantics.|

Existing notebook reader accepts one existing-file before/after modification. JSONL acceptance **tidak membuatnya mendukung RES-Q multi-file/add/delete**. RepairExample/BeforeSnapshot dapat merepresentasikan banyak files, tetapi adapters/parser/generator/scorers production belum ada. Dokumen ini tidak mengimplementasikan features tersebut.

## 10. Nama, judul, dan klaim akhir

**BifocalGemma tetap sesuai** untuk reader/generator dual capacity. Judul dapat mengikuti bukti akhirnya:

1. *BifocalGemma: Evaluating Bidirectional Repository Encoding for Local Code Repair* — target lama; butuh real repair/context evidence.
2. *BifocalGemma: Studying Encoder–Decoder Capacity Allocation for Repository-Context Code Editing and Repair* — memperjelas RQ dan RES-Q general edits.
3. *BifocalGemma: Direct Adaptation of Base Encoders and Instruction-Tuned Decoders for Repository-Context Editing* — bila feasibility/limits adaptation menjadi hasil utama.

Tidak ada rename otomatis. “Local” perlu offline inference/resource evidence; Molab 96 GB training tidak membuktikan pair muat consumer 6 GB atau lebih hemat total VRAM.

## 11. Gap yang belum terjawab

- Exact official Clean-PR download/revision/license belum dicocokkan.
- Conda/native gold tests di Molab belum berjalan; candidate bukan execution-certified.
- Rules Paper Track winner-license/eligibility/team clauses belum lengkap terbaca.
- Complete core-snapshot length/quality distribution, clone/env closure dan long-context kernel performance belum ada.
- Tidak ada claim zero contamination, admitted 20k, speedup, pasti GPU-hours atau model repair scores.

## 12. Referensi pelengkap

- [SWE-Fixer official code](https://github.com/InternLM/SWE-Fixer), [dataset API](https://huggingface.co/api/datasets/internlm/SWE-Fixer-Train-110K?blobs=true).
- [SWE-rebench execution fork](https://github.com/SWE-rebench/SWE-bench-fork).
- [RES-Q data/card](https://huggingface.co/datasets/Qurrent/RES-Q), [paper](https://arxiv.org/html/2406.16801v1), [native testbed](https://github.com/Qurrent-AI/RES-Q/blob/main/resq/testbeds/base.py).
- [RepoQA data loader](https://github.com/evalplus/repoqa/blob/main/repoqa/data.py), [release2024-06-23](https://github.com/evalplus/repoqa_release/releases/tag/2024-06-23).
- [BugsInPy reproduction study](https://mir.cs.illinois.edu/~marinov/publications/AguilarETAL23ReproducingBugsInPy.pdf).
- [RepoExec data](https://huggingface.co/datasets/Fsoft-AIC/RepoExec), [DevEval data](https://huggingface.co/datasets/LJ0815/DevEval), [RepoBench data](https://huggingface.co/datasets/tianyang/repobench_python_v1.1).

Pin source versions sebelum actual run. Counts dalam riset ini menyatakan scope pemeriksaan yang tercatat, bukan sertifikasi dataset atau success model.
