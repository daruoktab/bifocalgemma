# BifocalGemma: Research Scaffold for Local Code Repair

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![GitHub](https://img.shields.io/badge/GitHub-bifocalgemma-green)](https://github.com/daruoktab/bifocalgemma)

**Paper:** *BifocalGemma: Evaluating Bidirectional Repository Encoding for Local Code Repair*

BifocalGemma meneliti bagaimana kapasitas reader dan generator memengaruhi pemahaman
codebase serta perbaikan bug lintas file. Rancangan aktif memakai **encoder Gemma 4
base + decoder Gemma 4 IT**, langsung dibangun dengan merged attention, lalu
supervised fine-tuning. Teacher network dan distillation bersifat opsional.

Default tuning memakai koleksi private HF **`mixed_context_enc32k_dec8k_v3`**:
**44.408 train / 4.934 dev**, gabungan editing EditPackFT, CommitPackFT Python,
NextCoder Python, serta repair/localization/dependency completion dari snapshot
pre-fix SWE-rebench. Target mengikuti task: unified diff, path JSON, atau kode.
Split dan source revision dibekukan; overflow ditolak tanpa pemotongan diam-diam.
**RL dan VM/Docker tidak diperlukan pada jalur tuning.** Lisensi kode seed sintetis
NextCoder belum terverifikasi. [Rilis context v3](docs/CONTEXT_DATA_RELEASE_V3.md)
menjelaskan cohort aktif; [audit v2](docs/DATASET_REMEDIATION_V2.md) mencatat
fondasi kurasi sebelumnya. Status dan SHA rilis ada di
[dataset releases](docs/DATASET_RELEASES.md).

## Matriks penelitian

| Pasangan | Encoder | Decoder | Ordinary word embeddings lintas komponen |
| --- | --- | --- | --- |
| E2B–E2B IT | E2B base | E2B IT | Tied dari initialization, anchor decoder IT |
| E4B–E4B IT | E4B base | E4B IT | Tied dari initialization, anchor decoder IT |
| E2B–E4B IT | E2B base | E4B IT | Terpisah |
| E4B–E2B IT | E4B base | E2B IT | Terpisah |

Dua baseline utama adalah **E2B-IT dan E4B-IT decoder-only**. Decoder input/LM head
tetap tied internal; PLE antar encoder/decoder terpisah. T5Gemma/T5Gemma 2 menjadi
related work, dengan benchmark utama dibatasi pada enam model ini.

Kontribusi yang ditargetkan: adaptasi encoder-decoder Edge untuk codebase understanding/
repair, mixed initialization base/IT, serta studi alokasi kapasitas. Keunggulan kualitas
atau memori belum menjadi hasil eksperimen. [Desain studi](docs/EXPERIMENT_DESIGN.md)
menjelaskan kontrol dan confound symmetric-only tying.

## Susunan blok saat ini

| Tingkat | Yang tersedia | Makna |
| --- | --- | --- |
| Implemented | Registry/config, collator, word tying, CE, serializer, verified offline tokenization, local patch normalization, repository exports/context selection | Utilitas sesuai kontraknya; dataset fetching/harness terpisah |
| Reference | Joint-softmax merged attention, GQA, asymmetric projection, tiny assembly | Prototipe numerik kecil; short native greedy generation/scorer tersedia, optimized cache/backend pending |
| Native tiny | Full-depth Gemma encoder/merged decoder dan dua baseline, random weights | PLE/RoPE/KV sharing terintegrasi pada dimensi kecil |
| Offline SFT | Original text checkpoint loading, attention LoRA, tiled/recomputed training, v2 optimizer resume, sequential upload/evaluation | Recipe terimplementasi; hasil kualitas pretrained harus diukur di Molab |
| Placeholder | Temporal-cache/kernel optimization, agent loop, Docker harness dan general checkpoint factory | Signature dan acceptance criteria ada; fail secara eksplisit |
| Optional | Separated gated bridge dan distillation slot | Bukan recipe training default |

Tidak ada checkpoint BifocalGemma pretrained, training data besar, atau hasil code
repair yang disertakan. Model full-size diperiksa pada meta; tes native memakai
bobot acak yang diperkecil. CLI smoke merakit model reference yang lebih sederhana
lagi dan selalu menyatakan mode synthetic, bukan benchmark score.

## Repository study milestone — 8 Oktober 2026

[Notebook standalone](notebooks/bifocalgemma_repository_study.py) menyiapkan
source-pinned repo exports, matched-cohort SFT keenam model, checkpoint v2 inference,
greedy patch/function/completion generation, serta opt-in native scoring.
Notebook study menyediakan generation/scoring terpisah. Tiga training notebooks
menjalankan full training → upload → benchmark → upload secara berurutan;
masing-masing 21 cell dengan 20 isi identik dan menerima koleksi HF beku.
Setiap notebook menghasilkan empat run: dua model × recipe base AdamW dan
optimized GrokOrScale LoRA, dengan checkpoint/resume/benchmark terpisah.
[Default dan metodologi pembanding](docs/TRAINING_RECIPE_COMPARISON.md).

Validation nyata: synthetic pipeline smoke, RepoQA Python100 prepared, dan
HumanEvalFix 164/164 gold reference repairs lolos native ai scorer. Ini **bukan
model benchmark scores**. Temporal cache/long-context kernels dan real RES-Q
gold/dependency validation masih diperlukan. [Panduan implementasi](docs/REPOSITORY_STUDY_IMPLEMENTATION.md)
menjelaskan commands, cell map dan batas readiness.

## Mulai dari blok kecil

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'

python -m bifocalgemma inspect --experiment BifocalGemma-E4B-E2B
python -m bifocalgemma validate-config --config configs/experiments.json
python -m bifocalgemma smoke --experiment all --device cpu
python -m bifocalgemma smoke-native --experiment all --device cpu
python -m unittest discover -s tests
pyrefly check
ty check
ruff check --fix
ruff format --check .
```

Perintah tersebut tidak mengunduh bobot model. smoke melakukan satu synthetic
forward/backward per pasangan, tanpa optimizer/training riil. CUDA dapat dipilih
untuk reference/native tiny jika tersedia; ia tidak memuat model E2B/E4B asli.
[Native local validation](docs/NATIVE_LOCAL_VALIDATION.md) menjelaskan API, cache
tokenizer, patch direction SWE-smith, dan batas pengujian.

```powershell
python -m bifocalgemma train --experiment BifocalGemma-E2B-E2B
python -m bifocalgemma evaluate --predictions outputs/predictions.jsonl
```

train/evaluate sengaja exit 2 dan menyebut blok yang belum terintegrasi serta
tautan docs. Placeholder tidak menghasilkan checkpoint atau skor palsu.

## Notebook marimo / Molab

[Architecture test standalone](notebooks/bifocalgemma_architecture_test.py) berisi
13 cell: control center, environment, config/meta shapes, batch/loss, merged attention,
model, tiny tests, optional full-size BF16 probe untuk GPU 96 GB, serta report JSON.
Unggah satu file tersebut ke Molab. Panduan ada pada
[marimo architecture test](docs/MARIMO_ARCHITECTURE_TEST.md).

```powershell
python -m marimo check --strict notebooks/bifocalgemma_architecture_test.py
python -m marimo edit notebooks/bifocalgemma_architecture_test.py
```

Untuk **checkpoint asli dan pengukuran training GPU 96 GB**, unggah
[pretrained memory probe](notebooks/bifocalgemma_pretrained_probe.py). Pilot memakai
encoder base / decoder IT dan attention LoRA, lalu menaikkan jumlah model resident
1–4 dengan round-robin versus overlapping forward graphs. Hasil VRAM ditulis dari
runtime Molab. [Panduan pretrained probe](docs/PRETRAINED_GPU_PROBE.md) menjelaskan
download/cache, provenance, optimizer policy dan interpretasi PASS/OOM/guard.

Tiga notebook khusus **training** tersedia:
[simetris](notebooks/bifocalgemma_train_symmetric.py),
[asimetris](notebooks/bifocalgemma_train_asymmetric.py), dan
[baseline](notebooks/bifocalgemma_train_baseline.py). Masing-masing 21 cell,
20 isi cell identik; hanya tiga default control-center berbeda. Cell Training policy mengatur optimizer groups, LR/warmup serta freeze policy. Pipeline mencakup
Pinned private HF dataset → LoRA SFT → dev loss → upload → benchmark → upload.
RUN_TRAINING, RUN_BENCHMARKS dan upload default True. Satu model resident;
dua checkpoint terakhir per model/recipe disimpan, dengan resume stage dan prediksi.
Setiap notebook menjalankan dua model × dua recipe, sehingga menghasilkan empat
hasil SFT. Validation penuh setiap25% total update, checkpoint setiap10%, ditambah
hasil akhir. Base AdamW dan optimized GrokOrScale LoRA dimulai dari pretrained
dan seed yang sama; optimized masih hipotesis eksperimen.
[Panduan training notebooks](docs/TRAINING_NOTEBOOKS.md)
menjelaskan template bersama dan cara menjaga cell tetap sinkron.

| Keluarga | Marimo | Jupyter, tanpa outputs |
| --- | --- | --- |
| Simetris | [Python](notebooks/bifocalgemma_train_symmetric.py) | [ipynb](notebooks/bifocalgemma_train_symmetric.ipynb) |
| Asimetris | [Python](notebooks/bifocalgemma_train_asymmetric.py) | [ipynb](notebooks/bifocalgemma_train_asymmetric.ipynb) |
| Decoder-only | [Python](notebooks/bifocalgemma_train_baseline.py) | [ipynb](notebooks/bifocalgemma_train_baseline.ipynb) |

Default training kini memakai **checkpointing + tiled attention FP32 + chunked loss**
untuk menjaga objective pada input sampai32K dan shifted decoder budget8K termasuk
prefix. Baseline total budget41472; overflow ditolak tanpa pemotongan diam-diam.
Monitor live menampilkan model/tahap, train/dev CE,
progres validasi, GPU/RAM, throughput, LR dan checkpoint/error. Preflight panjang maksimum
menjalankan forward/backward tanpa memperbarui bobot. Lihat
[training hemat VRAM](docs/MEMORY_EFFICIENT_TRAINING.md); penghematan pretrained dan
kelayakan keenam konfigurasi pada32K/8K tetap harus diukur di Molab.
[Sequential training](docs/SEQUENTIAL_TRAINING.md) menjelaskan durabilitas dan RAM cache.

## Setup dari checkout baru

```powershell
git clone https://github.com/daruoktab/bifocalgemma.git
cd bifocalgemma
conda create -n ai python=3.13
conda activate ai
uv pip install --python (Get-Command python).Source -e ".[dev]"
```

Jika ai sudah ada, cukup aktifkan environment itu. Dependency installation berbeda
dari download bobot. Untuk source checkout tanpa editable install, gunakan PYTHONPATH
seperti contoh di atas. [Config training](configs/train.example.json) menunggu
profiling untuk hyperparameters; teacher_required=false.

## Struktur repo

| Folder | Isi |
| --- | --- |
| `src/bifocalgemma/` | Implementasi model, data, training, evaluasi, dan CLI |
| `notebooks/` | Notebook standalone dan ekspor Jupyter training |
| `tools/` | Generator notebook, template bersama, kurasi dan audit data |
| `tests/` | Tes dan fixture kecil untuk kontrak/numerik/resume |
| `configs/` dan `examples/` | Config serta contoh penggunaan |
| `docs/` | Panduan aktif, riset, status dan riwayat validasi |
| `outputs/` | Cache, hasil, dan arsip eksperimen lokal; diabaikan Git |

File `working-molab-v8-exp-temp.py` adalah referensi lokal yang diabaikan Git.
Edit template/modul bersama lalu regenerate notebook; cara reproduksi ada di
[panduan notebook](docs/TRAINING_NOTEBOOKS.md).

## Jalur membaca dan membangun

Mulai dari [index docs](docs/README.md), kemudian:

1. [Proposal](docs/proposal.md) dan [desain eksperimen](docs/EXPERIMENT_DESIGN.md).
2. [Arsitektur/tensor contracts](docs/ARCHITECTURE.md).
3. [Build guide: tiap blok dan acceptance criteria](docs/BUILD_GUIDE.md).
4. [Data](docs/DATA_PIPELINE.md), [training](docs/TRAINING.md), [evaluasi](docs/EVALUATION.md).
5. [Audit seluruh JSON](docs/GEMMA4_JSON_AUDIT.md) dan [referensi backbone](docs/GEMMA4_EDGE_ARCHITECTURE.md).
6. [Log riset](docs/RESEARCH_LOG.md), [hasil validasi](docs/VALIDATION.md), dan [panduan tim](docs/TEAM_PLAYBOOK.md).

Parameter efektif berbeda dari storage fisik. Word tying tidak menyatukan PLE,
dan pengurangan self-KV bukan pengurangan total VRAM. Dataset, waktu/hardware
dan hasil benchmark harus diukur serta diulang melalui harness.

Kontributor: [daruoktab](https://github.com/daruoktab) dan [attmhd](https://github.com/attmhd).
Kode memakai [Apache License 2.0](LICENSE). [Ketentuan kompetisi](docs/COMPETITION_GUIDE.md)
yang belum terverifikasi tetap diberi status demikian.

Notebook Molab selalu mengambil koleksi dataset beku dari HF; tidak ada kurasi ulang
atau pilihan raw-source di notebook. Default satu epoch penuh dan upload artefak run
selesai aktif. Pengguna dapat mengubah RUN_TRAINING/HF_UPLOAD_ENABLED menjadi False.
Koleksi aktif v3:44.408 train /4.934 dev. Lihat
[perbandingan recipe](docs/TRAINING_RECIPE_COMPARISON.md) untuk angka default,
urutan12 run, checkpoint, dan batas bukti.

Dataset releases are staged in the separate private HF dataset repository
[daruokta/bifocalgemma-data](https://huggingface.co/datasets/daruokta/bifocalgemma-data).
See [release guide](docs/DATASET_RELEASES.md) for immutable revisions, current
collection readiness and benchmark/gold separation. Data payloads remain outside
this Git repository.
