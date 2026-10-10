# BifocalGemma: Research Scaffold for Local Code Repair

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![GitHub](https://img.shields.io/badge/GitHub-bifocalgemma-green)](https://github.com/daruoktab/bifocalgemma)

**Paper:** *BifocalGemma: Evaluating Bidirectional Repository Encoding for Local Code Repair*

BifocalGemma meneliti bagaimana kapasitas reader dan generator memengaruhi pemahaman
codebase serta perbaikan bug lintas file. Rancangan aktif memakai **encoder Gemma 4
base + decoder Gemma 4 IT**, langsung dibangun dengan merged attention, lalu
supervised fine-tuning. Teacher network dan distillation bersifat opsional.

Tuning awal memakai **offline instruction-to-patch SFT**, dengan EditPackFT sebagai
data utama dan CommitPackFT Python opsional setelah dedup. Data berupa kode sebelum
edit + instruksi + target patch. **RL dan VM/Docker tidak diperlukan pada jalur
tuning**; agent rollouts/SWE-bench environments ditunda. Single-file editing pilot
akan dilanjutkan dengan static multi-file data untuk menguji pemahaman codebase.

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
| Placeholder | Pretrained weight/production/cache integration, dataset fetching/AST, production trainer, agent, Docker harness dan general checkpoint factory | Signature dan acceptance criteria ada; fail secara eksplisit |
| Optional | Separated gated bridge dan distillation slot | Bukan recipe training default |

Tidak ada checkpoint BifocalGemma pretrained, training data besar, atau hasil code
repair yang disertakan. Model full-size diperiksa pada meta; tes native memakai
bobot acak yang diperkecil. CLI smoke merakit model reference yang lebih sederhana
lagi dan selalu menyatakan mode synthetic, bukan benchmark score.

## Repository study milestone — 8 Oktober 2026

[Notebook standalone](notebooks/bifocalgemma_repository_study.py) menyiapkan
source-pinned repo exports, matched-cohort SFT keenam model, checkpoint v2 inference,
greedy patch/function/completion generation, serta opt-in native scoring.
Default actions/network/execution disabled. Dua training notebooks tetap 20 cell
dengan 18 isi identik dan menerima repository train exports melalui local_jsonl.

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
ruff check .
ruff format --check .
ty check .
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

Dua notebook khusus **training** tersedia:
[simetris](notebooks/bifocalgemma_train_symmetric.py) dan
[asimetris](notebooks/bifocalgemma_train_asymmetric.py). Keduanya 20 cell,
19 isi cell identik; hanya tiga default control-center berbeda. Cell Training policy mengatur optimizer groups, LR/warmup serta freeze policy. Pipeline mencakup
EditPackFT/JSONL → LoRA SFT → dev loss → checkpoint/resume. RUN_TRAINING=False
saat pertama dibuka. [Panduan training notebooks](docs/TRAINING_NOTEBOOKS.md)
menjelaskan template bersama dan cara menjaga cell tetap sinkron.

## Setup dari checkout baru

```powershell
git clone https://github.com/daruoktab/bifocalgemma.git
cd bifocalgemma
conda create -n ai python=3.12
conda activate ai
python -m pip install -e ".[dev]"
```

Jika ai sudah ada, cukup aktifkan environment itu. Dependency installation berbeda
dari download bobot. Untuk source checkout tanpa editable install, gunakan PYTHONPATH
seperti contoh di atas. [Config training](configs/train.example.json) menunggu
profiling untuk hyperparameters; teacher_required=false.

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

Full-corpus admission dan complete-epoch defaults dijelaskan dalam [panduan](docs/FULL_CORPUS_TRAINING.md). Kurasi penuh sedang berjalan; jumlah upstream bukan jumlah yang sudah lolos training admission. Pilot model asli E2B–E2B berhasil 100 steps dengan peak allocated 23,38 GiB; ini belum skor repair.

Manual Molab upload kini menyiapkan corpus melalui cell PREPARE_FULL_POOL. Dependency mengikuti env ai (Torch 2.14.1/cu130, Transformers 5.19.0). Foundation terverifikasi: 44.333 train / 4.926 dev; main repository corpus memakai snapshot berlisensi dan public-issue context. Lihat [panduan](docs/FULL_CORPUS_TRAINING.md).

Dataset releases are staged in the separate private HF dataset repository
[daruokta/bifocalgemma-data](https://huggingface.co/datasets/daruokta/bifocalgemma-data).
See [release guide](docs/DATASET_RELEASES.md) for immutable revisions, current
collection readiness and benchmark/gold separation. Data payloads remain outside
this Git repository.
