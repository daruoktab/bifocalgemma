# Peta modul BifocalGemma

Implementasi aktif disatukan ke modul canonical tanpa direktori `v2/` atau `v3/`.
Label versi pada beberapa format data, recipe, alias API, dan receipt lama tetap
merupakan identitas kompatibilitas, bukan lokasi source.

## Model, data, dan evaluasi

| Berkas | Tanggung jawab |
| --- | --- |
| [config.py](../src/bifocalgemma/config.py) | Registry konstruksi enam model, matriks training sepuluh model, backbone/revision, benchmark |
| [configuration_bifocalgemma.py](../src/bifocalgemma/configuration_bifocalgemma.py) | Konfigurasi Hugging Face |
| [modeling_bifocalgemma.py](../src/bifocalgemma/modeling_bifocalgemma.py) | Native model, tiny smoke, save/load, generation |
| [models/bridge.py](../src/bifocalgemma/models/bridge.py) | SwiGLUDualGateBridge |
| [models/cross_attention.py](../src/bifocalgemma/models/cross_attention.py) | Decoupled full-rank cross-attention dan cached/tiled paths |
| [models/embeddings.py](../src/bifocalgemma/models/embeddings.py) | Kebijakan embedding |
| [models/checkpoint_transfer.py](../src/bifocalgemma/models/checkpoint_transfer.py) | Transfer checkpoint dan loader lokal |
| [models/memory_efficient.py](../src/bifocalgemma/models/memory_efficient.py) | Tiling dan gradient checkpointing |
| [data/curriculum.py](../src/bifocalgemma/data/curriculum.py) | Helper comprehension/replay |
| [evaluation/benchmark.py](../src/bifocalgemma/evaluation/benchmark.py) | Batched decode, benchmark cohorts, sizing batch |
| [evaluation/context.py](../src/bifocalgemma/evaluation/context.py) | Prompt dan scoring task context |
| [evaluation/gsm8k.py](../src/bifocalgemma/evaluation/gsm8k.py) | GSM8K prompt dan jawaban numerik |

## Training

| Berkas | Tanggung jawab |
| --- | --- |
| [training/matrix.py](../src/bifocalgemma/training/matrix.py) | Entry point training, export, benchmark, dan reuse |
| [training/schedule.py](../src/bifocalgemma/training/schedule.py) | Schedule immutable dan manifest fail-closed |
| [training/stage.py](../src/bifocalgemma/training/stage.py) | Parameter aktif, optimizer groups, LR/warmup dua tahap |
| [training/objectives.py](../src/bifocalgemma/training/objectives.py) | Diff weighting, CE, pointer auxiliary loss, optimizer |
| [training/runtime.py](../src/bifocalgemma/training/runtime.py) | Memory guards, tokenization, beta, query-injected context |
| [training/checkpoint.py](../src/bifocalgemma/training/checkpoint.py) | Save, restore, prune, dan sinkronisasi matrix |
| [training/hub.py](../src/bifocalgemma/training/hub.py) | Identitas artefak dan upload |
| [training/diagnostics.py](../src/bifocalgemma/training/diagnostics.py) | Diagnosis kegagalan |
| [training/preflight.py](../src/bifocalgemma/training/preflight.py) | Pemeriksaan resource sebelum training |
| [training/fast_pipeline.py](../src/bifocalgemma/training/fast_pipeline.py) | Workflow fast-training terpisah |

## Notebook dan bundle

Kesepuluh notebook `notebooks/train_<model_tag>.py` (underscore pada nama berkas,
misalnya `train_2base_2it.py` untuk kunci `2base-2it`) memuat ZIP yang diverifikasi
SHA-256 melalui [generator matrix](../tools/generate_full_matrix_notebooks.py).
[Facade Molab](../tools/molab_matrix_support.py) mengekspos API canonical.
[Launcher](../tools/notebook_launcher.py) memeriksa dependency serta modul yang
sudah termuat sebelum aktivasi runtime.

Bundle berisi source, config, contoh repository-study, LICENSE, dan helper runtime.
Bobot, token HF, cache, dan hasil training tidak termasuk. Notebook repository-study
memiliki [generator terpisah](../tools/generate_repository_notebook.py).

```powershell
python tools/generate_full_matrix_notebooks.py --check --require-published
python tools/generate_repository_notebook.py --check
python -m pytest tests/test_matrix_runtime_loader.py -q
pyrefly check
ty check --python (Get-Command python).Source src tools tests
ruff check --fix src tools tests tests_e2e notebooks
Get-ChildItem notebooks/*.py | ForEach-Object { python -m marimo check --strict $_.FullName }
```

Gunakan environment proyek dan konfigurasi interpreter Pyrefly yang sesuai.
Setelah perubahan source, jalankan `graphify update .` dan regenerate bundle
sesuai [panduan rilis](RUNTIME_RELEASE.md).

## Kompatibilitas run lama

Baseline bersih tidak menerima checkpoint dan schedule sebelum penyatuan.
Jangan mengarahkan run lama ke notebook baru. Publikasi runtime hanya menambah
ZIP dengan nama berdasarkan hash, lalu notebook baru dipin ke commit immutable.
`runtime/latest.zip`, ZIP lama, dan artefak checkpoint remote dipertahankan.
