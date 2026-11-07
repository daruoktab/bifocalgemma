# Peta Repo: V2, V3, dan Runtime Bersama

Dokumen ini menjelaskan batas antara arsitektur V2 (legacy) dan V3 (produksi),
supaya perubahan pada satu generasi tidak menyentuh generasi lain.

## Ringkas

| Jalur | Status | Lokasi |
| --- | --- | --- |
| V3 — `swiglu_dual_gate_decoupled_cross_v3` | **Aktif**, dipakai semua notebook training | `src/bifocalgemma/v3/` |
| V2 — `normalized_merged_v2` | Legacy, dipakai test/audit/probe | `src/bifocalgemma/models/v2/` |
| Runtime bersama | Dipakai V2 dan V3 | `models/memory_efficient.py`, `models/inference_cache.py`, `models/flex_attention.py` |
| Entry notebook | Facade tipis ke V3 | `tools/molab_matrix_support.py` |

## Struktur `src/bifocalgemma/`

```
config.py                  ExperimentSpec, CheckpointRef, registry model
contracts.py               TokenizedRepairExample, Seq2SeqBatch, ModelOutput
scaffold.py                daftar blok M01..M07 dan FeatureNotReadyError

models/
  v2/                      ← LEGACY V2 (merged attention)
    alignment.py           NormalizedAlignment (linear + RMSNorm)
    merged_attention.py    ReferenceMergedAttention, build_joint_visibility
    native.py              NativeMergedAttention, NativeTinyModel, smoke
    pretrained.py          loader checkpoint V2 + adapter inference V2
    checkpoint_transfer.py konversi bobot Gemma → modul native
    bridge.py              GatedCrossAttentionBridge (V2 opsional)
    routing.py             RepoContextRouter
    code_graph_projector.py ASTGraphProjector
    reference.py           TinyReferenceModel
    embeddings.py          apply_word_embedding_policy
    model.py, cache.py, initialization.py   placeholder M04–M06
  memory_efficient.py      tiling + gradient checkpointing (dipakai V2 & V3)
  inference_cache.py       InferenceCache + CachedAttention (dipakai V2 & V3)
  flex_attention.py        fused attention opsional (dipakai tiling)

v3/                        ← PRODUKSI V3
  constants.py             repo HF, step budget, MATRIX_MODELS, BENCHMARKS, tier VRAM
  runtime.py               allocator guard, beta_N, query-injected context, patch tokenizer
  bridge.py                SwiGLUDualGateBridge
  attention.py             Bf16FullLinear, FullCrossTiledAttention,
                           FullCrossCachedAttention, configure_full_rank_cross_attention
  objectives.py            diff token weights, streaming pointer loss, optimizer groups
  stage.py                 configure_matrix_stage (freeze, LR, warmup)
  curriculum.py            anti-forgetting comprehension replay
  benchmark.py             batched greedy decode, 8 benchmark, eval batch sizing
  checkpoint.py            save/resume/prune + synchronisasi HF
  matrix.py                make_experiment_spec, run_matrix_* (entry point)

data/, evaluation/, inference/, training/, agent/   → infrastruktur bersama
training/v2.py, v2_hub.py, v2_preflight.py, v2_diagnostics.py   → pipeline V2
```

## Aturan batas

1. **Kode arsitektur baru hanya boleh masuk `v3/`.** Jangan menambah kelas atau
   fungsi training V3 ke `tools/molab_matrix_support.py`; file itu hanya facade.
2. **`v3/` tidak boleh mengimpor `models/v2/` kecuali `v2.native`.** V3 masih
   memakai `NativeTinyModel` sebagai kerangka Gemma 4 sebelum attention-nya
   diganti; ketergantungan ini eksplisit dan satupun arah sebaliknya tidak ada.
3. **`models/v2/` tidak boleh mengimpor `v3/`.** Kalau butuh fungsi bersama,
   pindahkan ke `models/` atau `training/` netral.
4. **Tidak ada monkey-patch baru.** `configure_full_rank_cross_attention` masih
   mengganti `__class__` dan `inference_cache.CachedAttention` agar kompatibel
   dengan checkpoint dan notebook yang sudah ada. Perbaikan bug berikutnya
   sebaiknya mengganti mekanisme ini dengan komposisi eksplisit.
5. **Label `architecture_version` masih `"normalized_merged_v2"`.** Ini disengaja:
   `training/policy.validate_training_policy` mensyaratkannya, dan belum diubah
   pada pass pemisahan struktur ini. Jangan mengandalkan label itu untuk
   membedakan artefak; gunakan hash bundle dan kelas bridge/attention.

## Notebook dan runtime bundle

Kesepuluh `notebooks/train_*.py` memanggil:

```python
ms = importlib.import_module("tools.molab_matrix_support")
ms.run_matrix_training_and_benchmark(...)
```

`tools/molab_matrix_support.py` sekarang hanya memuat ulang `bifocalgemma.v3`.
Runtime notebook dipin sebagai zip SHA-256 dari HF. Karena bundle lama belum
memuat `src/bifocalgemma/v3/`, facade menyisipkan `src/` repo ke `sys.path`
saat dijalankan dari checkout. Di Molab, bundle lama tetap memakai file
monolitik lamanya sampai bundle baru diterbitkan.

Setelah perubahan struktural seperti ini, hash `matrix_bundle()` berubah.
`python tools/generate_full_matrix_notebooks.py --check` akan gagal sampai:

```bash
python tools/generate_full_matrix_notebooks.py --publish
```

dijalankan dengan kredensial HF. Perintah itu mengunggah zip baru, menulis
`notebooks/runtime-manifest.json`, dan meregenerasi sepuluh notebook.

## Di mana memperbaiki bug tertentu

| Gejala | Berkas |
| --- | --- |
| Gate cross terlalu lemah/kuat, init gate | `v3/attention.py`, `v3/bridge.py` |
| Cross-KV hanya dua ruang untuk layer shared | `v3/attention.py` (`configure_full_rank_cross_attention`) |
| Bobot diff salah sasaran | `v3/objectives.py` (`compute_diff_token_weights`) |
| Pointer loss terlalu kuat / label noise | `v3/objectives.py` (`streaming_pointer_auxiliary_loss`) |
| Stage 1 membekukan/melatih parameter salah | `v3/stage.py` |
| Campuran replay / truncation 4096 | `v3/curriculum.py` |
| Evaluasi OOM atau batch salah | `v3/benchmark.py` |
| Checkpoint tidak bisa resume / hash mismatch | `v3/checkpoint.py` |
| Urutan training, upload, benchmark | `v3/matrix.py` |
| Arsitektur V2 / merged attention | `models/v2/` |

## Verifikasi cepat

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/test_molab_matrix_notebooks.py -q
python -m pytest tests/test_v2_models.py tests/test_native_integration.py -q
ruff check src tests tools
ty check src
marimo check --strict notebooks/*.py

# Pyrefly 1.4.0 tidak membaca site-packages dari python-interpreter-path,
# sehingga torch/transformers terlihat "hilang". Beri site-packages eksplisit:
$env:PYTHONPATH = "$env:CONDA_PREFIX\Lib\site-packages"
pyrefly check src tools tests
```

Catatan: `ruff format --check .` masih gagal pada beberapa berkas lama yang
belum pernah diformat (mis. `training/benchmark_upload.py`, `tests/test_gsm8k.py`).
Itu kondisi sebelum pemisahan V2/V3; jalankan `ruff format` pada berkas yang
Anda ubah saja agar diff tetap fokus.