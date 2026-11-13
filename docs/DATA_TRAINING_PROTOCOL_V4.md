# Data and training protocol V4

This document describes the implemented local protocol for the main ten-model matrix, version `bifocalgemma_shared_schedule_2026_11_12`. Local tests exercise all ten tiny model configurations, shared loaders, label masks, stage transitions, interruption/resume, partial-backward OOM recovery, effective benchmark settings, and trained-adapter export/reload. These checks do not establish pretrained model quality. The current source runtime is published at immutable Hub revision `f6f208e4ca09ce42cc4dce11a293e8a3460fb5d6`, including the dashboard repair and inherited-host UI correction without changing training semantics; see [runtime release verification](RUNTIME_RELEASE.md).

## Training exposure

The frozen training pool contains 44,408 original examples. A single versioned, deterministic schedule specifies their order before model-specific tokenization. Each original ID appears exactly once, paired with its canonical input and target. At an effective batch of four, this yields 11,102 optimizer updates. Paired models consume the first 22,204 presentations in stage 1 and the remainder in stage 2: 5,551 updates per stage. Decoder-only baselines consume the same complete order in one stage. These counts apply to the complete frozen pool; a smaller pilot must state its own counts and stage boundaries.

The main matrix no longer replaces repair examples with synthetic replay. The schedule manifest binds original IDs, input/target hashes, canonical encoder lengths, dataset identity, algorithm, seed and effective batch. Each update checks the loader's IDs against its scheduled window and emits IDs, stage, supervised-token counts by task, and example counts by source. Checkpoints retain the next-example cursor and the last 100 update records; the final report binds the complete schedule signature. This is not a separate, full training-event journal. Tokenization rejects overflow rather than truncating. Canonical length bins are computed before family-specific routing, so baseline prefix length cannot change the order.

The default matrix objective is target-only, uniformly weighted cross-entropy (`uniform_target_v1`). The legacy marker-weighted objective remains available for explicit experiments; it is not the main mixed-task default. No claim is made that either objective or a particular task mixture is empirically optimal. `stage1_steps=0` permits an explicitly identified joint-from-start control.

A checkpoint binds the schedule and run identity and records the global next-example position, optimizer state and random state. Resume must reject an incompatible dataset, schedule or model configuration before mutating model state. Checkpoints must either be taken at a completed optimizer-update boundary or carry enough in-flight accumulation state to reproduce an uninterrupted update. The stage boundary must not restart the shared schedule.

Packed token caches bind the prompt recipe and hashes of the prompt/tokenization source files. Final training reports retain completed stage summaries when a completed checkpoint is restored for export, and record the realized cursor and ID-order hash. Historical training-equivalence approvals remain historical: this changed schedule/objective must not be remapped to an old training hash simply to synchronize a test.

## Evaluation contract

Benchmark settings are resolved and recorded for each case from a versioned protocol. The planned output limit is the minimum of the row request, the suite limit and the remaining decoder sequence budget. For the currently profiled release this retains a 32-token row request for QA when it fits, and caps CrossCodeLong at the suite's 2,048-token output limit when the row asks for more. These are output limits; the decoder's full sequence limit also includes the actual prompt. An empty remaining budget is an explicit failure, not silent truncation.

Strict letter-only accuracy is primary; resilient extraction is a diagnostic on the same predictions. Reports name both metrics and include invalid-output counts, length-limit counts, and denominators. A rate is its count divided by the corresponding denominator. The effective protocol binds actual runtime configuration, declared model spec, sample IDs/seed, public/gold fingerprints, generation policy and semantic scorer fingerprint. Outer pipeline report reuse no longer bypasses this validation. A changed contract requires a separate output; signatures are never rewritten to approve old receipts.

Prediction journals without a state identity are rejected. Each stored prediction also binds the protocol signature, so copying or restoring a journal from another protocol cannot silently adopt old scores.

## Grounded data supplement

`tools/build_grounded_training_pool.py` builds a separate local pool from admitted pre-fix training contexts, with inherited train/dev groups, source revisions, snapshot hashes and per-file license evidence. It excludes foundation rows without this snapshot provenance. It never reads a parent's repair target or any benchmark question. Three deterministic tasks are available: exact source-line extraction, four-option source-line MCQ, and masked source-line reconstruction. Prompts contain explicit line locators; validation reconstructs each prompt and target from recorded evidence and rejects tampering. Reconstruction evidence is retained for auditing but is not passed to the model as context.

These are source retrieval and denoising proxies. They do not replace human-authored software-understanding QA, new long repair cases, functional repair validation or an independent long-context dev set. Mixture ratios and quality effects remain pilot questions. The generator does not alter the pinned release or choose a new main-matrix mixture. Every admitted row must fit all ten real, hash-verified local tokenizer configurations; overflow is counted and rejected consistently.

The main runner accepts `pool_directory` for a separately named, hash-verified local pool. All three matrix entry points propagate this option. Its identity and one-pass update count are derived from its manifest. Passing a smaller `total_steps` explicitly selects a prefix pilot; exceeding one pass is rejected. Keep local-pool pilots in a separate `output_root`. Training and subsequent benchmark calls must use matching pool and stage settings.

```powershell
conda run --no-capture-output -n ai python tools/build_grounded_training_pool.py --source outputs/mixed_context_enc32k_dec8k_v3 --output outputs/grounded_source_proxy_v1_pilot --tokenizer-cache outputs/context_tokenizers --max-parents 100
conda run --no-capture-output -n ai python tools/validate_training_schedule.py --pool outputs/mixed_context_enc32k_dec8k_v3 --output outputs/shared_schedule_v4_audit --stage1-updates 5551
```

The first command intentionally refuses an existing output directory. Reuse an already generated, validated pool, or choose a new named destination; do not overwrite release artifacts.

The local 2026-11-12 build admitted 294 train and 300 dev rows (98/100 per task), rejecting six candidates for token overflow. Its pool fingerprint is `2ed408a3fc806000418e49ce9750a122ce273456fc8161319a7d9ca994f079fa`. The complete pinned-pool audit retained all 44,408 train and 4,934 dev rows, with no duplicate/missing training IDs and 22,204 presentations in each stage. The pinned schedule signature is `2451993cc2a02d17c8c2f2d76eba76b8b67fcf7a1b36a72bff3b5bc02b61d5f8`. Local artifacts live under `outputs/grounded_source_proxy_v1_pilot` and `outputs/shared_schedule_v4_pinned_audit`; neither changes the published release.

## Interpretation and release checks

The matched schedule controls raw example exposure. It does not make paired and baseline models identical in architecture, tokenization, trainable parameters, stage policy, compute or memory use. Report the comparison as a whole-system comparison. A claim about architecture alone requires additional controls and ablations. Measure context sensitivity, functional repair, held-out task quality, throughput and peak memory under specified conditions; report uncertainty and training-seed variation where appropriate.

Before a long run, verify the schedule and raw inputs, ten loader traces, tokenizer compatibility, post-collation labels and weights, stage transition and interruption/resume equivalence. Test export/reload and imports from the actual pinned notebook runtime. A local preview bundle is not evidence that an immutable remote runtime has been published. Preserve old release artifacts and use a new protocol/run identity for changed training semantics.

## Focused local checks

The project environment is conda `ai`. The following tests use existing tiny fixtures; they do not download pretrained model weights or perform a full training run:

```powershell
conda run --no-capture-output -n ai python -m pytest tests/test_shared_schedule.py tests/test_matrix_schedule_runtime.py tests/test_effective_benchmark_protocol.py tests/test_grounded_tasks.py -q
```

After intentionally regenerating the runtime and notebooks, verify the bundle and all ten generated files with:

```powershell
conda run --no-capture-output -n ai python tools/generate_full_matrix_notebooks.py --check
```

The runtime tests train tiny models locally, including a complete matrix entry-point run on a local fixture pool, then resume a completed checkpoint and verify that exported trained weights remain identical. Export tests compare non-padding logits across baseline, symmetric and asymmetric models, and ensure independently configured encoders remain untied. This caught and fixed a Transformers tied-weight cache problem that shape-only tests missed. No pretrained weights are downloaded by these tests.

The actual generated supplement also passed `prepare_curated_training` with the production lossless packed cache for `e2b-it` and `4base-2it`. Both loaders retained all 294 train/300 dev IDs, produced the same target-token digests and schedule signature, and passed post-collation label checks. Its one-pass horizon is 74 updates, including an explicit final partial window of two examples. The shared local-pool signature is `cc5995954dd564db9c6b20c261ab0ce4435f6d9f2bc1a7b02ecf94a5d86353d7`. This verifies the production loader for these two configurations; all ten tokenizer combinations were checked by the generator, and all ten stage executions were checked separately with tiny integer-token fixtures.

Local validation artifacts are under `outputs/validation_v4`: the full pytest XML, `actual_pool_loader_report.json`, the written loader probe, and the written isolated-runtime probe. The final runtime ZIP was checked in a `python -I` process that asserted imported module paths belonged to the extracted bundle, then performed a tiny baseline and asymmetric update. Runtime SHA-256: `cf70346b7c90c626c7c238fcea0b540e37b9425e9ea64179536b568948270556`.

Required repository checks use the selected `ai` environment: `pyrefly check --min-severity=warn`, `ty check`, `ruff check --fix`, and `marimo check --strict` on all ten affected notebooks. Regenerate and recheck the local runtime after source changes, then run `graphify update .`. Full training, remote publication, functional-repair pilots and seed/uncertainty studies are separate empirical work.

## Final local acceptance, 2026-11-12

The final complete suite passed: **649 passed, 4 skipped, 1 xfailed**, with **44 subtests passed**, in 965.26 seconds. The single warning came from the existing fast-pipeline smoke test and Triton's default FP-fusion setting. Receipt: `outputs/validation_v4/pytest_final.xml`.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
conda run --no-capture-output -n ai python -m pytest -q --tb=short --junitxml=outputs/validation_v4/pytest_final.xml
```

Pyrefly (`--min-severity=warn`), ty, Ruff (`--fix`), strict marimo validation of all ten matrix notebooks, and deterministic notebook/runtime `--check` all passed. The final isolated runtime check passed for baseline and asymmetric tiny stages. The knowledge graph was updated after code changes. The old training-equivalence approval was preserved and does not authorize this protocol. Pretrained quality results and an immutable remote publication are not claimed by these local acceptance checks.

The subsequent bootstrap release correction published the same source ZIP, pinned all ten notebooks to the immutable revision, and added a release check rejecting `LOCAL-PREVIEW`. An anonymous network test started without a local ZIP/cache and executed the rendered notebook loader with a temporary marimo cell filename. Its checksum and imported module paths were verified; 62 focused bootstrap/notebook tests passed. This publication does not change model or training semantics.
