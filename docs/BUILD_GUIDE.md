# Build Guide: Blok Konstruksi dan Acceptance Criteria

Setiap blok memiliki ID, file sumber, kontrak dan batas readiness. Implemented/reference
tidak membuat placeholder sesudahnya otomatis siap. Teacher/conversion/distillation
tidak berada pada jalur default direct SFT.

## Peta blok

| ID | Status | Source | Peran |
| --- | --- | --- | --- |
| [C01](#c01) | implemented | [config.py](../src/bifocalgemma/config.py) | Six-model specification and checkpoint provenance |
| [D01](#d01) | implemented | [contracts.py](../src/bifocalgemma/contracts.py) | Repository, example, batch and prediction contracts |
| [D02](#d02) | implemented | [data/collator.py](../src/bifocalgemma/data/collator.py) | Pretokenized teacher-forcing batches |
| [D03](#d03) | placeholder | [data/tokenization.py](../src/bifocalgemma/data/tokenization.py) | Pinned base/IT tokenizers and formatting |
| [D04](#d04) | placeholder | [data/adapters.py](../src/bifocalgemma/data/adapters.py) | Training dataset adapters and overlap audit |
| [D05](#d05) | placeholder | [data/ast.py](../src/bifocalgemma/data/ast.py) | Tree-sitter features aligned to original token offsets |
| [M01](#m01) | reference | [models/merged_attention.py](../src/bifocalgemma/models/merged_attention.py) | Joint softmax, GQA and asymmetric projection |
| [M02](#m02) | implemented | [models/embeddings.py](../src/bifocalgemma/models/embeddings.py) | Symmetric word parameter sharing |
| [M03](#m03) | reference | [models/reference.py](../src/bifocalgemma/models/reference.py) | Tiny offline assembly and backward |
| [M04](#m04) | placeholder | [models/initialization.py](../src/bifocalgemma/models/initialization.py) | Base/IT weight mapping into Gemma layers |
| [M05](#m05) | placeholder | [models/model.py](../src/bifocalgemma/models/model.py) | Full Gemma encoder/merged decoder integration |
| [M06](#m06) | placeholder | [models/cache.py](../src/bifocalgemma/models/cache.py) | Production self/cross KV and repository cache |
| [T01](#t01) | implemented | [training/objectives.py](../src/bifocalgemma/training/objectives.py) | Patch-only supervised cross entropy |
| [T02](#t02) | placeholder | [training/trainer.py](../src/bifocalgemma/training/trainer.py) | Profiled PEFT optimizer and training loop |
| [T03](#t03) | placeholder | [training/profiling.py](../src/bifocalgemma/training/profiling.py) | Actual memory and latency measurements |
| [T04](#t04) | optional_placeholder | [training/distillation.py](../src/bifocalgemma/training/distillation.py) | Optional later teacher losses |
| [I01](#i01) | placeholder | [inference/generation.py](../src/bifocalgemma/inference/generation.py) | IT-aware generation and stopping |
| [A01](#a01) | placeholder | [agent/loop.py](../src/bifocalgemma/agent/loop.py) | Tool parsing and bounded developer-agent loop |
| [E01](#e01) | placeholder | [evaluation/harness.py](../src/bifocalgemma/evaluation/harness.py) | Real Docker repair evaluation |
| [E02](#e02) | implemented | [evaluation/predictions.py](../src/bifocalgemma/evaluation/predictions.py) | SWE-bench prediction JSONL serialization |
| [S01](#s01) | placeholder | [checkpointing.py](../src/bifocalgemma/checkpointing.py) | Alias-safe checkpoint save/load |

## Urutan penyelesaian

1. C01/D01/D02/M01/M02/M03/T01: reference offline dan kontrak yang tersedia.
2. M04/M05: native tiny integration, tanpa memuat full pretrained weights lokal.
3. D03/D04/T03: tokenizer/data riil dan profiling pada hardware yang sesuai.
4. T02/S01: direct SFT, checkpoint dan resume; mulai pilot E2B–E2B lalu empat pasangan.
5. I01/M06/A01/E01: generation, caches, tools dan real repair evaluation.
6. Analisis keenam model dan optional ablations yang budget/protocol-nya jelas.

D05 dan T04 bersifat optional. E02 tersedia lebih awal untuk record prediction.

## Cara mencoba

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
python -m bifocalgemma inspect --experiment BifocalGemma-E2B-E4B
python -m bifocalgemma validate-config
python -m bifocalgemma smoke --experiment all --device cpu
python -m unittest discover -s tests
```

train/evaluate belum berjalan sebagai production path: exit 2 dengan ID dan link ini.
Tidak ada skor/checkpoint dummy atau fallback model synthetic pada factory native.

## Detail tiap blok

<a id="c01"></a>

### C01 — Six-model specification and checkpoint provenance

**Status:** implemented. **Source:** [config.py](../src/bifocalgemma/config.py).

- Dependencies: none.
- Contract: get_experiment(name), validate_experiment_matrix(path) → pinned ExperimentSpec; no network.
- Invariants: Canonical six IDs, base/IT role/revision, direct_sft, distillation=false, symmetric-only cross word tying.
- Acceptance/readiness: Registry and JSON agree; unknown IDs/config policy mismatch rejected. See CLI inspect/validate-config.

<a id="d01"></a>

### D01 — Repository, example, batch and prediction contracts

**Status:** implemented. **Source:** [contracts.py](../src/bifocalgemma/contracts.py).

- Dependencies: C01 for model-specific use.
- Contract: Repository/RepairExample/TokenizedRepairExample/Seq2SeqBatch/Prediction/Report dataclasses.
- Invariants: Pre-fix snapshot, relative paths, gold target separate, Long IDs/labels and Bool masks.
- Acceptance/readiness: Adapters validate raw records; contract type hints alone are not complete source-data validation.

<a id="d02"></a>

### D02 — Pretokenized teacher-forcing batches

**Status:** implemented. **Source:** [data/collator.py](../src/bifocalgemma/data/collator.py).

- Dependencies: D01.
- Contract: RepairCollator(list[TokenizedRepairExample]) → left-padded Seq2SeqBatch.
- Invariants: Prefix+target shifted once; prefix/pad -100; last prefix predicts first target; decoder-only batches have empty encoder arrays.
- Acceptance/readiness: Target shift/padding/position tests and zero prefix gradient pass; no tokenizer or dataset fetching.

<a id="d03"></a>

### D03 — Pinned base/IT tokenizers and formatting

**Status:** placeholder. **Source:** [data/tokenization.py](../src/bifocalgemma/data/tokenization.py).

- Dependencies: C01, D01.
- Contract: tokenize_repair_example(example,spec,tokenizer_cache) → TokenizedRepairExample.
- Invariants: Pinned base BOS once, IT Jinja prefix, target closing, deterministic context selection and original offsets.
- Acceptance/readiness: Cached tokenizer/fixture round-trip, BOS/stop/template tests, no target in context, truncation manifest. Formatter exists; loader/tokenization still fail.

<a id="d04"></a>

### D04 — Training dataset adapters and overlap audit

**Status:** placeholder. **Source:** [data/adapters.py](../src/bifocalgemma/data/adapters.py).

- Dependencies: D01.
- Contract: SWESmithAdapter/CommitPackFTAdapter/SWEGymAdapter.iter_examples(manifest,split) → records.
- Invariants: Source revision, base_commit reconstruction, license/filter manifest and overlap exclusions.
- Acceptance/readiness: Small verified real fixture reconstructs before/after patch and split audit; task/commit IDs not mixed across splits.

<a id="d05"></a>

### D05 — Tree-sitter features aligned to original token offsets

**Status:** placeholder. **Source:** [data/ast.py](../src/bifocalgemma/data/ast.py).

- Dependencies: D03, D04.
- Contract: extract_token_ast_features(snapshot,original_offsets) → node types/depths per token.
- Invariants: Byte/character alignment, pinned grammar, special/pad handling; no implicit GNN or graph edges.
- Acceptance/readiness: Unicode/indentation/function-boundary cases align to original source; AST off by default until matched ablation.

<a id="m01"></a>

### M01 — Joint softmax, GQA and asymmetric projection

**Status:** reference. **Source:** [models/merged_attention.py](../src/bifocalgemma/models/merged_attention.py).

- Dependencies: D01/D02 at assembly.
- Contract: ReferenceMergedAttention(X,H,encoder_valid,decoder_valid) → [B,Ldec,Ddec].
- Invariants: One joint softmax, shared K/V, GQA, H alignment and branch-specific visibility.
- Acceptance/readiness: Joint competition, future-token isolation, encoder influence, all-masked finite backward, and local-self/global-cross tests pass. Native RoPE/PLE/cache absent.

<a id="m02"></a>

### M02 — Symmetric word parameter sharing

**Status:** implemented. **Source:** [models/embeddings.py](../src/bifocalgemma/models/embeddings.py).

- Dependencies: C01.
- Contract: apply_word_embedding_policy(encoder_words,decoder_words,lm_head,tie_encoder_decoder=...) → policy summary.
- Invariants: Symmetric parameter alias anchored to decoder; internal decoder head tie; PLE not passed/shared.
- Acceptance/readiness: Parameter identity, decoder anchor retention, dual gradient accumulation and optimizer dedup are verified. Native device/reload alias tests required at S01.

<a id="m03"></a>

### M03 — Tiny offline assembly and backward

**Status:** reference. **Source:** [models/reference.py](../src/bifocalgemma/models/reference.py).

- Dependencies: D02, M01, M02, T01.
- Contract: TinyReferenceModel.forward(batch) → ModelOutput; run_reference_smoke(spec,device) → synthetic report.
- Invariants: Generic one-layer encoder + reference merged layer, reduced widths/vocabulary, no pretrained or scores.
- Acceptance/readiness: Four pairs finite forward/backward, shared flags match policy, output explicitly synthetic. It is not a native Gemma replacement.

<a id="m04"></a>

### M04 — Base/IT weight mapping into Gemma layers

**Status:** placeholder. **Source:** [models/initialization.py](../src/bifocalgemma/models/initialization.py).

- Dependencies: C01, M02.
- Contract: describe_initialization(spec) → provenance plan; initialize_pretrained_components(spec) → InitializedComponents(encoder, decoder, H adapter, provenance).
- Invariants: Encoder body/PLE base, decoder IT, symmetric ordinary words IT; native norm/head/RoPE/MLP and asymmetric new adapter.
- Acceptance/readiness: Explicit source mapping and missing/new tensor report, source hashes, no implicit download, meta size validation; cannot mark ready from describe-only metadata.

<a id="m05"></a>

### M05 — Full Gemma encoder/merged decoder integration

**Status:** placeholder. **Source:** [models/model.py](../src/bifocalgemma/models/model.py).

- Dependencies: M04, M01/M02 semantics.
- Contract: BifocalGemmaModel.forward(batch) → logits/loss; encode_context/decode_logits → native tensor operations.
- Invariants: Replace every decoder attention while preserving Gemma residual/FFN/PLE and same-forward KV sharing.
- Acceptance/readiness: Tiny native output/gradient/mask/position/PLE tests; empty encoder matches mapped decoder; all four geometry cases. Eager/SDPA first: naive Flash sliding window must not truncate encoder keys.

<a id="m06"></a>

### M06 — Production self/cross KV and repository cache

**Status:** placeholder. **Source:** [models/cache.py](../src/bifocalgemma/models/cache.py).

- Dependencies: M05.
- Contract: RepositoryCache.get_or_encode(context_fingerprint) → EncoderMemory; future decoder cache wiring.
- Invariants: Frozen inference only, full context/model/tokenizer/AST fingerprint, static cross-KV and dynamic self-KV ownership.
- Acceptance/readiness: Prefill/incremental parity beyond window, shared-layer ownership, stale-context invalidation, actual byte/latency measurements.

<a id="t01"></a>

### T01 — Patch-only supervised cross entropy

**Status:** implemented. **Source:** [training/objectives.py](../src/bifocalgemma/training/objectives.py).

- Dependencies: D02.
- Contract: patch_cross_entropy(logits,aligned_labels) → scalar loss.
- Invariants: No second shift; ignored prefix/pad; all-ignored differentiable zero.
- Acceptance/readiness: Correct target gradient and finite ignored loss pass; does not establish training convergence.

<a id="t02"></a>

### T02 — Profiled PEFT optimizer and training loop

**Status:** placeholder. **Source:** [training/trainer.py](../src/bifocalgemma/training/trainer.py).

- Dependencies: D03/D04, M04/M05, T03, T01, S01 for resume.
- Contract: train_direct_sft(spec,data_manifest,SFTConfig,output_dir) → future training artifacts.
- Invariants: No required teacher; profiled precision/budget/PEFT, unique parameters, recorded seed/data/checkpoint.
- Acceptance/readiness: Small real overfit/resume and held-out retention tests; nonzero trainable scopes; actual hardware profile. Current CLI fail-fast T02 is intentional.

<a id="t03"></a>

### T03 — Actual memory and latency measurements

**Status:** placeholder. **Source:** [training/profiling.py](../src/bifocalgemma/training/profiling.py).

- Dependencies: M04/M05, D02.
- Contract: profile_experiment(spec,context_tokens,target_tokens) → measured profile.
- Invariants: Weights/optimizer/activation/self-KV/cross-KV/cold-warm latency separate, measured not extrapolated.
- Acceptance/readiness: Reproducible hardware/profile artifact with allocator peaks and token lengths; null training config resolved from measurements.

<a id="t04"></a>

### T04 — Optional later teacher losses

**Status:** optional_placeholder. **Source:** [training/distillation.py](../src/bifocalgemma/training/distillation.py).

- Dependencies: only a future optional research design.
- Contract: distillation_loss(student_logits,teacher_logits) → future scalar loss.
- Invariants: Alignment, masks, teacher freeze, temperature and forward/storage cost declared.
- Acceptance/readiness: Justified retention/quality question, matched control and compute accounting. Not imported by default trainer.

<a id="i01"></a>

### I01 — IT-aware generation and stopping

**Status:** placeholder. **Source:** [inference/generation.py](../src/bifocalgemma/inference/generation.py).

- Dependencies: D03, M05, S01; M06 for cached performance.
- Contract: generate_patch(spec,example,max_new_tokens) → RepairPrediction.
- Invariants: IT prefix/EOS, no gold target, patch/thinking/tool boundaries, generation budget.
- Acceptance/readiness: First uncached tiny generation then cached parity, termination/handoff/parser tests; no fabricated patches from placeholder.

<a id="a01"></a>

### A01 — Tool parsing and bounded developer-agent loop

**Status:** placeholder. **Source:** [agent/loop.py](../src/bifocalgemma/agent/loop.py).

- Dependencies: I01, M06 as applicable.
- Contract: DeveloperAgent.run(example,max_steps) → traced prediction.
- Invariants: Read/search/edit/test with workspace bounds, budgets, timeouts and full context refresh.
- Acceptance/readiness: Recorded deterministic toy tool trace then real isolated tasks; same allowed tools across six models.

<a id="e01"></a>

### E01 — Real Docker repair evaluation

**Status:** placeholder. **Source:** [evaluation/harness.py](../src/bifocalgemma/evaluation/harness.py).

- Dependencies: I01/E02 predictions; A01 if agent used.
- Contract: SWEBenchHarness.evaluate(path,dataset_revision,run_id) → EvaluationReport.
- Invariants: Docker environment, FAIL_TO_PASS/PASS_TO_PASS, fixed task denominator and infrastructure errors.
- Acceptance/readiness: Pinned tiny real harness fixture/report, distinct run IDs, valid failure accounting; reporting table cannot certify harness readiness.

<a id="e02"></a>

### E02 — SWE-bench prediction JSONL serialization

**Status:** implemented. **Source:** [evaluation/predictions.py](../src/bifocalgemma/evaluation/predictions.py).

- Dependencies: D01.
- Contract: write_swebench_predictions(records,path) → official-shaped JSONL.
- Invariants: instance_id/model_name_or_path/model_patch only; no duplicate task IDs or gold metadata.
- Acceptance/readiness: Wire shape/UTF-8 tests; serializer does not grade correctness.

<a id="s01"></a>

### S01 — Alias-safe checkpoint save/load

**Status:** placeholder. **Source:** [checkpointing.py](../src/bifocalgemma/checkpointing.py).

- Dependencies: M04/M05/M02, T02 config.
- Contract: save_checkpoint(model,path), restore_checkpoint(path) → future alias-safe state.
- Invariants: Model/config/source/tokenizer/data/version and optimizer state; rebuild ties before resume.
- Acceptance/readiness: Save/reload logits parity, tied aliases preserved, PLE separate, deterministic resume and incompatible version rejection.

## Integrasi docs dan kode

FeatureNotReadyError dari scaffold.py selalu menyebut ID dan section guide ini.
Saat blok selesai: implement acceptance, jalankan checks relevan, update status registry,
kontrak/caller/docs serta VALIDATION. Source path harus tetap ada; placeholder tidak
diganti dengan pass/return zero/empty dataset yang terlihat seperti hasil berhasil.

Dokumen detail: [architecture](ARCHITECTURE.md), [data](DATA_PIPELINE.md),
[training](TRAINING.md), [evaluation](EVALUATION.md).
