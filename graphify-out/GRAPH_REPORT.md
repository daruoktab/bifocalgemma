# Graph Report - bifocalgemma  (2026-11-03)

## Corpus Check
- 262 files · ~337,494 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: .ipynb 3, (none) 2, .jsonl 2)

## Summary
- 1934 nodes · 5915 edges · 101 communities (75 shown, 26 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 588 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a2c0428a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- torch
- test_editing_manifest.py
- remediation.py
- prepare_colab_pilot.py
- releases.py
- FeatureNotReadyError
- ExperimentSpec
- atomic_json
- shared_execution.py
- stages.py
- memory_probe.py
- subprocess
- test_repository_corpus.py
- v2.py
- json
- CodeOverlapIndex
- test_gsm8k.py
- tokenization.py
- config.py
- test_shared_execution.py
- telemetry.py
- run_sequential_pipeline
- RepositorySnapshot
- configure_memory_efficient_training
- test_molab_pilot.py
- Tensor
- stable_fingerprint
- memory_efficient.py
- gsm8k.py
- zipfile
- repository_study.py
- test_molab_matrix_notebooks.py
- TokenizedRepairExample
- repository_sft.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- build_native_tiny_model
- molab_matrix_support.py
- workflow
- repository_corpus.py
- NativeTinyModel
- probe
- test_completed_benchmark_with_pending_upload_resumes_without_model_load
- curation.py
- examples
- test_repository_study.py
- data/context.py
- ast
- build_repository_corpus
- fit_offline_sft
- model.py
- packed_tokens.py
- load_verified_tokenizer
- TestTrainingPolicy
- test_snapshot_verifies_replacement_before_pruning
- data/offline_sft.py
- GatedCrossAttentionBridge
- test_context_benchmark_resume.py
- policy.py
- patch_matrix_tokenization
- pathlib
- adapters.py
- TestBifocalGemmaArchitecture
- prepare_colab_probe.py
- RepoContextRouter
- contracts.py
- TestGemma4EdgeTopology
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- test_data_remediation.py
- test_notebook_hf_flow.py
- optimizers.py
- get_v2_experiment
- load_local_native_model
- ASTGraphProjector
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- context_tasks.py
- GitSnapshotStore
- TestNotebookCellParity
- SwiGLUDualGateBridge
- hashlib
- native_python.py
- CachedAttention
- evaluate_sft_loss
- tempfile
- bifocalgemma
- configure_full_rank_cross_attention
- initialization.py
- harness.py
- test_complete_epoch_training_budget_all_families
- NativeMergedAttention
- run_length_grid
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- build_bifocalgemma_v3_optimizer
- TextTokenizer
- test_profile_publish_parent_guard_keeps_concurrent_winner

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 79 edges
2. `stable_fingerprint()` - 62 edges
3. `TokenizedRepairExample` - 49 edges
4. `load_verified_tokenizer()` - 49 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `configure_memory_efficient_training()` - 42 edges
8. `build_native_tiny_model()` - 41 edges
9. `FeatureNotReadyError` - 39 edges
10. `RepositorySnapshot` - 38 edges

## Surprising Connections (you probably didn't know these)
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `publish()` --indirect_call--> `profile()`  [INFERRED]
  tests/test_shared_execution.py → tools/profile_repository_lengths.py
- `make_experiment_spec()` --uses--> `CheckpointRef`  [INFERRED]
  tools/molab_matrix_support.py → src/bifocalgemma/config.py
- `make_experiment_spec()` --uses--> `ExperimentSpec`  [INFERRED]
  tools/molab_matrix_support.py → src/bifocalgemma/config.py
- `run_batched_context_benchmark()` --uses--> `ExperimentSpec`  [INFERRED]
  tools/molab_matrix_support.py → src/bifocalgemma/config.py

## Import Cycles
- None detected.

## Communities (101 total, 26 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.06
Nodes (22): test_exported_worker_has_parity_context_and_exact_adapter_checks(), pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install() (+14 more)

### Community 1 - "torch"
Cohesion: 0.07
Nodes (6): apply_word_embedding_policy(), EmbeddingPolicyResult, build_native_model(), chunked_linear_cross_entropy(), TestWordEmbeddingPolicy, test_notebook_stress_helpers_match_shared_library()

### Community 2 - "test_editing_manifest.py"
Cohesion: 0.29
Nodes (9): editing_manifest(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation() (+1 more)

### Community 3 - "remediation.py"
Cohesion: 0.11
Nodes (18): export_pool(), read_arrow_rows(), split_connected_records(), token_profile(), build_full_pool(), rows(), audit_rows(), build_remediated_pool() (+10 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.12
Nodes (20): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset(), evaluate_pilot_checkpoint(), extract_resolvable_span() (+12 more)

### Community 5 - "releases.py"
Cohesion: 0.08
Nodes (28): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+20 more)

### Community 6 - "FeatureNotReadyError"
Cohesion: 0.11
Nodes (8): extract_token_ast_features(), TokenASTFeatures, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError, distillation_loss(), profile_experiment()

### Community 7 - "ExperimentSpec"
Cohesion: 0.15
Nodes (9): ExperimentSpec, validate_experiment_matrix(), main(), run_native_smoke(), run_reference_smoke(), TinyReferenceModel, SFTConfig, train_direct_sft() (+1 more)

### Community 8 - "atomic_json"
Cohesion: 0.08
Nodes (17): activity(), BenchmarkUploadQueue, retry_delay(), atomic_json(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden() (+9 more)

### Community 9 - "shared_execution.py"
Cohesion: 0.08
Nodes (34): RepairCollator, __getattr__(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings() (+26 more)

### Community 10 - "stages.py"
Cohesion: 0.11
Nodes (15): file_digest(), checked_prefix(), recover_remote_run(), _synchronize_once(), latest_checkpoint(), retained_checkpoints(), validate_checkpoint_files(), snapshot() (+7 more)

### Community 11 - "memory_probe.py"
Cohesion: 0.26
Nodes (6): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), tensor_bytes(), TestMemoryProbe

### Community 12 - "subprocess"
Cohesion: 0.20
Nodes (3): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), upload_in_chunks()

### Community 13 - "test_repository_corpus.py"
Cohesion: 0.23
Nodes (9): context_license_evidence(), patch_license_evidence(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch(), test_license_document_include_is_resolved_inside_snapshot() (+1 more)

### Community 14 - "v2.py"
Cohesion: 0.11
Nodes (16): source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics(), stratified_subset(), _execute_v2(), save_guard() (+8 more)

### Community 15 - "json"
Cohesion: 0.12
Nodes (15): compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit_pool(), audit_views(), main(), audit(), emit(), main() (+7 more)

### Community 16 - "CodeOverlapIndex"
Cohesion: 0.18
Nodes (20): code_shingles(), code_signature(), CodeOverlapIndex, curate_paired_sources(), nextcoder_row(), build_benchmark_index(), raw(), test_cross_source_duplicates_and_nextcoder_variants() (+12 more)

### Community 17 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 18 - "tokenization.py"
Cohesion: 0.14
Nodes (7): format_encoder_context(), repair_prompt_ids(), cached_logits(), InferenceCache, test_alignment_initialization_with_cuda_default_device(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only()

### Community 19 - "config.py"
Cohesion: 0.10
Nodes (5): BackboneProfile, CheckpointRef, _pair(), _ref(), RoutedRepoBatch

### Community 20 - "test_shared_execution.py"
Cohesion: 0.13
Nodes (6): __getattr__(), patch_cross_entropy(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline()

### Community 21 - "telemetry.py"
Cohesion: 0.09
Nodes (10): format_duration(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard (+2 more)

### Community 22 - "run_sequential_pipeline"
Cohesion: 0.11
Nodes (13): remote_reset_plan(), reset_local_benchmarks(), reset_state(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint() (+5 more)

### Community 23 - "RepositorySnapshot"
Cohesion: 0.11
Nodes (16): RepairExample, RepositoryFile, RepositorySnapshot, example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash() (+8 more)

### Community 24 - "configure_memory_efficient_training"
Cohesion: 0.18
Nodes (19): configure_memory_efficient_training(), fit_curated_sft(), configure_policy_model(), default_training_profiles(), unified_recipe(), fit_repository_sft(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence() (+11 more)

### Community 25 - "test_molab_pilot.py"
Cohesion: 0.09
Nodes (19): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only() (+11 more)

### Community 26 - "Tensor"
Cohesion: 0.12
Nodes (11): test_v3_streaming_pointer_loss_and_differential_diff_weighting(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), forward(), __init__(), differential_diff_cross_entropy(), _pointer_chunk_forward() (+3 more)

### Community 27 - "stable_fingerprint"
Cohesion: 0.17
Nodes (13): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), audit_repository_sources(), fetch_benchmark_files() (+5 more)

### Community 28 - "memory_efficient.py"
Cohesion: 0.06
Nodes (32): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), run_training_memory_preflight(), tiled_attention(), _TiledAttention (+24 more)

### Community 29 - "gsm8k.py"
Cohesion: 0.19
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 30 - "zipfile"
Cohesion: 0.18
Nodes (11): test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render(), main() (+3 more)

### Community 31 - "repository_study.py"
Cohesion: 0.22
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), generate_patch(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (15): test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_pointer_loss_gradient_checkpointing_and_trivial_filtering(), test_prepare_sampled_benchmark_collection_deterministic_100_cases() (+7 more)

### Community 34 - "TokenizedRepairExample"
Cohesion: 0.20
Nodes (4): TokenizedRepairExample, synthetic_stress_target(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 35 - "repository_sft.py"
Cohesion: 0.24
Nodes (7): prepare_editing_records(), split_editing_records(), prepare_curated_training(), tokenize(), prepare_repository_training(), tokenize_repository_record(), TestOfflineEditing

### Community 37 - "build_native_tiny_model"
Cohesion: 0.17
Nodes (5): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head(), test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory()

### Community 38 - "molab_matrix_support.py"
Cohesion: 0.08
Nodes (21): test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight(), batched_greedy_decode(), clamped_triton_kernel_options(), configure_full_rank_cross_attention(), encode_comprehension_prefix(), ensure_hardware_memory_guards(), infer_benchmark_weight_tier(), is_cross_bridge_parameter() (+13 more)

### Community 39 - "workflow"
Cohesion: 0.12
Nodes (11): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), workflow(), benchmark(), download() (+3 more)

### Community 40 - "repository_corpus.py"
Cohesion: 0.14
Nodes (15): apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), patch_paths(), safe_repo_path(), path_license_evidence(), read_text_archive(), recognize_license() (+7 more)

### Community 41 - "NativeTinyModel"
Cohesion: 0.13
Nodes (4): ModelOutput, Seq2SeqBatch, NormalizedAlignment, NativeTinyModel

### Community 42 - "probe"
Cohesion: 0.11
Nodes (14): native_config(), representative_cohort(), shared_contract(), append_event(), download_model(), progress(), maximum_probe(), emit() (+6 more)

### Community 43 - "test_completed_benchmark_with_pending_upload_resumes_without_model_load"
Cohesion: 0.17
Nodes (12): comparison_recipes(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume(), benchmark() (+4 more)

### Community 44 - "curation.py"
Cohesion: 0.14
Nodes (10): code_tokens(), CodeOverlapMatch, count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_curated_warning_target_retained_and_summary_distinguishes_duplicate(), test_errors_are_still_rejected_and_other_warnings_still_visible() (+2 more)

### Community 45 - "examples"
Cohesion: 0.09
Nodes (18): TestTokenizerVerification, examples(), TestOfflineSFT, prepare(), calibration(), guard(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), prepare() (+10 more)

### Community 46 - "test_repository_study.py"
Cohesion: 0.13
Nodes (16): prepare_repository_record(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions, ScriptedModel (+8 more)

### Community 47 - "data/context.py"
Cohesion: 0.48
Nodes (4): ContextWindow, render_window(), select_repository_context(), _terms()

### Community 48 - "ast"
Cohesion: 0.22
Nodes (7): test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), main(), module_definitions(), render(), use_embedded_imports(), main(), render_eval_script()

### Community 49 - "build_repository_corpus"
Cohesion: 0.16
Nodes (6): ArchiveSnapshotStore, build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_failed_download_removes_partial_bytes(), test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 50 - "fit_offline_sft"
Cohesion: 0.23
Nodes (10): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest() (+2 more)

### Community 51 - "model.py"
Cohesion: 0.15
Nodes (4): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel, build_pretrained_model()

### Community 52 - "packed_tokens.py"
Cohesion: 0.23
Nodes (5): build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 53 - "load_verified_tokenizer"
Cohesion: 0.10
Nodes (24): read_source_rows(), decode_text(), load_verified_tokenizer(), completion_metrics(), decoder_prompt(), greedy_decode(), prompt_ids(), run_context_benchmark() (+16 more)

### Community 55 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "data/offline_sft.py"
Cohesion: 0.14
Nodes (14): normalize_context_task(), curated_pool_digest(), load_curated_pool(), validate_context_license_record(), test_context_license_record_covers_all_visible_paths(), code_fragment_signatures(), completion_from_snapshot(), context_pool() (+6 more)

### Community 58 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 59 - "policy.py"
Cohesion: 0.12
Nodes (9): ProbeLoRALinear, build_training_groups(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence(), run_recipe_comparison(), test_inference_v2_adapter_roundtrip_and_baseline_group() (+1 more)

### Community 60 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract(), format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task()

### Community 61 - "pathlib"
Cohesion: 0.10
Nodes (7): TinyLossModel, test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_training_resume_keeps_elapsed_from_step_five_through_ten(), test_training_failure_never_launches_benchmarks(), load_runtime()

### Community 62 - "adapters.py"
Cohesion: 0.23
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 64 - "prepare_colab_probe.py"
Cohesion: 0.36
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 66 - "contracts.py"
Cohesion: 0.16
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 68 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.22
Nodes (4): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412(), synchronize_matrix_v3()

### Community 70 - "test_data_remediation.py"
Cohesion: 0.26
Nodes (12): merge_lineage(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage() (+4 more)

### Community 71 - "test_notebook_hf_flow.py"
Cohesion: 0.24
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 72 - "optimizers.py"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 73 - "get_v2_experiment"
Cohesion: 0.13
Nodes (8): get_v2_experiment(), _DummyTokenizer, test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_alignment_stays_fp32_after_bf16_conversion(), test_encoder_bidirectional_window_boundary_matches_native(), test_v3_wall_time_checkpoint_resume_is_exact(), make()

### Community 74 - "load_local_native_model"
Cohesion: 0.31
Nodes (4): transfer_text_checkpoint(), load_local_native_model(), TestCheckpointTransfer, test_local_loader_preflights_and_preserves_shared_words()

### Community 77 - "context_tasks.py"
Cohesion: 0.19
Nodes (7): repair_preimage_coverage(), task_prompt(), tokenize_context_task(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context()

### Community 81 - "hashlib"
Cohesion: 0.31
Nodes (7): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), repository_editing_record()

### Community 82 - "native_python.py"
Cohesion: 0.36
Nodes (6): canonical_snapshot_hash(), NativeTaskResult, score_python_patch(), summarize_native_results(), test_native_score_pass_fail_disabled_and_runtime(), test_unittest_results_and_early_exit_do_not_false_pass()

### Community 84 - "evaluate_sft_loss"
Cohesion: 0.33
Nodes (3): evaluate_sft_loss(), validation_batches(), evaluate_source_copy_probes()

### Community 85 - "tempfile"
Cohesion: 0.31
Nodes (3): main(), main(), run_cmd()

### Community 92 - "initialization.py"
Cohesion: 0.32
Nodes (4): describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents

### Community 94 - "test_complete_epoch_training_budget_all_families"
Cohesion: 0.40
Nodes (4): validate_curated_pool_limits(), test_complete_epoch_training_budget_all_families(), test_frozen_budget_matches_all_six_controls(), test_invalid_requested_pool_budget_is_rejected()

### Community 96 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 674 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `prepare_colab_pilot.py`, `shared_execution.py`, `stages.py`, `v2.py`, `json`, `run_sequential_pipeline`, `stable_fingerprint`, `test_molab_matrix_notebooks.py`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `molab_matrix_support.py`, `workflow`, `probe`, `test_completed_benchmark_with_pending_upload_resumes_without_model_load`, `examples`, `fit_offline_sft`, `packed_tokens.py`, `load_verified_tokenizer`, `test_snapshot_verifies_replacement_before_pruning`, `data/offline_sft.py`, `test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06448979591836734 - nodes in this community are weakly interconnected._
- **Why does `load_verified_tokenizer()` connect `load_verified_tokenizer` to `remediation.py`, `prepare_colab_pilot.py`, `v2.py`, `json`, `test_gsm8k.py`, `tokenization.py`, `config.py`, `RepositorySnapshot`, `configure_memory_efficient_training`, `stable_fingerprint`, `gsm8k.py`, `repository_study.py`, `repository_sft.py`, `molab_matrix_support.py`, `repository_corpus.py`, `probe`, `examples`, `build_repository_corpus`, `data/offline_sft.py`, `hashlib`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `torch` be split into smaller, more focused modules?**
  _Cohesion score 0.06988120195667366 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `test_notebook_dependencies.py`, `torch`, `test_editing_manifest.py`, `prepare_colab_pilot.py`, `ExperimentSpec`, `shared_execution.py`, `memory_probe.py`, `v2.py`, `CodeOverlapIndex`, `tokenization.py`, `test_shared_execution.py`, `RepositorySnapshot`, `configure_memory_efficient_training`, `test_molab_pilot.py`, `memory_efficient.py`, `test_molab_matrix_notebooks.py`, `repository_sft.py`, `molab_matrix_support.py`, `probe`, `examples`, `packed_tokens.py`, `patch_matrix_tokenization`, `prepare_colab_probe.py`, `contracts.py`, `context_tasks.py`, `test_complete_epoch_training_budget_all_families`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._