# Graph Report - bifocalgemma  (2026-11-10)

## Corpus Check
- 290 files · ~365,393 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 18 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2378 nodes · 6970 edges · 127 communities (82 shown, 45 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 707 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5f9230ce`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- curation.py
- run_sequential_pipeline
- prepare_colab_pilot.py
- releases.py
- __main__.py
- test_training_recipes.py
- test_shared_execution.py
- RepositorySnapshot
- _compute_per_row_cross_query
- test_gsm8k.py
- subprocess
- save_matrix_checkpoint
- os
- NativeTinyModel
- tempfile
- shared_execution.py
- FeatureNotReadyError
- molab_pilot_support.py
- calibrate_shared_profile
- json
- test_tier5_adversarial.py
- test_repository_study.py
- CodeOverlapIndex
- BifocalGemmaConfig
- test_v2_lifecycle.py
- torch
- FastTrainingPipeline
- hashlib
- Seq2SeqBatch
- test_molab_matrix_notebooks.py
- v3/objectives.py
- prepare_colab_probe.py
- TokenizedRepairExample
- repository_corpus.py
- pytest
- adversarial_r3_probe.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- .from_pretrained_backbones
- TestGemma4EdgeTopology
- data/offline_sft.py
- PackedTokenizedDataset
- test_molab_remediation.py
- test_colab_pilot.py
- BifocalGemmaConfig
- matrix.py
- profile_repository_lengths.py
- default_training_profiles
- repository_study.py
- gsm8k.py
- native.py
- typing
- test_memory_efficient_training.py
- adapters.py
- test_benchmark_prompt_budget.py
- math
- v2_preflight.py
- test_repository_corpus.py
- test_v2_fixed_execution.py
- fit_offline_sft
- TestBifocalGemmaArchitecture
- configure_memory_efficient_training
- build_native_tiny_model
- tokenization.py
- examples
- test_training_telemetry_v2.py
- stages.py
- run_length_grid
- v2.py
- optimizers.py
- test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract
- test_adversarial_reponiah_depth_and_ratio_extremes
- TestNotebookCellParity
- SwiGLUDualGateBridge
- resolve_optimal_eval_batch_size
- test_editing_manifest.py
- configuration_bifocalgemma.py
- matrix.py
- config.py
- tier5
- BifocalGemmaForCausalLM
- bifocalgemma
- apply_word_embedding_policy
- test_context_tasks.py
- evaluation/context.py
- v2/bridge.py
- pretrained.py
- telemetry.py
- workflow
- tokenize_repair_example
- harness.py
- data/context.py
- test_v2_workflow.py
- GitSnapshotStore
- RepositoryPreparationLedger
- strict_letter
- audit_dataset_release.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- prepare_colab_eval.py
- get_v2_experiment
- test_adversarial_mcq_adverb_preambles
- test_adversarial_mcq_lowercase_letter_normalization
- test_adversarial_metrics_keyword_identifier_inflation
- test_adversarial_metrics_identical_sequences
- test_adversarial_metrics_unicode_identifiers_blind_spot
- test_adversarial_metrics_pure_operator_zero_division_guard
- test_adversarial_mcq_isolated_formatting
- test_adversarial_mcq_ambiguity_guards
- test_adversarial_mcq_verbose_reasoning_preambles

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 83 edges
2. `stable_fingerprint()` - 65 edges
3. `load_verified_tokenizer()` - 55 edges
4. `TokenizedRepairExample` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `RepairCollator` - 40 edges
10. `run_matrix_training()` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/source_identity.py
- `publish()` --indirect_call--> `profile()`  [INFERRED]
  tests/test_shared_execution.py → tools/profile_repository_lengths.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (127 total, 45 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "remediation.py"
Cohesion: 0.11
Nodes (24): split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+16 more)

### Community 2 - "curation.py"
Cohesion: 0.16
Nodes (10): read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+2 more)

### Community 3 - "run_sequential_pipeline"
Cohesion: 0.12
Nodes (13): load_sft_adapters_for_inference(), remote_reset_plan(), reset_local_benchmarks(), reset_state(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress() (+5 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.11
Nodes (23): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset() (+15 more)

### Community 5 - "releases.py"
Cohesion: 0.09
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 6 - "__main__.py"
Cohesion: 0.11
Nodes (10): ExperimentSpec, validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents, SFTConfig (+2 more)

### Community 7 - "test_training_recipes.py"
Cohesion: 0.15
Nodes (13): comparison_recipes(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume(), benchmark() (+5 more)

### Community 8 - "test_shared_execution.py"
Cohesion: 0.10
Nodes (14): seal_report(), save(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend() (+6 more)

### Community 9 - "RepositorySnapshot"
Cohesion: 0.15
Nodes (12): RepositoryFile, RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, test_dependency_completion_removes_target_and_retains_real_helper() (+4 more)

### Community 10 - "_compute_per_row_cross_query"
Cohesion: 0.24
Nodes (3): _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 12 - "subprocess"
Cohesion: 0.08
Nodes (9): probe_prompt_neutrality_scan(), probe_unicode_and_metrics(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), main(), run_cmd() (+1 more)

### Community 13 - "save_matrix_checkpoint"
Cohesion: 0.16
Nodes (7): Bf16FullLinear, load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix_v3()

### Community 14 - "os"
Cohesion: 0.17
Nodes (7): ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 16 - "tempfile"
Cohesion: 0.17
Nodes (11): canonical_snapshot_hash(), apply_patch_to_snapshot(), extract_unified_diff(), example_from_record(), NativeTaskResult, score_python_patch(), summarize_native_results(), patch_syntax() (+3 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.23
Nodes (9): candidate_settings(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), verify_report(), check_evidence() (+1 more)

### Community 18 - "FeatureNotReadyError"
Cohesion: 0.11
Nodes (8): extract_token_ast_features(), TokenASTFeatures, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError, distillation_loss(), profile_experiment()

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.09
Nodes (16): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), test_upload_retry_verification_and_base_prefix_only(), version(), digest() (+8 more)

### Community 20 - "calibrate_shared_profile"
Cohesion: 0.17
Nodes (10): benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), error_kind(), numerical_probe(), probe_optimizer() (+2 more)

### Community 21 - "json"
Cohesion: 0.10
Nodes (27): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), build_packed_cache() (+19 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.18
Nodes (4): test_adversarial_mcq_ruling_out_distractors_vulnerability(), test_adversarial_metrics_disjoint_identifiers(), test_adversarial_metrics_multiline_diff_inflation(), test_adversarial_prompt_neutrality_matrix()

### Community 23 - "test_repository_study.py"
Cohesion: 0.08
Nodes (22): prepare_repository_record(), prepare_repository_record_with_snapshot(), RepositoryTask, authoritative_empirical_results(), project_root(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row() (+14 more)

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.08
Nodes (35): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+27 more)

### Community 25 - "BifocalGemmaConfig"
Cohesion: 0.13
Nodes (13): export_matrix_pretrained_model(), FastTrainingConfig, test_fast_training_config_and_pipeline_baseline_single_stage(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_save_and_from_pretrained_roundtrip(), test_enable_memory_efficient_training_and_softcapping(), test_export_matrix_pretrained_model(), test_export_matrix_pretrained_model_baseline() (+5 more)

### Community 27 - "test_v2_lifecycle.py"
Cohesion: 0.07
Nodes (12): retry_delay(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_retry_after_date_and_network_failures() (+4 more)

### Community 28 - "torch"
Cohesion: 0.11
Nodes (16): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+8 more)

### Community 30 - "hashlib"
Cohesion: 0.09
Nodes (26): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), source_fingerprints(), WithoutDocstrings (+18 more)

### Community 31 - "Seq2SeqBatch"
Cohesion: 0.13
Nodes (7): restore_checkpoint(), save_checkpoint(), ModelOutput, Seq2SeqBatch, BifocalGemmaModel, build_pretrained_model(), TinyReferenceModel

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (6): test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_telemetry_history_appends_all_active_stages()

### Community 34 - "v3/objectives.py"
Cohesion: 0.09
Nodes (12): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.43
Nodes (3): declared_limit_example(), main(), render()

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.07
Nodes (12): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), run_reference_smoke(), __getattr__(), synthetic_stress_target(), patch_cross_entropy() (+4 more)

### Community 37 - "repository_corpus.py"
Cohesion: 0.17
Nodes (10): patch_paths(), safe_repo_path(), context_license_evidence(), path_license_evidence(), prepare_repository_corpus(), recognize_license(), resolved_license(), repo_identity() (+2 more)

### Community 38 - "pytest"
Cohesion: 0.04
Nodes (29): completion_metrics(), pytest_configure(), reference_resilient_mcq_parser(), validate_empirical_results_schema(), validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template() (+21 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.20
Nodes (5): extract_mcq_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_redos_catastrophic_backtracking(), verify_gate_round_2_resolved_cases()

### Community 41 - ".from_pretrained_backbones"
Cohesion: 0.13
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 43 - "data/offline_sft.py"
Cohesion: 0.13
Nodes (14): normalize_context_task(), repair_preimage_coverage(), task_prompt(), tokenize_context_task(), curated_pool_digest(), load_curated_pool(), prepare_editing_records(), split_editing_records() (+6 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "test_colab_pilot.py"
Cohesion: 0.11
Nodes (6): pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), _prepare(), prepare_runtime()

### Community 47 - "BifocalGemmaConfig"
Cohesion: 0.13
Nodes (8): BifocalGemmaConfig, _resolve_backbone_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 48 - "matrix.py"
Cohesion: 0.13
Nodes (11): CheckpointRef, evaluate_sft_loss(), validation_batches(), update_policy_learning_rates(), benchmark_directory_name(), build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example() (+3 more)

### Community 49 - "profile_repository_lengths.py"
Cohesion: 0.38
Nodes (3): distribution(), main(), profile()

### Community 50 - "default_training_profiles"
Cohesion: 0.24
Nodes (3): default_training_profiles(), make_identity(), TestTrainingPolicy

### Community 51 - "repository_study.py"
Cohesion: 0.22
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), generate_diagnostic_text(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 52 - "gsm8k.py"
Cohesion: 0.19
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 53 - "native.py"
Cohesion: 0.07
Nodes (8): NormalizedAlignment, build_joint_visibility(), ReferenceMergedAttention, build_native_model(), native_config(), NativeMergedAttention, run_native_smoke(), TestMergedAttentionReference

### Community 54 - "typing"
Cohesion: 0.06
Nodes (12): ensure_assets_on_disk(), load_asset_json(), CachedAttention, configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group() (+4 more)

### Community 55 - "test_memory_efficient_training.py"
Cohesion: 0.20
Nodes (13): chunked_linear_cross_entropy(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+5 more)

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 58 - "math"
Cohesion: 0.15
Nodes (8): fit_curated_sft(), build_training_groups(), validate_training_policy(), resolve_training_cadence(), run_recipe_comparison(), test_curated_resume_rejects_runtime_drift(), test_inference_v2_adapter_roundtrip_and_baseline_group(), main()

### Community 59 - "v2_preflight.py"
Cohesion: 0.17
Nodes (10): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), select_execution(), emit(), check_followers() (+2 more)

### Community 60 - "test_repository_corpus.py"
Cohesion: 0.22
Nodes (11): validate_context_license_record(), patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch() (+3 more)

### Community 61 - "test_v2_fixed_execution.py"
Cohesion: 0.20
Nodes (11): original(), guard(), examples(), fixture_guard(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), measure() (+3 more)

### Community 62 - "fit_offline_sft"
Cohesion: 0.08
Nodes (12): fit_offline_sft(), emit(), model_stage(), DurableClock, TinyLossModel, test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten() (+4 more)

### Community 64 - "configure_memory_efficient_training"
Cohesion: 0.16
Nodes (19): configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), configure_matrix_stage(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), test_cached_uncached_and_batched_greedy_decode_equivalence() (+11 more)

### Community 65 - "build_native_tiny_model"
Cohesion: 0.16
Nodes (5): build_native_tiny_model(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words()

### Community 66 - "tokenization.py"
Cohesion: 0.13
Nodes (11): DeveloperAgent, RepairExample, RepairPrediction, format_encoder_context(), write_swebench_predictions(), generate_patch(), greedy_decode(), GreedyGeneration (+3 more)

### Community 67 - "examples"
Cohesion: 0.09
Nodes (14): TestTokenizerVerification, examples(), TestOfflineSFT, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), calibration(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight() (+6 more)

### Community 69 - "test_training_telemetry_v2.py"
Cohesion: 0.20
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 70 - "stages.py"
Cohesion: 0.11
Nodes (14): file_digest(), checked_prefix(), recover_remote_run(), _synchronize_once(), retained_checkpoints(), validate_checkpoint_files(), snapshot(), synchronize_preflight() (+6 more)

### Community 72 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 74 - "v2.py"
Cohesion: 0.12
Nodes (19): shared_contract(), latest_checkpoint(), append_event(), benchmark_jobs(), encoder_diagnostics(), stratified_subset(), download_model(), _execute_v2() (+11 more)

### Community 76 - "optimizers.py"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 77 - "test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract"
Cohesion: 0.12
Nodes (7): critical_attention_beta(), format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 81 - "resolve_optimal_eval_batch_size"
Cohesion: 0.40
Nodes (3): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 82 - "test_editing_manifest.py"
Cohesion: 0.07
Nodes (24): editing_manifest(), repository_editing_record(), limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer (+16 more)

### Community 84 - "matrix.py"
Cohesion: 0.14
Nodes (9): BenchmarkUploadQueue, configure_full_rank_cross_attention(), run_matrix_benchmark(), run_matrix_stage(), run_matrix_training(), run_matrix_training_and_benchmark(), progress(), clamped_triton_kernel_options() (+1 more)

### Community 85 - "config.py"
Cohesion: 0.09
Nodes (11): BackboneProfile, _pair(), _ref(), cpu_tree(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), Float32Alignment (+3 more)

### Community 87 - "tier5"
Cohesion: 0.18
Nodes (5): test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_redos_scaling(), test_adversarial_mcq_split_ambiguity_behavior(), test_adversarial_reponiah_deterministic_reproducibility(), test_adversarial_reponiah_scorer_negation_vulnerability()

### Community 91 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 92 - "test_context_tasks.py"
Cohesion: 0.40
Nodes (3): task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline()

### Community 93 - "evaluation/context.py"
Cohesion: 0.07
Nodes (27): decode_text(), build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), prompt_ids() (+19 more)

### Community 94 - "v2/bridge.py"
Cohesion: 0.08
Nodes (4): GatedCrossAttentionBridge, RMSNorm, ASTGraphProjector, RoutedRepoBatch

### Community 96 - "telemetry.py"
Cohesion: 0.09
Nodes (11): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog (+3 more)

### Community 97 - "workflow"
Cohesion: 0.22
Nodes (6): workflow(), benchmark(), download(), execution(), fit(), prepare()

### Community 106 - "data/context.py"
Cohesion: 0.48
Nodes (4): ContextWindow, render_window(), select_repository_context(), _terms()

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.20
Nodes (7): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), run()

### Community 111 - "strict_letter"
Cohesion: 0.29
Nodes (4): fallback_letter_regex(), strict_letter(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 112 - "audit_dataset_release.py"
Cohesion: 0.52
Nodes (4): audit(), emit(), main(), validate_all_patches()

### Community 115 - "get_v2_experiment"
Cohesion: 0.12
Nodes (11): get_v2_experiment(), cached_logits(), _DummyTokenizer, test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only() (+3 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 897 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `evaluation/context.py` to `run_sequential_pipeline`, `prepare_colab_pilot.py`, `test_training_recipes.py`, `test_shared_execution.py`, `save_matrix_checkpoint`, `os`, `shared_execution.py`, `calibrate_shared_profile`, `json`, `test_v2_lifecycle.py`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `matrix.py`, `typing`, `v2_preflight.py`, `test_v2_fixed_execution.py`, `fit_offline_sft`, `examples`, `stages.py`, `v2.py`, `matrix.py`, `config.py`, `workflow`, `test_v2_workflow.py`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `.from_pretrained_backbones`, `BifocalGemmaConfig`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `configuration_bifocalgemma.py`, `typing`, `BifocalGemmaConfig`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10793650793650794 - nodes in this community are weakly interconnected._
- **Why does `load_verified_tokenizer()` connect `json` to `remediation.py`, `curation.py`, `prepare_colab_pilot.py`, `test_gsm8k.py`, `CodeOverlapIndex`, `FastTrainingPipeline`, `hashlib`, `TokenizedRepairExample`, `repository_corpus.py`, `data/offline_sft.py`, `matrix.py`, `profile_repository_lengths.py`, `repository_study.py`, `gsm8k.py`, `typing`, `tokenization.py`, `examples`, `v2.py`, `matrix.py`, `evaluation/context.py`, `tokenize_repair_example`, `audit_dataset_release.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._