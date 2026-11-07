# Graph Report - bifocalgemma  (2026-11-07)

## Corpus Check
- 290 files · ~364,185 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2353 nodes · 6919 edges · 118 communities (86 shown, 32 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 709 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c1b225cc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- calibrate_shared_profile
- v2.py
- prepare_colab_pilot.py
- releases.py
- FeatureNotReadyError
- data/offline_sft.py
- typing
- repository.py
- test_pretrained_hf_model.py
- test_gsm8k.py
- subprocess
- atomic_json
- ast
- Seq2SeqBatch
- get_v2_experiment
- FastTrainingPipeline
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- molab_pilot_support.py
- test_memory_efficient_training.py
- score_context_case
- test_tier5_adversarial.py
- v2_preflight.py
- curation.py
- gsm8k.py
- contracts.py
- full_pool.py
- tiled_attention
- comparison.py
- test_molab_pilot.py
- load_verified_tokenizer
- Tensor
- matrix.py
- GatedCrossAttentionBridge
- TokenizedRepairExample
- repository_corpus.py
- test_tier1_features.py
- TestMemoryProbe
- audit_context_tokens.py
- config.py
- adversarial_r3_probe.py
- argparse
- audit_context_release.py
- test_molab_remediation.py
- swesmith.py
- completion_metrics
- benchmark_upload.py
- pytest
- adapters.py
- repository_study.py
- shared_execution.py
- apply_word_embedding_policy
- ReferenceMergedAttention
- test_one_model_training_stops_and_completed_resume_does_not_reload
- test_training_telemetry_v2.py
- BifocalGemmaConfig
- transfer_text_checkpoint
- profile_repository_lengths.py
- PackedTokenizedDataset
- build_native_tiny_model
- molab_matrix_support.py
- TestBifocalGemmaArchitecture
- test_molab_matrix_notebooks.py
- configure_memory_efficient_training
- TestTrainingPolicy
- optimizers.py
- tests_e2e/conftest.py
- molab_matrix_support.py
- initialization.py
- configure_full_rank_cross_attention
- json
- fit_offline_sft
- random
- ASTGraphProjector
- run_length_grid
- TestNotebookCellParity
- v3/__init__.py
- SyntheticRepairDataset
- BenchmarkUploadQueue
- modeling_bifocalgemma.py
- patch_matrix_tokenization
- BifocalGemmaForCausalLM
- bifocalgemma
- test_repository_study.py
- sys
- score_context_case
- v2_diagnostics.py
- config.py
- telemetry.py
- .from_pretrained_backbones
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- check_followers
- Journal
- benchmark.py
- validate_experiment_matrix
- data/context.py
- calibration
- training/offline_sft.py
- test_v2_workflow.py
- SwiGLUDualGateBridge
- workflow
- test_completed_benchmark_with_pending_upload_resumes_without_model_load
- test_snapshot_verifies_replacement_before_pruning
- cache.py
- test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training
- test_profile_publish_parent_guard_keeps_concurrent_winner

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 83 edges
2. `stable_fingerprint()` - 62 edges
3. `TokenizedRepairExample` - 53 edges
4. `load_verified_tokenizer()` - 52 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `RepairCollator` - 40 edges
10. `configure_policy_model()` - 39 edges

## Surprising Connections (you probably didn't know these)
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `test_fast_training_config_and_pipeline_baseline_single_stage()` --uses--> `FastTrainingConfig`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/fast_pipeline.py

## Import Cycles
- None detected.

## Communities (118 total, 32 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "remediation.py"
Cohesion: 0.12
Nodes (23): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+15 more)

### Community 2 - "calibrate_shared_profile"
Cohesion: 0.11
Nodes (14): native_config(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind() (+6 more)

### Community 3 - "v2.py"
Cohesion: 0.09
Nodes (15): source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), _execute_v2(), save_guard(), RemoteHeartbeat, synchronize_preflight() (+7 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.13
Nodes (20): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset(), evaluate_pilot_checkpoint(), extract_resolvable_span() (+12 more)

### Community 5 - "releases.py"
Cohesion: 0.08
Nodes (28): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+20 more)

### Community 6 - "FeatureNotReadyError"
Cohesion: 0.12
Nodes (8): EvaluationReport, SWEBenchHarness, main(), FeatureNotReadyError, distillation_loss(), SFTConfig, train_direct_sft(), test_train_fast_cli_baseline_2b()

### Community 7 - "data/offline_sft.py"
Cohesion: 0.10
Nodes (21): normalize_context_task(), curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines(), load_curated_pool(), prepare_editing_records(), repository_editing_record() (+13 more)

### Community 8 - "typing"
Cohesion: 0.08
Nodes (4): NormalizedAlignment, build_native_model(), RoutedRepoBatch, test_notebook_stress_helpers_match_shared_library()

### Community 9 - "repository.py"
Cohesion: 0.12
Nodes (13): RepositoryFile, apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), patch_paths(), safe_repo_path(), GitSnapshotStore, prepare_repository_record_with_snapshot() (+5 more)

### Community 10 - "test_pretrained_hf_model.py"
Cohesion: 0.10
Nodes (16): export_matrix_pretrained_model(), FastTrainingConfig, configure_probe_training(), ProbeLoRALinear, test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload() (+8 more)

### Community 11 - "test_gsm8k.py"
Cohesion: 0.06
Nodes (19): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+11 more)

### Community 12 - "subprocess"
Cohesion: 0.15
Nodes (7): NativeTaskResult, score_python_patch(), summarize_native_results(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), run_cmd()

### Community 13 - "atomic_json"
Cohesion: 0.09
Nodes (29): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), build_packed_cache() (+21 more)

### Community 14 - "ast"
Cohesion: 0.16
Nodes (10): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), main(), main(), render_eval_script(), declared_limit_example() (+2 more)

### Community 15 - "Seq2SeqBatch"
Cohesion: 0.17
Nodes (4): ModelOutput, Seq2SeqBatch, __getattr__(), patch_cross_entropy()

### Community 16 - "get_v2_experiment"
Cohesion: 0.15
Nodes (10): get_v2_experiment(), cached_logits(), CachedAttention, test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only(), test_encoder_bidirectional_window_boundary_matches_native() (+2 more)

### Community 17 - "FastTrainingPipeline"
Cohesion: 0.15
Nodes (3): FastTrainingPipeline, run_fast_training(), test_fast_training_config_and_pipeline_baseline_single_stage()

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.13
Nodes (12): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), digest(), journaled_loss(), evaluate(), PilotStorage, run(), _run() (+4 more)

### Community 20 - "test_memory_efficient_training.py"
Cohesion: 0.17
Nodes (13): chunked_linear_cross_entropy(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+5 more)

### Community 21 - "score_context_case"
Cohesion: 0.14
Nodes (11): repair_preimage_coverage(), task_prompt(), tokenize_context_task(), score_context_case(), task(), test_dependency_completion_removes_target_and_retains_real_helper(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline() (+3 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (20): extract_mcq_letter(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling(), test_adversarial_mcq_ruling_out_distractors_vulnerability() (+12 more)

### Community 23 - "v2_preflight.py"
Cohesion: 0.18
Nodes (11): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution(), save() (+3 more)

### Community 24 - "curation.py"
Cohesion: 0.09
Nodes (32): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits() (+24 more)

### Community 25 - "gsm8k.py"
Cohesion: 0.20
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 26 - "contracts.py"
Cohesion: 0.08
Nodes (14): DeveloperAgent, RepairExample, RepairPrediction, extract_token_ast_features(), TokenASTFeatures, decode_text(), format_encoder_context(), tokenize_repair_example() (+6 more)

### Community 27 - "full_pool.py"
Cohesion: 0.11
Nodes (25): export_pool(), file_digest(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool() (+17 more)

### Community 28 - "tiled_attention"
Cohesion: 0.12
Nodes (16): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+8 more)

### Community 30 - "test_molab_pilot.py"
Cohesion: 0.13
Nodes (14): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+6 more)

### Community 31 - "load_verified_tokenizer"
Cohesion: 0.16
Nodes (11): load_verified_tokenizer(), fit_curated_sft(), prepare_curated_training(), tokenize(), default_training_profiles(), fit_repository_sft(), prepare_repository_training(), assets() (+3 more)

### Community 32 - "Tensor"
Cohesion: 0.09
Nodes (14): build_bifocalgemma_v3_optimizer(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids() (+6 more)

### Community 34 - "matrix.py"
Cohesion: 0.10
Nodes (15): configure_full_rank_cross_attention(), load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix_v3(), run_matrix_benchmark() (+7 more)

### Community 35 - "GatedCrossAttentionBridge"
Cohesion: 0.10
Nodes (3): GatedCrossAttentionBridge, RMSNorm, TestGemma4EdgeTopology

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.14
Nodes (11): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), run_native_smoke(), synthetic_stress_target(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation() (+3 more)

### Community 37 - "repository_corpus.py"
Cohesion: 0.10
Nodes (17): build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license(), RepositoryPreparationLedger (+9 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 40 - "audit_context_tokens.py"
Cohesion: 0.13
Nodes (9): decoder_prompt(), prompt_ids(), test_adversarial_context_containing_forbidden_phrases(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer, audit() (+1 more)

### Community 41 - "config.py"
Cohesion: 0.18
Nodes (8): BackboneProfile, CheckpointRef, ExperimentSpec, _pair(), _ref(), run_reference_smoke(), TinyReferenceModel, profile_experiment()

### Community 42 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 43 - "argparse"
Cohesion: 0.18
Nodes (9): audit(), emit(), main(), validate_all_patches(), main(), main(), module_definitions(), render() (+1 more)

### Community 44 - "audit_context_release.py"
Cohesion: 0.21
Nodes (7): ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "swesmith.py"
Cohesion: 0.26
Nodes (8): RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 47 - "completion_metrics"
Cohesion: 0.09
Nodes (13): completion_metrics(), reference_resilient_mcq_parser(), validate_paper_manuscript(), test_f3_crosscodelong_continuous_metrics(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation() (+5 more)

### Community 48 - "benchmark_upload.py"
Cohesion: 0.23
Nodes (3): test_external_activity_reader_does_not_need_running_kernel(), inspect(), main()

### Community 49 - "pytest"
Cohesion: 0.07
Nodes (14): test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), examples(), TestOfflineSFT, TinyLossModel, test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend() (+6 more)

### Community 50 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 51 - "repository_study.py"
Cohesion: 0.19
Nodes (20): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), prepare_repository_record(), RepositoryTask, load_local_native_model() (+12 more)

### Community 52 - "shared_execution.py"
Cohesion: 0.21
Nodes (11): retry_delay(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), verify_report(), test_retry_after_date_and_network_failures() (+3 more)

### Community 53 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 54 - "ReferenceMergedAttention"
Cohesion: 0.17
Nodes (4): build_joint_visibility(), ReferenceMergedAttention, NativeMergedAttention, TestMergedAttentionReference

### Community 55 - "test_one_model_training_stops_and_completed_resume_does_not_reload"
Cohesion: 0.29
Nodes (3): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory()

### Community 56 - "test_training_telemetry_v2.py"
Cohesion: 0.20
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_training_resume_keeps_elapsed_from_step_five_through_ten(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 57 - "BifocalGemmaConfig"
Cohesion: 0.17
Nodes (3): BifocalGemmaConfig, test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_pretrained_model_subfolder_support_and_repo_creation()

### Community 58 - "transfer_text_checkpoint"
Cohesion: 0.17
Nodes (6): transfer_text_checkpoint(), gpu_memory(), optimizer_bytes(), profile_training_group(), tensor_bytes(), TestCheckpointTransfer

### Community 59 - "profile_repository_lengths.py"
Cohesion: 0.28
Nodes (5): test_shared_profile_upload_retries_without_losing_local_selection(), publish(), distribution(), main(), profile()

### Community 61 - "build_native_tiny_model"
Cohesion: 0.18
Nodes (4): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words()

### Community 64 - "test_molab_matrix_notebooks.py"
Cohesion: 0.09
Nodes (8): prepare_sampled_benchmark_collection(), test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_prepare_sampled_benchmark_collection_deterministic_100_cases(), test_telemetry_history_appends_all_active_stages()

### Community 65 - "configure_memory_efficient_training"
Cohesion: 0.10
Nodes (22): configure_memory_efficient_training(), NativeTinyModel, configure_policy_model(), unified_recipe(), make_experiment_spec(), configure_matrix_stage(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence() (+14 more)

### Community 69 - "optimizers.py"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 70 - "tests_e2e/conftest.py"
Cohesion: 0.10
Nodes (9): authoritative_empirical_results(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization() (+1 more)

### Community 71 - "molab_matrix_support.py"
Cohesion: 0.13
Nodes (6): test_exported_worker_has_parity_context_and_exact_adapter_checks(), _DummyTokenizer, pilot_module(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), prepare_runtime()

### Community 72 - "initialization.py"
Cohesion: 0.32
Nodes (4): describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents

### Community 74 - "json"
Cohesion: 0.10
Nodes (16): remote_reset_plan(), reset_local_benchmarks(), reset_state(), checked_prefix(), recover_remote_run(), _synchronize_once(), latest_checkpoint(), retained_checkpoints() (+8 more)

### Community 75 - "fit_offline_sft"
Cohesion: 0.26
Nodes (9): run_context_benchmark(), activity(), fit_offline_sft(), emit(), model_stage(), resource_snapshot(), DurableClock, run_batched_context_benchmark() (+1 more)

### Community 76 - "random"
Cohesion: 0.27
Nodes (5): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 78 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 80 - "v3/__init__.py"
Cohesion: 0.09
Nodes (8): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, is_zero_weight_decay_bridge_parameter(), clamped_triton_kernel_options(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 81 - "SyntheticRepairDataset"
Cohesion: 0.13
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 82 - "BenchmarkUploadQueue"
Cohesion: 0.07
Nodes (16): BenchmarkUploadQueue, limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_parent_conflict_refreshes_but_keeps_retry_bounded() (+8 more)

### Community 84 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 91 - "test_repository_study.py"
Cohesion: 0.10
Nodes (20): canonical_snapshot_hash(), example_from_record(), project_root(), CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions (+12 more)

### Community 93 - "score_context_case"
Cohesion: 0.12
Nodes (10): build_repo_niah_case(), fallback_letter_regex(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), score_repo_niah_case() (+2 more)

### Community 94 - "v2_diagnostics.py"
Cohesion: 0.28
Nodes (5): evaluate_sft_loss(), validation_batches(), encoder_diagnostics(), stratified_subset(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order()

### Community 95 - "config.py"
Cohesion: 0.15
Nodes (4): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel, build_pretrained_model()

### Community 96 - "telemetry.py"
Cohesion: 0.09
Nodes (11): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog (+3 more)

### Community 97 - ".from_pretrained_backbones"
Cohesion: 0.29
Nodes (3): _resolve_backbone_weights(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 99 - "check_followers"
Cohesion: 0.32
Nodes (7): append_event(), download_model(), progress(), emit(), check_followers(), progress(), progress()

### Community 101 - "benchmark.py"
Cohesion: 0.17
Nodes (7): greedy_decode(), GreedyGeneration, InferenceCache, batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 103 - "data/context.py"
Cohesion: 0.48
Nodes (4): ContextWindow, render_window(), select_repository_context(), _terms()

### Community 105 - "calibration"
Cohesion: 0.13
Nodes (13): test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), TestTokenizerVerification, calibration(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt() (+5 more)

### Community 106 - "training/offline_sft.py"
Cohesion: 0.12
Nodes (13): cpu_tree(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), build_training_groups(), Float32Alignment, optimizer_group_manifest(), update_policy_learning_rates() (+5 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.25
Nodes (6): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), run()

### Community 109 - "workflow"
Cohesion: 0.22
Nodes (6): workflow(), benchmark(), download(), execution(), fit(), prepare()

### Community 111 - "test_completed_benchmark_with_pending_upload_resumes_without_model_load"
Cohesion: 0.15
Nodes (13): comparison_recipes(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume(), benchmark() (+5 more)

### Community 115 - "test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training"
Cohesion: 0.50
Nodes (3): test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 881 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `calibrate_shared_profile`, `v2.py`, `prepare_colab_pilot.py`, `subprocess`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `molab_pilot_support.py`, `score_context_case`, `v2_preflight.py`, `full_pool.py`, `Tensor`, `matrix.py`, `audit_context_tokens.py`, `audit_context_release.py`, `benchmark_upload.py`, `shared_execution.py`, `test_molab_matrix_notebooks.py`, `json`, `fit_offline_sft`, `BenchmarkUploadQueue`, `score_context_case`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `check_followers`, `benchmark.py`, `calibration`, `training/offline_sft.py`, `test_v2_workflow.py`, `workflow`, `test_completed_benchmark_with_pending_upload_resumes_without_model_load`, `test_snapshot_verifies_replacement_before_pruning`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08858858858858859 - nodes in this community are weakly interconnected._
- **Why does `TestBifocalGemmaArchitecture` connect `TestBifocalGemmaArchitecture` to `typing`, `GatedCrossAttentionBridge`, `ASTGraphProjector`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11931818181818182 - nodes in this community are weakly interconnected._
- **Why does `TestTrainingPolicy` connect `TestTrainingPolicy` to `pytest`, `test_pretrained_hf_model.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._