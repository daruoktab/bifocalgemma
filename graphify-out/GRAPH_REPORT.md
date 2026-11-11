# Graph Report - bifocalgemma  (2026-11-12)

## Corpus Check
- 296 files · ~381,888 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 26 file(s) not represented in the graph (top: .tex 11, (none) 3, .ipynb 3)

## Summary
- 2464 nodes · 7117 edges · 128 communities (90 shown, 38 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 772 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `573d2535`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- molab_pilot_support.py
- curation.py
- copy
- prepare_colab_pilot.py
- test_completed_benchmark_with_pending_upload_resumes_without_model_load
- model.py
- pathlib
- test_shared_execution.py
- json
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- test_molab_matrix_notebooks.py
- test_molab_pilot.py
- NativeTinyModel
- fit_offline_sft
- v2_preflight.py
- examples
- torch
- stages.py
- load_verified_tokenizer
- test_tier5_adversarial.py
- test_repository_study.py
- curation.py
- test_memory_efficient_training.py
- FeatureNotReadyError
- hashlib
- torch
- calibration
- v3/__init__.py
- RMSNorm
- matrix.py
- prepare_colab_probe.py
- data/context.py
- load_verified_tokenizer
- test_tier1_features.py
- adversarial_r3_probe.py
- tests_e2e/conftest.py
- JSONLRepairDataset
- TestGemma4EdgeTopology
- test_training_telemetry_v2.py
- prepare_context_data.py
- test_molab_remediation.py
- pytest
- BifocalGemmaForCausalLM
- strict_letter
- configure_policy_model
- default_training_profiles
- repository_study.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- build_joint_visibility
- memory_probe.py
- PackedTokenizedDataset
- adapters.py
- math
- fit_offline_sft
- atomic_json
- repository_corpus.py
- test_data_remediation.py
- shared_execution.py
- TestBifocalGemmaArchitecture
- matrix.py
- optimizers.py
- RepairExample
- examples
- test_colab_pilot.py
- file_digest
- calibrate_shared_profile
- RepositorySnapshot
- apply_word_embedding_policy
- argparse
- CodeOverlapIndex
- build_native_tiny_model
- run
- GitSnapshotStore
- releases.py
- ast
- benchmark.py
- BifocalGemmaForCausalLM
- bifocalgemma
- decode_text
- v2.py
- seal_report
- telemetry.py
- FastTrainingConfig
- probe
- reference_resilient_mcq_parser
- RepairPrediction
- shutil
- TestNotebookCellParity
- test_context_benchmark_resume.py
- evaluation/context.py
- audit_context_tokens.py
- profile_repository_lengths.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- test_fast_pipeline.py
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- TokenizedRepairExample
- bifocalgemma/__init__.py
- GatedCrossAttentionBridge
- TestOfflineSFT
- test_profile_publish_parent_guard_keeps_concurrent_winner
- policy.py
- typing
- patch_matrix_tokenization

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 84 edges
2. `stable_fingerprint()` - 67 edges
3. `load_verified_tokenizer()` - 57 edges
4. `TokenizedRepairExample` - 52 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `run_matrix_training()` - 41 edges
10. `RepairCollator` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()` --calls--> `resolve_optimal_eval_batch_size()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/benchmark.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (128 total, 38 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "molab_pilot_support.py"
Cohesion: 0.09
Nodes (18): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), digest() (+10 more)

### Community 2 - "curation.py"
Cohesion: 0.15
Nodes (11): read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+3 more)

### Community 3 - "copy"
Cohesion: 0.08
Nodes (19): remote_reset_plan(), reset_local_benchmarks(), reset_state(), BenchmarkUploadQueue, synchronize_run(), resolve_training_cadence(), run_recipe_comparison(), run_sequential_pipeline() (+11 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.09
Nodes (26): tokenize(), tokenize_repository_record(), build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage() (+18 more)

### Community 5 - "test_completed_benchmark_with_pending_upload_resumes_without_model_load"
Cohesion: 0.17
Nodes (11): limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume(), benchmark() (+3 more)

### Community 6 - "model.py"
Cohesion: 0.13
Nodes (4): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel, build_pretrained_model()

### Community 7 - "pathlib"
Cohesion: 0.09
Nodes (3): RepoContextRouter, RoutedRepoBatch, test_notebook_stress_helpers_match_shared_library()

### Community 8 - "test_shared_execution.py"
Cohesion: 0.17
Nodes (5): __getattr__(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline()

### Community 9 - "json"
Cohesion: 0.11
Nodes (17): normalize_context_task(), repair_preimage_coverage(), load_curated_pool(), write_view(), compare_predictions(), test_repair_requires_visible_preimage_without_refilling_context(), test_view_record_json_is_lossless_and_gsm_shots_are_explicit(), test_paired_bootstrap_and_cohort_rejection() (+9 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.13
Nodes (5): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 12 - "subprocess"
Cohesion: 0.09
Nodes (7): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "test_molab_matrix_notebooks.py"
Cohesion: 0.07
Nodes (25): configure_memory_efficient_training(), unified_recipe(), save_matrix_checkpoint(), make_experiment_spec(), configure_matrix_stage(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_configure_full_rank_cross_attention_v3_specs() (+17 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.23
Nodes (9): test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render(), source_bundle() (+1 more)

### Community 15 - "NativeTinyModel"
Cohesion: 0.14
Nodes (4): ModelOutput, native_config(), NativeMergedAttention, NativeTinyModel

### Community 16 - "fit_offline_sft"
Cohesion: 0.19
Nodes (8): run_context_benchmark(), activity(), atomic_json(), DurableClock, prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), test_prepare_sampled_benchmark_collection_deterministic_100_cases()

### Community 17 - "v2_preflight.py"
Cohesion: 0.21
Nodes (8): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), select_execution(), emit(), save()

### Community 18 - "examples"
Cohesion: 0.10
Nodes (17): TestTokenizerVerification, examples(), TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), calibration() (+9 more)

### Community 19 - "torch"
Cohesion: 0.08
Nodes (7): NormalizedAlignment, ASTGraphProjector, build_native_model(), run_native_smoke(), chunked_linear_cross_entropy(), patch_cross_entropy(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients()

### Community 20 - "stages.py"
Cohesion: 0.14
Nodes (7): retry_delay(), upload_initial_results(), test_external_activity_reader_does_not_need_running_kernel(), test_retry_after_date_and_network_failures(), bounded_threads(), inspect(), main()

### Community 21 - "load_verified_tokenizer"
Cohesion: 0.17
Nodes (14): load_verified_tokenizer(), digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids() (+6 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+15 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.09
Nodes (15): canonical_snapshot_hash(), NativeTaskResult, score_python_patch(), summarize_native_results(), project_root(), bounded_threads(), diff(), ExecutingScoringOptions (+7 more)

### Community 24 - "curation.py"
Cohesion: 0.11
Nodes (16): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure() (+8 more)

### Community 25 - "test_memory_efficient_training.py"
Cohesion: 0.18
Nodes (11): run_training_memory_preflight(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native(), test_optimized_long_window_causality_context_and_all_masked_backward() (+3 more)

### Community 26 - "FeatureNotReadyError"
Cohesion: 0.06
Nodes (18): validate_experiment_matrix(), EvaluationReport, extract_token_ast_features(), TokenASTFeatures, SWEBenchHarness, main(), EncoderMemory, RepositoryCache (+10 more)

### Community 27 - "hashlib"
Cohesion: 0.12
Nodes (17): curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines(), prepare_editing_records(), repository_editing_record(), validate_context_license_record(), notebook_function() (+9 more)

### Community 28 - "torch"
Cohesion: 0.10
Nodes (15): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+7 more)

### Community 30 - "calibration"
Cohesion: 0.16
Nodes (10): original(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), measure(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure() (+2 more)

### Community 31 - "v3/__init__.py"
Cohesion: 0.17
Nodes (7): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), format_query_injected_context(), safe_tokenize_context_task(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 34 - "matrix.py"
Cohesion: 0.08
Nodes (15): build_bifocalgemma_v3_optimizer(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), is_zero_weight_decay_bridge_parameter(), _pointer_chunk_forward() (+7 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.36
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "data/context.py"
Cohesion: 0.19
Nodes (12): ContextWindow, render_window(), select_repository_context(), _terms(), prepare_repository_record(), prepare_repository_record_with_snapshot(), repo_identity(), repository_task_from_row() (+4 more)

### Community 37 - "load_verified_tokenizer"
Cohesion: 0.13
Nodes (10): BackboneProfile, CheckpointRef, ExperimentSpec, _pair(), _ref(), format_encoder_context(), tokenize_repair_example(), generate_patch() (+2 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (12): validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync() (+4 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 40 - "tests_e2e/conftest.py"
Cohesion: 0.10
Nodes (9): authoritative_empirical_results(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization() (+1 more)

### Community 41 - "JSONLRepairDataset"
Cohesion: 0.14
Nodes (5): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 43 - "test_training_telemetry_v2.py"
Cohesion: 0.22
Nodes (3): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 44 - "prepare_context_data.py"
Cohesion: 0.14
Nodes (17): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), tokenize_context_task(), test_f2_mcq_token_budget_specification() (+9 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "pytest"
Cohesion: 0.13
Nodes (6): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion(), main()

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.09
Nodes (17): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization() (+9 more)

### Community 48 - "strict_letter"
Cohesion: 0.25
Nodes (4): _extract_match_letter(), fallback_letter_regex(), strict_letter(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 49 - "configure_policy_model"
Cohesion: 0.08
Nodes (23): get_v2_experiment(), split_editing_records(), fit_curated_sft(), configure_policy_model(), default_training_profiles(), comparison_recipes(), fit_repository_sft(), prepare_repository_training() (+15 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 53 - "build_joint_visibility"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 54 - "memory_probe.py"
Cohesion: 0.14
Nodes (11): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), run_length_grid(), synthetic_stress_target(), tensor_bytes(), test_length_grid_matches_baseline_budget_and_stops_after_failure() (+3 more)

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "math"
Cohesion: 0.15
Nodes (6): test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 58 - "fit_offline_sft"
Cohesion: 0.20
Nodes (12): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), validation_batches() (+4 more)

### Community 59 - "atomic_json"
Cohesion: 0.14
Nodes (7): append_event(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), progress()

### Community 60 - "repository_corpus.py"
Cohesion: 0.09
Nodes (19): ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license() (+11 more)

### Community 61 - "test_data_remediation.py"
Cohesion: 0.24
Nodes (13): merge_lineage(), preference(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed() (+5 more)

### Community 62 - "shared_execution.py"
Cohesion: 0.38
Nodes (6): publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), verify_report()

### Community 64 - "matrix.py"
Cohesion: 0.08
Nodes (23): stable_fingerprint(), fetch_collection(), benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), evaluate_sft_loss(), latest_checkpoint() (+15 more)

### Community 65 - "optimizers.py"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 66 - "RepairExample"
Cohesion: 0.22
Nodes (9): RepairExample, example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records() (+1 more)

### Community 67 - "examples"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 69 - "test_colab_pilot.py"
Cohesion: 0.13
Nodes (7): test_exported_worker_has_parity_context_and_exact_adapter_checks(), _DummyTokenizer, pilot_module(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), _prepare(), prepare_runtime()

### Community 70 - "file_digest"
Cohesion: 0.13
Nodes (9): file_digest(), checked_prefix(), recover_remote_run(), retry_delay(), _synchronize_once(), validate_checkpoint_files(), snapshot(), synchronize_v2() (+1 more)

### Community 72 - "calibrate_shared_profile"
Cohesion: 0.21
Nodes (9): benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), probe_optimizer() (+1 more)

### Community 74 - "RepositorySnapshot"
Cohesion: 0.14
Nodes (14): RepositoryFile, RepositorySnapshot, apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), patch_syntax(), test_dependency_completion_removes_target_and_retains_real_helper() (+6 more)

### Community 76 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "argparse"
Cohesion: 0.20
Nodes (7): main(), main(), render(), main(), module_definitions(), render(), use_embedded_imports()

### Community 78 - "CodeOverlapIndex"
Cohesion: 0.12
Nodes (28): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+20 more)

### Community 79 - "build_native_tiny_model"
Cohesion: 0.15
Nodes (6): build_native_tiny_model(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words()

### Community 81 - "run"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+4 more)

### Community 83 - "releases.py"
Cohesion: 0.10
Nodes (23): archive_files(), audit_split_disjointness(), build_release(), digest(), main(), publish_release(), rows(), unpack_archive() (+15 more)

### Community 84 - "ast"
Cohesion: 0.23
Nodes (7): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), main(), render_eval_script(), code_fragment_signatures()

### Community 85 - "benchmark.py"
Cohesion: 0.07
Nodes (11): GreedyGeneration, CachedAttention, InferenceCache, configure_full_rank_cross_attention(), batched_greedy_decode(), infer_benchmark_weight_tier(), SwiGLUDualGateBridge, clamped_triton_kernel_options() (+3 more)

### Community 91 - "decode_text"
Cohesion: 0.15
Nodes (11): task_prompt(), decode_text(), score_context_case(), score_repo_niah_case(), generate_diagnostic_text(), cached_logits(), audit(), audit_controls() (+3 more)

### Community 92 - "v2.py"
Cohesion: 0.10
Nodes (18): load_asset_json(), load_local_native_model(), load_sft_adapters_for_inference(), prepare_curated_training(), benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics(), stratified_subset() (+10 more)

### Community 95 - "seal_report"
Cohesion: 0.27
Nodes (7): seal_report(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), valid_profile(), check_evidence()

### Community 96 - "telemetry.py"
Cohesion: 0.09
Nodes (9): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard (+1 more)

### Community 97 - "FastTrainingConfig"
Cohesion: 0.11
Nodes (10): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed() (+2 more)

### Community 100 - "probe"
Cohesion: 0.18
Nodes (9): numerical_probe(), representative_cohort(), shared_contract(), source_fingerprint(), version(), check_followers(), probe(), progress() (+1 more)

### Community 101 - "reference_resilient_mcq_parser"
Cohesion: 0.24
Nodes (4): reference_resilient_mcq_parser(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 102 - "RepairPrediction"
Cohesion: 0.31
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 105 - "shutil"
Cohesion: 0.28
Nodes (4): source_fingerprints(), WithoutDocstrings, test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()

### Community 107 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 108 - "evaluation/context.py"
Cohesion: 0.10
Nodes (12): build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), greedy_decode() (+4 more)

### Community 109 - "audit_context_tokens.py"
Cohesion: 0.24
Nodes (6): prompt_ids(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), audit(), main()

### Community 110 - "profile_repository_lengths.py"
Cohesion: 0.24
Nodes (5): test_shared_profile_upload_retries_without_losing_local_selection(), publish(), distribution(), main(), profile()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.12
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 115 - "TokenizedRepairExample"
Cohesion: 0.10
Nodes (15): Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), build_packed_cache(), open_packed_cache(), run_reference_smoke(), TinyReferenceModel (+7 more)

### Community 121 - "policy.py"
Cohesion: 0.29
Nodes (3): build_training_groups(), Float32Alignment, validate_training_policy()

### Community 126 - "typing"
Cohesion: 0.09
Nodes (4): ensure_assets_on_disk(), transfer_text_checkpoint(), ProbeLoRALinear, is_cross_bridge_parameter()

### Community 134 - "patch_matrix_tokenization"
Cohesion: 0.20
Nodes (3): patch_matrix_tokenization(), safe_load_verified_tokenizer(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 933 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **38 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `fit_offline_sft` to `molab_pilot_support.py`, `copy`, `prepare_colab_pilot.py`, `test_completed_benchmark_with_pending_upload_resumes_without_model_load`, `pathlib`, `json`, `subprocess`, `v2_preflight.py`, `examples`, `stages.py`, `calibration`, `matrix.py`, `prepare_context_data.py`, `fit_offline_sft`, `atomic_json`, `shared_execution.py`, `matrix.py`, `examples`, `file_digest`, `calibrate_shared_profile`, `run`, `benchmark.py`, `decode_text`, `v2.py`, `probe`, `evaluation/context.py`, `audit_context_tokens.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `TokenizedRepairExample`, `typing`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `load_verified_tokenizer()` connect `load_verified_tokenizer` to `curation.py`, `prepare_colab_pilot.py`, `json`, `test_gsm8k.py`, `test_molab_pilot.py`, `fit_offline_sft`, `examples`, `curation.py`, `FastTrainingPipeline`, `load_verified_tokenizer`, `prepare_context_data.py`, `configure_policy_model`, `repository_study.py`, `repository_corpus.py`, `matrix.py`, `file_digest`, `benchmark.py`, `decode_text`, `v2.py`, `evaluation/context.py`, `audit_context_tokens.py`, `profile_repository_lengths.py`, `typing`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Should `molab_pilot_support.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09302325581395349 - nodes in this community are weakly interconnected._
- **Why does `TestGemma4EdgeTopology` connect `TestGemma4EdgeTopology` to `torch`, `GatedCrossAttentionBridge`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._