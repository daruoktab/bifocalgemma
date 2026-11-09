# Graph Report - bifocalgemma  (2026-11-09)

## Corpus Check
- 290 files · ~363,889 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 18 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2369 nodes · 6953 edges · 118 communities (84 shown, 34 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 705 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `dcde8e41`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- curation.py
- atomic_json
- prepare_colab_pilot.py
- releases.py
- config.py
- data/offline_sft.py
- test_shared_execution.py
- repository_corpus.py
- v3/__init__.py
- test_gsm8k.py
- subprocess
- save_matrix_checkpoint
- json
- NativeTinyModel
- prepare_colab_probe.py
- shared_execution.py
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- molab_pilot_support.py
- BifocalGemmaConfig
- prepare_context_data.py
- test_tier5_adversarial.py
- tests_e2e/conftest.py
- CodeOverlapIndex
- test_pretrained_hf_model.py
- gsm8k.py
- benchmark_upload.py
- torch
- JSONLRepairDataset
- source_bundle_bytes
- replace
- test_molab_matrix_notebooks.py
- v3/objectives.py
- test_memory_efficient_training.py
- TokenizedRepairExample
- repository_corpus.py
- test_tier1_features.py
- adversarial_r3_probe.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- run_sequential_pipeline
- TestGemma4EdgeTopology
- test_completed_benchmark_with_pending_upload_resumes_without_model_load
- PackedTokenizedDataset
- test_molab_remediation.py
- swesmith.py
- test_tier2_boundary.py
- random
- examples
- TestTrainingPolicy
- repository_study.py
- test_tier4_scenarios.py
- ReferenceMergedAttention
- test_molab_pilot.py
- test_snapshot_verifies_replacement_before_pruning
- adapters.py
- test_benchmark_prompt_budget.py
- memory_probe.py
- ast
- stable_fingerprint
- fit_offline_sft
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- TestBifocalGemmaArchitecture
- audit_dataset_release.py
- build_native_tiny_model
- load_verified_tokenizer
- pathlib
- math
- curate_paired_sources
- run_length_grid
- v2.py
- benchmark.py
- typing
- build_policy_optimizer
- TestNotebookCellParity
- SwiGLUDualGateBridge
- FastTrainingPipeline
- test_context_benchmark_resume.py
- bifocalgemma/__init__.py
- matrix.py
- native_python.py
- importlib
- BifocalGemmaForCausalLM
- bifocalgemma
- test_repository_study.py
- test_context_tasks.py
- evaluation/context.py
- GatedCrossAttentionBridge
- test_training_telemetry_v2.py
- telemetry.py
- build_repository_corpus
- test_context_benchmark_mcq_budget_floor_execution
- molab_matrix_support.py
- test_notebook_hf_flow.py
- configure_policy_model
- test_v2_workflow.py
- .from_pretrained_backbones
- ASTGraphProjector
- GitSnapshotStore
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- evaluate
- get_v2_experiment
- __init__

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 83 edges
2. `stable_fingerprint()` - 65 edges
3. `TokenizedRepairExample` - 53 edges
4. `load_verified_tokenizer()` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `RepairCollator` - 40 edges
10. `run_matrix_training()` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `save_report()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_pilot.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (118 total, 34 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.06
Nodes (19): native_config(), NativeMergedAttention, simulated_environment(), version(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt() (+11 more)

### Community 1 - "remediation.py"
Cohesion: 0.12
Nodes (24): code_signature(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+16 more)

### Community 2 - "curation.py"
Cohesion: 0.11
Nodes (15): code_tokens(), export_pool(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool() (+7 more)

### Community 3 - "atomic_json"
Cohesion: 0.07
Nodes (23): file_digest(), build_packed_cache(), open_packed_cache(), activity(), checked_prefix(), recover_remote_run(), _synchronize_once(), atomic_json() (+15 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.10
Nodes (23): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset(), evaluate_pilot_checkpoint() (+15 more)

### Community 5 - "releases.py"
Cohesion: 0.08
Nodes (28): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+20 more)

### Community 6 - "config.py"
Cohesion: 0.04
Nodes (30): DeveloperAgent, restore_checkpoint(), save_checkpoint(), BackboneProfile, CheckpointRef, ExperimentSpec, _pair(), _ref() (+22 more)

### Community 7 - "data/offline_sft.py"
Cohesion: 0.11
Nodes (20): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines() (+12 more)

### Community 8 - "test_shared_execution.py"
Cohesion: 0.11
Nodes (13): completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_profile_publish_parent_guard_keeps_concurrent_winner() (+5 more)

### Community 9 - "repository_corpus.py"
Cohesion: 0.13
Nodes (12): nextcoder_row(), apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), context_license_evidence(), path_license_evidence(), recognize_license() (+4 more)

### Community 10 - "v3/__init__.py"
Cohesion: 0.12
Nodes (8): TiledMergedAttention, Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, clamped_triton_kernel_options(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (14): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+6 more)

### Community 12 - "subprocess"
Cohesion: 0.11
Nodes (5): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), run_cmd(), upload_in_chunks()

### Community 13 - "save_matrix_checkpoint"
Cohesion: 0.22
Nodes (7): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), test_save_and_restore_matrix_checkpoint_and_prune(), test_stage1_to_stage2_optimizer_state_transfer()

### Community 14 - "json"
Cohesion: 0.09
Nodes (18): normalize_context_task(), repair_preimage_coverage(), compare_predictions(), test_repair_requires_visible_preimage_without_refilling_context(), test_paired_bootstrap_and_cohort_rejection(), audit_pool(), audit_views(), main() (+10 more)

### Community 16 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 17 - "shared_execution.py"
Cohesion: 0.09
Nodes (32): configure_memory_efficient_training(), retry_delay(), build_training_groups(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial() (+24 more)

### Community 18 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.14
Nodes (10): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), digest(), Journal, PilotStorage, run(), _run() (+2 more)

### Community 20 - "BifocalGemmaConfig"
Cohesion: 0.23
Nodes (3): BifocalGemmaConfig, test_fast_training_pipeline_features_and_export(), test_pretrained_model_subfolder_support_and_repo_creation()

### Community 21 - "prepare_context_data.py"
Cohesion: 0.23
Nodes (10): tokenize_context_task(), load_curated_pool(), code_fragment_signatures(), completion_from_snapshot(), context_pool(), log(), main(), mixed_pool() (+2 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+15 more)

### Community 23 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.16
Nodes (19): code_shingles(), CodeOverlapIndex, CodeOverlapMatch, validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_cross_source_duplicates_and_nextcoder_variants(), test_disk_index_matches_memory_and_syntax_repairs_are_admitted() (+11 more)

### Community 25 - "test_pretrained_hf_model.py"
Cohesion: 0.12
Nodes (15): export_matrix_pretrained_model(), FastTrainingConfig, test_fast_training_config_and_pipeline_baseline_single_stage(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip() (+7 more)

### Community 26 - "gsm8k.py"
Cohesion: 0.18
Nodes (14): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+6 more)

### Community 27 - "benchmark_upload.py"
Cohesion: 0.09
Nodes (9): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_retry_after_date_and_network_failures() (+1 more)

### Community 28 - "torch"
Cohesion: 0.08
Nodes (18): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), run_training_memory_preflight(), tiled_attention(), _TiledAttention (+10 more)

### Community 29 - "JSONLRepairDataset"
Cohesion: 0.15
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 30 - "source_bundle_bytes"
Cohesion: 0.19
Nodes (10): test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), main(), render() (+2 more)

### Community 31 - "replace"
Cohesion: 0.11
Nodes (14): TestTokenizerVerification, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), test_atomic_status_retries_windows_sharing_violation_without_partial_write(), replace(), test_pilot_subset_is_deterministic_and_stratified() (+6 more)

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.08
Nodes (14): unified_recipe(), make_experiment_spec(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_matrix_baseline_single_stage_resolution() (+6 more)

### Community 34 - "v3/objectives.py"
Cohesion: 0.10
Nodes (12): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 35 - "test_memory_efficient_training.py"
Cohesion: 0.14
Nodes (14): __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion() (+6 more)

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.07
Nodes (11): ModelOutput, RepairPrediction, Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), write_swebench_predictions(), run_reference_smoke() (+3 more)

### Community 37 - "repository_corpus.py"
Cohesion: 0.13
Nodes (14): validate_context_license_record(), ArchiveSnapshotStore, patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read() (+6 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (13): task_prompt(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f2_mcq_token_budget_specification(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval() (+5 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 41 - "run_sequential_pipeline"
Cohesion: 0.11
Nodes (15): remote_reset_plan(), reset_local_benchmarks(), reset_state(), synchronize_run(), resolve_training_cadence(), run_sequential_pipeline(), publish_final(), save_benchmark_progress() (+7 more)

### Community 43 - "test_completed_benchmark_with_pending_upload_resumes_without_model_load"
Cohesion: 0.14
Nodes (15): validate_training_policy(), comparison_recipes(), run_recipe_comparison(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload() (+7 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.08
Nodes (7): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues(), test_matrix_quarantine_legacy_checkpoint_vs_clean_resume(), test_mcq_budget_floor_in_v3_benchmark_logic()

### Community 46 - "swesmith.py"
Cohesion: 0.16
Nodes (12): RepositoryFile, RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, test_dependency_completion_removes_target_and_retains_real_helper() (+4 more)

### Community 47 - "test_tier2_boundary.py"
Cohesion: 0.15
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 48 - "random"
Cohesion: 0.24
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 49 - "examples"
Cohesion: 0.16
Nodes (11): original(), calibration(), guard(), examples(), fixture_guard(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt() (+3 more)

### Community 51 - "repository_study.py"
Cohesion: 0.22
Nodes (20): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), generate_diagnostic_text(), fetch_tokenizer_assets() (+12 more)

### Community 52 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 53 - "ReferenceMergedAttention"
Cohesion: 0.11
Nodes (6): apply_word_embedding_policy(), EmbeddingPolicyResult, build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference, TestWordEmbeddingPolicy

### Community 54 - "test_molab_pilot.py"
Cohesion: 0.20
Nodes (6): test_failed_final_upload_remains_pending(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), render(), journaled_loss()

### Community 55 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 58 - "memory_probe.py"
Cohesion: 0.12
Nodes (12): load_local_native_model(), load_sft_adapters_for_inference(), configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), synthetic_stress_target() (+4 more)

### Community 59 - "ast"
Cohesion: 0.22
Nodes (7): test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), main(), module_definitions(), render(), use_embedded_imports(), main(), render_eval_script()

### Community 60 - "stable_fingerprint"
Cohesion: 0.42
Nodes (8): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), benchmarks()

### Community 61 - "fit_offline_sft"
Cohesion: 0.13
Nodes (14): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest() (+6 more)

### Community 64 - "audit_dataset_release.py"
Cohesion: 0.26
Nodes (5): audit(), emit(), main(), validate_all_patches(), main()

### Community 65 - "build_native_tiny_model"
Cohesion: 0.13
Nodes (7): transfer_text_checkpoint(), build_native_tiny_model(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head(), test_v3_wall_time_checkpoint_resume_is_exact(), make()

### Community 66 - "load_verified_tokenizer"
Cohesion: 0.17
Nodes (9): RepairExample, format_encoder_context(), load_verified_tokenizer(), tokenize_repair_example(), generate_patch(), repair_prompt_ids(), prepare_repository_training(), assets() (+1 more)

### Community 67 - "pathlib"
Cohesion: 0.06
Nodes (7): ensure_assets_on_disk(), load_asset_json(), Float32Alignment, test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), test_notebook_stress_helpers_match_shared_library()

### Community 69 - "math"
Cohesion: 0.15
Nodes (6): build_bifocalgemma_v3_optimizer(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), configure_matrix_stage(), test_v3_partitioned_optimizer_weight_decay_invariants(), main()

### Community 70 - "curate_paired_sources"
Cohesion: 0.18
Nodes (9): curate_paired_sources(), count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_curated_warning_target_retained_and_summary_distinguishes_duplicate(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples() (+1 more)

### Community 72 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 74 - "v2.py"
Cohesion: 0.10
Nodes (21): source_fingerprints(), WithoutDocstrings, append_event(), benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics(), stratified_subset(), download_model() (+13 more)

### Community 76 - "benchmark.py"
Cohesion: 0.15
Nodes (8): greedy_decode(), GreedyGeneration, InferenceCache, configure_full_rank_cross_attention(), batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 77 - "typing"
Cohesion: 0.09
Nodes (6): CachedAttention, NormalizedAlignment, build_native_model(), run_native_smoke(), RepoContextRouter, RoutedRepoBatch

### Community 78 - "build_policy_optimizer"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 82 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 84 - "matrix.py"
Cohesion: 0.11
Nodes (13): update_policy_learning_rates(), run_matrix_benchmark(), snapshot(), run_matrix_stage(), run_matrix_training(), run_matrix_training_and_benchmark(), progress(), snapshot() (+5 more)

### Community 85 - "native_python.py"
Cohesion: 0.19
Nodes (11): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), example_from_record(), NativeTaskResult, score_python_patch() (+3 more)

### Community 87 - "importlib"
Cohesion: 0.22
Nodes (6): test_exported_worker_has_parity_context_and_exact_adapter_checks(), pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), prepare_runtime()

### Community 91 - "test_repository_study.py"
Cohesion: 0.11
Nodes (21): prepare_repository_record(), prepare_repository_record_with_snapshot(), repo_identity(), repository_task_from_row(), RepositoryTask, bounded_threads(), CharacterTokenizer, diff() (+13 more)

### Community 92 - "test_context_tasks.py"
Cohesion: 0.20
Nodes (6): fallback_letter_regex(), strict_letter(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 93 - "evaluation/context.py"
Cohesion: 0.10
Nodes (18): decode_text(), build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), prompt_ids() (+10 more)

### Community 95 - "test_training_telemetry_v2.py"
Cohesion: 0.20
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 96 - "telemetry.py"
Cohesion: 0.09
Nodes (11): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog (+3 more)

### Community 97 - "build_repository_corpus"
Cohesion: 0.26
Nodes (4): build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 99 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 106 - "configure_policy_model"
Cohesion: 0.19
Nodes (9): fit_curated_sft(), prepare_curated_training(), tokenize(), configure_policy_model(), default_training_profiles(), fit_repository_sft(), tokenize_repository_record(), test_curated_resume_rejects_runtime_drift() (+1 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.20
Nodes (7): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), run()

### Community 108 - ".from_pretrained_backbones"
Cohesion: 0.29
Nodes (3): _resolve_backbone_weights(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 114 - "evaluate"
Cohesion: 0.50
Nodes (3): evaluate(), event_callback(), event()

### Community 115 - "get_v2_experiment"
Cohesion: 0.14
Nodes (9): get_v2_experiment(), cached_logits(), _DummyTokenizer, test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only() (+1 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 890 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `prepare_colab_pilot.py`, `json`, `shared_execution.py`, `prepare_context_data.py`, `benchmark_upload.py`, `replace`, `TokenizedRepairExample`, `repository_corpus.py`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `run_sequential_pipeline`, `test_completed_benchmark_with_pending_upload_resumes_without_model_load`, `examples`, `test_snapshot_verifies_replacement_before_pruning`, `memory_probe.py`, `stable_fingerprint`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `v2.py`, `benchmark.py`, `matrix.py`, `evaluation/context.py`, `test_v2_workflow.py`, `evaluate`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06382978723404255 - nodes in this community are weakly interconnected._
- **Why does `test_snapshot_verifies_replacement_before_pruning()` connect `test_snapshot_verifies_replacement_before_pruning` to `atomic_json`, `TestTrainingPolicy`, `pathlib`, `fit_offline_sft`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11942959001782531 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `atomic_json`, `prepare_colab_pilot.py`, `data/offline_sft.py`, `json`, `prepare_colab_probe.py`, `shared_execution.py`, `prepare_context_data.py`, `CodeOverlapIndex`, `torch`, `JSONLRepairDataset`, `test_molab_matrix_notebooks.py`, `test_memory_efficient_training.py`, `PackedTokenizedDataset`, `random`, `TestTrainingPolicy`, `test_molab_pilot.py`, `memory_probe.py`, `load_verified_tokenizer`, `pathlib`, `typing`, `importlib`, `configure_policy_model`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._