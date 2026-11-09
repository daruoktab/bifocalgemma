# Graph Report - bifocalgemma  (2026-11-09)

## Corpus Check
- 290 files · ~365,366 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 18 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2378 nodes · 6969 edges · 106 communities (72 shown, 34 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 707 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b69e3c42`
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
- v3/__init__.py
- test_gsm8k.py
- subprocess
- save_matrix_checkpoint
- json
- NativeTinyModel
- re
- shared_execution.py
- FeatureNotReadyError
- molab_pilot_support.py
- calibrate_shared_profile
- stable_fingerprint
- test_tier5_adversarial.py
- tests_e2e/conftest.py
- CodeOverlapIndex
- BifocalGemmaConfig
- ExperimentSpec
- atomic_json
- torch
- FastTrainingPipeline
- source_bundle_bytes
- Seq2SeqBatch
- test_molab_matrix_notebooks.py
- v3/objectives.py
- argparse
- torch
- repository_corpus.py
- test_tier1_features.py
- adversarial_r3_probe.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- .from_pretrained_backbones
- TestGemma4EdgeTopology
- benchmark_upload.py
- PackedTokenizedDataset
- test_molab_remediation.py
- test_tier4_scenarios.py
- test_tier2_boundary.py
- build_anti_forgetting_curriculum
- profile_repository_lengths.py
- fit_offline_sft
- hashlib
- RepoContextRouter
- ReferenceMergedAttention
- fast_pipeline.py
- adapters.py
- test_benchmark_prompt_budget.py
- TokenizedRepairExample
- generate_training_notebooks.py
- packed_tokens.py
- ASTGraphProjector
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- TestBifocalGemmaArchitecture
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- build_native_tiny_model
- tokenization.py
- examples
- is_cross_bridge_parameter
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- run_length_grid
- v2.py
- matrix.py
- test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract
- parametrize
- TestNotebookCellParity
- SwiGLUDualGateBridge
- resolve_optimal_eval_batch_size
- test_context_benchmark_resume.py
- configuration_bifocalgemma.py
- matrix.py
- pathlib
- load_asset_json
- BifocalGemmaForCausalLM
- bifocalgemma
- ScriptedModel
- test_context_tasks.py
- evaluation/context.py
- GatedCrossAttentionBridge
- CachedAttention
- telemetry.py
- JSONLRepairDataset
- test_v2_workflow.py
- configure_memory_efficient_training

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
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/source_identity.py
- `test_fast_training_config_and_pipeline_baseline_single_stage()` --uses--> `FastTrainingConfig`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/fast_pipeline.py
- `test_fast_training_pipeline_features_and_export()` --calls--> `FastTrainingConfig`  [INFERRED]
  tests/test_molab_remediation.py → src/bifocalgemma/training/fast_pipeline.py
- `test_fast_training_config_step_reconciliation()` --uses--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py

## Import Cycles
- None detected.

## Communities (106 total, 34 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.06
Nodes (22): pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt() (+14 more)

### Community 1 - "remediation.py"
Cohesion: 0.12
Nodes (22): split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+14 more)

### Community 2 - "curation.py"
Cohesion: 0.12
Nodes (20): file_digest(), read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+12 more)

### Community 3 - "run_sequential_pipeline"
Cohesion: 0.18
Nodes (10): reset_local_benchmarks(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint(), StageLedger, test_reset_archives_evaluation_once_without_changing_training() (+2 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.05
Nodes (45): task_prompt(), decode_text(), load_verified_tokenizer(), digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k() (+37 more)

### Community 5 - "releases.py"
Cohesion: 0.09
Nodes (25): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+17 more)

### Community 6 - "__main__.py"
Cohesion: 0.12
Nodes (10): validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents, run_native_smoke(), SFTConfig (+2 more)

### Community 7 - "test_training_recipes.py"
Cohesion: 0.17
Nodes (12): validate_training_policy(), comparison_recipes(), resolve_training_cadence(), run_recipe_comparison(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), test_four_fresh_runs_per_notebook_and_independent_resume() (+4 more)

### Community 8 - "test_shared_execution.py"
Cohesion: 0.09
Nodes (17): seal_report(), save(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), calibration(), guard(), test_followers_wait_for_one_profile_and_pin_it_locally() (+9 more)

### Community 9 - "RepositorySnapshot"
Cohesion: 0.11
Nodes (15): RepositoryFile, RepositorySnapshot, git_environment(), GitSnapshotStore, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash() (+7 more)

### Community 10 - "v3/__init__.py"
Cohesion: 0.14
Nodes (7): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, clamped_triton_kernel_options(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.06
Nodes (20): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+12 more)

### Community 12 - "subprocess"
Cohesion: 0.11
Nodes (11): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), audit(), emit(), main(), validate_all_patches(), main(), main(), main() (+3 more)

### Community 13 - "save_matrix_checkpoint"
Cohesion: 0.14
Nodes (12): latest_checkpoint(), retry_windows_file_operation(), validate_checkpoint_files(), load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint() (+4 more)

### Community 14 - "json"
Cohesion: 0.13
Nodes (9): write_view(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), test_view_record_json_is_lossless_and_gsm_shots_are_explicit(), audit_views(), main(), source_notices(), main() (+1 more)

### Community 16 - "re"
Cohesion: 0.11
Nodes (18): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path() (+10 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.17
Nodes (13): retry_delay(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), current_capacity(), execution_settings() (+5 more)

### Community 18 - "FeatureNotReadyError"
Cohesion: 0.09
Nodes (10): EvaluationReport, extract_token_ast_features(), TokenASTFeatures, SWEBenchHarness, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError (+2 more)

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.10
Nodes (18): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), measure(), measure() (+10 more)

### Community 20 - "calibrate_shared_profile"
Cohesion: 0.11
Nodes (14): native_config(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe() (+6 more)

### Community 21 - "stable_fingerprint"
Cohesion: 0.07
Nodes (37): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), normalize_context_task(), repair_preimage_coverage() (+29 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (22): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+14 more)

### Community 23 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.08
Nodes (35): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+27 more)

### Community 25 - "BifocalGemmaConfig"
Cohesion: 0.10
Nodes (14): BifocalGemmaConfig, FastTrainingConfig, test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip() (+6 more)

### Community 26 - "ExperimentSpec"
Cohesion: 0.15
Nodes (5): restore_checkpoint(), save_checkpoint(), ExperimentSpec, BifocalGemmaModel, build_pretrained_model()

### Community 27 - "atomic_json"
Cohesion: 0.07
Nodes (19): activity(), BenchmarkUploadQueue, atomic_json(), append_event(), maximum_probe(), activity(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail() (+11 more)

### Community 28 - "torch"
Cohesion: 0.10
Nodes (16): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+8 more)

### Community 30 - "source_bundle_bytes"
Cohesion: 0.16
Nodes (12): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+4 more)

### Community 31 - "Seq2SeqBatch"
Cohesion: 0.15
Nodes (5): ModelOutput, Seq2SeqBatch, TinyReferenceModel, __getattr__(), patch_cross_entropy()

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.09
Nodes (9): export_matrix_pretrained_model(), make_experiment_spec(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_matrix_baseline_single_stage_resolution(), make_identity(), test_telemetry_history_appends_all_active_stages(), test_export_matrix_pretrained_model() (+1 more)

### Community 34 - "v3/objectives.py"
Cohesion: 0.10
Nodes (13): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+5 more)

### Community 35 - "argparse"
Cohesion: 0.15
Nodes (8): main(), main(), render(), main(), declared_limit_example(), main(), render(), main()

### Community 36 - "torch"
Cohesion: 0.06
Nodes (7): transfer_text_checkpoint(), build_joint_visibility(), build_native_model(), RoutedRepoBatch, chunked_linear_cross_entropy(), test_notebook_stress_helpers_match_shared_library(), TestCheckpointTransfer

### Community 37 - "repository_corpus.py"
Cohesion: 0.08
Nodes (19): ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license() (+11 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 41 - ".from_pretrained_backbones"
Cohesion: 0.13
Nodes (5): _resolve_backbone_weights(), collate_fast_batch(), SyntheticRepairDataset, test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 43 - "benchmark_upload.py"
Cohesion: 0.19
Nodes (4): test_external_activity_reader_does_not_need_running_kernel(), test_keyboard_interrupt_is_not_swallowed(), inspect(), main()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.05
Nodes (11): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast() (+3 more)

### Community 46 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 47 - "test_tier2_boundary.py"
Cohesion: 0.15
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 48 - "build_anti_forgetting_curriculum"
Cohesion: 0.25
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 49 - "profile_repository_lengths.py"
Cohesion: 0.32
Nodes (4): publish(), distribution(), main(), profile()

### Community 50 - "fit_offline_sft"
Cohesion: 0.09
Nodes (21): run_training_memory_preflight(), load_sft_adapters_for_inference(), fit_curated_sft(), cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit() (+13 more)

### Community 51 - "hashlib"
Cohesion: 0.08
Nodes (43): get_experiment(), canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), normalized_code(), prepare_crosscode_tasks() (+35 more)

### Community 53 - "ReferenceMergedAttention"
Cohesion: 0.13
Nodes (5): apply_word_embedding_policy(), EmbeddingPolicyResult, ReferenceMergedAttention, TestMergedAttentionReference, TestWordEmbeddingPolicy

### Community 54 - "fast_pipeline.py"
Cohesion: 0.17
Nodes (7): configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), tensor_bytes(), TestMemoryProbe

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 58 - "TokenizedRepairExample"
Cohesion: 0.20
Nodes (7): TokenizedRepairExample, RepairCollator, __getattr__(), run_reference_smoke(), synthetic_stress_target(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 59 - "generate_training_notebooks.py"
Cohesion: 0.43
Nodes (4): main(), module_definitions(), render(), use_embedded_imports()

### Community 60 - "packed_tokens.py"
Cohesion: 0.26
Nodes (6): build_packed_cache(), open_packed_cache(), compare_predictions(), test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes(), test_paired_bootstrap_and_cohort_rejection()

### Community 64 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 65 - "build_native_tiny_model"
Cohesion: 0.08
Nodes (10): build_native_tiny_model(), NativeMergedAttention, benchmark_candidate(), probe_optimizer(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, test_last_token_logits_match_full_head(), factory() (+2 more)

### Community 66 - "tokenization.py"
Cohesion: 0.14
Nodes (9): DeveloperAgent, RepairExample, RepairPrediction, format_encoder_context(), tokenize_repair_example(), write_swebench_predictions(), generate_patch(), repair_prompt_ids() (+1 more)

### Community 67 - "examples"
Cohesion: 0.06
Nodes (21): TestTokenizerVerification, examples(), TestOfflineSFT, TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored() (+13 more)

### Community 69 - "is_cross_bridge_parameter"
Cohesion: 0.15
Nodes (6): build_bifocalgemma_v3_optimizer(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), configure_matrix_stage(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_v3_partitioned_optimizer_weight_decay_invariants()

### Community 72 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 74 - "v2.py"
Cohesion: 0.08
Nodes (26): prepare_curated_training(), tokenize(), checked_prefix(), recover_remote_run(), _synchronize_once(), source_fingerprints(), WithoutDocstrings, benchmark_directory_name() (+18 more)

### Community 76 - "matrix.py"
Cohesion: 0.06
Nodes (13): greedy_decode(), GreedyGeneration, InferenceCache, build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), Float32Alignment (+5 more)

### Community 77 - "test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 78 - "parametrize"
Cohesion: 0.33
Nodes (4): test_adversarial_reponiah_depth_and_ratio_extremes(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo()

### Community 81 - "resolve_optimal_eval_batch_size"
Cohesion: 0.40
Nodes (3): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 82 - "test_context_benchmark_resume.py"
Cohesion: 0.13
Nodes (11): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization() (+3 more)

### Community 84 - "matrix.py"
Cohesion: 0.18
Nodes (7): configure_full_rank_cross_attention(), run_matrix_benchmark(), run_matrix_stage(), run_matrix_training(), run_matrix_training_and_benchmark(), progress(), ensure_hardware_memory_guards()

### Community 85 - "pathlib"
Cohesion: 0.07
Nodes (7): BackboneProfile, CheckpointRef, _pair(), _ref(), remote_reset_plan(), reset_state(), test_training_failure_never_launches_benchmarks()

### Community 92 - "test_context_tasks.py"
Cohesion: 0.18
Nodes (7): fallback_letter_regex(), strict_letter(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 93 - "evaluation/context.py"
Cohesion: 0.11
Nodes (15): build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), prompt_ids(), run_context_benchmark() (+7 more)

### Community 96 - "telemetry.py"
Cohesion: 0.06
Nodes (13): format_stage_display(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html() (+5 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), workflow(), benchmark(), download() (+4 more)

### Community 115 - "configure_memory_efficient_training"
Cohesion: 0.07
Nodes (32): get_v2_experiment(), cached_logits(), configure_memory_efficient_training(), NormalizedAlignment, unified_recipe(), _DummyTokenizer, test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence() (+24 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 896 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `curation.py`, `run_sequential_pipeline`, `prepare_colab_pilot.py`, `test_training_recipes.py`, `test_shared_execution.py`, `save_matrix_checkpoint`, `json`, `shared_execution.py`, `molab_pilot_support.py`, `calibrate_shared_profile`, `stable_fingerprint`, `v3/objectives.py`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `benchmark_upload.py`, `packed_tokens.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `examples`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `v2.py`, `matrix.py`, `matrix.py`, `pathlib`, `evaluation/context.py`, `test_v2_workflow.py`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05519480519480519 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `torch`, `JSONLRepairDataset`, `.from_pretrained_backbones`, `test_molab_remediation.py`, `configuration_bifocalgemma.py`, `fast_pipeline.py`, `BifocalGemmaConfig`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Why does `TestBifocalGemmaArchitecture` connect `TestBifocalGemmaArchitecture` to `torch`, `RepoContextRouter`, `._run_bridge_test`, `ASTGraphProjector`, `GatedCrossAttentionBridge`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._