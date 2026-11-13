# Graph Report - bifocalgemma  (2026-11-13)

## Corpus Check
- 307 files · ~392,219 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2625 nodes · 7636 edges · 125 communities (84 shown, 41 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 828 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5c3f53cd`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- test_molab_matrix_notebooks.py
- stable_fingerprint
- copy
- prepare_colab_pilot.py
- molab_pilot_support.py
- FeatureNotReadyError
- memory_probe.py
- json
- curation.py
- FullCrossCachedAttention
- eval_code.py
- subprocess
- test_matrix_runtime_loader.py
- hashlib
- schedule.py
- re
- shared_execution.py
- examples
- NativeTinyModel
- os
- run_matrix_training
- test_tier5_adversarial.py
- test_repository_study.py
- CodeOverlapIndex
- configure_memory_efficient_training
- TokenizedRepairExample
- data/offline_sft.py
- torch
- FastTrainingPipeline
- project_root
- test_memory_efficient_training.py
- GatedCrossAttentionBridge
- typing
- prepare_colab_probe.py
- repository.py
- normalize_context_task
- pytest
- fit_offline_sft
- NativeTinyModel
- v3/__init__.py
- benchmark.py
- test_context_benchmark_resume.py
- adversarial_r3_probe.py
- test_context_benchmark_mcq_budget_floor_execution
- test_training_recipes.py
- BifocalGemmaForCausalLM
- test_tier3_combinations.py
- test_matrix_schedule_runtime.py
- evaluation/context.py
- repository_study.py
- contracts.py
- examples
- matrix.py
- apply_word_embedding_policy
- RepoContextRouter
- reference_resilient_mcq_parser
- run_sequential_pipeline
- TestGemma4EdgeTopology
- JSONLRepairDataset
- repository_corpus.py
- TestBifocalGemmaArchitecture
- configure_full_rank_cross_attention
- test_gsm8k.py
- test_wiggly_dashboard_lifecycle.py
- ASTGraphProjector
- BifocalGemmaModel
- test_offline_sft_training.py
- adapters.py
- re
- build_joint_visibility
- native_config
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- build_native_tiny_model
- WigglyDashboard
- test_v2_workflow.py
- render_dashboard
- releases.py
- test_v2_fixed_execution.py
- test_context_tasks.py
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- BifocalGemmaForCausalLM
- bifocalgemma
- _DummyTokenizer
- v2.py
- workflow
- harness.py
- wiggly_dashboard.py
- test_tier4_scenarios.py
- test_notebook_hf_flow.py
- atomic_json
- resolve_optimal_eval_batch_size
- typing
- argparse
- generation.py
- tokenize_repair_example
- TrainingDashboard
- test_fast_pipeline.py
- config.py
- PackedTokenizedDataset
- bifocalgemma/__init__.py
- RecordingStream
- test_matrix_quarantine_legacy_checkpoint_vs_clean_resume
- test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues
- test_shared_execution.py

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 84 edges
2. `stable_fingerprint()` - 69 edges
3. `load_verified_tokenizer()` - 58 edges
4. `TokenizedRepairExample` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `run_matrix_training()` - 46 edges
7. `_execute_v2()` - 45 edges
8. `configure_memory_efficient_training()` - 44 edges
9. `build_native_tiny_model()` - 43 edges
10. `RepairCollator` - 42 edges

## Surprising Connections (you probably didn't know these)
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/models/v2/native.py
- `test_telemetry_history_appends_all_active_stages()` --calls--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `save_report()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_pilot.py → src/bifocalgemma/training/stages.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (125 total, 41 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.06
Nodes (23): pilot_module(), dependency_state(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_stale_loaded_ui_version_rejects_each_ready_fast_path(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install() (+15 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.12
Nodes (6): test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_dashboard_stage_display_formatting(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_per_model_hf_repo(), test_telemetry_history_appends_all_active_stages()

### Community 2 - "stable_fingerprint"
Cohesion: 0.17
Nodes (13): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), build_packed_cache() (+5 more)

### Community 3 - "copy"
Cohesion: 0.13
Nodes (9): load_sft_adapters_for_inference(), build_policy_optimizer(), GrokOrScaleLoRA, validate_optimizer_config(), build_training_groups(), Float32Alignment, validate_training_policy(), resolve_training_cadence() (+1 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.10
Nodes (24): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset() (+16 more)

### Community 5 - "molab_pilot_support.py"
Cohesion: 0.09
Nodes (19): fail_second_microbatch(), test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), test_upload_retry_verification_and_base_prefix_only() (+11 more)

### Community 6 - "FeatureNotReadyError"
Cohesion: 0.11
Nodes (8): extract_token_ast_features(), TokenASTFeatures, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError, distillation_loss(), profile_experiment()

### Community 7 - "memory_probe.py"
Cohesion: 0.10
Nodes (11): transfer_text_checkpoint(), gpu_memory(), optimizer_bytes(), profile_training_group(), run_length_grid(), synthetic_stress_target(), tensor_bytes(), test_length_grid_matches_baseline_budget_and_stops_after_failure() (+3 more)

### Community 8 - "json"
Cohesion: 0.07
Nodes (14): ensure_assets_on_disk(), load_asset_json(), load_verified_tokenizer(), compare_predictions(), remote_reset_plan(), download_model(), _synchronize_once(), test_remote_reset_is_scoped_and_receipt_hashes_match() (+6 more)

### Community 9 - "curation.py"
Cohesion: 0.08
Nodes (27): export_pool(), file_digest(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool() (+19 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.12
Nodes (7): attention_tile(), Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), forward()

### Community 11 - "eval_code.py"
Cohesion: 0.13
Nodes (6): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 12 - "subprocess"
Cohesion: 0.12
Nodes (7): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.24
Nodes (13): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+5 more)

### Community 14 - "hashlib"
Cohesion: 0.07
Nodes (24): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), WithoutDocstrings, test_exported_worker_has_parity_context_and_exact_adapter_checks(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects() (+16 more)

### Community 15 - "schedule.py"
Cohesion: 0.11
Nodes (22): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+14 more)

### Community 16 - "re"
Cohesion: 0.14
Nodes (14): repair_preimage_coverage(), apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), NativeTaskResult, score_python_patch(), summarize_native_results() (+6 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.08
Nodes (33): retry_delay(), benchmark_candidate(), calibrate_shared_profile(), save(), trial(), candidate_settings(), error_kind(), numerical_probe() (+25 more)

### Community 18 - "examples"
Cohesion: 0.07
Nodes (12): TestTokenizerVerification, test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), bounded_threads(), test_atomic_status_retries_windows_sharing_violation_without_partial_write(), replace(), test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory() (+4 more)

### Community 19 - "NativeTinyModel"
Cohesion: 0.10
Nodes (5): build_native_model(), run_native_smoke(), RoutedRepoBatch, chunked_linear_cross_entropy(), checkpoint()

### Community 20 - "os"
Cohesion: 0.18
Nodes (5): audit_pool(), audit_views(), main(), main(), source_notices()

### Community 21 - "run_matrix_training"
Cohesion: 0.12
Nodes (10): RemoteHeartbeat, retry_delay(), latest_checkpoint(), run_training_and_benchmark(), resolve_training_contract(), run_matrix_benchmark(), run_matrix_training(), run_matrix_training_and_benchmark() (+2 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+15 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.11
Nodes (17): CodeOverlapMatch, prepare_repository_record(), repository_task_from_row(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions, fixture() (+9 more)

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.08
Nodes (40): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits(), empty_syntax_warning_summary() (+32 more)

### Community 25 - "configure_memory_efficient_training"
Cohesion: 0.10
Nodes (27): get_v2_experiment(), cached_logits(), configure_memory_efficient_training(), fit_curated_sft(), configure_policy_model(), default_training_profiles(), unified_recipe(), fit_repository_sft() (+19 more)

### Community 26 - "TokenizedRepairExample"
Cohesion: 0.16
Nodes (9): TokenizedRepairExample, RepairCollator, __getattr__(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), TestDataAndLoss (+1 more)

### Community 27 - "data/offline_sft.py"
Cohesion: 0.09
Nodes (25): normalize_context_task(), tokenize_context_task(), curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines(), load_curated_pool(), prepare_editing_records() (+17 more)

### Community 28 - "torch"
Cohesion: 0.10
Nodes (14): flex_kernel_options(), fused_attention(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention, TiledMergedAttention (+6 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.09
Nodes (3): FastTrainingPipeline, run_fast_training(), configure_probe_training()

### Community 31 - "test_memory_efficient_training.py"
Cohesion: 0.11
Nodes (15): run_training_memory_preflight(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+7 more)

### Community 34 - "typing"
Cohesion: 0.09
Nodes (13): build_bifocalgemma_v3_optimizer(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids() (+5 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "repository.py"
Cohesion: 0.17
Nodes (10): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), git_environment(), example_from_record(), GitSnapshotStore (+2 more)

### Community 37 - "normalize_context_task"
Cohesion: 0.19
Nodes (10): generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record(), parent(), test_evaluation_and_unlicensed_sources_rejected(), test_grounded_generation_answerability_provenance_and_target_independence() (+2 more)

### Community 38 - "pytest"
Cohesion: 0.05
Nodes (24): task_prompt(), authoritative_empirical_results(), pytest_configure(), reference_resilient_mcq_parser(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema() (+16 more)

### Community 39 - "fit_offline_sft"
Cohesion: 0.14
Nodes (14): cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint() (+6 more)

### Community 40 - "NativeTinyModel"
Cohesion: 0.10
Nodes (6): ModelOutput, Seq2SeqBatch, NormalizedAlignment, NativeTinyModel, __getattr__(), patch_cross_entropy()

### Community 41 - "v3/__init__.py"
Cohesion: 0.17
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), is_zero_weight_decay_bridge_parameter(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 42 - "benchmark.py"
Cohesion: 0.09
Nodes (22): prompt_ids(), run_context_benchmark(), activity(), score_context_case(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), mcq_report_metrics() (+14 more)

### Community 43 - "test_context_benchmark_resume.py"
Cohesion: 0.14
Nodes (10): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization() (+2 more)

### Community 44 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 45 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 46 - "test_training_recipes.py"
Cohesion: 0.11
Nodes (16): orthogonal_direction(), comparison_recipes(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload() (+8 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.07
Nodes (20): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), ProbeLoRALinear, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_pretrained_model_subfolder_support_and_repo_creation() (+12 more)

### Community 49 - "test_matrix_schedule_runtime.py"
Cohesion: 0.17
Nodes (10): run_matrix_stage(), backward_window(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download() (+2 more)

### Community 50 - "evaluation/context.py"
Cohesion: 0.08
Nodes (14): build_repo_niah_case(), decoder_prompt(), _extract_match_letter(), fallback_letter_regex(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk() (+6 more)

### Community 51 - "repository_study.py"
Cohesion: 0.20
Nodes (20): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), load_local_native_model(), fetch_tokenizer_assets(), generate_run() (+12 more)

### Community 52 - "contracts.py"
Cohesion: 0.10
Nodes (16): DeveloperAgent, RepairExample, RepairPrediction, RepositoryFile, RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord (+8 more)

### Community 53 - "examples"
Cohesion: 0.09
Nodes (11): examples(), TestOfflineSFT, TinyLossModel, test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit(), test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored() (+3 more)

### Community 54 - "matrix.py"
Cohesion: 0.12
Nodes (15): target_fingerprint(), retry_windows_file_operation(), validate_checkpoint_files(), load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint() (+7 more)

### Community 55 - "apply_word_embedding_policy"
Cohesion: 0.22
Nodes (4): apply_word_embedding_policy(), EmbeddingPolicyResult, TinyReferenceModel, TestWordEmbeddingPolicy

### Community 57 - "reference_resilient_mcq_parser"
Cohesion: 0.17
Nodes (4): test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 58 - "run_sequential_pipeline"
Cohesion: 0.17
Nodes (11): reset_local_benchmarks(), reset_state(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint(), StageLedger (+3 more)

### Community 61 - "JSONLRepairDataset"
Cohesion: 0.14
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_with_jsonl_data_dir()

### Community 62 - "repository_corpus.py"
Cohesion: 0.08
Nodes (20): count_syntax_warnings(), ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive() (+12 more)

### Community 64 - "configure_full_rank_cross_attention"
Cohesion: 0.10
Nodes (10): configure_full_rank_cross_attention(), clamped_triton_kernel_options(), critical_attention_beta(), ensure_hardware_memory_guards(), format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task() (+2 more)

### Community 65 - "test_gsm8k.py"
Cohesion: 0.17
Nodes (8): FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train(), test_verified_preparation_and_input_only_loading()

### Community 66 - "test_wiggly_dashboard_lifecycle.py"
Cohesion: 0.22
Nodes (5): model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 69 - "BifocalGemmaModel"
Cohesion: 0.13
Nodes (4): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel, build_pretrained_model()

### Community 70 - "test_offline_sft_training.py"
Cohesion: 0.10
Nodes (4): TestNotebookCellParity, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 72 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 74 - "re"
Cohesion: 0.28
Nodes (8): code_fragment_signatures(), completion_from_snapshot(), context_pool(), log(), main(), mixed_pool(), prepare_core(), source_python_paths()

### Community 76 - "build_joint_visibility"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 78 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 79 - "build_native_tiny_model"
Cohesion: 0.12
Nodes (6): build_native_tiny_model(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, TestCheckpointTransfer, TestMemoryProbe, test_last_token_logits_match_full_head()

### Community 81 - "test_v2_workflow.py"
Cohesion: 0.20
Nodes (7): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), run()

### Community 82 - "render_dashboard"
Cohesion: 0.25
Nodes (4): format_duration(), format_stage_display(), render_dashboard(), test_monitor_escapes_errors_updates_validation_and_exports_html()

### Community 83 - "releases.py"
Cohesion: 0.06
Nodes (41): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+33 more)

### Community 84 - "test_v2_fixed_execution.py"
Cohesion: 0.14
Nodes (9): calibration(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure(), test_preflight_upload_verifies_exact_evidence_without_checkpoint() (+1 more)

### Community 85 - "test_context_tasks.py"
Cohesion: 0.22
Nodes (6): strict_letter(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 92 - "v2.py"
Cohesion: 0.11
Nodes (21): prepare_curated_training(), tokenize(), checked_prefix(), recover_remote_run(), source_fingerprints(), benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics() (+13 more)

### Community 93 - "workflow"
Cohesion: 0.25
Nodes (5): workflow(), benchmark(), download(), execution(), fit()

### Community 96 - "wiggly_dashboard.py"
Cohesion: 0.20
Nodes (3): LastSeenWidget, phase_category(), TrainingLossLog

### Community 101 - "test_tier4_scenarios.py"
Cohesion: 0.32
Nodes (3): test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 104 - "atomic_json"
Cohesion: 0.07
Nodes (24): BenchmarkUploadQueue, emit(), atomic_json(), append_event(), loss_chart(), download_model(), progress(), emit() (+16 more)

### Community 105 - "resolve_optimal_eval_batch_size"
Cohesion: 0.40
Nodes (3): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 106 - "typing"
Cohesion: 0.08
Nodes (3): CachedAttention, SwiGLUDualGateBridge, test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()

### Community 107 - "argparse"
Cohesion: 0.12
Nodes (11): audit(), emit(), main(), validate_all_patches(), main(), main(), render(), distribution() (+3 more)

### Community 108 - "generation.py"
Cohesion: 0.14
Nodes (12): decode_text(), format_encoder_context(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids(), _safe_apply_chat_template() (+4 more)

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.07
Nodes (14): FastTrainingConfig, DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+6 more)

### Community 114 - "config.py"
Cohesion: 0.11
Nodes (15): BackboneProfile, CheckpointRef, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), main(), describe_initialization() (+7 more)

### Community 127 - "test_shared_execution.py"
Cohesion: 0.13
Nodes (9): test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), test_shared_profile_upload_retries_without_losing_local_selection(), publish() (+1 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 979 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **41 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `stable_fingerprint`, `copy`, `prepare_colab_pilot.py`, `json`, `re`, `shared_execution.py`, `examples`, `NativeTinyModel`, `os`, `run_matrix_training`, `benchmark.py`, `test_training_recipes.py`, `evaluation/context.py`, `examples`, `matrix.py`, `run_sequential_pipeline`, `re`, `test_v2_workflow.py`, `test_v2_fixed_execution.py`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `v2.py`, `workflow`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0602322206095791 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `copy`, `json`, `curation.py`, `hashlib`, `shared_execution.py`, `os`, `run_matrix_training`, `CodeOverlapIndex`, `configure_memory_efficient_training`, `data/offline_sft.py`, `benchmark.py`, `test_matrix_schedule_runtime.py`, `evaluation/context.py`, `matrix.py`, `run_sequential_pipeline`, `repository_corpus.py`, `re`, `releases.py`, `v2.py`, `argparse`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12418300653594772 - nodes in this community are weakly interconnected._
- **Why does `test_snapshot_verifies_replacement_before_pruning()` connect `examples` to `atomic_json`, `v2.py`, `examples`, `fit_offline_sft`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._