# Graph Report - bifocalgemma  (2026-11-12)

## Corpus Check
- 287 files · ~375,086 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2328 nodes · 6818 edges · 125 communities (75 shown, 50 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 756 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ad8a5995`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- test_molab_matrix_notebooks.py
- curation.py
- sequential.py
- prepare_colab_pilot.py
- repository_sft.py
- config.py
- CachedAttention
- test_shared_execution.py
- stages.py
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- test_molab_matrix_notebooks.py
- test_molab_pilot.py
- NativeTinyModel
- run_batched_context_benchmark
- shared_execution.py
- pytest
- torch
- json
- gsm8k.py
- greedy_decode
- curation.py
- test_memory_efficient_training.py
- TokenizedRepairExample
- hashlib
- torch
- FastTrainingPipeline
- test_v2_fixed_execution.py
- TestEvaluationReporting
- RMSNorm
- matrix.py
- prepare_colab_probe.py
- data/context.py
- fast_pipeline.py
- fit_offline_sft
- JSONLRepairDataset
- TestGemma4EdgeTopology
- test_context_benchmark_resume.py
- repository_corpus.py
- test_molab_remediation.py
- chunked_linear_cross_entropy
- BifocalGemmaForCausalLM
- random
- test_v2_models.py
- test_benchmark_prompt_budget.py
- repository_study.py
- test_molab_pilot.py
- ASTGraphProjector
- SwiGLUDualGateBridge
- test_snapshot_verifies_replacement_before_pruning
- adapters.py
- training/offline_sft.py
- build_joint_visibility
- test_repository_corpus.py
- test_data_remediation.py
- build_repository_corpus
- TestBifocalGemmaArchitecture
- matrix.py
- configure_probe_training
- RepairExample
- test_curation.py
- Journal
- workflow
- molab_pilot_support.py
- test_repository_study.py
- test_construction_blocks.py
- hashlib
- inspect_python_target
- build_native_tiny_model
- run_length_grid
- run
- releases.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- distillation.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_colab_probe.py
- v2.py
- RepoContextRouter
- test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training
- telemetry.py
- pathlib
- TestNotebookCellParity
- atomic_json
- evaluation/context.py
- test_fast_pipeline.py
- stable_fingerprint
- bifocalgemma/__init__.py
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
9. `run_matrix_training()` - 42 edges
10. `RepairCollator` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `save_report()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_pilot.py → src/bifocalgemma/training/stages.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/source_identity.py

## Import Cycles
- None detected.

## Communities (125 total, 50 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.06
Nodes (22): _DummyTokenizer, pilot_module(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt() (+14 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.06
Nodes (6): test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_telemetry_history_appends_all_active_stages(), main()

### Community 2 - "curation.py"
Cohesion: 0.12
Nodes (15): file_digest(), read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+7 more)

### Community 3 - "sequential.py"
Cohesion: 0.09
Nodes (17): reset_local_benchmarks(), reset_state(), synchronize_run(), validate_optimizer_config(), Float32Alignment, validate_training_policy(), resolve_training_cadence(), run_recipe_comparison() (+9 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.05
Nodes (45): evaluate_sft_loss(), validation_batches(), GrokOrScaleLoRA, orthogonal_direction(), comparison_recipes(), encoder_diagnostics(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load() (+37 more)

### Community 5 - "repository_sft.py"
Cohesion: 0.29
Nodes (6): prepare_editing_records(), split_editing_records(), default_training_profiles(), fit_repository_sft(), prepare_repository_training(), TestOfflineEditing

### Community 6 - "config.py"
Cohesion: 0.05
Nodes (26): restore_checkpoint(), save_checkpoint(), BackboneProfile, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), EvaluationReport (+18 more)

### Community 8 - "test_shared_execution.py"
Cohesion: 0.13
Nodes (9): test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), test_shared_profile_upload_retries_without_losing_local_selection() (+1 more)

### Community 9 - "stages.py"
Cohesion: 0.09
Nodes (10): checked_prefix(), recover_remote_run(), retry_delay(), _synchronize_once(), retained_checkpoints(), validate_checkpoint_files(), synchronize_v2(), upload_initial_results() (+2 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.11
Nodes (7): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.16
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 12 - "subprocess"
Cohesion: 0.10
Nodes (10): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), audit(), emit(), main(), validate_all_patches(), main(), main(), main() (+2 more)

### Community 13 - "test_molab_matrix_notebooks.py"
Cohesion: 0.11
Nodes (27): get_v2_experiment(), task_prompt(), cached_logits(), configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes() (+19 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.17
Nodes (11): test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render(), main() (+3 more)

### Community 15 - "NativeTinyModel"
Cohesion: 0.09
Nodes (8): Seq2SeqBatch, NormalizedAlignment, NativeTinyModel, test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412(), test_fast_pipeline_batch_edge_cases(), test_alignment_initialization_with_cuda_default_device()

### Community 16 - "run_batched_context_benchmark"
Cohesion: 0.18
Nodes (9): decoder_prompt(), prompt_ids(), run_context_benchmark(), activity(), DurableClock, prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity() (+1 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.07
Nodes (36): retry_delay(), build_policy_optimizer(), build_training_groups(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial() (+28 more)

### Community 18 - "pytest"
Cohesion: 0.07
Nodes (13): examples(), TestOfflineSFT, TinyLossModel, test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes() (+5 more)

### Community 19 - "torch"
Cohesion: 0.06
Nodes (5): transfer_text_checkpoint(), build_native_model(), RoutedRepoBatch, ProbeLoRALinear, test_notebook_stress_helpers_match_shared_library()

### Community 20 - "json"
Cohesion: 0.07
Nodes (25): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), compare_predictions(), remote_reset_plan() (+17 more)

### Community 21 - "gsm8k.py"
Cohesion: 0.08
Nodes (25): DeveloperAgent, RepairPrediction, decode_text(), digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k() (+17 more)

### Community 23 - "greedy_decode"
Cohesion: 0.33
Nodes (3): greedy_decode(), ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards()

### Community 24 - "curation.py"
Cohesion: 0.15
Nodes (21): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), count_syntax_warnings() (+13 more)

### Community 25 - "test_memory_efficient_training.py"
Cohesion: 0.22
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 26 - "TokenizedRepairExample"
Cohesion: 0.17
Nodes (7): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), synthetic_stress_target(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 27 - "hashlib"
Cohesion: 0.26
Nodes (10): editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical() (+2 more)

### Community 28 - "torch"
Cohesion: 0.12
Nodes (14): fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention, TiledMergedAttention (+6 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.07
Nodes (3): collate_fast_batch(), FastTrainingPipeline, run_fast_training()

### Community 30 - "test_v2_fixed_execution.py"
Cohesion: 0.26
Nodes (7): calibration(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure()

### Community 31 - "TestEvaluationReporting"
Cohesion: 0.15
Nodes (6): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 34 - "matrix.py"
Cohesion: 0.08
Nodes (15): build_bifocalgemma_v3_optimizer(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), is_zero_weight_decay_bridge_parameter(), _pointer_chunk_forward() (+7 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.36
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "data/context.py"
Cohesion: 0.48
Nodes (4): ContextWindow, render_window(), select_repository_context(), _terms()

### Community 39 - "fit_offline_sft"
Cohesion: 0.23
Nodes (5): fit_offline_sft(), emit(), model_stage(), make_identity(), TestTrainingPolicy

### Community 41 - "JSONLRepairDataset"
Cohesion: 0.18
Nodes (4): JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 43 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 44 - "repository_corpus.py"
Cohesion: 0.11
Nodes (16): tokenize_context_task(), patch_paths(), context_license_evidence(), path_license_evidence(), recognize_license(), resolved_license(), repository_task_from_row(), code_fragment_signatures() (+8 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.08
Nodes (17): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization() (+9 more)

### Community 48 - "random"
Cohesion: 0.24
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 50 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 52 - "test_molab_pilot.py"
Cohesion: 0.16
Nodes (5): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), StageStore

### Community 55 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 58 - "training/offline_sft.py"
Cohesion: 0.33
Nodes (6): cpu_tree(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest(), retry_windows_file_operation()

### Community 59 - "build_joint_visibility"
Cohesion: 0.17
Nodes (3): build_joint_visibility(), native_config(), NativeMergedAttention

### Community 60 - "test_repository_corpus.py"
Cohesion: 0.21
Nodes (11): validate_context_license_record(), patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch() (+3 more)

### Community 61 - "test_data_remediation.py"
Cohesion: 0.12
Nodes (22): split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+14 more)

### Community 62 - "build_repository_corpus"
Cohesion: 0.26
Nodes (4): build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 64 - "matrix.py"
Cohesion: 0.07
Nodes (29): benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), update_policy_learning_rates(), configure_full_rank_cross_attention(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size() (+21 more)

### Community 65 - "configure_probe_training"
Cohesion: 0.27
Nodes (6): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), tensor_bytes(), TestMemoryProbe

### Community 66 - "RepairExample"
Cohesion: 0.09
Nodes (22): CheckpointRef, RepairExample, RepositoryFile, RepositorySnapshot, example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord (+14 more)

### Community 67 - "test_curation.py"
Cohesion: 0.17
Nodes (12): export_pool(), validate_curated_pool_limits(), fit_curated_sft(), raw(), test_complete_epoch_training_budget_all_families(), test_cross_source_duplicates_and_nextcoder_variants(), test_curated_resume_rejects_runtime_drift(), test_frozen_budget_matches_all_six_controls() (+4 more)

### Community 69 - "Journal"
Cohesion: 0.31
Nodes (5): test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), measure(), Journal

### Community 70 - "workflow"
Cohesion: 0.22
Nodes (6): workflow(), benchmark(), download(), execution(), fit(), prepare()

### Community 72 - "molab_pilot_support.py"
Cohesion: 0.18
Nodes (9): test_completed_run_retries_upload_without_loading_model(), digest(), journaled_loss(), evaluate(), PilotStorage, run(), _run(), event_callback() (+1 more)

### Community 74 - "test_repository_study.py"
Cohesion: 0.07
Nodes (26): canonical_snapshot_hash(), apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), safe_repo_path(), GitSnapshotStore, prepare_repository_record(), prepare_repository_record_with_snapshot() (+18 more)

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.07
Nodes (11): ModelOutput, apply_word_embedding_policy(), EmbeddingPolicyResult, ReferenceMergedAttention, run_reference_smoke(), TinyReferenceModel, __getattr__(), patch_cross_entropy() (+3 more)

### Community 77 - "hashlib"
Cohesion: 0.12
Nodes (16): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), normalize_context_task(), repair_preimage_coverage(), curated_pool_digest(), editing_record() (+8 more)

### Community 78 - "inspect_python_target"
Cohesion: 0.23
Nodes (5): inspect_python_target(), summarize_admitted_warning_records(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples(), test_warning_has_provenance_without_stderr_or_source_mutation()

### Community 79 - "build_native_tiny_model"
Cohesion: 0.16
Nodes (5): build_native_tiny_model(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words()

### Community 80 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 81 - "run"
Cohesion: 0.20
Nodes (7): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), run()

### Community 83 - "releases.py"
Cohesion: 0.08
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 91 - "test_colab_probe.py"
Cohesion: 0.70
Nodes (3): boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens()

### Community 92 - "v2.py"
Cohesion: 0.09
Nodes (20): load_local_native_model(), load_sft_adapters_for_inference(), source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), stratified_subset(), download_model() (+12 more)

### Community 95 - "test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training"
Cohesion: 0.50
Nodes (3): test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare()

### Community 96 - "telemetry.py"
Cohesion: 0.07
Nodes (13): flex_kernel_options(), format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category() (+5 more)

### Community 102 - "pathlib"
Cohesion: 0.13
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 107 - "atomic_json"
Cohesion: 0.08
Nodes (18): BenchmarkUploadQueue, atomic_json(), append_event(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush() (+10 more)

### Community 108 - "evaluation/context.py"
Cohesion: 0.08
Nodes (17): build_repo_niah_case(), completion_metrics(), _extract_match_letter(), extract_mcq_letter(), fallback_letter_regex(), format_crosscodelong_summary(), generate_repo_niah_haystack(), make_distractor_python_chunk() (+9 more)

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.08
Nodes (12): FastTrainingConfig, DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+4 more)

### Community 115 - "stable_fingerprint"
Cohesion: 0.13
Nodes (10): load_curated_pool(), stable_fingerprint(), build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, prepare_curated_training(), tokenize(), tokenize_repository_record() (+2 more)

### Community 134 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 873 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **50 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_native_tiny_model()` connect `build_native_tiny_model` to `configure_probe_training`, `test_curation.py`, `prepare_colab_pilot.py`, `config.py`, `fit_offline_sft`, `workflow`, `test_molab_matrix_notebooks.py`, `NativeTinyModel`, `BifocalGemmaForCausalLM`, `test_v2_models.py`, `pytest`, `torch`, `test_snapshot_verifies_replacement_before_pruning`, `test_memory_efficient_training.py`, `test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.057329462989840346 - nodes in this community are weakly interconnected._
- **Why does `FastTrainingPipeline` connect `FastTrainingPipeline` to `test_molab_matrix_notebooks.py`, `fast_pipeline.py`, `config.py`, `JSONLRepairDataset`, `NativeTinyModel`, `test_fast_pipeline.py`, `BifocalGemmaForCausalLM`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.057692307692307696 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `atomic_json` to `curation.py`, `sequential.py`, `prepare_colab_pilot.py`, `stages.py`, `subprocess`, `test_molab_matrix_notebooks.py`, `run_batched_context_benchmark`, `shared_execution.py`, `pytest`, `torch`, `json`, `test_v2_fixed_execution.py`, `matrix.py`, `repository_corpus.py`, `test_snapshot_verifies_replacement_before_pruning`, `training/offline_sft.py`, `matrix.py`, `workflow`, `run`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `v2.py`, `evaluation/context.py`, `stable_fingerprint`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._