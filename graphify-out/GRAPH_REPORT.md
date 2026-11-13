# Graph Report - bifocalgemma  (2026-11-13)

## Corpus Check
- 307 files · ~393,654 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2634 nodes · 7664 edges · 136 communities (92 shown, 44 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 829 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `57a1fe78`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_dashboard_dependency_state.py
- test_molab_matrix_notebooks.py
- stable_fingerprint
- math
- prepare_colab_pilot.py
- hashlib
- molab_pilot_support.py
- models/checkpoint_transfer.py
- test_notebook_dependencies.py
- full_pool.py
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- test_matrix_runtime_loader.py
- test_molab_pilot.py
- schedule.py
- hashlib
- shared_execution.py
- test_snapshot_verifies_replacement_before_pruning
- native.py
- atomic_json
- run_matrix_training
- test_tier5_adversarial.py
- test_repository_study.py
- test_data_remediation.py
- molab_matrix_support.py
- TokenizedRepairExample
- test_editing_manifest.py
- tiled_attention
- FastTrainingPipeline
- remediation.py
- test_memory_efficient_training.py
- GatedCrossAttentionBridge
- typing
- prepare_colab_probe.py
- repository.py
- normalize_context_task
- test_tier1_features.py
- default_training_profiles
- NativeTinyModel
- random
- protocol.py
- test_context_benchmark_resume.py
- adversarial_r3_probe.py
- test_molab_remediation.py
- calibrate_shared_profile
- BifocalGemmaForCausalLM
- pathlib
- test_matrix_schedule_runtime.py
- evaluation/context.py
- repository_study.py
- contracts.py
- examples
- patches.py
- test_construction_blocks.py
- RepoContextRouter
- reference_resilient_mcq_parser
- stages.py
- TestGemma4EdgeTopology
- JSONLRepairDataset
- repository_corpus.py
- TestBifocalGemmaArchitecture
- configure_full_rank_cross_attention
- torch
- test_wiggly_dashboard_lifecycle.py
- test_curation.py
- ASTGraphProjector
- training/offline_sft.py
- test_offline_sft_training.py
- adapters.py
- benchmark.py
- build_joint_visibility
- notebook_dependencies.py
- CodeOverlapIndex
- memory_probe.py
- WigglyDashboard
- test_v2_workflow.py
- telemetry.py
- releases.py
- test_v2_fixed_execution.py
- Any
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- BifocalGemmaForCausalLM
- bifocalgemma
- test_colab_pilot.py
- v2.py
- data/offline_sft.py
- harness.py
- TrainingLossLog
- test_context_tasks.py
- test_notebook_hf_flow.py
- test_combination_eval_metrics_to_paper_data_serialization
- benchmark_upload.py
- curation.py
- matrix.py
- json
- generation.py
- test_loss_journal_preserves_token_weighting_and_reuses_examples
- resolve_optimal_eval_batch_size
- test_fast_pipeline.py
- config.py
- packed_tokens.py
- fast_pipeline.py
- test_scenario_scientific_paper_document_validation
- optimizers.py
- test_apply_waits_for_submission_and_valid_token_without_native_imports
- prepare_colab_eval.py
- test_benchmark_prompt_budget.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- test_shared_execution.py
- dependency_state
- test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold
- test_failed_final_upload_remains_pending
- project_root
- TiledEncoderAttention
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- flex_attention.py

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
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_telemetry_history_appends_all_active_stages()` --calls--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_fast_training_config_step_reconciliation()` --calls--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/models/v2/native.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (136 total, 44 thin omitted)

### Community 0 - "test_dashboard_dependency_state.py"
Cohesion: 0.22
Nodes (13): test_changed_transitive_loaded_ui_dependency_also_requires_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_upload_and_repeated_setup_with_preloaded_marimo() (+5 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.06
Nodes (38): get_v2_experiment(), InferenceCache, configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), batched_greedy_decode(), make_experiment_spec(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes() (+30 more)

### Community 2 - "stable_fingerprint"
Cohesion: 0.14
Nodes (20): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), repair_preimage_coverage(), stable_fingerprint() (+12 more)

### Community 3 - "math"
Cohesion: 0.13
Nodes (10): fit_curated_sft(), validate_optimizer_config(), build_training_groups(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence(), run_recipe_comparison() (+2 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.08
Nodes (30): evaluate_sft_loss(), validation_batches(), encoder_diagnostics(), test_train_fast_cli_baseline_2b(), test_atomic_status_retries_windows_sharing_violation_without_partial_write(), replace(), build_full_cross_cohorts(), build_source_copy_probes() (+22 more)

### Community 5 - "hashlib"
Cohesion: 0.19
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 6 - "molab_pilot_support.py"
Cohesion: 0.13
Nodes (11): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), digest(), Journal, journaled_loss(), PilotStorage, run() (+3 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.18
Nodes (8): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_new_torch_pin_forces_reinstall_despite_old_receipt_or_matching_install(), test_preloaded_torch_requires_real_kernel_restart(), test_preloaded_torchvision_alone_requires_restart()

### Community 9 - "full_pool.py"
Cohesion: 0.18
Nodes (10): read_arrow_rows(), split_connected_records(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+2 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.16
Nodes (4): _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 12 - "subprocess"
Cohesion: 0.10
Nodes (7): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.31
Nodes (11): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_release_check_rejects_local_preview() (+3 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.16
Nodes (12): test_invalid_generation_modes_cannot_publish(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render() (+4 more)

### Community 15 - "schedule.py"
Cohesion: 0.11
Nodes (22): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+14 more)

### Community 16 - "hashlib"
Cohesion: 0.14
Nodes (17): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), ContextWindow, render_window(), select_repository_context(), _terms() (+9 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.16
Nodes (14): publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), current_capacity(), execution_settings(), _gpu_context() (+6 more)

### Community 18 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 19 - "native.py"
Cohesion: 0.06
Nodes (11): NormalizedAlignment, build_native_tiny_model(), native_config(), NativeMergedAttention, run_native_smoke(), RoutedRepoBatch, TestNativeIntegration, version() (+3 more)

### Community 20 - "atomic_json"
Cohesion: 0.12
Nodes (11): ArchiveSnapshotStore, atomic_json(), test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main(), main(), source_notices() (+3 more)

### Community 21 - "run_matrix_training"
Cohesion: 0.08
Nodes (15): benchmark_directory_name(), RemoteHeartbeat, retry_delay(), target_fingerprint(), source_fingerprints(), WithoutDocstrings, run_training_and_benchmark(), resolve_training_contract() (+7 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+15 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.11
Nodes (15): canonical_snapshot_hash(), NativeTaskResult, score_python_patch(), summarize_native_results(), CharacterTokenizer, diff(), ExecutingScoringOptions, fixture() (+7 more)

### Community 24 - "test_data_remediation.py"
Cohesion: 0.30
Nodes (11): remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage(), test_heldout_role_rejected() (+3 more)

### Community 25 - "molab_matrix_support.py"
Cohesion: 0.16
Nodes (3): pilot_module(), _prepare(), prepare_runtime()

### Community 26 - "TokenizedRepairExample"
Cohesion: 0.09
Nodes (14): Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), run_length_grid(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens() (+6 more)

### Community 27 - "test_editing_manifest.py"
Cohesion: 0.29
Nodes (9): editing_manifest(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation() (+1 more)

### Community 28 - "tiled_attention"
Cohesion: 0.31
Nodes (8): fused_attention(), attention_tile(), tiled_attention(), test_flex_backward_head256_sequence_above_small_sequence_branch(), test_flex_joint_gqa_matches_reference_on_cuda(), test_flex_structured_native_masks_across_sequence_lengths(), test_joint_gqa_output_and_gradients_at_native_head_widths(), forward()

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.07
Nodes (15): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+7 more)

### Community 30 - "remediation.py"
Cohesion: 0.14
Nodes (11): token_profile(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+3 more)

### Community 31 - "test_memory_efficient_training.py"
Cohesion: 0.11
Nodes (16): run_training_memory_preflight(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+8 more)

### Community 34 - "typing"
Cohesion: 0.15
Nodes (8): compute_diff_token_weights(), differential_diff_cross_entropy(), _pointer_chunk_forward(), _streaming_lse_pointer_nll(), streaming_pointer_auxiliary_loss(), StreamingPointerLoss, test_pointer_loss_gradient_checkpointing_and_trivial_filtering(), test_v3_streaming_pointer_loss_and_differential_diff_weighting()

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "repository.py"
Cohesion: 0.21
Nodes (9): git_environment(), GitSnapshotStore, prepare_repository_record(), prepare_repository_record_with_snapshot(), repo_identity(), repository_task_from_row(), RepositoryTask, test_quarantine_fork_alias_and_grouped_ingestion() (+1 more)

### Community 37 - "normalize_context_task"
Cohesion: 0.16
Nodes (13): normalize_context_task(), task_prompt(), tokenize_context_task(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record() (+5 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval(), test_f6_repo_core_syntax_tests_executable() (+2 more)

### Community 39 - "default_training_profiles"
Cohesion: 0.22
Nodes (4): default_training_profiles(), fit_repository_sft(), make_identity(), TestTrainingPolicy

### Community 40 - "NativeTinyModel"
Cohesion: 0.18
Nodes (3): ModelOutput, build_native_model(), NativeTinyModel

### Community 41 - "random"
Cohesion: 0.27
Nodes (5): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 42 - "protocol.py"
Cohesion: 0.15
Nodes (12): build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), mcq_report_metrics(), _positive_integer(), _runtime_model_identity(), scorer_fingerprint(), test_effective_allowance() (+4 more)

### Community 43 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 44 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (11): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast() (+3 more)

### Community 46 - "calibrate_shared_profile"
Cohesion: 0.18
Nodes (9): benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe() (+1 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.08
Nodes (17): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), ProbeLoRALinear, test_model_to_bfloat16_preserves_fp32_master_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization() (+9 more)

### Community 48 - "pathlib"
Cohesion: 0.09
Nodes (10): pytest_configure(), reference_resilient_mcq_parser(), validate_empirical_results_schema(), validate_paper_manuscript(), test_f2_mcq_token_budget_specification(), test_f8_scientific_paper_structure(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation() (+2 more)

### Community 49 - "test_matrix_schedule_runtime.py"
Cohesion: 0.19
Nodes (8): fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), test_partial_backward_oom_replays_entire_window_without_double_gradient(), test_trained_adapters_export_reload_logit_parity()

### Community 50 - "evaluation/context.py"
Cohesion: 0.11
Nodes (11): build_repo_niah_case(), _extract_match_letter(), fallback_letter_regex(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval() (+3 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+10 more)

### Community 52 - "contracts.py"
Cohesion: 0.10
Nodes (18): CheckpointRef, RepairExample, RepositoryFile, RepositorySnapshot, example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord (+10 more)

### Community 53 - "examples"
Cohesion: 0.07
Nodes (25): comparison_recipes(), examples(), TestOfflineSFT, TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored() (+17 more)

### Community 54 - "patches.py"
Cohesion: 0.14
Nodes (10): apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), validate_code_eval_snapshots(), patch_syntax(), test_missing_old_file_is_still_rejected(), test_modify_add_delete_preserves_unrelated_unicode_bytes(), test_multifile_add_delete_unicode_and_path_guards() (+2 more)

### Community 55 - "test_construction_blocks.py"
Cohesion: 0.08
Nodes (10): apply_word_embedding_policy(), EmbeddingPolicyResult, ReferenceMergedAttention, TinyReferenceModel, __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy(), TestMergedAttentionReference (+2 more)

### Community 57 - "reference_resilient_mcq_parser"
Cohesion: 0.17
Nodes (4): test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 58 - "stages.py"
Cohesion: 0.14
Nodes (5): remote_reset_plan(), reset_local_benchmarks(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), main()

### Community 61 - "JSONLRepairDataset"
Cohesion: 0.11
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 62 - "repository_corpus.py"
Cohesion: 0.10
Nodes (18): safe_repo_path(), build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license() (+10 more)

### Community 64 - "configure_full_rank_cross_attention"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 65 - "torch"
Cohesion: 0.30
Nodes (4): joint_visibility_tile(), mask_tile(), _TiledAttention, sample_encoder_mass()

### Community 67 - "test_curation.py"
Cohesion: 0.19
Nodes (11): export_pool(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_frozen_budget_matches_all_six_controls(), test_frozen_pool_verifies_checksums_before_use(), test_invalid_requested_pool_budget_is_rejected(), test_license_repo_and_benchmark_gates() (+3 more)

### Community 69 - "training/offline_sft.py"
Cohesion: 0.29
Nodes (7): cpu_tree(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest(), retained_checkpoints(), retry_windows_file_operation()

### Community 70 - "test_offline_sft_training.py"
Cohesion: 0.09
Nodes (6): TestNotebookCellParity, TestOfflineEditing, test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 72 - "adapters.py"
Cohesion: 0.19
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 74 - "benchmark.py"
Cohesion: 0.13
Nodes (14): decoder_prompt(), run_context_benchmark(), activity(), score_context_case(), fit_offline_sft(), emit(), model_stage(), resource_snapshot() (+6 more)

### Community 77 - "notebook_dependencies.py"
Cohesion: 0.20
Nodes (5): test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols(), test_source_bundle_checksum_failure_does_not_activate_source(), activate_notebook_bundle(), import_native_gemma4()

### Community 78 - "CodeOverlapIndex"
Cohesion: 0.35
Nodes (10): code_shingles(), CodeOverlapIndex, test_disk_index_matches_memory_and_syntax_repairs_are_admitted(), test_full_overlap_index_skips_impossible_blob_reads(), test_near_matching_and_short_fragment_policy(), test_overlap_bound_preserves_near_matches_and_heuristic_index(), test_readonly_partial_postings_never_use_full_intersection_bound(), test_index_with_legacy_normalization_is_rejected() (+2 more)

### Community 79 - "memory_probe.py"
Cohesion: 0.22
Nodes (7): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), TestMemoryProbe

### Community 80 - "WigglyDashboard"
Cohesion: 0.22
Nodes (7): phase_category(), WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 81 - "test_v2_workflow.py"
Cohesion: 0.11
Nodes (13): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+5 more)

### Community 82 - "telemetry.py"
Cohesion: 0.16
Nodes (5): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), LastSeenWidget

### Community 83 - "releases.py"
Cohesion: 0.08
Nodes (25): archive_files(), audit_split_disjointness(), build_release(), digest(), main(), publish_release(), rows(), unpack_archive() (+17 more)

### Community 84 - "test_v2_fixed_execution.py"
Cohesion: 0.11
Nodes (11): calibration(), guard(), examples(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), measure(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence() (+3 more)

### Community 85 - "Any"
Cohesion: 0.22
Nodes (4): authoritative_empirical_results(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 87 - "test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume"
Cohesion: 0.17
Nodes (3): chunked_differential_cross_entropy(), checkpoint(), test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume()

### Community 91 - "test_colab_pilot.py"
Cohesion: 0.22
Nodes (3): _DummyTokenizer, test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order()

### Community 92 - "v2.py"
Cohesion: 0.05
Nodes (45): load_asset_json(), file_digest(), fetch_collection(), load_local_native_model(), load_sft_adapters_for_inference(), BenchmarkUploadQueue, prepare_curated_training(), checked_prefix() (+37 more)

### Community 93 - "data/offline_sft.py"
Cohesion: 0.16
Nodes (10): curated_pool_digest(), git_text_lines(), load_curated_pool(), prepare_editing_records(), split_editing_records(), tokenize(), prepare_repository_training(), tokenize_repository_record() (+2 more)

### Community 101 - "test_context_tasks.py"
Cohesion: 0.33
Nodes (4): task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 104 - "benchmark_upload.py"
Cohesion: 0.12
Nodes (7): retry_delay(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_retry_after_date_and_network_failures()

### Community 105 - "curation.py"
Cohesion: 0.12
Nodes (14): code_signature(), code_tokens(), CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target() (+6 more)

### Community 106 - "matrix.py"
Cohesion: 0.05
Nodes (22): Bf16FullLinear, configure_full_rank_cross_attention(), SwiGLUDualGateBridge, load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint() (+14 more)

### Community 107 - "json"
Cohesion: 0.10
Nodes (17): compare_predictions(), prompt_ids(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), audit(), main(), audit(), emit() (+9 more)

### Community 108 - "generation.py"
Cohesion: 0.12
Nodes (11): RepairPrediction, decode_text(), write_swebench_predictions(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids() (+3 more)

### Community 109 - "test_loss_journal_preserves_token_weighting_and_reuses_examples"
Cohesion: 0.29
Nodes (6): fail_second_microbatch(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), evaluate(), event_callback(), event()

### Community 111 - "resolve_optimal_eval_batch_size"
Cohesion: 0.40
Nodes (3): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.10
Nodes (3): DummyTrackingTokenizer, test_jsonl_dataset_query_injection_into_encoder(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step()

### Community 114 - "config.py"
Cohesion: 0.04
Nodes (27): DeveloperAgent, restore_checkpoint(), save_checkpoint(), BackboneProfile, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix() (+19 more)

### Community 115 - "packed_tokens.py"
Cohesion: 0.17
Nodes (5): build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 118 - "optimizers.py"
Cohesion: 0.21
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), probe_optimizer(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 124 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 127 - "test_shared_execution.py"
Cohesion: 0.09
Nodes (15): seal_report(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline() (+7 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 980 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **44 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `test_molab_matrix_notebooks.py`, `stable_fingerprint`, `prepare_colab_pilot.py`, `shared_execution.py`, `test_snapshot_verifies_replacement_before_pruning`, `run_matrix_training`, `calibrate_shared_profile`, `evaluation/context.py`, `examples`, `patches.py`, `stages.py`, `training/offline_sft.py`, `benchmark.py`, `test_v2_workflow.py`, `test_v2_fixed_execution.py`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `v2.py`, `benchmark_upload.py`, `matrix.py`, `json`, `test_loss_journal_preserves_token_weighting_and_reuses_examples`, `packed_tokens.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06101190476190476 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `test_molab_remediation.py`, `BifocalGemmaForCausalLM`, `test_fast_pipeline.py`, `fast_pipeline.py`, `FastTrainingPipeline`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Should `stable_fingerprint` be split into smaller, more focused modules?**
  _Cohesion score 0.14022988505747128 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `test_molab_matrix_notebooks.py`, `math`, `prepare_colab_pilot.py`, `shared_execution.py`, `native.py`, `test_editing_manifest.py`, `test_memory_efficient_training.py`, `prepare_colab_probe.py`, `normalize_context_task`, `random`, `test_matrix_schedule_runtime.py`, `contracts.py`, `examples`, `test_construction_blocks.py`, `JSONLRepairDataset`, `torch`, `test_curation.py`, `memory_probe.py`, `test_colab_pilot.py`, `data/offline_sft.py`, `test_loss_journal_preserves_token_weighting_and_reuses_examples`, `config.py`, `packed_tokens.py`, `fast_pipeline.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._