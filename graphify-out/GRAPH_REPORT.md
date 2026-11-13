# Graph Report - bifocalgemma  (2026-11-13)

## Corpus Check
- 307 files · ~394,559 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2637 nodes · 7670 edges · 137 communities (92 shown, 45 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 830 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9b04faec`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_dashboard_dependency_state.py
- test_molab_matrix_notebooks.py
- stable_fingerprint
- copy
- prepare_colab_pilot.py
- hashlib
- molab_pilot_support.py
- models/checkpoint_transfer.py
- test_notebook_dependencies.py
- FeatureNotReadyError
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- test_matrix_runtime_loader.py
- test_molab_pilot.py
- schedule.py
- hashlib
- shared_execution.py
- test_snapshot_verifies_replacement_before_pruning
- build_native_tiny_model
- get_v2_experiment
- run_matrix_training
- test_tier5_adversarial.py
- test_repository_study.py
- eval_code.py
- molab_matrix_support.py
- torch
- re
- tiled_attention
- FastTrainingPipeline
- load_verified_tokenizer
- test_memory_efficient_training.py
- GatedCrossAttentionBridge
- typing
- prepare_colab_probe.py
- repository.py
- normalize_context_task
- pytest
- model.py
- NativeTinyModel
- configure_memory_efficient_training
- protocol.py
- test_context_benchmark_resume.py
- adversarial_r3_probe.py
- test_molab_remediation.py
- calibrate_shared_profile
- BifocalGemmaForCausalLM
- replace
- test_matrix_schedule_runtime.py
- evaluation/context.py
- repository_study.py
- contracts.py
- fit_offline_sft
- patches.py
- apply_word_embedding_policy
- RepoContextRouter
- reference_resilient_mcq_parser
- pathlib
- TestGemma4EdgeTopology
- JSONLRepairDataset
- repository_corpus.py
- TestBifocalGemmaArchitecture
- configure_full_rank_cross_attention
- memory_efficient.py
- test_wiggly_dashboard_lifecycle.py
- load_asset_json
- ASTGraphProjector
- pilot_e2b_e2b
- test_training_telemetry_v2.py
- adapters.py
- benchmark.py
- build_joint_visibility
- notebook_dependencies.py
- SwiGLUDualGateBridge
- memory_probe.py
- WigglyDashboard
- test_v2_workflow.py
- render_dashboard
- releases.py
- examples
- generate_full_matrix_notebooks.py
- atomic_json
- BifocalGemmaForCausalLM
- bifocalgemma
- _DummyTokenizer
- _execute_v2
- curated_sft.py
- harness.py
- wiggly_dashboard.py
- test_context_tasks.py
- test_notebook_hf_flow.py
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- benchmark_upload.py
- curation.py
- matrix.py
- json
- generation.py
- test_molab_pilot.py
- .snapshot
- FastTrainingConfig
- config.py
- PackedTokenizedDataset
- bifocalgemma/__init__.py
- test_scenario_scientific_paper_document_validation
- TestNotebookCellParity
- test_apply_waits_for_submission_and_valid_token_without_native_imports
- evaluate_humaneval_subset
- run_length_grid
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- test_shared_execution.py
- dependency_state
- run_matrix_stage
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- run_pilot_stage
- TiledEncoderAttention
- mask_tile
- prepare_sampled_benchmark_collection
- CachedAttention

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
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/models/v2/native.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_telemetry_history_appends_all_active_stages()` --calls--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_dashboard_stage_display_formatting()` --calls--> `format_stage_display()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (137 total, 45 thin omitted)

### Community 0 - "test_dashboard_dependency_state.py"
Cohesion: 0.21
Nodes (14): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_upload_and_repeated_setup_with_preloaded_marimo() (+6 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.12
Nodes (7): test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_dashboard_stage_display_formatting(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_per_model_hf_repo(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight(), test_telemetry_history_appends_all_active_stages()

### Community 2 - "stable_fingerprint"
Cohesion: 0.18
Nodes (17): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), read_source_rows() (+9 more)

### Community 3 - "copy"
Cohesion: 0.08
Nodes (17): GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), build_training_groups(), Float32Alignment, validate_training_policy(), comparison_recipes(), resolve_training_cadence() (+9 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.19
Nodes (11): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), encode_chat_pair(), extract_resolvable_span(), extract_single_window_solution_repair(), normalize_and_recount_patch(), record_context_text() (+3 more)

### Community 5 - "hashlib"
Cohesion: 0.13
Nodes (18): score_context_case(), digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids() (+10 more)

### Community 6 - "molab_pilot_support.py"
Cohesion: 0.20
Nodes (6): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_journal_integrity_and_phase_isolation(), digest(), Journal, StageStore, write_json()

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.18
Nodes (9): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_new_torch_pin_forces_reinstall_despite_old_receipt_or_matching_install(), test_preloaded_torch_requires_real_kernel_restart() (+1 more)

### Community 9 - "FeatureNotReadyError"
Cohesion: 0.11
Nodes (8): extract_token_ast_features(), TokenASTFeatures, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError, distillation_loss(), profile_experiment()

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.19
Nodes (3): _compute_per_row_cross_query(), FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.16
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 12 - "subprocess"
Cohesion: 0.09
Nodes (6): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main()

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.34
Nodes (11): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_release_check_rejects_local_preview() (+3 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.18
Nodes (8): test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), render(), main(), render(), source_bundle(), source_bundle_bytes(), upload_in_chunks()

### Community 15 - "schedule.py"
Cohesion: 0.10
Nodes (23): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+15 more)

### Community 16 - "hashlib"
Cohesion: 0.11
Nodes (15): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_exported_worker_has_parity_context_and_exact_adapter_checks() (+7 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.10
Nodes (26): retry_delay(), build_policy_optimizer(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), shared_contract(), validate_profile() (+18 more)

### Community 18 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 19 - "build_native_tiny_model"
Cohesion: 0.10
Nodes (8): build_native_tiny_model(), NativeMergedAttention, TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head(), factory(), test_fp32_alignment_on_bf16_backbone(), model()

### Community 20 - "get_v2_experiment"
Cohesion: 0.11
Nodes (14): get_v2_experiment(), cached_logits(), InferenceCache, test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only(), test_encoder_bidirectional_window_boundary_matches_native() (+6 more)

### Community 21 - "run_matrix_training"
Cohesion: 0.10
Nodes (15): benchmark_directory_name(), download_model(), target_fingerprint(), source_fingerprints(), WithoutDocstrings, latest_checkpoint(), run_training_and_benchmark(), resolve_training_contract() (+7 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (22): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+14 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.08
Nodes (16): authoritative_empirical_results(), project_root(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), CharacterTokenizer, diff(), ExecutingScoringOptions (+8 more)

### Community 24 - "eval_code.py"
Cohesion: 0.16
Nodes (6): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 25 - "molab_matrix_support.py"
Cohesion: 0.15
Nodes (4): pilot_module(), test_matrix_launcher_preserves_reported_molab_host_widget_versions(), _prepare(), prepare_runtime()

### Community 26 - "torch"
Cohesion: 0.04
Nodes (15): ModelOutput, Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), build_native_model(), run_reference_smoke(), TinyReferenceModel (+7 more)

### Community 27 - "re"
Cohesion: 0.12
Nodes (18): normalize_context_task(), repair_preimage_coverage(), curated_pool_digest(), editing_manifest(), load_curated_pool(), repository_editing_record(), validate_context_license_record(), test_repository_pool_requires_context_license_policy_and_coverage() (+10 more)

### Community 28 - "tiled_attention"
Cohesion: 0.25
Nodes (9): flex_kernel_options(), fused_attention(), attention_tile(), tiled_attention(), test_flex_backward_head256_sequence_above_small_sequence_branch(), test_flex_joint_gqa_matches_reference_on_cuda(), test_flex_structured_native_masks_across_sequence_lengths(), test_joint_gqa_output_and_gradients_at_native_head_widths() (+1 more)

### Community 30 - "load_verified_tokenizer"
Cohesion: 0.07
Nodes (36): export_pool(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_full_pool(), rows(), fetch_benchmark_files() (+28 more)

### Community 31 - "test_memory_efficient_training.py"
Cohesion: 0.22
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 34 - "typing"
Cohesion: 0.09
Nodes (13): backward_window(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids() (+5 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "repository.py"
Cohesion: 0.20
Nodes (11): ContextWindow, render_window(), select_repository_context(), _terms(), prepare_repository_record(), prepare_repository_record_with_snapshot(), repository_task_from_row(), RepositoryTask (+3 more)

### Community 37 - "normalize_context_task"
Cohesion: 0.13
Nodes (13): generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples() (+5 more)

### Community 38 - "pytest"
Cohesion: 0.05
Nodes (23): pytest_configure(), validate_empirical_results_schema(), validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f2_mcq_token_budget_specification() (+15 more)

### Community 39 - "model.py"
Cohesion: 0.15
Nodes (4): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel, build_pretrained_model()

### Community 40 - "NativeTinyModel"
Cohesion: 0.16
Nodes (3): NormalizedAlignment, NativeTinyModel, chunked_linear_cross_entropy()

### Community 41 - "configure_memory_efficient_training"
Cohesion: 0.24
Nodes (13): configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline() (+5 more)

### Community 42 - "protocol.py"
Cohesion: 0.16
Nodes (12): build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), mcq_report_metrics(), _positive_integer(), _runtime_model_identity(), scorer_fingerprint(), test_effective_allowance() (+4 more)

### Community 43 - "test_context_benchmark_resume.py"
Cohesion: 0.14
Nodes (10): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+2 more)

### Community 44 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (8): score_repo_niah_case(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases(), test_adversarial_reponiah_scorer_negation_vulnerability()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "calibrate_shared_profile"
Cohesion: 0.10
Nodes (18): task_prompt(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind() (+10 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.08
Nodes (16): BifocalGemmaConfig, _resolve_backbone_weights(), ProbeLoRALinear, test_model_to_bfloat16_preserves_fp32_master_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation() (+8 more)

### Community 48 - "replace"
Cohesion: 0.14
Nodes (8): evaluate_sft_loss(), validation_batches(), encoder_diagnostics(), TestTokenizerVerification, calibration(), test_atomic_status_retries_windows_sharing_violation_without_partial_write(), replace(), evaluate_source_copy_probes()

### Community 49 - "test_matrix_schedule_runtime.py"
Cohesion: 0.21
Nodes (8): fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), test_partial_backward_oom_replays_entire_window_without_double_gradient(), test_trained_adapters_export_reload_logit_parity()

### Community 50 - "evaluation/context.py"
Cohesion: 0.14
Nodes (9): decode_text(), build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), generate_diagnostic_text() (+1 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+10 more)

### Community 52 - "contracts.py"
Cohesion: 0.11
Nodes (16): RepairExample, RepositoryFile, RepositorySnapshot, example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash() (+8 more)

### Community 53 - "fit_offline_sft"
Cohesion: 0.09
Nodes (18): cpu_tree(), file_digest(), fit_offline_sft(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest(), update_policy_learning_rates(), retained_checkpoints() (+10 more)

### Community 54 - "patches.py"
Cohesion: 0.17
Nodes (13): canonical_snapshot_hash(), apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), NativeTaskResult, score_python_patch(), summarize_native_results() (+5 more)

### Community 55 - "apply_word_embedding_policy"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 57 - "reference_resilient_mcq_parser"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 58 - "pathlib"
Cohesion: 0.07
Nodes (7): remote_reset_plan(), reset_state(), loss_chart(), test_external_activity_reader_does_not_need_running_kernel(), test_training_failure_never_launches_benchmarks(), inspect(), main()

### Community 61 - "JSONLRepairDataset"
Cohesion: 0.10
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_with_jsonl_data_dir()

### Community 62 - "repository_corpus.py"
Cohesion: 0.08
Nodes (20): ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license() (+12 more)

### Community 64 - "configure_full_rank_cross_attention"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 65 - "memory_efficient.py"
Cohesion: 0.21
Nodes (5): joint_visibility_tile(), sample_encoder_mass(), stratified_subset(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), test_pilot_subset_is_deterministic_and_stratified()

### Community 66 - "test_wiggly_dashboard_lifecycle.py"
Cohesion: 0.16
Nodes (6): model_opens(), RecordingStream, test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 67 - "load_asset_json"
Cohesion: 0.18
Nodes (3): ensure_assets_on_disk(), load_asset_json(), native_config()

### Community 69 - "pilot_e2b_e2b"
Cohesion: 0.22
Nodes (6): target_fingerprint(), configure_full_rank_cross_attention(), __init__(), pilot_e2b_e2b(), save_report(), tokenize_row()

### Community 70 - "test_training_telemetry_v2.py"
Cohesion: 0.22
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 72 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 74 - "benchmark.py"
Cohesion: 0.13
Nodes (14): decoder_prompt(), prompt_ids(), run_context_benchmark(), activity(), emit(), model_stage(), resource_snapshot(), DurableClock (+6 more)

### Community 76 - "build_joint_visibility"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 77 - "notebook_dependencies.py"
Cohesion: 0.20
Nodes (5): test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols(), test_source_bundle_checksum_failure_does_not_activate_source(), activate_notebook_bundle(), import_native_gemma4()

### Community 79 - "memory_probe.py"
Cohesion: 0.13
Nodes (10): export_matrix_pretrained_model(), configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), tensor_bytes(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_export_matrix_pretrained_model() (+2 more)

### Community 81 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), workflow(), benchmark(), download() (+4 more)

### Community 82 - "render_dashboard"
Cohesion: 0.25
Nodes (4): format_duration(), format_stage_display(), render_dashboard(), test_monitor_escapes_errors_updates_validation_and_exports_html()

### Community 83 - "releases.py"
Cohesion: 0.08
Nodes (31): archive_files(), audit_split_disjointness(), build_release(), digest(), main(), publish_release(), rows(), unpack_archive() (+23 more)

### Community 84 - "examples"
Cohesion: 0.29
Nodes (9): fail_second_microbatch(), original(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), measure(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence() (+1 more)

### Community 85 - "generate_full_matrix_notebooks.py"
Cohesion: 0.36
Nodes (5): test_invalid_generation_modes_cannot_publish(), format_code(), main(), matrix_bundle(), render_notebook()

### Community 87 - "atomic_json"
Cohesion: 0.09
Nodes (19): file_digest(), build_packed_cache(), open_packed_cache(), checked_prefix(), recover_remote_run(), RemoteHeartbeat, retry_delay(), _synchronize_once() (+11 more)

### Community 92 - "_execute_v2"
Cohesion: 0.07
Nodes (24): fetch_collection(), reset_local_benchmarks(), prepare_curated_training(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint() (+16 more)

### Community 93 - "curated_sft.py"
Cohesion: 0.09
Nodes (17): tokenize_context_task(), prepare_editing_records(), split_editing_records(), run_training_memory_preflight(), fit_curated_sft(), tokenize(), default_training_profiles(), fit_repository_sft() (+9 more)

### Community 96 - "wiggly_dashboard.py"
Cohesion: 0.20
Nodes (3): LastSeenWidget, phase_category(), TrainingLossLog

### Community 101 - "test_context_tasks.py"
Cohesion: 0.14
Nodes (8): _extract_match_letter(), fallback_letter_regex(), strict_letter(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 103 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 104 - "benchmark_upload.py"
Cohesion: 0.12
Nodes (7): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos()

### Community 105 - "curation.py"
Cohesion: 0.09
Nodes (32): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), build_benchmark_index() (+24 more)

### Community 106 - "matrix.py"
Cohesion: 0.08
Nodes (15): GreedyGeneration, Bf16FullLinear, configure_full_rank_cross_attention(), FullCrossCachedAttention, batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), build_bifocalgemma_v3_optimizer() (+7 more)

### Community 107 - "json"
Cohesion: 0.10
Nodes (13): compare_predictions(), audit(), main(), audit(), emit(), main(), validate_all_patches(), main() (+5 more)

### Community 108 - "generation.py"
Cohesion: 0.17
Nodes (7): DeveloperAgent, RepairPrediction, format_encoder_context(), write_swebench_predictions(), generate_patch(), repair_prompt_ids(), _safe_apply_chat_template()

### Community 109 - "test_molab_pilot.py"
Cohesion: 0.13
Nodes (12): test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), journaled_loss(), evaluate(), PilotStorage (+4 more)

### Community 113 - "FastTrainingConfig"
Cohesion: 0.06
Nodes (14): FastTrainingConfig, DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+6 more)

### Community 114 - "config.py"
Cohesion: 0.11
Nodes (14): BackboneProfile, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan (+6 more)

### Community 123 - "evaluate_humaneval_subset"
Cohesion: 0.33
Nodes (4): evaluate_humaneval_subset(), evaluate_pilot_checkpoint(), main(), render()

### Community 124 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 127 - "test_shared_execution.py"
Cohesion: 0.09
Nodes (9): __getattr__(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_shared_profile_upload_retries_without_losing_local_selection() (+1 more)

### Community 129 - "run_matrix_stage"
Cohesion: 0.12
Nodes (14): CheckpointRef, prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), make_experiment_spec(), run_matrix_stage(), test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold() (+6 more)

### Community 131 - "run_pilot_stage"
Cohesion: 0.40
Nodes (3): configure_pilot_stage(), is_cross_bridge_parameter(), run_pilot_stage()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 981 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `stable_fingerprint`, `copy`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `hashlib`, `prepare_sampled_benchmark_collection`, `run_pilot_stage`, `subprocess`, `shared_execution.py`, `test_snapshot_verifies_replacement_before_pruning`, `get_v2_experiment`, `run_matrix_training`, `torch`, `calibrate_shared_profile`, `replace`, `evaluation/context.py`, `pathlib`, `pilot_e2b_e2b`, `benchmark.py`, `test_v2_workflow.py`, `releases.py`, `examples`, `_execute_v2`, `benchmark_upload.py`, `matrix.py`, `json`, `test_molab_pilot.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11695906432748537 - nodes in this community are weakly interconnected._
- **Why does `FeatureNotReadyError` connect `FeatureNotReadyError` to `model.py`, `adapters.py`, `generation.py`, `config.py`, `torch`, `harness.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Should `copy` be split into smaller, more focused modules?**
  _Cohesion score 0.08309178743961353 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `hashlib`, `shared_execution.py`, `run_matrix_training`, `torch`, `re`, `load_verified_tokenizer`, `calibrate_shared_profile`, `test_matrix_schedule_runtime.py`, `evaluation/context.py`, `pathlib`, `repository_corpus.py`, `benchmark.py`, `releases.py`, `atomic_json`, `_execute_v2`, `curated_sft.py`, `curation.py`, `matrix.py`, `json`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._