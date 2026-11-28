# Graph Report - bifocalgemma  (2026-11-28)

## Corpus Check
- 262 files · ~382,731 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 17 file(s) not represented in the graph (top: .tex 5, (none) 2, .jsonl 2)

## Summary
- 2746 nodes · 7509 edges · 188 communities (97 shown, 91 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 863 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `78351c85`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- load_verified_tokenizer
- test_tier1_features.py
- benchmark_upload.py
- FastTrainingConfig
- RepositorySnapshot
- atomic_json
- test_notebook_dependencies.py
- test_memory_efficient_training.py
- tokenization.py
- full_pool.py
- memory_probe.py
- schedule.py
- run_matrix_stage
- test_repository_study.py
- TokenizedRepairExample
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- reference_resilient_mcq_parser
- NativeTinyModel
- pathlib
- build_native_tiny_model
- adversarial_r3_probe.py
- gsm8k.py
- FastTrainingPipeline
- inspect_python_target
- test_gsm8k.py
- curation.py
- test_training_telemetry.py
- run_external_benchmark
- json
- load_local_native_model
- prepare_context_data.py
- atomic_json
- test_recover_remote_run_ignores_dot_temp_and_heartbeat_files
- TrainingDashboard
- SwiGLUDualGateBridge
- run_batched_context_benchmark
- build_repository_corpus
- evaluation/context.py
- BifocalGemmaForCausalLM
- test_data_remediation.py
- repository_sft.py
- TestBifocalGemmaArchitecture
- test_dashboard_dependency_state.py
- repository_study.py
- test_context_benchmark_resume.py
- matrix.py
- test_v4_checkpoint_and_patches.py
- test_matrix_schedule_runtime.py
- json
- test_v4_architecture.py
- compute_step_loss
- hashlib
- test_shared_schedule.py
- is_cross_bridge_parameter
- fit_offline_sft
- test_editing_manifest.py
- stable_fingerprint
- torch
- test_context_tasks.py
- run_batched_context_benchmark
- test_repository_corpus.py
- extract_mcq_letter
- test_construction_blocks.py
- .snapshot
- examples
- test_training_recipes.py
- scorer_fingerprint
- optimizers.py
- benchmark.py
- .config
- test_v4_optimizer_schedule.py
- training/diagnostics.py
- source_fingerprints
- bifocalgemma
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- MockQwenTokenizer
- test_v4_data_disjoint.py
- dependency_state
- test_tier1_features.py
- test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues
- _ProportionalTokenizer
- Seq2SeqBatch
- eval_external_baselines.py
- BifocalGemmaConfig
- WigglyDashboard
- patch_matrix_tokenization
- test_fast_pipeline.py
- test_effective_benchmark_protocol.py
- test_matrix_runtime_loader.py
- make_experiment_spec
- patch_cross_entropy
- wiggly_dashboard.py
- resolve_diff_token_ids
- RepositorySnapshot
- RecordingStream
- test_adversarial_mcq_isolated_formatting
- test_molab_remediation.py
- test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract
- NativeMergedAttention
- JSONLRepairDataset
- test_adversarial_mcq_redos_scaling
- modeling_bifocalgemma.py
- test_context_benchmark_mcq_budget_floor_execution
- packed_tokens.py
- test_adversarial_mcq_ruling_out_distractors_vulnerability
- molab_matrix_support.py
- TinyLossModel
- FullCrossCachedAttention
- bifocalgemma/__init__.py
- notebook_launcher.py
- get_v2_experiment
- context_tasks.py
- ast
- test_molab_matrix_notebooks.py

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 65 edges
2. `atomic_json()` - 61 edges
3. `BifocalGemmaForCausalLM` - 55 edges
4. `run_matrix_training()` - 53 edges
5. `load_verified_tokenizer()` - 51 edges
6. `RepositorySnapshot` - 45 edges
7. `run_matrix_benchmark()` - 41 edges
8. `TokenizedRepairExample` - 40 edges
9. `BifocalGemmaConfig` - 39 edges
10. `FastTrainingPipeline` - 38 edges

## Surprising Connections (you probably didn't know these)
- `test_make_experiment_spec_and_hub_prefix_for_all_ten_models()` --calls--> `checked_prefix()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/hub.py
- `test_v3_critical_attention_beta_and_triton_tile_clamping()` --calls--> `_compute_per_row_cross_query()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/models/cross_attention.py
- `test_pipeline_state_ledger_creation_for_remote_resumption()` --calls--> `StageLedger`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/stages.py
- `_run_loss()` --calls--> `streaming_pointer_auxiliary_loss()`  [INFERRED]
  tests/test_sliding_bifocal_routing.py → src/bifocalgemma/training/objectives.py
- `test_fast_training_config_step_reconciliation()` --uses--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (188 total, 91 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.20
Nodes (9): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution(), save() (+1 more)

### Community 1 - "load_verified_tokenizer"
Cohesion: 0.12
Nodes (14): CheckpointRef, ExperimentSpec, _pair(), _ref(), decode_text(), format_encoder_context(), load_verified_tokenizer(), tokenize_repair_example() (+6 more)

### Community 2 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 3 - "benchmark_upload.py"
Cohesion: 0.10
Nodes (12): BenchmarkUploadQueue, is_transient_lfs_pointer_error(), retry_delay(), append_event(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden() (+4 more)

### Community 5 - "FastTrainingConfig"
Cohesion: 0.11
Nodes (10): validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), FastTrainingConfig, TestConstructionEntrypoints, test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_pipeline_skips_training_when_already_completed() (+2 more)

### Community 6 - "RepositorySnapshot"
Cohesion: 0.10
Nodes (27): RepositoryFile, RepositorySnapshot, _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application() (+19 more)

### Community 7 - "atomic_json"
Cohesion: 0.15
Nodes (16): build_and_verify_mixed_pool(), build_and_verify_release(), _build_v4_notices(), _extract_python_fim_from_code(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers(), _log() (+8 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.11
Nodes (14): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+6 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.25
Nodes (13): configure_memory_efficient_training(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+5 more)

### Community 11 - "tokenization.py"
Cohesion: 0.09
Nodes (13): DeveloperAgent, RepairExample, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter (+5 more)

### Community 13 - "full_pool.py"
Cohesion: 0.15
Nodes (11): read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+3 more)

### Community 14 - "memory_probe.py"
Cohesion: 0.15
Nodes (10): gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), run_length_grid(), tensor_bytes(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_length_grid_matches_baseline_budget_and_stops_after_failure() (+2 more)

### Community 15 - "schedule.py"
Cohesion: 0.10
Nodes (19): prepare_curated_training(), _build_schedule_for_epochs(), _validate_num_epochs(), _build_group_schedule(), resolve_training_cadence(), _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records() (+11 more)

### Community 16 - "run_matrix_stage"
Cohesion: 0.21
Nodes (10): pack_window_microbatches(), run_matrix_stage(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_length_grouped_microbatch_packing_preserves_gradients_and_backoff(), test_partial_backward_oom_replays_entire_window_without_double_gradient() (+2 more)

### Community 17 - "test_repository_study.py"
Cohesion: 0.08
Nodes (24): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), v4_repository_task_from_row(), example_from_record(), prepare_repository_record() (+16 more)

### Community 18 - "TokenizedRepairExample"
Cohesion: 0.11
Nodes (11): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), fit_curated_sft(), synthetic_stress_target(), TestDataAndLoss, test_curated_resume_rejects_runtime_drift() (+3 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.16
Nodes (6): MockCausalLM, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(), test_windowed_file_section_disjoint_split_and_train_eval_parity(), _write_repair_fixture()

### Community 20 - "eval_code.py"
Cohesion: 0.15
Nodes (6): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.11
Nodes (11): test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_metrics_disjoint_identifiers(), test_adversarial_metrics_identical_sequences(), test_adversarial_metrics_keyword_identifier_inflation(), test_adversarial_metrics_multiline_diff_inflation(), test_adversarial_metrics_pure_operator_zero_division_guard() (+3 more)

### Community 23 - "reference_resilient_mcq_parser"
Cohesion: 0.14
Nodes (6): test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 24 - "NativeTinyModel"
Cohesion: 0.21
Nodes (5): ModelOutput, NativeTinyModel, test_cross_scale_frozen_by_default_and_configurable(), test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), spy_sdpa()

### Community 25 - "pathlib"
Cohesion: 0.12
Nodes (6): test_repository_notebook_bundle_is_synchronized(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 26 - "build_native_tiny_model"
Cohesion: 0.16
Nodes (4): build_native_tiny_model(), TestNativeIntegration, TestMemoryProbe, test_last_token_logits_match_full_head()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "gsm8k.py"
Cohesion: 0.17
Nodes (13): _clean_extracted_number(), digest(), fetch_gsm8k_sources(), fingerprint(), generate_run(), gsm8k_prompt_ids(), load_gsm8k(), main() (+5 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.10
Nodes (3): FastTrainingPipeline, run_fast_training(), test_fast_training_pipeline_end_to_end_smoke()

### Community 30 - "inspect_python_target"
Cohesion: 0.19
Nodes (7): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples(), test_warning_has_provenance_without_stderr_or_source_mutation()

### Community 31 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 32 - "curation.py"
Cohesion: 0.10
Nodes (29): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+21 more)

### Community 33 - "test_training_telemetry.py"
Cohesion: 0.17
Nodes (3): test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime()

### Community 34 - "run_external_benchmark"
Cohesion: 0.14
Nodes (13): format_crosscodelong_summary(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort(), load_hf_external_model_and_tokenizer() (+5 more)

### Community 35 - "json"
Cohesion: 0.09
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 36 - "load_local_native_model"
Cohesion: 0.21
Nodes (5): load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), TestCheckpointTransfer, test_local_loader_preflights_and_preserves_shared_words()

### Community 37 - "prepare_context_data.py"
Cohesion: 0.10
Nodes (23): adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), completion_from_snapshot(), completions_from_snapshot(), context_pool() (+15 more)

### Community 38 - "atomic_json"
Cohesion: 0.06
Nodes (22): file_digest(), load_matrix_adapters_for_inference(), _milestone_checkpoint_names(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix() (+14 more)

### Community 39 - "test_recover_remote_run_ignores_dot_temp_and_heartbeat_files"
Cohesion: 0.15
Nodes (3): download(), test_recover_remote_run_ignores_dot_temp_and_heartbeat_files(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only()

### Community 40 - "TrainingDashboard"
Cohesion: 0.13
Nodes (8): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_dashboard_stage_display_formatting(), test_telemetry_history_appends_all_active_stages()

### Community 41 - "SwiGLUDualGateBridge"
Cohesion: 0.13
Nodes (4): NormalizedAlignment, SwiGLUDualGateBridge, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init()

### Community 42 - "run_batched_context_benchmark"
Cohesion: 0.14
Nodes (14): _paired_budget_for(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer(), repair_report_metrics() (+6 more)

### Community 43 - "build_repository_corpus"
Cohesion: 0.16
Nodes (6): ArchiveSnapshotStore, build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_failed_download_removes_partial_bytes(), test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 44 - "evaluation/context.py"
Cohesion: 0.06
Nodes (43): _build_paired_eval_header_and_row(), build_repo_niah_case(), completion_metrics(), _context_already_includes_instruction(), _count_niah_tokens(), decoder_prompt(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens() (+35 more)

### Community 45 - "BifocalGemmaForCausalLM"
Cohesion: 0.06
Nodes (17): BifocalGemmaForCausalLM, build_native_model(), export_matrix_pretrained_model(), _resolve_backbone_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload() (+9 more)

### Community 47 - "test_data_remediation.py"
Cohesion: 0.12
Nodes (21): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+13 more)

### Community 48 - "repository_sft.py"
Cohesion: 0.33
Nodes (6): prepare_editing_records(), split_editing_records(), tokenize(), fit_repository_sft(), prepare_repository_training(), tokenize_repository_record()

### Community 50 - "test_dashboard_dependency_state.py"
Cohesion: 0.18
Nodes (15): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+7 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 52 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 53 - "matrix.py"
Cohesion: 0.06
Nodes (30): reset_local_benchmarks(), reset_state(), benchmark_directory_name(), download_model(), target_fingerprint(), _backfill_final_stage(), _example_sequence_lengths(), _final_stage_info() (+22 more)

### Community 54 - "test_v4_checkpoint_and_patches.py"
Cohesion: 0.08
Nodes (5): test_matrix_resume_accepts_token_identical_prompt_source_reformat(), test_patch_paths_ignores_hunk_body_dash_and_plus_prefixes(), test_publish_release_replace_current_deletes_audits(), test_v4_milestone_checkpoints_survive_pruning_and_hub_retention(), _write_ready_ckpt()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "json"
Cohesion: 0.10
Nodes (10): compare_predictions(), main(), main(), inspect(), main(), distribution(), main(), profile() (+2 more)

### Community 58 - "test_v4_architecture.py"
Cohesion: 0.09
Nodes (13): critical_attention_beta(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_decoupled_full_attn_cross_kv_fp32_master_weights_and_gradients(), test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence(), test_full_cross_cached_attention_sdpa_guard_fallback_and_non_left_padded_step_valid(), test_gate_init_tanh_resolution_dynamic_and_v4_fallback() (+5 more)

### Community 59 - "compute_step_loss"
Cohesion: 0.09
Nodes (17): backward_window(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_example_loss_weights(), compute_step_loss() (+9 more)

### Community 60 - "hashlib"
Cohesion: 0.16
Nodes (11): normalize_context_task(), curated_pool_digest(), load_curated_pool(), repository_editing_record(), remote_reset_plan(), test_remote_reset_is_scoped_and_receipt_hashes_match(), audit(), emit() (+3 more)

### Community 61 - "test_shared_schedule.py"
Cohesion: 0.28
Nodes (12): rows(), schedule(), test_all_originals_once_with_disjoint_stage_windows(), test_atomic_manifest_reuse_and_fail_closed_mismatch(), test_deterministic_model_independent_schedule_and_mapping(), test_each_semantic_mutation_changes_signature(), test_empty_required_fields_rejected(), test_foundation_context_and_target_patch_are_bound() (+4 more)

### Community 62 - "is_cross_bridge_parameter"
Cohesion: 0.17
Nodes (5): configure_probe_training(), build_bifocalgemma_optimizer(), is_cross_bridge_parameter(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration()

### Community 63 - "fit_offline_sft"
Cohesion: 0.26
Nodes (11): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), build_training_groups() (+3 more)

### Community 64 - "test_editing_manifest.py"
Cohesion: 0.29
Nodes (8): editing_manifest(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation(), test_repository_eval_is_rejected()

### Community 66 - "stable_fingerprint"
Cohesion: 0.17
Nodes (13): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), prompt_ids() (+5 more)

### Community 67 - "torch"
Cohesion: 0.08
Nodes (20): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), _kv_slice_bounds(), mask_tile(), sdpa_attention(), sdpa_cross_attention() (+12 more)

### Community 68 - "test_context_tasks.py"
Cohesion: 0.09
Nodes (9): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_prompt_ids_instance_id_isolation_and_crosscodelong_normalization(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose() (+1 more)

### Community 69 - "run_batched_context_benchmark"
Cohesion: 0.15
Nodes (8): prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), greedy_decode(), run_context_benchmark(), activity(), run_repo_niah_eval(), DurableClock

### Community 72 - "test_repository_corpus.py"
Cohesion: 0.12
Nodes (15): is_permissive_license(), repair_preimage_coverage(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), recognize_license(), resolved_license() (+7 more)

### Community 74 - "extract_mcq_letter"
Cohesion: 0.18
Nodes (7): extract_mcq_letter(), strict_letter(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_split_ambiguity_behavior(), test_adversarial_mcq_verbose_reasoning_preambles(), test_resilient_mcq_uses_terminal_final_answer_after_questioned_candidate()

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 79 - "test_training_recipes.py"
Cohesion: 0.18
Nodes (5): comparison_recipes(), limit_threads(), test_fp32_alignment_on_bf16_backbone(), test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 80 - "scorer_fingerprint"
Cohesion: 0.22
Nodes (4): _extract_match_letter(), fallback_letter_regex(), score_repo_niah_case(), scorer_fingerprint()

### Community 82 - "optimizers.py"
Cohesion: 0.24
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 83 - "benchmark.py"
Cohesion: 0.13
Nodes (9): batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), greedy_decode(), GreedyGeneration, cached_logits(), InferenceCache, _run_cached() (+1 more)

### Community 85 - "test_v4_optimizer_schedule.py"
Cohesion: 0.07
Nodes (24): is_zero_weight_decay_bridge_parameter(), default_training_profiles(), Float32Alignment, training_policy_table(), update_policy_learning_rates(), validate_training_policy(), v4_group_defaults(), v4_optimizer_defaults() (+16 more)

### Community 87 - "training/diagnostics.py"
Cohesion: 0.32
Nodes (4): encoder_diagnostics(), stratified_subset(), evaluate_sft_loss(), validation_batches()

### Community 89 - "source_fingerprints"
Cohesion: 0.29
Nodes (3): source_fingerprints(), WithoutDocstrings, test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes()

### Community 91 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 96 - "test_v4_data_disjoint.py"
Cohesion: 0.06
Nodes (28): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), generate_grounded_pool_tasks() (+20 more)

### Community 102 - "test_tier1_features.py"
Cohesion: 0.07
Nodes (15): authoritative_empirical_results(), pytest_configure(), reference_resilient_mcq_parser(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), validate_paper_manuscript() (+7 more)

### Community 107 - "eval_external_baselines.py"
Cohesion: 0.38
Nodes (4): test_eval_external_baselines_cli_dry_run(), _flatten_csv_args(), main(), parse_args()

### Community 108 - "BifocalGemmaConfig"
Cohesion: 0.09
Nodes (11): BifocalGemmaConfig, _check_transformers_version(), native_config(), test_cached_logits_decoder_only_not_polluted_by_cross_attention(), test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_from_pretrained_device_map_and_missing_bridge_validation(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_generate_multi_eos_per_row_stopping_and_cache_parity() (+3 more)

### Community 109 - "WigglyDashboard"
Cohesion: 0.29
Nodes (6): WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.22
Nodes (4): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.14
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_jsonl_dataset_query_injection_into_encoder(), test_warmup_cosine_learning_rate_schedule()

### Community 114 - "test_effective_benchmark_protocol.py"
Cohesion: 0.15
Nodes (7): test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold(), decode(), test_effective_allowance(), test_effective_spec_binds_gold_caps_and_runtime_spec(), test_invalid_allowance_fails_explicitly(), test_prepare_sampled_benchmark_collection_invalidates_stale_cache(), test_suite_caps_and_ten_model_allowances()

### Community 115 - "test_matrix_runtime_loader.py"
Cohesion: 0.22
Nodes (14): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+6 more)

### Community 116 - "make_experiment_spec"
Cohesion: 0.05
Nodes (26): make_experiment_spec(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer, _CharTokenizer, _DummyEvalModel, test_effective_spec_includes_overflow_fallback_policy() (+18 more)

### Community 120 - "wiggly_dashboard.py"
Cohesion: 0.20
Nodes (3): LastSeenWidget, phase_category(), TrainingLossLog

### Community 122 - "RepositorySnapshot"
Cohesion: 0.12
Nodes (12): EvaluationReport, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, NativeTaskResult, score_python_patch() (+4 more)

### Community 130 - "test_molab_remediation.py"
Cohesion: 0.11
Nodes (12): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), format_code(), main(), matrix_bundle(), render_notebook() (+4 more)

### Community 137 - "NativeMergedAttention"
Cohesion: 0.19
Nodes (3): NativeMergedAttention, build_joint_visibility(), TestMergedAttentionReference

### Community 142 - "JSONLRepairDataset"
Cohesion: 0.09
Nodes (7): ensure_assets_on_disk(), load_asset_json(), _resolve_eot_token_id(), collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 172 - "modeling_bifocalgemma.py"
Cohesion: 0.07
Nodes (4): BackboneProfile, BaselineScores, CachedAttention, test_notebook_stress_helpers_match_shared_library()

### Community 173 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 174 - "packed_tokens.py"
Cohesion: 0.13
Nodes (6): build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, test_packed_cache_can_be_built_without_tensor_runtime(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 177 - "TinyLossModel"
Cohesion: 0.18
Nodes (4): TestOfflineSFT, TinyLossModel, test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 180 - "FullCrossCachedAttention"
Cohesion: 0.08
Nodes (15): Bf16FullLinear, _compute_per_row_cross_query(), configure_full_rank_cross_attention(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv() (+7 more)

### Community 185 - "notebook_launcher.py"
Cohesion: 0.44
Nodes (5): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 191 - "context_tasks.py"
Cohesion: 0.10
Nodes (30): _anchor_index_for_repair(), _anti_shortcut_budget(), _apply_decoder_chat_template(), _count_context_tokens(), DisjointContextSplit, dynamic_sliding_bifocal_split(), _DynamicBlock, _DynamicSection (+22 more)

### Community 195 - "ast"
Cohesion: 0.18
Nodes (8): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), TestOfflineEditing, TestTokenRoleAndDataIdentity

### Community 197 - "test_molab_matrix_notebooks.py"
Cohesion: 0.07
Nodes (19): configure_policy_model(), unified_recipe(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_matrix_baseline_single_stage_resolution() (+11 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1061 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **91 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_verified_tokenizer()` connect `load_verified_tokenizer` to `atomic_json`, `full_pool.py`, `JSONLRepairDataset`, `schedule.py`, `gsm8k.py`, `FastTrainingPipeline`, `test_gsm8k.py`, `prepare_context_data.py`, `atomic_json`, `build_repository_corpus`, `evaluation/context.py`, `test_data_remediation.py`, `repository_sft.py`, `repository_study.py`, `matrix.py`, `json`, `hashlib`, `stable_fingerprint`, `run_batched_context_benchmark`, `test_repository_corpus.py`, `benchmark.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `stable_fingerprint()` (e.g. with `test_generated_preparation_consumes_frozen_raw_repository_pool()` and `test_legacy_paired_manifest_remains_byte_identical()`) actually correct?**
  _`stable_fingerprint()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `load_verified_tokenizer` be split into smaller, more focused modules?**
  _Cohesion score 0.11724137931034483 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `torch`, `SwiGLUDualGateBridge`, `modeling_bifocalgemma.py`, `BifocalGemmaConfig`, `JSONLRepairDataset`, `memory_probe.py`, `run_matrix_stage`, `FullCrossCachedAttention`, `bifocalgemma/__init__.py`, `test_v4_architecture.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `test_tier1_features.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11965811965811966 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `preflight.py`, `full_pool.py`, `schedule.py`, `pathlib`, `curation.py`, `json`, `prepare_context_data.py`, `atomic_json`, `build_repository_corpus`, `evaluation/context.py`, `packed_tokens.py`, `test_data_remediation.py`, `repository_sft.py`, `matrix.py`, `test_matrix_schedule_runtime.py`, `json`, `hashlib`, `test_editing_manifest.py`, `run_batched_context_benchmark`, `test_repository_corpus.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._