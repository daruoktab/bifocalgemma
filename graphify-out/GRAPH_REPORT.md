# Graph Report - bifocalgemma  (2026-11-28)

## Corpus Check
- 238 files · ~354,835 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: (none) 2, .jsonl 2, .toml 2)

## Summary
- 2589 nodes · 7187 edges · 184 communities (91 shown, 93 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 838 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e501253e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- checkpoint.py
- modeling_bifocalgemma.py
- atomic_json
- FastTrainingPipeline
- RepositorySnapshot
- atomic_json
- test_notebook_dependencies.py
- test_memory_efficient_training.py
- tokenization.py
- full_pool.py
- test_pretrained_hf_model.py
- schedule.py
- RepositorySnapshot
- run_matrix_stage
- TokenizedRepairExample
- test_v4_eval_parity.py
- eval_code.py
- BifocalGemmaForCausalLM
- test_v4_data_disjoint.py
- NativeTinyModel
- subprocess
- build_native_tiny_model
- fit_offline_sft
- gsm8k.py
- .run
- reset_hf_benchmarks.py
- test_gsm8k.py
- curation.py
- test_training_telemetry.py
- run_external_benchmark
- json
- run_batched_context_benchmark
- prepare_context_data.py
- json
- types
- telemetry.py
- .from_pretrained
- run_batched_context_benchmark
- repository_corpus.py
- evaluation/context.py
- remediation.py
- test_eval_remediation.py
- SwiGLUDualGateBridge
- test_dashboard_dependency_state.py
- repository_study.py
- test_context_benchmark_resume.py
- run_matrix_training
- test_v4_checkpoint_and_patches.py
- test_matrix_schedule_runtime.py
- load_local_native_model
- test_v4_architecture.py
- compute_step_loss
- hashlib
- RepositoryFile
- test_benchmark_prompt_budget.py
- copy
- test_benchmark_upload.py
- test_evaluation.py
- memory_efficient.py
- test_context_tasks.py
- memory_probe.py
- NativeMergedAttention
- test_construction_blocks.py
- .snapshot
- examples
- TrainingDashboard
- score_context_case
- chunked_differential_cross_entropy
- benchmark.py
- tokenize_context_task
- test_v4_optimizer_schedule.py
- training/diagnostics.py
- source_fingerprints
- bifocalgemma
- test_molab_remediation.py
- .from_pretrained_backbones
- normalize_context_task
- dependency_state
- build_repo_niah_case
- configure_probe_training
- GrokOrScaleLoRA
- Seq2SeqBatch
- resolve_optimal_eval_batch_size
- BifocalGemmaConfig
- WigglyDashboard
- _CharTokenizer
- test_fast_pipeline.py
- _extract_match_letter
- test_matrix_runtime_loader.py
- make_experiment_spec
- patch_cross_entropy
- gsm8k_prompt_ids
- matrix.py
- RepairExample
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold
- test_molab_remediation.py
- CachedAttention
- config.py
- JSONLRepairDataset
- pathlib
- test_context_benchmark_mcq_budget_floor_execution
- PackedTokenizedDataset
- TinyLossModel
- .forward
- bifocalgemma/__init__.py
- notebook_launcher.py
- context_tasks.py
- TestTokenRoleAndDataIdentity
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
- `test_number_lines_false_preserves_genuine_digit_colon_source_lines()` --calls--> `split_context_for_mode()`  [INFERRED]
  tests/test_eval_remediation.py → src/bifocalgemma/evaluation/context.py
- `test_repair_record_type_has_same_prefix_as_explicit_repair_task()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `generate()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing()` --calls--> `score_context_case()`  [INFERRED]
  tests/test_v4_eval_parity.py → src/bifocalgemma/evaluation/context.py
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (184 total, 93 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.19
Nodes (10): build_policy_optimizer(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution() (+2 more)

### Community 1 - "checkpoint.py"
Cohesion: 0.08
Nodes (20): load_asset_json(), is_transient_lfs_pointer_error(), load_matrix_adapters_for_inference(), _milestone_checkpoint_names(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint() (+12 more)

### Community 3 - "atomic_json"
Cohesion: 0.17
Nodes (9): BenchmarkUploadQueue, retry_delay(), atomic_json(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed() (+1 more)

### Community 5 - "FastTrainingPipeline"
Cohesion: 0.13
Nodes (10): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_training_pipeline_baseline_2b(), test_fast_training_pipeline_with_jsonl_data_dir() (+2 more)

### Community 6 - "RepositorySnapshot"
Cohesion: 0.13
Nodes (17): _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines() (+9 more)

### Community 7 - "atomic_json"
Cohesion: 0.14
Nodes (17): is_permissive_license(), build_and_verify_mixed_pool(), build_and_verify_release(), _build_v4_notices(), _extract_python_fim_from_code(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers() (+9 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.11
Nodes (17): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+9 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.22
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 11 - "tokenization.py"
Cohesion: 0.09
Nodes (12): DeveloperAgent, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter, extract_token_ast_features() (+4 more)

### Community 13 - "full_pool.py"
Cohesion: 0.18
Nodes (9): read_arrow_rows(), token_profile(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main(), progress() (+1 more)

### Community 14 - "test_pretrained_hf_model.py"
Cohesion: 0.17
Nodes (6): ProbeLoRALinear, test_model_to_bfloat16_preserves_fp32_master_weights(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_save_and_from_pretrained_roundtrip(), test_fast_training_config_step_reconciliation()

### Community 15 - "schedule.py"
Cohesion: 0.11
Nodes (24): _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule() (+16 more)

### Community 16 - "RepositorySnapshot"
Cohesion: 0.17
Nodes (16): RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), example_from_record(), NativeTaskResult (+8 more)

### Community 17 - "run_matrix_stage"
Cohesion: 0.15
Nodes (13): _resolve_example_diff_flag(), run_matrix_stage(), backward_window(), _split_microbatch_groups(), compute_example_loss_weights(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows() (+5 more)

### Community 18 - "TokenizedRepairExample"
Cohesion: 0.13
Nodes (9): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), synthetic_stress_target(), TestDataAndLoss, test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix() (+1 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.08
Nodes (9): MockCausalLM, MockQwenTokenizer, _ProportionalTokenizer, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(), test_reponiah_32k_zero_overflow_for_bifocalgemma_and_qwen25(), test_windowed_file_section_disjoint_split_and_train_eval_parity() (+1 more)

### Community 20 - "eval_code.py"
Cohesion: 0.15
Nodes (6): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "BifocalGemmaForCausalLM"
Cohesion: 0.17
Nodes (3): BifocalGemmaForCausalLM, test_bifocalgemma_merge_and_unload(), test_enable_memory_efficient_training_and_softcapping()

### Community 23 - "test_v4_data_disjoint.py"
Cohesion: 0.31
Nodes (6): _load_prepare_module(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_multifunction_completion_extraction_removes_single_return_cap(), test_v4_quarantine_licenses_dedup_and_nextcoder_rejection(), _write_mock_curated_pool()

### Community 24 - "NativeTinyModel"
Cohesion: 0.12
Nodes (16): NativeTinyModel, configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_run_matrix_stage_execution_batch2_and_frozen_decoder_invariance() (+8 more)

### Community 25 - "subprocess"
Cohesion: 0.18
Nodes (3): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), test_repository_notebook_bundle_is_synchronized(), format_code()

### Community 27 - "fit_offline_sft"
Cohesion: 0.27
Nodes (11): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest() (+3 more)

### Community 28 - "gsm8k.py"
Cohesion: 0.18
Nodes (14): _clean_extracted_number(), digest(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer(), fetch_gsm8k_sources(), fingerprint(), generate_run() (+6 more)

### Community 29 - ".run"
Cohesion: 0.12
Nodes (3): test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule()

### Community 31 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 32 - "curation.py"
Cohesion: 0.07
Nodes (37): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+29 more)

### Community 33 - "test_training_telemetry.py"
Cohesion: 0.13
Nodes (5): _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime(), RecordingStream

### Community 34 - "run_external_benchmark"
Cohesion: 0.10
Nodes (18): prepare_sampled_benchmark_collection(), format_crosscodelong_summary(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort() (+10 more)

### Community 35 - "json"
Cohesion: 0.08
Nodes (28): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+20 more)

### Community 36 - "run_batched_context_benchmark"
Cohesion: 0.24
Nodes (6): run_batched_context_benchmark(), activity(), _paired_budget_for(), run_context_benchmark(), activity(), DurableClock

### Community 37 - "prepare_context_data.py"
Cohesion: 0.11
Nodes (21): adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), completion_from_snapshot(), _conversation_to_instruction_context_target(), _estimate_tokens() (+13 more)

### Community 38 - "json"
Cohesion: 0.07
Nodes (31): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), file_digest(), audit_repository_sources() (+23 more)

### Community 39 - "types"
Cohesion: 0.08
Nodes (5): test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), download(), test_recover_remote_run_ignores_dot_temp_and_heartbeat_files(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only(), test_standalone_export_requires_matching_remote_contents()

### Community 40 - "telemetry.py"
Cohesion: 0.13
Nodes (7): format_duration(), format_stage_display(), _gpu_utilization_snapshot(), loss_chart(), render_dashboard(), TrainingLossLog, test_dashboard_stage_display_formatting()

### Community 41 - ".from_pretrained"
Cohesion: 0.18
Nodes (7): export_matrix_pretrained_model(), test_from_pretrained_device_map_and_missing_bridge_validation(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_export_matrix_pretrained_model(), test_export_matrix_pretrained_model_baseline()

### Community 42 - "run_batched_context_benchmark"
Cohesion: 0.11
Nodes (17): build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer(), repair_report_metrics(), _runtime_model_identity() (+9 more)

### Community 43 - "repository_corpus.py"
Cohesion: 0.06
Nodes (33): repair_preimage_coverage(), patch_paths(), ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus() (+25 more)

### Community 44 - "evaluation/context.py"
Cohesion: 0.08
Nodes (29): _build_paired_eval_header_and_row(), _context_already_includes_instruction(), decoder_prompt(), _deduplicate_longcodeqa_headers(), encoder_context(), _extract_full_python_code(), greedy_decode(), _has_disjoint_structured_context() (+21 more)

### Community 47 - "remediation.py"
Cohesion: 0.12
Nodes (22): split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+14 more)

### Community 48 - "test_eval_remediation.py"
Cohesion: 0.15
Nodes (8): _CharTokenizer, _DummyEvalModel, test_effective_spec_includes_overflow_fallback_policy(), test_external_runner_resume_and_signature_guard(), test_number_lines_false_preserves_genuine_digit_colon_source_lines(), test_overflow_fallback_recomputes_budget_and_records_metadata(), test_recount_functional_pass_rate_returns_none_on_partial_execution(), _write_mcq_benchmark()

### Community 49 - "SwiGLUDualGateBridge"
Cohesion: 0.06
Nodes (5): NormalizedAlignment, SwiGLUDualGateBridge, TestBifocalGemmaArchitecture, TestGemma4EdgeTopology, test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight()

### Community 50 - "test_dashboard_dependency_state.py"
Cohesion: 0.14
Nodes (15): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+7 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+10 more)

### Community 52 - "test_context_benchmark_resume.py"
Cohesion: 0.14
Nodes (10): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization() (+2 more)

### Community 53 - "run_matrix_training"
Cohesion: 0.08
Nodes (20): _backfill_final_stage(), _example_sequence_lengths(), _final_stage_info(), _latest_checkpoint_matches(), _local_standalone_export_valid(), pack_window_microbatches(), _reusable_training_report(), run_matrix_benchmark() (+12 more)

### Community 54 - "test_v4_checkpoint_and_patches.py"
Cohesion: 0.08
Nodes (5): test_matrix_resume_accepts_token_identical_prompt_source_reformat(), test_patch_paths_ignores_hunk_body_dash_and_plus_prefixes(), test_publish_release_replace_current_deletes_audits(), test_v4_milestone_checkpoints_survive_pruning_and_hub_retention(), _write_ready_ckpt()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "load_local_native_model"
Cohesion: 0.21
Nodes (5): load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), TestCheckpointTransfer, test_local_loader_preflights_and_preserves_shared_words()

### Community 58 - "test_v4_architecture.py"
Cohesion: 0.10
Nodes (13): critical_attention_beta(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence(), test_decoupled_upper_full_attention_cross_kv_topology(), test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), spy_sdpa() (+5 more)

### Community 59 - "compute_step_loss"
Cohesion: 0.10
Nodes (14): compute_diff_token_weights(), _compute_row_weights(), compute_step_loss(), is_unified_diff_target_tokens(), _pointer_chunk_forward(), _streaming_lse_pointer_nll(), streaming_pointer_auxiliary_loss(), StreamingPointerLoss (+6 more)

### Community 60 - "hashlib"
Cohesion: 0.10
Nodes (21): normalize_context_task(), curated_pool_digest(), editing_record(), git_text_lines(), load_curated_pool(), prepare_editing_records(), repository_editing_record(), split_editing_records() (+13 more)

### Community 61 - "RepositoryFile"
Cohesion: 0.18
Nodes (5): RepositoryFile, tokenize_repair_example(), test_dependency_completion_removes_target_and_retains_real_helper(), TestRealOfflineTokenizers, test_applied_python_patch_reports_syntax_independently()

### Community 62 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 63 - "copy"
Cohesion: 0.08
Nodes (18): remote_reset_plan(), reset_local_benchmarks(), reset_state(), is_zero_weight_decay_bridge_parameter(), validate_optimizer_config(), _build_group_schedule(), build_training_groups(), default_training_profiles() (+10 more)

### Community 64 - "test_benchmark_upload.py"
Cohesion: 0.19
Nodes (3): test_external_activity_reader_does_not_need_running_kernel(), inspect(), main()

### Community 67 - "memory_efficient.py"
Cohesion: 0.07
Nodes (21): flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), _kv_slice_bounds(), mask_tile(), sdpa_attention() (+13 more)

### Community 68 - "test_context_tasks.py"
Cohesion: 0.10
Nodes (7): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_tokenize_context_task_preserves_target_indentation_and_newlines()

### Community 69 - "memory_probe.py"
Cohesion: 0.22
Nodes (8): gpu_memory(), optimizer_bytes(), profile_training_group(), run_length_grid(), tensor_bytes(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 78 - "examples"
Cohesion: 0.19
Nodes (5): make_identity(), examples(), TestTrainingPolicy, test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 79 - "TrainingDashboard"
Cohesion: 0.29
Nodes (3): TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 80 - "score_context_case"
Cohesion: 0.16
Nodes (12): completion_metrics(), extract_mcq_letter(), normalize_completion_prediction(), score_context_case(), _sequence_and_identifier_metrics(), strict_letter(), _strip_echoed_suffix(), _strip_fim_markers() (+4 more)

### Community 82 - "chunked_differential_cross_entropy"
Cohesion: 0.20
Nodes (7): chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), differential_diff_cross_entropy(), _normalized_token_coefficients(), checkpoint(), test_per_example_mean_and_length_capped_eliminate_short_target_starvation()

### Community 83 - "benchmark.py"
Cohesion: 0.11
Nodes (15): RepairExample, decode_text(), format_encoder_context(), batched_greedy_decode(), generate_gsm8k(), generate_diagnostic_text(), generate_patch(), greedy_decode() (+7 more)

### Community 84 - "tokenize_context_task"
Cohesion: 0.22
Nodes (9): _apply_decoder_chat_template(), reserved_output_token_budget(), _resolve_eot_token_id(), task_prompt(), tokenize_context_task(), _use_disjoint_mode(), test_paired_false_baseline_and_v4_namespace_invariance(), _numbered_line_numbers() (+1 more)

### Community 85 - "test_v4_optimizer_schedule.py"
Cohesion: 0.09
Nodes (19): build_bifocalgemma_optimizer(), update_policy_learning_rates(), v4_group_defaults(), v4_optimizer_defaults(), build_bifocalgemma_adamw(), build_partitioned_param_groups(), configure_matrix_stage(), _resolve_group_settings() (+11 more)

### Community 87 - "training/diagnostics.py"
Cohesion: 0.32
Nodes (4): encoder_diagnostics(), stratified_subset(), evaluate_sft_loss(), validation_batches()

### Community 89 - "source_fingerprints"
Cohesion: 0.29
Nodes (4): source_fingerprints(), WithoutDocstrings, test_reusable_training_report_backfills_missing_final_stage_when_verified(), _build_run()

### Community 91 - "test_molab_remediation.py"
Cohesion: 0.10
Nodes (7): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues(), test_matrix_quarantine_legacy_checkpoint_vs_clean_resume(), test_mcq_budget_floor_in_v3_benchmark_logic()

### Community 92 - ".from_pretrained_backbones"
Cohesion: 0.24
Nodes (4): build_native_model(), _resolve_backbone_weights(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 96 - "normalize_context_task"
Cohesion: 0.07
Nodes (25): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), generate_grounded_pool_tasks() (+17 more)

### Community 102 - "build_repo_niah_case"
Cohesion: 0.25
Nodes (6): build_repo_niah_case(), _count_niah_tokens(), _encoded_context_tokens(), generate_repo_niah_haystack(), make_distractor_python_chunk(), _render_niah_prompt_tokens()

### Community 105 - "Seq2SeqBatch"
Cohesion: 0.25
Nodes (3): ModelOutput, Seq2SeqBatch, test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss()

### Community 107 - "resolve_optimal_eval_batch_size"
Cohesion: 0.40
Nodes (3): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_v4_config_constants_and_external_registry()

### Community 108 - "BifocalGemmaConfig"
Cohesion: 0.13
Nodes (7): BifocalGemmaConfig, _check_transformers_version(), native_config(), test_bridge_init_preserved_across_post_init(), test_generate_multi_eos_per_row_stopping_and_cache_parity(), test_generate_right_padded_matches_left_padded(), test_decoupled_full_attn_cross_kv_fp32_master_weights_and_gradients()

### Community 109 - "WigglyDashboard"
Cohesion: 0.22
Nodes (8): LastSeenWidget, phase_category(), WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.15
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 115 - "test_matrix_runtime_loader.py"
Cohesion: 0.22
Nodes (16): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+8 more)

### Community 116 - "make_experiment_spec"
Cohesion: 0.14
Nodes (16): make_experiment_spec(), _prompt_ids_with_metadata(), _line_numbers(), _RecordingTokenizer, _SimpleTokenizer, test_eval_decoder_overflow_backoff_keeps_prefix_within_tight_budget(), test_grounded_line_lookup_stays_in_encoder_and_reconstruction_targets_exact_line(), test_gsm8k_paired_eval_matches_clean_prose_without_line_numbers_or_duplication() (+8 more)

### Community 121 - "matrix.py"
Cohesion: 0.06
Nodes (18): configure_full_rank_cross_attention(), fit_curated_sft(), benchmark_directory_name(), target_fingerprint(), _build_schedule_for_epochs(), _resolve_profile_settings(), resolve_training_contract(), _validate_num_epochs() (+10 more)

### Community 122 - "RepairExample"
Cohesion: 0.23
Nodes (7): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 130 - "test_molab_remediation.py"
Cohesion: 0.24
Nodes (5): matrix_bundle(), main(), render(), source_bundle(), source_bundle_bytes()

### Community 137 - "config.py"
Cohesion: 0.12
Nodes (12): BackboneProfile, CheckpointRef, ExperimentSpec, get_v2_experiment(), _pair(), _ref(), validate_experiment_matrix(), BaselineScores (+4 more)

### Community 142 - "JSONLRepairDataset"
Cohesion: 0.15
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 172 - "pathlib"
Cohesion: 0.06
Nodes (11): EvaluationReport, TestTokenizerVerification, test_notebook_stress_helpers_match_shared_library(), ExecutingScoringOptions, ScoringOptions, ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards(), test_last_token_logits_match_full_head() (+3 more)

### Community 173 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 177 - "TinyLossModel"
Cohesion: 0.18
Nodes (4): TestOfflineSFT, TinyLossModel, test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 180 - ".forward"
Cohesion: 0.12
Nodes (8): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossTiledAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), _resolve_gate_init_tanh(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step()

### Community 185 - "notebook_launcher.py"
Cohesion: 0.19
Nodes (6): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 191 - "context_tasks.py"
Cohesion: 0.11
Nodes (24): _anchor_index_for_repair(), _anti_shortcut_budget(), _count_context_tokens(), DisjointContextSplit, dynamic_sliding_bifocal_split(), _DynamicBlock, _DynamicSection, _extract_dynamic_sections() (+16 more)

### Community 197 - "test_molab_matrix_notebooks.py"
Cohesion: 0.09
Nodes (5): test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_pointer_loss_gradient_checkpointing_and_trivial_filtering(), test_v3_critical_attention_beta_and_triton_tile_clamping()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 986 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **93 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `preflight.py`, `modeling_bifocalgemma.py`, `config.py`, `test_memory_efficient_training.py`, `run_matrix_stage`, `NativeTinyModel`, `curation.py`, `json`, `pathlib`, `PackedTokenizedDataset`, `RepositoryFile`, `context_tasks.py`, `memory_efficient.py`, `.run_benchmark_suite`, `examples`, `benchmark.py`, `tokenize_context_task`, `test_v4_optimizer_schedule.py`, `normalize_context_task`, `configure_probe_training`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `stable_fingerprint()` (e.g. with `test_generated_preparation_consumes_frozen_raw_repository_pool()` and `test_legacy_paired_manifest_remains_byte_identical()`) actually correct?**
  _`stable_fingerprint()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `checkpoint.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07822410147991543 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaConfig` connect `BifocalGemmaConfig` to `modeling_bifocalgemma.py`, `FastTrainingPipeline`, `.from_pretrained`, `Seq2SeqBatch`, `test_pretrained_hf_model.py`, `JSONLRepairDataset`, `benchmark.py`, `.forward`, `BifocalGemmaForCausalLM`, `bifocalgemma/__init__.py`, `test_v4_architecture.py`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `.from_pretrained_backbones`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `modeling_bifocalgemma.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09191583610188261 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `modeling_bifocalgemma.py`, `memory_efficient.py`, `FastTrainingPipeline`, `.from_pretrained`, `Seq2SeqBatch`, `BifocalGemmaConfig`, `.generate`, `test_pretrained_hf_model.py`, `JSONLRepairDataset`, `run_matrix_stage`, `benchmark.py`, `.forward`, `bifocalgemma/__init__.py`, `test_v4_architecture.py`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `.from_pretrained_backbones`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._