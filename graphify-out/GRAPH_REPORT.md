# Graph Report - bifocalgemma  (2026-11-20)

## Corpus Check
- 255 files · ~353,430 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2526 nodes · 6950 edges · 185 communities (86 shown, 99 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 781 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `846b23c6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- TrainingDashboard
- RepositorySnapshot
- benchmark.py
- build_native_tiny_model
- re
- pathlib
- test_dashboard_dependency_state.py
- test_memory_efficient_training.py
- FastTrainingConfig
- test_matrix_runtime_loader.py
- curation.py
- schedule.py
- benchmark_upload.py
- prepare_context_data.py
- json
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- hashlib
- pytest
- context_tasks.py
- patch_cross_entropy
- adversarial_r3_probe.py
- torch
- FastTrainingPipeline
- test_v4_data_disjoint.py
- curriculum.py
- stable_fingerprint
- external_runner.py
- examples
- FeatureNotReadyError
- prepare_context_data.py
- synchronize_matrix
- generation.py
- test_molab_matrix_notebooks.py
- is_cross_bridge_parameter
- protocol.py
- load_local_native_model
- evaluation/context.py
- test_molab_remediation.py
- test_pretrained_hf_model.py
- sys
- bifocalgemma/__init__.py
- test_tier4_scenarios.py
- repository_study.py
- test_repository_study.py
- run_matrix_stage
- adapters.py
- test_matrix_schedule_runtime.py
- fit_offline_sft
- .config
- objectives.py
- packed_tokens.py
- JSONLRepairDataset
- remote_reset_plan
- memory_probe.py
- matrix.py
- test_training_telemetry.py
- TestBifocalGemmaArchitecture
- test_context_tasks.py
- test_construction_blocks.py
- repository_corpus.py
- tempfile
- test_construction_blocks.py
- Tensor
- NativeTinyModel
- WigglyDashboard
- telemetry.py
- json
- modeling_bifocalgemma.py
- test_upload_recovery_receipts.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_benchmark_prompt_budget.py
- run_matrix_training
- RepairCollator
- remediation.py
- normalize_context_task
- eval_external_baselines.py
- tests_e2e/conftest.py
- test_v4_architecture.py
- .from_pretrained_backbones
- BifocalGemmaConfig
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- JSONLRepairDataset
- .snapshot
- patch_matrix_tokenization
- DummyTrackingTokenizer
- __main__.py
- PackedTokenizedDataset
- test_optimized_native_resume_matches_full_and_rejects_different_recipe
- RepairPrediction
- test_applied_python_patch_reports_syntax_independently
- generate_repo_niah_haystack
- repository_sft.py
- fit_curated_sft
- load_verified_tokenizer
- test_tier1_features.py
- tests_e2e/conftest.py
- build_repository_corpus
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- test_evaluation.py
- RecordingStream
- test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold
- TestCheckpointTransfer
- test_generate_multi_eos_per_row_stopping_and_cache_parity
- NativeMergedAttention
- source_fingerprints
- FullCrossCachedAttention
- TrainingLossLog

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 67 edges
2. `atomic_json()` - 56 edges
3. `run_matrix_training()` - 50 edges
4. `load_verified_tokenizer()` - 50 edges
5. `BifocalGemmaForCausalLM` - 49 edges
6. `RepositorySnapshot` - 45 edges
7. `run_matrix_benchmark()` - 40 edges
8. `TokenizedRepairExample` - 40 edges
9. `BifocalGemmaConfig` - 37 edges
10. `FastTrainingPipeline` - 37 edges

## Surprising Connections (you probably didn't know these)
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_repair_record_type_has_same_prefix_as_explicit_repair_task()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `generate()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/models/bridge.py
- `test_swiglu_bridge_non_redundant_single_post_gain()` --uses--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_v4_architecture.py → src/bifocalgemma/models/bridge.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (185 total, 99 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.11
Nodes (14): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe() (+6 more)

### Community 1 - "TrainingDashboard"
Cohesion: 0.18
Nodes (5): loss_chart(), render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 2 - "RepositorySnapshot"
Cohesion: 0.13
Nodes (16): RepairExample, RepositoryFile, RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms() (+8 more)

### Community 3 - "benchmark.py"
Cohesion: 0.16
Nodes (8): batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), greedy_decode(), GreedyGeneration, InferenceCache, test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight(), test_v4_config_constants_and_external_registry()

### Community 5 - "build_native_tiny_model"
Cohesion: 0.24
Nodes (3): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head()

### Community 6 - "re"
Cohesion: 0.17
Nodes (17): _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines() (+9 more)

### Community 7 - "pathlib"
Cohesion: 0.10
Nodes (13): file_digest(), recover_remote_run(), _copy_atomic(), retry_delay(), _synchronize_once(), synchronize_run(), atomic_json(), retry_windows_file_operation() (+5 more)

### Community 8 - "test_dashboard_dependency_state.py"
Cohesion: 0.06
Nodes (36): dependency_state(), test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready() (+28 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.13
Nodes (15): run_training_memory_preflight(), __getattr__(), patch_cross_entropy(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients() (+7 more)

### Community 11 - "FastTrainingConfig"
Cohesion: 0.12
Nodes (10): FastTrainingConfig, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule(), test_fast_training_config_and_pipeline_baseline_single_stage() (+2 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.23
Nodes (16): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+8 more)

### Community 14 - "curation.py"
Cohesion: 0.08
Nodes (31): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits() (+23 more)

### Community 15 - "schedule.py"
Cohesion: 0.08
Nodes (31): _build_schedule_for_epochs(), _schedule_signature_matches(), _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json() (+23 more)

### Community 16 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (17): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed() (+9 more)

### Community 17 - "prepare_context_data.py"
Cohesion: 0.29
Nodes (7): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), code_fragment_signatures()

### Community 18 - "json"
Cohesion: 0.08
Nodes (18): curated_pool_digest(), write_view(), compare_predictions(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), test_view_record_json_is_lossless_and_gsm_shots_are_explicit(), audit(), emit() (+10 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.06
Nodes (14): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+6 more)

### Community 20 - "eval_code.py"
Cohesion: 0.14
Nodes (7): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): extract_mcq_letter(), strict_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization() (+15 more)

### Community 23 - "hashlib"
Cohesion: 0.16
Nodes (16): _clean_extracted_number(), digest(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k() (+8 more)

### Community 24 - "pytest"
Cohesion: 0.17
Nodes (11): build_bifocalgemma_optimizer(), build_bifocalgemma_adamw(), configure_matrix_stage(), test_consolidated_optimizer_builder_parity_and_decoder_only_apples_to_apples(), test_diff_marker_v1_strictly_gated_to_repair_unified_diffs_only(), test_matrix_stage_microbatch_invariance_under_heterogeneous_lengths(), test_per_group_grad_clipping_isolates_gate_scalars_and_gate_proj_from_bridge(), test_six_group_partitioning_and_argon_116_invariants() (+3 more)

### Community 25 - "context_tasks.py"
Cohesion: 0.18
Nodes (16): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), _parse_file_sections(), _render_entries(), _resolve_eot_token_id() (+8 more)

### Community 26 - "patch_cross_entropy"
Cohesion: 0.20
Nodes (4): examples(), TestOfflineSFT, TinyLossModel, test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "torch"
Cohesion: 0.12
Nodes (17): flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), mask_tile(), sdpa_attention(), sdpa_cross_attention() (+9 more)

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.19
Nodes (7): _CharTokenizer, _load_prepare_module(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_format_numbered_code_and_exports(), test_mixed_pool_v4_curriculum_grounded_and_curated_sft(), test_multifunction_completion_extraction_removes_single_return_cap(), test_v4_quarantine_licenses_dedup_and_nextcoder_rejection()

### Community 31 - "curriculum.py"
Cohesion: 0.21
Nodes (8): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 32 - "stable_fingerprint"
Cohesion: 0.12
Nodes (22): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), read_arrow_rows(), audit_repository_sources() (+14 more)

### Community 34 - "external_runner.py"
Cohesion: 0.12
Nodes (15): prepare_sampled_benchmark_collection(), build_repo_niah_case(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort() (+7 more)

### Community 36 - "FeatureNotReadyError"
Cohesion: 0.16
Nodes (5): DeveloperAgent, extract_token_ast_features(), TokenASTFeatures, BuildBlock, FeatureNotReadyError

### Community 37 - "prepare_context_data.py"
Cohesion: 0.11
Nodes (19): is_permissive_license(), split_connected_records(), adapt_v4_row(), build_gsm8k_fingerprints(), completion_from_snapshot(), completions_from_snapshot(), context_pool(), _conversation_to_instruction_context_target() (+11 more)

### Community 38 - "synchronize_matrix"
Cohesion: 0.17
Nodes (11): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix(), snapshot(), snapshot() (+3 more)

### Community 39 - "generation.py"
Cohesion: 0.23
Nodes (7): decode_text(), format_encoder_context(), generate_diagnostic_text(), generate_patch(), repair_prompt_ids(), _safe_apply_chat_template(), cached_logits()

### Community 40 - "test_molab_matrix_notebooks.py"
Cohesion: 0.08
Nodes (22): make_experiment_spec(), configure_full_rank_cross_attention(), configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), clamped_triton_kernel_options(), StageLedger, test_cached_uncached_and_batched_greedy_decode_equivalence() (+14 more)

### Community 41 - "is_cross_bridge_parameter"
Cohesion: 0.20
Nodes (5): is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), build_partitioned_param_groups(), classify_bifocalgemma_parameter(), _resolve_group_settings()

### Community 42 - "protocol.py"
Cohesion: 0.09
Nodes (18): _extract_match_letter(), fallback_letter_regex(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer() (+10 more)

### Community 43 - "load_local_native_model"
Cohesion: 0.16
Nodes (5): build_native_model(), load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), test_local_loader_preflights_and_preserves_shared_words()

### Community 44 - "evaluation/context.py"
Cohesion: 0.07
Nodes (29): completion_metrics(), _context_already_includes_instruction(), decoder_prompt(), _deduplicate_longcodeqa_headers(), encoder_context(), _extract_full_python_code(), format_crosscodelong_summary(), greedy_decode() (+21 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 47 - "test_pretrained_hf_model.py"
Cohesion: 0.15
Nodes (8): test_from_pretrained_raises_file_not_found_when_weights_missing(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip(), test_enable_memory_efficient_training_and_softcapping(), test_fast_training_pipeline_baseline_2b()

### Community 48 - "sys"
Cohesion: 0.18
Nodes (7): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), format_code(), matrix_bundle(), main(), render(), source_bundle(), source_bundle_bytes()

### Community 50 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 51 - "repository_study.py"
Cohesion: 0.25
Nodes (17): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+9 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.11
Nodes (17): v4_repository_task_from_row(), repository_task_from_row(), RepositoryTask, project_root(), CharacterTokenizer, diff(), ExecutingScoringOptions, fixture() (+9 more)

### Community 53 - "run_matrix_stage"
Cohesion: 0.25
Nodes (8): run_matrix_stage(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_partial_backward_oom_replays_entire_window_without_double_gradient(), test_stage1_to_stage2_continuous_warmup_offset_and_per_group_grad_norms(), test_trained_adapters_export_reload_logit_parity()

### Community 54 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "fit_offline_sft"
Cohesion: 0.11
Nodes (23): run_batched_context_benchmark(), activity(), prompt_ids(), _prompt_ids_impl(), run_context_benchmark(), activity(), repair_report_metrics(), encoder_diagnostics() (+15 more)

### Community 59 - "objectives.py"
Cohesion: 0.07
Nodes (22): _resolve_example_diff_flag(), backward_window(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_example_loss_weights() (+14 more)

### Community 60 - "packed_tokens.py"
Cohesion: 0.36
Nodes (4): build_packed_cache(), open_packed_cache(), test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 62 - "remote_reset_plan"
Cohesion: 0.14
Nodes (9): remote_reset_plan(), reset_local_benchmarks(), reset_state(), checked_prefix(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models() (+1 more)

### Community 63 - "memory_probe.py"
Cohesion: 0.18
Nodes (8): configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), synthetic_stress_target(), tensor_bytes(), TestMemoryProbe

### Community 64 - "matrix.py"
Cohesion: 0.07
Nodes (19): ensure_assets_on_disk(), load_asset_json(), BackboneProfile, download_model(), resolve_training_contract(), _validate_num_epochs(), build_training_groups(), default_training_profiles() (+11 more)

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.17
Nodes (3): test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime()

### Community 68 - "test_context_tasks.py"
Cohesion: 0.10
Nodes (7): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_short_answer_scoring_rejects_unrequested_prose(), test_tokenize_context_task_preserves_target_indentation_and_newlines()

### Community 69 - "test_construction_blocks.py"
Cohesion: 0.06
Nodes (8): run_length_grid(), TestTokenRoleAndDataIdentity, test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library(), test_notebook_sweep_persists_results_per_case(), limit_threads(), test_fp32_alignment_on_bf16_backbone()

### Community 72 - "repository_corpus.py"
Cohesion: 0.12
Nodes (16): repair_preimage_coverage(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license(), resolved_license() (+8 more)

### Community 74 - "tempfile"
Cohesion: 0.11
Nodes (12): EvaluationReport, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, NativeTaskResult, score_python_patch() (+4 more)

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 80 - "WigglyDashboard"
Cohesion: 0.35
Nodes (7): LastSeenWidget, WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 82 - "telemetry.py"
Cohesion: 0.17
Nodes (4): format_duration(), format_stage_display(), phase_category(), test_dashboard_stage_display_formatting()

### Community 83 - "json"
Cohesion: 0.09
Nodes (25): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+17 more)

### Community 85 - "modeling_bifocalgemma.py"
Cohesion: 0.07
Nodes (9): NormalizedAlignment, SwiGLUDualGateBridge, FullCrossTiledAttention, CachedAttention, TiledEncoderAttention, TiledMergedAttention, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init() (+1 more)

### Community 87 - "test_upload_recovery_receipts.py"
Cohesion: 0.07
Nodes (13): editing_manifest(), repository_editing_record(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation() (+5 more)

### Community 91 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 92 - "run_matrix_training"
Cohesion: 0.09
Nodes (16): RemoteHeartbeat, target_fingerprint(), _backfill_final_stage(), _final_stage_info(), _latest_checkpoint_matches(), _local_standalone_export_valid(), _reusable_training_report(), run_matrix_benchmark() (+8 more)

### Community 93 - "RepairCollator"
Cohesion: 0.12
Nodes (8): Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), TestDataAndLoss, test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_fast_pipeline_batch_edge_cases(), test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 95 - "remediation.py"
Cohesion: 0.10
Nodes (24): token_profile(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+16 more)

### Community 96 - "normalize_context_task"
Cohesion: 0.18
Nodes (12): normalize_context_task(), generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record(), parent() (+4 more)

### Community 101 - "eval_external_baselines.py"
Cohesion: 0.32
Nodes (4): test_eval_external_baselines_cli_dry_run(), _flatten_csv_args(), main(), parse_args()

### Community 102 - "tests_e2e/conftest.py"
Cohesion: 0.10
Nodes (9): authoritative_empirical_results(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization() (+1 more)

### Community 103 - "test_v4_architecture.py"
Cohesion: 0.13
Nodes (10): critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_decoupled_upper_full_attention_cross_kv_topology(), test_sdpa_attention_structured_masks_match_tiled_eager(), test_sdpa_cross_attention_matches_eager_and_handles_all_masked_rows() (+2 more)

### Community 104 - ".from_pretrained_backbones"
Cohesion: 0.29
Nodes (3): _resolve_backbone_weights(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 105 - "BifocalGemmaConfig"
Cohesion: 0.17
Nodes (5): BifocalGemmaConfig, _check_transformers_version(), native_config(), test_decoupled_full_attn_cross_kv_fp32_master_weights_and_gradients(), test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence()

### Community 106 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 108 - "JSONLRepairDataset"
Cohesion: 0.29
Nodes (3): JSONLRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 114 - "__main__.py"
Cohesion: 0.19
Nodes (5): validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 122 - "repository_sft.py"
Cohesion: 0.20
Nodes (8): prepare_editing_records(), split_editing_records(), prepare_curated_training(), tokenize(), fit_repository_sft(), prepare_repository_training(), tokenize_repository_record(), TestOfflineEditing

### Community 124 - "load_verified_tokenizer"
Cohesion: 0.13
Nodes (8): CheckpointRef, ExperimentSpec, get_v2_experiment(), _pair(), _ref(), load_verified_tokenizer(), tokenize_repair_example(), TestTokenizerVerification

### Community 130 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 134 - "tests_e2e/conftest.py"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 137 - "build_repository_corpus"
Cohesion: 0.13
Nodes (11): export_pool(), load_curated_pool(), ArchiveSnapshotStore, build_repository_corpus(), RepositoryPreparationLedger, repo_identity(), test_frozen_pool_verifies_checksums_before_use(), test_repository_pool_requires_context_license_policy_and_coverage() (+3 more)

### Community 142 - "test_export_matrix_pretrained_model_with_subfolder_and_tokenizer"
Cohesion: 0.25
Nodes (3): export_matrix_pretrained_model(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_export_matrix_pretrained_model_baseline()

### Community 177 - "source_fingerprints"
Cohesion: 0.33
Nodes (3): source_fingerprints(), WithoutDocstrings, test_f5_training_semantics_change_rejects_legacy_approval()

### Community 180 - "FullCrossCachedAttention"
Cohesion: 0.13
Nodes (7): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), _resolve_gate_init_tanh()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 978 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **99 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_verified_tokenizer()` connect `load_verified_tokenizer` to `benchmark.py`, `pathlib`, `build_repository_corpus`, `json`, `test_v4_eval_parity.py`, `hashlib`, `stable_fingerprint`, `prepare_context_data.py`, `generation.py`, `evaluation/context.py`, `repository_study.py`, `fit_offline_sft`, `matrix.py`, `repository_corpus.py`, `JSONLRepairDataset`, `run_matrix_training`, `RepairCollator`, `remediation.py`, `repository_sft.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `preflight.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11330049261083744 - nodes in this community are weakly interconnected._
- **Why does `run_matrix_training()` connect `run_matrix_training` to `TrainingDashboard`, `pathlib`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `schedule.py`, `benchmark_upload.py`, `json`, `stable_fingerprint`, `synchronize_matrix`, `test_molab_matrix_notebooks.py`, `test_molab_remediation.py`, `source_fingerprints`, `run_matrix_stage`, `test_matrix_schedule_runtime.py`, `objectives.py`, `matrix.py`, `json`, `patch_matrix_tokenization`, `repository_sft.py`, `load_verified_tokenizer`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `RepositorySnapshot` be split into smaller, more focused modules?**
  _Cohesion score 0.12903225806451613 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `preflight.py`, `pathlib`, `build_repository_corpus`, `curation.py`, `json`, `hashlib`, `external_runner.py`, `prepare_context_data.py`, `test_molab_matrix_notebooks.py`, `evaluation/context.py`, `test_matrix_schedule_runtime.py`, `fit_offline_sft`, `packed_tokens.py`, `matrix.py`, `repository_corpus.py`, `json`, `test_upload_recovery_receipts.py`, `run_matrix_training`, `remediation.py`, `repository_sft.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.10336817653890824 - nodes in this community are weakly interconnected._