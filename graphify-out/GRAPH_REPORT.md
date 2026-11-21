# Graph Report - bifocalgemma  (2026-11-21)

## Corpus Check
- 257 files · ~370,432 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2623 nodes · 7192 edges · 190 communities (88 shown, 102 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 892 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `48b99705`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- wiggly_dashboard.py
- test_v4_architecture.py
- benchmark_upload.py
- build_native_tiny_model
- RepositorySnapshot
- stages.py
- test_dashboard_dependency_state.py
- test_memory_efficient_training.py
- test_effective_benchmark_protocol.py
- test_matrix_runtime_loader.py
- test_notebook_dependencies.py
- schedule.py
- build_and_publish_v4_dataset.py
- test_tier2_boundary.py
- TokenizedRepairExample
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- hashlib
- test_molab_matrix_notebooks.py
- context_tasks.py
- audit_dataset_release.py
- adversarial_r3_probe.py
- typing
- FastTrainingPipeline
- test_v4_data_disjoint.py
- curriculum.py
- curation.py
- external_runner.py
- examples
- pathlib
- prepare_context_data.py
- matrix.py
- test_gsm8k.py
- TrainingDashboard
- TestBifocalGemmaArchitecture
- benchmark.py
- scorer_fingerprint
- evaluation/context.py
- test_molab_remediation.py
- generate_full_matrix_notebooks.py
- test_offline_sft_training.py
- configuration_bifocalgemma.py
- test_tier4_scenarios.py
- repository_study.py
- test_repository_study.py
- run_matrix_stage
- CodeOverlapIndex
- test_matrix_schedule_runtime.py
- notebook_launcher.py
- BifocalGemmaForCausalLM
- objectives.py
- hashlib
- notebook_dependencies.py
- test_tier3_combinations.py
- GrokOrScaleLoRA
- test_training_telemetry.py
- NativeTinyModel
- test_context_tasks.py
- test_pretrained_probe_helpers.py
- test_repository_corpus.py
- subprocess
- test_construction_blocks.py
- fit_offline_sft
- test_editing_manifest.py
- generate_repository_notebook.py
- restore_matrix_checkpoint
- build_repository_corpus
- training/diagnostics.py
- math
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- TestTokenRoleAndDataIdentity
- bifocalgemma
- releases.py
- types
- Seq2SeqBatch
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- normalize_context_task
- test_context_benchmark_mcq_budget_floor_execution
- test_tier1_features.py
- .config
- pytest
- test_optimized_native_resume_matches_full_and_rejects_different_recipe
- tokenize_context_task
- RecordingStream
- .snapshot
- patch_matrix_tokenization
- DummyTrackingTokenizer
- test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only
- PackedTokenizedDataset
- json
- test_construction_blocks.py
- WigglyDashboard
- _CharTokenizer
- curated_sft.py
- test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- stable_fingerprint
- dependency_state
- test_prompt_ids_instance_id_isolation_and_crosscodelong_normalization
- fast_pipeline.py
- test_evaluation.py
- SwiGLUDualGateBridge
- test_benchmark_prompt_budget.py
- test_publication_only_adds_verified_immutable_runtime
- test_archive_download_retries_incomplete_transport
- test_fast_pipeline_respects_learnable_cross_scale_default_and_override
- configure_full_rank_cross_attention
- patch_cross_entropy
- CachedAttention
- tests_e2e/conftest.py

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 67 edges
2. `atomic_json()` - 61 edges
3. `run_matrix_training()` - 51 edges
4. `load_verified_tokenizer()` - 51 edges
5. `BifocalGemmaForCausalLM` - 49 edges
6. `RepositorySnapshot` - 45 edges
7. `run_matrix_benchmark()` - 41 edges
8. `TokenizedRepairExample` - 41 edges
9. `run_matrix_stage()` - 39 edges
10. `FastTrainingPipeline` - 38 edges

## Surprising Connections (you probably didn't know these)
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/modeling_bifocalgemma.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/models/bridge.py
- `test_fast_training_config_step_reconciliation()` --calls--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py
- `test_repair_record_type_has_same_prefix_as_explicit_repair_task()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `generate()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (190 total, 102 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.20
Nodes (9): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution(), save() (+1 more)

### Community 1 - "wiggly_dashboard.py"
Cohesion: 0.15
Nodes (5): format_stage_display(), LastSeenWidget, phase_category(), TrainingLossLog, test_dashboard_stage_display_formatting()

### Community 2 - "test_v4_architecture.py"
Cohesion: 0.10
Nodes (8): critical_attention_beta(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_gate_init_tanh_resolution_dynamic_and_v4_fallback(), test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa(), test_sdpa_attention_structured_masks_match_tiled_eager(), test_sdpa_cross_attention_matches_eager_and_handles_all_masked_rows()

### Community 3 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (17): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed() (+9 more)

### Community 5 - "build_native_tiny_model"
Cohesion: 0.14
Nodes (5): build_native_tiny_model(), run_native_smoke(), TestNativeIntegration, TestMemoryProbe, test_last_token_logits_match_full_head()

### Community 6 - "RepositorySnapshot"
Cohesion: 0.11
Nodes (26): RepositoryFile, RepositorySnapshot, _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application() (+18 more)

### Community 7 - "stages.py"
Cohesion: 0.08
Nodes (16): remote_reset_plan(), reset_local_benchmarks(), reset_state(), checked_prefix(), recover_remote_run(), _copy_atomic(), retry_delay(), _synchronize_once() (+8 more)

### Community 8 - "test_dashboard_dependency_state.py"
Cohesion: 0.14
Nodes (15): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+7 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.24
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 11 - "test_effective_benchmark_protocol.py"
Cohesion: 0.11
Nodes (11): batched_greedy_decode(), greedy_decode(), GreedyGeneration, test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold(), decode(), test_effective_allowance(), test_effective_spec_binds_gold_caps_and_runtime_spec(), test_invalid_allowance_fails_explicitly() (+3 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.34
Nodes (11): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_release_check_rejects_local_preview() (+3 more)

### Community 14 - "test_notebook_dependencies.py"
Cohesion: 0.18
Nodes (12): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_new_torch_pin_forces_reinstall_despite_old_receipt_or_matching_install(), test_preloaded_torch_requires_real_kernel_restart() (+4 more)

### Community 15 - "schedule.py"
Cohesion: 0.09
Nodes (28): _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule() (+20 more)

### Community 16 - "build_and_publish_v4_dataset.py"
Cohesion: 0.12
Nodes (17): atomic_json(), build_and_verify_mixed_pool(), build_and_verify_release(), _build_v4_notices(), _extract_python_fim_from_code(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers() (+9 more)

### Community 17 - "test_tier2_boundary.py"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 18 - "TokenizedRepairExample"
Cohesion: 0.15
Nodes (8): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), synthetic_stress_target(), TestDataAndLoss, test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes(), test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.07
Nodes (7): MockCausalLM, MockQwenTokenizer, _ProportionalTokenizer, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(), test_windowed_file_section_disjoint_split_and_train_eval_parity()

### Community 20 - "eval_code.py"
Cohesion: 0.14
Nodes (7): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (26): completion_metrics(), extract_mcq_letter(), _sequence_and_identifier_metrics(), strict_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards() (+18 more)

### Community 23 - "hashlib"
Cohesion: 0.09
Nodes (18): _clean_extracted_number(), digest(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+10 more)

### Community 24 - "test_molab_matrix_notebooks.py"
Cohesion: 0.14
Nodes (19): make_experiment_spec(), configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), fixture_model(), test_trained_adapters_export_reload_logit_parity(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_comprehension_targets_stay_inside_baseline_visible_prefix() (+11 more)

### Community 25 - "context_tasks.py"
Cohesion: 0.24
Nodes (11): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), _parse_file_sections(), _render_entries(), _split_crosscodelong_context() (+3 more)

### Community 26 - "audit_dataset_release.py"
Cohesion: 0.20
Nodes (5): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), audit(), emit(), main(), validate_all_patches()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "typing"
Cohesion: 0.06
Nodes (20): flex_kernel_options(), fused_attention(), cached_logits(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), mask_tile(), sdpa_attention() (+12 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.05
Nodes (18): FastTrainingConfig, FastTrainingPipeline, JSONLRepairDataset, run_fast_training(), is_cross_bridge_parameter(), test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke() (+10 more)

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.24
Nodes (9): _load_prepare_module(), _numbered_line_numbers(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_format_numbered_code_and_exports(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_multifunction_completion_extraction_removes_single_return_cap(), test_v4_quarantine_licenses_dedup_and_nextcoder_rejection() (+1 more)

### Community 31 - "curriculum.py"
Cohesion: 0.21
Nodes (8): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 32 - "curation.py"
Cohesion: 0.09
Nodes (27): export_pool(), file_digest(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool() (+19 more)

### Community 34 - "external_runner.py"
Cohesion: 0.10
Nodes (20): prepare_sampled_benchmark_collection(), format_crosscodelong_summary(), score_repo_niah_case(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch() (+12 more)

### Community 35 - "examples"
Cohesion: 0.27
Nodes (3): make_identity(), examples(), TestTrainingPolicy

### Community 36 - "pathlib"
Cohesion: 0.05
Nodes (33): DeveloperAgent, BackboneProfile, CheckpointRef, ExperimentSpec, get_v2_experiment(), _pair(), _ref(), validate_experiment_matrix() (+25 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.09
Nodes (26): is_permissive_license(), normalize_context_task(), load_curated_pool(), context_license_evidence(), test_repository_pool_requires_context_license_policy_and_coverage(), adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints() (+18 more)

### Community 38 - "matrix.py"
Cohesion: 0.07
Nodes (31): load_matrix_adapters_for_inference(), synchronize_matrix(), benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), _backfill_final_stage(), _build_schedule_for_epochs() (+23 more)

### Community 39 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 40 - "TrainingDashboard"
Cohesion: 0.15
Nodes (6): format_duration(), loss_chart(), render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 42 - "benchmark.py"
Cohesion: 0.09
Nodes (18): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), run_batched_context_benchmark(), activity(), run_context_benchmark(), activity(), build_effective_benchmark_spec(), effective_generation_budget() (+10 more)

### Community 43 - "scorer_fingerprint"
Cohesion: 0.20
Nodes (8): _extract_match_letter(), fallback_letter_regex(), score_context_case(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer(), score_gsm8k(), scorer_fingerprint()

### Community 44 - "evaluation/context.py"
Cohesion: 0.07
Nodes (31): build_repo_niah_case(), _context_already_includes_instruction(), _count_niah_tokens(), decoder_prompt(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens(), encoder_context(), _extract_full_python_code() (+23 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.10
Nodes (7): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues(), test_matrix_quarantine_legacy_checkpoint_vs_clean_resume(), test_mcq_budget_floor_in_v3_benchmark_logic()

### Community 47 - "generate_full_matrix_notebooks.py"
Cohesion: 0.33
Nodes (6): test_invalid_generation_modes_cannot_publish(), format_code(), main(), matrix_bundle(), render_notebook(), source_bundle_bytes()

### Community 48 - "test_offline_sft_training.py"
Cohesion: 0.11
Nodes (5): TestOfflineSFT, TinyLossModel, limit_threads(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 49 - "configuration_bifocalgemma.py"
Cohesion: 0.08
Nodes (4): _check_transformers_version(), native_config(), BifocalGemmaPreTrainedModel, NativeMergedAttention

### Community 50 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 51 - "repository_study.py"
Cohesion: 0.22
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), summarize_native_results(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.12
Nodes (18): v4_repository_task_from_row(), prepare_repository_record(), repository_task_from_row(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions, fixture() (+10 more)

### Community 53 - "run_matrix_stage"
Cohesion: 0.10
Nodes (17): run_matrix_stage(), training_policy_table(), update_policy_learning_rates(), build_bifocalgemma_adamw(), clip_optimizer_group_grads(), configure_matrix_stage(), test_stage1_to_stage2_continuous_warmup_offset_and_per_group_grad_norms(), _synthetic_cohort() (+9 more)

### Community 54 - "CodeOverlapIndex"
Cohesion: 0.06
Nodes (45): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits() (+37 more)

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.07
Nodes (9): fixture_data(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_partial_backward_oom_replays_entire_window_without_double_gradient(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work() (+1 more)

### Community 56 - "notebook_launcher.py"
Cohesion: 0.36
Nodes (6): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 58 - "BifocalGemmaForCausalLM"
Cohesion: 0.04
Nodes (27): BifocalGemmaConfig, BifocalGemmaForCausalLM, _resolve_backbone_weights(), FullCrossTiledAttention, InferenceCache, ProbeLoRALinear, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init() (+19 more)

### Community 59 - "objectives.py"
Cohesion: 0.08
Nodes (19): _resolve_example_diff_flag(), backward_window(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_example_loss_weights() (+11 more)

### Community 60 - "hashlib"
Cohesion: 0.09
Nodes (20): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context() (+12 more)

### Community 62 - "notebook_dependencies.py"
Cohesion: 0.20
Nodes (5): test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols(), test_source_bundle_checksum_failure_does_not_activate_source(), activate_notebook_bundle(), import_native_gemma4()

### Community 63 - "test_tier3_combinations.py"
Cohesion: 0.19
Nodes (4): validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 64 - "GrokOrScaleLoRA"
Cohesion: 0.24
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.20
Nodes (3): test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime()

### Community 67 - "NativeTinyModel"
Cohesion: 0.22
Nodes (5): ModelOutput, NativeTinyModel, test_cross_scale_frozen_by_default_and_configurable(), test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), spy_sdpa()

### Community 68 - "test_context_tasks.py"
Cohesion: 0.12
Nodes (7): repair_preimage_coverage(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_tokenize_context_task_preserves_target_indentation_and_newlines()

### Community 69 - "test_pretrained_probe_helpers.py"
Cohesion: 0.20
Nodes (5): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library(), test_notebook_sweep_persists_results_per_case()

### Community 72 - "test_repository_corpus.py"
Cohesion: 0.44
Nodes (7): patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_context_files_are_licensed_separately_from_patch(), test_license_document_include_is_resolved_inside_snapshot(), test_nested_licensing_overrides_root_and_traversal_rejected()

### Community 74 - "subprocess"
Cohesion: 0.21
Nodes (7): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "fit_offline_sft"
Cohesion: 0.28
Nodes (12): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), build_training_groups() (+4 more)

### Community 79 - "test_editing_manifest.py"
Cohesion: 0.26
Nodes (9): editing_manifest(), repository_editing_record(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation() (+1 more)

### Community 80 - "generate_repository_notebook.py"
Cohesion: 0.33
Nodes (3): main(), render(), source_bundle()

### Community 82 - "restore_matrix_checkpoint"
Cohesion: 0.24
Nodes (4): prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint()

### Community 84 - "training/diagnostics.py"
Cohesion: 0.32
Nodes (4): encoder_diagnostics(), stratified_subset(), evaluate_sft_loss(), validation_batches()

### Community 85 - "math"
Cohesion: 0.06
Nodes (15): build_bifocalgemma_optimizer(), compute_step_loss(), extract_trivial_token_ids(), is_zero_weight_decay_bridge_parameter(), resolve_diff_token_ids(), streaming_pointer_auxiliary_loss(), _build_group_schedule(), Float32Alignment (+7 more)

### Community 91 - "releases.py"
Cohesion: 0.08
Nodes (28): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+20 more)

### Community 95 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 96 - "normalize_context_task"
Cohesion: 0.25
Nodes (7): generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record(), test_context_license_record_covers_all_visible_paths()

### Community 101 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 102 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 104 - "pytest"
Cohesion: 0.43
Nodes (4): parent(), test_evaluation_and_unlicensed_sources_rejected(), test_grounded_generation_answerability_provenance_and_target_independence(), test_grounded_mutation_rejected()

### Community 105 - "test_optimized_native_resume_matches_full_and_rejects_different_recipe"
Cohesion: 0.29
Nodes (4): comparison_recipes(), test_fp32_alignment_on_bf16_backbone(), test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 106 - "tokenize_context_task"
Cohesion: 0.38
Nodes (5): _resolve_eot_token_id(), task_prompt(), tokenize_context_task(), _use_disjoint_mode(), test_disjoint_split_completion_zero_overlap_and_three_way_parity()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 116 - "json"
Cohesion: 0.07
Nodes (18): ArchiveSnapshotStore, test_f5_training_semantics_change_rejects_legacy_approval(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main() (+10 more)

### Community 120 - "WigglyDashboard"
Cohesion: 0.29
Nodes (6): WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 122 - "curated_sft.py"
Cohesion: 0.29
Nodes (6): prepare_editing_records(), split_editing_records(), default_training_profiles(), fit_repository_sft(), prepare_repository_training(), TestOfflineEditing

### Community 130 - "stable_fingerprint"
Cohesion: 0.10
Nodes (21): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), curated_pool_digest(), stable_fingerprint() (+13 more)

### Community 142 - "fast_pipeline.py"
Cohesion: 0.09
Nodes (12): ensure_assets_on_disk(), load_asset_json(), export_matrix_pretrained_model(), collate_fast_batch(), SyntheticRepairDataset, configure_probe_training(), gpu_memory(), optimizer_bytes() (+4 more)

### Community 146 - "test_evaluation.py"
Cohesion: 0.15
Nodes (3): test_crosscodelong_normalize_completion_prediction_strips_fences_and_suffix_overlap(), test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity(), test_prompt_ids_instance_id_isolation_and_gold_rejection()

### Community 171 - "SwiGLUDualGateBridge"
Cohesion: 0.14
Nodes (4): NormalizedAlignment, SwiGLUDualGateBridge, test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight(), test_swiglu_bridge_non_redundant_single_post_gain()

### Community 172 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 180 - "configure_full_rank_cross_attention"
Cohesion: 0.09
Nodes (11): Bf16FullLinear, _compute_per_row_cross_query(), configure_full_rank_cross_attention(), FullCrossCachedAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), _resolve_gate_init_tanh() (+3 more)

### Community 196 - "tests_e2e/conftest.py"
Cohesion: 0.15
Nodes (6): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1017 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **102 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `test_v4_architecture.py`, `fast_pipeline.py`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `configuration_bifocalgemma.py`, `configure_full_rank_cross_attention`, `typing`, `FastTrainingPipeline`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_v4_architecture.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10333333333333333 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `build_and_publish_v4_dataset.py` to `preflight.py`, `curation.py`, `stable_fingerprint`, `external_runner.py`, `benchmark_upload.py`, `prepare_context_data.py`, `matrix.py`, `stages.py`, `benchmark.py`, `evaluation/context.py`, `schedule.py`, `json`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `objectives.py`, `types`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Should `benchmark_upload.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07293868921775898 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `curation.py`, `preflight.py`, `external_runner.py`, `audit_dataset_release.py`, `prepare_context_data.py`, `matrix.py`, `stages.py`, `benchmark.py`, `evaluation/context.py`, `test_editing_manifest.py`, `build_repository_corpus`, `json`, `CodeOverlapIndex`, `test_matrix_schedule_runtime.py`, `curated_sft.py`, `releases.py`, `hashlib`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._