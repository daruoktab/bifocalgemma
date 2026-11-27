# Graph Report - bifocalgemma  (2026-11-27)

## Corpus Check
- 260 files · ~376,557 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 17 file(s) not represented in the graph (top: .tex 5, (none) 2, .jsonl 2)

## Summary
- 2680 nodes · 7346 edges · 181 communities (87 shown, 94 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 812 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9ee31f83`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- run_matrix_stage
- test_v4_architecture.py
- benchmark_upload.py
- modeling_bifocalgemma.py
- RepositorySnapshot
- tests_e2e/conftest.py
- test_notebook_dependencies.py
- test_memory_efficient_training.py
- FeatureNotReadyError
- test_matrix_runtime_loader.py
- test_pretrained_hf_model.py
- schedule.py
- make_experiment_spec
- stable_fingerprint
- TokenizedRepairExample
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- test_tier2_boundary.py
- NativeTinyModel
- context_tasks.py
- gsm8k.py
- adversarial_r3_probe.py
- BifocalGemmaConfig
- FastTrainingPipeline
- test_v4_data_disjoint.py
- curriculum.py
- curation.py
- test_training_telemetry.py
- run_external_benchmark
- test_dashboard_dependency_state.py
- scorer_fingerprint
- prepare_context_data.py
- fast_pipeline.py
- hub.py
- math
- TestBifocalGemmaArchitecture
- scorer_fingerprint
- test_gsm8k.py
- evaluation/context.py
- .generate
- remediation.py
- configure_policy_model
- releases.py
- SwiGLUDualGateBridge
- repository_study.py
- test_repository_study.py
- .configure_stage1
- test_v4_milestone_checkpoints_survive_pruning_and_hub_retention
- test_matrix_schedule_runtime.py
- generate_full_matrix_notebooks.py
- BifocalGemmaForCausalLM
- objectives.py
- repository_corpus.py
- test_shared_schedule.py
- audit_context_release.py
- .configure_stage1
- notebook_launcher.py
- .from_pretrained
- memory_efficient.py
- test_context_tasks.py
- test_tier4_scenarios.py
- test_context_benchmark_mcq_budget_floor_execution
- subprocess
- test_construction_blocks.py
- fit_offline_sft
- notebook_dependencies.py
- generate_repository_notebook.py
- matrix.py
- RepositorySnapshot
- typing
- ExperimentSpec
- pathlib
- bifocalgemma
- build_and_publish_v4_dataset.py
- test_overflow_fallback_recomputes_budget_and_records_metadata
- test_training_recipes.py
- pytest
- test_molab_remediation.py
- test_tier1_features.py
- TestTokenRoleAndDataIdentity
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- FastTrainingConfig
- test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa
- tokenization.py
- test_tokenize_context_task_preserves_target_indentation_and_newlines
- patch_matrix_tokenization
- test_fast_pipeline.py
- bifocalgemma/__init__.py
- patch_cross_entropy
- argparse
- _CharTokenizer
- dependency_state
- run_batched_context_benchmark
- hashlib
- test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- .snapshot
- CachedAttention
- memory_probe.py
- JSONLRepairDataset
- sdpa_cross_attention
- test_molab_matrix_notebooks.py

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 65 edges
2. `atomic_json()` - 61 edges
3. `BifocalGemmaForCausalLM` - 59 edges
4. `load_verified_tokenizer()` - 51 edges
5. `run_matrix_training()` - 51 edges
6. `RepositorySnapshot` - 45 edges
7. `TokenizedRepairExample` - 41 edges
8. `run_matrix_benchmark()` - 41 edges
9. `FastTrainingPipeline` - 40 edges
10. `BifocalGemmaConfig` - 39 edges

## Surprising Connections (you probably didn't know these)
- `test_number_lines_false_preserves_genuine_digit_colon_source_lines()` --calls--> `split_context_for_mode()`  [INFERRED]
  tests/test_eval_remediation.py → src/bifocalgemma/evaluation/context.py
- `test_recount_functional_pass_rate_returns_none_on_partial_execution()` --calls--> `repair_report_metrics()`  [INFERRED]
  tests/test_eval_remediation.py → src/bifocalgemma/evaluation/protocol.py
- `test_fast_training_config_step_reconciliation()` --uses--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_telemetry_history_appends_all_active_stages()` --calls--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (181 total, 94 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.11
Nodes (15): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), build_training_groups(), validate_training_policy(), current_capacity(), execution_settings() (+7 more)

### Community 1 - "run_matrix_stage"
Cohesion: 0.12
Nodes (9): _build_schedule_for_epochs(), _resolve_example_diff_flag(), run_matrix_stage(), backward_window(), compute_example_loss_weights(), SharedSchedule, clip_optimizer_group_grads(), test_matrix_stage_microbatch_invariance_under_heterogeneous_lengths() (+1 more)

### Community 2 - "test_v4_architecture.py"
Cohesion: 0.12
Nodes (13): configure_full_rank_cross_attention(), critical_attention_beta(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_cross_scale_frozen_by_default_and_configurable(), test_decoupled_upper_full_attention_cross_kv_topology(), test_full_cross_cached_attention_sdpa_guard_fallback_and_non_left_padded_step_valid(), _run_cached() (+5 more)

### Community 3 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (21): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush() (+13 more)

### Community 5 - "modeling_bifocalgemma.py"
Cohesion: 0.10
Nodes (7): FullCrossCachedAttention, cached_logits(), InferenceCache, save_matrix_checkpoint(), is_cross_bridge_parameter(), clamped_triton_kernel_options(), test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence()

### Community 6 - "RepositorySnapshot"
Cohesion: 0.11
Nodes (22): RepositoryFile, _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff() (+14 more)

### Community 7 - "tests_e2e/conftest.py"
Cohesion: 0.10
Nodes (9): authoritative_empirical_results(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization() (+1 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.18
Nodes (12): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_new_torch_pin_forces_reinstall_despite_old_receipt_or_matching_install(), test_preloaded_torch_requires_real_kernel_restart() (+4 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.25
Nodes (13): configure_memory_efficient_training(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+5 more)

### Community 11 - "FeatureNotReadyError"
Cohesion: 0.09
Nodes (12): DeveloperAgent, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter, extract_token_ast_features() (+4 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.38
Nodes (11): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_release_check_rejects_local_preview() (+3 more)

### Community 14 - "test_pretrained_hf_model.py"
Cohesion: 0.14
Nodes (9): export_matrix_pretrained_model(), _resolve_backbone_weights(), test_bifocalgemma_config_initialization_and_serialization(), test_export_matrix_pretrained_model(), test_export_matrix_pretrained_model_baseline(), test_fast_training_config_step_reconciliation(), test_fast_training_pipeline_baseline_2b(), test_from_pretrained_backbones_baseline() (+1 more)

### Community 15 - "schedule.py"
Cohesion: 0.12
Nodes (18): prepare_curated_training(), tokenize(), _schedule_signature_matches(), _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer() (+10 more)

### Community 16 - "make_experiment_spec"
Cohesion: 0.17
Nodes (11): make_experiment_spec(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_partial_backward_oom_replays_entire_window_without_double_gradient(), test_stage1_to_stage2_continuous_warmup_offset_and_per_group_grad_norms(), test_trained_adapters_export_reload_logit_parity() (+3 more)

### Community 17 - "stable_fingerprint"
Cohesion: 0.24
Nodes (12): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), prompt_ids() (+4 more)

### Community 18 - "TokenizedRepairExample"
Cohesion: 0.09
Nodes (12): Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), fit_curated_sft(), synthetic_stress_target(), TestDataAndLoss (+4 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.07
Nodes (11): _prompt_ids_impl(), MockCausalLM, MockQwenTokenizer, _ProportionalTokenizer, test_eval_external_baselines_cli_dry_run(), test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic() (+3 more)

### Community 20 - "eval_code.py"
Cohesion: 0.09
Nodes (8): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity(), test_prompt_ids_instance_id_isolation_and_gold_rejection(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): extract_mcq_letter(), strict_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization() (+15 more)

### Community 23 - "test_tier2_boundary.py"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 24 - "NativeTinyModel"
Cohesion: 0.14
Nodes (5): ModelOutput, NativeTinyModel, NormalizedAlignment, test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), spy_sdpa()

### Community 25 - "context_tasks.py"
Cohesion: 0.17
Nodes (17): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), _parse_file_sections(), _render_entries(), _resolve_eot_token_id() (+9 more)

### Community 26 - "gsm8k.py"
Cohesion: 0.05
Nodes (22): digest(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k(), main() (+14 more)

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "BifocalGemmaConfig"
Cohesion: 0.17
Nodes (5): BifocalGemmaConfig, test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_generate_multi_eos_per_row_stopping_and_cache_parity(), test_generate_right_padded_matches_left_padded(), test_decoupled_full_attn_cross_kv_fp32_master_weights_and_gradients()

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.18
Nodes (9): v4_repository_task_from_row(), _load_prepare_module(), _numbered_line_numbers(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_multifunction_completion_extraction_removes_single_return_cap(), test_v4_quarantine_licenses_dedup_and_nextcoder_rejection() (+1 more)

### Community 31 - "curriculum.py"
Cohesion: 0.19
Nodes (8): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), test_mixed_pool_v4_curriculum_grounded_and_curated_sft()

### Community 32 - "curation.py"
Cohesion: 0.07
Nodes (33): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits(), count_syntax_warnings() (+25 more)

### Community 33 - "test_training_telemetry.py"
Cohesion: 0.05
Nodes (14): LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard, _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime() (+6 more)

### Community 34 - "run_external_benchmark"
Cohesion: 0.08
Nodes (18): prepare_sampled_benchmark_collection(), activity(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort() (+10 more)

### Community 35 - "test_dashboard_dependency_state.py"
Cohesion: 0.18
Nodes (12): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+4 more)

### Community 36 - "scorer_fingerprint"
Cohesion: 0.16
Nodes (10): _extract_match_letter(), fallback_letter_regex(), score_context_case(), score_repo_niah_case(), _clean_extracted_number(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer() (+2 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.10
Nodes (24): is_permissive_license(), normalize_context_task(), adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), completion_from_snapshot() (+16 more)

### Community 38 - "fast_pipeline.py"
Cohesion: 0.09
Nodes (6): ensure_assets_on_disk(), load_asset_json(), _check_transformers_version(), native_config(), build_bifocalgemma_optimizer(), extract_trivial_token_ids()

### Community 39 - "hub.py"
Cohesion: 0.05
Nodes (17): remote_reset_plan(), reset_state(), checked_prefix(), recover_remote_run(), _copy_atomic(), retry_delay(), _synchronize_once(), synchronize_run() (+9 more)

### Community 40 - "math"
Cohesion: 0.08
Nodes (11): is_transient_lfs_pointer_error(), _build_group_schedule(), update_policy_learning_rates(), append_event(), format_duration(), format_stage_display(), loss_chart(), render_dashboard() (+3 more)

### Community 42 - "scorer_fingerprint"
Cohesion: 0.12
Nodes (18): run_context_benchmark(), _paired_budget_for(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer() (+10 more)

### Community 43 - "test_gsm8k.py"
Cohesion: 0.16
Nodes (8): FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train(), test_verified_preparation_and_input_only_loading()

### Community 44 - "evaluation/context.py"
Cohesion: 0.06
Nodes (34): build_repo_niah_case(), completion_metrics(), _context_already_includes_instruction(), _count_niah_tokens(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens(), encoder_context(), _extract_full_python_code() (+26 more)

### Community 45 - ".generate"
Cohesion: 0.20
Nodes (3): BifocalGemmaForCausalLM, test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 47 - "remediation.py"
Cohesion: 0.07
Nodes (38): export_pool(), file_digest(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool() (+30 more)

### Community 48 - "configure_policy_model"
Cohesion: 0.18
Nodes (12): configure_policy_model(), training_policy_table(), unified_recipe(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_run_matrix_stage_execution_batch2_and_frozen_decoder_invariance(), test_save_and_restore_matrix_checkpoint_and_prune(), test_stage1_to_stage2_optimizer_state_transfer(), test_v3_partitioned_optimizer_weight_decay_invariants() (+4 more)

### Community 49 - "releases.py"
Cohesion: 0.08
Nodes (32): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+24 more)

### Community 50 - "SwiGLUDualGateBridge"
Cohesion: 0.10
Nodes (7): SwiGLUDualGateBridge, Bf16FullLinear, FullCrossTiledAttention, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight()

### Community 51 - "repository_study.py"
Cohesion: 0.16
Nodes (25): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), prepare_repository_record(), repository_task_from_row(), RepositoryTask (+17 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.10
Nodes (11): CodeOverlapMatch, project_root(), diff(), ExecutingScoringOptions, fixture(), ScoringOptions, ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards() (+3 more)

### Community 54 - "test_v4_milestone_checkpoints_survive_pruning_and_hub_retention"
Cohesion: 0.13
Nodes (3): test_publish_release_replace_current_deletes_audits(), test_v4_milestone_checkpoints_survive_pruning_and_hub_retention(), _write_ready_ckpt()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "generate_full_matrix_notebooks.py"
Cohesion: 0.21
Nodes (7): test_invalid_generation_modes_cannot_publish(), test_publication_only_adds_verified_immutable_runtime(), download(), format_code(), main(), matrix_bundle(), render_notebook()

### Community 59 - "objectives.py"
Cohesion: 0.07
Nodes (17): chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_step_loss(), differential_diff_cross_entropy(), is_unified_diff_target_tokens() (+9 more)

### Community 60 - "repository_corpus.py"
Cohesion: 0.13
Nodes (14): context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license(), resolved_license(), archive() (+6 more)

### Community 61 - "test_shared_schedule.py"
Cohesion: 0.28
Nodes (12): rows(), schedule(), test_all_originals_once_with_disjoint_stage_windows(), test_atomic_manifest_reuse_and_fail_closed_mismatch(), test_deterministic_model_independent_schedule_and_mapping(), test_each_semantic_mutation_changes_signature(), test_empty_required_fields_rejected(), test_foundation_context_and_target_patch_are_bound() (+4 more)

### Community 62 - "audit_context_release.py"
Cohesion: 0.20
Nodes (8): ArchiveSnapshotStore, repo_identity(), test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 64 - "notebook_launcher.py"
Cohesion: 0.30
Nodes (7): _write_mcq_benchmark(), test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 66 - ".from_pretrained"
Cohesion: 0.21
Nodes (6): test_cached_logits_decoder_only_not_polluted_by_cross_attention(), test_from_pretrained_device_map_and_missing_bridge_validation(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_save_and_from_pretrained_roundtrip()

### Community 67 - "memory_efficient.py"
Cohesion: 0.08
Nodes (18): flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), _kv_slice_bounds(), mask_tile(), sdpa_attention() (+10 more)

### Community 68 - "test_context_tasks.py"
Cohesion: 0.12
Nodes (9): repair_preimage_coverage(), task(), test_dependency_completion_removes_target_and_retains_real_helper(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context() (+1 more)

### Community 69 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 72 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 74 - "subprocess"
Cohesion: 0.12
Nodes (9): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), fixture_records(), TestSWESmithContract (+1 more)

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "fit_offline_sft"
Cohesion: 0.07
Nodes (23): encoder_diagnostics(), stratified_subset(), cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage() (+15 more)

### Community 78 - "notebook_dependencies.py"
Cohesion: 0.20
Nodes (5): test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols(), test_source_bundle_checksum_failure_does_not_activate_source(), activate_notebook_bundle(), import_native_gemma4()

### Community 80 - "generate_repository_notebook.py"
Cohesion: 0.33
Nodes (4): main(), render(), source_bundle(), source_bundle_bytes()

### Community 82 - "matrix.py"
Cohesion: 0.05
Nodes (37): reset_local_benchmarks(), load_matrix_adapters_for_inference(), _milestone_checkpoint_names(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), synchronize_matrix(), benchmark_directory_name() (+29 more)

### Community 83 - "RepositorySnapshot"
Cohesion: 0.18
Nodes (13): RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), example_from_record(), prepare_repository_record_with_snapshot() (+5 more)

### Community 85 - "typing"
Cohesion: 0.08
Nodes (14): is_zero_weight_decay_bridge_parameter(), default_training_profiles(), Float32Alignment, v4_group_defaults(), v4_optimizer_defaults(), build_bifocalgemma_adamw(), build_partitioned_param_groups(), classify_bifocalgemma_parameter() (+6 more)

### Community 87 - "ExperimentSpec"
Cohesion: 0.16
Nodes (7): ExperimentSpec, get_v2_experiment(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 89 - "pathlib"
Cohesion: 0.07
Nodes (8): BackboneProfile, CheckpointRef, _pair(), _ref(), EvaluationReport, BaselineScores, TestTokenizerVerification, test_notebook_stress_helpers_match_shared_library()

### Community 91 - "build_and_publish_v4_dataset.py"
Cohesion: 0.15
Nodes (16): build_and_verify_mixed_pool(), build_and_verify_release(), _build_v4_notices(), _extract_python_fim_from_code(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers(), _log() (+8 more)

### Community 92 - "test_overflow_fallback_recomputes_budget_and_records_metadata"
Cohesion: 0.15
Nodes (6): _CharTokenizer, _DummyEvalModel, test_effective_spec_includes_overflow_fallback_policy(), test_external_runner_resume_and_signature_guard(), test_number_lines_false_preserves_genuine_digit_colon_source_lines(), test_recount_functional_pass_rate_returns_none_on_partial_execution()

### Community 93 - "test_training_recipes.py"
Cohesion: 0.25
Nodes (4): comparison_recipes(), limit_threads(), test_fp32_alignment_on_bf16_backbone(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 96 - "pytest"
Cohesion: 0.05
Nodes (22): generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record(), build_packed_cache(), open_packed_cache() (+14 more)

### Community 101 - "test_molab_remediation.py"
Cohesion: 0.12
Nodes (6): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_legacy_checkpoint_vs_clean_resume(), test_mcq_budget_floor_in_v3_benchmark_logic()

### Community 102 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (12): source_fingerprints(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval() (+4 more)

### Community 104 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 105 - "FastTrainingConfig"
Cohesion: 0.22
Nodes (5): FastTrainingConfig, test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_pipeline_respects_learnable_cross_scale_default_and_override()

### Community 108 - "tokenization.py"
Cohesion: 0.11
Nodes (13): RepairExample, decode_text(), format_encoder_context(), tokenize_repair_example(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration (+5 more)

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.25
Nodes (4): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.15
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 116 - "argparse"
Cohesion: 0.13
Nodes (6): main(), main(), distribution(), main(), profile(), main()

### Community 121 - "run_batched_context_benchmark"
Cohesion: 0.12
Nodes (12): batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), run_batched_context_benchmark(), activity(), _paired_budget_for(), decoder_prompt(), _prompt_ids_with_metadata() (+4 more)

### Community 122 - "hashlib"
Cohesion: 0.07
Nodes (28): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines() (+20 more)

### Community 137 - "memory_probe.py"
Cohesion: 0.14
Nodes (11): configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), run_length_grid(), tensor_bytes(), test_length_grid_matches_baseline_budget_and_stops_after_failure() (+3 more)

### Community 142 - "JSONLRepairDataset"
Cohesion: 0.15
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 180 - "sdpa_cross_attention"
Cohesion: 0.14
Nodes (8): _compute_per_row_cross_query(), _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), _resolve_gate_init_tanh(), sdpa_cross_attention(), _sdpa_supported(), test_compute_per_row_cross_query_has_no_host_sync_calls()

### Community 197 - "test_molab_matrix_notebooks.py"
Cohesion: 0.08
Nodes (10): test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_per_model_hf_repo(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_pointer_loss_gradient_checkpointing_and_trivial_filtering() (+2 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1042 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **94 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `.generate` to `.from_pretrained`, `memory_efficient.py`, `test_v4_architecture.py`, `modeling_bifocalgemma.py`, `fast_pipeline.py`, `test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa`, `test_pretrained_hf_model.py`, `JSONLRepairDataset`, `make_experiment_spec`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `bifocalgemma/__init__.py`, `SwiGLUDualGateBridge`, `.configure_stage1`, `BifocalGemmaForCausalLM`, `BifocalGemmaConfig`, `FastTrainingPipeline`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `stable_fingerprint()` (e.g. with `test_generated_preparation_consumes_frozen_raw_repository_pool()` and `test_legacy_paired_manifest_remains_byte_identical()`) actually correct?**
  _`stable_fingerprint()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `preflight.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1103448275862069 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `curation.py`, `pytest`, `preflight.py`, `run_external_benchmark`, `prepare_context_data.py`, `hub.py`, `scorer_fingerprint`, `evaluation/context.py`, `remediation.py`, `schedule.py`, `releases.py`, `matrix.py`, `make_experiment_spec`, `test_matrix_schedule_runtime.py`, `hashlib`, `repository_corpus.py`, `audit_context_release.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `run_matrix_stage` be split into smaller, more focused modules?**
  _Cohesion score 0.11692307692307692 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `run_external_benchmark` to `pytest`, `preflight.py`, `benchmark_upload.py`, `modeling_bifocalgemma.py`, `prepare_context_data.py`, `hub.py`, `math`, `scorer_fingerprint`, `build_and_publish_v4_dataset.py`, `evaluation/context.py`, `fit_offline_sft`, `schedule.py`, `stable_fingerprint`, `matrix.py`, `releases.py`, `run_batched_context_benchmark`, `objectives.py`, `audit_context_release.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._