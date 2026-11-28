# Graph Report - bifocalgemma  (2026-11-28)

## Corpus Check
- 246 files · ~361,389 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: (none) 2, .jsonl 2, .toml 2)

## Summary
- 2740 nodes · 7482 edges · 178 communities (88 shown, 90 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 863 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8a5dcf17`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- TestTokenizerVerification
- test_tier1_features.py
- benchmark_upload.py
- test_pretrained_hf_model.py
- RepositorySnapshot
- atomic_json
- test_notebook_dependencies.py
- test_memory_efficient_training.py
- tokenization.py
- curation.py
- ProbeLoRALinear
- schedule.py
- native_python.py
- re
- TokenizedRepairExample
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- test_v4_data_disjoint.py
- NativeTinyModel
- subprocess
- build_native_tiny_model
- adversarial_r3_probe.py
- gsm8k.py
- FastTrainingPipeline
- benchmark_reset.py
- test_gsm8k.py
- curation.py
- test_training_telemetry.py
- run_external_benchmark
- json
- test_data_releases.py
- prepare_context_data.py
- json
- test_recover_remote_run_ignores_dot_temp_and_heartbeat_files
- telemetry.py
- curriculum.py
- run_batched_context_benchmark
- build_repository_corpus
- evaluation/context.py
- BifocalGemmaForCausalLM
- test_data_remediation.py
- test_eval_remediation.py
- SwiGLUDualGateBridge
- test_dashboard_dependency_state.py
- repository_study.py
- test_context_benchmark_resume.py
- matrix.py
- test_v4_checkpoint_and_patches.py
- test_matrix_schedule_runtime.py
- argparse
- test_v4_architecture.py
- compute_step_loss
- hashlib
- context_pool
- test_benchmark_prompt_budget.py
- policy.py
- test_benchmark_upload.py
- test_evaluation.py
- tiled_attention
- test_context_tasks.py
- .attempt
- limited
- data/context.py
- test_construction_blocks.py
- .snapshot
- fit_offline_sft
- TrainingDashboard
- scorer_fingerprint
- ScriptedModel
- benchmark.py
- TestConstructionEntrypoints
- test_v4_optimizer_schedule.py
- training/diagnostics.py
- test_upload_recovery_receipts.py
- bifocalgemma
- test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues
- normalize_context_task
- dependency_state
- test_tier1_features.py
- test_matrix_quarantine_legacy_checkpoint_vs_clean_resume
- Seq2SeqBatch
- BifocalGemmaConfig
- WigglyDashboard
- DummyTrackingTokenizer
- test_matrix_runtime_loader.py
- make_experiment_spec
- patch_cross_entropy
- TrainingLossLog
- resolve_diff_token_ids
- RepairExample
- RecordingStream
- test_molab_remediation.py
- fast_pipeline.py
- JSONLRepairDataset
- pathlib
- test_context_benchmark_mcq_budget_floor_execution
- PackedTokenizedDataset
- test_v4_checkpoint_and_patches.py
- math
- objectives.py
- configure_probe_training
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
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_swiglu_bridge_non_redundant_single_post_gain()` --uses--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_v4_architecture.py → src/bifocalgemma/models/bridge.py
- `test_pipeline_state_ledger_creation_for_remote_resumption()` --calls--> `StageLedger`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/stages.py
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `fetch()` --calls--> `repository_task_from_row()`  [INFERRED]
  tools/prepare_context_data.py → src/bifocalgemma/data/repository.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (178 total, 90 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.11
Nodes (14): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe() (+6 more)

### Community 2 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (12): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f2_mcq_token_budget_specification(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval() (+4 more)

### Community 3 - "benchmark_upload.py"
Cohesion: 0.16
Nodes (4): BenchmarkUploadQueue, is_transient_lfs_pointer_error(), append_event(), test_interval_throttles_periodic_upload_but_final_can_flush()

### Community 5 - "test_pretrained_hf_model.py"
Cohesion: 0.13
Nodes (8): FastTrainingConfig, test_fast_training_config_and_pipeline_baseline_single_stage(), test_bifocalgemma_config_initialization_and_serialization(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_training_config_step_reconciliation(), test_fast_pipeline_resume_identity_mismatch_and_corrupt_checkpoint(), test_fast_pipeline_respects_learnable_cross_scale_default_and_override()

### Community 6 - "RepositorySnapshot"
Cohesion: 0.12
Nodes (24): RepositoryFile, RepositorySnapshot, _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application() (+16 more)

### Community 7 - "atomic_json"
Cohesion: 0.15
Nodes (16): build_and_verify_mixed_pool(), build_and_verify_release(), _build_v4_notices(), _extract_python_fim_from_code(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers(), _log() (+8 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.11
Nodes (14): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+6 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.24
Nodes (9): run_training_memory_preflight(), examples(), make_model(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native(), test_optimized_long_window_causality_context_and_all_masked_backward(), test_optimized_resume_and_validation_loss_only() (+1 more)

### Community 11 - "tokenization.py"
Cohesion: 0.11
Nodes (10): DeveloperAgent, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter, extract_token_ast_features(), TokenASTFeatures (+2 more)

### Community 13 - "curation.py"
Cohesion: 0.09
Nodes (20): curate_paired_sources(), nextcoder_row(), read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows() (+12 more)

### Community 14 - "ProbeLoRALinear"
Cohesion: 0.13
Nodes (7): FullCrossTiledAttention, ProbeLoRALinear, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip()

### Community 15 - "schedule.py"
Cohesion: 0.10
Nodes (26): _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule() (+18 more)

### Community 16 - "native_python.py"
Cohesion: 0.10
Nodes (17): canonical_snapshot_hash(), NativeTaskResult, score_python_patch(), summarize_native_results(), authoritative_empirical_results(), project_root(), sample_crosscode_row(), sample_mcq_row() (+9 more)

### Community 17 - "re"
Cohesion: 0.13
Nodes (14): repair_preimage_coverage(), context_license_evidence(), path_license_evidence(), recognize_license(), resolved_license(), v4_repository_task_from_row(), prepare_repository_record(), prepare_repository_record_with_snapshot() (+6 more)

### Community 18 - "TokenizedRepairExample"
Cohesion: 0.10
Nodes (15): TokenizedRepairExample, RepairCollator, __getattr__(), synthetic_stress_target(), TestDataAndLoss, fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows() (+7 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.07
Nodes (8): MockCausalLM, MockQwenTokenizer, _ProportionalTokenizer, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(), test_reponiah_32k_zero_overflow_for_bifocalgemma_and_qwen25(), test_windowed_file_section_disjoint_split_and_train_eval_parity()

### Community 20 - "eval_code.py"
Cohesion: 0.15
Nodes (6): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): extract_mcq_letter(), strict_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization() (+15 more)

### Community 23 - "test_v4_data_disjoint.py"
Cohesion: 0.15
Nodes (11): _CharTokenizer, _load_prepare_module(), _numbered_line_numbers(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_disjoint_split_completion_zero_overlap_and_three_way_parity(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_mixed_pool_v4_curriculum_grounded_and_curated_sft() (+3 more)

### Community 24 - "NativeTinyModel"
Cohesion: 0.11
Nodes (18): NativeTinyModel, configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_run_matrix_stage_execution_batch2_and_frozen_decoder_invariance() (+10 more)

### Community 26 - "build_native_tiny_model"
Cohesion: 0.14
Nodes (7): build_native_tiny_model(), run_native_smoke(), fit_curated_sft(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, test_last_token_logits_match_full_head(), model()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "gsm8k.py"
Cohesion: 0.06
Nodes (26): digest(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k(), main() (+18 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.08
Nodes (9): main(), FastTrainingPipeline, run_fast_training(), test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+1 more)

### Community 30 - "benchmark_reset.py"
Cohesion: 0.15
Nodes (8): remote_reset_plan(), reset_local_benchmarks(), reset_state(), checked_prefix(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training(), main()

### Community 31 - "test_gsm8k.py"
Cohesion: 0.16
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 32 - "curation.py"
Cohesion: 0.11
Nodes (21): code_shingles(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_cross_source_duplicates_and_nextcoder_variants() (+13 more)

### Community 33 - "test_training_telemetry.py"
Cohesion: 0.15
Nodes (3): test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime()

### Community 34 - "run_external_benchmark"
Cohesion: 0.11
Nodes (18): format_crosscodelong_summary(), score_repo_niah_case(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort() (+10 more)

### Community 35 - "json"
Cohesion: 0.14
Nodes (16): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+8 more)

### Community 36 - "test_data_releases.py"
Cohesion: 0.15
Nodes (11): pool(), release(), test_archive_source_escape_rejected(), test_eval_public_and_gold_separate(), test_existing_local_pool_cannot_bypass_pinned_identity(), test_fetch_pinned_pool_roundtrip_and_reuse(), test_moving_revision_rejected_without_network(), test_publish_refuses_public_repo_before_upload() (+3 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.12
Nodes (15): is_permissive_license(), adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), _conversation_to_instruction_context_target(), _estimate_tokens() (+7 more)

### Community 38 - "json"
Cohesion: 0.10
Nodes (21): file_digest(), build_packed_cache(), open_packed_cache(), compare_predictions(), prompt_ids(), recover_remote_run(), _copy_atomic(), retry_delay() (+13 more)

### Community 39 - "test_recover_remote_run_ignores_dot_temp_and_heartbeat_files"
Cohesion: 0.15
Nodes (3): download(), test_recover_remote_run_ignores_dot_temp_and_heartbeat_files(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only()

### Community 40 - "telemetry.py"
Cohesion: 0.13
Nodes (8): format_duration(), format_stage_display(), _gpu_utilization_snapshot(), loss_chart(), render_dashboard(), LastSeenWidget, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_dashboard_stage_display_formatting()

### Community 41 - "curriculum.py"
Cohesion: 0.22
Nodes (7): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples()

### Community 42 - "run_batched_context_benchmark"
Cohesion: 0.07
Nodes (26): prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), _paired_budget_for(), run_context_benchmark(), activity(), build_effective_benchmark_spec(), effective_generation_budget() (+18 more)

### Community 43 - "build_repository_corpus"
Cohesion: 0.11
Nodes (15): ArchiveSnapshotStore, build_repository_corpus(), patch_license_evidence(), prepare_repository_corpus(), read_text_archive(), RepositoryPreparationLedger, archive(), test_archive_commit_budget_and_static_license() (+7 more)

### Community 44 - "evaluation/context.py"
Cohesion: 0.08
Nodes (31): reserved_output_token_budget(), _build_paired_eval_header_and_row(), build_repo_niah_case(), _context_already_includes_instruction(), _count_niah_tokens(), decoder_prompt(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens() (+23 more)

### Community 45 - "BifocalGemmaForCausalLM"
Cohesion: 0.10
Nodes (4): BifocalGemmaForCausalLM, test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_enable_memory_efficient_training_and_softcapping(), test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa()

### Community 47 - "test_data_remediation.py"
Cohesion: 0.19
Nodes (16): code_signature(), merge_lineage(), preference(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined() (+8 more)

### Community 48 - "test_eval_remediation.py"
Cohesion: 0.17
Nodes (7): _CharTokenizer, _DummyEvalModel, test_effective_spec_includes_overflow_fallback_policy(), test_external_runner_resume_and_signature_guard(), test_number_lines_false_preserves_genuine_digit_colon_source_lines(), test_overflow_fallback_recomputes_budget_and_records_metadata(), _write_mcq_benchmark()

### Community 49 - "SwiGLUDualGateBridge"
Cohesion: 0.06
Nodes (6): NormalizedAlignment, SwiGLUDualGateBridge, _resolve_gate_init_tanh(), TestBifocalGemmaArchitecture, TestGemma4EdgeTopology, test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight()

### Community 50 - "test_dashboard_dependency_state.py"
Cohesion: 0.15
Nodes (18): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+10 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 52 - "test_context_benchmark_resume.py"
Cohesion: 0.16
Nodes (5): FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch(), transient()

### Community 53 - "matrix.py"
Cohesion: 0.04
Nodes (43): configure_full_rank_cross_attention(), load_matrix_adapters_for_inference(), _milestone_checkpoint_names(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix() (+35 more)

### Community 54 - "test_v4_checkpoint_and_patches.py"
Cohesion: 0.12
Nodes (4): test_matrix_resume_accepts_token_identical_prompt_source_reformat(), test_publish_release_replace_current_deletes_audits(), test_v4_milestone_checkpoints_survive_pruning_and_hub_retention(), _write_ready_ckpt()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "argparse"
Cohesion: 0.15
Nodes (5): main(), main(), distribution(), main(), profile()

### Community 58 - "test_v4_architecture.py"
Cohesion: 0.08
Nodes (17): sdpa_attention(), clamped_triton_kernel_options(), critical_attention_beta(), format_query_injected_context(), safe_tokenize_context_task(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity() (+9 more)

### Community 59 - "compute_step_loss"
Cohesion: 0.07
Nodes (22): _resolve_example_diff_flag(), backward_window(), chunked_differential_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_example_loss_weights(), compute_step_loss() (+14 more)

### Community 60 - "hashlib"
Cohesion: 0.08
Nodes (35): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), export_pool(), curated_pool_digest() (+27 more)

### Community 61 - "context_pool"
Cohesion: 0.19
Nodes (10): split_connected_records(), completion_from_snapshot(), completions_from_snapshot(), context_pool(), log(), main(), mixed_pool(), prepare_core() (+2 more)

### Community 62 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 63 - "policy.py"
Cohesion: 0.17
Nodes (7): _build_group_schedule(), build_training_groups(), Float32Alignment, training_policy_table(), validate_training_policy(), classify_bifocalgemma_parameter(), test_build_training_groups_supports_unified_v4_all_families_and_legacy()

### Community 64 - "test_benchmark_upload.py"
Cohesion: 0.24
Nodes (3): test_external_activity_reader_does_not_need_running_kernel(), inspect(), main()

### Community 67 - "tiled_attention"
Cohesion: 0.12
Nodes (10): flex_kernel_options(), fused_attention(), tiled_attention(), TiledEncoderAttention, test_flex_backward_head256_sequence_above_small_sequence_branch(), test_flex_joint_gqa_matches_reference_on_cuda(), test_flex_structured_native_masks_across_sequence_lengths(), test_joint_gqa_output_and_gradients_at_native_head_widths() (+2 more)

### Community 68 - "test_context_tasks.py"
Cohesion: 0.10
Nodes (7): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_short_answer_scoring_rejects_unrequested_prose(), test_tokenize_context_task_preserves_target_indentation_and_newlines()

### Community 69 - ".attempt"
Cohesion: 0.29
Nodes (4): retry_delay(), test_keyboard_interrupt_is_not_swallowed(), test_retry_after_date_and_network_failures(), test_transient_lfs_pointer_400_is_retryable_while_other_400_is_fatal()

### Community 72 - "limited"
Cohesion: 0.29
Nodes (7): limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync()

### Community 74 - "data/context.py"
Cohesion: 0.48
Nodes (4): ContextWindow, render_window(), select_repository_context(), _terms()

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 78 - "fit_offline_sft"
Cohesion: 0.10
Nodes (17): cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint() (+9 more)

### Community 80 - "scorer_fingerprint"
Cohesion: 0.09
Nodes (20): completion_metrics(), _extract_full_python_code(), _extract_match_letter(), fallback_letter_regex(), _merge_full_function_into_file(), normalize_completion_prediction(), score_context_case(), _score_full_function_repair() (+12 more)

### Community 83 - "benchmark.py"
Cohesion: 0.09
Nodes (18): RepairPrediction, decode_text(), format_encoder_context(), batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), write_swebench_predictions(), generate_diagnostic_text() (+10 more)

### Community 85 - "test_v4_optimizer_schedule.py"
Cohesion: 0.09
Nodes (16): build_bifocalgemma_optimizer(), update_policy_learning_rates(), build_bifocalgemma_adamw(), build_partitioned_param_groups(), clip_optimizer_group_grads(), configure_matrix_stage(), _resolve_group_settings(), _synthetic_cohort() (+8 more)

### Community 87 - "training/diagnostics.py"
Cohesion: 0.17
Nodes (7): joint_visibility_tile(), _kv_slice_bounds(), mask_tile(), _TiledAttention, encoder_diagnostics(), sample_encoder_mass(), stratified_subset()

### Community 89 - "test_upload_recovery_receipts.py"
Cohesion: 0.09
Nodes (9): source_fingerprints(), WithoutDocstrings, retained_checkpoints(), test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), checkpoint(), test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), test_reusable_training_report_backfills_missing_final_stage_when_verified(), _build_run() (+1 more)

### Community 96 - "normalize_context_task"
Cohesion: 0.10
Nodes (21): normalize_context_task(), generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), editing_manifest(), repository_editing_record() (+13 more)

### Community 102 - "test_tier1_features.py"
Cohesion: 0.06
Nodes (17): pytest_configure(), reference_resilient_mcq_parser(), validate_empirical_results_schema(), validate_paper_manuscript(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation() (+9 more)

### Community 108 - "BifocalGemmaConfig"
Cohesion: 0.09
Nodes (13): BifocalGemmaConfig, _resolve_backbone_weights(), test_from_pretrained_device_map_and_missing_bridge_validation(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_generate_multi_eos_per_row_stopping_and_cache_parity(), test_generate_right_padded_matches_left_padded(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype() (+5 more)

### Community 109 - "WigglyDashboard"
Cohesion: 0.26
Nodes (7): phase_category(), WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 115 - "test_matrix_runtime_loader.py"
Cohesion: 0.21
Nodes (17): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+9 more)

### Community 116 - "make_experiment_spec"
Cohesion: 0.14
Nodes (13): make_experiment_spec(), _line_numbers(), _RecordingTokenizer, _SimpleTokenizer, test_eval_decoder_overflow_backoff_keeps_prefix_within_tight_budget(), test_grounded_line_lookup_stays_in_encoder_and_reconstruction_targets_exact_line(), test_gsm8k_paired_eval_matches_clean_prose_without_line_numbers_or_duplication(), test_long_context_dynamic_split_disjoint_and_zero_gold_leakage() (+5 more)

### Community 118 - "patch_cross_entropy"
Cohesion: 0.21
Nodes (5): ModelOutput, __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients()

### Community 122 - "RepairExample"
Cohesion: 0.13
Nodes (11): RepairExample, example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, tokenize_repair_example() (+3 more)

### Community 130 - "test_molab_remediation.py"
Cohesion: 0.11
Nodes (10): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_mcq_budget_floor_in_v3_benchmark_logic(), matrix_bundle(), main(), render() (+2 more)

### Community 137 - "fast_pipeline.py"
Cohesion: 0.06
Nodes (14): ensure_assets_on_disk(), load_asset_json(), CheckpointRef, ExperimentSpec, get_v2_experiment(), _pair(), _ref(), validate_experiment_matrix() (+6 more)

### Community 142 - "JSONLRepairDataset"
Cohesion: 0.09
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 172 - "pathlib"
Cohesion: 0.06
Nodes (12): BackboneProfile, EvaluationReport, BaselineScores, comparison_recipes(), TinyLossModel, test_notebook_stress_helpers_match_shared_library(), ExecutingScoringOptions, ScoringOptions (+4 more)

### Community 173 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 177 - "math"
Cohesion: 0.24
Nodes (4): resolve_training_cadence(), v4_group_defaults(), v4_optimizer_defaults(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated()

### Community 180 - "objectives.py"
Cohesion: 0.07
Nodes (13): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), CachedAttention, attention_tile() (+5 more)

### Community 183 - "configure_probe_training"
Cohesion: 0.11
Nodes (6): BifocalGemmaPreTrainedModel, export_matrix_pretrained_model(), configure_probe_training(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_export_matrix_pretrained_model(), test_export_matrix_pretrained_model_baseline()

### Community 185 - "notebook_launcher.py"
Cohesion: 0.29
Nodes (5): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 191 - "context_tasks.py"
Cohesion: 0.10
Nodes (29): _anchor_index_for_repair(), _anti_shortcut_budget(), _apply_decoder_chat_template(), _count_context_tokens(), DisjointContextSplit, dynamic_sliding_bifocal_split(), _DynamicBlock, _DynamicSection (+21 more)

### Community 197 - "test_molab_matrix_notebooks.py"
Cohesion: 0.08
Nodes (6): test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_matrix_baseline_single_stage_resolution(), make_identity(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_pointer_loss_gradient_checkpointing_and_trivial_filtering()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1061 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **90 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `stable_fingerprint()` connect `hashlib` to `preflight.py`, `normalize_context_task`, `json`, `test_data_releases.py`, `prepare_context_data.py`, `json`, `run_batched_context_benchmark`, `build_repository_corpus`, `evaluation/context.py`, `curation.py`, `test_data_remediation.py`, `re`, `matrix.py`, `test_matrix_schedule_runtime.py`, `context_pool`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `stable_fingerprint()` (e.g. with `test_generated_preparation_consumes_frozen_raw_repository_pool()` and `test_legacy_paired_manifest_remains_byte_identical()`) actually correct?**
  _`stable_fingerprint()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `preflight.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11330049261083744 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `json` to `preflight.py`, `test_benchmark_upload.py`, `run_external_benchmark`, `benchmark_upload.py`, `.attempt`, `prepare_context_data.py`, `atomic_json`, `run_batched_context_benchmark`, `evaluation/context.py`, `benchmark.py`, `matrix.py`, `test_upload_recovery_receipts.py`, `hashlib`, `context_pool`, `benchmark_reset.py`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `test_tier1_features.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11083743842364532 - nodes in this community are weakly interconnected._
- **Why does `RecordingStream` connect `RecordingStream` to `test_training_telemetry.py`, `WigglyDashboard`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._