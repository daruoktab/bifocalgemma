# Graph Report - bifocalgemma  (2026-11-21)

## Corpus Check
- 257 files · ~369,563 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2614 nodes · 7175 edges · 198 communities (98 shown, 100 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 877 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0d66654f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- wiggly_dashboard.py
- test_applied_python_patch_reports_syntax_independently
- benchmark_upload.py
- build_native_tiny_model
- re
- hub.py
- test_notebook_dependencies.py
- test_memory_efficient_training.py
- FastTrainingPipeline
- test_matrix_runtime_loader.py
- curation.py
- schedule.py
- build_and_publish_v4_dataset.py
- test_dashboard_dependency_state.py
- stages.py
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- hashlib
- test_molab_matrix_notebooks.py
- context_tasks.py
- patch_cross_entropy
- adversarial_r3_probe.py
- memory_efficient.py
- test_v4_data_disjoint.py
- curriculum.py
- full_pool.py
- external_runner.py
- examples
- contracts.py
- prepare_context_data.py
- synchronize_matrix
- test_gsm8k.py
- TrainingDashboard
- SwiGLUDualGateBridge
- scorer_fingerprint
- load_local_native_model
- evaluation/context.py
- test_molab_remediation.py
- BifocalGemmaForCausalLM
- sys
- bifocalgemma/__init__.py
- test_scenario_end_to_end_mock_evaluation_pipeline
- repository_study.py
- RepositorySnapshot
- configure_matrix_stage
- CodeOverlapIndex
- test_matrix_schedule_runtime.py
- remediation.py
- BifocalGemmaConfig
- objectives.py
- hashlib
- JSONLRepairDataset
- atomic_json
- TestMemoryProbe
- math
- test_training_telemetry.py
- NativeTinyModel
- test_context_tasks.py
- run_length_grid
- repository_corpus.py
- subprocess
- test_construction_blocks.py
- training/offline_sft.py
- test_editing_manifest.py
- test_shared_schedule.py
- run_matrix_stage
- build_repository_corpus
- JSONLRepairDataset
- pathlib
- test_upload_recovery_receipts.py
- TestTokenRoleAndDataIdentity
- bifocalgemma
- releases.py
- run_matrix_training
- RepairCollator
- test_data_remediation.py
- normalize_context_task
- .configure_stage1
- pytest
- test_v4_architecture.py
- BifocalGemmaForCausalLM
- adapt_v4_row
- FullCrossCachedAttention
- notebook_launcher.py
- .snapshot
- patch_matrix_tokenization
- test_fast_pipeline.py
- __main__.py
- packed_tokens.py
- json
- WigglyDashboard
- .from_pretrained_backbones
- data/offline_sft.py
- TokenizedRepairExample
- mixed_pool_v4
- stable_fingerprint
- benchmark.py
- memory_probe.py
- test_evaluation.py
- native_config
- test_benchmark_prompt_budget.py
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- test_compute_per_row_cross_query_vectorized_parity
- NativeMergedAttention
- dependency_state
- FullCrossCachedAttention
- _ProportionalTokenizer
- patch_cross_entropy
- audit_dataset_release.py
- source_fingerprints
- profile_repository_lengths.py
- ScriptedModel
- TestCheckpointTransfer
- CachedAttention
- test_generate_multi_eos_per_row_stopping_and_cache_parity
- audit_context_tokens.py
- normalize_math_question
- project_root

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 67 edges
2. `atomic_json()` - 61 edges
3. `run_matrix_training()` - 51 edges
4. `load_verified_tokenizer()` - 51 edges
5. `BifocalGemmaForCausalLM` - 50 edges
6. `RepositorySnapshot` - 45 edges
7. `run_matrix_benchmark()` - 41 edges
8. `TokenizedRepairExample` - 40 edges
9. `FastTrainingPipeline` - 38 edges
10. `fit_offline_sft()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `test_v4_config_constants_and_external_registry()` --calls--> `infer_benchmark_weight_tier()`  [INFERRED]
  tests/test_v4_eval_parity.py → src/bifocalgemma/evaluation/benchmark.py
- `test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()` --calls--> `resolve_optimal_eval_batch_size()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/evaluation/benchmark.py
- `test_prepare_sampled_benchmark_collection_deterministic_100_cases()` --calls--> `prepare_sampled_benchmark_collection()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/evaluation/benchmark.py
- `fetch()` --calls--> `repository_task_from_row()`  [INFERRED]
  tools/prepare_context_data.py → src/bifocalgemma/data/repository.py
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (198 total, 100 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.19
Nodes (10): build_policy_optimizer(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution() (+2 more)

### Community 1 - "wiggly_dashboard.py"
Cohesion: 0.15
Nodes (4): format_stage_display(), LastSeenWidget, TrainingLossLog, test_dashboard_stage_display_formatting()

### Community 3 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (18): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush() (+10 more)

### Community 6 - "re"
Cohesion: 0.13
Nodes (18): _add_missing_git_headers(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines(), _is_file_header_pair() (+10 more)

### Community 7 - "hub.py"
Cohesion: 0.11
Nodes (11): remote_reset_plan(), reset_state(), checked_prefix(), recover_remote_run(), _copy_atomic(), RemoteHeartbeat, retry_delay(), _synchronize_once() (+3 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.11
Nodes (17): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+9 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.20
Nodes (12): run_training_memory_preflight(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 11 - "FastTrainingPipeline"
Cohesion: 0.16
Nodes (10): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_training_pipeline_features_and_export(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_training_config_step_reconciliation() (+2 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.27
Nodes (13): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_publication_only_adds_verified_immutable_runtime() (+5 more)

### Community 14 - "curation.py"
Cohesion: 0.14
Nodes (13): code_signature(), code_tokens(), curate_paired_sources(), nextcoder_row(), count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records() (+5 more)

### Community 15 - "schedule.py"
Cohesion: 0.12
Nodes (15): _build_schedule_for_epochs(), _validate_num_epochs(), default_training_profiles(), _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer() (+7 more)

### Community 16 - "build_and_publish_v4_dataset.py"
Cohesion: 0.17
Nodes (14): build_and_verify_mixed_pool(), build_and_verify_release(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers(), _log(), main(), _measure_exact_tokens() (+6 more)

### Community 17 - "test_dashboard_dependency_state.py"
Cohesion: 0.14
Nodes (15): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+7 more)

### Community 18 - "stages.py"
Cohesion: 0.15
Nodes (7): ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.10
Nodes (7): MockCausalLM, MockQwenTokenizer, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(), test_v4_config_constants_and_external_registry(), _write_repair_fixture()

### Community 20 - "eval_code.py"
Cohesion: 0.15
Nodes (6): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (25): completion_metrics(), extract_mcq_letter(), _sequence_and_identifier_metrics(), strict_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards() (+17 more)

### Community 23 - "hashlib"
Cohesion: 0.20
Nodes (12): _clean_extracted_number(), digest(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+4 more)

### Community 24 - "test_molab_matrix_notebooks.py"
Cohesion: 0.12
Nodes (20): make_experiment_spec(), configure_full_rank_cross_attention(), configure_memory_efficient_training(), is_cross_bridge_parameter(), configure_policy_model(), unified_recipe(), v4_group_defaults(), fixture_model() (+12 more)

### Community 25 - "context_tasks.py"
Cohesion: 0.15
Nodes (19): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), _parse_file_sections(), _render_entries(), _resolve_eot_token_id() (+11 more)

### Community 26 - "patch_cross_entropy"
Cohesion: 0.18
Nodes (3): TestOfflineSFT, TinyLossModel, test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "memory_efficient.py"
Cohesion: 0.08
Nodes (19): flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), mask_tile(), sdpa_attention(), sdpa_cross_attention() (+11 more)

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.17
Nodes (9): _CharTokenizer, _load_prepare_module(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_mixed_pool_v4_curriculum_grounded_and_curated_sft(), test_multifunction_completion_extraction_removes_single_return_cap(), test_v4_quarantine_licenses_dedup_and_nextcoder_rejection() (+1 more)

### Community 31 - "curriculum.py"
Cohesion: 0.19
Nodes (9): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length() (+1 more)

### Community 32 - "full_pool.py"
Cohesion: 0.22
Nodes (9): read_arrow_rows(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main(), progress() (+1 more)

### Community 34 - "external_runner.py"
Cohesion: 0.09
Nodes (22): format_crosscodelong_summary(), _resolve_repair_mode(), score_context_case(), _score_full_function_repair(), score_repo_niah_case(), _single_repair_target_file(), _build_reponiah_cohort(), _collect_int_ids() (+14 more)

### Community 35 - "examples"
Cohesion: 0.19
Nodes (5): make_identity(), examples(), TestTrainingPolicy, test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 36 - "contracts.py"
Cohesion: 0.06
Nodes (23): DeveloperAgent, CheckpointRef, EvaluationReport, RepairExample, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter (+15 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.16
Nodes (10): split_connected_records(), completion_from_snapshot(), completions_from_snapshot(), context_pool(), log(), main(), mixed_pool(), prepare_core() (+2 more)

### Community 38 - "synchronize_matrix"
Cohesion: 0.19
Nodes (9): load_matrix_adapters_for_inference(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix(), snapshot(), snapshot(), validate_checkpoint_files() (+1 more)

### Community 39 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 40 - "TrainingDashboard"
Cohesion: 0.22
Nodes (4): render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 41 - "SwiGLUDualGateBridge"
Cohesion: 0.06
Nodes (7): SwiGLUDualGateBridge, _resolve_gate_init_tanh(), TestBifocalGemmaArchitecture, TestGemma4EdgeTopology, test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads(), test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight(), test_swiglu_bridge_non_redundant_single_post_gain()

### Community 42 - "scorer_fingerprint"
Cohesion: 0.07
Nodes (24): _extract_match_letter(), fallback_letter_regex(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer(), score_gsm8k(), build_effective_benchmark_spec(), effective_generation_budget() (+16 more)

### Community 43 - "load_local_native_model"
Cohesion: 0.21
Nodes (5): load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), test_inference_v2_adapter_roundtrip_and_baseline_group(), test_local_loader_preflights_and_preserves_shared_words()

### Community 44 - "evaluation/context.py"
Cohesion: 0.07
Nodes (25): build_repo_niah_case(), _context_already_includes_instruction(), _count_niah_tokens(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens(), encoder_context(), _extract_full_python_code(), generate_repo_niah_haystack() (+17 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 48 - "sys"
Cohesion: 0.17
Nodes (9): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), format_code(), main(), matrix_bundle(), render_notebook(), main(), render(), source_bundle() (+1 more)

### Community 49 - "bifocalgemma/__init__.py"
Cohesion: 0.12
Nodes (4): BifocalGemmaPreTrainedModel, export_matrix_pretrained_model(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_export_matrix_pretrained_model_baseline()

### Community 51 - "repository_study.py"
Cohesion: 0.25
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+10 more)

### Community 52 - "RepositorySnapshot"
Cohesion: 0.09
Nodes (29): RepositoryFile, RepositorySnapshot, canonical_snapshot_hash(), apply_patch_to_snapshot(), validate_code_eval_snapshots(), v4_repository_task_from_row(), example_from_record(), prepare_repository_record() (+21 more)

### Community 53 - "configure_matrix_stage"
Cohesion: 0.10
Nodes (16): training_policy_table(), update_policy_learning_rates(), build_bifocalgemma_adamw(), build_partitioned_param_groups(), configure_matrix_stage(), _resolve_group_settings(), _synthetic_cohort(), test_build_training_groups_supports_unified_v4_all_families_and_legacy() (+8 more)

### Community 54 - "CodeOverlapIndex"
Cohesion: 0.15
Nodes (17): code_shingles(), CodeOverlapIndex, CodeOverlapMatch, export_pool(), raw(), test_disk_index_matches_memory_and_syntax_repairs_are_admitted(), test_frozen_pool_verifies_checksums_before_use(), test_full_overlap_index_skips_impossible_blob_reads() (+9 more)

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "remediation.py"
Cohesion: 0.14
Nodes (11): token_profile(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), preference(), retokenize_records() (+3 more)

### Community 58 - "BifocalGemmaConfig"
Cohesion: 0.11
Nodes (13): BifocalGemmaConfig, test_from_pretrained_raises_file_not_found_when_weights_missing(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip() (+5 more)

### Community 59 - "objectives.py"
Cohesion: 0.07
Nodes (23): _resolve_example_diff_flag(), backward_window(), build_bifocalgemma_optimizer(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights() (+15 more)

### Community 60 - "hashlib"
Cohesion: 0.17
Nodes (13): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), ContextWindow, render_window(), select_repository_context(), _terms() (+5 more)

### Community 62 - "atomic_json"
Cohesion: 0.14
Nodes (9): activity(), activity(), reset_local_benchmarks(), atomic_json(), StageLedger, test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training() (+1 more)

### Community 64 - "math"
Cohesion: 0.07
Nodes (15): is_zero_weight_decay_bridge_parameter(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), _build_group_schedule(), build_training_groups(), Float32Alignment, validate_training_policy() (+7 more)

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.13
Nodes (5): _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime(), RecordingStream

### Community 67 - "NativeTinyModel"
Cohesion: 0.22
Nodes (4): NativeTinyModel, test_cross_scale_frozen_by_default_and_configurable(), test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), test_gate_init_tanh_resolution_dynamic_and_v4_fallback()

### Community 68 - "test_context_tasks.py"
Cohesion: 0.09
Nodes (9): repair_preimage_coverage(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose() (+1 more)

### Community 69 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 72 - "repository_corpus.py"
Cohesion: 0.16
Nodes (12): context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read() (+4 more)

### Community 74 - "subprocess"
Cohesion: 0.16
Nodes (8): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract, TestTokenizerVerification

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "training/offline_sft.py"
Cohesion: 0.17
Nodes (14): encoder_diagnostics(), stratified_subset(), cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage() (+6 more)

### Community 79 - "test_editing_manifest.py"
Cohesion: 0.24
Nodes (9): editing_manifest(), repository_editing_record(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation() (+1 more)

### Community 80 - "test_shared_schedule.py"
Cohesion: 0.28
Nodes (12): rows(), schedule(), test_all_originals_once_with_disjoint_stage_windows(), test_atomic_manifest_reuse_and_fail_closed_mismatch(), test_deterministic_model_independent_schedule_and_mapping(), test_each_semantic_mutation_changes_signature(), test_empty_required_fields_rejected(), test_foundation_context_and_target_patch_are_bound() (+4 more)

### Community 82 - "run_matrix_stage"
Cohesion: 0.18
Nodes (9): prune_matrix_checkpoints(), run_matrix_stage(), progress(), fixture_data(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_partial_backward_oom_replays_entire_window_without_double_gradient(), test_stage1_to_stage2_continuous_warmup_offset_and_per_group_grad_norms() (+1 more)

### Community 83 - "build_repository_corpus"
Cohesion: 0.26
Nodes (4): build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 85 - "pathlib"
Cohesion: 0.05
Nodes (13): ensure_assets_on_disk(), load_asset_json(), BackboneProfile, BaselineScores, cached_logits(), ProbeLoRALinear, comparison_recipes(), v4_optimizer_defaults() (+5 more)

### Community 87 - "test_upload_recovery_receipts.py"
Cohesion: 0.09
Nodes (5): test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), download(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only(), test_reusable_training_report_backfills_missing_final_stage_when_verified(), test_standalone_export_requires_matching_remote_contents()

### Community 91 - "releases.py"
Cohesion: 0.10
Nodes (24): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+16 more)

### Community 92 - "run_matrix_training"
Cohesion: 0.10
Nodes (16): download_model(), _backfill_final_stage(), _final_stage_info(), _latest_checkpoint_matches(), _local_standalone_export_valid(), _resolve_profile_settings(), resolve_training_contract(), _reusable_training_report() (+8 more)

### Community 93 - "RepairCollator"
Cohesion: 0.15
Nodes (7): Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), TestDataAndLoss, test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 95 - "test_data_remediation.py"
Cohesion: 0.24
Nodes (13): merge_lineage(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage() (+5 more)

### Community 96 - "normalize_context_task"
Cohesion: 0.22
Nodes (10): normalize_context_task(), generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), parent(), test_evaluation_and_unlicensed_sources_rejected() (+2 more)

### Community 101 - ".configure_stage1"
Cohesion: 0.20
Nodes (3): test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule()

### Community 102 - "pytest"
Cohesion: 0.04
Nodes (28): authoritative_empirical_results(), pytest_configure(), reference_resilient_mcq_parser(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), validate_paper_manuscript() (+20 more)

### Community 103 - "test_v4_architecture.py"
Cohesion: 0.15
Nodes (5): clamped_triton_kernel_options(), critical_attention_beta(), ensure_hardware_memory_guards(), test_v3_critical_attention_beta_and_triton_tile_clamping(), test_compute_per_row_cross_query_on_cuda_stays_on_device()

### Community 104 - "BifocalGemmaForCausalLM"
Cohesion: 0.13
Nodes (3): BifocalGemmaForCausalLM, spy_sdpa(), test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa()

### Community 105 - "adapt_v4_row"
Cohesion: 0.24
Nodes (7): is_permissive_license(), adapt_v4_row(), _conversation_to_instruction_context_target(), _estimate_tokens(), _extract_repo_from_v4_row(), _extract_row_license(), _resolve_v4_license_evidence()

### Community 106 - "FullCrossCachedAttention"
Cohesion: 0.20
Nodes (4): FullCrossCachedAttention, test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence(), test_full_cross_cached_attention_sdpa_guard_fallback_and_non_left_padded_step_valid(), _run_cached()

### Community 108 - "notebook_launcher.py"
Cohesion: 0.30
Nodes (7): test_matrix_launcher_upload_and_repeated_setup_with_preloaded_marimo(), test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.15
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 114 - "__main__.py"
Cohesion: 0.13
Nodes (9): ExperimentSpec, get_v2_experiment(), _pair(), _ref(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke() (+1 more)

### Community 115 - "packed_tokens.py"
Cohesion: 0.14
Nodes (7): file_digest(), build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, test_packed_cache_can_be_built_without_tensor_runtime(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 116 - "json"
Cohesion: 0.13
Nodes (11): write_view(), compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), test_view_record_json_is_lossless_and_gsm_shots_are_explicit(), main(), main(), inspect(), main() (+3 more)

### Community 120 - "WigglyDashboard"
Cohesion: 0.26
Nodes (7): phase_category(), WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 122 - "data/offline_sft.py"
Cohesion: 0.11
Nodes (14): curated_pool_digest(), git_text_lines(), load_curated_pool(), prepare_editing_records(), split_editing_records(), validate_context_license_record(), prepare_curated_training(), tokenize() (+6 more)

### Community 123 - "TokenizedRepairExample"
Cohesion: 0.22
Nodes (6): validate_curated_pool_limits(), fit_curated_sft(), test_complete_epoch_training_budget_all_families(), test_curated_resume_rejects_runtime_drift(), test_frozen_budget_matches_all_six_controls(), test_invalid_requested_pool_budget_is_rejected()

### Community 124 - "mixed_pool_v4"
Cohesion: 0.27
Nodes (6): _all_code_fragment_signatures(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), _is_quarantined_repo(), mixed_pool_v4(), _row_repo_candidates()

### Community 130 - "stable_fingerprint"
Cohesion: 0.19
Nodes (11): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), benchmark_directory_name() (+3 more)

### Community 137 - "benchmark.py"
Cohesion: 0.10
Nodes (18): decode_text(), batched_greedy_decode(), infer_benchmark_weight_tier(), prepare_sampled_benchmark_collection(), resolve_optimal_eval_batch_size(), run_batched_context_benchmark(), decoder_prompt(), greedy_decode() (+10 more)

### Community 142 - "memory_probe.py"
Cohesion: 0.33
Nodes (5): gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes()

### Community 146 - "test_evaluation.py"
Cohesion: 0.20
Nodes (3): test_crosscodelong_normalize_completion_prediction_strips_fences_and_suffix_overlap(), test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity(), test_prompt_ids_instance_id_isolation_and_gold_rejection()

### Community 171 - "native_config"
Cohesion: 0.25
Nodes (3): _check_transformers_version(), native_config(), NormalizedAlignment

### Community 172 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 173 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 174 - "test_compute_per_row_cross_query_vectorized_parity"
Cohesion: 0.32
Nodes (4): test_compute_per_row_cross_query_vectorized_parity(), test_decoupled_upper_full_attention_cross_kv_topology(), test_sdpa_attention_structured_masks_match_tiled_eager(), test_sdpa_cross_attention_matches_eager_and_handles_all_masked_rows()

### Community 180 - "FullCrossCachedAttention"
Cohesion: 0.11
Nodes (9): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossTiledAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init() (+1 more)

### Community 185 - "patch_cross_entropy"
Cohesion: 0.33
Nodes (3): __getattr__(), patch_cross_entropy(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients()

### Community 186 - "audit_dataset_release.py"
Cohesion: 0.52
Nodes (4): audit(), emit(), main(), validate_all_patches()

### Community 187 - "source_fingerprints"
Cohesion: 0.33
Nodes (3): source_fingerprints(), WithoutDocstrings, test_f5_training_semantics_change_rejects_legacy_approval()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1011 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **100 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `test_generate_multi_eos_per_row_stopping_and_cache_parity`, `FullCrossCachedAttention`, `FastTrainingPipeline`, `test_compute_per_row_cross_query_vectorized_parity`, `BifocalGemmaForCausalLM`, `bifocalgemma/__init__.py`, `run_matrix_stage`, `FullCrossCachedAttention`, `pathlib`, `.from_pretrained_backbones`, `BifocalGemmaConfig`, `memory_efficient.py`, `RepairCollator`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `benchmark_upload.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0707070707070707 - nodes in this community are weakly interconnected._
- **Why does `load_verified_tokenizer()` connect `contracts.py` to `stable_fingerprint`, `hub.py`, `benchmark.py`, `schedule.py`, `build_and_publish_v4_dataset.py`, `hashlib`, `full_pool.py`, `prepare_context_data.py`, `test_gsm8k.py`, `evaluation/context.py`, `repository_study.py`, `remediation.py`, `audit_dataset_release.py`, `hashlib`, `audit_context_tokens.py`, `repository_corpus.py`, `subprocess`, `build_repository_corpus`, `JSONLRepairDataset`, `pathlib`, `run_matrix_training`, `json`, `data/offline_sft.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Should `re` be split into smaller, more focused modules?**
  _Cohesion score 0.12878787878787878 - nodes in this community are weakly interconnected._
- **Should `hub.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1076923076923077 - nodes in this community are weakly interconnected._