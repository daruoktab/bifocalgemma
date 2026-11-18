# Graph Report - bifocalgemma  (2026-11-18)

## Corpus Check
- 249 files · ~329,553 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2289 nodes · 6197 edges · 179 communities (79 shown, 100 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 730 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c07224a4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- TrainingDashboard
- test_editing_manifest.py
- modeling_bifocalgemma.py
- build_native_tiny_model
- re
- hub.py
- test_notebook_dependencies.py
- FullCrossCachedAttention
- FastTrainingPipeline
- test_matrix_runtime_loader.py
- curation.py
- schedule.py
- benchmark_upload.py
- prepare_context_data.py
- json
- test_gsm8k.py
- eval_code.py
- test_tier5_adversarial.py
- hashlib
- load_curated_pool
- test_dashboard_dependency_state.py
- patch_cross_entropy
- adversarial_r3_probe.py
- torch
- random
- full_pool.py
- fit_offline_sft
- examples
- FeatureNotReadyError
- prepare_context_data.py
- synchronize_matrix
- collections_abc
- test_molab_matrix_notebooks.py
- test_memory_efficient_training.py
- protocol.py
- load_local_native_model
- evaluation/context.py
- test_molab_remediation.py
- BifocalGemmaConfig
- sys
- bifocalgemma/__init__.py
- notebook_launcher.py
- repository_study.py
- RepositorySnapshot
- run_matrix_stage
- fixture
- test_matrix_schedule_runtime.py
- Seq2SeqBatch
- SwiGLUDualGateBridge
- compute_step_loss
- test_tier4_scenarios.py
- JSONLRepairDataset
- remote_reset_plan
- memory_probe.py
- copy
- test_training_telemetry.py
- TestBifocalGemmaArchitecture
- test_context_tasks.py
- pathlib
- repository_corpus.py
- tempfile
- test_construction_blocks.py
- NativeMergedAttention
- NativeTinyModel
- WigglyDashboard
- wiggly_dashboard.py
- json
- JSONLRepairDataset
- test_generate_multi_eos_per_row_stopping_and_cache_parity
- test_upload_recovery_receipts.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_benchmark_prompt_budget.py
- run_matrix_training
- RepairCollator
- remediation.py
- normalize_context_task
- stable_fingerprint
- test_tier3_combinations.py
- matrix.py
- patch_cross_entropy
- .from_pretrained_backbones
- TestMemoryProbe
- .snapshot
- profile_repository_lengths.py
- test_fast_pipeline.py
- ExperimentSpec
- PackedTokenizedDataset
- test_optimized_native_resume_matches_full_and_rejects_different_recipe
- repository_sft.py
- load_verified_tokenizer
- test_tier1_features.py
- tests_e2e/conftest.py
- audit_dataset_release.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- RecordingStream
- TestTokenRoleAndDataIdentity
- .update_widgets

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 65 edges
2. `atomic_json()` - 52 edges
3. `run_matrix_training()` - 50 edges
4. `load_verified_tokenizer()` - 50 edges
5. `RepositorySnapshot` - 41 edges
6. `BifocalGemmaForCausalLM` - 41 edges
7. `run_matrix_benchmark()` - 40 edges
8. `TokenizedRepairExample` - 39 edges
9. `FastTrainingPipeline` - 37 edges
10. `fit_offline_sft()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()` --calls--> `resolve_optimal_eval_batch_size()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/evaluation/benchmark.py
- `test_fast_training_config_step_reconciliation()` --calls--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry.py → src/bifocalgemma/training/source_identity.py
- `test_context_license_record_covers_all_visible_paths()` --calls--> `validate_context_license_record()`  [INFERRED]
  tests/test_repository_corpus.py → src/bifocalgemma/data/offline_sft.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (179 total, 100 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.19
Nodes (10): build_training_groups(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution() (+2 more)

### Community 1 - "TrainingDashboard"
Cohesion: 0.22
Nodes (4): render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 2 - "test_editing_manifest.py"
Cohesion: 0.26
Nodes (9): editing_manifest(), repository_editing_record(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation() (+1 more)

### Community 3 - "modeling_bifocalgemma.py"
Cohesion: 0.06
Nodes (8): ensure_assets_on_disk(), load_asset_json(), _check_transformers_version(), native_config(), cached_logits(), CachedAttention, InferenceCache, download_model()

### Community 6 - "re"
Cohesion: 0.11
Nodes (24): _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines() (+16 more)

### Community 7 - "hub.py"
Cohesion: 0.14
Nodes (10): benchmark_directory_name(), checked_prefix(), recover_remote_run(), _copy_atomic(), RemoteHeartbeat, retry_delay(), _synchronize_once(), synchronize_run() (+2 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.12
Nodes (17): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+9 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.23
Nodes (3): FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "FastTrainingPipeline"
Cohesion: 0.13
Nodes (13): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule(), test_fast_training_config_and_pipeline_baseline_single_stage() (+5 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.21
Nodes (17): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+9 more)

### Community 14 - "curation.py"
Cohesion: 0.06
Nodes (39): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits(), count_syntax_warnings() (+31 more)

### Community 15 - "schedule.py"
Cohesion: 0.10
Nodes (25): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+17 more)

### Community 16 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (17): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed() (+9 more)

### Community 17 - "prepare_context_data.py"
Cohesion: 0.21
Nodes (8): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), refresh_patch_targets(), code_fragment_signatures()

### Community 18 - "json"
Cohesion: 0.10
Nodes (17): compare_predictions(), source_fingerprints(), WithoutDocstrings, test_f5_training_semantics_change_rejects_legacy_approval(), test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), audit_pool() (+9 more)

### Community 19 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 20 - "eval_code.py"
Cohesion: 0.17
Nodes (5): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): extract_mcq_letter(), strict_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization() (+15 more)

### Community 23 - "hashlib"
Cohesion: 0.14
Nodes (16): decode_text(), _clean_extracted_number(), digest(), extract_gsm8k_answer(), extract_resilient_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k() (+8 more)

### Community 24 - "load_curated_pool"
Cohesion: 0.22
Nodes (4): curated_pool_digest(), load_curated_pool(), build_pool(), main()

### Community 25 - "test_dashboard_dependency_state.py"
Cohesion: 0.11
Nodes (16): dependency_state(), test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready() (+8 more)

### Community 26 - "patch_cross_entropy"
Cohesion: 0.20
Nodes (3): TestOfflineSFT, TinyLossModel, test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "torch"
Cohesion: 0.11
Nodes (15): flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention (+7 more)

### Community 31 - "random"
Cohesion: 0.24
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 32 - "full_pool.py"
Cohesion: 0.14
Nodes (14): file_digest(), read_arrow_rows(), split_connected_records(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+6 more)

### Community 34 - "fit_offline_sft"
Cohesion: 0.11
Nodes (22): prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), prompt_ids(), run_context_benchmark(), activity(), encoder_diagnostics(), stratified_subset() (+14 more)

### Community 35 - "examples"
Cohesion: 0.27
Nodes (3): make_identity(), examples(), TestTrainingPolicy

### Community 36 - "FeatureNotReadyError"
Cohesion: 0.09
Nodes (15): DeveloperAgent, EvaluationReport, RepairExample, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter (+7 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.19
Nodes (10): export_pool(), ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), completion_from_snapshot(), context_pool(), log(), main(), mixed_pool() (+2 more)

### Community 38 - "synchronize_matrix"
Cohesion: 0.15
Nodes (10): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix(), snapshot(), snapshot() (+2 more)

### Community 39 - "collections_abc"
Cohesion: 0.18
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 40 - "test_molab_matrix_notebooks.py"
Cohesion: 0.11
Nodes (20): make_experiment_spec(), configure_memory_efficient_training(), configure_policy_model(), validate_training_policy(), unified_recipe(), fit_repository_sft(), configure_matrix_stage(), test_cached_uncached_and_batched_greedy_decode_equivalence() (+12 more)

### Community 41 - "test_memory_efficient_training.py"
Cohesion: 0.19
Nodes (13): run_training_memory_preflight(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+5 more)

### Community 42 - "protocol.py"
Cohesion: 0.08
Nodes (19): _extract_match_letter(), fallback_letter_regex(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer() (+11 more)

### Community 43 - "load_local_native_model"
Cohesion: 0.17
Nodes (6): load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), TestCheckpointTransfer, test_inference_v2_adapter_roundtrip_and_baseline_group(), test_local_loader_preflights_and_preserves_shared_words()

### Community 44 - "evaluation/context.py"
Cohesion: 0.10
Nodes (18): build_repo_niah_case(), completion_metrics(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), normalize_completion_prediction(), run_repo_niah_eval() (+10 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 47 - "BifocalGemmaConfig"
Cohesion: 0.10
Nodes (12): BifocalGemmaConfig, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation() (+4 more)

### Community 48 - "sys"
Cohesion: 0.19
Nodes (5): matrix_bundle(), main(), render(), source_bundle(), source_bundle_bytes()

### Community 50 - "notebook_launcher.py"
Cohesion: 0.35
Nodes (6): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 51 - "repository_study.py"
Cohesion: 0.12
Nodes (31): get_experiment(), canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), normalized_code(), prepare_crosscode_tasks() (+23 more)

### Community 52 - "RepositorySnapshot"
Cohesion: 0.07
Nodes (21): RepositoryFile, RepositorySnapshot, CodeOverlapMatch, test_dependency_completion_removes_target_and_retains_real_helper(), TestRealOfflineTokenizers, test_missing_old_file_is_still_rejected(), test_modify_add_delete_preserves_unrelated_unicode_bytes(), CharacterTokenizer (+13 more)

### Community 53 - "run_matrix_stage"
Cohesion: 0.16
Nodes (10): run_matrix_stage(), update_policy_learning_rates(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), test_partial_backward_oom_replays_entire_window_without_double_gradient() (+2 more)

### Community 54 - "fixture"
Cohesion: 0.15
Nodes (6): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.12
Nodes (4): download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 58 - "SwiGLUDualGateBridge"
Cohesion: 0.16
Nodes (3): NormalizedAlignment, SwiGLUDualGateBridge, test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()

### Community 59 - "compute_step_loss"
Cohesion: 0.06
Nodes (17): backward_window(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), compute_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids() (+9 more)

### Community 60 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 62 - "remote_reset_plan"
Cohesion: 0.08
Nodes (11): remote_reset_plan(), reset_local_benchmarks(), reset_state(), atomic_json(), StageLedger, test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training() (+3 more)

### Community 63 - "memory_probe.py"
Cohesion: 0.13
Nodes (14): export_matrix_pretrained_model(), configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), run_length_grid(), tensor_bytes() (+6 more)

### Community 64 - "copy"
Cohesion: 0.14
Nodes (7): default_training_profiles(), Float32Alignment, comparison_recipes(), resolve_training_cadence(), limit_threads(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_fp32_alignment_on_bf16_backbone()

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.20
Nodes (3): test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime()

### Community 68 - "test_context_tasks.py"
Cohesion: 0.06
Nodes (14): repair_preimage_coverage(), format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment() (+6 more)

### Community 69 - "pathlib"
Cohesion: 0.06
Nodes (6): BackboneProfile, CheckpointRef, _pair(), _ref(), BaselineScores, test_notebook_stress_helpers_match_shared_library()

### Community 72 - "repository_corpus.py"
Cohesion: 0.17
Nodes (11): context_license_evidence(), patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch() (+3 more)

### Community 74 - "tempfile"
Cohesion: 0.13
Nodes (8): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), fixture_records(), TestSWESmithContract

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 80 - "WigglyDashboard"
Cohesion: 0.47
Nodes (6): WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 83 - "json"
Cohesion: 0.09
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 87 - "test_upload_recovery_receipts.py"
Cohesion: 0.11
Nodes (3): test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), download(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only()

### Community 91 - "test_benchmark_prompt_budget.py"
Cohesion: 0.09
Nodes (8): _context_already_includes_instruction(), decoder_prompt(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer, test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity(), test_prompt_ids_instance_id_isolation_and_gold_rejection()

### Community 92 - "run_matrix_training"
Cohesion: 0.11
Nodes (15): _backfill_final_stage(), _final_stage_info(), _latest_checkpoint_matches(), _local_standalone_export_valid(), resolve_training_contract(), _reusable_training_report(), run_matrix_benchmark(), run_matrix_training() (+7 more)

### Community 93 - "RepairCollator"
Cohesion: 0.17
Nodes (7): TokenizedRepairExample, RepairCollator, __getattr__(), synthetic_stress_target(), TestDataAndLoss, test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes(), test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 95 - "remediation.py"
Cohesion: 0.13
Nodes (22): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), remediate_records() (+14 more)

### Community 96 - "normalize_context_task"
Cohesion: 0.20
Nodes (10): normalize_context_task(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record(), parent(), test_evaluation_and_unlicensed_sources_rejected() (+2 more)

### Community 101 - "stable_fingerprint"
Cohesion: 0.30
Nodes (9): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+1 more)

### Community 102 - "test_tier3_combinations.py"
Cohesion: 0.19
Nodes (4): validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 103 - "matrix.py"
Cohesion: 0.09
Nodes (11): _compute_per_row_cross_query(), configure_full_rank_cross_attention(), build_bifocalgemma_optimizer(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), clamped_triton_kernel_options(), critical_attention_beta(), ensure_hardware_memory_guards() (+3 more)

### Community 105 - ".from_pretrained_backbones"
Cohesion: 0.20
Nodes (4): build_native_model(), _resolve_backbone_weights(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 110 - "profile_repository_lengths.py"
Cohesion: 0.47
Nodes (3): distribution(), main(), profile()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.15
Nodes (4): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_jsonl_dataset_query_injection_into_encoder()

### Community 114 - "ExperimentSpec"
Cohesion: 0.16
Nodes (7): ExperimentSpec, get_v2_experiment(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 122 - "repository_sft.py"
Cohesion: 0.16
Nodes (10): _resolve_eot_token_id(), task_prompt(), tokenize_context_task(), prepare_editing_records(), split_editing_records(), prepare_curated_training(), tokenize(), prepare_repository_training() (+2 more)

### Community 124 - "load_verified_tokenizer"
Cohesion: 0.10
Nodes (12): format_encoder_context(), load_verified_tokenizer(), tokenize_repair_example(), batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), generate_diagnostic_text(), greedy_decode() (+4 more)

### Community 130 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 134 - "tests_e2e/conftest.py"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 137 - "audit_dataset_release.py"
Cohesion: 0.31
Nodes (5): token_profile(), audit(), emit(), main(), validate_all_patches()

### Community 177 - ".update_widgets"
Cohesion: 0.22
Nodes (3): format_stage_display(), phase_category(), test_dashboard_stage_display_formatting()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 883 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **100 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NativeTinyModel` connect `NativeTinyModel` to `modeling_bifocalgemma.py`, `build_native_tiny_model`, `test_molab_matrix_notebooks.py`, `.from_pretrained_backbones`, `run_matrix_stage`, `test_matrix_schedule_runtime.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `modeling_bifocalgemma.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06086956521739131 - nodes in this community are weakly interconnected._
- **Why does `RepairCollator` connect `RepairCollator` to `preflight.py`, `fit_offline_sft`, `modeling_bifocalgemma.py`, `test_context_tasks.py`, `matrix.py`, `test_memory_efficient_training.py`, `TestMemoryProbe`, `load_verified_tokenizer`, `ExperimentSpec`, `RepositorySnapshot`, `run_matrix_stage`, `Seq2SeqBatch`, `torch`, `JSONLRepairDataset`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `re` be split into smaller, more focused modules?**
  _Cohesion score 0.11153846153846154 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `remote_reset_plan` to `full_pool.py`, `preflight.py`, `fit_offline_sft`, `stable_fingerprint`, `synchronize_matrix`, `hub.py`, `matrix.py`, `prepare_context_data.py`, `evaluation/context.py`, `run_matrix_training`, `benchmark_upload.py`, `schedule.py`, `json`, `test_upload_recovery_receipts.py`, `compute_step_loss`, `load_verified_tokenizer`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._