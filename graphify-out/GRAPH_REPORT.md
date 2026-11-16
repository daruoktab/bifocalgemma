# Graph Report - bifocalgemma  (2026-11-16)

## Corpus Check
- 241 files · ~317,651 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2288 nodes · 6194 edges · 180 communities (84 shown, 96 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 733 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `071b331f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- math
- TrainingDashboard
- test_editing_manifest.py
- pathlib
- build_native_tiny_model
- re
- atomic_json
- test_notebook_dependencies.py
- FullCrossCachedAttention
- re
- test_matrix_runtime_loader.py
- curation.py
- schedule.py
- benchmark_upload.py
- prepare_context_data.py
- argparse
- test_gsm8k.py
- eval_code.py
- test_tier5_adversarial.py
- hashlib
- build_grounded_training_pool.py
- test_dashboard_dependency_state.py
- patch_cross_entropy
- adversarial_r3_probe.py
- torch
- FastTrainingPipeline
- run_repo_niah_eval
- random
- test_data_remediation.py
- benchmark.py
- examples
- FeatureNotReadyError
- build_repository_corpus
- adapters.py
- test_context_benchmark_mcq_budget_floor_execution
- make_experiment_spec
- test_memory_efficient_training.py
- protocol.py
- load_local_native_model
- evaluation/context.py
- generate_full_matrix_notebooks.py
- BifocalGemmaConfig
- test_evaluation.py
- bifocalgemma/__init__.py
- notebook_launcher.py
- repository_study.py
- RepairExample
- fit_offline_sft
- test_repository_study.py
- test_matrix_schedule_runtime.py
- Seq2SeqBatch
- contracts.py
- compute_step_loss
- test_tier4_scenarios.py
- SyntheticRepairDataset
- remote_reset_plan
- memory_probe.py
- pytest
- test_training_telemetry.py
- SwiGLUDualGateBridge
- context_tasks.py
- test_construction_blocks.py
- repository_corpus.py
- test_upload_recovery_receipts.py
- test_construction_blocks.py
- collections_abc
- copy
- fit_curated_sft
- WigglyDashboard
- wiggly_dashboard.py
- json
- .from_pretrained_backbones
- StreamingPointerLoss
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- BifocalGemmaForCausalLM
- bifocalgemma
- test_benchmark_prompt_budget.py
- matrix.py
- test_molab_remediation.py
- remediation.py
- subprocess
- stable_fingerprint
- dependency_state
- patch_matrix_tokenization
- Tensor
- test_molab_matrix_notebooks.py
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- .snapshot
- FastTrainingConfig
- ExperimentSpec
- PackedTokenizedDataset
- chunked_linear_cross_entropy
- stable_fingerprint
- test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only
- tokenization.py
- test_tier1_features.py
- tests_e2e/conftest.py
- audit_dataset_release.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- tokenize_repair_example
- CheckpointRef
- test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold
- DummyTrackingTokenizer
- test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues
- TestTokenRoleAndDataIdentity
- repair_preimage_coverage
- test_matrix_quarantine_legacy_checkpoint_vs_clean_resume

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 65 edges
2. `atomic_json()` - 52 edges
3. `load_verified_tokenizer()` - 50 edges
4. `run_matrix_training()` - 50 edges
5. `RepositorySnapshot` - 41 edges
6. `TokenizedRepairExample` - 41 edges
7. `BifocalGemmaForCausalLM` - 41 edges
8. `run_matrix_benchmark()` - 40 edges
9. `FastTrainingPipeline` - 37 edges
10. `fit_offline_sft()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()` --calls--> `resolve_optimal_eval_batch_size()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/evaluation/benchmark.py
- `test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline()` --calls--> `prompt_ids()`  [INFERRED]
  tests/test_context_tasks.py → src/bifocalgemma/evaluation/context.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry.py → src/bifocalgemma/training/source_identity.py
- `test_telemetry_history_appends_all_active_stages()` --calls--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (180 total, 96 thin omitted)

### Community 0 - "math"
Cohesion: 0.09
Nodes (17): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), build_training_groups(), Float32Alignment, validate_training_policy(), current_capacity() (+9 more)

### Community 1 - "TrainingDashboard"
Cohesion: 0.22
Nodes (4): render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 2 - "test_editing_manifest.py"
Cohesion: 0.32
Nodes (8): editing_manifest(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation(), test_repository_eval_is_rejected()

### Community 3 - "pathlib"
Cohesion: 0.07
Nodes (9): BackboneProfile, BaselineScores, ProbeLoRALinear, build_bifocalgemma_optimizer(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), format_duration(), loss_chart() (+1 more)

### Community 5 - "build_native_tiny_model"
Cohesion: 0.13
Nodes (6): build_native_tiny_model(), configure_probe_training(), TestNativeIntegration, TestMemoryProbe, test_inference_v2_adapter_roundtrip_and_baseline_group(), test_last_token_logits_match_full_head()

### Community 6 - "re"
Cohesion: 0.17
Nodes (17): _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines() (+9 more)

### Community 7 - "atomic_json"
Cohesion: 0.11
Nodes (13): synchronize_matrix(), checked_prefix(), recover_remote_run(), _copy_atomic(), RemoteHeartbeat, retry_delay(), _synchronize_once(), synchronize_run() (+5 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.12
Nodes (17): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+9 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.10
Nodes (8): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), clamped_triton_kernel_options(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "re"
Cohesion: 0.19
Nodes (13): RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), example_from_record(), prepare_repository_record_with_snapshot() (+5 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.21
Nodes (17): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+9 more)

### Community 14 - "curation.py"
Cohesion: 0.09
Nodes (31): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits() (+23 more)

### Community 15 - "schedule.py"
Cohesion: 0.10
Nodes (25): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+17 more)

### Community 16 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (17): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed() (+9 more)

### Community 17 - "prepare_context_data.py"
Cohesion: 0.14
Nodes (17): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), repository_task_from_row(), TestOfflineEditing (+9 more)

### Community 18 - "argparse"
Cohesion: 0.10
Nodes (9): compare_predictions(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), audit(), main(), main(), inspect(), main() (+1 more)

### Community 19 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 20 - "eval_code.py"
Cohesion: 0.17
Nodes (5): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (21): extract_mcq_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling() (+13 more)

### Community 23 - "hashlib"
Cohesion: 0.20
Nodes (13): load_verified_tokenizer(), _clean_extracted_number(), digest(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids() (+5 more)

### Community 24 - "build_grounded_training_pool.py"
Cohesion: 0.12
Nodes (5): build_pool(), main(), distribution(), main(), profile()

### Community 25 - "test_dashboard_dependency_state.py"
Cohesion: 0.18
Nodes (12): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+4 more)

### Community 26 - "patch_cross_entropy"
Cohesion: 0.12
Nodes (5): __getattr__(), patch_cross_entropy(), TestOfflineSFT, TinyLossModel, test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (8): score_repo_niah_case(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "torch"
Cohesion: 0.11
Nodes (15): flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention (+7 more)

### Community 30 - "run_repo_niah_eval"
Cohesion: 0.14
Nodes (7): build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), _single_repair_target_file()

### Community 31 - "random"
Cohesion: 0.12
Nodes (12): TokenizedRepairExample, RepairCollator, build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), __getattr__(), synthetic_stress_target() (+4 more)

### Community 32 - "test_data_remediation.py"
Cohesion: 0.26
Nodes (11): record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage(), test_heldout_role_rejected(), test_indentation_changes_are_not_identical() (+3 more)

### Community 34 - "benchmark.py"
Cohesion: 0.11
Nodes (15): batched_greedy_decode(), infer_benchmark_weight_tier(), prepare_sampled_benchmark_collection(), resolve_optimal_eval_batch_size(), run_batched_context_benchmark(), activity(), prompt_ids(), run_context_benchmark() (+7 more)

### Community 35 - "examples"
Cohesion: 0.14
Nodes (9): default_training_profiles(), comparison_recipes(), resolve_training_cadence(), make_identity(), examples(), TestTrainingPolicy, test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_optimized_native_resume_matches_full_and_rejects_different_recipe() (+1 more)

### Community 36 - "FeatureNotReadyError"
Cohesion: 0.21
Nodes (4): extract_token_ast_features(), TokenASTFeatures, BuildBlock, FeatureNotReadyError

### Community 37 - "build_repository_corpus"
Cohesion: 0.16
Nodes (7): ArchiveSnapshotStore, build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, repo_identity(), test_failed_download_removes_partial_bytes(), test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 38 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 39 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 40 - "make_experiment_spec"
Cohesion: 0.09
Nodes (28): make_experiment_spec(), configure_full_rank_cross_attention(), configure_memory_efficient_training(), load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint() (+20 more)

### Community 41 - "test_memory_efficient_training.py"
Cohesion: 0.24
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 42 - "protocol.py"
Cohesion: 0.13
Nodes (12): build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), _positive_integer(), _runtime_model_identity(), test_effective_allowance(), test_effective_spec_binds_gold_caps_and_runtime_spec(), test_invalid_allowance_fails_explicitly() (+4 more)

### Community 43 - "load_local_native_model"
Cohesion: 0.21
Nodes (5): load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), TestCheckpointTransfer, test_local_loader_preflights_and_preserves_shared_words()

### Community 44 - "evaluation/context.py"
Cohesion: 0.11
Nodes (18): completion_metrics(), _extract_match_letter(), fallback_letter_regex(), normalize_completion_prediction(), score_context_case(), _sequence_and_identifier_metrics(), strict_letter(), _strip_echoed_suffix() (+10 more)

### Community 45 - "generate_full_matrix_notebooks.py"
Cohesion: 0.19
Nodes (6): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), matrix_bundle(), main(), render(), source_bundle(), source_bundle_bytes()

### Community 47 - "BifocalGemmaConfig"
Cohesion: 0.08
Nodes (17): BifocalGemmaConfig, export_matrix_pretrained_model(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_generate_multi_eos_per_row_stopping_and_cache_parity(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype() (+9 more)

### Community 50 - "notebook_launcher.py"
Cohesion: 0.35
Nodes (6): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 51 - "repository_study.py"
Cohesion: 0.17
Nodes (24): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), prepare_repository_record(), read_source_rows(), RepositoryTask (+16 more)

### Community 52 - "RepairExample"
Cohesion: 0.17
Nodes (8): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract, TestTokenizerVerification

### Community 53 - "fit_offline_sft"
Cohesion: 0.15
Nodes (16): encoder_diagnostics(), stratified_subset(), cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage() (+8 more)

### Community 54 - "test_repository_study.py"
Cohesion: 0.09
Nodes (13): authoritative_empirical_results(), project_root(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), diff(), ExecutingScoringOptions, fixture() (+5 more)

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.08
Nodes (7): ModelOutput, NativeTinyModel, test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "Seq2SeqBatch"
Cohesion: 0.25
Nodes (3): Seq2SeqBatch, test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_fast_pipeline_batch_edge_cases()

### Community 58 - "contracts.py"
Cohesion: 0.13
Nodes (7): EvaluationReport, RepositoryFile, test_dependency_completion_removes_target_and_retains_real_helper(), TestRealOfflineTokenizers, test_missing_old_file_is_still_rejected(), test_modify_add_delete_preserves_unrelated_unicode_bytes(), test_applied_python_patch_reports_syntax_independently()

### Community 59 - "compute_step_loss"
Cohesion: 0.25
Nodes (5): backward_window(), compute_diff_token_weights(), compute_step_loss(), differential_diff_cross_entropy(), test_v3_streaming_pointer_loss_and_differential_diff_weighting()

### Community 60 - "test_tier4_scenarios.py"
Cohesion: 0.29
Nodes (3): test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 62 - "remote_reset_plan"
Cohesion: 0.22
Nodes (6): remote_reset_plan(), reset_local_benchmarks(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training()

### Community 63 - "memory_probe.py"
Cohesion: 0.15
Nodes (9): gpu_memory(), optimizer_bytes(), profile_training_group(), run_length_grid(), tensor_bytes(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library() (+1 more)

### Community 64 - "pytest"
Cohesion: 0.10
Nodes (6): test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule(), limit_threads(), test_fp32_alignment_on_bf16_backbone()

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.14
Nodes (5): _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime(), RecordingStream

### Community 67 - "SwiGLUDualGateBridge"
Cohesion: 0.07
Nodes (5): NormalizedAlignment, SwiGLUDualGateBridge, TestBifocalGemmaArchitecture, TestGemma4EdgeTopology, test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()

### Community 68 - "context_tasks.py"
Cohesion: 0.07
Nodes (20): normalize_context_task(), _resolve_eot_token_id(), task_prompt(), tokenize_context_task(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task() (+12 more)

### Community 72 - "repository_corpus.py"
Cohesion: 0.13
Nodes (13): context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), recognize_license(), resolved_license(), archive(), test_archive_commit_budget_and_static_license() (+5 more)

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "collections_abc"
Cohesion: 0.23
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 78 - "copy"
Cohesion: 0.10
Nodes (6): ensure_assets_on_disk(), load_asset_json(), _check_transformers_version(), native_config(), NativeMergedAttention, TestMergedAttentionReference

### Community 79 - "fit_curated_sft"
Cohesion: 0.25
Nodes (4): run_training_memory_preflight(), fit_curated_sft(), test_curated_resume_rejects_runtime_drift(), version()

### Community 80 - "WigglyDashboard"
Cohesion: 0.26
Nodes (7): phase_category(), WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 82 - "wiggly_dashboard.py"
Cohesion: 0.15
Nodes (4): format_stage_display(), LastSeenWidget, TrainingLossLog, test_dashboard_stage_display_formatting()

### Community 83 - "json"
Cohesion: 0.09
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 85 - "StreamingPointerLoss"
Cohesion: 0.25
Nodes (3): extract_trivial_token_ids(), StreamingPointerLoss, test_pointer_loss_gradient_checkpointing_and_trivial_filtering()

### Community 91 - "test_benchmark_prompt_budget.py"
Cohesion: 0.16
Nodes (6): _context_already_includes_instruction(), decoder_prompt(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 92 - "matrix.py"
Cohesion: 0.07
Nodes (23): benchmark_directory_name(), download_model(), target_fingerprint(), _backfill_final_stage(), _final_stage_info(), _latest_checkpoint_matches(), _local_standalone_export_valid(), resolve_training_contract() (+15 more)

### Community 93 - "test_molab_remediation.py"
Cohesion: 0.18
Nodes (4): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast()

### Community 95 - "remediation.py"
Cohesion: 0.09
Nodes (24): export_pool(), read_arrow_rows(), split_connected_records(), token_profile(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+16 more)

### Community 101 - "stable_fingerprint"
Cohesion: 0.10
Nodes (22): file_digest(), audit_repository_sources(), curated_pool_digest(), load_curated_pool(), prepare_editing_records(), repository_editing_record(), split_editing_records(), stable_fingerprint() (+14 more)

### Community 103 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 104 - "Tensor"
Cohesion: 0.36
Nodes (3): _pointer_chunk_forward(), _streaming_lse_pointer_nll(), streaming_pointer_auxiliary_loss()

### Community 105 - "test_molab_matrix_notebooks.py"
Cohesion: 0.15
Nodes (6): test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_per_model_hf_repo(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 106 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 113 - "FastTrainingConfig"
Cohesion: 0.12
Nodes (10): FastTrainingConfig, JSONLRepairDataset, test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_training_pipeline_features_and_export(), test_fast_pipeline_skips_training_when_already_completed() (+2 more)

### Community 114 - "ExperimentSpec"
Cohesion: 0.16
Nodes (7): ExperimentSpec, get_v2_experiment(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 116 - "chunked_linear_cross_entropy"
Cohesion: 0.33
Nodes (4): chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), checkpoint()

### Community 120 - "stable_fingerprint"
Cohesion: 0.27
Nodes (8): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), test_f2_mcq_token_budget_specification(), benchmarks()

### Community 124 - "tokenization.py"
Cohesion: 0.17
Nodes (10): RepairExample, decode_text(), format_encoder_context(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids() (+2 more)

### Community 130 - "test_tier1_features.py"
Cohesion: 0.09
Nodes (14): pytest_configure(), validate_empirical_results_schema(), validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics() (+6 more)

### Community 134 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 137 - "audit_dataset_release.py"
Cohesion: 0.52
Nodes (4): audit(), emit(), main(), validate_all_patches()

### Community 171 - "CheckpointRef"
Cohesion: 0.67
Nodes (3): CheckpointRef, _pair(), _ref()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 882 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **96 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FastTrainingPipeline` connect `FastTrainingPipeline` to `pytest`, `pathlib`, `FullCrossCachedAttention`, `BifocalGemmaConfig`, `FastTrainingConfig`, `ExperimentSpec`, `Seq2SeqBatch`, `BifocalGemmaForCausalLM`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `math` be split into smaller, more focused modules?**
  _Cohesion score 0.08858858858858859 - nodes in this community are weakly interconnected._
- **Why does `RepairCollator` connect `random` to `math`, `pathlib`, `context_tasks.py`, `stable_fingerprint`, `build_native_tiny_model`, `make_experiment_spec`, `test_memory_efficient_training.py`, `fit_curated_sft`, `matrix.py`, `ExperimentSpec`, `tokenize_repair_example`, `fit_offline_sft`, `Seq2SeqBatch`, `contracts.py`, `torch`, `FastTrainingPipeline`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.07231638418079096 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `math`, `benchmark.py`, `test_editing_manifest.py`, `build_repository_corpus`, `atomic_json`, `repository_corpus.py`, `audit_dataset_release.py`, `evaluation/context.py`, `curation.py`, `prepare_context_data.py`, `argparse`, `json`, `test_matrix_schedule_runtime.py`, `hashlib`, `stable_fingerprint`, `build_grounded_training_pool.py`, `matrix.py`, `remediation.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._