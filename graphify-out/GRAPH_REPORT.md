# Graph Report - bifocalgemma  (2026-11-18)

## Corpus Check
- 249 files · ~329,652 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2289 nodes · 6196 edges · 187 communities (88 shown, 99 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 730 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6c65153e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- TrainingDashboard
- repository_corpus.py
- modeling_bifocalgemma.py
- build_native_tiny_model
- re
- stages.py
- test_notebook_dependencies.py
- scorer_fingerprint
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
- test_curation.py
- test_dashboard_dependency_state.py
- patch_cross_entropy
- adversarial_r3_probe.py
- torch
- random
- full_pool.py
- run_batched_context_benchmark
- examples
- FeatureNotReadyError
- prepare_context_data.py
- synchronize_matrix
- collections_abc
- test_molab_matrix_notebooks.py
- test_context_benchmark_resume.py
- protocol.py
- load_local_native_model
- evaluation/context.py
- test_molab_remediation.py
- test_pretrained_hf_model.py
- sys
- bifocalgemma/__init__.py
- notebook_launcher.py
- repository_study.py
- test_repository_study.py
- run_matrix_stage
- audit_context_release.py
- test_matrix_schedule_runtime.py
- training/offline_sft.py
- inspect_python_target
- compute_step_loss
- copy
- JSONLRepairDataset
- remote_reset_plan
- memory_probe.py
- math
- test_training_telemetry.py
- SwiGLUDualGateBridge
- test_context_tasks.py
- contracts.py
- repository_corpus.py
- tempfile
- test_construction_blocks.py
- NativeMergedAttention
- NativeTinyModel
- WigglyDashboard
- telemetry.py
- json
- JSONLRepairDataset
- test_hf_pretrained_model.py
- test_upload_recovery_receipts.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_benchmark_prompt_budget.py
- run_matrix_training
- RepairCollator
- remediation.py
- normalize_context_task
- stable_fingerprint
- tests_e2e/conftest.py
- matrix.py
- fast_pipeline.py
- BifocalGemmaConfig
- TestMemoryProbe
- .snapshot
- patch_matrix_tokenization
- test_fast_pipeline.py
- config.py
- PackedTokenizedDataset
- test_training_recipes.py
- sys
- repository_sft.py
- load_verified_tokenizer
- test_tier1_features.py
- tests_e2e/conftest.py
- build_repository_corpus
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- test_evaluation.py
- RecordingStream
- ProbeLoRALinear
- run_length_grid
- TestMergedAttentionReference
- ast.py
- source_fingerprints
- dependency_state
- TrainingLossLog

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 65 edges
2. `atomic_json()` - 52 edges
3. `load_verified_tokenizer()` - 50 edges
4. `run_matrix_training()` - 50 edges
5. `RepositorySnapshot` - 41 edges
6. `BifocalGemmaForCausalLM` - 41 edges
7. `run_matrix_benchmark()` - 40 edges
8. `TokenizedRepairExample` - 39 edges
9. `FastTrainingPipeline` - 37 edges
10. `fit_offline_sft()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_full_overlap_index_skips_impossible_blob_reads()` --calls--> `CodeOverlapIndex`  [INFERRED]
  tests/test_curation.py → src/bifocalgemma/data/curation.py
- `test_near_matching_and_short_fragment_policy()` --calls--> `CodeOverlapIndex`  [INFERRED]
  tests/test_curation.py → src/bifocalgemma/data/curation.py
- `test_readonly_partial_postings_never_use_full_intersection_bound()` --calls--> `CodeOverlapIndex`  [INFERRED]
  tests/test_curation.py → src/bifocalgemma/data/curation.py
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (187 total, 99 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.20
Nodes (9): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution(), save() (+1 more)

### Community 2 - "repository_corpus.py"
Cohesion: 0.14
Nodes (12): ContextWindow, render_window(), select_repository_context(), _terms(), context_license_evidence(), path_license_evidence(), recognize_license(), resolved_license() (+4 more)

### Community 3 - "modeling_bifocalgemma.py"
Cohesion: 0.09
Nodes (8): batched_greedy_decode(), infer_benchmark_weight_tier(), greedy_decode(), GreedyGeneration, FullCrossCachedAttention, cached_logits(), CachedAttention, InferenceCache

### Community 5 - "build_native_tiny_model"
Cohesion: 0.18
Nodes (4): build_native_tiny_model(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head()

### Community 6 - "re"
Cohesion: 0.12
Nodes (23): RepositoryFile, RepositorySnapshot, _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application() (+15 more)

### Community 7 - "stages.py"
Cohesion: 0.14
Nodes (14): file_digest(), build_packed_cache(), open_packed_cache(), checked_prefix(), recover_remote_run(), _copy_atomic(), _synchronize_once(), synchronize_run() (+6 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.12
Nodes (14): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+6 more)

### Community 10 - "scorer_fingerprint"
Cohesion: 0.13
Nodes (12): _extract_match_letter(), fallback_letter_regex(), score_context_case(), score_repo_niah_case(), strict_letter(), _clean_extracted_number(), extract_gsm8k_answer(), extract_release_answer() (+4 more)

### Community 11 - "FastTrainingPipeline"
Cohesion: 0.12
Nodes (15): _initialization_plan(), main(), FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule() (+7 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.25
Nodes (13): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_publication_only_adds_verified_immutable_runtime() (+5 more)

### Community 14 - "curation.py"
Cohesion: 0.18
Nodes (14): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), test_cross_source_duplicates_and_nextcoder_variants() (+6 more)

### Community 15 - "schedule.py"
Cohesion: 0.10
Nodes (24): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+16 more)

### Community 16 - "benchmark_upload.py"
Cohesion: 0.08
Nodes (11): retry_delay(), append_event(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed() (+3 more)

### Community 17 - "prepare_context_data.py"
Cohesion: 0.16
Nodes (9): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), TestOfflineEditing, TestTokenRoleAndDataIdentity (+1 more)

### Community 18 - "json"
Cohesion: 0.09
Nodes (16): curated_pool_digest(), compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit(), main(), audit(), emit(), main() (+8 more)

### Community 19 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 20 - "eval_code.py"
Cohesion: 0.17
Nodes (5): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (20): extract_mcq_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling() (+12 more)

### Community 23 - "hashlib"
Cohesion: 0.23
Nodes (11): digest(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k(), main() (+3 more)

### Community 24 - "test_curation.py"
Cohesion: 0.14
Nodes (15): export_pool(), split_connected_records(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_disk_index_matches_memory_and_syntax_repairs_are_admitted(), test_frozen_budget_matches_all_six_controls(), test_frozen_pool_verifies_checksums_before_use() (+7 more)

### Community 25 - "test_dashboard_dependency_state.py"
Cohesion: 0.18
Nodes (15): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+7 more)

### Community 26 - "patch_cross_entropy"
Cohesion: 0.16
Nodes (9): fit_offline_sft(), emit(), model_stage(), fit_repository_sft(), resource_snapshot(), TestOfflineSFT, TinyLossModel, test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated() (+1 more)

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "torch"
Cohesion: 0.06
Nodes (30): FullCrossTiledAttention, _norm_weight_dtype(), flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), mask_tile() (+22 more)

### Community 31 - "random"
Cohesion: 0.22
Nodes (7): TokenizedRepairExample, build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 32 - "full_pool.py"
Cohesion: 0.18
Nodes (10): read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+2 more)

### Community 34 - "run_batched_context_benchmark"
Cohesion: 0.18
Nodes (9): prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), prompt_ids(), run_context_benchmark(), activity(), DurableClock, test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline() (+1 more)

### Community 35 - "examples"
Cohesion: 0.27
Nodes (3): make_identity(), examples(), TestTrainingPolicy

### Community 36 - "FeatureNotReadyError"
Cohesion: 0.10
Nodes (16): DeveloperAgent, RepairExample, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter (+8 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.15
Nodes (14): normalize_context_task(), repair_preimage_coverage(), _resolve_eot_token_id(), task_prompt(), tokenize_context_task(), load_curated_pool(), test_repair_requires_visible_preimage_without_refilling_context(), completion_from_snapshot() (+6 more)

### Community 38 - "synchronize_matrix"
Cohesion: 0.17
Nodes (8): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix(), snapshot(), snapshot()

### Community 39 - "collections_abc"
Cohesion: 0.24
Nodes (4): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 40 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (23): make_experiment_spec(), configure_full_rank_cross_attention(), configure_memory_efficient_training(), is_cross_bridge_parameter(), configure_policy_model(), unified_recipe(), configure_matrix_stage(), test_cached_uncached_and_batched_greedy_decode_equivalence() (+15 more)

### Community 41 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 42 - "protocol.py"
Cohesion: 0.10
Nodes (16): build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer(), _runtime_model_identity(), test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold() (+8 more)

### Community 43 - "load_local_native_model"
Cohesion: 0.20
Nodes (5): build_native_model(), load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), test_local_loader_preflights_and_preserves_shared_words()

### Community 44 - "evaluation/context.py"
Cohesion: 0.11
Nodes (13): build_repo_niah_case(), completion_metrics(), format_crosscodelong_summary(), generate_repo_niah_haystack(), make_distractor_python_chunk(), normalize_completion_prediction(), _sequence_and_identifier_metrics(), _single_repair_target_file() (+5 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 47 - "test_pretrained_hf_model.py"
Cohesion: 0.14
Nodes (8): _resolve_backbone_weights(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_enable_memory_efficient_training_and_softcapping(), test_export_matrix_pretrained_model_baseline(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 48 - "sys"
Cohesion: 0.24
Nodes (7): test_invalid_generation_modes_cannot_publish(), format_code(), main(), matrix_bundle(), render_notebook(), source_bundle(), source_bundle_bytes()

### Community 50 - "notebook_launcher.py"
Cohesion: 0.35
Nodes (6): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.07
Nodes (25): canonical_snapshot_hash(), example_from_record(), prepare_repository_record(), RepositoryTask, NativeTaskResult, score_python_patch(), summarize_native_results(), authoritative_empirical_results() (+17 more)

### Community 53 - "run_matrix_stage"
Cohesion: 0.14
Nodes (11): resolve_optimal_eval_batch_size(), export_matrix_pretrained_model(), run_matrix_stage(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_partial_backward_oom_replays_entire_window_without_double_gradient() (+3 more)

### Community 54 - "audit_context_release.py"
Cohesion: 0.21
Nodes (7): ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "training/offline_sft.py"
Cohesion: 0.20
Nodes (9): encoder_diagnostics(), stratified_subset(), cpu_tree(), evaluate_sft_loss(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), validation_batches() (+1 more)

### Community 58 - "inspect_python_target"
Cohesion: 0.19
Nodes (8): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_curated_warning_target_retained_and_summary_distinguishes_duplicate(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples(), test_warning_has_provenance_without_stderr_or_source_mutation()

### Community 59 - "compute_step_loss"
Cohesion: 0.11
Nodes (13): backward_window(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), compute_step_loss(), differential_diff_cross_entropy(), _pointer_chunk_forward() (+5 more)

### Community 60 - "copy"
Cohesion: 0.30
Nodes (11): remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage(), test_heldout_role_rejected() (+3 more)

### Community 62 - "remote_reset_plan"
Cohesion: 0.13
Nodes (7): remote_reset_plan(), reset_local_benchmarks(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training(), main()

### Community 63 - "memory_probe.py"
Cohesion: 0.33
Nodes (5): gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes()

### Community 64 - "math"
Cohesion: 0.14
Nodes (11): fit_curated_sft(), validate_optimizer_config(), build_training_groups(), default_training_profiles(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence() (+3 more)

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.17
Nodes (3): test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime()

### Community 67 - "SwiGLUDualGateBridge"
Cohesion: 0.06
Nodes (5): NormalizedAlignment, SwiGLUDualGateBridge, TestBifocalGemmaArchitecture, TestGemma4EdgeTopology, test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()

### Community 68 - "test_context_tasks.py"
Cohesion: 0.12
Nodes (5): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_tokenize_context_task_preserves_target_indentation_and_newlines()

### Community 69 - "contracts.py"
Cohesion: 0.11
Nodes (3): EvaluationReport, test_exact_synthetic_lengths_preserve_shift_and_closing_turn(), test_notebook_stress_helpers_match_shared_library()

### Community 72 - "repository_corpus.py"
Cohesion: 0.29
Nodes (9): patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch(), test_license_document_include_is_resolved_inside_snapshot() (+1 more)

### Community 74 - "tempfile"
Cohesion: 0.15
Nodes (7): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 79 - "NativeTinyModel"
Cohesion: 0.16
Nodes (4): ModelOutput, NativeTinyModel, __getattr__(), patch_cross_entropy()

### Community 80 - "WigglyDashboard"
Cohesion: 0.24
Nodes (6): WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 82 - "telemetry.py"
Cohesion: 0.13
Nodes (8): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), LastSeenWidget, phase_category(), test_monitor_escapes_errors_updates_validation_and_exports_html(), test_dashboard_stage_display_formatting()

### Community 83 - "json"
Cohesion: 0.09
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 85 - "test_hf_pretrained_model.py"
Cohesion: 0.17
Nodes (5): test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_bridge_init_preserved_across_post_init(), test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_generate_multi_eos_per_row_stopping_and_cache_parity()

### Community 87 - "test_upload_recovery_receipts.py"
Cohesion: 0.09
Nodes (4): test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), download(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only(), test_standalone_export_requires_matching_remote_contents()

### Community 91 - "test_benchmark_prompt_budget.py"
Cohesion: 0.16
Nodes (6): _context_already_includes_instruction(), decoder_prompt(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 92 - "run_matrix_training"
Cohesion: 0.08
Nodes (20): BenchmarkUploadQueue, download_model(), RemoteHeartbeat, _backfill_final_stage(), _final_stage_info(), _latest_checkpoint_matches(), _local_standalone_export_valid(), resolve_training_contract() (+12 more)

### Community 93 - "RepairCollator"
Cohesion: 0.11
Nodes (5): Seq2SeqBatch, RepairCollator, __getattr__(), write_swebench_predictions(), TestDataAndLoss

### Community 95 - "remediation.py"
Cohesion: 0.16
Nodes (10): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+2 more)

### Community 96 - "normalize_context_task"
Cohesion: 0.10
Nodes (19): generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), editing_manifest(), repository_editing_record(), validate_context_license_record(), repository_row() (+11 more)

### Community 101 - "stable_fingerprint"
Cohesion: 0.19
Nodes (11): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), retry_delay() (+3 more)

### Community 102 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): pytest_configure(), reference_resilient_mcq_parser(), validate_empirical_results_schema(), validate_paper_manuscript(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks(), test_scenario_end_to_end_mock_evaluation_pipeline() (+2 more)

### Community 103 - "matrix.py"
Cohesion: 0.07
Nodes (12): _compute_per_row_cross_query(), benchmark_directory_name(), target_fingerprint(), build_bifocalgemma_optimizer(), extract_trivial_token_ids(), is_zero_weight_decay_bridge_parameter(), resolve_diff_token_ids(), clamped_triton_kernel_options() (+4 more)

### Community 105 - "BifocalGemmaConfig"
Cohesion: 0.17
Nodes (3): BifocalGemmaConfig, _check_transformers_version(), native_config()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.14
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 114 - "config.py"
Cohesion: 0.11
Nodes (10): BackboneProfile, CheckpointRef, ExperimentSpec, get_v2_experiment(), _pair(), _ref(), validate_experiment_matrix(), BaselineScores (+2 more)

### Community 116 - "test_training_recipes.py"
Cohesion: 0.17
Nodes (5): comparison_recipes(), limit_threads(), test_fp32_alignment_on_bf16_backbone(), test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 120 - "sys"
Cohesion: 0.21
Nodes (3): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), main(), render()

### Community 122 - "repository_sft.py"
Cohesion: 0.25
Nodes (6): prepare_editing_records(), split_editing_records(), prepare_curated_training(), tokenize(), prepare_repository_training(), tokenize_repository_record()

### Community 124 - "load_verified_tokenizer"
Cohesion: 0.18
Nodes (6): load_verified_tokenizer(), tokenize_repair_example(), greedy_decode(), run_repo_niah_eval(), TestRealOfflineTokenizers, TestTokenizerVerification

### Community 130 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 134 - "tests_e2e/conftest.py"
Cohesion: 0.14
Nodes (6): test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 137 - "build_repository_corpus"
Cohesion: 0.26
Nodes (4): build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 146 - "test_evaluation.py"
Cohesion: 0.18
Nodes (4): test_crosscodelong_normalize_completion_prediction_strips_fences_and_suffix_overlap(), test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity(), test_prompt_ids_instance_id_isolation_and_gold_rejection(), test_resilient_mcq_uses_terminal_final_answer_after_questioned_candidate()

### Community 173 - "ProbeLoRALinear"
Cohesion: 0.33
Nodes (4): ProbeLoRALinear, test_model_to_bfloat16_preserves_fp32_master_weights(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip()

### Community 174 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 883 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **99 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `stable_fingerprint` to `preflight.py`, `run_batched_context_benchmark`, `modeling_bifocalgemma.py`, `prepare_context_data.py`, `synchronize_matrix`, `stages.py`, `matrix.py`, `evaluation/context.py`, `run_matrix_training`, `benchmark_upload.py`, `schedule.py`, `json`, `audit_context_release.py`, `test_upload_recovery_receipts.py`, `compute_step_loss`, `load_verified_tokenizer`, `remote_reset_plan`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `repository_corpus.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14153846153846153 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaPreTrainedModel` connect `bifocalgemma/__init__.py` to `BifocalGemmaForCausalLM`, `modeling_bifocalgemma.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `modeling_bifocalgemma.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08912655971479501 - nodes in this community are weakly interconnected._
- **Why does `run_matrix_training()` connect `run_matrix_training` to `TrainingDashboard`, `stable_fingerprint`, `synchronize_matrix`, `matrix.py`, `test_molab_matrix_notebooks.py`, `stages.py`, `test_molab_remediation.py`, `patch_matrix_tokenization`, `schedule.py`, `load_verified_tokenizer`, `source_fingerprints`, `json`, `json`, `run_matrix_stage`, `test_matrix_schedule_runtime.py`, `repository_sft.py`, `copy`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._