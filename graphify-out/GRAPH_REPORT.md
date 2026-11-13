# Graph Report - bifocalgemma  (2026-11-13)

## Corpus Check
- 307 files · ~392,995 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2632 nodes · 7656 edges · 140 communities (94 shown, 46 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 829 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `08df73aa`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- test_molab_matrix_notebooks.py
- stable_fingerprint
- fast_pipeline.py
- prepare_colab_pilot.py
- hashlib
- distillation.py
- models/checkpoint_transfer.py
- types
- full_pool.py
- FullCrossCachedAttention
- eval_code.py
- subprocess
- test_matrix_runtime_loader.py
- test_molab_pilot.py
- schedule.py
- re
- shared_execution.py
- test_snapshot_verifies_replacement_before_pruning
- native.py
- json
- matrix.py
- test_tier5_adversarial.py
- test_repository_study.py
- test_data_remediation.py
- get_v2_experiment
- TokenizedRepairExample
- test_editing_manifest.py
- torch
- FastTrainingPipeline
- remediation.py
- test_memory_efficient_training.py
- GatedCrossAttentionBridge
- typing
- prepare_colab_probe.py
- repository.py
- normalize_context_task
- test_tier1_features.py
- fit_offline_sft
- NativeTinyModel
- v3/__init__.py
- protocol.py
- test_context_benchmark_resume.py
- adversarial_r3_probe.py
- test_molab_remediation.py
- test_training_recipes.py
- BifocalGemmaForCausalLM
- tests_e2e/conftest.py
- test_matrix_schedule_runtime.py
- evaluation/context.py
- repository_study.py
- contracts.py
- examples
- native_python.py
- apply_word_embedding_policy
- RepoContextRouter
- reference_resilient_mcq_parser
- run_sequential_pipeline
- TestGemma4EdgeTopology
- JSONLRepairDataset
- repository_corpus.py
- TestBifocalGemmaArchitecture
- configure_full_rank_cross_attention
- test_gsm8k.py
- test_wiggly_dashboard_lifecycle.py
- test_curation.py
- ASTGraphProjector
- BifocalGemmaModel
- test_offline_sft_training.py
- adapters.py
- atomic_json
- build_joint_visibility
- JSONLRepairDataset
- CodeOverlapIndex
- build_native_tiny_model
- WigglyDashboard
- test_v2_workflow.py
- telemetry.py
- releases.py
- test_v2_fixed_execution.py
- strict_letter
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- bifocalgemma
- test_colab_pilot.py
- v2.py
- data/offline_sft.py
- seal_report
- TrainingLossLog
- pytest
- test_notebook_hf_flow.py
- benchmark_upload.py
- curation.py
- typing
- argparse
- load_verified_tokenizer
- inspect_python_target
- TrainingDashboard
- test_fast_pipeline.py
- config.py
- packed_tokens.py
- BifocalGemmaPreTrainedModel
- collections_abc
- optimizers.py
- stages.py
- tempfile
- test_benchmark_prompt_budget.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- examples
- SwiGLUDualGateBridge
- generate_full_matrix_notebooks.py
- load_asset_json
- generate_training_notebooks.py
- TiledEncoderAttention
- run_length_grid
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- flex_attention.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- test_profile_publish_parent_guard_keeps_concurrent_winner

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 84 edges
2. `stable_fingerprint()` - 69 edges
3. `load_verified_tokenizer()` - 58 edges
4. `TokenizedRepairExample` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `run_matrix_training()` - 46 edges
7. `_execute_v2()` - 45 edges
8. `configure_memory_efficient_training()` - 44 edges
9. `build_native_tiny_model()` - 43 edges
10. `RepairCollator` - 42 edges

## Surprising Connections (you probably didn't know these)
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_telemetry_history_appends_all_active_stages()` --calls--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/models/v2/native.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_dashboard_stage_display_formatting()` --calls--> `format_stage_display()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (140 total, 46 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.05
Nodes (33): pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), dependency_state(), test_changed_transitive_loaded_ui_dependency_also_requires_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version() (+25 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.08
Nodes (30): configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), configure_matrix_stage(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), fixture_model() (+22 more)

### Community 2 - "stable_fingerprint"
Cohesion: 0.31
Nodes (9): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+1 more)

### Community 3 - "fast_pipeline.py"
Cohesion: 0.09
Nodes (13): export_matrix_pretrained_model(), load_local_native_model(), configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), tensor_bytes() (+5 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.05
Nodes (42): fail_second_microbatch(), test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), measure(), digest() (+34 more)

### Community 5 - "hashlib"
Cohesion: 0.17
Nodes (14): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+6 more)

### Community 9 - "full_pool.py"
Cohesion: 0.19
Nodes (11): file_digest(), read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+3 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.11
Nodes (7): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "eval_code.py"
Cohesion: 0.16
Nodes (6): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 12 - "subprocess"
Cohesion: 0.13
Nodes (6): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.31
Nodes (11): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_release_check_rejects_local_preview() (+3 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.15
Nodes (10): test_exported_worker_has_parity_context_and_exact_adapter_checks(), test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), render(), main(), render() (+2 more)

### Community 15 - "schedule.py"
Cohesion: 0.12
Nodes (21): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), ScheduleRecord, SharedSchedule (+13 more)

### Community 16 - "re"
Cohesion: 0.12
Nodes (13): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), extract_unified_diff(), refresh_patch_targets(), patch_syntax() (+5 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.09
Nodes (24): benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe() (+16 more)

### Community 18 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 19 - "native.py"
Cohesion: 0.09
Nodes (6): NormalizedAlignment, build_native_model(), native_config(), NativeMergedAttention, RoutedRepoBatch, version()

### Community 20 - "json"
Cohesion: 0.12
Nodes (12): write_view(), test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), test_view_record_json_is_lossless_and_gsm_shots_are_explicit(), audit_pool(), audit_views(), main(), main(), source_notices() (+4 more)

### Community 21 - "matrix.py"
Cohesion: 0.07
Nodes (27): benchmark_directory_name(), checked_prefix(), download_model(), recover_remote_run(), RemoteHeartbeat, retry_delay(), _synchronize_once(), target_fingerprint() (+19 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+15 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.12
Nodes (7): ExecutingScoringOptions, ScoringOptions, ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards(), test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words(), test_preparation_reuses_one_verified_snapshot_without_changing_export()

### Community 24 - "test_data_remediation.py"
Cohesion: 0.26
Nodes (12): merge_lineage(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage() (+4 more)

### Community 25 - "get_v2_experiment"
Cohesion: 0.12
Nodes (11): get_v2_experiment(), cached_logits(), InferenceCache, test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only(), test_encoder_bidirectional_window_boundary_matches_native() (+3 more)

### Community 26 - "TokenizedRepairExample"
Cohesion: 0.09
Nodes (9): Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), synthetic_stress_target(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn(), test_notebook_stress_helpers_match_shared_library() (+1 more)

### Community 27 - "test_editing_manifest.py"
Cohesion: 0.26
Nodes (10): editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical() (+2 more)

### Community 28 - "torch"
Cohesion: 0.18
Nodes (12): fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, sample_encoder_mass(), test_flex_backward_head256_sequence_above_small_sequence_branch() (+4 more)

### Community 30 - "remediation.py"
Cohesion: 0.16
Nodes (10): split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), preference(), retokenize_records() (+2 more)

### Community 31 - "test_memory_efficient_training.py"
Cohesion: 0.22
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 34 - "typing"
Cohesion: 0.09
Nodes (13): backward_window(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids() (+5 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "repository.py"
Cohesion: 0.18
Nodes (8): git_environment(), GitSnapshotStore, prepare_repository_record(), prepare_repository_record_with_snapshot(), RepositoryTask, CharacterTokenizer, test_quarantine_fork_alias_and_grouped_ingestion(), test_source_whitelist_and_gold_separation()

### Community 37 - "normalize_context_task"
Cohesion: 0.19
Nodes (11): normalize_context_task(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record(), parent(), test_evaluation_and_unlicensed_sources_rejected() (+3 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (12): task_prompt(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval() (+4 more)

### Community 39 - "fit_offline_sft"
Cohesion: 0.11
Nodes (18): cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint() (+10 more)

### Community 40 - "NativeTinyModel"
Cohesion: 0.12
Nodes (6): ModelOutput, NativeTinyModel, __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy(), checkpoint()

### Community 41 - "v3/__init__.py"
Cohesion: 0.11
Nodes (11): configure_full_rank_cross_attention(), batched_greedy_decode(), infer_benchmark_weight_tier(), prepare_sampled_benchmark_collection(), resolve_optimal_eval_batch_size(), build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example() (+3 more)

### Community 42 - "protocol.py"
Cohesion: 0.12
Nodes (14): build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), mcq_report_metrics(), _positive_integer(), _runtime_model_identity(), scorer_fingerprint(), test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold() (+6 more)

### Community 43 - "test_context_benchmark_resume.py"
Cohesion: 0.13
Nodes (11): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization() (+3 more)

### Community 44 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "test_training_recipes.py"
Cohesion: 0.17
Nodes (12): comparison_recipes(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), test_four_fresh_runs_per_notebook_and_independent_resume(), benchmark() (+4 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.10
Nodes (16): BifocalGemmaConfig, BifocalGemmaForCausalLM, _resolve_backbone_weights(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization() (+8 more)

### Community 48 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 49 - "test_matrix_schedule_runtime.py"
Cohesion: 0.09
Nodes (6): fixture_data(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), test_partial_backward_oom_replays_entire_window_without_double_gradient()

### Community 50 - "evaluation/context.py"
Cohesion: 0.11
Nodes (17): decode_text(), build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), prompt_ids() (+9 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 52 - "contracts.py"
Cohesion: 0.11
Nodes (20): RepairExample, RepositoryFile, RepositorySnapshot, apply_patch_to_snapshot(), example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord (+12 more)

### Community 53 - "examples"
Cohesion: 0.12
Nodes (7): TestOfflineSFT, TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 54 - "native_python.py"
Cohesion: 0.18
Nodes (11): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), validate_code_eval_snapshots(), NativeTaskResult, score_python_patch() (+3 more)

### Community 55 - "apply_word_embedding_policy"
Cohesion: 0.18
Nodes (4): apply_word_embedding_policy(), EmbeddingPolicyResult, TinyReferenceModel, TestWordEmbeddingPolicy

### Community 57 - "reference_resilient_mcq_parser"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 58 - "run_sequential_pipeline"
Cohesion: 0.12
Nodes (14): remote_reset_plan(), reset_local_benchmarks(), reset_state(), synchronize_run(), resolve_training_cadence(), run_recipe_comparison(), run_sequential_pipeline(), publish_final() (+6 more)

### Community 62 - "repository_corpus.py"
Cohesion: 0.07
Nodes (33): repair_preimage_coverage(), tokenize_context_task(), patch_paths(), safe_repo_path(), ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence() (+25 more)

### Community 64 - "configure_full_rank_cross_attention"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 65 - "test_gsm8k.py"
Cohesion: 0.17
Nodes (8): FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train(), test_verified_preparation_and_input_only_loading()

### Community 67 - "test_curation.py"
Cohesion: 0.19
Nodes (11): export_pool(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_frozen_budget_matches_all_six_controls(), test_frozen_pool_verifies_checksums_before_use(), test_invalid_requested_pool_budget_is_rejected(), test_license_repo_and_benchmark_gates() (+3 more)

### Community 69 - "BifocalGemmaModel"
Cohesion: 0.13
Nodes (4): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel, build_pretrained_model()

### Community 70 - "test_offline_sft_training.py"
Cohesion: 0.11
Nodes (5): TestNotebookCellParity, test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 72 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 74 - "atomic_json"
Cohesion: 0.15
Nodes (10): activity(), atomic_json(), append_event(), activity(), test_preflight_upload_verifies_exact_evidence_without_checkpoint(), event(), save(), check_followers() (+2 more)

### Community 76 - "build_joint_visibility"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 77 - "JSONLRepairDataset"
Cohesion: 0.15
Nodes (3): JSONLRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 78 - "CodeOverlapIndex"
Cohesion: 0.35
Nodes (10): code_shingles(), CodeOverlapIndex, test_disk_index_matches_memory_and_syntax_repairs_are_admitted(), test_full_overlap_index_skips_impossible_blob_reads(), test_near_matching_and_short_fragment_policy(), test_overlap_bound_preserves_near_matches_and_heuristic_index(), test_readonly_partial_postings_never_use_full_intersection_bound(), test_index_with_legacy_normalization_is_rejected() (+2 more)

### Community 79 - "build_native_tiny_model"
Cohesion: 0.11
Nodes (6): run_training_memory_preflight(), build_native_tiny_model(), fit_curated_sft(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, TestCheckpointTransfer

### Community 80 - "WigglyDashboard"
Cohesion: 0.29
Nodes (6): WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 81 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+4 more)

### Community 82 - "telemetry.py"
Cohesion: 0.15
Nodes (6): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), LastSeenWidget, phase_category()

### Community 83 - "releases.py"
Cohesion: 0.10
Nodes (24): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+16 more)

### Community 84 - "test_v2_fixed_execution.py"
Cohesion: 0.23
Nodes (9): guard(), examples(), fixture_guard(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure() (+1 more)

### Community 85 - "strict_letter"
Cohesion: 0.25
Nodes (4): _extract_match_letter(), fallback_letter_regex(), strict_letter(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 91 - "test_colab_pilot.py"
Cohesion: 0.22
Nodes (3): _DummyTokenizer, test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order()

### Community 92 - "v2.py"
Cohesion: 0.11
Nodes (18): source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics(), stratified_subset(), download_model(), _execute_v2() (+10 more)

### Community 93 - "data/offline_sft.py"
Cohesion: 0.20
Nodes (7): curated_pool_digest(), git_text_lines(), load_curated_pool(), prepare_curated_training(), tokenize(), tokenize_repository_record(), build_pool()

### Community 95 - "seal_report"
Cohesion: 0.23
Nodes (9): seal_report(), save(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), valid_profile(), check_evidence() (+1 more)

### Community 101 - "pytest"
Cohesion: 0.14
Nodes (8): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.24
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 104 - "benchmark_upload.py"
Cohesion: 0.12
Nodes (8): BenchmarkUploadQueue, retry_delay(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed()

### Community 105 - "curation.py"
Cohesion: 0.23
Nodes (7): code_signature(), code_tokens(), CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), test_cross_source_duplicates_and_nextcoder_variants(), test_unparseable_fallback_preserves_whitespace()

### Community 106 - "typing"
Cohesion: 0.06
Nodes (5): CachedAttention, build_bifocalgemma_v3_optimizer(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), clamped_triton_kernel_options()

### Community 107 - "argparse"
Cohesion: 0.13
Nodes (8): audit(), main(), main(), main(), distribution(), main(), profile(), main()

### Community 108 - "load_verified_tokenizer"
Cohesion: 0.11
Nodes (15): CheckpointRef, prepare_editing_records(), split_editing_records(), format_encoder_context(), load_verified_tokenizer(), tokenize_repair_example(), generate_diagnostic_text(), generate_patch() (+7 more)

### Community 109 - "inspect_python_target"
Cohesion: 0.19
Nodes (7): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples(), test_warning_has_provenance_without_stderr_or_source_mutation()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.08
Nodes (13): FastTrainingConfig, DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+5 more)

### Community 114 - "config.py"
Cohesion: 0.06
Nodes (24): BackboneProfile, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), EvaluationReport, extract_token_ast_features(), TokenASTFeatures (+16 more)

### Community 115 - "packed_tokens.py"
Cohesion: 0.13
Nodes (7): build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, compare_predictions(), test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes(), test_paired_bootstrap_and_cohort_rejection()

### Community 117 - "collections_abc"
Cohesion: 0.23
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 118 - "optimizers.py"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 122 - "stages.py"
Cohesion: 0.25
Nodes (3): snapshot(), synchronize_v2(), upload_initial_results()

### Community 123 - "tempfile"
Cohesion: 0.31
Nodes (5): audit(), emit(), main(), validate_all_patches(), main()

### Community 124 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 127 - "examples"
Cohesion: 0.11
Nodes (17): examples(), prepare(), calibration(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight() (+9 more)

### Community 129 - "generate_full_matrix_notebooks.py"
Cohesion: 0.36
Nodes (5): test_invalid_generation_modes_cannot_publish(), format_code(), main(), matrix_bundle(), render_notebook()

### Community 131 - "generate_training_notebooks.py"
Cohesion: 0.43
Nodes (4): main(), module_definitions(), render(), use_embedded_imports()

### Community 133 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 980 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **46 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `fast_pipeline.py`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `typing`, `JSONLRepairDataset`, `BifocalGemmaPreTrainedModel`, `BifocalGemmaForCausalLM`, `FastTrainingPipeline`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05336538461538461 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `atomic_json` to `stable_fingerprint`, `prepare_colab_pilot.py`, `re`, `shared_execution.py`, `test_snapshot_verifies_replacement_before_pruning`, `json`, `matrix.py`, `fit_offline_sft`, `NativeTinyModel`, `v3/__init__.py`, `test_training_recipes.py`, `evaluation/context.py`, `contracts.py`, `run_sequential_pipeline`, `repository_corpus.py`, `test_v2_workflow.py`, `test_v2_fixed_execution.py`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `v2.py`, `seal_report`, `benchmark_upload.py`, `argparse`, `packed_tokens.py`, `stages.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `examples`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08333333333333333 - nodes in this community are weakly interconnected._
- **Why does `SwiGLUDualGateBridge` connect `SwiGLUDualGateBridge` to `v3/__init__.py`, `typing`, `BifocalGemmaForCausalLM`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._