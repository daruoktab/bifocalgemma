# Graph Report - bifocalgemma  (2026-11-09)

## Corpus Check
- 290 files · ~363,898 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 18 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2369 nodes · 6952 edges · 110 communities (76 shown, 34 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 704 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `440330e5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- curation.py
- atomic_json
- prepare_colab_pilot.py
- releases.py
- config.py
- data/offline_sft.py
- test_profile_publish_parent_guard_keeps_concurrent_winner
- repository.py
- v3/__init__.py
- test_gsm8k.py
- subprocess
- save_matrix_checkpoint
- json
- NativeTinyModel
- ast
- shared_execution.py
- FeatureNotReadyError
- molab_pilot_support.py
- native_config
- prepare_context_data.py
- test_tier5_adversarial.py
- Any
- CodeOverlapIndex
- BifocalGemmaConfig
- gsm8k.py
- BenchmarkUploadQueue
- torch
- .run
- source_bundle_bytes
- patch_cross_entropy
- test_molab_matrix_notebooks.py
- v3/objectives.py
- test_memory_efficient_training.py
- TokenizedRepairExample
- repository_corpus.py
- typing
- adversarial_r3_probe.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- test_colab_pilot.py
- TestGemma4EdgeTopology
- apply_word_embedding_policy
- PackedTokenizedDataset
- test_molab_remediation.py
- swesmith.py
- test_tier2_boundary.py
- random
- profile_repository_lengths.py
- TestTrainingPolicy
- repository_study.py
- transfer_text_checkpoint
- ReferenceMergedAttention
- configure_probe_training
- test_snapshot_verifies_replacement_before_pruning
- adapters.py
- test_benchmark_prompt_budget.py
- memory_probe.py
- generate_training_notebooks.py
- stable_fingerprint
- fit_offline_sft
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- TestBifocalGemmaArchitecture
- resolve_diff_token_ids
- build_native_tiny_model
- tokenization.py
- pathlib
- is_cross_bridge_parameter
- curate_paired_sources
- run_length_grid
- v2.py
- benchmark.py
- test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract
- NativeMergedAttention
- TestNotebookCellParity
- SwiGLUDualGateBridge
- test_benchmark_upload.py
- fast_pipeline.py
- matrix.py
- native_python.py
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- BifocalGemmaForCausalLM
- bifocalgemma
- test_repository_study.py
- test_context_tasks.py
- evaluation/context.py
- GatedCrossAttentionBridge
- test_training_telemetry_v2.py
- telemetry.py
- build_repository_corpus
- project_root
- test_notebook_hf_flow.py
- test_v2_workflow.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- copy

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 83 edges
2. `stable_fingerprint()` - 65 edges
3. `TokenizedRepairExample` - 53 edges
4. `load_verified_tokenizer()` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `RepairCollator` - 40 edges
10. `run_matrix_training()` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `event()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py
- `save()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (110 total, 34 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.05
Nodes (22): pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt() (+14 more)

### Community 1 - "remediation.py"
Cohesion: 0.12
Nodes (23): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+15 more)

### Community 2 - "curation.py"
Cohesion: 0.13
Nodes (14): export_pool(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows() (+6 more)

### Community 3 - "atomic_json"
Cohesion: 0.06
Nodes (31): file_digest(), run_training_memory_preflight(), remote_reset_plan(), reset_local_benchmarks(), reset_state(), checked_prefix(), recover_remote_run(), _synchronize_once() (+23 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.09
Nodes (28): tokenize(), evaluate_sft_loss(), validation_batches(), tokenize_repository_record(), build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention() (+20 more)

### Community 5 - "releases.py"
Cohesion: 0.09
Nodes (25): archive_files(), audit_split_disjointness(), build_release(), digest(), main(), publish_release(), rows(), unpack_archive() (+17 more)

### Community 6 - "config.py"
Cohesion: 0.07
Nodes (17): BackboneProfile, CheckpointRef, _pair(), _ref(), validate_experiment_matrix(), EvaluationReport, extract_token_ast_features(), TokenASTFeatures (+9 more)

### Community 7 - "data/offline_sft.py"
Cohesion: 0.15
Nodes (15): editing_manifest(), editing_record(), git_text_lines(), prepare_editing_records(), repository_editing_record(), split_editing_records(), notebook_function(), repository_row() (+7 more)

### Community 9 - "repository.py"
Cohesion: 0.12
Nodes (16): RepositoryFile, RepositorySnapshot, apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), patch_paths(), safe_repo_path(), example_from_record() (+8 more)

### Community 10 - "v3/__init__.py"
Cohesion: 0.12
Nodes (6): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, clamped_triton_kernel_options(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (14): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+6 more)

### Community 12 - "subprocess"
Cohesion: 0.09
Nodes (11): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), audit(), emit(), main(), validate_all_patches(), main(), main(), main() (+3 more)

### Community 13 - "save_matrix_checkpoint"
Cohesion: 0.11
Nodes (11): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), configure_matrix_stage(), test_save_and_restore_matrix_checkpoint_and_prune(), test_stage1_to_stage2_optimizer_state_transfer() (+3 more)

### Community 14 - "json"
Cohesion: 0.12
Nodes (11): compare_predictions(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), audit_views(), main(), audit(), main(), main() (+3 more)

### Community 16 - "ast"
Cohesion: 0.12
Nodes (12): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), test_exported_worker_has_parity_context_and_exact_adapter_checks(), main(), main(), render_eval_script() (+4 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.05
Nodes (45): retry_delay(), fit_curated_sft(), build_policy_optimizer(), validate_optimizer_config(), build_training_groups(), default_training_profiles(), Float32Alignment, update_policy_learning_rates() (+37 more)

### Community 18 - "FeatureNotReadyError"
Cohesion: 0.06
Nodes (14): restore_checkpoint(), save_checkpoint(), ExperimentSpec, ModelOutput, Seq2SeqBatch, EncoderMemory, RepositoryCache, initialize_pretrained_components() (+6 more)

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.10
Nodes (17): save(), test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), measure(), digest() (+9 more)

### Community 21 - "prepare_context_data.py"
Cohesion: 0.16
Nodes (14): normalize_context_task(), repair_preimage_coverage(), task_prompt(), tokenize_context_task(), load_curated_pool(), repository_task_from_row(), audit_pool(), completion_from_snapshot() (+6 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (21): extract_mcq_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling() (+13 more)

### Community 23 - "Any"
Cohesion: 0.22
Nodes (4): authoritative_empirical_results(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.15
Nodes (24): code_shingles(), code_signature(), CodeOverlapIndex, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families() (+16 more)

### Community 25 - "BifocalGemmaConfig"
Cohesion: 0.07
Nodes (23): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), FastTrainingConfig, FastTrainingPipeline, run_fast_training(), ProbeLoRALinear, test_fast_training_pipeline_features_and_export() (+15 more)

### Community 26 - "gsm8k.py"
Cohesion: 0.19
Nodes (14): curated_pool_digest(), digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids() (+6 more)

### Community 27 - "BenchmarkUploadQueue"
Cohesion: 0.24
Nodes (4): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed()

### Community 28 - "torch"
Cohesion: 0.07
Nodes (20): flex_kernel_options(), fused_attention(), cached_logits(), CachedAttention, attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention() (+12 more)

### Community 29 - ".run"
Cohesion: 0.08
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 30 - "source_bundle_bytes"
Cohesion: 0.14
Nodes (14): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+6 more)

### Community 31 - "patch_cross_entropy"
Cohesion: 0.19
Nodes (4): __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy(), checkpoint()

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (6): test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_telemetry_history_appends_all_active_stages()

### Community 34 - "v3/objectives.py"
Cohesion: 0.14
Nodes (9): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), _pointer_chunk_forward(), _streaming_lse_pointer_nll(), streaming_pointer_auxiliary_loss(), StreamingPointerLoss (+1 more)

### Community 35 - "test_memory_efficient_training.py"
Cohesion: 0.27
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.08
Nodes (9): TokenizedRepairExample, RepairCollator, __getattr__(), build_native_model(), run_native_smoke(), run_reference_smoke(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn() (+1 more)

### Community 37 - "repository_corpus.py"
Cohesion: 0.12
Nodes (16): normalized_code(), validate_context_license_record(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), recognize_license(), resolved_license() (+8 more)

### Community 38 - "typing"
Cohesion: 0.06
Nodes (19): pytest_configure(), reference_resilient_mcq_parser(), validate_empirical_results_schema(), validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser() (+11 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 41 - "test_colab_pilot.py"
Cohesion: 0.20
Nodes (4): _DummyTokenizer, test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order()

### Community 43 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "swesmith.py"
Cohesion: 0.20
Nodes (7): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 47 - "test_tier2_boundary.py"
Cohesion: 0.09
Nodes (9): completion_metrics(), test_f3_crosscodelong_continuous_metrics(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard(), test_combination_crosscode_continuous_metrics_aggregation() (+1 more)

### Community 48 - "random"
Cohesion: 0.17
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), is_zero_weight_decay_bridge_parameter(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 49 - "profile_repository_lengths.py"
Cohesion: 0.28
Nodes (5): test_shared_profile_upload_retries_without_losing_local_selection(), publish(), distribution(), main(), profile()

### Community 51 - "repository_study.py"
Cohesion: 0.22
Nodes (22): get_experiment(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), load_verified_tokenizer(), generate_diagnostic_text(), fetch_tokenizer_assets() (+14 more)

### Community 53 - "ReferenceMergedAttention"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 55 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "adapters.py"
Cohesion: 0.23
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (5): decoder_prompt(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 58 - "memory_probe.py"
Cohesion: 0.23
Nodes (7): load_local_native_model(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), test_local_loader_preflights_and_preserves_shared_words()

### Community 59 - "generate_training_notebooks.py"
Cohesion: 0.43
Nodes (4): main(), module_definitions(), render(), use_embedded_imports()

### Community 60 - "stable_fingerprint"
Cohesion: 0.16
Nodes (13): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), build_packed_cache() (+5 more)

### Community 61 - "fit_offline_sft"
Cohesion: 0.21
Nodes (12): run_context_benchmark(), activity(), cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint() (+4 more)

### Community 63 - "TestBifocalGemmaArchitecture"
Cohesion: 0.06
Nodes (3): ASTGraphProjector, RepoContextRouter, TestBifocalGemmaArchitecture

### Community 64 - "resolve_diff_token_ids"
Cohesion: 0.25
Nodes (3): extract_trivial_token_ids(), resolve_diff_token_ids(), test_pointer_loss_gradient_checkpointing_and_trivial_filtering()

### Community 65 - "build_native_tiny_model"
Cohesion: 0.22
Nodes (3): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head()

### Community 66 - "tokenization.py"
Cohesion: 0.11
Nodes (9): DeveloperAgent, RepairExample, RepairPrediction, format_encoder_context(), tokenize_repair_example(), write_swebench_predictions(), generate_patch(), repair_prompt_ids() (+1 more)

### Community 67 - "pathlib"
Cohesion: 0.06
Nodes (25): examples(), TestOfflineSFT, TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), calibration() (+17 more)

### Community 70 - "curate_paired_sources"
Cohesion: 0.14
Nodes (9): code_tokens(), CodeOverlapMatch, count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples() (+1 more)

### Community 72 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 74 - "v2.py"
Cohesion: 0.08
Nodes (24): prepare_curated_training(), shared_contract(), source_fingerprints(), WithoutDocstrings, append_event(), benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics() (+16 more)

### Community 76 - "benchmark.py"
Cohesion: 0.11
Nodes (13): prompt_ids(), GreedyGeneration, InferenceCache, configure_full_rank_cross_attention(), batched_greedy_decode(), infer_benchmark_weight_tier(), prepare_sampled_benchmark_collection(), resolve_optimal_eval_batch_size() (+5 more)

### Community 77 - "test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract"
Cohesion: 0.25
Nodes (3): format_query_injected_context(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 82 - "test_benchmark_upload.py"
Cohesion: 0.05
Nodes (28): GrokOrScaleLoRA, orthogonal_direction(), comparison_recipes(), limited(), fail(), test_fatal_errors_not_hidden(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict() (+20 more)

### Community 83 - "fast_pipeline.py"
Cohesion: 0.11
Nodes (3): ensure_assets_on_disk(), load_asset_json(), BifocalGemmaPreTrainedModel

### Community 84 - "matrix.py"
Cohesion: 0.11
Nodes (11): fetch_collection(), run_matrix_benchmark(), run_matrix_stage(), run_matrix_training(), run_matrix_training_and_benchmark(), check_legacy_artifacts(), progress(), ensure_hardware_memory_guards() (+3 more)

### Community 85 - "native_python.py"
Cohesion: 0.21
Nodes (10): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), NativeTaskResult, score_python_patch(), summarize_native_results() (+2 more)

### Community 91 - "test_repository_study.py"
Cohesion: 0.12
Nodes (17): prepare_repository_record(), prepare_repository_record_with_snapshot(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions (+9 more)

### Community 92 - "test_context_tasks.py"
Cohesion: 0.20
Nodes (6): fallback_letter_regex(), strict_letter(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 93 - "evaluation/context.py"
Cohesion: 0.14
Nodes (10): decode_text(), build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), score_context_case() (+2 more)

### Community 94 - "GatedCrossAttentionBridge"
Cohesion: 0.12
Nodes (3): GatedCrossAttentionBridge, RMSNorm, RoutedRepoBatch

### Community 95 - "test_training_telemetry_v2.py"
Cohesion: 0.15
Nodes (5): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_training_resume_keeps_elapsed_from_step_five_through_ten(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 96 - "telemetry.py"
Cohesion: 0.08
Nodes (11): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog (+3 more)

### Community 97 - "build_repository_corpus"
Cohesion: 0.14
Nodes (8): ArchiveSnapshotStore, build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, repo_identity(), test_failed_download_removes_partial_bytes(), test_preparation_ledger_preserves_records_and_rejects_new_settings(), source_notices()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+4 more)

### Community 115 - "copy"
Cohesion: 0.08
Nodes (33): get_v2_experiment(), configure_memory_efficient_training(), NormalizedAlignment, load_sft_adapters_for_inference(), configure_policy_model(), unified_recipe(), fit_repository_sft(), make_experiment_spec() (+25 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 891 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `curation.py`, `prepare_colab_pilot.py`, `json`, `shared_execution.py`, `molab_pilot_support.py`, `prepare_context_data.py`, `BenchmarkUploadQueue`, `patch_cross_entropy`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `test_snapshot_verifies_replacement_before_pruning`, `stable_fingerprint`, `fit_offline_sft`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `pathlib`, `v2.py`, `benchmark.py`, `test_benchmark_upload.py`, `matrix.py`, `evaluation/context.py`, `build_repository_corpus`, `test_v2_workflow.py`, `copy`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05388471177944862 - nodes in this community are weakly interconnected._
- **Why does `TestBifocalGemmaArchitecture` connect `TestBifocalGemmaArchitecture` to `GatedCrossAttentionBridge`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11931818181818182 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `BifocalGemmaConfig`, `fast_pipeline.py`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._