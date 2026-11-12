# Graph Report - bifocalgemma  (2026-11-12)

## Corpus Check
- 305 files · ~390,761 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2596 nodes · 7569 edges · 123 communities (80 shown, 43 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 820 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `013f2393`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- test_molab_matrix_notebooks.py
- stable_fingerprint
- copy
- prepare_colab_pilot.py
- repository_sft.py
- contracts.py
- test_pretrained_probe_helpers.py
- types
- remediation.py
- FullCrossCachedAttention
- test_gsm8k.py
- pathlib
- test_matrix_runtime_loader.py
- test_molab_pilot.py
- schedule.py
- fit_offline_sft
- shared_execution.py
- examples
- NativeTinyModel
- json
- hashlib
- test_tier5_adversarial.py
- test_repository_study.py
- CodeOverlapIndex
- configure_memory_efficient_training
- TokenizedRepairExample
- hashlib
- torch
- tests_e2e/conftest.py
- patch_cross_entropy
- GatedCrossAttentionBridge
- typing
- prepare_colab_probe.py
- repository.py
- normalize_context_task
- test_tier1_features.py
- TestTrainingPolicy
- NativeTinyModel
- JSONLRepairDataset
- load_verified_tokenizer
- test_context_benchmark_resume.py
- adversarial_r3_probe.py
- test_molab_remediation.py
- test_training_recipes.py
- BifocalGemmaForCausalLM
- test_tier3_combinations.py
- test_matrix_schedule_runtime.py
- test_benchmark_prompt_budget.py
- repository_study.py
- contracts.py
- test_sequential_stages.py
- v3/__init__.py
- apply_word_embedding_policy
- RepoContextRouter
- reference_resilient_mcq_parser
- models/checkpoint_transfer.py
- TestGemma4EdgeTopology
- SyntheticRepairDataset
- repository_corpus.py
- TestBifocalGemmaArchitecture
- matrix.py
- configure_probe_training
- pytest
- inspect_python_target
- ASTGraphProjector
- BifocalGemmaModel
- test_training_telemetry_v2.py
- TestNotebookCellParity
- re
- build_joint_visibility
- ast
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- build_native_tiny_model
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- test_v2_workflow.py
- profile_repository_lengths.py
- fetch_collection
- test_v2_fixed_execution.py
- test_context_tasks.py
- optimizers.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_colab_pilot.py
- v2.py
- checkpointing.py
- test_snapshot_verifies_replacement_before_pruning
- stages.py
- test_tier4_scenarios.py
- test_notebook_hf_flow.py
- cache.py
- atomic_json
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- inference_cache.py
- argparse
- protocol.py
- prepare_colab_eval.py
- DummyTrackingTokenizer
- test_fast_pipeline.py
- config.py
- file_digest
- bifocalgemma/__init__.py
- TestOfflineSFT
- test_shared_execution.py

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
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/models/v2/native.py
- `test_telemetry_history_appends_all_active_stages()` --calls--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_prepare_sampled_benchmark_collection_deterministic_100_cases()` --calls--> `prepare_sampled_benchmark_collection()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/benchmark.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (123 total, 43 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.07
Nodes (8): test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_per_model_hf_repo(), test_prepare_sampled_benchmark_collection_deterministic_100_cases(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight(), test_telemetry_history_appends_all_active_stages(), _prepare()

### Community 2 - "stable_fingerprint"
Cohesion: 0.13
Nodes (21): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), read_arrow_rows(), audit_repository_sources() (+13 more)

### Community 3 - "copy"
Cohesion: 0.10
Nodes (13): run_training_memory_preflight(), load_sft_adapters_for_inference(), fit_curated_sft(), prepare_curated_training(), validate_optimizer_config(), build_training_groups(), default_training_profiles(), Float32Alignment (+5 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.05
Nodes (40): fail_second_microbatch(), test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), measure(), digest() (+32 more)

### Community 5 - "repository_sft.py"
Cohesion: 0.24
Nodes (7): tokenize_context_task(), token_profile(), prepare_editing_records(), split_editing_records(), tokenize(), prepare_repository_training(), tokenize_repository_record()

### Community 6 - "contracts.py"
Cohesion: 0.06
Nodes (18): DeveloperAgent, RepairExample, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter (+10 more)

### Community 7 - "test_pretrained_probe_helpers.py"
Cohesion: 0.24
Nodes (5): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library(), test_notebook_sweep_persists_results_per_case()

### Community 8 - "types"
Cohesion: 0.08
Nodes (8): remote_reset_plan(), reset_local_benchmarks(), reset_state(), retained_checkpoints(), test_remote_reset_is_scoped_and_receipt_hashes_match(), checkpoint(), test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), main()

### Community 9 - "remediation.py"
Cohesion: 0.11
Nodes (23): split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+15 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.11
Nodes (7): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 12 - "pathlib"
Cohesion: 0.06
Nodes (17): apply_patch_to_snapshot(), extract_unified_diff(), safe_repo_path(), NativeTaskResult, score_python_patch(), summarize_native_results(), patch_syntax(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch() (+9 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.29
Nodes (12): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+4 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.14
Nodes (14): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+6 more)

### Community 15 - "schedule.py"
Cohesion: 0.11
Nodes (24): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+16 more)

### Community 16 - "fit_offline_sft"
Cohesion: 0.23
Nodes (11): cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), validation_batches() (+3 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.07
Nodes (35): build_policy_optimizer(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind() (+27 more)

### Community 18 - "examples"
Cohesion: 0.12
Nodes (15): TestTokenizerVerification, examples(), prepare(), calibration(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), prepare(), bounded_threads(), test_atomic_status_retries_windows_sharing_violation_without_partial_write() (+7 more)

### Community 19 - "NativeTinyModel"
Cohesion: 0.07
Nodes (5): NormalizedAlignment, build_native_model(), native_config(), NativeMergedAttention, RoutedRepoBatch

### Community 20 - "json"
Cohesion: 0.10
Nodes (21): curated_pool_digest(), load_curated_pool(), audit_split_disjointness(), build_release(), main(), publish_release(), rows(), verify_release() (+13 more)

### Community 21 - "hashlib"
Cohesion: 0.21
Nodes (12): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_run(), gsm8k_prompt_ids(), load_gsm8k(), main() (+4 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (20): completion_metrics(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling() (+12 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.11
Nodes (15): prepare_repository_record(), RepositoryTask, bounded_threads(), CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions (+7 more)

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.13
Nodes (28): code_shingles(), code_signature(), CodeOverlapIndex, curate_paired_sources(), export_pool(), nextcoder_row(), validate_curated_pool_limits(), count_syntax_warnings() (+20 more)

### Community 25 - "configure_memory_efficient_training"
Cohesion: 0.08
Nodes (33): get_v2_experiment(), cached_logits(), configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), fit_repository_sft(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence() (+25 more)

### Community 26 - "TokenizedRepairExample"
Cohesion: 0.10
Nodes (13): ExperimentSpec, Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), run_native_smoke(), run_reference_smoke(), TinyReferenceModel (+5 more)

### Community 27 - "hashlib"
Cohesion: 0.15
Nodes (12): code_tokens(), CodeOverlapMatch, editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool() (+4 more)

### Community 28 - "torch"
Cohesion: 0.12
Nodes (15): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+7 more)

### Community 30 - "tests_e2e/conftest.py"
Cohesion: 0.15
Nodes (6): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 31 - "patch_cross_entropy"
Cohesion: 0.21
Nodes (4): __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients()

### Community 34 - "typing"
Cohesion: 0.05
Nodes (16): ProbeLoRALinear, SwiGLUDualGateBridge, backward_window(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids() (+8 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.36
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "repository.py"
Cohesion: 0.16
Nodes (11): RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), git_environment(), validate_code_eval_snapshots() (+3 more)

### Community 37 - "normalize_context_task"
Cohesion: 0.27
Nodes (7): normalize_context_task(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record(), test_context_license_record_covers_all_visible_paths()

### Community 38 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 42 - "load_verified_tokenizer"
Cohesion: 0.06
Nodes (35): task_prompt(), decode_text(), load_verified_tokenizer(), build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode() (+27 more)

### Community 43 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 44 - "adversarial_r3_probe.py"
Cohesion: 0.11
Nodes (10): extract_mcq_letter(), strict_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases() (+2 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "test_training_recipes.py"
Cohesion: 0.16
Nodes (13): comparison_recipes(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume() (+5 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.07
Nodes (19): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype() (+11 more)

### Community 48 - "test_tier3_combinations.py"
Cohesion: 0.19
Nodes (4): validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 49 - "test_matrix_schedule_runtime.py"
Cohesion: 0.16
Nodes (13): CheckpointRef, make_experiment_spec(), run_matrix_stage(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights() (+5 more)

### Community 50 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+10 more)

### Community 52 - "contracts.py"
Cohesion: 0.09
Nodes (13): RepositoryFile, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, tokenize_repair_example(), test_dependency_completion_removes_target_and_retains_real_helper() (+5 more)

### Community 53 - "test_sequential_stages.py"
Cohesion: 0.17
Nodes (6): TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 54 - "v3/__init__.py"
Cohesion: 0.09
Nodes (17): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example() (+9 more)

### Community 55 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 57 - "reference_resilient_mcq_parser"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 62 - "repository_corpus.py"
Cohesion: 0.08
Nodes (20): ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license() (+12 more)

### Community 64 - "matrix.py"
Cohesion: 0.07
Nodes (21): benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), configure_full_rank_cross_attention(), synchronize_matrix_v3(), resolve_training_contract(), run_matrix_benchmark() (+13 more)

### Community 65 - "configure_probe_training"
Cohesion: 0.22
Nodes (7): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), TestMemoryProbe

### Community 66 - "pytest"
Cohesion: 0.43
Nodes (4): parent(), test_evaluation_and_unlicensed_sources_rejected(), test_grounded_generation_answerability_provenance_and_target_independence(), test_grounded_mutation_rejected()

### Community 67 - "inspect_python_target"
Cohesion: 0.23
Nodes (5): inspect_python_target(), summarize_admitted_warning_records(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples(), test_warning_has_provenance_without_stderr_or_source_mutation()

### Community 70 - "test_training_telemetry_v2.py"
Cohesion: 0.16
Nodes (5): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 74 - "re"
Cohesion: 0.18
Nodes (12): repair_preimage_coverage(), patch_paths(), repository_task_from_row(), code_fragment_signatures(), completion_from_snapshot(), context_pool(), log(), main() (+4 more)

### Community 76 - "build_joint_visibility"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 77 - "ast"
Cohesion: 0.14
Nodes (15): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), source_fingerprints(), WithoutDocstrings (+7 more)

### Community 78 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 79 - "build_native_tiny_model"
Cohesion: 0.15
Nodes (6): build_native_tiny_model(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words()

### Community 81 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+4 more)

### Community 82 - "profile_repository_lengths.py"
Cohesion: 0.38
Nodes (3): distribution(), main(), profile()

### Community 83 - "fetch_collection"
Cohesion: 0.13
Nodes (14): archive_files(), digest(), fetch_collection(), unpack_archive(), pool(), release(), test_archive_source_escape_rejected(), test_existing_local_pool_cannot_bypass_pinned_identity() (+6 more)

### Community 84 - "test_v2_fixed_execution.py"
Cohesion: 0.17
Nodes (8): guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure(), test_preflight_upload_verifies_exact_evidence_without_checkpoint(), test_stage_counters_do_not_leak_and_flex_backward_tiles_cover_head256()

### Community 85 - "test_context_tasks.py"
Cohesion: 0.33
Nodes (4): task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context()

### Community 87 - "optimizers.py"
Cohesion: 0.32
Nodes (3): GrokOrScaleLoRA, orthogonal_direction(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 91 - "test_colab_pilot.py"
Cohesion: 0.16
Nodes (5): _DummyTokenizer, pilot_module(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), prepare_runtime()

### Community 92 - "v2.py"
Cohesion: 0.08
Nodes (28): load_asset_json(), load_local_native_model(), checked_prefix(), recover_remote_run(), _synchronize_once(), synchronize_run(), save_benchmark_progress(), upload_checkpoint() (+20 more)

### Community 96 - "stages.py"
Cohesion: 0.06
Nodes (15): retry_delay(), publish_with_retry(), format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget (+7 more)

### Community 101 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.31
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 104 - "atomic_json"
Cohesion: 0.09
Nodes (19): BenchmarkUploadQueue, retry_delay(), emit(), run_sequential_pipeline(), publish_final(), atomic_json(), StageLedger, append_event() (+11 more)

### Community 107 - "argparse"
Cohesion: 0.12
Nodes (9): compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit(), emit(), main(), validate_all_patches(), main(), inspect() (+1 more)

### Community 108 - "protocol.py"
Cohesion: 0.10
Nodes (14): _extract_match_letter(), fallback_letter_regex(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), _positive_integer(), _runtime_model_identity(), scorer_fingerprint() (+6 more)

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.08
Nodes (15): ensure_assets_on_disk(), FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2() (+7 more)

### Community 114 - "config.py"
Cohesion: 0.07
Nodes (16): BackboneProfile, _pair(), _ref(), validate_experiment_matrix(), EvaluationReport, SWEBenchHarness, main(), describe_initialization() (+8 more)

### Community 115 - "file_digest"
Cohesion: 0.17
Nodes (6): file_digest(), build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 127 - "test_shared_execution.py"
Cohesion: 0.13
Nodes (7): test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_shared_profile_upload_retries_without_losing_local_selection(), publish()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 966 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **43 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `stable_fingerprint`, `copy`, `prepare_colab_pilot.py`, `types`, `pathlib`, `fit_offline_sft`, `shared_execution.py`, `examples`, `json`, `configure_memory_efficient_training`, `load_verified_tokenizer`, `test_training_recipes.py`, `v3/__init__.py`, `matrix.py`, `re`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `test_v2_workflow.py`, `test_v2_fixed_execution.py`, `v2.py`, `test_snapshot_verifies_replacement_before_pruning`, `stages.py`, `file_digest`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `copy`, `repository_sft.py`, `remediation.py`, `shared_execution.py`, `json`, `CodeOverlapIndex`, `hashlib`, `load_verified_tokenizer`, `test_matrix_schedule_runtime.py`, `repository_corpus.py`, `matrix.py`, `re`, `ast`, `fetch_collection`, `v2.py`, `stages.py`, `atomic_json`, `argparse`, `file_digest`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06818181818181818 - nodes in this community are weakly interconnected._
- **Should `stable_fingerprint` be split into smaller, more focused modules?**
  _Cohesion score 0.13068181818181818 - nodes in this community are weakly interconnected._