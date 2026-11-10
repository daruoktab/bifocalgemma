# Graph Report - bifocalgemma  (2026-11-10)

## Corpus Check
- 292 files · ~368,810 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 19 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2441 nodes · 7082 edges · 120 communities (89 shown, 31 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 780 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `80781e61`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- molab_pilot_support.py
- curation.py
- sequential.py
- prepare_colab_pilot.py
- gsm8k.py
- config.py
- math
- test_v2_fixed_execution.py
- hashlib
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- v3/__init__.py
- test_molab_pilot.py
- NativeTinyModel
- repository_corpus.py
- shared_execution.py
- fit_offline_sft
- native.py
- tests_e2e/conftest.py
- os
- test_tier5_adversarial.py
- test_repository_study.py
- build_repository_corpus
- configure_memory_efficient_training
- run_matrix_stage
- is_cross_bridge_parameter
- torch
- FastTrainingPipeline
- test_v2_fixed_execution.py
- greedy_decode
- test_molab_matrix_notebooks.py
- v3/objectives.py
- argparse
- test_pretrained_hf_model.py
- load_verified_tokenizer
- test_tier1_features.py
- adversarial_r3_probe.py
- test_tier2_boundary.py
- fast_pipeline.py
- TestGemma4EdgeTopology
- test_training_telemetry_v2.py
- json
- test_molab_remediation.py
- run_repo_niah_eval
- BifocalGemmaForCausalLM
- build_repository_corpus
- profile_repository_lengths.py
- default_training_profiles
- repository_study.py
- native_config
- modeling_bifocalgemma.py
- run_length_grid
- TokenizedRepairExample
- adapters.py
- test_tier4_scenarios.py
- RepoContextRouter
- benchmark_upload.py
- test_repository_corpus.py
- remediation.py
- sequential.py
- TestBifocalGemmaArchitecture
- matrix.py
- test_training_recipes.py
- contracts.py
- examples
- test_combination_eval_metrics_to_paper_data_serialization
- pathlib
- build_joint_visibility
- RepositorySnapshot
- patch_cross_entropy
- patch_matrix_tokenization
- CodeOverlapIndex
- build_native_tiny_model
- test_v2_workflow.py
- releases.py
- test_data_releases.py
- bifocalgemma/__init__.py
- SwiGLUDualGateBridge
- GatedCrossAttentionBridge
- bifocalgemma
- v2_preflight.py
- GrokOrScaleLoRA
- atomic_json
- configure_probe_training
- telemetry.py
- FastTrainingConfig
- greedy_decode
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- TiledEncoderAttention
- release_gpu_profile.py
- load_curated_pool
- test_v2_workflow.py
- test_benchmark_prompt_budget.py
- replace
- ASTGraphProjector
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- hashlib
- pytest

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 83 edges
2. `stable_fingerprint()` - 65 edges
3. `load_verified_tokenizer()` - 55 edges
4. `TokenizedRepairExample` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `run_matrix_training()` - 40 edges
10. `RepairCollator` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `save_report()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_pilot.py → src/bifocalgemma/training/stages.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/source_identity.py

## Import Cycles
- None detected.

## Communities (120 total, 31 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "molab_pilot_support.py"
Cohesion: 0.09
Nodes (18): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), test_upload_retry_verification_and_base_prefix_only(), measure() (+10 more)

### Community 2 - "curation.py"
Cohesion: 0.13
Nodes (10): read_arrow_rows(), split_connected_records(), token_profile(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), prepare_full_corpus() (+2 more)

### Community 3 - "sequential.py"
Cohesion: 0.11
Nodes (13): remote_reset_plan(), reset_local_benchmarks(), reset_state(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint() (+5 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.10
Nodes (24): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset() (+16 more)

### Community 5 - "gsm8k.py"
Cohesion: 0.16
Nodes (15): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+7 more)

### Community 6 - "config.py"
Cohesion: 0.04
Nodes (26): DeveloperAgent, restore_checkpoint(), save_checkpoint(), ExperimentSpec, validate_experiment_matrix(), extract_token_ast_features(), TokenASTFeatures, SWEBenchHarness (+18 more)

### Community 7 - "math"
Cohesion: 0.21
Nodes (6): build_training_groups(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence(), run_recipe_comparison()

### Community 8 - "test_v2_fixed_execution.py"
Cohesion: 0.13
Nodes (9): seal_report(), completed_evidence(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch() (+1 more)

### Community 9 - "hashlib"
Cohesion: 0.10
Nodes (14): map_nextcoder_to_bifocal(), normalize_context_task(), curated_pool_digest(), editing_record(), git_text_lines(), task(), test_dependency_completion_removes_target_and_retains_real_helper(), test_eval_role_and_invisible_localization_target_rejected() (+6 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.13
Nodes (5): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (14): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+6 more)

### Community 12 - "subprocess"
Cohesion: 0.13
Nodes (7): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "v3/__init__.py"
Cohesion: 0.26
Nodes (5): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.20
Nodes (9): test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render(), source_bundle() (+1 more)

### Community 16 - "repository_corpus.py"
Cohesion: 0.09
Nodes (26): repair_preimage_coverage(), task_prompt(), tokenize_context_task(), apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), patch_paths(), safe_repo_path() (+18 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.19
Nodes (10): benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe() (+2 more)

### Community 18 - "fit_offline_sft"
Cohesion: 0.14
Nodes (15): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest() (+7 more)

### Community 19 - "native.py"
Cohesion: 0.11
Nodes (3): NormalizedAlignment, build_native_model(), RoutedRepoBatch

### Community 20 - "tests_e2e/conftest.py"
Cohesion: 0.14
Nodes (7): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema()

### Community 21 - "os"
Cohesion: 0.21
Nodes (7): ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (19): completion_metrics(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling() (+11 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.09
Nodes (23): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), prepare_repository_record(), prepare_repository_record_with_snapshot(), RepositoryTask (+15 more)

### Community 24 - "build_repository_corpus"
Cohesion: 0.19
Nodes (7): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples(), test_warning_has_provenance_without_stderr_or_source_mutation()

### Community 25 - "configure_memory_efficient_training"
Cohesion: 0.09
Nodes (27): get_v2_experiment(), cached_logits(), InferenceCache, configure_memory_efficient_training(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), bounded_threads(), examples() (+19 more)

### Community 26 - "run_matrix_stage"
Cohesion: 0.21
Nodes (8): prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), run_matrix_stage(), test_run_matrix_stage_execution_batch2_and_frozen_decoder_invariance(), test_save_and_restore_matrix_checkpoint_and_prune(), test_stage1_to_stage2_optimizer_state_transfer()

### Community 27 - "is_cross_bridge_parameter"
Cohesion: 0.19
Nodes (6): build_bifocalgemma_v3_optimizer(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), configure_matrix_stage(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_v3_partitioned_optimizer_weight_decay_invariants()

### Community 28 - "torch"
Cohesion: 0.16
Nodes (13): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, sample_encoder_mass() (+5 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.07
Nodes (3): collate_fast_batch(), FastTrainingPipeline, run_fast_training()

### Community 30 - "test_v2_fixed_execution.py"
Cohesion: 0.14
Nodes (10): guard(), examples(), fixture_guard(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure() (+2 more)

### Community 31 - "greedy_decode"
Cohesion: 0.20
Nodes (6): greedy_decode(), GreedyGeneration, batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.11
Nodes (11): make_experiment_spec(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_configure_full_rank_cross_attention_v3_specs(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast() (+3 more)

### Community 34 - "v3/objectives.py"
Cohesion: 0.10
Nodes (12): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 35 - "argparse"
Cohesion: 0.16
Nodes (7): compare_predictions(), main(), main(), render(), declared_limit_example(), main(), render()

### Community 36 - "test_pretrained_hf_model.py"
Cohesion: 0.14
Nodes (8): Seq2SeqBatch, FastTrainingConfig, test_bifocalgemma_config_initialization_and_serialization(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_training_config_step_reconciliation(), test_fast_training_pipeline_baseline_2b(), test_train_fast_cli_baseline_2b()

### Community 37 - "load_verified_tokenizer"
Cohesion: 0.09
Nodes (17): BackboneProfile, CheckpointRef, _pair(), _ref(), prepare_editing_records(), split_editing_records(), decode_text(), load_verified_tokenizer() (+9 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f2_mcq_token_budget_specification(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.11
Nodes (10): extract_mcq_letter(), fallback_letter_regex(), strict_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), verify_gate_round_2_resolved_cases() (+2 more)

### Community 40 - "test_tier2_boundary.py"
Cohesion: 0.15
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 41 - "fast_pipeline.py"
Cohesion: 0.11
Nodes (6): ensure_assets_on_disk(), load_asset_json(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 43 - "test_training_telemetry_v2.py"
Cohesion: 0.10
Nodes (5): TestNotebookCellParity, test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 44 - "json"
Cohesion: 0.11
Nodes (18): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), audit_repository_sources(), main() (+10 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "run_repo_niah_eval"
Cohesion: 0.11
Nodes (9): build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), make_distractor_python_chunk(), run_repo_niah_eval(), score_repo_niah_case(), probe_reponiah_negation_coverage(), test_adversarial_reponiah_deterministic_reproducibility() (+1 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.07
Nodes (14): BifocalGemmaConfig, BifocalGemmaForCausalLM, _resolve_backbone_weights(), ProbeLoRALinear, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype() (+6 more)

### Community 48 - "build_repository_corpus"
Cohesion: 0.26
Nodes (4): build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 49 - "profile_repository_lengths.py"
Cohesion: 0.24
Nodes (5): test_shared_profile_upload_retries_without_losing_local_selection(), publish(), distribution(), main(), profile()

### Community 50 - "default_training_profiles"
Cohesion: 0.19
Nodes (5): default_training_profiles(), fit_repository_sft(), make_identity(), test_inference_v2_adapter_roundtrip_and_baseline_group(), TestTrainingPolicy

### Community 51 - "repository_study.py"
Cohesion: 0.18
Nodes (20): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), GitSnapshotStore, read_source_rows(), fetch_tokenizer_assets() (+12 more)

### Community 52 - "native_config"
Cohesion: 0.20
Nodes (3): native_config(), NativeMergedAttention, version()

### Community 53 - "modeling_bifocalgemma.py"
Cohesion: 0.11
Nodes (4): CachedAttention, clamped_triton_kernel_options(), critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 54 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 55 - "TokenizedRepairExample"
Cohesion: 0.13
Nodes (8): TokenizedRepairExample, RepairCollator, __getattr__(), PackedTokenizedDataset, synthetic_stress_target(), TestDataAndLoss, test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes(), test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 59 - "benchmark_upload.py"
Cohesion: 0.09
Nodes (13): BenchmarkUploadQueue, retry_delay(), publish_with_retry(), resolve_shared_profile(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden() (+5 more)

### Community 60 - "test_repository_corpus.py"
Cohesion: 0.22
Nodes (11): validate_context_license_record(), patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch() (+3 more)

### Community 61 - "remediation.py"
Cohesion: 0.13
Nodes (21): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+13 more)

### Community 62 - "sequential.py"
Cohesion: 0.11
Nodes (9): run_training_memory_preflight(), load_sft_adapters_for_inference(), evaluate_sft_loss(), validation_batches(), resource_snapshot(), TrainingDashboard, encoder_diagnostics(), stratified_subset() (+1 more)

### Community 64 - "matrix.py"
Cohesion: 0.07
Nodes (35): fetch_collection(), prepare_curated_training(), configure_policy_model(), unified_recipe(), shared_contract(), source_fingerprints(), latest_checkpoint(), append_event() (+27 more)

### Community 65 - "test_training_recipes.py"
Cohesion: 0.14
Nodes (14): comparison_recipes(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume() (+6 more)

### Community 66 - "contracts.py"
Cohesion: 0.07
Nodes (9): EvaluationReport, ModelOutput, RepairPrediction, write_swebench_predictions(), apply_word_embedding_policy(), EmbeddingPolicyResult, TinyReferenceModel, TestWordEmbeddingPolicy (+1 more)

### Community 67 - "examples"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 69 - "test_combination_eval_metrics_to_paper_data_serialization"
Cohesion: 0.22
Nodes (3): test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 70 - "pathlib"
Cohesion: 0.08
Nodes (16): file_digest(), checked_prefix(), recover_remote_run(), _synchronize_once(), retained_checkpoints(), retry_windows_file_operation(), validate_checkpoint_files(), synchronize_preflight() (+8 more)

### Community 72 - "build_joint_visibility"
Cohesion: 0.21
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 74 - "RepositorySnapshot"
Cohesion: 0.12
Nodes (15): RepairExample, RepositoryFile, RepositorySnapshot, example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash() (+7 more)

### Community 76 - "patch_cross_entropy"
Cohesion: 0.16
Nodes (3): __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy()

### Community 77 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 78 - "CodeOverlapIndex"
Cohesion: 0.18
Nodes (17): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), test_cross_source_duplicates_and_nextcoder_variants() (+9 more)

### Community 79 - "build_native_tiny_model"
Cohesion: 0.12
Nodes (7): transfer_text_checkpoint(), build_native_tiny_model(), fit_curated_sft(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head()

### Community 81 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+4 more)

### Community 82 - "releases.py"
Cohesion: 0.16
Nodes (13): archive_files(), audit_split_disjointness(), build_release(), digest(), main(), rows(), unpack_archive(), validate_code_eval_snapshots() (+5 more)

### Community 83 - "test_data_releases.py"
Cohesion: 0.15
Nodes (12): publish_release(), verify_release(), pool(), release(), test_archive_source_escape_rejected(), test_eval_public_and_gold_separate(), test_existing_local_pool_cannot_bypass_pinned_identity(), test_fetch_pinned_pool_roundtrip_and_reuse() (+4 more)

### Community 91 - "v2_preflight.py"
Cohesion: 0.21
Nodes (8): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), select_execution(), emit(), save()

### Community 92 - "GrokOrScaleLoRA"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 93 - "atomic_json"
Cohesion: 0.10
Nodes (18): decoder_prompt(), prompt_ids(), run_context_benchmark(), activity(), atomic_json(), DurableClock, prepare_sampled_benchmark_collection(), run_batched_context_benchmark() (+10 more)

### Community 95 - "configure_probe_training"
Cohesion: 0.13
Nodes (10): export_matrix_pretrained_model(), configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), tensor_bytes(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_export_matrix_pretrained_model() (+2 more)

### Community 96 - "telemetry.py"
Cohesion: 0.09
Nodes (10): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard (+2 more)

### Community 97 - "FastTrainingConfig"
Cohesion: 0.06
Nodes (9): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule() (+1 more)

### Community 100 - "greedy_decode"
Cohesion: 0.33
Nodes (3): greedy_decode(), ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards()

### Community 101 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 105 - "release_gpu_profile.py"
Cohesion: 0.27
Nodes (8): publish_profile(), read_remote_profile(), validate_profile(), verify_report(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), check_evidence(), main()

### Community 106 - "load_curated_pool"
Cohesion: 0.16
Nodes (13): export_pool(), load_curated_pool(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_frozen_budget_matches_all_six_controls(), test_frozen_pool_verifies_checksums_before_use(), test_invalid_requested_pool_budget_is_rejected() (+5 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.07
Nodes (24): editing_manifest(), repository_editing_record(), limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer (+16 more)

### Community 108 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 109 - "replace"
Cohesion: 0.20
Nodes (7): test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), calibration(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), replace(), prepare()

### Community 114 - "hashlib"
Cohesion: 0.16
Nodes (10): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), sanitize_posix_path(), main(), module_definitions(), render(), use_embedded_imports(), main() (+2 more)

### Community 115 - "pytest"
Cohesion: 0.11
Nodes (11): test_exported_worker_has_parity_context_and_exact_adapter_checks(), _DummyTokenizer, pilot_module(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation() (+3 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 922 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `sequential.py`, `prepare_colab_pilot.py`, `hashlib`, `repository_corpus.py`, `shared_execution.py`, `os`, `test_v2_fixed_execution.py`, `json`, `run_repo_niah_eval`, `benchmark_upload.py`, `sequential.py`, `matrix.py`, `test_training_recipes.py`, `examples`, `pathlib`, `test_v2_workflow.py`, `v2_preflight.py`, `release_gpu_profile.py`, `load_curated_pool`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `molab_pilot_support.py`, `prepare_colab_pilot.py`, `config.py`, `v3/__init__.py`, `repository_corpus.py`, `fit_offline_sft`, `native.py`, `configure_memory_efficient_training`, `run_matrix_stage`, `torch`, `FastTrainingPipeline`, `test_molab_matrix_notebooks.py`, `argparse`, `load_verified_tokenizer`, `fast_pipeline.py`, `json`, `sequential.py`, `contracts.py`, `build_native_tiny_model`, `v2_preflight.py`, `configure_probe_training`, `load_curated_pool`, `test_v2_workflow.py`, `pytest`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `molab_pilot_support.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08902439024390243 - nodes in this community are weakly interconnected._
- **Why does `TestGemma4EdgeTopology` connect `TestGemma4EdgeTopology` to `native.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._