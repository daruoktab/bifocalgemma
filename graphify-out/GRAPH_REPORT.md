# Graph Report - bifocalgemma  (2026-11-07)

## Corpus Check
- 290 files · ~364,657 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2348 nodes · 6918 edges · 106 communities (76 shown, 30 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 706 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7722d623`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- examples
- v2.py
- prepare_colab_pilot.py
- releases.py
- FeatureNotReadyError
- data/offline_sft.py
- checkpoint.py
- subprocess
- BifocalGemmaConfig
- test_gsm8k.py
- subprocess
- atomic_json
- prepare_colab_probe.py
- NativeTinyModel
- curation.py
- FastTrainingPipeline
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- molab_pilot_support.py
- shared_execution.py
- data/offline_sft.py
- test_tier5_adversarial.py
- test_shared_execution.py
- test_colab_pilot.py
- gsm8k.py
- contracts.py
- full_pool.py
- tiled_attention
- test_memory_efficient_training.py
- source_bundle_bytes
- re
- Tensor
- typing
- GatedCrossAttentionBridge
- TokenizedRepairExample
- test_context_benchmark_resume.py
- test_tier1_features.py
- reference_resilient_mcq_parser
- RepoContextRouter
- test_shared_execution.py
- ast
- build_native_tiny_model
- test_molab_remediation.py
- dataclasses
- test_tier4_scenarios.py
- adapters.py
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- ReferenceMergedAttention
- repository_study.py
- bifocalgemma/__init__.py
- apply_word_embedding_policy
- shutil
- examples
- test_training_telemetry_v2.py
- types
- transfer_text_checkpoint
- ScriptedModel
- PackedTokenizedDataset
- wiggly_dashboard.py
- ast.py
- TestBifocalGemmaArchitecture
- test_molab_matrix_notebooks.py
- test_molab_matrix_notebooks.py
- ASTGraphProjector
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- tests_e2e/conftest.py
- molab_matrix_support.py
- TestConstructionEntrypoints
- default_training_profiles
- argparse
- fit_offline_sft
- matrix.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- run_length_grid
- TestNotebookCellParity
- .__init__
- benchmark_upload.py
- modeling_bifocalgemma.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_repository_study.py
- score_context_case
- TestGemma4EdgeTopology
- config.py
- test_context_tasks.py
- test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation
- shared_execution.py
- repository_corpus.py
- test_v2_fixed_execution.py
- math
- test_v2_workflow.py
- json
- run_context_benchmark
- test_training_recipes.py

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 83 edges
2. `stable_fingerprint()` - 62 edges
3. `TokenizedRepairExample` - 53 edges
4. `load_verified_tokenizer()` - 52 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `RepairCollator` - 40 edges
10. `configure_policy_model()` - 39 edges

## Surprising Connections (you probably didn't know these)
- `test_train_fast_cli_baseline_2b()` --calls--> `main()`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/__main__.py
- `test_fast_training_config_step_reconciliation()` --uses--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()` --calls--> `resolve_optimal_eval_batch_size()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/benchmark.py

## Import Cycles
- None detected.

## Communities (106 total, 30 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "remediation.py"
Cohesion: 0.10
Nodes (24): split_connected_records(), token_profile(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage() (+16 more)

### Community 2 - "examples"
Cohesion: 0.11
Nodes (10): examples(), TestOfflineEditing, TestOfflineSFT, TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored() (+2 more)

### Community 3 - "v2.py"
Cohesion: 0.09
Nodes (24): source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics(), stratified_subset(), download_model(), _execute_v2() (+16 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.10
Nodes (24): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset() (+16 more)

### Community 5 - "releases.py"
Cohesion: 0.09
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 6 - "FeatureNotReadyError"
Cohesion: 0.08
Nodes (14): EvaluationReport, SWEBenchHarness, main(), EncoderMemory, RepositoryCache, describe_initialization(), InitializationPlan, initialize_pretrained_components() (+6 more)

### Community 7 - "data/offline_sft.py"
Cohesion: 0.26
Nodes (10): editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical() (+2 more)

### Community 8 - "checkpoint.py"
Cohesion: 0.13
Nodes (9): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix_v3(), build_bifocalgemma_v3_optimizer(), is_cross_bridge_parameter() (+1 more)

### Community 9 - "subprocess"
Cohesion: 0.10
Nodes (17): RepositoryFile, apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), patch_paths(), safe_repo_path(), GitSnapshotStore, NativeTaskResult (+9 more)

### Community 10 - "BifocalGemmaConfig"
Cohesion: 0.11
Nodes (11): BifocalGemmaConfig, _resolve_backbone_weights(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_enable_memory_efficient_training_and_softcapping(), test_fast_pipeline_batch_edge_cases(), test_fast_training_config_step_reconciliation() (+3 more)

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 12 - "subprocess"
Cohesion: 0.10
Nodes (7): probe_prompt_neutrality_scan(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main()

### Community 13 - "atomic_json"
Cohesion: 0.15
Nodes (13): synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint(), atomic_json(), StageLedger, test_reset_archives_evaluation_once_without_changing_training() (+5 more)

### Community 14 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 15 - "NativeTinyModel"
Cohesion: 0.12
Nodes (5): ModelOutput, NormalizedAlignment, NativeTinyModel, chunked_linear_cross_entropy(), checkpoint()

### Community 16 - "curation.py"
Cohesion: 0.08
Nodes (36): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+28 more)

### Community 17 - "FastTrainingPipeline"
Cohesion: 0.11
Nodes (6): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_baseline_2b(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.09
Nodes (18): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), test_upload_retry_verification_and_base_prefix_only(), measure() (+10 more)

### Community 20 - "shared_execution.py"
Cohesion: 0.17
Nodes (11): calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe(), representative_cohort() (+3 more)

### Community 21 - "data/offline_sft.py"
Cohesion: 0.09
Nodes (23): normalize_context_task(), repair_preimage_coverage(), tokenize_context_task(), curated_pool_digest(), load_curated_pool(), prepare_editing_records(), split_editing_records(), validate_context_license_record() (+15 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.06
Nodes (26): completion_metrics(), extract_mcq_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_redos_catastrophic_backtracking(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles() (+18 more)

### Community 23 - "test_shared_execution.py"
Cohesion: 0.15
Nodes (11): seal_report(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), test_shared_profile_upload_retries_without_losing_local_selection(), publish() (+3 more)

### Community 24 - "test_colab_pilot.py"
Cohesion: 0.18
Nodes (5): _DummyTokenizer, pilot_module(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order()

### Community 25 - "gsm8k.py"
Cohesion: 0.14
Nodes (16): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+8 more)

### Community 26 - "contracts.py"
Cohesion: 0.09
Nodes (15): DeveloperAgent, CheckpointRef, RepairExample, RepairPrediction, format_encoder_context(), load_verified_tokenizer(), tokenize_repair_example(), write_swebench_predictions() (+7 more)

### Community 27 - "full_pool.py"
Cohesion: 0.16
Nodes (18): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), read_arrow_rows(), audit_repository_sources() (+10 more)

### Community 28 - "tiled_attention"
Cohesion: 0.08
Nodes (16): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+8 more)

### Community 29 - "test_memory_efficient_training.py"
Cohesion: 0.20
Nodes (13): run_training_memory_preflight(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+5 more)

### Community 30 - "source_bundle_bytes"
Cohesion: 0.17
Nodes (10): test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render() (+2 more)

### Community 31 - "re"
Cohesion: 0.17
Nodes (10): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), refresh_patch_targets(), path_license_evidence() (+2 more)

### Community 32 - "Tensor"
Cohesion: 0.10
Nodes (12): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 34 - "typing"
Cohesion: 0.06
Nodes (13): Bf16FullLinear, _compute_per_row_cross_query(), configure_full_rank_cross_attention(), FullCrossCachedAttention, FullCrossTiledAttention, batched_greedy_decode(), infer_benchmark_weight_tier(), SwiGLUDualGateBridge (+5 more)

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.08
Nodes (11): Seq2SeqBatch, TokenizedRepairExample, RepairCollator, __getattr__(), run_reference_smoke(), TinyReferenceModel, synthetic_stress_target(), patch_cross_entropy() (+3 more)

### Community 37 - "test_context_benchmark_resume.py"
Cohesion: 0.14
Nodes (10): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization() (+2 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (13): task_prompt(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f2_mcq_token_budget_specification(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval() (+5 more)

### Community 39 - "reference_resilient_mcq_parser"
Cohesion: 0.15
Nodes (6): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement()

### Community 41 - "test_shared_execution.py"
Cohesion: 0.15
Nodes (6): calibration(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight()

### Community 43 - "ast"
Cohesion: 0.24
Nodes (6): main(), module_definitions(), render(), use_embedded_imports(), main(), render_eval_script()

### Community 44 - "build_native_tiny_model"
Cohesion: 0.09
Nodes (8): build_native_tiny_model(), native_config(), NativeMergedAttention, run_native_smoke(), TestNativeIntegration, TestMemoryProbe, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_embedded_runtime_extraction_from_fast_notebook(), test_embedded_runtime_extraction_from_real_matrix_notebook(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_legacy_checkpoint_vs_clean_resume() (+2 more)

### Community 46 - "dataclasses"
Cohesion: 0.17
Nodes (8): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract, TestTokenizerVerification

### Community 47 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 48 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 50 - "ReferenceMergedAttention"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+10 more)

### Community 53 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 54 - "shutil"
Cohesion: 0.29
Nodes (4): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), main(), render(), source_bundle()

### Community 55 - "examples"
Cohesion: 0.11
Nodes (9): bounded_threads(), test_atomic_status_retries_windows_sharing_violation_without_partial_write(), replace(), test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_pilot_subset_is_deterministic_and_stratified(), test_snapshot_verifies_replacement_before_pruning() (+1 more)

### Community 56 - "test_training_telemetry_v2.py"
Cohesion: 0.20
Nodes (4): __getattr__(), test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 57 - "types"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 61 - "wiggly_dashboard.py"
Cohesion: 0.09
Nodes (11): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog (+3 more)

### Community 64 - "test_molab_matrix_notebooks.py"
Cohesion: 0.07
Nodes (11): format_query_injected_context(), safe_tokenize_context_task(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo() (+3 more)

### Community 65 - "test_molab_matrix_notebooks.py"
Cohesion: 0.08
Nodes (33): get_v2_experiment(), cached_logits(), configure_memory_efficient_training(), configure_policy_model(), default_training_profiles(), run_recipe_comparison(), unified_recipe(), fit_repository_sft() (+25 more)

### Community 70 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 71 - "molab_matrix_support.py"
Cohesion: 0.13
Nodes (6): test_exported_worker_has_parity_context_and_exact_adapter_checks(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), _prepare(), prepare_runtime()

### Community 74 - "argparse"
Cohesion: 0.11
Nodes (12): compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit(), emit(), main(), validate_all_patches(), main(), inspect() (+4 more)

### Community 75 - "fit_offline_sft"
Cohesion: 0.17
Nodes (15): cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint() (+7 more)

### Community 76 - "matrix.py"
Cohesion: 0.09
Nodes (16): latest_checkpoint(), resolve_optimal_eval_batch_size(), build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), run_matrix_benchmark(), run_matrix_stage() (+8 more)

### Community 78 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 81 - ".__init__"
Cohesion: 0.15
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 82 - "benchmark_upload.py"
Cohesion: 0.10
Nodes (8): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos()

### Community 83 - "modeling_bifocalgemma.py"
Cohesion: 0.09
Nodes (11): export_matrix_pretrained_model(), configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), tensor_bytes(), test_bifocalgemma_merge_and_unload() (+3 more)

### Community 91 - "test_repository_study.py"
Cohesion: 0.13
Nodes (24): RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), example_from_record(), prepare_repository_record() (+16 more)

### Community 93 - "score_context_case"
Cohesion: 0.09
Nodes (17): decode_text(), build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), prompt_ids() (+9 more)

### Community 95 - "config.py"
Cohesion: 0.09
Nodes (10): restore_checkpoint(), save_checkpoint(), BackboneProfile, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), BifocalGemmaModel (+2 more)

### Community 97 - "test_context_tasks.py"
Cohesion: 0.18
Nodes (7): fallback_letter_regex(), strict_letter(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 100 - "test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 101 - "shared_execution.py"
Cohesion: 0.09
Nodes (22): retry_delay(), build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), build_training_groups(), benchmark_candidate(), probe_optimizer(), publish_profile() (+14 more)

### Community 103 - "repository_corpus.py"
Cohesion: 0.09
Nodes (17): ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), prepare_repository_corpus(), read_text_archive(), RepositoryPreparationLedger, repo_identity() (+9 more)

### Community 105 - "test_v2_fixed_execution.py"
Cohesion: 0.20
Nodes (6): guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure()

### Community 106 - "math"
Cohesion: 0.10
Nodes (7): CachedAttention, validate_optimizer_config(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence(), test_boundary_zero_identifier_f1_safeguard()

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (11): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+3 more)

### Community 108 - "json"
Cohesion: 0.09
Nodes (19): file_digest(), open_packed_cache(), remote_reset_plan(), reset_local_benchmarks(), reset_state(), checked_prefix(), recover_remote_run(), _synchronize_once() (+11 more)

### Community 109 - "run_context_benchmark"
Cohesion: 0.21
Nodes (7): run_context_benchmark(), activity(), DurableClock, prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), test_prepare_sampled_benchmark_collection_deterministic_100_cases()

### Community 111 - "test_training_recipes.py"
Cohesion: 0.13
Nodes (15): comparison_recipes(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload() (+7 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 877 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `v2.py`, `prepare_colab_pilot.py`, `checkpoint.py`, `subprocess`, `NativeTinyModel`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `shared_execution.py`, `data/offline_sft.py`, `test_shared_execution.py`, `full_pool.py`, `typing`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `examples`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `fit_offline_sft`, `matrix.py`, `benchmark_upload.py`, `score_context_case`, `shared_execution.py`, `test_v2_fixed_execution.py`, `test_v2_workflow.py`, `json`, `run_context_benchmark`, `test_training_recipes.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08858858858858859 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `BifocalGemmaConfig`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `test_molab_remediation.py`, `.__init__`, `FastTrainingPipeline`, `modeling_bifocalgemma.py`, `bifocalgemma/__init__.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1036036036036036 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `full_pool.py` to `remediation.py`, `v2.py`, `releases.py`, `shared_execution.py`, `repository_corpus.py`, `data/offline_sft.py`, `argparse`, `json`, `run_context_benchmark`, `atomic_json`, `curation.py`, `shared_execution.py`, `data/offline_sft.py`, `test_shared_execution.py`, `gsm8k.py`, `score_context_case`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._