# Graph Report - bifocalgemma  (2026-11-07)

## Corpus Check
- 289 files · ~361,943 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2282 nodes · 6818 edges · 112 communities (84 shown, 28 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 665 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0bbf2c10`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- TokenizedRepairExample
- argparse
- v2.py
- prepare_colab_pilot.py
- releases.py
- dataclasses
- editing_record
- test_context_benchmark_resume.py
- re
- torch
- test_gsm8k.py
- pathlib
- repository_corpus.py
- prepare_colab_probe.py
- NativeTinyModel
- curation.py
- curation.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- molab_pilot_support.py
- test_data_remediation.py
- remediation.py
- test_tier5_adversarial.py
- test_shared_execution.py
- get_v2_experiment
- gsm8k.py
- benchmark.py
- full_pool.py
- tiled_attention
- GitSnapshotStore
- test_molab_pilot.py
- shared_execution.py
- Tensor
- FeatureNotReadyError
- test_tier1_features.py
- tests_e2e/conftest.py
- load_verified_tokenizer
- modeling_bifocalgemma.py
- prepare_context_data.py
- test_memory_efficient_training.py
- build_native_tiny_model
- adapters.py
- RepositorySnapshot
- curriculum.py
- test_tier4_scenarios.py
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- TokenizedRepairExample
- repository_study.py
- bifocalgemma/__init__.py
- stable_fingerprint
- evaluation/context.py
- test_snapshot_verifies_replacement_before_pruning
- test_training_telemetry_v2.py
- profile_repository_lengths.py
- RepairPrediction
- test_repository_study.py
- test_combination_eval_metrics_to_paper_data_serialization
- wiggly_dashboard.py
- ast
- TestBifocalGemmaArchitecture
- matrix.py
- test_molab_matrix_notebooks.py
- ExperimentSpec
- default_training_profiles
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- sequential.py
- test_tier2_boundary.py
- pytest
- harness.py
- default_training_profiles
- audit_dataset_release.py
- fit_offline_sft
- atomic_json
- cache.py
- run_length_grid
- TestNotebookCellParity
- _DummyTokenizer
- fast_pipeline.py
- telemetry.py
- v2/__init__.py
- replace
- test_colab_probe.py
- BifocalGemmaForCausalLM
- bifocalgemma
- collections
- test_notebook_hf_flow.py
- run_repo_niah_eval
- TestGemma4EdgeTopology
- load_verified_tokenizer
- test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract
- strict_letter
- fit_curated_sft
- test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation
- optimizers.py
- memory_efficient.py
- build_repository_corpus
- test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt
- Float32Alignment
- workflow
- json
- run_batched_context_benchmark
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
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `event()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py
- `save()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (112 total, 28 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "TokenizedRepairExample"
Cohesion: 0.08
Nodes (13): TokenizedRepairExample, RepairCollator, __getattr__(), build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, run_training_memory_preflight(), synthetic_stress_target() (+5 more)

### Community 2 - "argparse"
Cohesion: 0.21
Nodes (5): compare_predictions(), main(), main(), render(), source_bundle()

### Community 3 - "v2.py"
Cohesion: 0.09
Nodes (23): publish_with_retry(), shared_contract(), source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics(), stratified_subset() (+15 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.10
Nodes (24): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset() (+16 more)

### Community 5 - "releases.py"
Cohesion: 0.09
Nodes (25): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+17 more)

### Community 6 - "dataclasses"
Cohesion: 0.08
Nodes (15): validate_experiment_matrix(), extract_token_ast_features(), TokenASTFeatures, main(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents (+7 more)

### Community 7 - "editing_record"
Cohesion: 0.20
Nodes (12): editing_manifest(), editing_record(), git_text_lines(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool() (+4 more)

### Community 8 - "test_context_benchmark_resume.py"
Cohesion: 0.14
Nodes (10): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+2 more)

### Community 9 - "re"
Cohesion: 0.11
Nodes (16): repair_preimage_coverage(), task_prompt(), apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), validate_code_eval_snapshots(), NativeTaskResult (+8 more)

### Community 10 - "torch"
Cohesion: 0.10
Nodes (18): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), FastTrainingConfig, test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload() (+10 more)

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (14): _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+6 more)

### Community 12 - "pathlib"
Cohesion: 0.12
Nodes (7): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "repository_corpus.py"
Cohesion: 0.15
Nodes (13): context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), recognize_license(), resolved_license(), archive(), test_archive_commit_budget_and_static_license() (+5 more)

### Community 14 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 16 - "curation.py"
Cohesion: 0.16
Nodes (23): code_shingles(), code_signature(), CodeOverlapIndex, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families() (+15 more)

### Community 17 - "curation.py"
Cohesion: 0.12
Nodes (7): code_tokens(), CodeOverlapMatch, validate_context_license_record(), test_context_license_record_covers_all_visible_paths(), audit(), main(), completion_from_snapshot()

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.08
Nodes (21): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters() (+13 more)

### Community 20 - "test_data_remediation.py"
Cohesion: 0.18
Nodes (16): merge_lineage(), preference(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed() (+8 more)

### Community 21 - "remediation.py"
Cohesion: 0.10
Nodes (16): token_profile(), count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), audit_rows(), build_remediated_pool(), event_key() (+8 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (21): completion_metrics(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling() (+13 more)

### Community 23 - "test_shared_execution.py"
Cohesion: 0.07
Nodes (10): completed_evidence(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), valid_profile() (+2 more)

### Community 24 - "get_v2_experiment"
Cohesion: 0.14
Nodes (15): get_v2_experiment(), cached_logits(), test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only(), test_encoder_bidirectional_window_boundary_matches_native(), test_v2_alignment_groups_masks_and_gradients() (+7 more)

### Community 25 - "gsm8k.py"
Cohesion: 0.17
Nodes (15): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+7 more)

### Community 26 - "benchmark.py"
Cohesion: 0.13
Nodes (12): decode_text(), format_encoder_context(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids(), InferenceCache (+4 more)

### Community 27 - "full_pool.py"
Cohesion: 0.18
Nodes (9): read_arrow_rows(), split_connected_records(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main(), progress() (+1 more)

### Community 28 - "tiled_attention"
Cohesion: 0.09
Nodes (14): flex_kernel_options(), fused_attention(), attention_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention, TiledMergedAttention (+6 more)

### Community 30 - "test_molab_pilot.py"
Cohesion: 0.16
Nodes (11): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+3 more)

### Community 31 - "shared_execution.py"
Cohesion: 0.08
Nodes (34): build_policy_optimizer(), build_training_groups(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind() (+26 more)

### Community 32 - "Tensor"
Cohesion: 0.10
Nodes (12): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 36 - "FeatureNotReadyError"
Cohesion: 0.10
Nodes (8): restore_checkpoint(), save_checkpoint(), ExperimentSpec, BifocalGemmaModel, build_pretrained_model(), FeatureNotReadyError, distillation_loss(), profile_experiment()

### Community 38 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 39 - "tests_e2e/conftest.py"
Cohesion: 0.14
Nodes (7): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema()

### Community 40 - "load_verified_tokenizer"
Cohesion: 0.20
Nodes (9): tokenize_context_task(), prepare_editing_records(), split_editing_records(), load_verified_tokenizer(), prepare_curated_training(), tokenize(), prepare_repository_training(), tokenize_repository_record() (+1 more)

### Community 41 - "modeling_bifocalgemma.py"
Cohesion: 0.12
Nodes (8): transfer_text_checkpoint(), configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), tensor_bytes(), TestCheckpointTransfer

### Community 42 - "prepare_context_data.py"
Cohesion: 0.15
Nodes (16): normalize_context_task(), export_pool(), curated_pool_digest(), load_curated_pool(), ArchiveSnapshotStore, test_repository_pool_requires_context_license_policy_and_coverage(), test_failed_download_removes_partial_bytes(), audit_pool() (+8 more)

### Community 43 - "test_memory_efficient_training.py"
Cohesion: 0.27
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 44 - "build_native_tiny_model"
Cohesion: 0.12
Nodes (5): build_native_tiny_model(), NativeMergedAttention, TestNativeIntegration, test_last_token_logits_match_full_head(), model()

### Community 45 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 46 - "RepositorySnapshot"
Cohesion: 0.14
Nodes (13): RepairExample, RepositoryFile, RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord (+5 more)

### Community 47 - "curriculum.py"
Cohesion: 0.27
Nodes (5): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 48 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 50 - "TokenizedRepairExample"
Cohesion: 0.08
Nodes (9): ModelOutput, Seq2SeqBatch, ReferenceMergedAttention, TinyReferenceModel, __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy(), TestMergedAttentionReference (+1 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), audit_repository_sources(), read_source_rows(), load_sft_adapters_for_inference(), fetch_tokenizer_assets(), generate_run() (+10 more)

### Community 53 - "stable_fingerprint"
Cohesion: 0.33
Nodes (9): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+1 more)

### Community 54 - "evaluation/context.py"
Cohesion: 0.15
Nodes (7): extract_mcq_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 55 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "test_training_telemetry_v2.py"
Cohesion: 0.24
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 57 - "profile_repository_lengths.py"
Cohesion: 0.28
Nodes (5): test_shared_profile_upload_retries_without_losing_local_selection(), publish(), distribution(), main(), profile()

### Community 58 - "RepairPrediction"
Cohesion: 0.31
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 59 - "test_repository_study.py"
Cohesion: 0.12
Nodes (21): canonical_snapshot_hash(), example_from_record(), prepare_repository_record(), prepare_repository_record_with_snapshot(), repo_identity(), repository_task_from_row(), RepositoryTask, CharacterTokenizer (+13 more)

### Community 60 - "test_combination_eval_metrics_to_paper_data_serialization"
Cohesion: 0.22
Nodes (3): test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 61 - "wiggly_dashboard.py"
Cohesion: 0.13
Nodes (5): format_duration(), LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard

### Community 62 - "ast"
Cohesion: 0.16
Nodes (11): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), main(), module_definitions(), render(), use_embedded_imports() (+3 more)

### Community 64 - "matrix.py"
Cohesion: 0.06
Nodes (17): Bf16FullLinear, _compute_per_row_cross_query(), configure_full_rank_cross_attention(), FullCrossCachedAttention, FullCrossTiledAttention, SwiGLUDualGateBridge, is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter() (+9 more)

### Community 65 - "test_molab_matrix_notebooks.py"
Cohesion: 0.16
Nodes (20): configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_comprehension_targets_stay_inside_baseline_visible_prefix() (+12 more)

### Community 66 - "ExperimentSpec"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 67 - "default_training_profiles"
Cohesion: 0.25
Nodes (6): default_training_profiles(), fit_repository_sft(), test_inference_v2_adapter_roundtrip_and_baseline_group(), test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare()

### Community 68 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 69 - "sequential.py"
Cohesion: 0.11
Nodes (14): reset_local_benchmarks(), synchronize_run(), validate_optimizer_config(), update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence(), run_sequential_pipeline(), publish_final() (+6 more)

### Community 70 - "test_tier2_boundary.py"
Cohesion: 0.15
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 71 - "pytest"
Cohesion: 0.20
Nodes (6): test_exported_worker_has_parity_context_and_exact_adapter_checks(), pilot_module(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_paired_bootstrap_and_cohort_rejection(), _prepare(), prepare_runtime()

### Community 74 - "audit_dataset_release.py"
Cohesion: 0.52
Nodes (4): audit(), emit(), main(), validate_all_patches()

### Community 75 - "fit_offline_sft"
Cohesion: 0.13
Nodes (13): cpu_tree(), file_digest(), fit_offline_sft(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest(), TestOfflineSFT (+5 more)

### Community 76 - "atomic_json"
Cohesion: 0.10
Nodes (23): file_digest(), checked_prefix(), recover_remote_run(), _synchronize_once(), atomic_json(), validate_checkpoint_files(), synchronize_v2(), upload_initial_results() (+15 more)

### Community 78 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 81 - "fast_pipeline.py"
Cohesion: 0.08
Nodes (6): collate_fast_batch(), FastTrainingPipeline, JSONLRepairDataset, run_fast_training(), SyntheticRepairDataset, build_bifocalgemma_v3_optimizer()

### Community 82 - "telemetry.py"
Cohesion: 0.07
Nodes (16): BenchmarkUploadQueue, retry_delay(), emit(), append_event(), loss_chart(), render_dashboard(), TrainingDashboard, test_durable_backoff_shared_across_models_and_final_retries_without_sleep() (+8 more)

### Community 83 - "v2/__init__.py"
Cohesion: 0.06
Nodes (6): CachedAttention, GatedCrossAttentionBridge, RMSNorm, ASTGraphProjector, build_native_model(), RoutedRepoBatch

### Community 84 - "replace"
Cohesion: 0.25
Nodes (5): TestTokenizerVerification, calibration(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), replace(), prepare()

### Community 85 - "test_colab_probe.py"
Cohesion: 0.70
Nodes (3): boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens()

### Community 91 - "collections"
Cohesion: 0.27
Nodes (6): ContextWindow, render_window(), select_repository_context(), _terms(), normalized_code(), score_diagnostics()

### Community 92 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 93 - "run_repo_niah_eval"
Cohesion: 0.16
Nodes (8): build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), score_context_case(), score_repo_niah_case()

### Community 95 - "load_verified_tokenizer"
Cohesion: 0.12
Nodes (6): BackboneProfile, CheckpointRef, _pair(), _ref(), tokenize_repair_example(), BaselineScores

### Community 97 - "strict_letter"
Cohesion: 0.17
Nodes (7): fallback_letter_regex(), strict_letter(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 100 - "test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation"
Cohesion: 0.20
Nodes (5): decoder_prompt(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 101 - "optimizers.py"
Cohesion: 0.32
Nodes (3): GrokOrScaleLoRA, orthogonal_direction(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 102 - "memory_efficient.py"
Cohesion: 0.12
Nodes (6): joint_visibility_tile(), sample_encoder_mass(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), test_atomic_status_retries_windows_sharing_violation_without_partial_write(), test_pilot_subset_is_deterministic_and_stratified(), test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos()

### Community 103 - "build_repository_corpus"
Cohesion: 0.26
Nodes (4): build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 107 - "workflow"
Cohesion: 0.13
Nodes (10): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), workflow(), benchmark(), download(), execution() (+2 more)

### Community 108 - "json"
Cohesion: 0.06
Nodes (15): remote_reset_plan(), reset_state(), evaluate_sft_loss(), validation_batches(), retry_windows_file_operation(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_external_activity_reader_does_not_need_running_kernel(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused() (+7 more)

### Community 109 - "run_batched_context_benchmark"
Cohesion: 0.18
Nodes (9): prompt_ids(), run_context_benchmark(), activity(), DurableClock, prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline() (+1 more)

### Community 111 - "test_training_recipes.py"
Cohesion: 0.15
Nodes (13): comparison_recipes(), run_recipe_comparison(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), test_four_fresh_runs_per_notebook_and_independent_resume() (+5 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 832 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **28 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `TokenizedRepairExample`, `v2.py`, `prepare_colab_pilot.py`, `re`, `curation.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `molab_pilot_support.py`, `test_shared_execution.py`, `get_v2_experiment`, `benchmark.py`, `shared_execution.py`, `prepare_context_data.py`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `stable_fingerprint`, `test_snapshot_verifies_replacement_before_pruning`, `sequential.py`, `telemetry.py`, `run_repo_niah_eval`, `memory_efficient.py`, `workflow`, `json`, `run_batched_context_benchmark`, `test_training_recipes.py`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `fast_pipeline.py`, `modeling_bifocalgemma.py`, `torch`, `bifocalgemma/__init__.py`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Should `TokenizedRepairExample` be split into smaller, more focused modules?**
  _Cohesion score 0.08097165991902834 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `prepare_colab_pilot.py`, `dataclasses`, `editing_record`, `re`, `prepare_colab_probe.py`, `curation.py`, `molab_pilot_support.py`, `shared_execution.py`, `load_verified_tokenizer`, `test_memory_efficient_training.py`, `RepositorySnapshot`, `curriculum.py`, `TokenizedRepairExample`, `test_molab_matrix_notebooks.py`, `default_training_profiles`, `fast_pipeline.py`, `test_colab_probe.py`, `load_verified_tokenizer`, `fit_curated_sft`, `memory_efficient.py`, `json`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._