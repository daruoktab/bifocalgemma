# Graph Report - bifocalgemma  (2026-11-07)

## Corpus Check
- 290 files · ~364,657 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2349 nodes · 6918 edges · 108 communities (83 shown, 25 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 708 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6e135dcc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- test_data_remediation.py
- examples
- _execute_v2
- prepare_colab_pilot.py
- releases.py
- contracts.py
- data/offline_sft.py
- torch
- subprocess
- BifocalGemmaConfig
- test_gsm8k.py
- subprocess
- stable_fingerprint
- prepare_colab_probe.py
- NativeTinyModel
- curation.py
- FastTrainingPipeline
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- molab_pilot_support.py
- shared_execution.py
- prepare_context_data.py
- test_tier5_adversarial.py
- test_shared_execution.py
- remediation.py
- gsm8k.py
- tokenization.py
- full_pool.py
- tiled_attention
- test_pretrained_hf_model.py
- hashlib
- re
- Tensor
- typing
- GatedCrossAttentionBridge
- TokenizedRepairExample
- repository_corpus.py
- test_tier1_features.py
- memory_probe.py
- typing
- extract_mcq_letter
- adversarial_r3_probe.py
- v2_preflight.py
- build_native_tiny_model
- test_molab_remediation.py
- RepositorySnapshot
- test_tier4_scenarios.py
- optimizers.py
- stages.py
- Seq2SeqBatch
- repository_study.py
- curation.py
- apply_word_embedding_policy
- JSONLRepairDataset
- examples
- test_training_telemetry_v2.py
- pathlib
- transfer_text_checkpoint
- release_gpu_profile.py
- PackedTokenizedDataset
- wiggly_dashboard.py
- data/context.py
- TestBifocalGemmaArchitecture
- test_molab_matrix_notebooks.py
- test_molab_matrix_notebooks.py
- GitSnapshotStore
- strict_letter
- tests_e2e/conftest.py
- molab_matrix_support.py
- __main__.py
- default_training_profiles
- argparse
- fit_offline_sft
- random
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- run_length_grid
- TestNotebookCellParity
- SwiGLUDualGateBridge
- SyntheticRepairDataset
- benchmark_upload.py
- modeling_bifocalgemma.py
- test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract
- BifocalGemmaForCausalLM
- bifocalgemma
- test_repository_study.py
- score_context_case
- TestGemma4EdgeTopology
- config.py
- format_stage_display
- context_tasks.py
- shared_execution.py
- test_repository_corpus.py
- test_v2_fixed_execution.py
- v2.py
- test_v2_workflow.py
- os
- atomic_json
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
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_prepare_sampled_benchmark_collection_deterministic_100_cases()` --calls--> `prepare_sampled_benchmark_collection()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/benchmark.py
- `test_train_fast_cli_baseline_2b()` --calls--> `main()`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/__main__.py

## Import Cycles
- None detected.

## Communities (108 total, 25 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "test_data_remediation.py"
Cohesion: 0.24
Nodes (13): merge_lineage(), preference(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed() (+5 more)

### Community 2 - "examples"
Cohesion: 0.13
Nodes (10): examples(), TestOfflineEditing, TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_training_resume_keeps_elapsed_from_step_five_through_ten() (+2 more)

### Community 3 - "_execute_v2"
Cohesion: 0.07
Nodes (25): latest_checkpoint(), StageLedger, benchmark_directory_name(), benchmark_jobs(), download_model(), _execute_v2(), save_guard(), snapshot() (+17 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.08
Nodes (30): evaluate_sft_loss(), validation_batches(), encoder_diagnostics(), TestTokenizerVerification, test_atomic_status_retries_windows_sharing_violation_without_partial_write(), replace(), build_full_cross_cohorts(), build_source_copy_probes() (+22 more)

### Community 5 - "releases.py"
Cohesion: 0.09
Nodes (26): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+18 more)

### Community 6 - "contracts.py"
Cohesion: 0.06
Nodes (18): DeveloperAgent, EvaluationReport, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter (+10 more)

### Community 7 - "data/offline_sft.py"
Cohesion: 0.13
Nodes (18): normalize_context_task(), curated_pool_digest(), editing_manifest(), load_curated_pool(), prepare_editing_records(), repository_editing_record(), split_editing_records(), validate_context_license_record() (+10 more)

### Community 8 - "torch"
Cohesion: 0.08
Nodes (8): CachedAttention, NormalizedAlignment, build_joint_visibility(), build_native_model(), chunked_linear_cross_entropy(), test_notebook_stress_helpers_match_shared_library(), test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion()

### Community 9 - "subprocess"
Cohesion: 0.15
Nodes (13): repair_preimage_coverage(), apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), NativeTaskResult, score_python_patch(), summarize_native_results() (+5 more)

### Community 10 - "BifocalGemmaConfig"
Cohesion: 0.18
Nodes (4): BifocalGemmaConfig, _resolve_backbone_weights(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 12 - "subprocess"
Cohesion: 0.19
Nodes (6): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), validate_all_patches(), main(), main(), run_cmd(), main()

### Community 13 - "stable_fingerprint"
Cohesion: 0.27
Nodes (9): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+1 more)

### Community 14 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 15 - "NativeTinyModel"
Cohesion: 0.11
Nodes (4): ModelOutput, native_config(), NativeMergedAttention, NativeTinyModel

### Community 16 - "curation.py"
Cohesion: 0.16
Nodes (22): code_shingles(), CodeOverlapIndex, curate_paired_sources(), export_pool(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_cross_source_duplicates_and_nextcoder_variants() (+14 more)

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.10
Nodes (16): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), digest() (+8 more)

### Community 20 - "shared_execution.py"
Cohesion: 0.24
Nodes (8): calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe(), representative_cohort()

### Community 21 - "prepare_context_data.py"
Cohesion: 0.20
Nodes (10): split_connected_records(), code_fragment_signatures(), completion_from_snapshot(), context_pool(), log(), main(), mixed_pool(), prepare_core() (+2 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.10
Nodes (12): test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_redos_scaling(), test_adversarial_metrics_disjoint_identifiers(), test_adversarial_metrics_identical_sequences(), test_adversarial_metrics_keyword_identifier_inflation(), test_adversarial_metrics_multiline_diff_inflation(), test_adversarial_metrics_pure_operator_zero_division_guard() (+4 more)

### Community 23 - "test_shared_execution.py"
Cohesion: 0.11
Nodes (12): seal_report(), completed_evidence(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), test_profile_publish_parent_guard_keeps_concurrent_winner() (+4 more)

### Community 24 - "remediation.py"
Cohesion: 0.11
Nodes (16): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure() (+8 more)

### Community 25 - "gsm8k.py"
Cohesion: 0.19
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 26 - "tokenization.py"
Cohesion: 0.11
Nodes (11): format_encoder_context(), tokenize_repair_example(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids(), cached_logits(), InferenceCache (+3 more)

### Community 27 - "full_pool.py"
Cohesion: 0.17
Nodes (11): read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+3 more)

### Community 28 - "tiled_attention"
Cohesion: 0.06
Nodes (29): fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention, TiledMergedAttention (+21 more)

### Community 29 - "test_pretrained_hf_model.py"
Cohesion: 0.10
Nodes (13): FastTrainingConfig, ProbeLoRALinear, test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip() (+5 more)

### Community 30 - "hashlib"
Cohesion: 0.12
Nodes (19): source_fingerprints(), WithoutDocstrings, test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), test_exported_worker_has_parity_context_and_exact_adapter_checks(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main() (+11 more)

### Community 31 - "re"
Cohesion: 0.33
Nodes (6): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines()

### Community 32 - "Tensor"
Cohesion: 0.09
Nodes (14): build_bifocalgemma_v3_optimizer(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids() (+6 more)

### Community 34 - "typing"
Cohesion: 0.06
Nodes (20): retry_windows_file_operation(), Bf16FullLinear, configure_full_rank_cross_attention(), FullCrossTiledAttention, batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), load_matrix_adapters_for_inference() (+12 more)

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.14
Nodes (8): TokenizedRepairExample, RepairCollator, __getattr__(), run_reference_smoke(), __getattr__(), patch_cross_entropy(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 37 - "repository_corpus.py"
Cohesion: 0.12
Nodes (10): build_repository_corpus(), context_license_evidence(), path_license_evidence(), prepare_repository_corpus(), recognize_license(), RepositoryPreparationLedger, resolved_license(), tokenize() (+2 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 39 - "memory_probe.py"
Cohesion: 0.20
Nodes (8): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), test_export_matrix_pretrained_model_baseline(), TestMemoryProbe

### Community 40 - "typing"
Cohesion: 0.09
Nodes (4): BackboneProfile, _pair(), _ref(), RoutedRepoBatch

### Community 41 - "extract_mcq_letter"
Cohesion: 0.14
Nodes (7): extract_mcq_letter(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_ruling_out_distractors_vulnerability(), test_adversarial_mcq_split_ambiguity_behavior(), test_adversarial_mcq_verbose_reasoning_preambles()

### Community 42 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics()

### Community 43 - "v2_preflight.py"
Cohesion: 0.21
Nodes (7): current_capacity(), _gpu_context(), maximum_probe(), _measure_phase(), select_execution(), emit(), save()

### Community 44 - "build_native_tiny_model"
Cohesion: 0.20
Nodes (4): build_native_tiny_model(), test_curated_resume_rejects_runtime_drift(), TestNativeIntegration, test_last_token_logits_match_full_head()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (9): FixtureGenerationModel, FixtureTokenizer, test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_embedded_runtime_extraction_from_fast_notebook(), test_embedded_runtime_extraction_from_real_matrix_notebook(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_legacy_checkpoint_vs_clean_resume() (+1 more)

### Community 46 - "RepositorySnapshot"
Cohesion: 0.14
Nodes (15): RepairExample, RepositoryFile, RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord (+7 more)

### Community 47 - "test_tier4_scenarios.py"
Cohesion: 0.18
Nodes (7): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens()

### Community 48 - "optimizers.py"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 49 - "stages.py"
Cohesion: 0.09
Nodes (18): file_digest(), retry_delay(), checked_prefix(), recover_remote_run(), _synchronize_once(), synchronize_run(), save_benchmark_progress(), upload_checkpoint() (+10 more)

### Community 50 - "Seq2SeqBatch"
Cohesion: 0.13
Nodes (5): Seq2SeqBatch, ReferenceMergedAttention, TinyReferenceModel, TestMergedAttentionReference, test_fast_pipeline_batch_edge_cases()

### Community 51 - "repository_study.py"
Cohesion: 0.22
Nodes (20): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), generate_diagnostic_text(), fetch_tokenizer_assets() (+12 more)

### Community 52 - "curation.py"
Cohesion: 0.24
Nodes (6): code_signature(), code_tokens(), CodeOverlapMatch, nextcoder_row(), test_indentation_changes_are_not_identical(), test_unparseable_fallback_preserves_whitespace()

### Community 53 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 55 - "examples"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "test_training_telemetry_v2.py"
Cohesion: 0.24
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 57 - "pathlib"
Cohesion: 0.08
Nodes (9): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion(), main(), render_eval_script(), main() (+1 more)

### Community 58 - "transfer_text_checkpoint"
Cohesion: 0.21
Nodes (4): transfer_text_checkpoint(), load_local_native_model(), TestCheckpointTransfer, test_local_loader_preflights_and_preserves_shared_words()

### Community 59 - "release_gpu_profile.py"
Cohesion: 0.39
Nodes (4): test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), check_evidence(), main()

### Community 60 - "PackedTokenizedDataset"
Cohesion: 0.18
Nodes (5): build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 61 - "wiggly_dashboard.py"
Cohesion: 0.14
Nodes (4): LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard

### Community 62 - "data/context.py"
Cohesion: 0.48
Nodes (4): ContextWindow, render_window(), select_repository_context(), _terms()

### Community 63 - "TestBifocalGemmaArchitecture"
Cohesion: 0.06
Nodes (3): ASTGraphProjector, RepoContextRouter, TestBifocalGemmaArchitecture

### Community 64 - "test_molab_matrix_notebooks.py"
Cohesion: 0.09
Nodes (8): test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_prepare_sampled_benchmark_collection_deterministic_100_cases(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight(), test_telemetry_history_appends_all_active_stages()

### Community 65 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (30): CheckpointRef, get_v2_experiment(), decode_text(), load_verified_tokenizer(), prompt_ids(), score_context_case(), configure_memory_efficient_training(), configure_policy_model() (+22 more)

### Community 69 - "strict_letter"
Cohesion: 0.29
Nodes (4): fallback_letter_regex(), strict_letter(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 70 - "tests_e2e/conftest.py"
Cohesion: 0.06
Nodes (17): completion_metrics(), authoritative_empirical_results(), pytest_configure(), reference_resilient_mcq_parser(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema() (+9 more)

### Community 71 - "molab_matrix_support.py"
Cohesion: 0.09
Nodes (7): stratified_subset(), _DummyTokenizer, pilot_module(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), _prepare(), prepare_runtime()

### Community 72 - "__main__.py"
Cohesion: 0.13
Nodes (10): validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents, run_native_smoke(), SFTConfig (+2 more)

### Community 74 - "argparse"
Cohesion: 0.10
Nodes (15): compare_predictions(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), audit(), main(), audit(), emit(), main() (+7 more)

### Community 75 - "fit_offline_sft"
Cohesion: 0.23
Nodes (8): cpu_tree(), file_digest(), fit_offline_sft(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest(), TestOfflineSFT

### Community 76 - "random"
Cohesion: 0.27
Nodes (5): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 77 - "test_export_matrix_pretrained_model_with_subfolder_and_tokenizer"
Cohesion: 0.25
Nodes (3): export_matrix_pretrained_model(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_export_matrix_pretrained_model()

### Community 78 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 82 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (17): BenchmarkUploadQueue, limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_parent_conflict_refreshes_but_keeps_retry_bounded() (+9 more)

### Community 91 - "test_repository_study.py"
Cohesion: 0.09
Nodes (23): canonical_snapshot_hash(), validate_code_eval_snapshots(), example_from_record(), prepare_repository_record(), prepare_repository_record_with_snapshot(), repository_task_from_row(), RepositoryTask, project_root() (+15 more)

### Community 93 - "score_context_case"
Cohesion: 0.09
Nodes (13): build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), score_repo_niah_case() (+5 more)

### Community 95 - "config.py"
Cohesion: 0.15
Nodes (5): restore_checkpoint(), save_checkpoint(), ExperimentSpec, BifocalGemmaModel, build_pretrained_model()

### Community 97 - "context_tasks.py"
Cohesion: 0.20
Nodes (6): task_prompt(), tokenize_context_task(), test_adversarial_prompt_neutrality_matrix(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline()

### Community 101 - "shared_execution.py"
Cohesion: 0.26
Nodes (8): build_training_groups(), benchmark_candidate(), probe_optimizer(), publish_profile(), read_remote_profile(), validate_profile(), execution_settings(), verify_report()

### Community 103 - "test_repository_corpus.py"
Cohesion: 0.18
Nodes (11): ArchiveSnapshotStore, patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch() (+3 more)

### Community 105 - "test_v2_fixed_execution.py"
Cohesion: 0.10
Nodes (15): flex_kernel_options(), original(), calibration(), guard(), examples(), fixture_guard(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_guard_failure_is_durable_and_rolls_back_probe_parameters() (+7 more)

### Community 106 - "v2.py"
Cohesion: 0.06
Nodes (22): run_training_memory_preflight(), load_sft_adapters_for_inference(), remote_reset_plan(), reset_local_benchmarks(), reset_state(), fit_curated_sft(), prepare_curated_training(), default_training_profiles() (+14 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.11
Nodes (13): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+5 more)

### Community 108 - "os"
Cohesion: 0.26
Nodes (5): audit_pool(), audit_views(), main(), main(), source_notices()

### Community 109 - "atomic_json"
Cohesion: 0.10
Nodes (22): run_context_benchmark(), activity(), emit(), publish_final(), shared_contract(), atomic_json(), append_event(), DurableClock (+14 more)

### Community 111 - "test_training_recipes.py"
Cohesion: 0.14
Nodes (14): comparison_recipes(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume() (+6 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 878 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `_execute_v2`, `prepare_colab_pilot.py`, `stable_fingerprint`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `shared_execution.py`, `prepare_context_data.py`, `Tensor`, `typing`, `v2_preflight.py`, `stages.py`, `examples`, `release_gpu_profile.py`, `PackedTokenizedDataset`, `test_molab_matrix_notebooks.py`, `argparse`, `benchmark_upload.py`, `score_context_case`, `shared_execution.py`, `test_v2_fixed_execution.py`, `v2.py`, `test_v2_workflow.py`, `os`, `test_training_recipes.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08858858858858859 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `examples`, `prepare_colab_pilot.py`, `contracts.py`, `data/offline_sft.py`, `torch`, `prepare_colab_probe.py`, `curation.py`, `molab_pilot_support.py`, `tokenization.py`, `tiled_attention`, `repository_corpus.py`, `memory_probe.py`, `v2_preflight.py`, `build_native_tiny_model`, `test_tier4_scenarios.py`, `JSONLRepairDataset`, `PackedTokenizedDataset`, `test_molab_matrix_notebooks.py`, `test_molab_matrix_notebooks.py`, `molab_matrix_support.py`, `__main__.py`, `random`, `modeling_bifocalgemma.py`, `context_tasks.py`, `v2.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `examples` be split into smaller, more focused modules?**
  _Cohesion score 0.13438735177865613 - nodes in this community are weakly interconnected._
- **Why does `ExperimentSpec` connect `config.py` to `test_molab_matrix_notebooks.py`, `typing`, `TokenizedRepairExample`, `contracts.py`, `typing`, `__main__.py`, `torch`, `build_native_tiny_model`, `atomic_json`, `NativeTinyModel`, `Seq2SeqBatch`, `repository_study.py`, `tokenization.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._