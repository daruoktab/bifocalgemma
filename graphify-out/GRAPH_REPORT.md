# Graph Report - bifocalgemma  (2026-11-08)

## Corpus Check
- 290 files · ~364,404 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2356 nodes · 6919 edges · 101 communities (72 shown, 29 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 711 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6059b3af`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- stable_fingerprint
- atomic_json
- prepare_colab_pilot.py
- releases.py
- dataclasses
- data/offline_sft.py
- native.py
- FeatureNotReadyError
- BifocalGemmaConfig
- test_gsm8k.py
- subprocess
- v2.py
- hashlib
- NativeTinyModel
- RepairPrediction
- shared_execution.py
- configure_memory_efficient_training
- molab_pilot_support.py
- torch
- prepare_context_data.py
- test_tier5_adversarial.py
- full_pool.py
- curation.py
- get_v2_experiment
- gsm8k.py
- BenchmarkUploadQueue
- tiled_attention
- JSONLRepairDataset
- test_molab_pilot.py
- tokenization.py
- Tensor
- matrix.py
- GatedCrossAttentionBridge
- TokenizedRepairExample
- repository_corpus.py
- test_tier1_features.py
- adversarial_r3_probe.py
- audit_context_tokens.py
- RepoContextRouter
- TestGemma4EdgeTopology
- comparison.py
- telemetry.py
- test_molab_remediation.py
- swesmith.py
- completion_metrics
- test_memory_efficient_training.py
- test_tier4_scenarios.py
- repository_study.py
- adapters.py
- test_construction_blocks.py
- v2_diagnostics.py
- test_snapshot_verifies_replacement_before_pruning
- test_training_telemetry_v2.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- transfer_text_checkpoint
- prepare_colab_probe.py
- test_notebook_hf_flow.py
- profile_repository_lengths.py
- harness.py
- TestBifocalGemmaArchitecture
- test_molab_matrix_notebooks.py
- build_native_tiny_model
- fit_offline_sft
- tests_e2e/conftest.py
- Float32Alignment
- pathlib
- benchmark.py
- ASTGraphProjector
- TestNotebookCellParity
- v3/__init__.py
- FastTrainingPipeline
- test_context_benchmark_resume.py
- modeling_bifocalgemma.py
- patch_matrix_tokenization
- BifocalGemmaForCausalLM
- bifocalgemma
- contracts.py
- run_repo_niah_eval
- matrix.py
- ExperimentSpec
- telemetry.py
- typing
- test_v2_workflow.py
- test_colab_pilot.py

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
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `event()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py
- `save()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (101 total, 29 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "remediation.py"
Cohesion: 0.10
Nodes (25): split_connected_records(), load_curated_pool(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage() (+17 more)

### Community 2 - "stable_fingerprint"
Cohesion: 0.10
Nodes (20): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), ArchiveSnapshotStore (+12 more)

### Community 3 - "atomic_json"
Cohesion: 0.08
Nodes (17): run_context_benchmark(), activity(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint(), atomic_json() (+9 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.07
Nodes (32): decode_text(), evaluate_sft_loss(), validation_batches(), encoder_diagnostics(), replace(), test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory() (+24 more)

### Community 5 - "releases.py"
Cohesion: 0.08
Nodes (28): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+20 more)

### Community 6 - "dataclasses"
Cohesion: 0.11
Nodes (11): validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents, BuildBlock, SFTConfig (+3 more)

### Community 7 - "data/offline_sft.py"
Cohesion: 0.11
Nodes (20): curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines(), prepare_editing_records(), repository_editing_record(), split_editing_records(), tokenize() (+12 more)

### Community 8 - "native.py"
Cohesion: 0.16
Nodes (3): __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy()

### Community 9 - "FeatureNotReadyError"
Cohesion: 0.09
Nodes (7): extract_token_ast_features(), TokenASTFeatures, EncoderMemory, RepositoryCache, FeatureNotReadyError, distillation_loss(), profile_experiment()

### Community 10 - "BifocalGemmaConfig"
Cohesion: 0.16
Nodes (5): BifocalGemmaConfig, test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_config_initialization_and_serialization(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (14): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+6 more)

### Community 12 - "subprocess"
Cohesion: 0.11
Nodes (11): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), audit(), emit(), main(), validate_all_patches(), main(), main(), main() (+3 more)

### Community 13 - "v2.py"
Cohesion: 0.10
Nodes (19): reset_local_benchmarks(), retry_delay(), publish_with_retry(), source_fingerprints(), WithoutDocstrings, StageLedger, benchmark_jobs(), stratified_subset() (+11 more)

### Community 14 - "hashlib"
Cohesion: 0.16
Nodes (12): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), test_exported_worker_has_parity_context_and_exact_adapter_checks(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), main(), module_definitions() (+4 more)

### Community 16 - "RepairPrediction"
Cohesion: 0.23
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 17 - "shared_execution.py"
Cohesion: 0.05
Nodes (43): build_policy_optimizer(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind() (+35 more)

### Community 18 - "configure_memory_efficient_training"
Cohesion: 0.15
Nodes (20): CheckpointRef, configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), configure_matrix_stage(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence() (+12 more)

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.05
Nodes (29): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), test_upload_retry_verification_and_base_prefix_only(), calibration() (+21 more)

### Community 20 - "torch"
Cohesion: 0.14
Nodes (13): export_matrix_pretrained_model(), FastTrainingConfig, test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip(), test_enable_memory_efficient_training_and_softcapping(), test_export_matrix_pretrained_model() (+5 more)

### Community 21 - "prepare_context_data.py"
Cohesion: 0.12
Nodes (17): normalize_context_task(), repair_preimage_coverage(), task_prompt(), tokenize_context_task(), patch_paths(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_requires_visible_preimage_without_refilling_context() (+9 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), fallback_letter_regex(), strict_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards() (+15 more)

### Community 23 - "full_pool.py"
Cohesion: 0.17
Nodes (11): read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+3 more)

### Community 24 - "curation.py"
Cohesion: 0.08
Nodes (35): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+27 more)

### Community 25 - "get_v2_experiment"
Cohesion: 0.08
Nodes (18): get_v2_experiment(), greedy_decode(), cached_logits(), CachedAttention, InferenceCache, ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards(), test_alignment_initialization_with_cuda_default_device() (+10 more)

### Community 26 - "gsm8k.py"
Cohesion: 0.19
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 27 - "BenchmarkUploadQueue"
Cohesion: 0.09
Nodes (16): BenchmarkUploadQueue, limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed() (+8 more)

### Community 28 - "tiled_attention"
Cohesion: 0.19
Nodes (10): fused_attention(), attention_tile(), mask_tile(), tiled_attention(), _TiledAttention, test_flex_backward_head256_sequence_above_small_sequence_branch(), test_flex_joint_gqa_matches_reference_on_cuda(), test_flex_structured_native_masks_across_sequence_lengths() (+2 more)

### Community 29 - "JSONLRepairDataset"
Cohesion: 0.12
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 30 - "test_molab_pilot.py"
Cohesion: 0.18
Nodes (11): test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render(), main() (+3 more)

### Community 31 - "tokenization.py"
Cohesion: 0.17
Nodes (6): RepairExample, format_encoder_context(), tokenize_repair_example(), generate_patch(), repair_prompt_ids(), TestRealOfflineTokenizers

### Community 32 - "Tensor"
Cohesion: 0.08
Nodes (15): build_bifocalgemma_v3_optimizer(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), is_zero_weight_decay_bridge_parameter(), _pointer_chunk_forward() (+7 more)

### Community 34 - "matrix.py"
Cohesion: 0.09
Nodes (13): ProbeLoRALinear, load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix_v3(), build_anti_forgetting_curriculum() (+5 more)

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.09
Nodes (13): TokenizedRepairExample, RepairCollator, __getattr__(), build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, run_reference_smoke(), TestDataAndLoss (+5 more)

### Community 37 - "repository_corpus.py"
Cohesion: 0.09
Nodes (19): validate_context_license_record(), build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license() (+11 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (8): score_repo_niah_case(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases(), test_adversarial_reponiah_scorer_negation_vulnerability()

### Community 40 - "audit_context_tokens.py"
Cohesion: 0.20
Nodes (5): decoder_prompt(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 43 - "comparison.py"
Cohesion: 0.31
Nodes (3): compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), main()

### Community 44 - "telemetry.py"
Cohesion: 0.12
Nodes (7): flex_kernel_options(), format_duration(), loss_chart(), render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_stage_counters_do_not_leak_and_flex_backward_tiles_cover_head256()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "swesmith.py"
Cohesion: 0.18
Nodes (8): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract, TestTokenizerVerification

### Community 47 - "completion_metrics"
Cohesion: 0.15
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 49 - "test_memory_efficient_training.py"
Cohesion: 0.27
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 50 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 51 - "repository_study.py"
Cohesion: 0.20
Nodes (22): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), load_verified_tokenizer(), summarize_native_results() (+14 more)

### Community 52 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 53 - "test_construction_blocks.py"
Cohesion: 0.11
Nodes (6): apply_word_embedding_policy(), EmbeddingPolicyResult, build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference, TestWordEmbeddingPolicy

### Community 54 - "v2_diagnostics.py"
Cohesion: 0.20
Nodes (4): joint_visibility_tile(), TiledEncoderAttention, TiledMergedAttention, sample_encoder_mass()

### Community 56 - "test_training_telemetry_v2.py"
Cohesion: 0.20
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 58 - "transfer_text_checkpoint"
Cohesion: 0.14
Nodes (11): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), run_length_grid(), synthetic_stress_target(), tensor_bytes(), test_length_grid_matches_baseline_budget_and_stops_after_failure() (+3 more)

### Community 59 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 60 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 61 - "profile_repository_lengths.py"
Cohesion: 0.32
Nodes (4): publish(), distribution(), main(), profile()

### Community 64 - "test_molab_matrix_notebooks.py"
Cohesion: 0.07
Nodes (8): test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads(), test_telemetry_history_appends_all_active_stages(), _prepare()

### Community 65 - "build_native_tiny_model"
Cohesion: 0.08
Nodes (12): build_native_model(), build_native_tiny_model(), native_config(), NativeMergedAttention, run_native_smoke(), load_local_native_model(), RoutedRepoBatch, TestNativeIntegration (+4 more)

### Community 67 - "fit_offline_sft"
Cohesion: 0.06
Nodes (25): cpu_tree(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest() (+17 more)

### Community 70 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 74 - "pathlib"
Cohesion: 0.09
Nodes (16): file_digest(), remote_reset_plan(), reset_state(), checked_prefix(), recover_remote_run(), _synchronize_once(), validate_checkpoint_files(), snapshot() (+8 more)

### Community 76 - "benchmark.py"
Cohesion: 0.13
Nodes (10): GreedyGeneration, DurableClock, batched_greedy_decode(), infer_benchmark_weight_tier(), prepare_sampled_benchmark_collection(), resolve_optimal_eval_batch_size(), run_batched_context_benchmark(), activity() (+2 more)

### Community 80 - "v3/__init__.py"
Cohesion: 0.09
Nodes (10): Bf16FullLinear, _compute_per_row_cross_query(), configure_full_rank_cross_attention(), FullCrossCachedAttention, FullCrossTiledAttention, SwiGLUDualGateBridge, clamped_triton_kernel_options(), critical_attention_beta() (+2 more)

### Community 82 - "test_context_benchmark_resume.py"
Cohesion: 0.18
Nodes (5): FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch(), transient()

### Community 83 - "modeling_bifocalgemma.py"
Cohesion: 0.12
Nodes (3): BifocalGemmaPreTrainedModel, _resolve_backbone_weights(), transfer_text_checkpoint()

### Community 84 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 91 - "contracts.py"
Cohesion: 0.07
Nodes (40): RepositoryFile, RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), apply_patch_to_snapshot() (+32 more)

### Community 93 - "run_repo_niah_eval"
Cohesion: 0.12
Nodes (8): build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), run_repo_niah_eval(), test_adversarial_reponiah_depth_and_ratio_extremes(), test_adversarial_reponiah_deterministic_reproducibility()

### Community 94 - "matrix.py"
Cohesion: 0.14
Nodes (10): benchmark_directory_name(), download_model(), target_fingerprint(), run_matrix_benchmark(), run_matrix_stage(), run_matrix_training(), run_matrix_training_and_benchmark(), check_legacy_artifacts() (+2 more)

### Community 95 - "ExperimentSpec"
Cohesion: 0.10
Nodes (11): restore_checkpoint(), save_checkpoint(), BackboneProfile, ExperimentSpec, _pair(), _ref(), ModelOutput, Seq2SeqBatch (+3 more)

### Community 96 - "telemetry.py"
Cohesion: 0.12
Nodes (6): format_stage_display(), LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard, test_dashboard_stage_display_formatting()

### Community 106 - "typing"
Cohesion: 0.05
Nodes (24): run_training_memory_preflight(), load_sft_adapters_for_inference(), fit_curated_sft(), prepare_curated_training(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), build_training_groups() (+16 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.11
Nodes (13): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+5 more)

### Community 115 - "test_colab_pilot.py"
Cohesion: 0.13
Nodes (8): _DummyTokenizer, pilot_module(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), prepare_runtime()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 884 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Seq2SeqBatch` connect `ExperimentSpec` to `TokenizedRepairExample`, `native.py`, `NativeTinyModel`, `FastTrainingPipeline`, `modeling_bifocalgemma.py`, `torch`, `test_construction_blocks.py`, `BifocalGemmaForCausalLM`, `contracts.py`, `JSONLRepairDataset`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `atomic_json` to `stable_fingerprint`, `prepare_colab_pilot.py`, `releases.py`, `v2.py`, `shared_execution.py`, `molab_pilot_support.py`, `prepare_context_data.py`, `get_v2_experiment`, `BenchmarkUploadQueue`, `Tensor`, `matrix.py`, `TokenizedRepairExample`, `test_snapshot_verifies_replacement_before_pruning`, `fit_offline_sft`, `pathlib`, `benchmark.py`, `contracts.py`, `run_repo_niah_eval`, `matrix.py`, `typing`, `test_v2_workflow.py`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09851551956815115 - nodes in this community are weakly interconnected._
- **Why does `TestGemma4EdgeTopology` connect `TestGemma4EdgeTopology` to `typing`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._