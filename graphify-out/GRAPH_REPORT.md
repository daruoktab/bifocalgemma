# Graph Report - bifocalgemma  (2026-11-09)

## Corpus Check
- 290 files · ~365,305 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2369 nodes · 6952 edges · 103 communities (76 shown, 27 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 704 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5c1b1283`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- audit_context_release.py
- atomic_json
- prepare_colab_pilot.py
- releases.py
- dataclasses
- data/offline_sft.py
- test_shared_execution.py
- FeatureNotReadyError
- v3/__init__.py
- test_gsm8k.py
- subprocess
- matrix.py
- json
- NativeTinyModel
- prepare_colab_probe.py
- shared_execution.py
- configure_memory_efficient_training
- molab_pilot_support.py
- torch
- prepare_context_data.py
- test_tier5_adversarial.py
- tests_e2e/conftest.py
- curation.py
- test_pretrained_hf_model.py
- gsm8k.py
- benchmark_upload.py
- torch
- JSONLRepairDataset
- source_bundle_bytes
- replace
- test_molab_matrix_notebooks.py
- v3/objectives.py
- patch_cross_entropy
- TokenizedRepairExample
- repository_corpus.py
- test_tier1_features.py
- adversarial_r3_probe.py
- test_v2_fixed_execution.py
- run_sequential_pipeline
- TestGemma4EdgeTopology
- test_completed_benchmark_with_pending_upload_resumes_without_model_load
- PackedTokenizedDataset
- test_molab_remediation.py
- swesmith.py
- test_tier2_boundary.py
- random
- examples
- TestTrainingPolicy
- repository_study.py
- test_tier4_scenarios.py
- contracts.py
- test_molab_pilot.py
- test_snapshot_verifies_replacement_before_pruning
- adapters.py
- test_benchmark_prompt_budget.py
- memory_probe.py
- argparse
- stable_fingerprint
- profile_repository_lengths.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- TestBifocalGemmaArchitecture
- audit_dataset_release.py
- build_native_tiny_model
- pathlib
- NativeMergedAttention
- transfer_text_checkpoint
- run_length_grid
- v2.py
- benchmark.py
- typing
- build_policy_optimizer
- TestNotebookCellParity
- SwiGLUDualGateBridge
- FastTrainingPipeline
- test_context_benchmark_resume.py
- bifocalgemma/__init__.py
- patch_matrix_tokenization
- fast_pipeline.py
- BifocalGemmaForCausalLM
- bifocalgemma
- re
- test_context_tasks.py
- evaluation/context.py
- telemetry.py
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- copy
- test_v2_workflow.py
- test_colab_pilot.py

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 83 edges
2. `stable_fingerprint()` - 65 edges
3. `load_verified_tokenizer()` - 53 edges
4. `TokenizedRepairExample` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `run_matrix_training()` - 40 edges
10. `RepairCollator` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/source_identity.py
- `test_train_fast_cli_baseline_2b()` --calls--> `main()`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/__main__.py
- `event()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (103 total, 27 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.06
Nodes (22): pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt() (+14 more)

### Community 1 - "remediation.py"
Cohesion: 0.07
Nodes (34): read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+26 more)

### Community 2 - "audit_context_release.py"
Cohesion: 0.17
Nodes (4): audit_views(), main(), main(), source_notices()

### Community 3 - "atomic_json"
Cohesion: 0.10
Nodes (17): file_digest(), build_packed_cache(), open_packed_cache(), checked_prefix(), recover_remote_run(), _synchronize_once(), atomic_json(), retained_checkpoints() (+9 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.11
Nodes (23): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset() (+15 more)

### Community 5 - "releases.py"
Cohesion: 0.09
Nodes (25): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+17 more)

### Community 6 - "dataclasses"
Cohesion: 0.08
Nodes (18): BackboneProfile, CheckpointRef, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), main(), describe_initialization() (+10 more)

### Community 7 - "data/offline_sft.py"
Cohesion: 0.12
Nodes (20): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines() (+12 more)

### Community 8 - "test_shared_execution.py"
Cohesion: 0.13
Nodes (9): test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), test_shared_profile_upload_retries_without_losing_local_selection() (+1 more)

### Community 9 - "FeatureNotReadyError"
Cohesion: 0.07
Nodes (9): DeveloperAgent, EvaluationReport, extract_token_ast_features(), TokenASTFeatures, SWEBenchHarness, EncoderMemory, RepositoryCache, FeatureNotReadyError (+1 more)

### Community 10 - "v3/__init__.py"
Cohesion: 0.11
Nodes (8): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, clamped_triton_kernel_options(), critical_attention_beta(), ensure_hardware_memory_guards(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.05
Nodes (32): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+24 more)

### Community 12 - "subprocess"
Cohesion: 0.13
Nodes (7): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "matrix.py"
Cohesion: 0.13
Nodes (11): retry_windows_file_operation(), load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), run_matrix_stage(), is_cross_bridge_parameter() (+3 more)

### Community 14 - "json"
Cohesion: 0.13
Nodes (9): write_view(), compare_predictions(), test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), test_view_record_json_is_lossless_and_gsm_shots_are_explicit(), test_paired_bootstrap_and_cohort_rejection(), audit(), main(), main() (+1 more)

### Community 16 - "prepare_colab_probe.py"
Cohesion: 0.25
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 17 - "shared_execution.py"
Cohesion: 0.08
Nodes (32): build_policy_optimizer(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind() (+24 more)

### Community 18 - "configure_memory_efficient_training"
Cohesion: 0.11
Nodes (24): get_v2_experiment(), configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), test_cached_uncached_and_batched_greedy_decode_equivalence() (+16 more)

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.14
Nodes (10): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), digest(), Journal, PilotStorage, run(), _run() (+2 more)

### Community 20 - "torch"
Cohesion: 0.10
Nodes (8): BifocalGemmaConfig, _resolve_backbone_weights(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_config_initialization_and_serialization(), test_enable_memory_efficient_training_and_softcapping(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 21 - "prepare_context_data.py"
Cohesion: 0.11
Nodes (17): normalize_context_task(), repair_preimage_coverage(), task_prompt(), tokenize_context_task(), load_curated_pool(), ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), audit_pool() (+9 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+15 more)

### Community 23 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 24 - "curation.py"
Cohesion: 0.08
Nodes (36): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+28 more)

### Community 25 - "test_pretrained_hf_model.py"
Cohesion: 0.10
Nodes (15): export_matrix_pretrained_model(), FastTrainingConfig, ProbeLoRALinear, test_fast_training_config_and_pipeline_baseline_single_stage(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip() (+7 more)

### Community 26 - "gsm8k.py"
Cohesion: 0.19
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 27 - "benchmark_upload.py"
Cohesion: 0.12
Nodes (8): BenchmarkUploadQueue, retry_delay(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_retry_after_date_and_network_failures()

### Community 28 - "torch"
Cohesion: 0.12
Nodes (13): flex_kernel_options(), fused_attention(), attention_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention, TiledMergedAttention (+5 more)

### Community 29 - "JSONLRepairDataset"
Cohesion: 0.13
Nodes (4): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export()

### Community 30 - "source_bundle_bytes"
Cohesion: 0.17
Nodes (11): test_exported_worker_has_parity_context_and_exact_adapter_checks(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), main() (+3 more)

### Community 31 - "replace"
Cohesion: 0.14
Nodes (8): evaluate_sft_loss(), validation_batches(), encoder_diagnostics(), TestTokenizerVerification, calibration(), test_atomic_status_retries_windows_sharing_violation_without_partial_write(), replace(), evaluate_source_copy_probes()

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.09
Nodes (6): prepare_sampled_benchmark_collection(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_prepare_sampled_benchmark_collection_deterministic_100_cases(), test_telemetry_history_appends_all_active_stages()

### Community 34 - "v3/objectives.py"
Cohesion: 0.09
Nodes (14): build_bifocalgemma_v3_optimizer(), chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), is_zero_weight_decay_bridge_parameter(), _pointer_chunk_forward() (+6 more)

### Community 35 - "patch_cross_entropy"
Cohesion: 0.10
Nodes (5): GatedCrossAttentionBridge, RMSNorm, __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy()

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.10
Nodes (14): RepairExample, RepairPrediction, TokenizedRepairExample, RepairCollator, __getattr__(), format_encoder_context(), tokenize_repair_example(), write_swebench_predictions() (+6 more)

### Community 37 - "repository_corpus.py"
Cohesion: 0.11
Nodes (16): validate_context_license_record(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license(), resolved_license() (+8 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 40 - "test_v2_fixed_execution.py"
Cohesion: 0.12
Nodes (7): format_duration(), loss_chart(), render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_preflight_upload_verifies_exact_evidence_without_checkpoint(), test_stage_counters_do_not_leak_and_flex_backward_tiles_cover_head256()

### Community 41 - "run_sequential_pipeline"
Cohesion: 0.23
Nodes (9): synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint(), StageLedger, test_reset_archives_evaluation_once_without_changing_training(), test_pipeline_state_ledger_creation_for_remote_resumption() (+1 more)

### Community 43 - "test_completed_benchmark_with_pending_upload_resumes_without_model_load"
Cohesion: 0.15
Nodes (13): comparison_recipes(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume(), benchmark() (+5 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "swesmith.py"
Cohesion: 0.24
Nodes (7): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 47 - "test_tier2_boundary.py"
Cohesion: 0.15
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 48 - "random"
Cohesion: 0.24
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 49 - "examples"
Cohesion: 0.20
Nodes (11): test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), measure(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence() (+3 more)

### Community 51 - "repository_study.py"
Cohesion: 0.13
Nodes (28): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), prepare_editing_records(), split_editing_records(), read_source_rows() (+20 more)

### Community 52 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 53 - "contracts.py"
Cohesion: 0.06
Nodes (13): restore_checkpoint(), save_checkpoint(), ModelOutput, Seq2SeqBatch, apply_word_embedding_policy(), EmbeddingPolicyResult, build_joint_visibility(), ReferenceMergedAttention (+5 more)

### Community 54 - "test_molab_pilot.py"
Cohesion: 0.25
Nodes (4): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), render()

### Community 55 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 58 - "memory_probe.py"
Cohesion: 0.22
Nodes (7): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), TestMemoryProbe

### Community 59 - "argparse"
Cohesion: 0.17
Nodes (8): test_external_activity_reader_does_not_need_running_kernel(), main(), main(), module_definitions(), render(), use_embedded_imports(), inspect(), main()

### Community 60 - "stable_fingerprint"
Cohesion: 0.18
Nodes (11): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), RepositoryPreparationLedger (+3 more)

### Community 61 - "profile_repository_lengths.py"
Cohesion: 0.32
Nodes (4): publish(), distribution(), main(), profile()

### Community 63 - "TestBifocalGemmaArchitecture"
Cohesion: 0.06
Nodes (3): ASTGraphProjector, RepoContextRouter, TestBifocalGemmaArchitecture

### Community 64 - "audit_dataset_release.py"
Cohesion: 0.52
Nodes (4): audit(), emit(), main(), validate_all_patches()

### Community 65 - "build_native_tiny_model"
Cohesion: 0.13
Nodes (8): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words(), test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_alignment_stays_fp32_after_bf16_conversion()

### Community 67 - "pathlib"
Cohesion: 0.07
Nodes (13): joint_visibility_tile(), cpu_tree(), file_digest(), restore_sft_checkpoint(), sample_encoder_mass(), examples(), TestOfflineSFT, TinyLossModel (+5 more)

### Community 72 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 74 - "v2.py"
Cohesion: 0.07
Nodes (29): load_sft_adapters_for_inference(), prepare_curated_training(), publish_with_retry(), resolve_shared_profile(), shared_contract(), source_fingerprints(), WithoutDocstrings, latest_checkpoint() (+21 more)

### Community 76 - "benchmark.py"
Cohesion: 0.12
Nodes (9): greedy_decode(), GreedyGeneration, configure_full_rank_cross_attention(), batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight(), ScriptedModel (+1 more)

### Community 77 - "typing"
Cohesion: 0.07
Nodes (9): cached_logits(), CachedAttention, InferenceCache, NormalizedAlignment, build_native_model(), RoutedRepoBatch, test_alignment_initialization_with_cuda_default_device(), test_cached_logits_match_full_prefix_and_cross_window() (+1 more)

### Community 78 - "build_policy_optimizer"
Cohesion: 0.32
Nodes (3): GrokOrScaleLoRA, orthogonal_direction(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 82 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 84 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 87 - "fast_pipeline.py"
Cohesion: 0.14
Nodes (3): ensure_assets_on_disk(), load_asset_json(), native_config()

### Community 91 - "re"
Cohesion: 0.07
Nodes (41): RepositoryFile, RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), apply_patch_to_snapshot() (+33 more)

### Community 92 - "test_context_tasks.py"
Cohesion: 0.18
Nodes (7): fallback_letter_regex(), strict_letter(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 93 - "evaluation/context.py"
Cohesion: 0.10
Nodes (19): decode_text(), build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), prompt_ids() (+11 more)

### Community 96 - "telemetry.py"
Cohesion: 0.07
Nodes (11): format_stage_display(), LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard, test_dashboard_stage_display_formatting(), test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes() (+3 more)

### Community 106 - "copy"
Cohesion: 0.08
Nodes (26): run_training_memory_preflight(), remote_reset_plan(), reset_local_benchmarks(), reset_state(), fit_curated_sft(), fit_offline_sft(), emit(), model_stage() (+18 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.11
Nodes (13): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+5 more)

### Community 115 - "test_colab_pilot.py"
Cohesion: 0.20
Nodes (4): stratified_subset(), _DummyTokenizer, test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 890 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `bifocalgemma/__init__.py`, `torch`, `contracts.py`, `fast_pipeline.py`, `test_pretrained_hf_model.py`, `JSONLRepairDataset`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05519480519480519 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `atomic_json` to `audit_context_release.py`, `prepare_colab_pilot.py`, `matrix.py`, `json`, `shared_execution.py`, `configure_memory_efficient_training`, `prepare_context_data.py`, `benchmark_upload.py`, `replace`, `test_molab_matrix_notebooks.py`, `test_v2_fixed_execution.py`, `run_sequential_pipeline`, `test_completed_benchmark_with_pending_upload_resumes_without_model_load`, `examples`, `repository_study.py`, `test_snapshot_verifies_replacement_before_pruning`, `argparse`, `stable_fingerprint`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `pathlib`, `v2.py`, `benchmark.py`, `evaluation/context.py`, `copy`, `test_v2_workflow.py`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07477288609364081 - nodes in this community are weakly interconnected._
- **Why does `TestBifocalGemmaArchitecture` connect `TestBifocalGemmaArchitecture` to `patch_cross_entropy`, `pathlib`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._