# Graph Report - bifocalgemma  (2026-11-08)

## Corpus Check
- 290 files · ~364,201 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2356 nodes · 6919 edges · 104 communities (71 shown, 33 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 711 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0c213bbb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- stable_fingerprint
- training/offline_sft.py
- prepare_colab_pilot.py
- releases.py
- config.py
- data/offline_sft.py
- typing
- re
- BifocalGemmaConfig
- test_gsm8k.py
- subprocess
- v2.py
- prepare_colab_eval.py
- NativeTinyModel
- RepairPrediction
- test_shared_execution.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- molab_pilot_support.py
- patch_cross_entropy
- test_context_tasks.py
- test_tier5_adversarial.py
- calibrate_shared_profile
- curation.py
- atomic_json
- load_verified_tokenizer
- BenchmarkUploadQueue
- tiled_attention
- JSONLRepairDataset
- hashlib
- shared_execution.py
- Tensor
- checkpoint.py
- GatedCrossAttentionBridge
- TokenizedRepairExample
- repository_corpus.py
- test_tier1_features.py
- adversarial_r3_probe.py
- audit_context_tokens.py
- TestGemma4EdgeTopology
- argparse
- benchmark_upload.py
- test_molab_remediation.py
- swesmith.py
- completion_metrics
- examples
- v2_preflight.py
- repository_study.py
- GatedCrossAttentionBridge
- apply_word_embedding_policy
- ReferenceMergedAttention
- test_snapshot_verifies_replacement_before_pruning
- test_training_telemetry_v2.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- transfer_text_checkpoint
- test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training
- PackedTokenizedDataset
- CachedAttention
- project_root
- TestBifocalGemmaArchitecture
- test_molab_matrix_notebooks.py
- configure_memory_efficient_training
- TestTrainingPolicy
- optimizers.py
- Any
- GitSnapshotStore
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- pathlib
- random
- ASTGraphProjector
- TestNotebookCellParity
- v3/__init__.py
- FastTrainingPipeline
- test_context_benchmark_resume.py
- modeling_bifocalgemma.py
- patch_matrix_tokenization
- BifocalGemmaForCausalLM
- bifocalgemma
- test_repository_study.py
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- evaluation/context.py
- matrix.py
- ExperimentSpec
- telemetry.py
- copy
- test_v2_workflow.py
- SwiGLUDualGateBridge
- test_completed_benchmark_with_pending_upload_resumes_without_model_load
- replace

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
- `test_fast_training_config_step_reconciliation()` --uses--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_train_fast_cli_baseline_2b()` --calls--> `main()`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/__main__.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (104 total, 33 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.07
Nodes (22): pilot_module(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt() (+14 more)

### Community 1 - "remediation.py"
Cohesion: 0.07
Nodes (26): export_pool(), read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_full_pool(), rows(), fetch_benchmark_files() (+18 more)

### Community 2 - "stable_fingerprint"
Cohesion: 0.17
Nodes (17): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+9 more)

### Community 3 - "training/offline_sft.py"
Cohesion: 0.14
Nodes (8): cpu_tree(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), optimizer_group_manifest(), retained_checkpoints(), retry_windows_file_operation(), test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume()

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.09
Nodes (27): evaluate_sft_loss(), validation_batches(), encoder_diagnostics(), build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__() (+19 more)

### Community 5 - "releases.py"
Cohesion: 0.07
Nodes (34): normalize_context_task(), repair_preimage_coverage(), load_curated_pool(), archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection() (+26 more)

### Community 6 - "config.py"
Cohesion: 0.06
Nodes (21): BackboneProfile, _pair(), _ref(), validate_experiment_matrix(), EvaluationReport, SWEBenchHarness, main(), EncoderMemory (+13 more)

### Community 7 - "data/offline_sft.py"
Cohesion: 0.11
Nodes (19): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines() (+11 more)

### Community 8 - "typing"
Cohesion: 0.08
Nodes (5): build_joint_visibility(), build_native_model(), RepoContextRouter, RoutedRepoBatch, chunked_linear_cross_entropy()

### Community 9 - "re"
Cohesion: 0.18
Nodes (11): apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), validate_code_eval_snapshots(), NativeTaskResult, score_python_patch(), summarize_native_results() (+3 more)

### Community 10 - "BifocalGemmaConfig"
Cohesion: 0.09
Nodes (13): BifocalGemmaConfig, ProbeLoRALinear, test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip(), test_enable_memory_efficient_training_and_softcapping() (+5 more)

### Community 11 - "test_gsm8k.py"
Cohesion: 0.05
Nodes (24): prepare_editing_records(), split_editing_records(), BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), prepare_repository_training() (+16 more)

### Community 12 - "subprocess"
Cohesion: 0.11
Nodes (7): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "v2.py"
Cohesion: 0.08
Nodes (26): load_sft_adapters_for_inference(), reset_local_benchmarks(), prepare_curated_training(), tokenize(), synchronize_run(), tokenize_repository_record(), run_sequential_pipeline(), publish_final() (+18 more)

### Community 15 - "NativeTinyModel"
Cohesion: 0.11
Nodes (5): NormalizedAlignment, native_config(), NativeMergedAttention, NativeTinyModel, version()

### Community 16 - "RepairPrediction"
Cohesion: 0.12
Nodes (8): DeveloperAgent, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter, write_swebench_predictions()

### Community 17 - "test_shared_execution.py"
Cohesion: 0.11
Nodes (15): seal_report(), save(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend() (+7 more)

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.10
Nodes (15): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), test_upload_retry_verification_and_base_prefix_only(), digest(), Journal (+7 more)

### Community 21 - "test_context_tasks.py"
Cohesion: 0.33
Nodes (4): task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context()

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.06
Nodes (26): completion_metrics(), extract_mcq_letter(), fallback_letter_regex(), strict_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards() (+18 more)

### Community 23 - "calibrate_shared_profile"
Cohesion: 0.17
Nodes (9): benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), probe_optimizer() (+1 more)

### Community 24 - "curation.py"
Cohesion: 0.06
Nodes (47): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), build_benchmark_index() (+39 more)

### Community 25 - "atomic_json"
Cohesion: 0.10
Nodes (18): activity(), numerical_probe(), shared_contract(), atomic_json(), append_event(), source_fingerprint(), prepare_sampled_benchmark_collection(), activity() (+10 more)

### Community 26 - "load_verified_tokenizer"
Cohesion: 0.07
Nodes (27): CheckpointRef, format_encoder_context(), load_verified_tokenizer(), tokenize_repair_example(), digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint() (+19 more)

### Community 27 - "BenchmarkUploadQueue"
Cohesion: 0.18
Nodes (6): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed()

### Community 28 - "tiled_attention"
Cohesion: 0.10
Nodes (13): flex_kernel_options(), fused_attention(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention, sample_encoder_mass() (+5 more)

### Community 29 - "JSONLRepairDataset"
Cohesion: 0.15
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 30 - "hashlib"
Cohesion: 0.11
Nodes (18): source_fingerprints(), WithoutDocstrings, test_exported_worker_has_parity_context_and_exact_adapter_checks(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code() (+10 more)

### Community 31 - "shared_execution.py"
Cohesion: 0.25
Nodes (8): retry_delay(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), verify_report(), test_retry_after_date_and_network_failures()

### Community 32 - "Tensor"
Cohesion: 0.10
Nodes (13): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+5 more)

### Community 34 - "checkpoint.py"
Cohesion: 0.11
Nodes (14): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix_v3(), build_bifocalgemma_v3_optimizer(), is_cross_bridge_parameter() (+6 more)

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.14
Nodes (7): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn(), TestMemoryProbe

### Community 37 - "repository_corpus.py"
Cohesion: 0.11
Nodes (16): ArchiveSnapshotStore, context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), recognize_license(), resolved_license(), archive() (+8 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.06
Nodes (21): pytest_configure(), reference_resilient_mcq_parser(), validate_empirical_results_schema(), validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser() (+13 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 40 - "audit_context_tokens.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 43 - "argparse"
Cohesion: 0.08
Nodes (17): compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit(), emit(), main(), validate_all_patches(), main(), main() (+9 more)

### Community 44 - "benchmark_upload.py"
Cohesion: 0.23
Nodes (3): test_external_activity_reader_does_not_need_running_kernel(), inspect(), main()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (11): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+3 more)

### Community 46 - "swesmith.py"
Cohesion: 0.10
Nodes (17): RepairExample, RepositoryFile, RepositorySnapshot, extract_token_ast_features(), TokenASTFeatures, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord (+9 more)

### Community 47 - "completion_metrics"
Cohesion: 0.15
Nodes (6): test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 49 - "examples"
Cohesion: 0.20
Nodes (4): TestOfflineSFT, TinyLossModel, test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_checkpoint_directory_publish_retries_windows_sharing_violation()

### Community 50 - "v2_preflight.py"
Cohesion: 0.24
Nodes (7): build_training_groups(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), select_execution()

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (19): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+11 more)

### Community 53 - "apply_word_embedding_policy"
Cohesion: 0.24
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 55 - "test_snapshot_verifies_replacement_before_pruning"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "test_training_telemetry_v2.py"
Cohesion: 0.15
Nodes (5): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_training_resume_keeps_elapsed_from_step_five_through_ten(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 58 - "transfer_text_checkpoint"
Cohesion: 0.12
Nodes (10): gpu_memory(), optimizer_bytes(), profile_training_group(), run_length_grid(), synthetic_stress_target(), tensor_bytes(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe() (+2 more)

### Community 59 - "test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training"
Cohesion: 0.50
Nodes (3): test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare()

### Community 64 - "test_molab_matrix_notebooks.py"
Cohesion: 0.07
Nodes (16): unified_recipe(), make_experiment_spec(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_configure_full_rank_cross_attention_v3_specs(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_matrix_baseline_single_stage_resolution() (+8 more)

### Community 65 - "configure_memory_efficient_training"
Cohesion: 0.06
Nodes (34): get_v2_experiment(), configure_memory_efficient_training(), transfer_text_checkpoint(), build_native_tiny_model(), _DummyTokenizer, test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage() (+26 more)

### Community 67 - "TestTrainingPolicy"
Cohesion: 0.27
Nodes (3): make_identity(), examples(), TestTrainingPolicy

### Community 69 - "optimizers.py"
Cohesion: 0.26
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 70 - "Any"
Cohesion: 0.22
Nodes (4): authoritative_empirical_results(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 73 - "test_preflight_upload_verifies_exact_evidence_without_checkpoint"
Cohesion: 0.18
Nodes (5): progress(), save_guard(), synchronize_preflight(), emit(), test_preflight_upload_verifies_exact_evidence_without_checkpoint()

### Community 74 - "pathlib"
Cohesion: 0.07
Nodes (16): file_digest(), build_packed_cache(), open_packed_cache(), remote_reset_plan(), reset_state(), checked_prefix(), recover_remote_run(), _synchronize_once() (+8 more)

### Community 76 - "random"
Cohesion: 0.15
Nodes (9): batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length() (+1 more)

### Community 80 - "v3/__init__.py"
Cohesion: 0.06
Nodes (10): attention_tile(), TiledMergedAttention, Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, clamped_triton_kernel_options(), critical_attention_beta() (+2 more)

### Community 81 - "FastTrainingPipeline"
Cohesion: 0.10
Nodes (7): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_baseline_2b(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 82 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 83 - "modeling_bifocalgemma.py"
Cohesion: 0.10
Nodes (6): BifocalGemmaPreTrainedModel, export_matrix_pretrained_model(), _resolve_backbone_weights(), configure_probe_training(), test_export_matrix_pretrained_model(), test_export_matrix_pretrained_model_baseline()

### Community 84 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 91 - "test_repository_study.py"
Cohesion: 0.10
Nodes (24): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), example_from_record(), prepare_repository_record(), prepare_repository_record_with_snapshot() (+16 more)

### Community 93 - "evaluation/context.py"
Cohesion: 0.09
Nodes (22): task_prompt(), tokenize_context_task(), decode_text(), build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode() (+14 more)

### Community 94 - "matrix.py"
Cohesion: 0.14
Nodes (11): benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), configure_full_rank_cross_attention(), run_matrix_benchmark(), run_matrix_stage(), run_matrix_training() (+3 more)

### Community 95 - "ExperimentSpec"
Cohesion: 0.10
Nodes (9): restore_checkpoint(), save_checkpoint(), ExperimentSpec, ModelOutput, Seq2SeqBatch, BifocalGemmaModel, build_pretrained_model(), run_reference_smoke() (+1 more)

### Community 96 - "telemetry.py"
Cohesion: 0.09
Nodes (11): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog (+3 more)

### Community 106 - "copy"
Cohesion: 0.13
Nodes (11): fit_curated_sft(), configure_policy_model(), default_training_profiles(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence(), run_recipe_comparison() (+3 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (11): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), workflow(), benchmark(), download() (+3 more)

### Community 111 - "test_completed_benchmark_with_pending_upload_resumes_without_model_load"
Cohesion: 0.14
Nodes (14): comparison_recipes(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload(), test_four_fresh_runs_per_notebook_and_independent_resume() (+6 more)

### Community 115 - "replace"
Cohesion: 0.13
Nodes (16): stratified_subset(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), original(), calibration(), guard(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters() (+8 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 883 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `stable_fingerprint`, `training/offline_sft.py`, `prepare_colab_pilot.py`, `releases.py`, `v2.py`, `test_shared_execution.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `calibrate_shared_profile`, `BenchmarkUploadQueue`, `shared_execution.py`, `Tensor`, `checkpoint.py`, `argparse`, `benchmark_upload.py`, `v2_preflight.py`, `test_snapshot_verifies_replacement_before_pruning`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `pathlib`, `v3/__init__.py`, `evaluation/context.py`, `matrix.py`, `test_v2_workflow.py`, `test_completed_benchmark_with_pending_upload_resumes_without_model_load`, `replace`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06547619047619048 - nodes in this community are weakly interconnected._
- **Why does `build_native_tiny_model()` connect `configure_memory_efficient_training` to `TestTrainingPolicy`, `TokenizedRepairExample`, `config.py`, `typing`, `copy`, `test_v2_workflow.py`, `NativeTinyModel`, `test_completed_benchmark_with_pending_upload_resumes_without_model_load`, `examples`, `modeling_bifocalgemma.py`, `test_snapshot_verifies_replacement_before_pruning`, `curation.py`, `load_verified_tokenizer`, `test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training`, `ExperimentSpec`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07183673469387755 - nodes in this community are weakly interconnected._
- **Why does `run_sequential_pipeline()` connect `v2.py` to `telemetry.py`, `configure_memory_efficient_training`, `stable_fingerprint`, `TokenizedRepairExample`, `releases.py`, `copy`, `pathlib`, `repository_study.py`, `atomic_json`, `BenchmarkUploadQueue`, `evaluation/context.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._