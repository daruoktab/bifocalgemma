# Graph Report - bifocalgemma  (2026-11-12)

## Corpus Check
- 295 files · ~381,780 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2466 nodes · 7118 edges · 118 communities (81 shown, 37 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 768 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `eb6909d2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- test_molab_matrix_notebooks.py
- curation.py
- atomic_json
- prepare_colab_pilot.py
- copy
- FeatureNotReadyError
- torch_nn
- test_shared_execution.py
- audit_context_release.py
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- test_molab_matrix_notebooks.py
- test_molab_pilot.py
- NativeTinyModel
- benchmark.py
- v2_preflight.py
- examples
- native.py
- json
- load_verified_tokenizer
- test_tier5_adversarial.py
- test_repository_study.py
- CodeOverlapIndex
- test_memory_efficient_training.py
- config.py
- hashlib
- torch
- FastTrainingPipeline
- calibration
- stable_fingerprint
- RMSNorm
- matrix.py
- prepare_colab_probe.py
- contracts.py
- load_verified_tokenizer
- test_tier1_features.py
- adversarial_r3_probe.py
- tests_e2e/conftest.py
- JSONLRepairDataset
- TestGemma4EdgeTopology
- test_training_telemetry_v2.py
- prepare_context_data.py
- test_molab_remediation.py
- patch_cross_entropy
- BifocalGemmaForCausalLM
- test_hf_pretrained_model.py
- test_v2_models.py
- test_benchmark_prompt_budget.py
- repository_study.py
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- ASTGraphProjector
- SwiGLUDualGateBridge
- harness.py
- adapters.py
- math
- fit_offline_sft
- audit_dataset_release.py
- repository_corpus.py
- test_data_remediation.py
- shared_execution.py
- TestBifocalGemmaArchitecture
- matrix.py
- optimizers.py
- RepairExample
- test_complete_epoch_training_budget_all_families
- test_colab_pilot.py
- generate_repository_notebook.py
- calibrate_shared_profile
- RepositorySnapshot
- test_construction_blocks.py
- generate_training_notebooks.py
- curation.py
- build_native_tiny_model
- run
- releases.py
- ast
- typing
- molab_matrix_support.py
- BifocalGemmaForCausalLM
- bifocalgemma
- TestCheckpointTransfer
- v2.py
- RepoContextRouter
- seal_report
- telemetry.py
- reference_resilient_mcq_parser
- RepairPrediction
- TestNotebookCellParity
- benchmark_upload.py
- evaluation/context.py
- profile_repository_lengths.py
- test_fast_pipeline.py
- TokenizedRepairExample
- bifocalgemma/__init__.py
- modeling_bifocalgemma.py
- patch_matrix_tokenization

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 84 edges
2. `stable_fingerprint()` - 67 edges
3. `load_verified_tokenizer()` - 57 edges
4. `TokenizedRepairExample` - 52 edges
5. `fit_offline_sft()` - 47 edges
6. `_execute_v2()` - 45 edges
7. `build_native_tiny_model()` - 43 edges
8. `configure_memory_efficient_training()` - 43 edges
9. `run_matrix_training()` - 41 edges
10. `RepairCollator` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/source_identity.py
- `publish()` --indirect_call--> `profile()`  [INFERRED]
  tests/test_shared_execution.py → tools/profile_repository_lengths.py

## Import Cycles
- None detected.

## Communities (118 total, 37 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.12
Nodes (7): test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads(), test_telemetry_history_appends_all_active_stages()

### Community 2 - "curation.py"
Cohesion: 0.12
Nodes (13): read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+5 more)

### Community 3 - "atomic_json"
Cohesion: 0.10
Nodes (18): activity(), reset_local_benchmarks(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint(), atomic_json() (+10 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.05
Nodes (40): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), measure(), digest(), Journal (+32 more)

### Community 5 - "copy"
Cohesion: 0.10
Nodes (20): validate_optimizer_config(), build_training_groups(), default_training_profiles(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), comparison_recipes(), resolve_training_cadence() (+12 more)

### Community 6 - "FeatureNotReadyError"
Cohesion: 0.07
Nodes (10): restore_checkpoint(), save_checkpoint(), EncoderMemory, RepositoryCache, BifocalGemmaModel, build_pretrained_model(), BuildBlock, FeatureNotReadyError (+2 more)

### Community 8 - "test_shared_execution.py"
Cohesion: 0.13
Nodes (11): seal_report(), completed_evidence(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch() (+3 more)

### Community 9 - "audit_context_release.py"
Cohesion: 0.13
Nodes (11): repair_preimage_coverage(), task_prompt(), tokenize_context_task(), ArchiveSnapshotStore, test_repair_requires_visible_preimage_without_refilling_context(), test_failed_download_removes_partial_bytes(), audit_pool(), audit_views() (+3 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.13
Nodes (4): Bf16FullLinear, FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.06
Nodes (19): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+11 more)

### Community 12 - "subprocess"
Cohesion: 0.08
Nodes (8): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), test_notebook_stress_helpers_match_shared_library(), main(), main(), main(), run_cmd(), main(), upload_in_chunks()

### Community 13 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (29): get_v2_experiment(), cached_logits(), configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), configure_matrix_stage(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes() (+21 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.17
Nodes (10): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook() (+2 more)

### Community 16 - "benchmark.py"
Cohesion: 0.09
Nodes (19): decode_text(), prompt_ids(), run_context_benchmark(), score_context_case(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids() (+11 more)

### Community 17 - "v2_preflight.py"
Cohesion: 0.21
Nodes (7): current_capacity(), _gpu_context(), maximum_probe(), _measure_phase(), select_execution(), emit(), save()

### Community 18 - "examples"
Cohesion: 0.05
Nodes (22): make_identity(), examples(), TinyLossModel, test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit(), test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored() (+14 more)

### Community 19 - "native.py"
Cohesion: 0.07
Nodes (8): NormalizedAlignment, build_joint_visibility(), ReferenceMergedAttention, build_native_model(), native_config(), NativeMergedAttention, RoutedRepoBatch, TestMergedAttentionReference

### Community 20 - "json"
Cohesion: 0.10
Nodes (11): compare_predictions(), remote_reset_plan(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_paired_bootstrap_and_cohort_rejection(), audit(), main(), main() (+3 more)

### Community 21 - "load_verified_tokenizer"
Cohesion: 0.20
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): completion_metrics(), extract_mcq_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting() (+15 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.11
Nodes (16): repository_task_from_row(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions, ScriptedModel (+8 more)

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.29
Nodes (11): code_shingles(), CodeOverlapIndex, CodeOverlapMatch, test_disk_index_matches_memory_and_syntax_repairs_are_admitted(), test_full_overlap_index_skips_impossible_blob_reads(), test_near_matching_and_short_fragment_policy(), test_overlap_bound_preserves_near_matches_and_heuristic_index(), test_readonly_partial_postings_never_use_full_intersection_bound() (+3 more)

### Community 25 - "test_memory_efficient_training.py"
Cohesion: 0.22
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 26 - "config.py"
Cohesion: 0.10
Nodes (15): BackboneProfile, ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan (+7 more)

### Community 27 - "hashlib"
Cohesion: 0.13
Nodes (15): curated_pool_digest(), editing_manifest(), editing_record(), git_text_lines(), prepare_editing_records(), repository_editing_record(), notebook_function(), repository_row() (+7 more)

### Community 28 - "torch"
Cohesion: 0.14
Nodes (13): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, sample_encoder_mass() (+5 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.10
Nodes (3): FastTrainingPipeline, run_fast_training(), configure_probe_training()

### Community 30 - "calibration"
Cohesion: 0.13
Nodes (11): guard(), examples(), fixture_guard(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure() (+3 more)

### Community 31 - "stable_fingerprint"
Cohesion: 0.30
Nodes (9): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+1 more)

### Community 34 - "matrix.py"
Cohesion: 0.10
Nodes (13): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+5 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 36 - "contracts.py"
Cohesion: 0.20
Nodes (6): extract_token_ast_features(), TokenASTFeatures, ContextWindow, render_window(), select_repository_context(), _terms()

### Community 37 - "load_verified_tokenizer"
Cohesion: 0.10
Nodes (9): ensure_assets_on_disk(), load_asset_json(), CheckpointRef, split_editing_records(), load_verified_tokenizer(), tokenize_repair_example(), prepare_repository_training(), assets() (+1 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.14
Nodes (6): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 40 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 41 - "JSONLRepairDataset"
Cohesion: 0.11
Nodes (5): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 43 - "test_training_telemetry_v2.py"
Cohesion: 0.20
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 44 - "prepare_context_data.py"
Cohesion: 0.17
Nodes (13): normalize_context_task(), load_curated_pool(), validate_context_license_record(), test_context_license_record_covers_all_visible_paths(), code_fragment_signatures(), completion_from_snapshot(), context_pool(), log() (+5 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "patch_cross_entropy"
Cohesion: 0.21
Nodes (3): __getattr__(), chunked_linear_cross_entropy(), patch_cross_entropy()

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.09
Nodes (16): BifocalGemmaConfig, export_matrix_pretrained_model(), _resolve_backbone_weights(), ProbeLoRALinear, test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload() (+8 more)

### Community 49 - "test_v2_models.py"
Cohesion: 0.20
Nodes (3): test_alignment_initialization_with_cuda_default_device(), test_alignment_stays_fp32_after_bf16_conversion(), threads()

### Community 50 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 51 - "repository_study.py"
Cohesion: 0.17
Nodes (22): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), GitSnapshotStore, prepare_repository_record(), read_source_rows() (+14 more)

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "math"
Cohesion: 0.15
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 58 - "fit_offline_sft"
Cohesion: 0.08
Nodes (22): run_training_memory_preflight(), fit_curated_sft(), prepare_curated_training(), cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit() (+14 more)

### Community 59 - "audit_dataset_release.py"
Cohesion: 0.52
Nodes (4): audit(), emit(), main(), validate_all_patches()

### Community 60 - "repository_corpus.py"
Cohesion: 0.10
Nodes (17): build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license(), RepositoryPreparationLedger (+9 more)

### Community 61 - "test_data_remediation.py"
Cohesion: 0.12
Nodes (22): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+14 more)

### Community 62 - "shared_execution.py"
Cohesion: 0.35
Nodes (7): publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), execution_settings(), verify_report()

### Community 64 - "matrix.py"
Cohesion: 0.06
Nodes (31): file_digest(), fetch_collection(), benchmark_directory_name(), checked_prefix(), download_model(), recover_remote_run(), RemoteHeartbeat, retry_delay() (+23 more)

### Community 65 - "optimizers.py"
Cohesion: 0.17
Nodes (6): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), benchmark_candidate(), probe_optimizer(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 66 - "RepairExample"
Cohesion: 0.15
Nodes (12): RepairExample, RepositoryFile, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, format_encoder_context() (+4 more)

### Community 67 - "test_complete_epoch_training_budget_all_families"
Cohesion: 0.40
Nodes (4): validate_curated_pool_limits(), test_complete_epoch_training_budget_all_families(), test_frozen_budget_matches_all_six_controls(), test_invalid_requested_pool_budget_is_rejected()

### Community 69 - "test_colab_pilot.py"
Cohesion: 0.13
Nodes (7): test_exported_worker_has_parity_context_and_exact_adapter_checks(), _DummyTokenizer, pilot_module(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), prepare_runtime()

### Community 70 - "generate_repository_notebook.py"
Cohesion: 0.47
Nodes (3): main(), render(), source_bundle()

### Community 72 - "calibrate_shared_profile"
Cohesion: 0.21
Nodes (8): calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe(), representative_cohort()

### Community 74 - "RepositorySnapshot"
Cohesion: 0.12
Nodes (19): RepositorySnapshot, canonical_snapshot_hash(), apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), patch_paths(), safe_repo_path(), example_from_record() (+11 more)

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.11
Nodes (6): Seq2SeqBatch, apply_word_embedding_policy(), EmbeddingPolicyResult, TinyReferenceModel, TestWordEmbeddingPolicy, test_fast_pipeline_batch_edge_cases()

### Community 77 - "generate_training_notebooks.py"
Cohesion: 0.43
Nodes (4): main(), module_definitions(), render(), use_embedded_imports()

### Community 78 - "curation.py"
Cohesion: 0.11
Nodes (21): code_signature(), code_tokens(), curate_paired_sources(), export_pool(), nextcoder_row(), count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target() (+13 more)

### Community 79 - "build_native_tiny_model"
Cohesion: 0.14
Nodes (4): build_native_tiny_model(), TestNativeIntegration, TestOfflineSFT, test_last_token_logits_match_full_head()

### Community 81 - "run"
Cohesion: 0.11
Nodes (13): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+5 more)

### Community 83 - "releases.py"
Cohesion: 0.08
Nodes (26): archive_files(), audit_split_disjointness(), build_release(), digest(), main(), publish_release(), rows(), unpack_archive() (+18 more)

### Community 84 - "ast"
Cohesion: 0.27
Nodes (6): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), main(), render_eval_script()

### Community 85 - "typing"
Cohesion: 0.12
Nodes (9): _compute_per_row_cross_query(), configure_full_rank_cross_attention(), build_bifocalgemma_v3_optimizer(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), clamped_triton_kernel_options(), critical_attention_beta(), ensure_hardware_memory_guards() (+1 more)

### Community 92 - "v2.py"
Cohesion: 0.07
Nodes (25): load_local_native_model(), load_sft_adapters_for_inference(), shared_contract(), source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), encoder_diagnostics() (+17 more)

### Community 95 - "seal_report"
Cohesion: 0.39
Nodes (4): test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), check_evidence(), main()

### Community 96 - "telemetry.py"
Cohesion: 0.11
Nodes (6): format_stage_display(), LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard, test_dashboard_stage_display_formatting()

### Community 101 - "reference_resilient_mcq_parser"
Cohesion: 0.15
Nodes (7): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline()

### Community 102 - "RepairPrediction"
Cohesion: 0.23
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 107 - "benchmark_upload.py"
Cohesion: 0.06
Nodes (20): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush() (+12 more)

### Community 108 - "evaluation/context.py"
Cohesion: 0.11
Nodes (12): build_repo_niah_case(), decoder_prompt(), _extract_match_letter(), fallback_letter_regex(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk() (+4 more)

### Community 110 - "profile_repository_lengths.py"
Cohesion: 0.32
Nodes (3): distribution(), main(), profile()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.09
Nodes (11): FastTrainingConfig, DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+3 more)

### Community 115 - "TokenizedRepairExample"
Cohesion: 0.05
Nodes (28): TokenizedRepairExample, RepairCollator, __getattr__(), build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, gpu_memory(), optimizer_bytes() (+20 more)

### Community 126 - "modeling_bifocalgemma.py"
Cohesion: 0.13
Nodes (3): transfer_text_checkpoint(), TiledEncoderAttention, TiledMergedAttention

### Community 134 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 934 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **37 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `prepare_colab_pilot.py`, `copy`, `audit_context_release.py`, `subprocess`, `test_molab_matrix_notebooks.py`, `benchmark.py`, `v2_preflight.py`, `examples`, `json`, `calibration`, `stable_fingerprint`, `matrix.py`, `prepare_context_data.py`, `fit_offline_sft`, `shared_execution.py`, `matrix.py`, `calibrate_shared_profile`, `run`, `v2.py`, `seal_report`, `benchmark_upload.py`, `evaluation/context.py`, `TokenizedRepairExample`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `FastTrainingPipeline` connect `FastTrainingPipeline` to `test_molab_matrix_notebooks.py`, `load_verified_tokenizer`, `JSONLRepairDataset`, `test_construction_blocks.py`, `BifocalGemmaForCausalLM`, `test_fast_pipeline.py`, `BifocalGemmaForCausalLM`, `config.py`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11695906432748537 - nodes in this community are weakly interconnected._
- **Why does `build_native_tiny_model()` connect `build_native_tiny_model` to `test_complete_epoch_training_budget_all_families`, `copy`, `test_molab_matrix_notebooks.py`, `curation.py`, `NativeTinyModel`, `BifocalGemmaForCausalLM`, `test_v2_models.py`, `examples`, `native.py`, `TokenizedRepairExample`, `run`, `test_memory_efficient_training.py`, `config.py`, `TestCheckpointTransfer`, `v2.py`, `calibration`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._