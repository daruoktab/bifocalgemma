# Graph Report - bifocalgemma  (2026-11-07)

## Corpus Check
- 290 files · ~364,808 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2352 nodes · 6924 edges · 116 communities (83 shown, 33 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 707 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `806f1241`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- remediation.py
- test_data_releases.py
- _execute_v2
- prepare_colab_pilot.py
- releases.py
- contracts.py
- test_editing_manifest.py
- typing
- repository.py
- BifocalGemmaConfig
- test_gsm8k.py
- subprocess
- stable_fingerprint
- prepare_colab_probe.py
- NativeTinyModel
- CodeOverlapIndex
- FastTrainingPipeline
- TinyLossModel
- molab_pilot_support.py
- test_memory_efficient_training.py
- prepare_context_data.py
- test_tier5_adversarial.py
- shared_execution.py
- remediation.py
- gsm8k.py
- contracts.py
- full_pool.py
- tiled_attention
- pytest
- test_molab_pilot.py
- hashlib
- Tensor
- matrix.py
- GatedCrossAttentionBridge
- TokenizedRepairExample
- repository_corpus.py
- test_tier1_features.py
- memory_probe.py
- test_context_benchmark_resume.py
- extract_mcq_letter
- adversarial_r3_probe.py
- ast
- test_curation.py
- test_molab_remediation.py
- swesmith.py
- completion_metrics
- benchmark_upload.py
- pathlib
- adapters.py
- repository_study.py
- build_bifocalgemma_v3_optimizer
- apply_word_embedding_policy
- RepoContextRouter
- examples
- test_training_telemetry_v2.py
- test_notebook_hf_flow.py
- transfer_text_checkpoint
- profile_repository_lengths.py
- PackedTokenizedDataset
- TrainingLossLog
- molab_matrix_support.py
- TestBifocalGemmaArchitecture
- test_molab_matrix_notebooks.py
- copy
- strict_letter
- tests_e2e/conftest.py
- molab_matrix_support.py
- __main__.py
- run_pilot_stage
- json
- fit_offline_sft
- random
- ASTGraphProjector
- run_length_grid
- TestNotebookCellParity
- benchmark.py
- SyntheticRepairDataset
- BenchmarkUploadQueue
- modeling_bifocalgemma.py
- v3/__init__.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_repository_study.py
- score_context_case
- TestGemma4EdgeTopology
- config.py
- telemetry.py
- test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention
- test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume
- WigglyDashboard
- Journal
- resolve_optimal_eval_batch_size
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos
- calibration
- sequential.py
- test_v2_workflow.py
- generate_repository_notebook.py
- atomic_json
- examples
- test_colab_probe.py
- test_adversarial_mcq_lowercase_letter_normalization
- test_adversarial_mcq_redos_scaling
- test_adversarial_mcq_ambiguity_guards

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
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `benchmark()` --calls--> `atomic_json()`  [INFERRED]
  tests/test_training_recipes.py → src/bifocalgemma/training/stages.py

## Import Cycles
- None detected.

## Communities (116 total, 33 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "remediation.py"
Cohesion: 0.12
Nodes (22): split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+14 more)

### Community 2 - "test_data_releases.py"
Cohesion: 0.15
Nodes (11): pool(), release(), test_archive_source_escape_rejected(), test_eval_public_and_gold_separate(), test_existing_local_pool_cannot_bypass_pinned_identity(), test_fetch_pinned_pool_roundtrip_and_reuse(), test_moving_revision_rejected_without_network(), test_publish_refuses_public_repo_before_upload() (+3 more)

### Community 3 - "_execute_v2"
Cohesion: 0.06
Nodes (35): fetch_collection(), recover_remote_run(), shared_contract(), source_fingerprints(), WithoutDocstrings, latest_checkpoint(), StageLedger, TrainingDashboard (+27 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.12
Nodes (19): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), encode_chat_pair(), evaluate_humaneval_subset(), evaluate_pilot_checkpoint(), evaluate_source_copy_probes(), extract_resolvable_span() (+11 more)

### Community 5 - "releases.py"
Cohesion: 0.13
Nodes (16): archive_files(), audit_split_disjointness(), build_release(), digest(), main(), publish_release(), rows(), unpack_archive() (+8 more)

### Community 6 - "contracts.py"
Cohesion: 0.09
Nodes (10): EvaluationReport, extract_token_ast_features(), TokenASTFeatures, SWEBenchHarness, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError (+2 more)

### Community 7 - "test_editing_manifest.py"
Cohesion: 0.23
Nodes (10): editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical() (+2 more)

### Community 8 - "typing"
Cohesion: 0.06
Nodes (8): BackboneProfile, CheckpointRef, _pair(), _ref(), native_config(), RoutedRepoBatch, chunked_linear_cross_entropy(), test_notebook_stress_helpers_match_shared_library()

### Community 9 - "repository.py"
Cohesion: 0.09
Nodes (25): RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), apply_patch_to_snapshot(), extract_unified_diff() (+17 more)

### Community 10 - "BifocalGemmaConfig"
Cohesion: 0.09
Nodes (13): BifocalGemmaConfig, _resolve_backbone_weights(), ProbeLoRALinear, test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip() (+5 more)

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (14): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting, FakeTokenizer, fixture_rows() (+6 more)

### Community 12 - "subprocess"
Cohesion: 0.09
Nodes (11): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), audit(), emit(), main(), validate_all_patches(), main(), main(), main() (+3 more)

### Community 13 - "stable_fingerprint"
Cohesion: 0.30
Nodes (9): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+1 more)

### Community 14 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 15 - "NativeTinyModel"
Cohesion: 0.15
Nodes (5): ModelOutput, Seq2SeqBatch, NativeTinyModel, TinyReferenceModel, test_fast_pipeline_batch_edge_cases()

### Community 16 - "CodeOverlapIndex"
Cohesion: 0.29
Nodes (11): code_shingles(), CodeOverlapIndex, CodeOverlapMatch, test_disk_index_matches_memory_and_syntax_repairs_are_admitted(), test_full_overlap_index_skips_impossible_blob_reads(), test_near_matching_and_short_fragment_policy(), test_overlap_bound_preserves_near_matches_and_heuristic_index(), test_readonly_partial_postings_never_use_full_intersection_bound() (+3 more)

### Community 17 - "FastTrainingPipeline"
Cohesion: 0.11
Nodes (7): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_fast_training_pipeline_features_and_export(), test_pretrained_model_subfolder_support_and_repo_creation(), test_fast_training_pipeline_baseline_2b(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 18 - "TinyLossModel"
Cohesion: 0.10
Nodes (5): TestOfflineSFT, TinyLossModel, test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit(), test_training_resume_keeps_elapsed_from_step_five_through_ten(), test_checkpoint_directory_publish_retries_windows_sharing_violation()

### Community 19 - "molab_pilot_support.py"
Cohesion: 0.14
Nodes (11): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), digest(), journaled_loss(), evaluate(), PilotStorage, run(), _run() (+3 more)

### Community 20 - "test_memory_efficient_training.py"
Cohesion: 0.27
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 21 - "prepare_context_data.py"
Cohesion: 0.09
Nodes (21): normalize_context_task(), repair_preimage_coverage(), task_prompt(), tokenize_context_task(), curated_pool_digest(), load_curated_pool(), task(), test_dependency_completion_removes_target_and_retains_real_helper() (+13 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.11
Nodes (11): test_adversarial_context_containing_forbidden_phrases(), test_adversarial_metrics_disjoint_identifiers(), test_adversarial_metrics_identical_sequences(), test_adversarial_metrics_keyword_identifier_inflation(), test_adversarial_metrics_multiline_diff_inflation(), test_adversarial_metrics_pure_operator_zero_division_guard(), test_adversarial_metrics_unicode_identifiers_blind_spot(), test_adversarial_prompt_neutrality_matrix() (+3 more)

### Community 23 - "shared_execution.py"
Cohesion: 0.05
Nodes (40): retry_delay(), build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), calibrate_shared_profile(), emit(), save() (+32 more)

### Community 24 - "remediation.py"
Cohesion: 0.14
Nodes (14): code_signature(), code_tokens(), curate_paired_sources(), nextcoder_row(), count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records() (+6 more)

### Community 25 - "gsm8k.py"
Cohesion: 0.13
Nodes (17): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+9 more)

### Community 26 - "contracts.py"
Cohesion: 0.10
Nodes (15): DeveloperAgent, RepairExample, RepairPrediction, RepositoryFile, decode_text(), format_encoder_context(), tokenize_repair_example(), generate_diagnostic_text() (+7 more)

### Community 27 - "full_pool.py"
Cohesion: 0.18
Nodes (9): read_arrow_rows(), token_profile(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main(), progress() (+1 more)

### Community 28 - "tiled_attention"
Cohesion: 0.06
Nodes (18): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+10 more)

### Community 29 - "pytest"
Cohesion: 0.23
Nodes (6): build_packed_cache(), open_packed_cache(), compare_predictions(), test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes(), test_paired_bootstrap_and_cohort_rejection()

### Community 30 - "test_molab_pilot.py"
Cohesion: 0.17
Nodes (11): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+3 more)

### Community 31 - "hashlib"
Cohesion: 0.14
Nodes (15): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), prepare_editing_records(), split_editing_records() (+7 more)

### Community 32 - "Tensor"
Cohesion: 0.10
Nodes (12): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 34 - "matrix.py"
Cohesion: 0.14
Nodes (11): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix_v3(), run_matrix_stage(), is_cross_bridge_parameter() (+3 more)

### Community 36 - "TokenizedRepairExample"
Cohesion: 0.12
Nodes (10): TokenizedRepairExample, RepairCollator, __getattr__(), write_swebench_predictions(), run_native_smoke(), run_reference_smoke(), __getattr__(), patch_cross_entropy() (+2 more)

### Community 37 - "repository_corpus.py"
Cohesion: 0.08
Nodes (24): audit_repository_sources(), validate_context_license_record(), ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus() (+16 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 39 - "memory_probe.py"
Cohesion: 0.12
Nodes (11): export_matrix_pretrained_model(), configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer() (+3 more)

### Community 40 - "test_context_benchmark_resume.py"
Cohesion: 0.18
Nodes (5): FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch(), transient()

### Community 41 - "extract_mcq_letter"
Cohesion: 0.17
Nodes (6): extract_mcq_letter(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_ruling_out_distractors_vulnerability(), test_adversarial_mcq_split_ambiguity_behavior(), test_adversarial_mcq_verbose_reasoning_preambles()

### Community 42 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 43 - "ast"
Cohesion: 0.24
Nodes (6): main(), module_definitions(), render(), use_embedded_imports(), main(), render_eval_script()

### Community 44 - "test_curation.py"
Cohesion: 0.21
Nodes (11): export_pool(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_curated_resume_rejects_runtime_drift(), test_frozen_budget_matches_all_six_controls(), test_frozen_pool_verifies_checksums_before_use(), test_invalid_requested_pool_budget_is_rejected() (+3 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_embedded_runtime_extraction_from_fast_notebook(), test_embedded_runtime_extraction_from_real_matrix_notebook(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "swesmith.py"
Cohesion: 0.21
Nodes (7): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 47 - "completion_metrics"
Cohesion: 0.09
Nodes (13): completion_metrics(), reference_resilient_mcq_parser(), validate_paper_manuscript(), test_f3_crosscodelong_continuous_metrics(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation() (+5 more)

### Community 48 - "benchmark_upload.py"
Cohesion: 0.23
Nodes (6): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), upload()

### Community 49 - "pathlib"
Cohesion: 0.10
Nodes (9): file_digest(), checked_prefix(), _synchronize_once(), validate_checkpoint_files(), save_guard(), synchronize_preflight(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend() (+1 more)

### Community 50 - "adapters.py"
Cohesion: 0.23
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 51 - "repository_study.py"
Cohesion: 0.25
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+10 more)

### Community 53 - "apply_word_embedding_policy"
Cohesion: 0.14
Nodes (5): apply_word_embedding_policy(), EmbeddingPolicyResult, ReferenceMergedAttention, TestMergedAttentionReference, TestWordEmbeddingPolicy

### Community 55 - "examples"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 56 - "test_training_telemetry_v2.py"
Cohesion: 0.24
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 57 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 59 - "profile_repository_lengths.py"
Cohesion: 0.28
Nodes (5): test_shared_profile_upload_retries_without_losing_local_selection(), publish(), distribution(), main(), profile()

### Community 64 - "test_molab_matrix_notebooks.py"
Cohesion: 0.09
Nodes (8): prepare_sampled_benchmark_collection(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_prepare_sampled_benchmark_collection_deterministic_100_cases(), test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads(), test_telemetry_history_appends_all_active_stages()

### Community 65 - "copy"
Cohesion: 0.08
Nodes (30): get_v2_experiment(), cached_logits(), configure_memory_efficient_training(), build_native_tiny_model(), configure_policy_model(), unified_recipe(), benchmark_candidate(), probe_optimizer() (+22 more)

### Community 69 - "strict_letter"
Cohesion: 0.29
Nodes (4): fallback_letter_regex(), strict_letter(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 70 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 71 - "molab_matrix_support.py"
Cohesion: 0.13
Nodes (7): test_exported_worker_has_parity_context_and_exact_adapter_checks(), _DummyTokenizer, pilot_module(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), prepare_runtime()

### Community 72 - "__main__.py"
Cohesion: 0.12
Nodes (9): validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents, SFTConfig, train_direct_sft() (+1 more)

### Community 73 - "run_pilot_stage"
Cohesion: 0.25
Nodes (5): configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), is_cross_bridge_parameter(), run_pilot_stage()

### Community 74 - "json"
Cohesion: 0.08
Nodes (15): remote_reset_plan(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_external_activity_reader_does_not_need_running_kernel(), TestTokenizerVerification, audit_views(), main(), audit() (+7 more)

### Community 75 - "fit_offline_sft"
Cohesion: 0.15
Nodes (17): run_context_benchmark(), cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint() (+9 more)

### Community 76 - "random"
Cohesion: 0.24
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 78 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 80 - "benchmark.py"
Cohesion: 0.09
Nodes (5): CachedAttention, Bf16FullLinear, configure_full_rank_cross_attention(), FullCrossCachedAttention, SwiGLUDualGateBridge

### Community 81 - "SyntheticRepairDataset"
Cohesion: 0.15
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 82 - "BenchmarkUploadQueue"
Cohesion: 0.18
Nodes (6): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed()

### Community 84 - "v3/__init__.py"
Cohesion: 0.11
Nodes (10): _compute_per_row_cross_query(), clamped_triton_kernel_options(), critical_attention_beta(), ensure_hardware_memory_guards(), format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task() (+2 more)

### Community 91 - "test_repository_study.py"
Cohesion: 0.12
Nodes (19): prepare_repository_record(), repository_task_from_row(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions (+11 more)

### Community 93 - "score_context_case"
Cohesion: 0.10
Nodes (13): build_repo_niah_case(), decoder_prompt(), format_crosscodelong_summary(), generate_repo_niah_haystack(), greedy_decode(), make_distractor_python_chunk(), prompt_ids(), run_repo_niah_eval() (+5 more)

### Community 95 - "config.py"
Cohesion: 0.15
Nodes (5): restore_checkpoint(), save_checkpoint(), ExperimentSpec, BifocalGemmaModel, build_pretrained_model()

### Community 96 - "telemetry.py"
Cohesion: 0.16
Nodes (6): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), test_monitor_escapes_errors_updates_validation_and_exports_html(), test_dashboard_stage_display_formatting()

### Community 97 - "test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 101 - "resolve_optimal_eval_batch_size"
Cohesion: 0.40
Nodes (3): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 105 - "calibration"
Cohesion: 0.21
Nodes (10): test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), calibration(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), measure() (+2 more)

### Community 106 - "sequential.py"
Cohesion: 0.08
Nodes (16): run_training_memory_preflight(), NormalizedAlignment, reset_local_benchmarks(), fit_curated_sft(), prepare_curated_training(), synchronize_run(), Float32Alignment, update_policy_learning_rates() (+8 more)

### Community 107 - "test_v2_workflow.py"
Cohesion: 0.20
Nodes (7): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), run()

### Community 108 - "generate_repository_notebook.py"
Cohesion: 0.47
Nodes (3): main(), render(), source_bundle()

### Community 109 - "atomic_json"
Cohesion: 0.11
Nodes (17): activity(), atomic_json(), append_event(), progress(), activity(), workflow(), benchmark(), download() (+9 more)

### Community 111 - "examples"
Cohesion: 0.07
Nodes (27): build_training_groups(), default_training_profiles(), comparison_recipes(), make_identity(), examples(), test_inference_v2_adapter_roundtrip_and_baseline_group(), test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory() (+19 more)

### Community 112 - "test_colab_probe.py"
Cohesion: 0.70
Nodes (3): boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 880 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `_execute_v2`, `prepare_colab_pilot.py`, `repository.py`, `stable_fingerprint`, `TinyLossModel`, `prepare_context_data.py`, `shared_execution.py`, `pytest`, `matrix.py`, `benchmark_upload.py`, `pathlib`, `examples`, `test_molab_matrix_notebooks.py`, `copy`, `run_pilot_stage`, `json`, `fit_offline_sft`, `benchmark.py`, `BenchmarkUploadQueue`, `score_context_case`, `test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `calibration`, `sequential.py`, `test_v2_workflow.py`, `examples`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `prepare_colab_pilot.py`, `test_editing_manifest.py`, `typing`, `prepare_colab_probe.py`, `test_memory_efficient_training.py`, `prepare_context_data.py`, `shared_execution.py`, `contracts.py`, `tiled_attention`, `pytest`, `hashlib`, `memory_probe.py`, `test_curation.py`, `PackedTokenizedDataset`, `copy`, `molab_matrix_support.py`, `random`, `SyntheticRepairDataset`, `modeling_bifocalgemma.py`, `calibration`, `sequential.py`, `examples`, `test_colab_probe.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Should `remediation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Why does `test_one_model_training_stops_and_completed_resume_does_not_reload()` connect `examples` to `pathlib`, `examples`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._