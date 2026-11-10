# Graph Report - bifocalgemma  (2026-11-11)

## Corpus Check
- 292 files · ~368,817 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 19 file(s) not represented in the graph (top: .tex 5, .ipynb 3, (none) 2)

## Summary
- 2441 nodes · 7079 edges · 118 communities (83 shown, 35 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 777 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f88d5f17`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- molab_pilot_support.py
- curation.py
- atomic_json
- prepare_colab_pilot.py
- gsm8k.py
- __main__.py
- RepositoryFile
- test_v2_fixed_execution.py
- json
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- v3/__init__.py
- test_molab_pilot.py
- NativeTinyModel
- re
- shared_execution.py
- fit_offline_sft
- v2/__init__.py
- tests_e2e/conftest.py
- os
- test_tier5_adversarial.py
- test_repository_study.py
- build_repository_corpus
- configure_memory_efficient_training
- FeatureNotReadyError
- test_editing_manifest.py
- torch
- FastTrainingPipeline
- calibration
- BifocalGemmaModel
- test_molab_matrix_notebooks.py
- v3/objectives.py
- prepare_colab_probe.py
- molab_matrix_support.py
- config.py
- test_tier1_features.py
- adversarial_r3_probe.py
- test_tier2_boundary.py
- fast_pipeline.py
- TestGemma4EdgeTopology
- test_training_telemetry_v2.py
- stable_fingerprint
- test_molab_remediation.py
- run_repo_niah_eval
- BifocalGemmaForCausalLM
- test_context_tasks.py
- test_context_benchmark_mcq_budget_floor_execution
- default_training_profiles
- repository_study.py
- RepairPrediction
- typing
- native.py
- packed_tokens.py
- adapters.py
- journaled_loss
- harness.py
- benchmark_upload.py
- repository_corpus.py
- remediation.py
- training/offline_sft.py
- TestBifocalGemmaArchitecture
- matrix.py
- policy.py
- apply_word_embedding_policy
- examples
- StageStore
- pathlib
- build_joint_visibility
- RepositorySnapshot
- RepairCollator
- digest
- CodeOverlapIndex
- build_native_tiny_model
- TestNotebookCellParity
- test_v2_workflow.py
- repository.py
- releases.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- SwiGLUDualGateBridge
- GatedCrossAttentionBridge
- BifocalGemmaForCausalLM
- bifocalgemma
- test_adversarial_mcq_isolated_formatting
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- atomic_json
- configure_probe_training
- telemetry.py
- FastTrainingConfig
- greedy_decode
- test_matrix_quarantine_legacy_checkpoint_vs_clean_resume
- test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues
- v2.py
- data/offline_sft.py
- test_context_benchmark_resume.py
- test_benchmark_prompt_budget.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
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
9. `RepairCollator` - 40 edges
10. `run_matrix_training()` - 40 edges

## Surprising Connections (you probably didn't know these)
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `save_report()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_pilot.py → src/bifocalgemma/training/stages.py
- `publish()` --indirect_call--> `profile()`  [INFERRED]
  tests/test_shared_execution.py → tools/profile_repository_lengths.py

## Import Cycles
- None detected.

## Communities (118 total, 35 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "molab_pilot_support.py"
Cohesion: 0.15
Nodes (7): test_completed_run_retries_upload_without_loading_model(), test_failed_final_upload_remains_pending(), test_upload_retry_verification_and_base_prefix_only(), PilotStorage, run(), _run(), write_json()

### Community 2 - "curation.py"
Cohesion: 0.16
Nodes (10): read_arrow_rows(), split_connected_records(), token_profile(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+2 more)

### Community 3 - "atomic_json"
Cohesion: 0.10
Nodes (25): load_sft_adapters_for_inference(), reset_local_benchmarks(), reset_state(), prepare_curated_training(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress() (+17 more)

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.10
Nodes (24): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair(), evaluate_humaneval_subset() (+16 more)

### Community 5 - "gsm8k.py"
Cohesion: 0.19
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+5 more)

### Community 6 - "__main__.py"
Cohesion: 0.12
Nodes (9): validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents, SFTConfig, train_direct_sft() (+1 more)

### Community 7 - "RepositoryFile"
Cohesion: 0.13
Nodes (11): RepositoryFile, apply_patch_to_snapshot(), extract_unified_diff(), git_environment(), GitSnapshotStore, patch_syntax(), test_dependency_completion_removes_target_and_retains_real_helper(), test_missing_old_file_is_still_rejected() (+3 more)

### Community 8 - "test_v2_fixed_execution.py"
Cohesion: 0.13
Nodes (9): test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(), test_profile_publish_parent_guard_keeps_concurrent_winner(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), test_shared_profile_upload_retries_without_losing_local_selection(), publish() (+1 more)

### Community 9 - "json"
Cohesion: 0.09
Nodes (13): ensure_assets_on_disk(), load_asset_json(), compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit(), emit(), main(), validate_all_patches() (+5 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.13
Nodes (5): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.06
Nodes (20): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+12 more)

### Community 12 - "subprocess"
Cohesion: 0.10
Nodes (12): NativeTaskResult, score_python_patch(), summarize_native_results(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), test_unittest_results_and_early_exit_do_not_false_pass(), audit(), main(), main() (+4 more)

### Community 13 - "v3/__init__.py"
Cohesion: 0.18
Nodes (7): TokenizedRepairExample, build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.16
Nodes (11): test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle(), render_notebook(), render(), main() (+3 more)

### Community 15 - "NativeTinyModel"
Cohesion: 0.15
Nodes (4): NativeTinyModel, test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 16 - "re"
Cohesion: 0.15
Nodes (16): repair_preimage_coverage(), task_prompt(), tokenize_context_task(), patch_paths(), repository_task_from_row(), test_repair_requires_visible_preimage_without_refilling_context(), audit_decoder_control(), code_fragment_signatures() (+8 more)

### Community 17 - "shared_execution.py"
Cohesion: 0.11
Nodes (24): build_policy_optimizer(), build_training_groups(), benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings() (+16 more)

### Community 18 - "fit_offline_sft"
Cohesion: 0.10
Nodes (15): examples(), TestOfflineSFT, TinyLossModel, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight() (+7 more)

### Community 19 - "v2/__init__.py"
Cohesion: 0.08
Nodes (6): NormalizedAlignment, ASTGraphProjector, native_config(), NativeMergedAttention, run_native_smoke(), RoutedRepoBatch

### Community 20 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 21 - "os"
Cohesion: 0.18
Nodes (10): normalize_context_task(), curated_pool_digest(), load_curated_pool(), validate_context_license_record(), test_context_license_record_covers_all_visible_paths(), audit_pool(), audit_views(), main() (+2 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.08
Nodes (17): test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling(), test_adversarial_mcq_ruling_out_distractors_vulnerability(), test_adversarial_mcq_split_ambiguity_behavior(), test_adversarial_mcq_verbose_reasoning_preambles() (+9 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.14
Nodes (13): prepare_repository_record(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions, test_context_budget_and_hidden_tests_are_not_rendered() (+5 more)

### Community 24 - "build_repository_corpus"
Cohesion: 0.19
Nodes (7): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples(), test_warning_has_provenance_without_stderr_or_source_mutation()

### Community 25 - "configure_memory_efficient_training"
Cohesion: 0.10
Nodes (24): cached_logits(), InferenceCache, configure_memory_efficient_training(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients() (+16 more)

### Community 26 - "FeatureNotReadyError"
Cohesion: 0.12
Nodes (7): extract_token_ast_features(), TokenASTFeatures, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError, distillation_loss()

### Community 27 - "test_editing_manifest.py"
Cohesion: 0.26
Nodes (10): editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical() (+2 more)

### Community 28 - "torch"
Cohesion: 0.11
Nodes (16): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+8 more)

### Community 30 - "calibration"
Cohesion: 0.27
Nodes (7): calibration(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure()

### Community 31 - "BifocalGemmaModel"
Cohesion: 0.19
Nodes (3): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.07
Nodes (21): unified_recipe(), make_experiment_spec(), build_bifocalgemma_v3_optimizer(), configure_matrix_stage(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support() (+13 more)

### Community 34 - "v3/objectives.py"
Cohesion: 0.09
Nodes (13): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+5 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 37 - "config.py"
Cohesion: 0.14
Nodes (9): BackboneProfile, CheckpointRef, ExperimentSpec, _pair(), _ref(), generate_patch(), repair_prompt_ids(), build_pretrained_model() (+1 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.13
Nodes (8): extract_mcq_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 40 - "test_tier2_boundary.py"
Cohesion: 0.09
Nodes (13): completion_metrics(), reference_resilient_mcq_parser(), validate_paper_manuscript(), test_f3_crosscodelong_continuous_metrics(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation() (+5 more)

### Community 41 - "fast_pipeline.py"
Cohesion: 0.13
Nodes (5): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 43 - "test_training_telemetry_v2.py"
Cohesion: 0.11
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), load_runtime()

### Community 44 - "stable_fingerprint"
Cohesion: 0.23
Nodes (9): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), stable_fingerprint(), test_f2_mcq_token_budget_specification() (+1 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.13
Nodes (5): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_mcq_budget_floor_in_v3_benchmark_logic()

### Community 46 - "run_repo_niah_eval"
Cohesion: 0.15
Nodes (8): decode_text(), build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), make_distractor_python_chunk(), run_repo_niah_eval(), score_context_case(), score_repo_niah_case()

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.09
Nodes (14): BifocalGemmaConfig, _resolve_backbone_weights(), ProbeLoRALinear, test_model_to_bfloat16_preserves_fp32_master_weights(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_bifocalgemma_merge_and_unload() (+6 more)

### Community 48 - "test_context_tasks.py"
Cohesion: 0.17
Nodes (7): fallback_letter_regex(), strict_letter(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 49 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 51 - "repository_study.py"
Cohesion: 0.21
Nodes (21): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), load_verified_tokenizer(), generate_diagnostic_text() (+13 more)

### Community 52 - "RepairPrediction"
Cohesion: 0.31
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 53 - "typing"
Cohesion: 0.07
Nodes (3): BifocalGemmaPreTrainedModel, CachedAttention, is_zero_weight_decay_bridge_parameter()

### Community 54 - "native.py"
Cohesion: 0.08
Nodes (14): transfer_text_checkpoint(), build_native_model(), load_local_native_model(), gpu_memory(), optimizer_bytes(), profile_training_group(), run_length_grid(), synthetic_stress_target() (+6 more)

### Community 55 - "packed_tokens.py"
Cohesion: 0.16
Nodes (5): build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "journaled_loss"
Cohesion: 0.29
Nodes (6): test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), measure(), journaled_loss(), evaluate(), event_callback()

### Community 59 - "benchmark_upload.py"
Cohesion: 0.09
Nodes (10): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos() (+2 more)

### Community 60 - "repository_corpus.py"
Cohesion: 0.08
Nodes (22): audit_repository_sources(), safe_repo_path(), ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus() (+14 more)

### Community 61 - "remediation.py"
Cohesion: 0.13
Nodes (21): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+13 more)

### Community 62 - "training/offline_sft.py"
Cohesion: 0.18
Nodes (10): cpu_tree(), evaluate_sft_loss(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), validation_batches(), optimizer_group_manifest(), update_policy_learning_rates() (+2 more)

### Community 64 - "matrix.py"
Cohesion: 0.06
Nodes (28): latest_checkpoint(), benchmark_directory_name(), target_fingerprint(), configure_full_rank_cross_attention(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), load_matrix_adapters_for_inference(), prune_matrix_checkpoints() (+20 more)

### Community 65 - "policy.py"
Cohesion: 0.09
Nodes (17): GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), Float32Alignment, validate_training_policy(), comparison_recipes(), resolve_training_cadence(), run_recipe_comparison() (+9 more)

### Community 66 - "apply_word_embedding_policy"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 67 - "examples"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 70 - "pathlib"
Cohesion: 0.12
Nodes (12): file_digest(), checked_prefix(), recover_remote_run(), _synchronize_once(), validate_checkpoint_files(), snapshot(), synchronize_preflight(), synchronize_v2() (+4 more)

### Community 72 - "build_joint_visibility"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 74 - "RepositorySnapshot"
Cohesion: 0.12
Nodes (13): RepairExample, RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, format_encoder_context() (+5 more)

### Community 76 - "RepairCollator"
Cohesion: 0.07
Nodes (10): ModelOutput, Seq2SeqBatch, RepairCollator, __getattr__(), run_reference_smoke(), TinyReferenceModel, __getattr__(), chunked_linear_cross_entropy() (+2 more)

### Community 77 - "digest"
Cohesion: 0.62
Nodes (3): test_journal_integrity_and_phase_isolation(), digest(), Journal

### Community 78 - "CodeOverlapIndex"
Cohesion: 0.13
Nodes (25): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), export_pool(), nextcoder_row() (+17 more)

### Community 79 - "build_native_tiny_model"
Cohesion: 0.09
Nodes (16): get_v2_experiment(), build_native_tiny_model(), configure_policy_model(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head() (+8 more)

### Community 81 - "test_v2_workflow.py"
Cohesion: 0.12
Nodes (12): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), test_training_failure_never_launches_benchmarks(), workflow(), benchmark() (+4 more)

### Community 82 - "repository.py"
Cohesion: 0.22
Nodes (8): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), validate_code_eval_snapshots(), example_from_record(), prepare_repository_record_with_snapshot()

### Community 83 - "releases.py"
Cohesion: 0.09
Nodes (26): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+18 more)

### Community 89 - "BifocalGemmaForCausalLM"
Cohesion: 0.10
Nodes (3): BifocalGemmaForCausalLM, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_pretrained_model_subfolder_support_and_repo_creation()

### Community 93 - "atomic_json"
Cohesion: 0.18
Nodes (12): decoder_prompt(), prompt_ids(), run_context_benchmark(), activity(), fit_offline_sft(), emit(), model_stage(), DurableClock (+4 more)

### Community 95 - "configure_probe_training"
Cohesion: 0.17
Nodes (5): export_matrix_pretrained_model(), configure_probe_training(), test_export_matrix_pretrained_model(), test_export_matrix_pretrained_model_baseline(), TestMemoryProbe

### Community 96 - "telemetry.py"
Cohesion: 0.08
Nodes (12): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, LastSeenWidget, phase_category(), TrainingLossLog (+4 more)

### Community 97 - "FastTrainingConfig"
Cohesion: 0.08
Nodes (12): FastTrainingConfig, DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+4 more)

### Community 100 - "greedy_decode"
Cohesion: 0.20
Nodes (6): greedy_decode(), greedy_decode(), GreedyGeneration, batched_greedy_decode(), ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards()

### Community 105 - "v2.py"
Cohesion: 0.08
Nodes (22): retry_delay(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), benchmark_jobs(), encoder_diagnostics() (+14 more)

### Community 106 - "data/offline_sft.py"
Cohesion: 0.09
Nodes (18): editing_record(), git_text_lines(), prepare_editing_records(), split_editing_records(), validate_curated_pool_limits(), run_training_memory_preflight(), fit_curated_sft(), tokenize() (+10 more)

### Community 107 - "test_context_benchmark_resume.py"
Cohesion: 0.13
Nodes (11): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization() (+3 more)

### Community 108 - "test_benchmark_prompt_budget.py"
Cohesion: 0.22
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 114 - "hashlib"
Cohesion: 0.11
Nodes (16): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), remote_reset_plan(), source_fingerprints(), WithoutDocstrings, test_remote_reset_is_scoped_and_receipt_hashes_match() (+8 more)

### Community 115 - "pytest"
Cohesion: 0.13
Nodes (9): test_exported_worker_has_parity_context_and_exact_adapter_checks(), _DummyTokenizer, pilot_module(), test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens() (+1 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 922 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `prepare_colab_pilot.py`, `json`, `subprocess`, `re`, `shared_execution.py`, `fit_offline_sft`, `os`, `configure_memory_efficient_training`, `calibration`, `v3/objectives.py`, `stable_fingerprint`, `run_repo_niah_eval`, `typing`, `packed_tokens.py`, `benchmark_upload.py`, `training/offline_sft.py`, `matrix.py`, `policy.py`, `examples`, `pathlib`, `test_v2_workflow.py`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `atomic_json`, `v2.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `v3/__init__.py` to `prepare_colab_pilot.py`, `re`, `shared_execution.py`, `fit_offline_sft`, `v2/__init__.py`, `configure_memory_efficient_training`, `test_editing_manifest.py`, `torch`, `FastTrainingPipeline`, `test_molab_matrix_notebooks.py`, `prepare_colab_probe.py`, `fast_pipeline.py`, `native.py`, `packed_tokens.py`, `journaled_loss`, `RepositorySnapshot`, `RepairCollator`, `build_native_tiny_model`, `configure_probe_training`, `v2.py`, `data/offline_sft.py`, `pytest`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `atomic_json` be split into smaller, more focused modules?**
  _Cohesion score 0.10365853658536585 - nodes in this community are weakly interconnected._
- **Why does `PackedTokenizedDataset` connect `packed_tokens.py` to `v3/__init__.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._