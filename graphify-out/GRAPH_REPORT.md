# Graph Report - bifocalgemma  (2026-11-11)

## Corpus Check
- 294 files · ~375,624 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2445 nodes · 7086 edges · 143 communities (97 shown, 46 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 777 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5242724f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- molab_pilot_support.py
- curation.py
- run_sequential_pipeline
- prepare_colab_pilot.py
- gsm8k.py
- __main__.py
- tempfile
- test_v2_fixed_execution.py
- json
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- v3/__init__.py
- test_molab_pilot.py
- NativeTinyModel
- re
- calibrate_shared_profile
- pytest
- v2/__init__.py
- tests_e2e/conftest.py
- atomic_json
- test_tier5_adversarial.py
- test_repository_study.py
- remediation.py
- test_memory_efficient_training.py
- FeatureNotReadyError
- test_editing_manifest.py
- torch
- .run
- calibration
- checkpointing.py
- test_molab_matrix_notebooks.py
- v3/objectives.py
- prepare_colab_probe.py
- RepairExample
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
- configure_memory_efficient_training
- default_training_profiles
- repository_study.py
- RepairPrediction
- typing
- memory_probe.py
- PackedTokenizedDataset
- adapters.py
- journaled_loss
- harness.py
- benchmark_upload.py
- repository_corpus.py
- remediation.py
- v2_preflight.py
- TestBifocalGemmaArchitecture
- matrix.py
- pathlib
- apply_word_embedding_policy
- examples
- get_v2_experiment
- pathlib
- test_v2_models.py
- swesmith.py
- RepairCollator
- BifocalGemmaConfig
- CodeOverlapIndex
- build_native_tiny_model
- v3/__init__.py
- run
- repository.py
- releases.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- SwiGLUDualGateBridge
- GatedCrossAttentionBridge
- BifocalGemmaForCausalLM
- bifocalgemma
- shutil
- RepositoryFile
- atomic_json
- benchmark_upload.py
- telemetry.py
- FastTrainingConfig
- greedy_decode
- shared_execution.py
- test_tier3_combinations.py
- v2.py
- editing_record
- test_context_benchmark_resume.py
- test_benchmark_prompt_budget.py
- test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit
- ast
- TokenizedRepairExample
- build_repository_corpus
- test_tier4_scenarios.py
- packed_tokens.py
- curation.py
- pilot_e2b_e2b
- native_config
- RepoContextRouter
- bifocalgemma/__init__.py
- test_notebook_hf_flow.py
- workflow
- ASTGraphProjector
- scaffold.py
- check_followers
- batched_greedy_decode
- test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract
- DummyTrackingTokenizer
- TrainingLossLog
- WigglyDashboard
- run_length_grid
- ast.py
- transfer_text_checkpoint
- CachedAttention
- configure_full_rank_cross_attention

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
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/v3/bridge.py
- `test_telemetry_history_appends_all_active_stages()` --uses--> `TrainingDashboard`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/telemetry.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes()` --calls--> `source_fingerprints()`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/source_identity.py

## Import Cycles
- None detected.

## Communities (143 total, 46 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.07
Nodes (19): pilot_module(), simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt() (+11 more)

### Community 1 - "molab_pilot_support.py"
Cohesion: 0.15
Nodes (10): test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), digest(), Journal, PilotStorage, run(), _run() (+2 more)

### Community 2 - "curation.py"
Cohesion: 0.16
Nodes (14): export_pool(), read_arrow_rows(), split_connected_records(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+6 more)

### Community 3 - "run_sequential_pipeline"
Cohesion: 0.23
Nodes (8): synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint(), StageLedger, test_reset_archives_evaluation_once_without_changing_training(), test_stage_receipt_cannot_skip_changed_or_missing_evidence()

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.15
Nodes (15): build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_pilot_stage(), encode_chat_pair(), extract_resolvable_span(), extract_single_window_solution_repair(), is_cross_bridge_parameter() (+7 more)

### Community 5 - "gsm8k.py"
Cohesion: 0.18
Nodes (13): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_run(), gsm8k_prompt_ids(), load_gsm8k(), main() (+5 more)

### Community 6 - "__main__.py"
Cohesion: 0.09
Nodes (15): ExperimentSpec, _pair(), _ref(), validate_experiment_matrix(), main(), describe_initialization(), InitializationPlan, initialize_pretrained_components() (+7 more)

### Community 7 - "tempfile"
Cohesion: 0.16
Nodes (12): apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), NativeTaskResult, score_python_patch(), summarize_native_results(), patch_syntax() (+4 more)

### Community 8 - "test_v2_fixed_execution.py"
Cohesion: 0.12
Nodes (14): seal_report(), save(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline() (+6 more)

### Community 9 - "json"
Cohesion: 0.09
Nodes (18): remote_reset_plan(), reset_local_benchmarks(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_external_activity_reader_does_not_need_running_kernel(), audit(), emit(), main() (+10 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.13
Nodes (5): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.08
Nodes (15): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), assets(), TestEvaluationReporting, FakeTokenizer (+7 more)

### Community 12 - "subprocess"
Cohesion: 0.13
Nodes (6): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), main(), main(), main(), run_cmd(), upload_in_chunks()

### Community 13 - "v3/__init__.py"
Cohesion: 0.24
Nodes (6): build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_comprehension_targets_stay_inside_baseline_visible_prefix()

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.16
Nodes (11): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+3 more)

### Community 16 - "re"
Cohesion: 0.11
Nodes (18): normalize_context_task(), repair_preimage_coverage(), task_prompt(), tokenize_context_task(), load_curated_pool(), validate_context_license_record(), task(), test_eval_role_and_invisible_localization_target_rejected() (+10 more)

### Community 17 - "calibrate_shared_profile"
Cohesion: 0.12
Nodes (14): benchmark_candidate(), calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe() (+6 more)

### Community 18 - "pytest"
Cohesion: 0.08
Nodes (6): TinyLossModel, test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight(), test_checkpoint_directory_publish_retries_windows_sharing_violation(), test_training_failure_never_launches_benchmarks()

### Community 19 - "v2/__init__.py"
Cohesion: 0.09
Nodes (6): NormalizedAlignment, build_joint_visibility(), ReferenceMergedAttention, build_native_model(), RoutedRepoBatch, TestMergedAttentionReference

### Community 20 - "tests_e2e/conftest.py"
Cohesion: 0.15
Nodes (6): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 21 - "atomic_json"
Cohesion: 0.17
Nodes (8): atomic_json(), retry_windows_file_operation(), benchmark(), audit_pool(), audit_views(), main(), main(), source_notices()

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.09
Nodes (16): extract_mcq_letter(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling(), test_adversarial_mcq_ruling_out_distractors_vulnerability(), test_adversarial_mcq_split_ambiguity_behavior() (+8 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.10
Nodes (18): prepare_repository_record(), repository_task_from_row(), RepositoryTask, bounded_threads(), CharacterTokenizer, diff(), ExecutingScoringOptions, fixture() (+10 more)

### Community 24 - "remediation.py"
Cohesion: 0.11
Nodes (16): count_syntax_warnings(), empty_syntax_warning_summary(), inspect_python_target(), summarize_admitted_warning_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure() (+8 more)

### Community 25 - "test_memory_efficient_training.py"
Cohesion: 0.24
Nodes (11): examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native(), test_optimized_long_window_causality_context_and_all_masked_backward() (+3 more)

### Community 26 - "FeatureNotReadyError"
Cohesion: 0.17
Nodes (3): EncoderMemory, RepositoryCache, FeatureNotReadyError

### Community 27 - "test_editing_manifest.py"
Cohesion: 0.26
Nodes (10): editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical() (+2 more)

### Community 28 - "torch"
Cohesion: 0.08
Nodes (20): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+12 more)

### Community 30 - "calibration"
Cohesion: 0.15
Nodes (11): TestTokenizerVerification, calibration(), guard(), examples(), test_guard_failure_is_durable_and_rolls_back_probe_parameters(), test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(), test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(), measure() (+3 more)

### Community 32 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (10): critical_attention_beta(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_centralized_hf_repo(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads() (+2 more)

### Community 34 - "v3/objectives.py"
Cohesion: 0.09
Nodes (12): chunked_differential_cross_entropy(), compute_diff_token_weights(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.29
Nodes (4): main(), declared_limit_example(), main(), render()

### Community 37 - "RepairExample"
Cohesion: 0.22
Nodes (9): RepairExample, decode_text(), format_encoder_context(), generate_gsm8k(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration (+1 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.13
Nodes (10): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_compatibility_hash_sync(), test_f6_repo_core_syntax_tests_executable(), test_f7_dataset_checkpoint_contract() (+2 more)

### Community 39 - "adversarial_r3_probe.py"
Cohesion: 0.11
Nodes (9): score_repo_niah_case(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases() (+1 more)

### Community 40 - "test_tier2_boundary.py"
Cohesion: 0.13
Nodes (9): completion_metrics(), reference_resilient_mcq_parser(), test_f3_crosscodelong_continuous_metrics(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement() (+1 more)

### Community 41 - "fast_pipeline.py"
Cohesion: 0.14
Nodes (5): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 43 - "test_training_telemetry_v2.py"
Cohesion: 0.20
Nodes (4): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 44 - "stable_fingerprint"
Cohesion: 0.17
Nodes (14): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), curated_pool_digest(), stable_fingerprint() (+6 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.06
Nodes (10): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+2 more)

### Community 46 - "run_repo_niah_eval"
Cohesion: 0.15
Nodes (6): build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), make_distractor_python_chunk(), run_repo_niah_eval(), test_adversarial_reponiah_deterministic_reproducibility()

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.10
Nodes (13): export_matrix_pretrained_model(), _resolve_backbone_weights(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_enable_memory_efficient_training_and_softcapping(), test_export_matrix_pretrained_model() (+5 more)

### Community 48 - "test_context_tasks.py"
Cohesion: 0.29
Nodes (4): fallback_letter_regex(), strict_letter(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 49 - "configure_memory_efficient_training"
Cohesion: 0.14
Nodes (21): CheckpointRef, configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), configure_matrix_stage(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence() (+13 more)

### Community 50 - "default_training_profiles"
Cohesion: 0.12
Nodes (8): make_identity(), examples(), TestOfflineSFT, test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(), factory(), prepare(), TestTrainingPolicy, test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 51 - "repository_study.py"
Cohesion: 0.18
Nodes (21): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), load_verified_tokenizer(), load_local_native_model(), fetch_tokenizer_assets() (+13 more)

### Community 52 - "RepairPrediction"
Cohesion: 0.23
Nodes (3): DeveloperAgent, RepairPrediction, write_swebench_predictions()

### Community 54 - "memory_probe.py"
Cohesion: 0.22
Nodes (7): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), TestMemoryProbe

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "journaled_loss"
Cohesion: 0.25
Nodes (7): test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), measure(), journaled_loss(), evaluate(), event_callback(), event()

### Community 59 - "benchmark_upload.py"
Cohesion: 0.12
Nodes (7): BenchmarkUploadQueue, test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos()

### Community 60 - "repository_corpus.py"
Cohesion: 0.11
Nodes (15): ArchiveSnapshotStore, context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), recognize_license(), resolved_license(), archive() (+7 more)

### Community 61 - "remediation.py"
Cohesion: 0.19
Nodes (16): code_signature(), merge_lineage(), preference(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined() (+8 more)

### Community 62 - "v2_preflight.py"
Cohesion: 0.13
Nodes (12): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), build_training_groups(), validate_training_policy(), current_capacity(), _gpu_context() (+4 more)

### Community 64 - "matrix.py"
Cohesion: 0.11
Nodes (14): latest_checkpoint(), configure_full_rank_cross_attention(), run_matrix_benchmark(), snapshot(), run_matrix_stage(), run_matrix_training(), run_matrix_training_and_benchmark(), check_legacy_artifacts() (+6 more)

### Community 65 - "pathlib"
Cohesion: 0.06
Nodes (28): ensure_assets_on_disk(), load_asset_json(), cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), restore_sft_checkpoint(), save_sft_checkpoint() (+20 more)

### Community 66 - "apply_word_embedding_policy"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 67 - "examples"
Cohesion: 0.14
Nodes (4): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory(), test_snapshot_verifies_replacement_before_pruning()

### Community 69 - "get_v2_experiment"
Cohesion: 0.11
Nodes (12): get_v2_experiment(), load_sft_adapters_for_inference(), download_model(), _DummyTokenizer, test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(), test_alignment_stays_fp32_after_bf16_conversion(), test_encoder_bidirectional_window_boundary_matches_native() (+4 more)

### Community 70 - "pathlib"
Cohesion: 0.11
Nodes (13): _terms(), file_digest(), checked_prefix(), recover_remote_run(), _synchronize_once(), retained_checkpoints(), validate_checkpoint_files(), synchronize_preflight() (+5 more)

### Community 72 - "test_v2_models.py"
Cohesion: 0.14
Nodes (7): cached_logits(), InferenceCache, test_alignment_initialization_with_cuda_default_device(), test_cached_logits_match_full_prefix_and_cross_window(), test_empty_and_fully_masked_context_matches_self_only(), test_v3_wall_time_checkpoint_resume_is_exact(), make()

### Community 74 - "swesmith.py"
Cohesion: 0.24
Nodes (7): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 76 - "RepairCollator"
Cohesion: 0.06
Nodes (10): BackboneProfile, ModelOutput, Seq2SeqBatch, BifocalGemmaModel, build_pretrained_model(), TinyReferenceModel, __getattr__(), chunked_linear_cross_entropy() (+2 more)

### Community 77 - "BifocalGemmaConfig"
Cohesion: 0.15
Nodes (6): BifocalGemmaConfig, ProbeLoRALinear, test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_bifocalgemma_merge_and_unload(), test_bifocalgemma_save_and_from_pretrained_roundtrip()

### Community 78 - "CodeOverlapIndex"
Cohesion: 0.17
Nodes (20): code_shingles(), CodeOverlapIndex, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_cross_source_duplicates_and_nextcoder_variants() (+12 more)

### Community 79 - "build_native_tiny_model"
Cohesion: 0.22
Nodes (3): build_native_tiny_model(), TestNativeIntegration, factory()

### Community 80 - "v3/__init__.py"
Cohesion: 0.20
Nodes (6): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), is_cross_bridge_parameter()

### Community 81 - "run"
Cohesion: 0.25
Nodes (6): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_old_benchmarks_are_reset_once_while_completed_training_is_reused(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), run()

### Community 82 - "repository.py"
Cohesion: 0.19
Nodes (10): RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), git_environment(), example_from_record(), GitSnapshotStore (+2 more)

### Community 83 - "releases.py"
Cohesion: 0.09
Nodes (27): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+19 more)

### Community 91 - "shutil"
Cohesion: 0.18
Nodes (7): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), main(), render(), module_definitions(), render(), use_embedded_imports(), source_bundle()

### Community 92 - "RepositoryFile"
Cohesion: 0.15
Nodes (6): RepositoryFile, tokenize_repair_example(), test_dependency_completion_removes_target_and_retains_real_helper(), TestRealOfflineTokenizers, test_missing_old_file_is_still_rejected(), test_modify_add_delete_preserves_unrelated_unicode_bytes()

### Community 93 - "atomic_json"
Cohesion: 0.16
Nodes (11): run_context_benchmark(), activity(), score_context_case(), emit(), model_stage(), resource_snapshot(), DurableClock, prepare_sampled_benchmark_collection() (+3 more)

### Community 95 - "benchmark_upload.py"
Cohesion: 0.23
Nodes (6): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), test_retry_after_date_and_network_failures(), upload()

### Community 96 - "telemetry.py"
Cohesion: 0.14
Nodes (7): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_dashboard_stage_display_formatting()

### Community 97 - "FastTrainingConfig"
Cohesion: 0.08
Nodes (12): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+4 more)

### Community 100 - "greedy_decode"
Cohesion: 0.33
Nodes (3): greedy_decode(), ScriptedModel, test_greedy_encoder_once_stop_mode_and_vocab_guards()

### Community 101 - "shared_execution.py"
Cohesion: 0.29
Nodes (8): retry_delay(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), execution_settings(), verify_report()

### Community 102 - "test_tier3_combinations.py"
Cohesion: 0.19
Nodes (4): validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 105 - "v2.py"
Cohesion: 0.14
Nodes (13): source_fingerprints(), WithoutDocstrings, benchmark_directory_name(), benchmark_jobs(), _execute_v2(), save_guard(), snapshot(), RemoteHeartbeat (+5 more)

### Community 106 - "editing_record"
Cohesion: 0.14
Nodes (7): editing_record(), git_text_lines(), prepare_editing_records(), split_editing_records(), prepare_repository_training(), TestNotebookCellParity, TestOfflineEditing

### Community 107 - "test_context_benchmark_resume.py"
Cohesion: 0.18
Nodes (5): FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch(), transient()

### Community 108 - "test_benchmark_prompt_budget.py"
Cohesion: 0.14
Nodes (7): decoder_prompt(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_prompt_neutrality_matrix(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 114 - "ast"
Cohesion: 0.18
Nodes (8): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), test_exported_worker_has_parity_context_and_exact_adapter_checks(), main(), render_eval_script(), code_fragment_signatures()

### Community 115 - "TokenizedRepairExample"
Cohesion: 0.13
Nodes (12): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), fit_curated_sft(), boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens() (+4 more)

### Community 117 - "build_repository_corpus"
Cohesion: 0.26
Nodes (4): build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 118 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 119 - "packed_tokens.py"
Cohesion: 0.27
Nodes (5): build_packed_cache(), open_packed_cache(), compare_predictions(), test_packed_cache_can_be_built_without_tensor_runtime(), test_paired_bootstrap_and_cohort_rejection()

### Community 122 - "curation.py"
Cohesion: 0.22
Nodes (4): code_tokens(), CodeOverlapMatch, token_profile(), tokenize_repository_record()

### Community 123 - "pilot_e2b_e2b"
Cohesion: 0.22
Nodes (6): evaluate_humaneval_subset(), evaluate_pilot_checkpoint(), normalize_and_recount_patch(), pilot_e2b_e2b(), save_report(), tokenize_row()

### Community 124 - "native_config"
Cohesion: 0.20
Nodes (3): native_config(), NativeMergedAttention, version()

### Community 128 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 129 - "workflow"
Cohesion: 0.22
Nodes (6): workflow(), benchmark(), download(), execution(), fit(), prepare()

### Community 132 - "check_followers"
Cohesion: 0.29
Nodes (6): prepare_curated_training(), tokenize(), append_event(), check_followers(), progress(), progress()

### Community 133 - "batched_greedy_decode"
Cohesion: 0.29
Nodes (4): batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 134 - "test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract"
Cohesion: 0.25
Nodes (3): format_query_injected_context(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 138 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 924 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **46 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `workflow`, `run_sequential_pipeline`, `prepare_colab_pilot.py`, `check_followers`, `tempfile`, `test_v2_fixed_execution.py`, `json`, `re`, `calibrate_shared_profile`, `calibration`, `stable_fingerprint`, `run_repo_niah_eval`, `typing`, `journaled_loss`, `benchmark_upload.py`, `v2_preflight.py`, `matrix.py`, `pathlib`, `examples`, `get_v2_experiment`, `pathlib`, `v3/__init__.py`, `run`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `atomic_json`, `benchmark_upload.py`, `shared_execution.py`, `v2.py`, `test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit`, `packed_tokens.py`, `pilot_e2b_e2b`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07419712070874862 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `prepare_colab_pilot.py`, `__main__.py`, `v3/__init__.py`, `re`, `test_memory_efficient_training.py`, `test_editing_manifest.py`, `torch`, `prepare_colab_probe.py`, `fast_pipeline.py`, `configure_memory_efficient_training`, `default_training_profiles`, `typing`, `memory_probe.py`, `PackedTokenizedDataset`, `journaled_loss`, `v2_preflight.py`, `pathlib`, `RepairCollator`, `CodeOverlapIndex`, `RepositoryFile`, `.run_benchmark_suite`, `packed_tokens.py`, `curation.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Should `molab_pilot_support.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1455026455026455 - nodes in this community are weakly interconnected._
- **Why does `FastTrainingPipeline` connect `FastTrainingConfig` to `test_molab_matrix_notebooks.py`, `__main__.py`, `fast_pipeline.py`, `BifocalGemmaForCausalLM`, `.run_benchmark_suite`, `typing`, `.run`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._