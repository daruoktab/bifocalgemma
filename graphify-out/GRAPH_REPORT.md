# Graph Report - bifocalgemma  (2026-11-12)

## Corpus Check
- 304 files · ~389,082 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 25 file(s) not represented in the graph (top: .tex 11, .ipynb 3, (none) 2)

## Summary
- 2580 nodes · 7516 edges · 130 communities (96 shown, 34 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 816 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8aa843f9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_notebook_dependencies.py
- test_molab_matrix_notebooks.py
- repository_corpus.py
- math
- prepare_colab_pilot.py
- curated_sft.py
- FeatureNotReadyError
- native.py
- seal_report
- remediation.py
- FullCrossCachedAttention
- test_gsm8k.py
- subprocess
- _execute_v2
- test_molab_pilot.py
- schedule.py
- config.py
- calibrate_shared_profile
- training/offline_sft.py
- NativeTinyModel
- json
- hashlib
- test_tier5_adversarial.py
- test_repository_study.py
- CodeOverlapIndex
- test_memory_efficient_training.py
- TokenizedRepairExample
- hashlib
- torch
- test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt
- TestEvaluationReporting
- v2/bridge.py
- matrix.py
- prepare_colab_probe.py
- re
- pytest
- test_tier1_features.py
- fit_offline_sft
- NativeTinyModel
- JSONLRepairDataset
- run_batched_context_benchmark
- BenchmarkUploadQueue
- adversarial_r3_probe.py
- test_molab_remediation.py
- test_training_recipes.py
- BifocalGemmaForCausalLM
- random
- test_matrix_schedule_runtime.py
- decoder_prompt
- repository_study.py
- contracts.py
- build_repo_niah_case
- typing
- test_one_model_training_stops_and_completed_resume_does_not_reload
- adapters.py
- reference_resilient_mcq_parser
- models/checkpoint_transfer.py
- v2_preflight.py
- test_repository_corpus.py
- test_data_remediation.py
- build_repository_corpus
- TestBifocalGemmaArchitecture
- matrix.py
- configure_probe_training
- load_verified_tokenizer
- curate_paired_sources
- shared_execution.py
- BifocalGemmaModel
- test_training_telemetry_v2.py
- molab_pilot_support.py
- re
- build_joint_visibility
- ast
- TrainingDashboard
- build_native_tiny_model
- native_python.py
- workflow
- generation.py
- releases.py
- test_preflight_upload_verifies_exact_evidence_without_checkpoint
- packed_tokens.py
- optimizers.py
- BifocalGemmaForCausalLM
- bifocalgemma
- test_colab_pilot.py
- v2.py
- SwiGLUDualGateBridge
- molab_matrix_support.py
- telemetry.py
- test_tier4_scenarios.py
- test_notebook_hf_flow.py
- run
- atomic_json
- resolve_optimal_eval_batch_size
- inference_cache.py
- pathlib
- protocol.py
- .snapshot
- build_policy_optimizer
- DummyTrackingTokenizer
- test_fast_pipeline.py
- __main__.py
- PackedTokenizedDataset
- bifocalgemma/__init__.py
- test_colab_probe.py
- code_signature
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- TestOfflineSFT
- TestCheckpointTransfer
- test_followers_wait_for_one_profile_and_pin_it_locally
- test_profile_publish_parent_guard_keeps_concurrent_winner
- patch_matrix_tokenization

## God Nodes (most connected - your core abstractions)
1. `atomic_json()` - 84 edges
2. `stable_fingerprint()` - 69 edges
3. `load_verified_tokenizer()` - 58 edges
4. `TokenizedRepairExample` - 53 edges
5. `fit_offline_sft()` - 47 edges
6. `run_matrix_training()` - 46 edges
7. `_execute_v2()` - 45 edges
8. `configure_memory_efficient_training()` - 44 edges
9. `build_native_tiny_model()` - 43 edges
10. `RepairCollator` - 42 edges

## Surprising Connections (you probably didn't know these)
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/models/v2/native.py
- `test_fast_training_config_step_reconciliation()` --calls--> `FastTrainingConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/training/fast_pipeline.py
- `test_widgets_update_in_place_and_restore_history_without_duplicate_points()` --uses--> `WigglyDashboard`  [INFERRED]
  tests/test_training_telemetry_v2.py → src/bifocalgemma/training/wiggly_dashboard.py
- `event()` --calls--> `atomic_json()`  [INFERRED]
  tools/prepare_colab_audit.py → src/bifocalgemma/training/stages.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (130 total, 34 thin omitted)

### Community 0 - "test_notebook_dependencies.py"
Cohesion: 0.09
Nodes (16): simulated_environment(), test_apply_waits_for_submission_and_valid_token_without_native_imports(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure() (+8 more)

### Community 1 - "test_molab_matrix_notebooks.py"
Cohesion: 0.09
Nodes (26): configure_memory_efficient_training(), configure_policy_model(), unified_recipe(), make_experiment_spec(), configure_matrix_stage(), test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(), test_full_rank_cross_attention_stage_training_and_cached_equivalence(), test_cached_uncached_and_batched_greedy_decode_equivalence() (+18 more)

### Community 2 - "repository_corpus.py"
Cohesion: 0.07
Nodes (37): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), export_pool(), read_arrow_rows() (+29 more)

### Community 3 - "math"
Cohesion: 0.21
Nodes (5): validate_optimizer_config(), Float32Alignment, update_policy_learning_rates(), validate_training_policy(), resolve_training_cadence()

### Community 4 - "prepare_colab_pilot.py"
Cohesion: 0.10
Nodes (25): target_fingerprint(), build_full_cross_cohorts(), build_source_copy_probes(), build_span_denoising_examples(), configure_full_rank_cross_attention(), __init__(), configure_pilot_stage(), encode_chat_pair() (+17 more)

### Community 5 - "curated_sft.py"
Cohesion: 0.13
Nodes (14): prepare_editing_records(), split_editing_records(), fit_curated_sft(), prepare_curated_training(), tokenize(), default_training_profiles(), fit_repository_sft(), prepare_repository_training() (+6 more)

### Community 6 - "FeatureNotReadyError"
Cohesion: 0.12
Nodes (7): extract_token_ast_features(), TokenASTFeatures, EncoderMemory, RepositoryCache, BuildBlock, FeatureNotReadyError, distillation_loss()

### Community 7 - "native.py"
Cohesion: 0.12
Nodes (6): run_length_grid(), chunked_linear_cross_entropy(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library(), test_notebook_sweep_persists_results_per_case()

### Community 8 - "seal_report"
Cohesion: 0.33
Nodes (7): seal_report(), completed_evidence(), test_complete_six_model_boundary_evidence_can_be_reviewed_offline(), test_partial_or_inconsistent_gpu_evidence_cannot_be_released(), test_profile_rejects_corruption_mismatched_guard_and_effective_batch(), valid_profile(), check_evidence()

### Community 9 - "remediation.py"
Cohesion: 0.14
Nodes (11): token_profile(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference() (+3 more)

### Community 10 - "FullCrossCachedAttention"
Cohesion: 0.13
Nodes (4): Bf16FullLinear, FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype()

### Community 11 - "test_gsm8k.py"
Cohesion: 0.17
Nodes (8): FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train(), test_verified_preparation_and_input_only_loading()

### Community 12 - "subprocess"
Cohesion: 0.08
Nodes (11): test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), audit(), emit(), main(), validate_all_patches(), main(), main(), main() (+3 more)

### Community 13 - "_execute_v2"
Cohesion: 0.10
Nodes (18): get_v2_experiment(), cached_logits(), benchmark_jobs(), encoder_diagnostics(), download_model(), _execute_v2(), progress(), save_guard() (+10 more)

### Community 14 - "test_molab_pilot.py"
Cohesion: 0.13
Nodes (14): test_failed_final_upload_remains_pending(), test_standalone_bundle_has_fixed_control_and_no_colab_side_effects(), test_upload_retry_verification_and_base_prefix_only(), test_runtime_archive_is_identical_with_windows_or_linux_newlines(), test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle(), format_code(), main(), matrix_bundle() (+6 more)

### Community 15 - "schedule.py"
Cohesion: 0.11
Nodes (23): build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule(), ScheduleRecord (+15 more)

### Community 16 - "config.py"
Cohesion: 0.16
Nodes (9): BackboneProfile, ExperimentSpec, _pair(), _ref(), describe_initialization(), InitializationPlan, initialize_pretrained_components(), InitializedComponents (+1 more)

### Community 17 - "calibrate_shared_profile"
Cohesion: 0.17
Nodes (10): calibrate_shared_profile(), emit(), save(), trial(), candidate_settings(), error_kind(), numerical_probe(), representative_cohort() (+2 more)

### Community 18 - "training/offline_sft.py"
Cohesion: 0.06
Nodes (20): stratified_subset(), examples(), TinyLossModel, test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(), calibration(), guard(), test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(), test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight() (+12 more)

### Community 19 - "NativeTinyModel"
Cohesion: 0.08
Nodes (6): NormalizedAlignment, build_native_model(), native_config(), NativeMergedAttention, RoutedRepoBatch, test_alignment_initialization_with_cuda_default_device()

### Community 20 - "json"
Cohesion: 0.07
Nodes (18): ensure_assets_on_disk(), curated_pool_digest(), load_curated_pool(), compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit_pool(), audit_views(), main() (+10 more)

### Community 21 - "hashlib"
Cohesion: 0.14
Nodes (16): digest(), extract_release_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k(), generate_run(), gsm8k_prompt_ids(), load_gsm8k() (+8 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (24): completion_metrics(), extract_mcq_letter(), strict_letter(), probe_unicode_and_metrics(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards() (+16 more)

### Community 23 - "test_repository_study.py"
Cohesion: 0.10
Nodes (13): authoritative_empirical_results(), project_root(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), CharacterTokenizer, diff(), ExecutingScoringOptions (+5 more)

### Community 24 - "CodeOverlapIndex"
Cohesion: 0.25
Nodes (13): code_shingles(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), test_disk_index_matches_memory_and_syntax_repairs_are_admitted(), test_full_overlap_index_skips_impossible_blob_reads(), test_license_repo_and_benchmark_gates(), test_near_matching_and_short_fragment_policy() (+5 more)

### Community 25 - "test_memory_efficient_training.py"
Cohesion: 0.13
Nodes (15): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+7 more)

### Community 26 - "TokenizedRepairExample"
Cohesion: 0.08
Nodes (9): Seq2SeqBatch, RepairCollator, __getattr__(), apply_word_embedding_policy(), EmbeddingPolicyResult, run_reference_smoke(), TinyReferenceModel, TestWordEmbeddingPolicy (+1 more)

### Community 27 - "hashlib"
Cohesion: 0.26
Nodes (10): editing_manifest(), repository_editing_record(), notebook_function(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical() (+2 more)

### Community 28 - "torch"
Cohesion: 0.12
Nodes (16): flex_kernel_options(), fused_attention(), attention_tile(), joint_visibility_tile(), mask_tile(), tiled_attention(), _TiledAttention, TiledEncoderAttention (+8 more)

### Community 31 - "TestEvaluationReporting"
Cohesion: 0.13
Nodes (6): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), TestEvaluationReporting

### Community 32 - "v2/bridge.py"
Cohesion: 0.11
Nodes (3): GatedCrossAttentionBridge, RMSNorm, TestGemma4EdgeTopology

### Community 34 - "matrix.py"
Cohesion: 0.09
Nodes (12): backward_window(), chunked_differential_cross_entropy(), compute_v3_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), _pointer_chunk_forward(), resolve_diff_token_ids(), _streaming_lse_pointer_nll() (+4 more)

### Community 35 - "prepare_colab_probe.py"
Cohesion: 0.25
Nodes (6): source_fingerprint(), main(), check_followers(), declared_limit_example(), main(), render()

### Community 36 - "re"
Cohesion: 0.18
Nodes (13): ContextWindow, render_window(), select_repository_context(), _terms(), recognize_license(), resolved_license(), prepare_repository_record(), prepare_repository_record_with_snapshot() (+5 more)

### Community 37 - "pytest"
Cohesion: 0.11
Nodes (18): normalize_context_task(), repair_preimage_coverage(), task_prompt(), tokenize_context_task(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task() (+10 more)

### Community 38 - "test_tier1_features.py"
Cohesion: 0.07
Nodes (16): validate_empirical_results_schema(), validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f2_mcq_token_budget_specification(), test_f3_crosscodelong_continuous_metrics() (+8 more)

### Community 39 - "fit_offline_sft"
Cohesion: 0.12
Nodes (16): cpu_tree(), evaluate_sft_loss(), file_digest(), fit_offline_sft(), emit(), model_stage(), restore_sft_checkpoint(), save_sft_checkpoint() (+8 more)

### Community 40 - "NativeTinyModel"
Cohesion: 0.18
Nodes (4): ModelOutput, NativeTinyModel, __getattr__(), patch_cross_entropy()

### Community 41 - "JSONLRepairDataset"
Cohesion: 0.11
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 42 - "run_batched_context_benchmark"
Cohesion: 0.20
Nodes (8): run_context_benchmark(), activity(), score_context_case(), DurableClock, prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), test_prepare_sampled_benchmark_collection_deterministic_100_cases()

### Community 43 - "BenchmarkUploadQueue"
Cohesion: 0.07
Nodes (17): BenchmarkUploadQueue, limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed(), test_parent_conflict_refreshes_but_keeps_retry_bounded() (+9 more)

### Community 44 - "adversarial_r3_probe.py"
Cohesion: 0.13
Nodes (7): score_repo_niah_case(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), verify_gate_round_2_resolved_cases()

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.07
Nodes (9): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+1 more)

### Community 46 - "test_training_recipes.py"
Cohesion: 0.15
Nodes (13): comparison_recipes(), run_recipe_comparison(), limit_threads(), test_all_training_precedes_benchmark_and_failure_stops_barrier(), test_completed_benchmark_with_pending_upload_resumes_without_model_load(), benchmark(), factory(), upload() (+5 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.08
Nodes (17): BifocalGemmaConfig, _resolve_backbone_weights(), ProbeLoRALinear, test_model_to_bfloat16_preserves_fp32_master_weights(), test_pretrained_model_subfolder_support_and_repo_creation(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation() (+9 more)

### Community 48 - "random"
Cohesion: 0.14
Nodes (8): TokenizedRepairExample, run_training_memory_preflight(), build_anti_forgetting_curriculum(), encode_comprehension_prefix(), is_comprehension_example(), synthesize_comprehension_examples(), TestDataAndLoss, test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()

### Community 49 - "test_matrix_schedule_runtime.py"
Cohesion: 0.21
Nodes (8): fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), test_partial_backward_oom_replays_entire_window_without_double_gradient(), test_trained_adapters_export_reload_logit_parity()

### Community 50 - "decoder_prompt"
Cohesion: 0.10
Nodes (9): decoder_prompt(), greedy_decode(), run_repo_niah_eval(), test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer, ScriptedModel (+1 more)

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+10 more)

### Community 52 - "contracts.py"
Cohesion: 0.11
Nodes (15): DeveloperAgent, RepairExample, RepairPrediction, RepositorySnapshot, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash() (+7 more)

### Community 53 - "build_repo_niah_case"
Cohesion: 0.18
Nodes (5): build_repo_niah_case(), format_crosscodelong_summary(), generate_repo_niah_haystack(), make_distractor_python_chunk(), test_adversarial_reponiah_deterministic_reproducibility()

### Community 54 - "typing"
Cohesion: 0.09
Nodes (9): _compute_per_row_cross_query(), save_matrix_checkpoint(), build_bifocalgemma_v3_optimizer(), compute_diff_token_weights(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), clamped_triton_kernel_options(), critical_attention_beta() (+1 more)

### Community 55 - "test_one_model_training_stops_and_completed_resume_does_not_reload"
Cohesion: 0.29
Nodes (3): test_one_model_training_stops_and_completed_resume_does_not_reload(), execution(), factory()

### Community 56 - "adapters.py"
Cohesion: 0.25
Nodes (5): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter

### Community 57 - "reference_resilient_mcq_parser"
Cohesion: 0.14
Nodes (6): test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 59 - "v2_preflight.py"
Cohesion: 0.18
Nodes (7): current_capacity(), _gpu_context(), maximum_probe(), _measure_phase(), select_execution(), emit(), save()

### Community 60 - "test_repository_corpus.py"
Cohesion: 0.21
Nodes (11): context_license_evidence(), patch_license_evidence(), path_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read() (+3 more)

### Community 61 - "test_data_remediation.py"
Cohesion: 0.32
Nodes (11): remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage(), test_heldout_role_rejected() (+3 more)

### Community 62 - "build_repository_corpus"
Cohesion: 0.16
Nodes (6): ArchiveSnapshotStore, build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_failed_download_removes_partial_bytes(), test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 63 - "TestBifocalGemmaArchitecture"
Cohesion: 0.06
Nodes (3): ASTGraphProjector, RepoContextRouter, TestBifocalGemmaArchitecture

### Community 64 - "matrix.py"
Cohesion: 0.08
Nodes (21): export_matrix_pretrained_model(), benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), configure_full_rank_cross_attention(), load_matrix_adapters_for_inference(), prune_matrix_checkpoints() (+13 more)

### Community 65 - "configure_probe_training"
Cohesion: 0.18
Nodes (8): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), synthetic_stress_target(), tensor_bytes(), test_exact_synthetic_lengths_preserve_shift_and_closing_turn(), TestMemoryProbe

### Community 66 - "load_verified_tokenizer"
Cohesion: 0.16
Nodes (5): load_asset_json(), CheckpointRef, load_verified_tokenizer(), tokenize_repair_example(), replace()

### Community 67 - "curate_paired_sources"
Cohesion: 0.14
Nodes (13): validate_curated_pool_limits(), raw(), test_complete_epoch_training_budget_all_families(), test_cross_source_duplicates_and_nextcoder_variants(), test_curated_resume_rejects_runtime_drift(), test_frozen_budget_matches_all_six_controls(), test_frozen_pool_verifies_checksums_before_use(), test_invalid_requested_pool_budget_is_rejected() (+5 more)

### Community 68 - "shared_execution.py"
Cohesion: 0.29
Nodes (8): retry_delay(), publish_profile(), publish_with_retry(), read_remote_profile(), resolve_shared_profile(), validate_profile(), execution_settings(), verify_report()

### Community 69 - "BifocalGemmaModel"
Cohesion: 0.13
Nodes (4): restore_checkpoint(), save_checkpoint(), BifocalGemmaModel, build_pretrained_model()

### Community 70 - "test_training_telemetry_v2.py"
Cohesion: 0.15
Nodes (5): test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(), test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_training_resume_keeps_elapsed_from_step_five_through_ten(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), load_runtime()

### Community 72 - "molab_pilot_support.py"
Cohesion: 0.08
Nodes (18): fail_second_microbatch(), test_checkpoint_restores_optimizer_rng_and_skips_partial(), test_completed_run_retries_upload_without_loading_model(), test_journal_integrity_and_phase_isolation(), test_loss_journal_preserves_token_weighting_and_reuses_examples(), original(), TestNotebookCellParity, measure() (+10 more)

### Community 74 - "re"
Cohesion: 0.17
Nodes (10): RepositoryFile, apply_patch_to_snapshot(), extract_unified_diff(), patch_paths(), safe_repo_path(), patch_syntax(), test_missing_old_file_is_still_rejected(), test_modify_add_delete_preserves_unrelated_unicode_bytes() (+2 more)

### Community 76 - "build_joint_visibility"
Cohesion: 0.24
Nodes (3): build_joint_visibility(), ReferenceMergedAttention, TestMergedAttentionReference

### Community 77 - "ast"
Cohesion: 0.11
Nodes (17): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), refresh_patch_targets(), source_fingerprints() (+9 more)

### Community 78 - "TrainingDashboard"
Cohesion: 0.22
Nodes (5): loss_chart(), render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 79 - "build_native_tiny_model"
Cohesion: 0.22
Nodes (3): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head()

### Community 80 - "native_python.py"
Cohesion: 0.33
Nodes (7): canonical_snapshot_hash(), example_from_record(), NativeTaskResult, score_python_patch(), summarize_native_results(), test_native_score_pass_fail_disabled_and_runtime(), test_unittest_results_and_early_exit_do_not_false_pass()

### Community 81 - "workflow"
Cohesion: 0.22
Nodes (6): workflow(), benchmark(), download(), execution(), fit(), prepare()

### Community 82 - "generation.py"
Cohesion: 0.19
Nodes (10): decode_text(), format_encoder_context(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids(), _safe_apply_chat_template() (+2 more)

### Community 83 - "releases.py"
Cohesion: 0.08
Nodes (28): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+20 more)

### Community 85 - "packed_tokens.py"
Cohesion: 0.31
Nodes (3): build_packed_cache(), open_packed_cache(), test_packed_cache_can_be_built_without_tensor_runtime()

### Community 87 - "optimizers.py"
Cohesion: 0.32
Nodes (3): GrokOrScaleLoRA, orthogonal_direction(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 91 - "test_colab_pilot.py"
Cohesion: 0.13
Nodes (7): test_exported_worker_has_parity_context_and_exact_adapter_checks(), _DummyTokenizer, pilot_module(), test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(), test_stratified_raw_subset_matches_tokenized_stratified_subset_order(), _prepare(), prepare_runtime()

### Community 92 - "v2.py"
Cohesion: 0.08
Nodes (17): file_digest(), checked_prefix(), recover_remote_run(), retry_delay(), _synchronize_once(), retained_checkpoints(), validate_checkpoint_files(), benchmark_directory_name() (+9 more)

### Community 96 - "telemetry.py"
Cohesion: 0.11
Nodes (7): format_duration(), format_stage_display(), LastSeenWidget, phase_category(), TrainingLossLog, WigglyDashboard, test_dashboard_stage_display_formatting()

### Community 101 - "test_tier4_scenarios.py"
Cohesion: 0.32
Nodes (3): test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 102 - "test_notebook_hf_flow.py"
Cohesion: 0.36
Nodes (5): cell_function(), test_completed_upload_summary_has_no_second_side_effect(), test_loader_always_verifies_pinned_collection_even_for_warm_cache(), test_pending_backup_is_not_displayed_as_completed(), test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion()

### Community 103 - "run"
Cohesion: 0.29
Nodes (5): test_all_six_models_follow_one_profile_instead_of_tuning_independently(), test_notebook_starts_with_token_without_any_mode_selection(), test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(), test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(), run()

### Community 104 - "atomic_json"
Cohesion: 0.08
Nodes (21): remote_reset_plan(), reset_local_benchmarks(), reset_state(), synchronize_run(), run_sequential_pipeline(), publish_final(), save_benchmark_progress(), upload_checkpoint() (+13 more)

### Community 105 - "resolve_optimal_eval_batch_size"
Cohesion: 0.40
Nodes (3): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 107 - "pathlib"
Cohesion: 0.08
Nodes (8): retry_windows_file_operation(), resource_snapshot(), pytest_configure(), reference_resilient_mcq_parser(), test_combination_mcq_scoring_across_benchmarks(), test_external_activity_reader_does_not_need_running_kernel(), inspect(), main()

### Community 108 - "protocol.py"
Cohesion: 0.09
Nodes (17): _extract_match_letter(), fallback_letter_regex(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), mcq_report_metrics(), _positive_integer(), _runtime_model_identity() (+9 more)

### Community 110 - "build_policy_optimizer"
Cohesion: 0.40
Nodes (3): build_policy_optimizer(), benchmark_candidate(), probe_optimizer()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.10
Nodes (15): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+7 more)

### Community 114 - "__main__.py"
Cohesion: 0.12
Nodes (8): validate_experiment_matrix(), EvaluationReport, SWEBenchHarness, main(), run_native_smoke(), SFTConfig, train_direct_sft(), TestConstructionEntrypoints

### Community 117 - "test_colab_probe.py"
Cohesion: 0.70
Nodes (3): boundary_builder(), test_declared_boundaries_cannot_hide_an_overflow_by_truncation(), test_declared_boundaries_reach_exact_input_length_without_new_target_tokens()

### Community 118 - "code_signature"
Cohesion: 0.33
Nodes (4): code_signature(), code_tokens(), nextcoder_row(), test_unparseable_fallback_preserves_whitespace()

### Community 127 - "test_followers_wait_for_one_profile_and_pin_it_locally"
Cohesion: 0.50
Nodes (3): test_followers_wait_for_one_profile_and_pin_it_locally(), fetch(), test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline()

### Community 134 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 964 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `repository_corpus.py`, `prepare_colab_pilot.py`, `_execute_v2`, `calibrate_shared_profile`, `training/offline_sft.py`, `json`, `prepare_colab_probe.py`, `run_batched_context_benchmark`, `BenchmarkUploadQueue`, `test_training_recipes.py`, `decoder_prompt`, `typing`, `v2_preflight.py`, `matrix.py`, `shared_execution.py`, `workflow`, `test_preflight_upload_verifies_exact_evidence_without_checkpoint`, `packed_tokens.py`, `v2.py`, `pathlib`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `atomic_json()` (e.g. with `test_completed_benchmark_with_pending_upload_resumes_without_model_load()` and `benchmark()`) actually correct?**
  _`atomic_json()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_notebook_dependencies.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Why does `build_native_tiny_model()` connect `build_native_tiny_model` to `test_molab_matrix_notebooks.py`, `curated_sft.py`, `native.py`, `_execute_v2`, `config.py`, `NativeTinyModel`, `hashlib`, `test_memory_efficient_training.py`, `test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt`, `fit_offline_sft`, `NativeTinyModel`, `test_training_recipes.py`, `BifocalGemmaForCausalLM`, `test_one_model_training_stops_and_completed_resume_does_not_reload`, `configure_probe_training`, `curate_paired_sources`, `workflow`, `__main__.py`, `TestOfflineSFT`, `TestCheckpointTransfer`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Should `test_molab_matrix_notebooks.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0851063829787234 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `random` to `test_molab_matrix_notebooks.py`, `prepare_colab_pilot.py`, `curated_sft.py`, `native.py`, `training/offline_sft.py`, `test_memory_efficient_training.py`, `TokenizedRepairExample`, `hashlib`, `torch`, `prepare_colab_probe.py`, `pytest`, `JSONLRepairDataset`, `test_matrix_schedule_runtime.py`, `contracts.py`, `v2_preflight.py`, `configure_probe_training`, `load_verified_tokenizer`, `curate_paired_sources`, `molab_pilot_support.py`, `packed_tokens.py`, `test_colab_pilot.py`, `pathlib`, `__main__.py`, `PackedTokenizedDataset`, `test_colab_probe.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._