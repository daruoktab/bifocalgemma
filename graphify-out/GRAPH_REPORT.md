# Graph Report - bifocalgemma  (2026-11-27)

## Corpus Check
- 260 files · ~376,883 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 17 file(s) not represented in the graph (top: .tex 5, (none) 2, .jsonl 2)

## Summary
- 2687 nodes · 7346 edges · 191 communities (100 shown, 91 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 814 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a28e1883`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- checkpoint.py
- test_compute_per_row_cross_query_vectorized_parity
- benchmark_upload.py
- pathlib
- RepositorySnapshot
- tests_e2e/conftest.py
- sys
- test_memory_efficient_training.py
- FeatureNotReadyError
- repository.py
- test_pretrained_hf_model.py
- matrix.py
- make_experiment_spec
- stable_fingerprint
- TokenizedRepairExample
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- test_tier2_boundary.py
- NativeTinyModel
- context_tasks.py
- gsm8k.py
- scorer_fingerprint
- BifocalGemmaConfig
- FastTrainingPipeline
- test_v4_data_disjoint.py
- curriculum.py
- curation.py
- test_training_telemetry.py
- run_external_benchmark
- fetch_collection
- gsm8k.py
- prepare_context_data.py
- benchmark_reset.py
- types
- TrainingDashboard
- TestBifocalGemmaArchitecture
- protocol.py
- test_gsm8k.py
- evaluation/context.py
- BifocalGemmaForCausalLM
- test_data_remediation.py
- math
- json
- configure_full_rank_cross_attention
- repository_study.py
- test_repository_study.py
- .configure_stage1
- test_v4_milestone_checkpoints_survive_pruning_and_hub_retention
- test_matrix_schedule_runtime.py
- hub.py
- objectives.py
- repository_corpus.py
- test_shared_schedule.py
- audit_context_release.py
- completion_metrics
- run_matrix_training
- memory_efficient.py
- test_context_tasks.py
- test_tier4_scenarios.py
- test_effective_benchmark_protocol.py
- subprocess
- test_construction_blocks.py
- fit_offline_sft
- WigglyDashboard
- argparse
- Any
- RepositorySnapshot
- objectives.py
- ExperimentSpec
- test_construction_blocks.py
- bifocalgemma
- build_and_publish_v4_dataset.py
- test_overflow_fallback_recomputes_budget_and_records_metadata
- wiggly_dashboard.py
- pytest
- test_molab_remediation.py
- test_tier1_features.py
- test_cross_attention.py
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- test_fast_pipeline_respects_learnable_cross_scale_default_and_override
- InferenceCache
- tokenization.py
- test_tokenize_context_task_preserves_target_indentation_and_newlines
- patch_matrix_tokenization
- test_fast_pipeline.py
- export_matrix_pretrained_model
- patch_cross_entropy
- collections
- _CharTokenizer
- test_tier3_combinations.py
- atomic_json
- hashlib
- run_matrix_benchmark
- test_benchmark_prompt_budget.py
- .snapshot
- CachedAttention
- comparison.py
- JSONLRepairDataset
- subprocess
- .config
- .from_pretrained_backbones
- PackedTokenizedDataset
- sdpa_cross_attention
- training/diagnostics.py
- sdpa_cross_attention
- source_fingerprints
- _ProportionalTokenizer
- Seq2SeqBatch
- tiled_attention
- audit_dataset_release.py
- test_molab_matrix_notebooks.py

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 65 edges
2. `atomic_json()` - 61 edges
3. `BifocalGemmaForCausalLM` - 59 edges
4. `run_matrix_training()` - 52 edges
5. `load_verified_tokenizer()` - 51 edges
6. `RepositorySnapshot` - 45 edges
7. `run_matrix_benchmark()` - 41 edges
8. `TokenizedRepairExample` - 41 edges
9. `FastTrainingPipeline` - 40 edges
10. `BifocalGemmaConfig` - 39 edges

## Surprising Connections (you probably didn't know these)
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length()` --calls--> `TokenizedRepairExample`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/contracts.py
- `test_comprehension_targets_stay_inside_baseline_visible_prefix()` --calls--> `TokenizedRepairExample`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/contracts.py
- `test_bifocalgemma_config_initialization_and_serialization()` --calls--> `BifocalGemmaConfig`  [INFERRED]
  tests/test_pretrained_hf_model.py → src/bifocalgemma/configuration_bifocalgemma.py
- `test_pipeline_state_ledger_creation_for_remote_resumption()` --calls--> `StageLedger`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/training/stages.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (191 total, 91 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.12
Nodes (15): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe() (+7 more)

### Community 1 - "checkpoint.py"
Cohesion: 0.09
Nodes (17): is_transient_lfs_pointer_error(), load_matrix_adapters_for_inference(), _milestone_checkpoint_names(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), synchronize_matrix() (+9 more)

### Community 2 - "test_compute_per_row_cross_query_vectorized_parity"
Cohesion: 0.24
Nodes (5): critical_attention_beta(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_sdpa_attention_structured_masks_match_tiled_eager(), test_sdpa_cross_attention_matches_eager_and_handles_all_masked_rows()

### Community 3 - "benchmark_upload.py"
Cohesion: 0.08
Nodes (19): BenchmarkUploadQueue, retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush() (+11 more)

### Community 5 - "pathlib"
Cohesion: 0.05
Nodes (10): ensure_assets_on_disk(), load_asset_json(), BackboneProfile, _check_transformers_version(), native_config(), FullCrossCachedAttention, ProbeLoRALinear, clamped_triton_kernel_options() (+2 more)

### Community 6 - "RepositorySnapshot"
Cohesion: 0.11
Nodes (26): RepositoryFile, RepositorySnapshot, _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application() (+18 more)

### Community 7 - "tests_e2e/conftest.py"
Cohesion: 0.15
Nodes (6): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 8 - "sys"
Cohesion: 0.06
Nodes (36): dependency_state(), test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready() (+28 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.28
Nodes (12): configure_memory_efficient_training(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 11 - "FeatureNotReadyError"
Cohesion: 0.08
Nodes (13): DeveloperAgent, EvaluationReport, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter (+5 more)

### Community 13 - "repository.py"
Cohesion: 0.12
Nodes (22): file_digest(), read_arrow_rows(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+14 more)

### Community 14 - "test_pretrained_hf_model.py"
Cohesion: 0.16
Nodes (9): FastTrainingConfig, test_fast_training_pipeline_features_and_export(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_training_config_step_reconciliation(), test_fast_training_pipeline_baseline_2b() (+1 more)

### Community 15 - "matrix.py"
Cohesion: 0.09
Nodes (19): _build_schedule_for_epochs(), _example_sequence_lengths(), pack_window_microbatches(), run_matrix_stage(), _split_microbatch_groups(), _validate_num_epochs(), _apportion_strata_counts(), build_shared_schedule() (+11 more)

### Community 16 - "make_experiment_spec"
Cohesion: 0.17
Nodes (11): CheckpointRef, make_experiment_spec(), _pair(), _ref(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted() (+3 more)

### Community 17 - "stable_fingerprint"
Cohesion: 0.23
Nodes (7): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), test_f2_mcq_token_budget_specification()

### Community 18 - "TokenizedRepairExample"
Cohesion: 0.12
Nodes (10): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), fit_curated_sft(), synthetic_stress_target(), TestDataAndLoss, test_curated_resume_rejects_runtime_drift() (+2 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.07
Nodes (9): infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), MockCausalLM, MockQwenTokenizer, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(), test_three_way_control_modes_longcodeqa_dedup_and_fim_marker() (+1 more)

### Community 20 - "eval_code.py"
Cohesion: 0.09
Nodes (9): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity(), test_prompt_ids_instance_id_isolation_and_gold_rejection() (+1 more)

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (19): test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization(), test_adversarial_mcq_redos_scaling(), test_adversarial_mcq_ruling_out_distractors_vulnerability(), test_adversarial_mcq_split_ambiguity_behavior() (+11 more)

### Community 23 - "test_tier2_boundary.py"
Cohesion: 0.14
Nodes (6): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement()

### Community 24 - "NativeTinyModel"
Cohesion: 0.21
Nodes (5): ModelOutput, NativeTinyModel, test_cross_scale_frozen_by_default_and_configurable(), test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), spy_sdpa()

### Community 25 - "context_tasks.py"
Cohesion: 0.15
Nodes (19): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), _parse_file_sections(), _render_entries(), _resolve_eot_token_id() (+11 more)

### Community 26 - "gsm8k.py"
Cohesion: 0.06
Nodes (17): build_native_model(), build_native_tiny_model(), NativeMergedAttention, load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), configure_probe_training(), gpu_memory() (+9 more)

### Community 27 - "scorer_fingerprint"
Cohesion: 0.09
Nodes (15): _extract_match_letter(), extract_mcq_letter(), fallback_letter_regex(), score_context_case(), score_repo_niah_case(), strict_letter(), scorer_fingerprint(), probe_mcq_delimiters_and_wrappers() (+7 more)

### Community 28 - "BifocalGemmaConfig"
Cohesion: 0.10
Nodes (11): BifocalGemmaConfig, test_bridge_init_preserved_across_post_init(), test_cached_logits_decoder_only_not_polluted_by_cross_attention(), test_from_pretrained_device_map_and_missing_bridge_validation(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_generate_multi_eos_per_row_stopping_and_cache_parity(), test_generate_right_padded_matches_left_padded(), test_pretrained_model_subfolder_support_and_repo_creation() (+3 more)

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.31
Nodes (6): _load_prepare_module(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_mixed_pool_v4_curriculum_grounded_and_curated_sft(), test_v4_quarantine_licenses_dedup_and_nextcoder_rejection(), _write_mock_curated_pool()

### Community 31 - "curriculum.py"
Cohesion: 0.22
Nodes (7): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples()

### Community 32 - "curation.py"
Cohesion: 0.08
Nodes (31): code_shingles(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits(), count_syntax_warnings() (+23 more)

### Community 33 - "test_training_telemetry.py"
Cohesion: 0.13
Nodes (5): _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime(), RecordingStream

### Community 34 - "run_external_benchmark"
Cohesion: 0.11
Nodes (17): format_crosscodelong_summary(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort(), load_hf_external_model_and_tokenizer() (+9 more)

### Community 35 - "fetch_collection"
Cohesion: 0.15
Nodes (14): digest(), fetch_collection(), unpack_archive(), verify_release(), pool(), release(), test_eval_public_and_gold_separate(), test_existing_local_pool_cannot_bypass_pinned_identity() (+6 more)

### Community 36 - "gsm8k.py"
Cohesion: 0.15
Nodes (16): _clean_extracted_number(), digest(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k() (+8 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.08
Nodes (31): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), is_permissive_license(), repair_preimage_coverage(), code_signature(), export_pool() (+23 more)

### Community 38 - "benchmark_reset.py"
Cohesion: 0.13
Nodes (7): remote_reset_plan(), reset_local_benchmarks(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training(), main()

### Community 39 - "types"
Cohesion: 0.10
Nodes (4): test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), download(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only(), test_standalone_export_requires_matching_remote_contents()

### Community 40 - "TrainingDashboard"
Cohesion: 0.22
Nodes (4): render_dashboard(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 41 - "TestBifocalGemmaArchitecture"
Cohesion: 0.09
Nodes (4): SwiGLUDualGateBridge, TestBifocalGemmaArchitecture, test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight(), test_swiglu_bridge_non_redundant_single_post_gain()

### Community 42 - "protocol.py"
Cohesion: 0.20
Nodes (10): build_effective_benchmark_spec(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer(), repair_report_metrics(), _runtime_model_identity(), _surface_strict_and_resilient() (+2 more)

### Community 43 - "test_gsm8k.py"
Cohesion: 0.17
Nodes (8): FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train(), test_verified_preparation_and_input_only_loading()

### Community 44 - "evaluation/context.py"
Cohesion: 0.08
Nodes (26): build_repo_niah_case(), _context_already_includes_instruction(), _count_niah_tokens(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens(), encoder_context(), generate_repo_niah_haystack(), greedy_decode() (+18 more)

### Community 45 - "BifocalGemmaForCausalLM"
Cohesion: 0.10
Nodes (3): BifocalGemmaForCausalLM, test_bifocalgemma_merge_and_unload(), test_enable_memory_efficient_training_and_softcapping()

### Community 47 - "test_data_remediation.py"
Cohesion: 0.15
Nodes (19): audit_rows(), event_key(), inspect_structure(), merge_lineage(), preference(), remediate_records(), WithoutDocstrings, record() (+11 more)

### Community 48 - "math"
Cohesion: 0.08
Nodes (18): NormalizedAlignment, _build_group_schedule(), build_training_groups(), default_training_profiles(), Float32Alignment, training_policy_table(), validate_training_policy(), comparison_recipes() (+10 more)

### Community 49 - "json"
Cohesion: 0.14
Nodes (14): archive_files(), audit_split_disjointness(), build_release(), main(), publish_release(), rows(), validate_code_eval_snapshots(), write_card() (+6 more)

### Community 50 - "configure_full_rank_cross_attention"
Cohesion: 0.16
Nodes (7): Bf16FullLinear, configure_full_rank_cross_attention(), FullCrossTiledAttention, _resolve_gate_init_tanh(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_decoupled_upper_full_attention_cross_kv_topology()

### Community 51 - "repository_study.py"
Cohesion: 0.22
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), prepare_repository_record(), fetch_tokenizer_assets(), generate_run() (+10 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.14
Nodes (13): bounded_threads(), CharacterTokenizer, diff(), ExecutingScoringOptions, fixture(), ScoringOptions, ScriptedModel, test_context_budget_and_hidden_tests_are_not_rendered() (+5 more)

### Community 53 - ".configure_stage1"
Cohesion: 0.20
Nodes (3): test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration(), test_warmup_cosine_learning_rate_schedule()

### Community 54 - "test_v4_milestone_checkpoints_survive_pruning_and_hub_retention"
Cohesion: 0.08
Nodes (6): test_fast_pipeline_resume_identity_mismatch_and_corrupt_checkpoint(), test_matrix_resume_accepts_token_identical_prompt_source_reformat(), test_patch_paths_ignores_hunk_body_dash_and_plus_prefixes(), test_publish_release_replace_current_deletes_audits(), test_v4_milestone_checkpoints_survive_pruning_and_hub_retention(), _write_ready_ckpt()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "hub.py"
Cohesion: 0.15
Nodes (9): checked_prefix(), recover_remote_run(), _copy_atomic(), RemoteHeartbeat, retry_delay(), _synchronize_once(), synchronize_run(), target_fingerprint() (+1 more)

### Community 59 - "objectives.py"
Cohesion: 0.08
Nodes (19): _resolve_example_diff_flag(), backward_window(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_example_loss_weights() (+11 more)

### Community 60 - "repository_corpus.py"
Cohesion: 0.10
Nodes (17): build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license(), RepositoryPreparationLedger (+9 more)

### Community 61 - "test_shared_schedule.py"
Cohesion: 0.25
Nodes (13): canonical_schedule_records(), rows(), schedule(), test_all_originals_once_with_disjoint_stage_windows(), test_atomic_manifest_reuse_and_fail_closed_mismatch(), test_deterministic_model_independent_schedule_and_mapping(), test_each_semantic_mutation_changes_signature(), test_empty_required_fields_rejected() (+5 more)

### Community 62 - "audit_context_release.py"
Cohesion: 0.24
Nodes (5): ArchiveSnapshotStore, test_failed_download_removes_partial_bytes(), audit_pool(), main(), source_notices()

### Community 64 - "completion_metrics"
Cohesion: 0.15
Nodes (10): completion_metrics(), _extract_full_python_code(), normalize_completion_prediction(), _sequence_and_identifier_metrics(), _strip_echoed_suffix(), _strip_fim_markers(), _strip_markdown_fences(), probe_unicode_and_metrics() (+2 more)

### Community 66 - "run_matrix_training"
Cohesion: 0.19
Nodes (6): _backfill_final_stage(), run_matrix_training(), check_legacy_artifacts(), snapshot(), latest_checkpoint(), StageLedger

### Community 67 - "memory_efficient.py"
Cohesion: 0.16
Nodes (7): joint_visibility_tile(), _kv_slice_bounds(), mask_tile(), _TiledAttention, TiledEncoderAttention, TiledMergedAttention, sample_encoder_mass()

### Community 68 - "test_context_tasks.py"
Cohesion: 0.14
Nodes (7): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose()

### Community 69 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 72 - "test_effective_benchmark_protocol.py"
Cohesion: 0.15
Nodes (8): test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold(), decode(), test_effective_allowance(), test_effective_spec_binds_gold_caps_and_runtime_spec(), test_invalid_allowance_fails_explicitly(), test_prepare_sampled_benchmark_collection_invalidates_stale_cache(), test_strict_primary_and_resilient_diagnostic_share_prediction(), test_suite_caps_and_ten_model_allowances()

### Community 74 - "subprocess"
Cohesion: 0.17
Nodes (8): _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract, TestTokenizerVerification

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "fit_offline_sft"
Cohesion: 0.11
Nodes (13): fit_offline_sft(), emit(), model_stage(), fit_repository_sft(), append_event(), examples(), TestOfflineSFT, TinyLossModel (+5 more)

### Community 78 - "WigglyDashboard"
Cohesion: 0.26
Nodes (7): phase_category(), WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 80 - "argparse"
Cohesion: 0.15
Nodes (9): audit_views(), main(), main(), main(), render(), inspect(), main(), main() (+1 more)

### Community 82 - "Any"
Cohesion: 0.15
Nodes (9): _final_stage_info(), _latest_checkpoint_matches(), _local_standalone_export_valid(), _resolve_profile_settings(), resolve_training_contract(), _reusable_training_report(), _schedule_signature_matches(), _standalone_export_exists() (+1 more)

### Community 83 - "RepositorySnapshot"
Cohesion: 0.22
Nodes (10): canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), NativeTaskResult, score_python_patch(), summarize_native_results() (+2 more)

### Community 85 - "objectives.py"
Cohesion: 0.07
Nodes (21): build_bifocalgemma_optimizer(), extract_trivial_token_ids(), is_cross_bridge_parameter(), is_zero_weight_decay_bridge_parameter(), resolve_diff_token_ids(), update_policy_learning_rates(), build_bifocalgemma_adamw(), build_partitioned_param_groups() (+13 more)

### Community 87 - "ExperimentSpec"
Cohesion: 0.18
Nodes (7): ExperimentSpec, get_v2_experiment(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 89 - "test_construction_blocks.py"
Cohesion: 0.08
Nodes (7): run_length_grid(), TestTokenRoleAndDataIdentity, test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library(), test_notebook_sweep_persists_results_per_case(), limit_threads()

### Community 91 - "build_and_publish_v4_dataset.py"
Cohesion: 0.15
Nodes (16): build_and_verify_mixed_pool(), build_and_verify_release(), _build_v4_notices(), _extract_python_fim_from_code(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers(), _log() (+8 more)

### Community 92 - "test_overflow_fallback_recomputes_budget_and_records_metadata"
Cohesion: 0.17
Nodes (6): _CharTokenizer, _DummyEvalModel, test_effective_spec_includes_overflow_fallback_policy(), test_external_runner_resume_and_signature_guard(), test_overflow_fallback_recomputes_budget_and_records_metadata(), _write_mcq_benchmark()

### Community 93 - "wiggly_dashboard.py"
Cohesion: 0.15
Nodes (4): format_stage_display(), LastSeenWidget, TrainingLossLog, test_dashboard_stage_display_formatting()

### Community 96 - "pytest"
Cohesion: 0.18
Nodes (9): generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), parent(), test_evaluation_and_unlicensed_sources_rejected(), test_grounded_generation_answerability_provenance_and_target_independence() (+1 more)

### Community 101 - "test_molab_remediation.py"
Cohesion: 0.05
Nodes (30): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+22 more)

### Community 102 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 103 - "test_cross_attention.py"
Cohesion: 0.26
Nodes (7): flex_kernel_options(), fused_attention(), attention_tile(), test_flex_backward_head256_sequence_above_small_sequence_branch(), test_flex_joint_gqa_matches_reference_on_cuda(), test_flex_structured_native_masks_across_sequence_lengths(), test_joint_gqa_output_and_gradients_at_native_head_widths()

### Community 104 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 107 - "InferenceCache"
Cohesion: 0.15
Nodes (6): InferenceCache, test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence(), test_full_cross_cached_attention_sdpa_guard_fallback_and_non_left_padded_step_valid(), _run_cached(), test_gate_init_tanh_resolution_dynamic_and_v4_fallback(), test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa()

### Community 108 - "tokenization.py"
Cohesion: 0.11
Nodes (14): RepairExample, example_from_record(), decode_text(), format_encoder_context(), tokenize_repair_example(), batched_greedy_decode(), generate_diagnostic_text(), generate_patch() (+6 more)

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.25
Nodes (4): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.15
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 114 - "export_matrix_pretrained_model"
Cohesion: 0.11
Nodes (5): BifocalGemmaPreTrainedModel, export_matrix_pretrained_model(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_export_matrix_pretrained_model(), test_export_matrix_pretrained_model_baseline()

### Community 116 - "collections"
Cohesion: 0.11
Nodes (10): editing_manifest(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation(), test_repository_eval_is_rejected() (+2 more)

### Community 120 - "test_tier3_combinations.py"
Cohesion: 0.19
Nodes (4): validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 121 - "atomic_json"
Cohesion: 0.14
Nodes (12): prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), _paired_budget_for(), decoder_prompt(), _prompt_ids_with_metadata(), run_context_benchmark(), activity() (+4 more)

### Community 122 - "hashlib"
Cohesion: 0.08
Nodes (27): normalize_context_task(), token_profile(), curated_pool_digest(), editing_record(), git_text_lines(), load_curated_pool(), prepare_editing_records(), repository_editing_record() (+19 more)

### Community 123 - "run_matrix_benchmark"
Cohesion: 0.18
Nodes (6): benchmark_directory_name(), download_model(), run_matrix_benchmark(), snapshot(), run_matrix_training_and_benchmark(), progress()

### Community 124 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 137 - "comparison.py"
Cohesion: 0.24
Nodes (3): compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), main()

### Community 142 - "JSONLRepairDataset"
Cohesion: 0.16
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 173 - ".from_pretrained_backbones"
Cohesion: 0.25
Nodes (3): _resolve_backbone_weights(), test_from_pretrained_backbones_baseline(), test_from_pretrained_backbones_loading()

### Community 175 - "sdpa_cross_attention"
Cohesion: 0.29
Nodes (3): build_joint_visibility(), sdpa_cross_attention(), _sdpa_supported()

### Community 177 - "training/diagnostics.py"
Cohesion: 0.32
Nodes (4): encoder_diagnostics(), stratified_subset(), evaluate_sft_loss(), validation_batches()

### Community 180 - "sdpa_cross_attention"
Cohesion: 0.18
Nodes (5): _compute_per_row_cross_query(), _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), test_compute_per_row_cross_query_has_no_host_sync_calls()

### Community 184 - "source_fingerprints"
Cohesion: 0.29
Nodes (4): source_fingerprints(), WithoutDocstrings, test_reusable_training_report_backfills_missing_final_stage_when_verified(), _build_run()

### Community 187 - "tiled_attention"
Cohesion: 0.38
Nodes (4): sdpa_attention(), tiled_attention(), test_kv_sliced_tiled_and_sdpa_attention_exact_forward_backward_parity(), test_sdpa_attention_safe_fallback_on_float64_and_custom_float_mask()

### Community 188 - "audit_dataset_release.py"
Cohesion: 0.52
Nodes (4): audit(), emit(), main(), validate_all_patches()

### Community 197 - "test_molab_matrix_notebooks.py"
Cohesion: 0.06
Nodes (20): configure_policy_model(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length(), test_cached_uncached_and_batched_greedy_decode_equivalence(), test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_configure_full_rank_cross_attention_v3_specs(), test_configure_matrix_stage_parameter_freezing_paired_and_baseline(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_fast_training_config_and_pipeline_baseline_single_stage() (+12 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1047 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **91 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `atomic_json()` connect `atomic_json` to `preflight.py`, `checkpoint.py`, `benchmark_upload.py`, `pathlib`, `repository.py`, `matrix.py`, `stable_fingerprint`, `run_external_benchmark`, `prepare_context_data.py`, `benchmark_reset.py`, `types`, `evaluation/context.py`, `json`, `hub.py`, `source_fingerprints`, `objectives.py`, `audit_context_release.py`, `run_matrix_training`, `argparse`, `build_and_publish_v4_dataset.py`, `hashlib`, `run_matrix_benchmark`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `stable_fingerprint()` (e.g. with `test_generated_preparation_consumes_frozen_raw_repository_pool()` and `test_legacy_paired_manifest_remains_byte_identical()`) actually correct?**
  _`stable_fingerprint()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `preflight.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11576354679802955 - nodes in this community are weakly interconnected._
- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `memory_efficient.py`, `pathlib`, `TestBifocalGemmaArchitecture`, `InferenceCache`, `.from_pretrained_backbones`, `test_pretrained_hf_model.py`, `make_experiment_spec`, `export_matrix_pretrained_model`, `configure_full_rank_cross_attention`, `Seq2SeqBatch`, `BifocalGemmaConfig`, `FastTrainingPipeline`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Should `checkpoint.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09388335704125178 - nodes in this community are weakly interconnected._
- **Why does `RepositorySnapshot` connect `RepositorySnapshot` to `.snapshot`, `subprocess`, `FeatureNotReadyError`, `tokenization.py`, `repository.py`, `evaluation/context.py`, `RepositorySnapshot`, `repository_study.py`, `test_repository_study.py`, `repository_corpus.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._