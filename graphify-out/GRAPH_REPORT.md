# Graph Report - bifocalgemma  (2026-11-21)

## Corpus Check
- 257 files · ~371,520 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2628 nodes · 7207 edges · 191 communities (91 shown, 100 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 884 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `05d21a63`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- telemetry.py
- test_v4_architecture.py
- benchmark_upload.py
- build_native_tiny_model
- patches.py
- benchmark_reset.py
- parametrize
- test_memory_efficient_training.py
- generation.py
- test_matrix_runtime_loader.py
- BifocalGemmaConfig
- schedule.py
- build_and_publish_v4_dataset.py
- test_tier2_boundary.py
- TokenizedRepairExample
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- hashlib
- test_molab_matrix_notebooks.py
- context_tasks.py
- pathlib
- adversarial_r3_probe.py
- typing
- FastTrainingPipeline
- test_v4_data_disjoint.py
- curriculum.py
- curation.py
- external_runner.py
- examples
- contracts.py
- prepare_context_data.py
- matrix.py
- test_gsm8k.py
- TrainingDashboard
- SwiGLUDualGateBridge
- run_context_benchmark
- scorer_fingerprint
- evaluation/context.py
- test_molab_remediation.py
- generate_full_matrix_notebooks.py
- test_offline_sft_training.py
- native_config
- test_tier4_scenarios.py
- repository_study.py
- test_repository_study.py
- copy
- CodeOverlapIndex
- test_matrix_schedule_runtime.py
- remediation.py
- BifocalGemmaForCausalLM
- objectives.py
- re
- RepositorySnapshot
- repository_corpus.py
- test_tier3_combinations.py
- GrokOrScaleLoRA
- test_training_telemetry.py
- torch
- test_context_tasks.py
- test_pretrained_probe_helpers.py
- test_repository_corpus.py
- subprocess
- test_construction_blocks.py
- training/offline_sft.py
- TiledEncoderAttention
- test_editing_manifest.py
- generate_repository_notebook.py
- restore_matrix_checkpoint
- build_repository_corpus
- memory_efficient.py
- policy.py
- ExperimentSpec
- test_offline_sft_training.py
- bifocalgemma
- releases.py
- benchmark.py
- Seq2SeqBatch
- checkpoint_transfer.py
- normalize_context_task
- test_context_benchmark_mcq_budget_floor_execution
- test_tier1_features.py
- test_data_remediation.py
- pytest
- test_optimized_native_resume_matches_full_and_rejects_different_recipe
- tokenization.py
- fit_curated_sft
- patch_matrix_tokenization
- DummyTrackingTokenizer
- bifocalgemma/__init__.py
- PackedTokenizedDataset
- json
- WigglyDashboard
- fit_offline_sft
- load_curated_pool
- Bf16FullLinear
- test_export_matrix_pretrained_model_with_subfolder_and_tokenizer
- stable_fingerprint
- build_joint_visibility
- TestMemoryProbe
- JSONLRepairDataset
- test_evaluation.py
- NativeMergedAttention
- test_benchmark_prompt_budget.py
- tokenize_repair_example
- test_matrix_quarantine_legacy_checkpoint_vs_clean_resume
- test_fast_pipeline_respects_learnable_cross_scale_default_and_override
- TestCheckpointTransfer
- configure_full_rank_cross_attention
- patch_cross_entropy
- tests_e2e/conftest.py

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 67 edges
2. `atomic_json()` - 61 edges
3. `run_matrix_training()` - 51 edges
4. `load_verified_tokenizer()` - 51 edges
5. `BifocalGemmaForCausalLM` - 49 edges
6. `RepositorySnapshot` - 45 edges
7. `run_matrix_benchmark()` - 41 edges
8. `TokenizedRepairExample` - 41 edges
9. `run_matrix_stage()` - 40 edges
10. `FastTrainingPipeline` - 38 edges

## Surprising Connections (you probably didn't know these)
- `download()` --calls--> `NativeTinyModel`  [INFERRED]
  tests/test_matrix_schedule_runtime.py → src/bifocalgemma/modeling_bifocalgemma.py
- `test_prepare_sampled_benchmark_collection_deterministic_100_cases()` --calls--> `prepare_sampled_benchmark_collection()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/evaluation/benchmark.py
- `fetch()` --calls--> `repository_task_from_row()`  [INFERRED]
  tools/prepare_context_data.py → src/bifocalgemma/data/repository.py
- `test_repair_record_type_has_same_prefix_as_explicit_repair_task()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `generate()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (191 total, 100 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.20
Nodes (9): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution(), save() (+1 more)

### Community 1 - "telemetry.py"
Cohesion: 0.15
Nodes (5): format_duration(), format_stage_display(), _gpu_utilization_snapshot(), phase_category(), test_dashboard_stage_display_formatting()

### Community 2 - "test_v4_architecture.py"
Cohesion: 0.09
Nodes (12): critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), spy_sdpa(), test_full_cross_cached_attention_sdpa_guard_fallback_and_non_left_padded_step_valid() (+4 more)

### Community 3 - "benchmark_upload.py"
Cohesion: 0.07
Nodes (18): BenchmarkUploadQueue, retry_delay(), append_event(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush() (+10 more)

### Community 5 - "build_native_tiny_model"
Cohesion: 0.22
Nodes (3): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head()

### Community 6 - "patches.py"
Cohesion: 0.14
Nodes (14): _add_missing_git_headers(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines(), _is_file_header_pair() (+6 more)

### Community 7 - "benchmark_reset.py"
Cohesion: 0.13
Nodes (9): remote_reset_plan(), reset_local_benchmarks(), reset_state(), checked_prefix(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models() (+1 more)

### Community 8 - "parametrize"
Cohesion: 0.05
Nodes (40): dependency_state(), test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready() (+32 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.24
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 11 - "generation.py"
Cohesion: 0.12
Nodes (13): decode_text(), format_encoder_context(), generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids(), _safe_apply_chat_template() (+5 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.38
Nodes (11): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_release_check_rejects_local_preview() (+3 more)

### Community 14 - "BifocalGemmaConfig"
Cohesion: 0.09
Nodes (16): BifocalGemmaConfig, _resolve_backbone_weights(), test_bridge_init_preserved_across_post_init(), test_from_pretrained_raises_file_not_found_when_weights_missing(), test_generate_multi_eos_per_row_stopping_and_cache_parity(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation() (+8 more)

### Community 15 - "schedule.py"
Cohesion: 0.08
Nodes (31): _build_schedule_for_epochs(), _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin() (+23 more)

### Community 16 - "build_and_publish_v4_dataset.py"
Cohesion: 0.14
Nodes (15): build_and_verify_mixed_pool(), build_and_verify_release(), _build_v4_notices(), fetch_all_v4_sources(), _iter_jsonl(), _load_gemma4_tokenizers(), _log(), main() (+7 more)

### Community 17 - "test_tier2_boundary.py"
Cohesion: 0.14
Nodes (6): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement()

### Community 18 - "TokenizedRepairExample"
Cohesion: 0.17
Nodes (7): TokenizedRepairExample, RepairCollator, __getattr__(), run_training_memory_preflight(), synthetic_stress_target(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.07
Nodes (10): _prompt_ids_impl(), MockCausalLM, MockQwenTokenizer, _ProportionalTokenizer, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(), test_reponiah_32k_zero_overflow_for_bifocalgemma_and_qwen25() (+2 more)

### Community 20 - "eval_code.py"
Cohesion: 0.14
Nodes (7): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.07
Nodes (23): extract_mcq_letter(), strict_letter(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards(), test_adversarial_mcq_isolated_formatting(), test_adversarial_mcq_lowercase_letter_normalization() (+15 more)

### Community 23 - "hashlib"
Cohesion: 0.12
Nodes (19): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), repository_editing_record(), load_verified_tokenizer() (+11 more)

### Community 24 - "test_molab_matrix_notebooks.py"
Cohesion: 0.08
Nodes (25): make_experiment_spec(), ModelOutput, export_matrix_pretrained_model(), NativeTinyModel, configure_full_rank_cross_attention(), _resolve_gate_init_tanh(), configure_memory_efficient_training(), save_matrix_checkpoint() (+17 more)

### Community 25 - "context_tasks.py"
Cohesion: 0.17
Nodes (17): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), _parse_file_sections(), _render_entries(), _resolve_eot_token_id() (+9 more)

### Community 26 - "pathlib"
Cohesion: 0.14
Nodes (4): audit_pool(), audit_views(), main(), main()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.12
Nodes (7): probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), verify_gate_round_2_resolved_cases()

### Community 28 - "typing"
Cohesion: 0.07
Nodes (10): BackboneProfile, CheckpointRef, _pair(), _ref(), CachedAttention, is_cross_bridge_parameter(), clamped_triton_kernel_options(), ensure_hardware_memory_guards() (+2 more)

### Community 29 - "FastTrainingPipeline"
Cohesion: 0.08
Nodes (13): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration() (+5 more)

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.16
Nodes (10): _CharTokenizer, _load_prepare_module(), _numbered_line_numbers(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_mixed_pool_v4_curriculum_grounded_and_curated_sft(), test_multifunction_completion_extraction_removes_single_return_cap() (+2 more)

### Community 31 - "curriculum.py"
Cohesion: 0.19
Nodes (9): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length() (+1 more)

### Community 32 - "curation.py"
Cohesion: 0.12
Nodes (16): read_arrow_rows(), split_connected_records(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files() (+8 more)

### Community 34 - "external_runner.py"
Cohesion: 0.11
Nodes (18): build_repo_niah_case(), format_crosscodelong_summary(), _render_niah_prompt_tokens(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch() (+10 more)

### Community 36 - "contracts.py"
Cohesion: 0.07
Nodes (14): DeveloperAgent, EvaluationReport, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter (+6 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.10
Nodes (24): export_pool(), adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), completion_from_snapshot(), completions_from_snapshot() (+16 more)

### Community 38 - "matrix.py"
Cohesion: 0.06
Nodes (30): benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), _backfill_final_stage(), _example_sequence_lengths(), _final_stage_info(), _latest_checkpoint_matches() (+22 more)

### Community 39 - "test_gsm8k.py"
Cohesion: 0.06
Nodes (12): FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train(), test_verified_preparation_and_input_only_loading() (+4 more)

### Community 40 - "TrainingDashboard"
Cohesion: 0.16
Nodes (6): loss_chart(), render_dashboard(), resource_snapshot(), TrainingDashboard, test_monitor_escapes_errors_updates_validation_and_exports_html(), test_telemetry_history_appends_all_active_stages()

### Community 41 - "SwiGLUDualGateBridge"
Cohesion: 0.06
Nodes (6): SwiGLUDualGateBridge, TestBifocalGemmaArchitecture, TestGemma4EdgeTopology, test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads(), test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight(), test_swiglu_bridge_non_redundant_single_post_gain()

### Community 42 - "run_context_benchmark"
Cohesion: 0.10
Nodes (18): run_context_benchmark(), activity(), build_effective_benchmark_spec(), effective_generation_budget(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer() (+10 more)

### Community 43 - "scorer_fingerprint"
Cohesion: 0.16
Nodes (10): _extract_match_letter(), fallback_letter_regex(), score_context_case(), score_repo_niah_case(), _clean_extracted_number(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer() (+2 more)

### Community 44 - "evaluation/context.py"
Cohesion: 0.07
Nodes (30): completion_metrics(), _context_already_includes_instruction(), _count_niah_tokens(), decoder_prompt(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens(), encoder_context(), _extract_full_python_code() (+22 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.12
Nodes (6): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues(), test_mcq_budget_floor_in_v3_benchmark_logic()

### Community 47 - "generate_full_matrix_notebooks.py"
Cohesion: 0.27
Nodes (6): test_invalid_generation_modes_cannot_publish(), test_publication_only_adds_verified_immutable_runtime(), download(), format_code(), main(), render_notebook()

### Community 48 - "test_offline_sft_training.py"
Cohesion: 0.19
Nodes (5): examples(), TestOfflineSFT, TinyLossModel, test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 50 - "test_tier4_scenarios.py"
Cohesion: 0.29
Nodes (3): test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 51 - "repository_study.py"
Cohesion: 0.25
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), read_source_rows(), fetch_tokenizer_assets(), generate_run() (+10 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.11
Nodes (20): canonical_snapshot_hash(), example_from_record(), NativeTaskResult, score_python_patch(), summarize_native_results(), CharacterTokenizer, diff(), ExecutingScoringOptions (+12 more)

### Community 53 - "copy"
Cohesion: 0.07
Nodes (21): build_bifocalgemma_optimizer(), default_training_profiles(), training_policy_table(), update_policy_learning_rates(), resolve_training_cadence(), unified_recipe(), v4_group_defaults(), v4_optimizer_defaults() (+13 more)

### Community 54 - "CodeOverlapIndex"
Cohesion: 0.12
Nodes (22): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), raw() (+14 more)

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.08
Nodes (14): run_matrix_stage(), fixture_data(), fixture_model(), test_all_ten_real_stage_loaders_consume_identical_windows(), test_interrupted_optimizer_boundary_resume_matches_uninterrupted(), test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export() (+6 more)

### Community 56 - "remediation.py"
Cohesion: 0.13
Nodes (10): inspect_python_target(), inspect_structure(), main(), preference(), refresh_patch_targets(), WithoutDocstrings, test_errors_are_still_rejected_and_other_warnings_still_visible(), test_final_cohort_summary_does_not_include_filtered_examples() (+2 more)

### Community 59 - "objectives.py"
Cohesion: 0.06
Nodes (23): _resolve_example_diff_flag(), backward_window(), chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_example_loss_weights() (+15 more)

### Community 60 - "re"
Cohesion: 0.12
Nodes (14): ContextWindow, render_window(), select_repository_context(), repair_preimage_coverage(), _terms(), git_environment(), v4_repository_task_from_row(), GitSnapshotStore (+6 more)

### Community 61 - "RepositorySnapshot"
Cohesion: 0.14
Nodes (11): RepositoryFile, RepositorySnapshot, apply_patch_to_snapshot(), patch_syntax(), test_dependency_completion_removes_target_and_retains_real_helper(), TestRealOfflineTokenizers, test_diagnose_patch_application_rescues_recount_without_weakening_strict(), test_missing_old_file_is_still_rejected() (+3 more)

### Community 62 - "repository_corpus.py"
Cohesion: 0.12
Nodes (7): ArchiveSnapshotStore, context_license_evidence(), path_license_evidence(), recognize_license(), resolved_license(), test_failed_download_removes_partial_bytes(), source_notices()

### Community 63 - "test_tier3_combinations.py"
Cohesion: 0.19
Nodes (4): validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 64 - "GrokOrScaleLoRA"
Cohesion: 0.24
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.12
Nodes (5): _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime(), RecordingStream

### Community 67 - "torch"
Cohesion: 0.21
Nodes (9): flex_kernel_options(), fused_attention(), attention_tile(), tiled_attention(), test_flex_backward_head256_sequence_above_small_sequence_branch(), test_flex_joint_gqa_matches_reference_on_cuda(), test_flex_structured_native_masks_across_sequence_lengths(), test_joint_gqa_output_and_gradients_at_native_head_widths() (+1 more)

### Community 68 - "test_context_tasks.py"
Cohesion: 0.10
Nodes (7): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_short_answer_scoring_rejects_unrequested_prose(), test_tokenize_context_task_preserves_target_indentation_and_newlines()

### Community 69 - "test_pretrained_probe_helpers.py"
Cohesion: 0.20
Nodes (5): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library(), test_notebook_sweep_persists_results_per_case()

### Community 72 - "test_repository_corpus.py"
Cohesion: 0.27
Nodes (9): patch_license_evidence(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport(), read(), test_context_files_are_licensed_separately_from_patch(), test_license_document_include_is_resolved_inside_snapshot() (+1 more)

### Community 74 - "subprocess"
Cohesion: 0.21
Nodes (8): RepairExample, _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord, snapshot_hash(), SWESmithRecord, fixture_records(), TestSWESmithContract

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "training/offline_sft.py"
Cohesion: 0.23
Nodes (8): cpu_tree(), evaluate_sft_loss(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), validation_batches(), optimizer_group_manifest(), retained_checkpoints()

### Community 79 - "test_editing_manifest.py"
Cohesion: 0.28
Nodes (8): editing_manifest(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation(), test_repository_eval_is_rejected()

### Community 80 - "generate_repository_notebook.py"
Cohesion: 0.26
Nodes (5): matrix_bundle(), main(), render(), source_bundle(), source_bundle_bytes()

### Community 82 - "restore_matrix_checkpoint"
Cohesion: 0.18
Nodes (8): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), synchronize_matrix(), snapshot(), snapshot(), validate_checkpoint_files()

### Community 83 - "build_repository_corpus"
Cohesion: 0.18
Nodes (7): count_syntax_warnings(), empty_syntax_warning_summary(), build_repository_corpus(), prepare_repository_corpus(), RepositoryPreparationLedger, test_curated_warning_target_retained_and_summary_distinguishes_duplicate(), test_preparation_ledger_preserves_records_and_rejects_new_settings()

### Community 84 - "memory_efficient.py"
Cohesion: 0.12
Nodes (10): joint_visibility_tile(), _kv_slice_bounds(), mask_tile(), sdpa_attention(), sdpa_cross_attention(), _sdpa_supported(), _TiledAttention, encoder_diagnostics() (+2 more)

### Community 85 - "policy.py"
Cohesion: 0.09
Nodes (14): NormalizedAlignment, configure_probe_training(), gpu_memory(), optimizer_bytes(), ProbeLoRALinear, profile_training_group(), tensor_bytes(), is_zero_weight_decay_bridge_parameter() (+6 more)

### Community 87 - "ExperimentSpec"
Cohesion: 0.16
Nodes (7): ExperimentSpec, get_v2_experiment(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 91 - "releases.py"
Cohesion: 0.08
Nodes (31): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+23 more)

### Community 92 - "benchmark.py"
Cohesion: 0.16
Nodes (7): batched_greedy_decode(), infer_benchmark_weight_tier(), prepare_sampled_benchmark_collection(), resolve_optimal_eval_batch_size(), run_batched_context_benchmark(), activity(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight()

### Community 95 - "checkpoint_transfer.py"
Cohesion: 0.17
Nodes (5): build_native_model(), load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), test_local_loader_preflights_and_preserves_shared_words()

### Community 96 - "normalize_context_task"
Cohesion: 0.21
Nodes (9): is_permissive_license(), normalize_context_task(), generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), validate_context_license_record() (+1 more)

### Community 101 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 102 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (12): validate_paper_manuscript(), test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval() (+4 more)

### Community 103 - "test_data_remediation.py"
Cohesion: 0.26
Nodes (12): merge_lineage(), remediate_records(), record(), test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined(), test_canonical_git_patch_batch_including_missing_final_newline(), test_conflicting_event_variants_are_all_quarantined(), test_doc_only_and_warning_targets_are_removed(), test_event_variants_choose_documented_export_and_merge_lineage() (+4 more)

### Community 104 - "pytest"
Cohesion: 0.43
Nodes (4): parent(), test_evaluation_and_unlicensed_sources_rejected(), test_grounded_generation_answerability_provenance_and_target_independence(), test_grounded_mutation_rejected()

### Community 105 - "test_optimized_native_resume_matches_full_and_rejects_different_recipe"
Cohesion: 0.29
Nodes (4): comparison_recipes(), test_fp32_alignment_on_bf16_backbone(), test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 109 - "fit_curated_sft"
Cohesion: 0.22
Nodes (6): validate_curated_pool_limits(), fit_curated_sft(), test_complete_epoch_training_budget_all_families(), test_curated_resume_rejects_runtime_drift(), test_frozen_budget_matches_all_six_controls(), test_invalid_requested_pool_budget_is_rejected()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 116 - "json"
Cohesion: 0.08
Nodes (11): compare_predictions(), test_external_activity_reader_does_not_need_running_kernel(), test_paired_bootstrap_and_cohort_rejection(), main(), main(), parse_args(), inspect(), main() (+3 more)

### Community 120 - "WigglyDashboard"
Cohesion: 0.24
Nodes (8): LastSeenWidget, TrainingLossLog, WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 121 - "fit_offline_sft"
Cohesion: 0.50
Nodes (4): fit_offline_sft(), emit(), model_stage(), DurableClock

### Community 122 - "load_curated_pool"
Cohesion: 0.14
Nodes (12): curated_pool_digest(), load_curated_pool(), prepare_editing_records(), split_editing_records(), prepare_curated_training(), tokenize(), prepare_repository_training(), assets() (+4 more)

### Community 130 - "stable_fingerprint"
Cohesion: 0.12
Nodes (23): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), file_digest(), stable_fingerprint() (+15 more)

### Community 142 - "JSONLRepairDataset"
Cohesion: 0.12
Nodes (5): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset, test_fast_training_pipeline_features_and_export(), test_fast_training_pipeline_with_jsonl_data_dir()

### Community 146 - "test_evaluation.py"
Cohesion: 0.15
Nodes (4): prompt_ids(), test_prompt_ids_instance_id_isolation_and_crosscodelong_normalization(), test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity(), test_prompt_ids_instance_id_isolation_and_gold_rejection()

### Community 172 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 180 - "configure_full_rank_cross_attention"
Cohesion: 0.17
Nodes (6): _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv()

### Community 196 - "tests_e2e/conftest.py"
Cohesion: 0.15
Nodes (6): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1017 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **100 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `test_v4_architecture.py`, `parametrize`, `TiledEncoderAttention`, `BifocalGemmaConfig`, `test_export_matrix_pretrained_model_with_subfolder_and_tokenizer`, `JSONLRepairDataset`, `bifocalgemma/__init__.py`, `configure_full_rank_cross_attention`, `Seq2SeqBatch`, `Bf16FullLinear`, `typing`, `FastTrainingPipeline`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `test_v4_architecture.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08994708994708994 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `preflight.py`, `hashlib`, `pathlib`, `curation.py`, `prepare_context_data.py`, `matrix.py`, `run_context_benchmark`, `evaluation/context.py`, `copy`, `CodeOverlapIndex`, `test_matrix_schedule_runtime.py`, `remediation.py`, `repository_corpus.py`, `test_editing_manifest.py`, `build_repository_corpus`, `releases.py`, `benchmark.py`, `test_data_remediation.py`, `load_curated_pool`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `benchmark_upload.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07171717171717172 - nodes in this community are weakly interconnected._
- **Why does `atomic_json()` connect `stable_fingerprint` to `preflight.py`, `benchmark_upload.py`, `benchmark_reset.py`, `schedule.py`, `build_and_publish_v4_dataset.py`, `pathlib`, `typing`, `external_runner.py`, `prepare_context_data.py`, `matrix.py`, `test_gsm8k.py`, `run_context_benchmark`, `evaluation/context.py`, `objectives.py`, `repository_corpus.py`, `restore_matrix_checkpoint`, `releases.py`, `benchmark.py`, `json`, `fit_offline_sft`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._