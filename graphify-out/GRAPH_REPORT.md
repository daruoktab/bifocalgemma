# Graph Report - bifocalgemma  (2026-11-20)

## Corpus Check
- 255 files · ~358,782 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2580 nodes · 7054 edges · 194 communities (99 shown, 95 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 835 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `140f21db`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- render_dashboard
- contracts.py
- pathlib
- build_native_tiny_model
- re
- stages.py
- test_notebook_dependencies.py
- test_memory_efficient_training.py
- FastTrainingPipeline
- test_matrix_runtime_loader.py
- curation.py
- schedule.py
- test_context_benchmark_resume.py
- test_dashboard_dependency_state.py
- json
- test_v4_eval_parity.py
- eval_code.py
- test_tier5_adversarial.py
- hashlib
- make_experiment_spec
- context_tasks.py
- patch_cross_entropy
- adversarial_r3_probe.py
- tiled_attention
- .run
- test_v4_data_disjoint.py
- curriculum.py
- full_pool.py
- external_runner.py
- examples
- dataclasses
- prepare_context_data.py
- synchronize_matrix
- test_gsm8k.py
- test_molab_matrix_notebooks.py
- SwiGLUDualGateBridge
- benchmark.py
- load_local_native_model
- evaluation/context.py
- test_molab_remediation.py
- BifocalGemmaForCausalLM
- sys
- bifocalgemma/__init__.py
- test_tier4_scenarios.py
- repository_study.py
- test_repository_study.py
- run_matrix_stage
- atomic_json
- test_matrix_schedule_runtime.py
- fit_offline_sft
- BifocalGemmaConfig
- objectives.py
- hashlib
- JSONLRepairDataset
- reset_local_benchmarks
- TestMemoryProbe
- math
- test_training_telemetry.py
- generation.py
- test_context_tasks.py
- run_length_grid
- repository_corpus.py
- RepositorySnapshot
- test_construction_blocks.py
- training/offline_sft.py
- test_editing_manifest.py
- stage.py
- TrainingLossLog
- releases.py
- modeling_bifocalgemma.py
- test_upload_recovery_receipts.py
- TestTokenRoleAndDataIdentity
- bifocalgemma
- test_data_releases.py
- run_matrix_training
- RepairCollator
- remediation.py
- normalize_context_task
- .configure_stage1
- tests_e2e/conftest.py
- test_v4_architecture.py
- sdpa_cross_attention
- fast_pipeline.py
- FullCrossCachedAttention
- notebook_launcher.py
- .snapshot
- patch_matrix_tokenization
- test_fast_pipeline.py
- __main__.py
- packed_tokens.py
- test_training_recipes.py
- pathlib
- WigglyDashboard
- GrokOrScaleLoRA
- stable_fingerprint
- TokenizedRepairExample
- os
- test_tier1_features.py
- tests_e2e/conftest.py
- batched_greedy_decode
- test_evaluation.py
- resolve_diff_token_ids
- test_effective_benchmark_protocol.py
- test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention
- _CharTokenizer
- NativeMergedAttention
- dependency_state
- FullCrossCachedAttention
- export_matrix_pretrained_model
- config.py
- _extract_match_letter
- source_fingerprints
- .update_widgets
- test_matrix_quarantine_legacy_checkpoint_vs_clean_resume
- audit_split_disjointness
- CachedAttention
- test_generate_multi_eos_per_row_stopping_and_cache_parity

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 67 edges
2. `atomic_json()` - 56 edges
3. `BifocalGemmaForCausalLM` - 50 edges
4. `run_matrix_training()` - 50 edges
5. `load_verified_tokenizer()` - 50 edges
6. `RepositorySnapshot` - 45 edges
7. `run_matrix_benchmark()` - 40 edges
8. `TokenizedRepairExample` - 40 edges
9. `FastTrainingPipeline` - 38 edges
10. `fit_offline_sft()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `test_repair_requires_visible_preimage_without_refilling_context()` --calls--> `repair_preimage_coverage()`  [INFERRED]
  tests/test_context_tasks.py → src/bifocalgemma/data/context_tasks.py
- `test_repair_record_type_has_same_prefix_as_explicit_repair_task()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `generate()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `fetch()` --calls--> `repository_task_from_row()`  [INFERRED]
  tools/prepare_context_data.py → src/bifocalgemma/data/repository.py
- `test_prepare_sampled_benchmark_collection_deterministic_100_cases()` --calls--> `prepare_sampled_benchmark_collection()`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/evaluation/benchmark.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (194 total, 95 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.20
Nodes (9): current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution(), save() (+1 more)

### Community 1 - "render_dashboard"
Cohesion: 0.16
Nodes (6): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), test_monitor_escapes_errors_updates_validation_and_exports_html(), test_dashboard_stage_display_formatting()

### Community 2 - "contracts.py"
Cohesion: 0.13
Nodes (5): DeveloperAgent, EvaluationReport, RepairPrediction, write_swebench_predictions(), test_missing_old_file_is_still_rejected()

### Community 3 - "pathlib"
Cohesion: 0.12
Nodes (12): retry_delay(), limited(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_external_activity_reader_does_not_need_running_kernel(), test_fatal_errors_not_hidden(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict() (+4 more)

### Community 5 - "build_native_tiny_model"
Cohesion: 0.22
Nodes (3): build_native_tiny_model(), TestNativeIntegration, test_last_token_logits_match_full_head()

### Community 6 - "re"
Cohesion: 0.17
Nodes (16): _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines(), _is_file_header_pair() (+8 more)

### Community 7 - "stages.py"
Cohesion: 0.20
Nodes (11): file_digest(), synchronize_matrix(), checked_prefix(), recover_remote_run(), _copy_atomic(), _synchronize_once(), synchronize_run(), snapshot() (+3 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.11
Nodes (17): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+9 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.24
Nodes (12): bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager(), test_mixed_precision_backbone_fp32_adapters_match_native() (+4 more)

### Community 11 - "FastTrainingPipeline"
Cohesion: 0.16
Nodes (11): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_training_pipeline_features_and_export(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_training_config_step_reconciliation() (+3 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.27
Nodes (13): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_local_preview_missing_archive_explains_publication_without_network(), test_publication_only_adds_verified_immutable_runtime() (+5 more)

### Community 14 - "curation.py"
Cohesion: 0.09
Nodes (28): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), count_syntax_warnings() (+20 more)

### Community 15 - "schedule.py"
Cohesion: 0.10
Nodes (26): _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json(), _length_bin(), persist_shared_schedule() (+18 more)

### Community 16 - "test_context_benchmark_resume.py"
Cohesion: 0.16
Nodes (5): FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch(), transient()

### Community 17 - "test_dashboard_dependency_state.py"
Cohesion: 0.16
Nodes (13): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+5 more)

### Community 18 - "json"
Cohesion: 0.09
Nodes (19): audit_pool(), audit_views(), main(), audit(), main(), audit(), emit(), main() (+11 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.06
Nodes (15): build_repo_niah_case(), _count_niah_tokens(), _encoded_context_tokens(), generate_repo_niah_haystack(), make_distractor_python_chunk(), _render_niah_prompt_tokens(), MockCausalLM, MockQwenTokenizer (+7 more)

### Community 20 - "eval_code.py"
Cohesion: 0.14
Nodes (7): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.06
Nodes (26): extract_mcq_letter(), strict_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), verify_gate_round_2_resolved_cases(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards() (+18 more)

### Community 23 - "hashlib"
Cohesion: 0.15
Nodes (16): _clean_extracted_number(), digest(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k() (+8 more)

### Community 24 - "make_experiment_spec"
Cohesion: 0.13
Nodes (22): make_experiment_spec(), NativeTinyModel, configure_full_rank_cross_attention(), configure_memory_efficient_training(), is_cross_bridge_parameter(), configure_policy_model(), unified_recipe(), fixture_model() (+14 more)

### Community 25 - "context_tasks.py"
Cohesion: 0.15
Nodes (18): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), is_permissive_license(), _parse_file_sections(), _render_entries() (+10 more)

### Community 26 - "patch_cross_entropy"
Cohesion: 0.18
Nodes (3): TestOfflineSFT, TinyLossModel, test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.13
Nodes (5): probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics(), main()

### Community 28 - "tiled_attention"
Cohesion: 0.15
Nodes (11): flex_kernel_options(), fused_attention(), attention_tile(), build_joint_visibility(), mask_tile(), tiled_attention(), _TiledAttention, test_flex_backward_head256_sequence_above_small_sequence_branch() (+3 more)

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.29
Nodes (7): _load_prepare_module(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_format_numbered_code_and_exports(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(), test_v4_quarantine_licenses_dedup_and_nextcoder_rejection(), _write_mock_curated_pool()

### Community 31 - "curriculum.py"
Cohesion: 0.20
Nodes (9): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length() (+1 more)

### Community 32 - "full_pool.py"
Cohesion: 0.18
Nodes (11): read_arrow_rows(), token_profile(), audit_repository_sources(), build_benchmark_index(), build_full_pool(), rows(), fetch_benchmark_files(), main() (+3 more)

### Community 34 - "external_runner.py"
Cohesion: 0.11
Nodes (17): prepare_sampled_benchmark_collection(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort(), load_hf_external_model_and_tokenizer() (+9 more)

### Community 35 - "examples"
Cohesion: 0.28
Nodes (3): make_identity(), examples(), TestTrainingPolicy

### Community 36 - "dataclasses"
Cohesion: 0.12
Nodes (9): CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter, SWESmithAdapter, extract_token_ast_features(), TokenASTFeatures, BuildBlock (+1 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.10
Nodes (24): repair_preimage_coverage(), adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), completion_from_snapshot(), completions_from_snapshot() (+16 more)

### Community 38 - "synchronize_matrix"
Cohesion: 0.21
Nodes (5): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint()

### Community 39 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 40 - "test_molab_matrix_notebooks.py"
Cohesion: 0.11
Nodes (9): TrainingDashboard, test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_per_model_hf_repo(), test_pipeline_state_ledger_creation_for_remote_resumption(), test_prepare_sampled_benchmark_collection_deterministic_100_cases() (+1 more)

### Community 41 - "SwiGLUDualGateBridge"
Cohesion: 0.06
Nodes (7): NormalizedAlignment, SwiGLUDualGateBridge, TestBifocalGemmaArchitecture, TestGemma4EdgeTopology, test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads(), test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight(), test_swiglu_bridge_non_redundant_single_post_gain()

### Community 42 - "benchmark.py"
Cohesion: 0.13
Nodes (18): decode_text(), run_batched_context_benchmark(), prompt_ids(), _prompt_ids_impl(), run_context_benchmark(), score_context_case(), build_effective_benchmark_spec(), effective_generation_budget() (+10 more)

### Community 43 - "load_local_native_model"
Cohesion: 0.14
Nodes (6): build_native_model(), load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), TestCheckpointTransfer, test_local_loader_preflights_and_preserves_shared_words()

### Community 44 - "evaluation/context.py"
Cohesion: 0.09
Nodes (19): _context_already_includes_instruction(), decoder_prompt(), _deduplicate_longcodeqa_headers(), encoder_context(), format_crosscodelong_summary(), greedy_decode(), _has_disjoint_structured_context(), _merge_full_function_into_file() (+11 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.08
Nodes (9): FixtureGenerationModel, FixtureTokenizer, test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_context_benchmark_mcq_budget_floor_execution(), test_fast_notebook_is_clean_without_embedded_b85(), test_matrix_notebooks_strict_line_count_and_ast(), test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues() (+1 more)

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.09
Nodes (4): ModelOutput, BifocalGemmaForCausalLM, test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_pretrained_model_subfolder_support_and_repo_creation()

### Community 48 - "sys"
Cohesion: 0.17
Nodes (7): test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(), format_code(), matrix_bundle(), main(), render(), source_bundle(), source_bundle_bytes()

### Community 50 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 51 - "repository_study.py"
Cohesion: 0.11
Nodes (32): get_experiment(), canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), normalized_code(), prepare_crosscode_tasks() (+24 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.08
Nodes (9): test_notebook_stress_helpers_match_shared_library(), diff(), ExecutingScoringOptions, fixture(), ScoringOptions, ScriptedModel, test_context_budget_and_hidden_tests_are_not_rendered(), test_greedy_encoder_once_stop_mode_and_vocab_guards() (+1 more)

### Community 53 - "run_matrix_stage"
Cohesion: 0.07
Nodes (21): run_matrix_stage(), backward_window(), build_bifocalgemma_optimizer(), compute_example_loss_weights(), compute_step_loss(), update_policy_learning_rates(), build_bifocalgemma_adamw(), clip_optimizer_group_grads() (+13 more)

### Community 54 - "atomic_json"
Cohesion: 0.14
Nodes (6): activity(), activity(), retry_delay(), atomic_json(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.10
Nodes (5): test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "fit_offline_sft"
Cohesion: 0.44
Nodes (5): fit_offline_sft(), emit(), model_stage(), append_event(), DurableClock

### Community 58 - "BifocalGemmaConfig"
Cohesion: 0.09
Nodes (14): BifocalGemmaConfig, _resolve_backbone_weights(), ProbeLoRALinear, test_from_pretrained_raises_file_not_found_when_weights_missing(), test_model_to_bfloat16_preserves_fp32_master_weights(), test_bifocalgemma_bin_serialization_and_dtype(), test_bifocalgemma_config_initialization_and_serialization(), test_bifocalgemma_for_causal_lm_forward_loss_and_generation() (+6 more)

### Community 59 - "objectives.py"
Cohesion: 0.09
Nodes (16): chunked_differential_cross_entropy(), chunked_linear_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), differential_diff_cross_entropy(), is_unified_diff_target_tokens(), _normalized_token_coefficients() (+8 more)

### Community 60 - "hashlib"
Cohesion: 0.15
Nodes (14): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), normalize_context_task(), curated_pool_digest(), editing_record(), git_text_lines() (+6 more)

### Community 61 - "JSONLRepairDataset"
Cohesion: 0.16
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 62 - "reset_local_benchmarks"
Cohesion: 0.33
Nodes (3): reset_local_benchmarks(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training()

### Community 63 - "TestMemoryProbe"
Cohesion: 0.27
Nodes (6): configure_probe_training(), gpu_memory(), optimizer_bytes(), profile_training_group(), tensor_bytes(), TestMemoryProbe

### Community 64 - "math"
Cohesion: 0.11
Nodes (11): _build_group_schedule(), build_training_groups(), default_training_profiles(), Float32Alignment, training_policy_table(), validate_training_policy(), comparison_recipes(), resolve_training_cadence() (+3 more)

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.14
Nodes (5): _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime(), RecordingStream

### Community 67 - "generation.py"
Cohesion: 0.17
Nodes (9): generate_diagnostic_text(), generate_patch(), greedy_decode(), GreedyGeneration, repair_prompt_ids(), _safe_apply_chat_template(), cached_logits(), test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold() (+1 more)

### Community 68 - "test_context_tasks.py"
Cohesion: 0.10
Nodes (8): task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose(), test_tokenize_context_task_preserves_target_indentation_and_newlines()

### Community 69 - "run_length_grid"
Cohesion: 0.40
Nodes (4): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_sweep_persists_results_per_case()

### Community 72 - "repository_corpus.py"
Cohesion: 0.08
Nodes (20): ArchiveSnapshotStore, build_repository_corpus(), context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), recognize_license() (+12 more)

### Community 74 - "RepositorySnapshot"
Cohesion: 0.09
Nodes (21): RepairExample, RepositoryFile, RepositorySnapshot, _build_canonical_patch(), example_from_record(), _git_apply(), normalize_swesmith_record(), NormalizedSWESmithRecord (+13 more)

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "training/offline_sft.py"
Cohesion: 0.18
Nodes (9): encoder_diagnostics(), stratified_subset(), cpu_tree(), evaluate_sft_loss(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), validation_batches() (+1 more)

### Community 79 - "test_editing_manifest.py"
Cohesion: 0.28
Nodes (8): editing_manifest(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation(), test_repository_eval_is_rejected()

### Community 80 - "stage.py"
Cohesion: 0.16
Nodes (6): is_zero_weight_decay_bridge_parameter(), v4_group_defaults(), v4_optimizer_defaults(), build_partitioned_param_groups(), classify_bifocalgemma_parameter(), _resolve_group_settings()

### Community 83 - "releases.py"
Cohesion: 0.15
Nodes (15): archive_files(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows(), unpack_archive() (+7 more)

### Community 85 - "modeling_bifocalgemma.py"
Cohesion: 0.10
Nodes (4): joint_visibility_tile(), TiledEncoderAttention, TiledMergedAttention, sample_encoder_mass()

### Community 87 - "test_upload_recovery_receipts.py"
Cohesion: 0.09
Nodes (5): retained_checkpoints(), test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), download(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only(), test_standalone_export_requires_matching_remote_contents()

### Community 91 - "test_data_releases.py"
Cohesion: 0.16
Nodes (10): pool(), release(), test_eval_public_and_gold_separate(), test_existing_local_pool_cannot_bypass_pinned_identity(), test_fetch_pinned_pool_roundtrip_and_reuse(), test_moving_revision_rejected_without_network(), test_publish_refuses_public_repo_before_upload(), test_release_original_bytes_and_view_schema() (+2 more)

### Community 92 - "run_matrix_training"
Cohesion: 0.08
Nodes (25): BenchmarkUploadQueue, benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), _backfill_final_stage(), _build_schedule_for_epochs(), _final_stage_info() (+17 more)

### Community 93 - "RepairCollator"
Cohesion: 0.12
Nodes (6): Seq2SeqBatch, RepairCollator, __getattr__(), __getattr__(), patch_cross_entropy(), TestDataAndLoss

### Community 95 - "remediation.py"
Cohesion: 0.10
Nodes (26): export_pool(), split_connected_records(), audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage() (+18 more)

### Community 96 - "normalize_context_task"
Cohesion: 0.22
Nodes (9): generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), parent(), test_evaluation_and_unlicensed_sources_rejected(), test_grounded_generation_answerability_provenance_and_target_independence() (+1 more)

### Community 102 - "tests_e2e/conftest.py"
Cohesion: 0.09
Nodes (10): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row(), validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation() (+2 more)

### Community 103 - "test_v4_architecture.py"
Cohesion: 0.11
Nodes (9): critical_attention_beta(), test_v3_critical_attention_beta_and_triton_tile_clamping(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_decoupled_full_attn_cross_kv_fp32_master_weights_and_gradients(), test_decoupled_upper_full_attention_cross_kv_topology(), test_sdpa_attention_structured_masks_match_tiled_eager() (+1 more)

### Community 104 - "sdpa_cross_attention"
Cohesion: 0.16
Nodes (6): sdpa_attention(), sdpa_cross_attention(), _sdpa_supported(), spy_sdpa(), test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa(), test_sdpa_attention_safe_fallback_on_float64_and_custom_float_mask()

### Community 106 - "FullCrossCachedAttention"
Cohesion: 0.20
Nodes (5): FullCrossCachedAttention, InferenceCache, test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence(), test_full_cross_cached_attention_sdpa_guard_fallback_and_non_left_padded_step_valid(), _run_cached()

### Community 108 - "notebook_launcher.py"
Cohesion: 0.28
Nodes (6): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.12
Nodes (7): clamped_triton_kernel_options(), ensure_hardware_memory_guards(), format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.15
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 114 - "__main__.py"
Cohesion: 0.16
Nodes (7): ExperimentSpec, get_v2_experiment(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 115 - "packed_tokens.py"
Cohesion: 0.14
Nodes (6): build_packed_cache(), open_packed_cache(), PackedTokenizedDataset, compare_predictions(), test_packed_cache_can_be_built_without_tensor_runtime(), test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes()

### Community 116 - "test_training_recipes.py"
Cohesion: 0.16
Nodes (6): test_paired_bootstrap_and_cohort_rejection(), limit_threads(), test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_fp32_alignment_on_bf16_backbone(), test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 120 - "WigglyDashboard"
Cohesion: 0.27
Nodes (7): LastSeenWidget, WigglyDashboard, model_opens(), test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(), widget_runtime()

### Community 121 - "GrokOrScaleLoRA"
Cohesion: 0.24
Nodes (5): build_policy_optimizer(), GrokOrScaleLoRA, orthogonal_direction(), validate_optimizer_config(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 122 - "stable_fingerprint"
Cohesion: 0.12
Nodes (18): CheckpointRef, fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), prepare_editing_records() (+10 more)

### Community 123 - "TokenizedRepairExample"
Cohesion: 0.12
Nodes (10): TokenizedRepairExample, validate_curated_pool_limits(), run_training_memory_preflight(), fit_curated_sft(), synthetic_stress_target(), test_complete_epoch_training_budget_all_families(), test_curated_resume_rejects_runtime_drift(), test_frozen_budget_matches_all_six_controls() (+2 more)

### Community 124 - "os"
Cohesion: 0.19
Nodes (5): NativeTaskResult, score_python_patch(), summarize_native_results(), test_static_helpers_and_official_tokenizer_class_do_not_import_torch(), test_native_score_pass_fail_disabled_and_runtime()

### Community 130 - "test_tier1_features.py"
Cohesion: 0.11
Nodes (12): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f2_mcq_token_budget_specification(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval() (+4 more)

### Community 134 - "tests_e2e/conftest.py"
Cohesion: 0.13
Nodes (9): completion_metrics(), _sequence_and_identifier_metrics(), reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement() (+1 more)

### Community 137 - "batched_greedy_decode"
Cohesion: 0.25
Nodes (5): batched_greedy_decode(), infer_benchmark_weight_tier(), resolve_optimal_eval_batch_size(), test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight(), test_v4_config_constants_and_external_registry()

### Community 146 - "test_evaluation.py"
Cohesion: 0.10
Nodes (9): _extract_full_python_code(), normalize_completion_prediction(), _strip_echoed_suffix(), _strip_fim_markers(), _strip_markdown_fences(), test_prompt_ids_instance_id_isolation_and_crosscodelong_normalization(), test_crosscodelong_normalize_completion_prediction_strips_fences_and_suffix_overlap(), test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity() (+1 more)

### Community 171 - "resolve_diff_token_ids"
Cohesion: 0.25
Nodes (3): extract_trivial_token_ids(), resolve_diff_token_ids(), test_pointer_loss_gradient_checkpointing_and_trivial_filtering()

### Community 172 - "test_effective_benchmark_protocol.py"
Cohesion: 0.10
Nodes (10): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer, test_effective_allowance(), test_effective_spec_binds_gold_caps_and_runtime_spec(), test_invalid_allowance_fails_explicitly(), test_prepare_sampled_benchmark_collection_invalidates_stale_cache() (+2 more)

### Community 173 - "test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention"
Cohesion: 0.29
Nodes (3): test_synchronize_matrix_retains_stage1_milestone_and_retries_412_contention(), create_commit(), make_412()

### Community 175 - "NativeMergedAttention"
Cohesion: 0.17
Nodes (4): _check_transformers_version(), native_config(), NativeMergedAttention, TestMergedAttentionReference

### Community 180 - "FullCrossCachedAttention"
Cohesion: 0.11
Nodes (9): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossTiledAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), _resolve_gate_init_tanh(), test_bfloat16_forward_backward_gradient_flow_and_optimizer_step() (+1 more)

### Community 185 - "config.py"
Cohesion: 0.47
Nodes (3): BackboneProfile, _pair(), _ref()

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 999 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **95 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `test_generate_multi_eos_per_row_stopping_and_cache_parity`, `.configure_stage1`, `test_v4_architecture.py`, `sdpa_cross_attention`, `fast_pipeline.py`, `FullCrossCachedAttention`, `FastTrainingPipeline`, `bifocalgemma/__init__.py`, `FullCrossCachedAttention`, `modeling_bifocalgemma.py`, `run_matrix_stage`, `export_matrix_pretrained_model`, `BifocalGemmaConfig`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `contracts.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13157894736842105 - nodes in this community are weakly interconnected._
- **Why does `TokenizedRepairExample` connect `TokenizedRepairExample` to `preflight.py`, `contracts.py`, `test_memory_efficient_training.py`, `schedule.py`, `make_experiment_spec`, `context_tasks.py`, `curriculum.py`, `examples`, `test_molab_matrix_notebooks.py`, `run_matrix_stage`, `TestMemoryProbe`, `RepositorySnapshot`, `test_editing_manifest.py`, `JSONLRepairDataset`, `modeling_bifocalgemma.py`, `RepairCollator`, `fast_pipeline.py`, `__main__.py`, `packed_tokens.py`, `stable_fingerprint`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._
- **Why does `run_matrix_training()` connect `run_matrix_training` to `math`, `synchronize_matrix`, `stages.py`, `test_molab_matrix_notebooks.py`, `resolve_diff_token_ids`, `test_molab_remediation.py`, `patch_matrix_tokenization`, `schedule.py`, `json`, `releases.py`, `run_matrix_stage`, `atomic_json`, `test_matrix_schedule_runtime.py`, `make_experiment_spec`, `stable_fingerprint`, `source_fingerprints`, `test_matrix_quarantine_legacy_checkpoint_vs_clean_resume`, `export_matrix_pretrained_model`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._