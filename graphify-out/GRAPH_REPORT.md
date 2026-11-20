# Graph Report - bifocalgemma  (2026-11-20)

## Corpus Check
- 255 files · ~358,783 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .tex 11, (none) 2, .jsonl 2)

## Summary
- 2581 nodes · 7053 edges · 184 communities (93 shown, 91 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 845 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `310c3617`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- preflight.py
- render_dashboard
- RepositorySnapshot
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
- unified_recipe
- context_tasks.py
- patch_cross_entropy
- adversarial_r3_probe.py
- torch
- test_v4_data_disjoint.py
- curriculum.py
- stable_fingerprint
- external_runner.py
- examples
- contracts.py
- prepare_context_data.py
- synchronize_matrix
- test_gsm8k.py
- test_molab_matrix_notebooks.py
- SwiGLUDualGateBridge
- protocol.py
- load_local_native_model
- evaluation/context.py
- test_molab_remediation.py
- BifocalGemmaForCausalLM
- sys
- bifocalgemma/__init__.py
- test_tier4_scenarios.py
- repository_study.py
- test_repository_study.py
- make_experiment_spec
- atomic_json
- test_matrix_schedule_runtime.py
- benchmark.py
- test_shared_schedule.py
- objectives.py
- score_context_case
- JSONLRepairDataset
- remote_reset_plan
- TestMemoryProbe
- math
- test_training_telemetry.py
- TestBifocalGemmaArchitecture
- test_context_tasks.py
- test_pretrained_probe_helpers.py
- repository_corpus.py
- hashlib
- test_construction_blocks.py
- training/offline_sft.py
- test_editing_manifest.py
- test_dashboard_created_before_context_registers_all_displayed_models
- wiggly_dashboard.py
- json
- modeling_bifocalgemma.py
- test_upload_recovery_receipts.py
- test_offline_sft_training.py
- bifocalgemma
- test_benchmark_prompt_budget.py
- run_matrix_training
- TokenizedRepairExample
- remediation.py
- normalize_context_task
- .configure_stage1
- tests_e2e/conftest.py
- test_v4_architecture.py
- run_matrix_stage
- fast_pipeline.py
- test_tier3_combinations.py
- notebook_launcher.py
- .snapshot
- patch_matrix_tokenization
- test_fast_pipeline.py
- __main__.py
- PackedTokenizedDataset
- test_training_recipes.py
- test_context_benchmark_mcq_budget_floor_execution
- WigglyDashboard
- GrokOrScaleLoRA
- repository_sft.py
- fit_curated_sft
- os
- test_tier1_features.py
- tests_e2e/conftest.py
- build_repository_corpus
- _ProportionalTokenizer
- test_evaluation.py
- patch_cross_entropy
- test_effective_benchmark_protocol.py
- _CharTokenizer
- NativeMergedAttention
- dependency_state
- FullCrossCachedAttention

## God Nodes (most connected - your core abstractions)
1. `stable_fingerprint()` - 67 edges
2. `atomic_json()` - 56 edges
3. `BifocalGemmaForCausalLM` - 50 edges
4. `load_verified_tokenizer()` - 50 edges
5. `run_matrix_training()` - 50 edges
6. `RepositorySnapshot` - 45 edges
7. `TokenizedRepairExample` - 40 edges
8. `run_matrix_benchmark()` - 40 edges
9. `FastTrainingPipeline` - 38 edges
10. `fit_offline_sft()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `test_repair_record_type_has_same_prefix_as_explicit_repair_task()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `generate()` --calls--> `decoder_prompt()`  [INFERRED]
  tests/test_benchmark_prompt_budget.py → src/bifocalgemma/evaluation/context.py
- `fetch()` --calls--> `repository_task_from_row()`  [INFERRED]
  tools/prepare_context_data.py → src/bifocalgemma/data/repository.py
- `decode()` --calls--> `GreedyGeneration`  [INFERRED]
  tests/test_effective_benchmark_protocol.py → src/bifocalgemma/inference/generation.py
- `test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads()` --calls--> `SwiGLUDualGateBridge`  [INFERRED]
  tests/test_molab_matrix_notebooks.py → src/bifocalgemma/models/bridge.py

## Import Cycles
- 2-file cycle: `src/bifocalgemma/evaluation/context.py -> src/bifocalgemma/evaluation/protocol.py -> src/bifocalgemma/evaluation/context.py`

## Communities (184 total, 91 thin omitted)

### Community 0 - "preflight.py"
Cohesion: 0.19
Nodes (10): build_policy_optimizer(), current_capacity(), execution_settings(), _gpu_context(), maximum_probe(), _measure_phase(), seal_report(), select_execution() (+2 more)

### Community 1 - "render_dashboard"
Cohesion: 0.18
Nodes (6): format_duration(), format_stage_display(), loss_chart(), render_dashboard(), test_monitor_escapes_errors_updates_validation_and_exports_html(), test_dashboard_stage_display_formatting()

### Community 2 - "RepositorySnapshot"
Cohesion: 0.12
Nodes (15): RepositoryFile, RepositorySnapshot, canonical_snapshot_hash(), ContextWindow, render_window(), select_repository_context(), _terms(), example_from_record() (+7 more)

### Community 3 - "pathlib"
Cohesion: 0.15
Nodes (6): retry_delay(), append_event(), test_external_activity_reader_does_not_need_running_kernel(), test_retry_after_date_and_network_failures(), inspect(), main()

### Community 5 - "build_native_tiny_model"
Cohesion: 0.16
Nodes (5): build_native_tiny_model(), TestNativeIntegration, TestCheckpointTransfer, test_last_token_logits_match_full_head(), test_local_loader_preflights_and_preserves_shared_words()

### Community 6 - "re"
Cohesion: 0.13
Nodes (21): _add_missing_git_headers(), apply_patch_to_snapshot(), apply_patch_with_recount(), _apply_validated_patch(), _build_canonical_patch(), diagnose_patch_application(), extract_unified_diff(), _git_text_lines() (+13 more)

### Community 7 - "stages.py"
Cohesion: 0.14
Nodes (16): file_digest(), build_packed_cache(), open_packed_cache(), synchronize_matrix(), checked_prefix(), recover_remote_run(), _copy_atomic(), _synchronize_once() (+8 more)

### Community 8 - "test_notebook_dependencies.py"
Cohesion: 0.11
Nodes (14): simulated_environment(), test_conda_base_is_rejected_before_any_install(), test_cpu_wheel_invalidates_matching_receipt(), test_failed_install_cannot_write_success_receipt(), test_force_reinstall_targets_kernel_and_receipt_skips_next_install(), test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(), test_native_import_diagnostic_preserves_original_dependency_failure(), test_native_import_loads_direct_modules_and_expected_symbols() (+6 more)

### Community 10 - "test_memory_efficient_training.py"
Cohesion: 0.25
Nodes (13): configure_memory_efficient_training(), bounded_threads(), examples(), make_model(), test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(), test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(), test_failure_records_stage_lengths_ids_and_no_completion(), test_full_stack_loss_logits_and_adapter_gradients_match_eager() (+5 more)

### Community 11 - "FastTrainingPipeline"
Cohesion: 0.16
Nodes (11): FastTrainingConfig, FastTrainingPipeline, run_fast_training(), test_warmup_cosine_learning_rate_schedule(), test_fast_training_config_and_pipeline_baseline_single_stage(), test_fast_pipeline_batch_edge_cases(), test_fast_pipeline_skips_training_when_already_completed(), test_fast_training_config_step_reconciliation() (+3 more)

### Community 13 - "test_matrix_runtime_loader.py"
Cohesion: 0.21
Nodes (17): archive_bytes(), loader(), no_network(), settings(), test_bad_download_never_activates_code(), test_corrupt_local_archive_uses_verified_cache(), test_invalid_generation_modes_cannot_publish(), test_local_preview_missing_archive_explains_publication_without_network() (+9 more)

### Community 14 - "curation.py"
Cohesion: 0.08
Nodes (32): code_shingles(), code_signature(), code_tokens(), CodeOverlapIndex, CodeOverlapMatch, curate_paired_sources(), nextcoder_row(), validate_curated_pool_limits() (+24 more)

### Community 15 - "schedule.py"
Cohesion: 0.13
Nodes (19): _build_schedule_for_epochs(), _schedule_signature_matches(), _apportion_strata_counts(), build_shared_schedule(), canonical_schedule_records(), _digest(), _integer(), _json() (+11 more)

### Community 16 - "test_context_benchmark_resume.py"
Cohesion: 0.15
Nodes (9): limited(), test_parent_conflict_refreshes_but_keeps_retry_bounded(), always_conflict(), sync(), FixtureGenerationModel, FixtureTokenizer, test_gold_in_public_row_is_rejected_before_tokenization(), test_prediction_journal_resume_and_checkpoint_mismatch() (+1 more)

### Community 17 - "test_dashboard_dependency_state.py"
Cohesion: 0.18
Nodes (15): test_changed_ui_constraint_profile_cannot_bypass_pending_restart(), test_changed_ui_install_becomes_ready_after_restart_without_reinstall(), test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(), test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(), test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(), test_install_with_preloaded_pinned_ui_module_requires_restart(), test_matching_loaded_and_installed_ui_version_remains_ready(), test_matrix_launcher_preserves_reported_molab_host_widget_versions() (+7 more)

### Community 18 - "json"
Cohesion: 0.08
Nodes (10): compare_predictions(), test_paired_bootstrap_and_cohort_rejection(), audit(), emit(), main(), main(), main(), distribution() (+2 more)

### Community 19 - "test_v4_eval_parity.py"
Cohesion: 0.10
Nodes (5): MockCausalLM, MockQwenTokenizer, test_external_runner_qwen_bos_none_and_eos_handling(), test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing(), test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic()

### Community 20 - "eval_code.py"
Cohesion: 0.14
Nodes (7): BaselineScores, _format_score(), print_comparison_table(), print_repair_comparison_table(), print_repair_experiment_table(), print_v4_parity_comparison_table(), TestEvaluationReporting

### Community 22 - "test_tier5_adversarial.py"
Cohesion: 0.06
Nodes (25): extract_mcq_letter(), strict_letter(), probe_mcq_delimiters_and_wrappers(), probe_mcq_preambles_and_ambiguity(), test_adversarial_context_containing_forbidden_phrases(), test_adversarial_mcq_adverb_preambles(), test_adversarial_mcq_ambiguity_guards(), test_adversarial_mcq_delimited_negation_and_conjunction_guards() (+17 more)

### Community 23 - "hashlib"
Cohesion: 0.15
Nodes (16): _clean_extracted_number(), digest(), extract_gsm8k_answer(), extract_release_answer(), extract_resilient_answer(), fetch_gsm8k_sources(), fingerprint(), generate_gsm8k() (+8 more)

### Community 24 - "unified_recipe"
Cohesion: 0.12
Nodes (9): unified_recipe(), v4_optimizer_defaults(), build_bifocalgemma_adamw(), clip_optimizer_group_grads(), test_matrix_baseline_single_stage_resolution(), test_consolidated_optimizer_builder_parity_and_decoder_only_apples_to_apples(), test_matrix_stage_microbatch_invariance_under_heterogeneous_lengths(), test_per_group_grad_clipping_isolates_gate_scalars_and_gate_proj_from_bridge() (+1 more)

### Community 25 - "context_tasks.py"
Cohesion: 0.18
Nodes (16): _anchor_index_for_repair(), DisjointContextSplit, _extract_numbered_entries(), format_numbered_code(), _is_numbered_block(), _parse_file_sections(), _render_entries(), _resolve_eot_token_id() (+8 more)

### Community 26 - "patch_cross_entropy"
Cohesion: 0.18
Nodes (4): TestOfflineSFT, TinyLossModel, test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(), test_training_resume_keeps_elapsed_from_step_five_through_ten()

### Community 27 - "adversarial_r3_probe.py"
Cohesion: 0.20
Nodes (4): probe_prompt_neutrality_scan(), probe_redos_catastrophic_backtracking(), probe_reponiah_negation_coverage(), probe_unicode_and_metrics()

### Community 28 - "torch"
Cohesion: 0.06
Nodes (20): flex_kernel_options(), fused_attention(), cached_logits(), attention_tile(), build_joint_visibility(), joint_visibility_tile(), mask_tile(), sdpa_attention() (+12 more)

### Community 30 - "test_v4_data_disjoint.py"
Cohesion: 0.15
Nodes (10): validate_context_license_record(), test_context_license_record_covers_all_visible_paths(), _load_prepare_module(), _numbered_line_numbers(), test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code(), test_disjoint_split_crosscodelong_repair_and_qa_layouts(), test_format_numbered_code_and_exports(), test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool() (+2 more)

### Community 31 - "curriculum.py"
Cohesion: 0.20
Nodes (9): build_anti_forgetting_curriculum(), build_v4_curriculum(), classify_v4_bucket(), encode_comprehension_prefix(), _encoder_token_count(), is_comprehension_example(), synthesize_comprehension_examples(), test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length() (+1 more)

### Community 32 - "stable_fingerprint"
Cohesion: 0.11
Nodes (23): fetch_public(), matched_eval_cohort(), prepare_crosscodelong(), prepare_longcodeqa(), prepare_sweqa(), write_context_eval(), read_arrow_rows(), token_profile() (+15 more)

### Community 34 - "external_runner.py"
Cohesion: 0.11
Nodes (17): format_crosscodelong_summary(), _build_reponiah_cohort(), _collect_int_ids(), extract_generated_tokens(), format_external_prompt(), generate_external_batch(), _load_benchmark_cohort(), load_hf_external_model_and_tokenizer() (+9 more)

### Community 35 - "examples"
Cohesion: 0.27
Nodes (3): make_identity(), examples(), TestTrainingPolicy

### Community 36 - "contracts.py"
Cohesion: 0.05
Nodes (25): DeveloperAgent, EvaluationReport, RepairExample, RepairPrediction, CommitPackFTAdapter, DatasetAdapter, EditPackFTAdapter, SWEGymAdapter (+17 more)

### Community 37 - "prepare_context_data.py"
Cohesion: 0.10
Nodes (25): export_pool(), split_connected_records(), adapt_v4_row(), _all_code_fragment_signatures(), build_gsm8k_fingerprints(), code_fragment_signatures(), _collect_benchmark_quarantine_assets(), completion_from_snapshot() (+17 more)

### Community 38 - "synchronize_matrix"
Cohesion: 0.13
Nodes (11): load_matrix_adapters_for_inference(), prune_matrix_checkpoints(), restore_matrix_checkpoint(), restore_optimizer_state_by_name(), save_matrix_checkpoint(), configure_policy_model(), test_save_and_restore_matrix_checkpoint_and_prune(), test_stage1_to_stage2_optimizer_state_transfer() (+3 more)

### Community 39 - "test_gsm8k.py"
Cohesion: 0.15
Nodes (9): assets(), FakeTokenizer, fixture_rows(), test_bad_generation_budget_rejected_before_loading_weights(), test_cli_rejects_prediction_from_another_prompt_protocol(), test_release_extraction_and_fixed_denominator(), test_six_generation_paths_cache_assets_and_never_read_gold(), test_target_is_separate_and_shots_use_only_train() (+1 more)

### Community 40 - "test_molab_matrix_notebooks.py"
Cohesion: 0.10
Nodes (11): resolve_optimal_eval_batch_size(), TrainingDashboard, test_comprehension_targets_stay_inside_baseline_visible_prefix(), test_configure_full_rank_cross_attention_v3_specs(), test_embedded_bundle_can_be_extracted_and_contains_matrix_support(), test_make_experiment_spec_and_hub_prefix_for_all_ten_models(), test_notebook_exists_and_is_valid_ast(), test_notebook_has_correct_metadata_and_per_model_hf_repo() (+3 more)

### Community 41 - "SwiGLUDualGateBridge"
Cohesion: 0.12
Nodes (5): NormalizedAlignment, SwiGLUDualGateBridge, test_bridge_init_preserved_across_post_init(), test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight(), test_swiglu_bridge_non_redundant_single_post_gain()

### Community 42 - "protocol.py"
Cohesion: 0.15
Nodes (11): _extract_match_letter(), fallback_letter_regex(), build_effective_benchmark_spec(), _fingerprint(), gsm8k_report_metrics(), mcq_report_metrics(), _positive_integer(), _runtime_model_identity() (+3 more)

### Community 43 - "load_local_native_model"
Cohesion: 0.16
Nodes (5): build_native_model(), load_local_native_model(), load_sft_adapters_for_inference(), transfer_text_checkpoint(), test_inference_v2_adapter_roundtrip_and_baseline_group()

### Community 44 - "evaluation/context.py"
Cohesion: 0.07
Nodes (29): build_repo_niah_case(), _context_already_includes_instruction(), _count_niah_tokens(), decoder_prompt(), _deduplicate_longcodeqa_headers(), _encoded_context_tokens(), encoder_context(), _extract_full_python_code() (+21 more)

### Community 45 - "test_molab_remediation.py"
Cohesion: 0.11
Nodes (6): test_clean_notebooks_have_no_embedded_b85_and_valid_runtime_archive(), test_clean_recipe_version_and_constants(), test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(), test_fast_notebook_is_clean_without_embedded_b85(), test_fast_training_pipeline_features_and_export(), test_matrix_notebooks_strict_line_count_and_ast()

### Community 47 - "BifocalGemmaForCausalLM"
Cohesion: 0.05
Nodes (22): BifocalGemmaConfig, BifocalGemmaForCausalLM, export_matrix_pretrained_model(), _resolve_backbone_weights(), configure_probe_training(), ProbeLoRALinear, test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss(), test_from_pretrained_raises_file_not_found_when_weights_missing() (+14 more)

### Community 48 - "sys"
Cohesion: 0.24
Nodes (5): matrix_bundle(), main(), render(), source_bundle(), source_bundle_bytes()

### Community 50 - "test_tier4_scenarios.py"
Cohesion: 0.24
Nodes (4): validate_paper_manuscript(), test_scenario_end_to_end_mock_evaluation_pipeline(), test_scenario_scientific_paper_document_validation(), test_scenario_training_compatibility_preflight()

### Community 51 - "repository_study.py"
Cohesion: 0.24
Nodes (18): get_experiment(), normalized_code(), prepare_crosscode_tasks(), prepare_repoqa_tasks(), score_diagnostics(), fetch_tokenizer_assets(), generate_run(), main() (+10 more)

### Community 52 - "test_repository_study.py"
Cohesion: 0.12
Nodes (17): v4_repository_task_from_row(), prepare_repository_record(), repo_identity(), repository_task_from_row(), RepositoryTask, CharacterTokenizer, diff(), ExecutingScoringOptions (+9 more)

### Community 53 - "make_experiment_spec"
Cohesion: 0.10
Nodes (21): make_experiment_spec(), batched_greedy_decode(), configure_full_rank_cross_attention(), is_cross_bridge_parameter(), update_policy_learning_rates(), configure_matrix_stage(), fixture_data(), fixture_model() (+13 more)

### Community 54 - "atomic_json"
Cohesion: 0.12
Nodes (7): retry_delay(), atomic_json(), test_durable_backoff_shared_across_models_and_final_retries_without_sleep(), fail(), test_fatal_errors_not_hidden(), test_interval_throttles_periodic_upload_but_final_can_flush(), test_keyboard_interrupt_is_not_swallowed()

### Community 55 - "test_matrix_schedule_runtime.py"
Cohesion: 0.07
Nodes (8): ModelOutput, Seq2SeqBatch, NativeTinyModel, test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(), download(), local_export(), test_recovered_completed_training_and_benchmark_resume_skip_completed_work(), download()

### Community 56 - "benchmark.py"
Cohesion: 0.13
Nodes (14): infer_benchmark_weight_tier(), prepare_sampled_benchmark_collection(), run_batched_context_benchmark(), activity(), run_context_benchmark(), activity(), repair_report_metrics(), fit_offline_sft() (+6 more)

### Community 58 - "test_shared_schedule.py"
Cohesion: 0.28
Nodes (12): rows(), schedule(), test_all_originals_once_with_disjoint_stage_windows(), test_atomic_manifest_reuse_and_fail_closed_mismatch(), test_deterministic_model_independent_schedule_and_mapping(), test_each_semantic_mutation_changes_signature(), test_empty_required_fields_rejected(), test_foundation_context_and_target_patch_are_bound() (+4 more)

### Community 59 - "objectives.py"
Cohesion: 0.06
Nodes (18): chunked_differential_cross_entropy(), compute(), compute_diff_token_weights(), _compute_row_weights(), compute_step_loss(), differential_diff_cross_entropy(), extract_trivial_token_ids(), is_unified_diff_target_tokens() (+10 more)

### Community 60 - "score_context_case"
Cohesion: 0.16
Nodes (10): completion_metrics(), normalize_completion_prediction(), score_context_case(), score_repo_niah_case(), _sequence_and_identifier_metrics(), _strip_echoed_suffix(), _strip_fim_markers(), verify_gate_round_2_resolved_cases() (+2 more)

### Community 61 - "JSONLRepairDataset"
Cohesion: 0.15
Nodes (3): collate_fast_batch(), JSONLRepairDataset, SyntheticRepairDataset

### Community 62 - "remote_reset_plan"
Cohesion: 0.13
Nodes (7): remote_reset_plan(), reset_local_benchmarks(), reset_state(), test_remote_reset_is_scoped_and_receipt_hashes_match(), test_reset_archives_benchmark_underscore_keys_too(), test_reset_archives_evaluation_once_without_changing_training(), main()

### Community 64 - "math"
Cohesion: 0.10
Nodes (13): is_zero_weight_decay_bridge_parameter(), validate_optimizer_config(), _build_group_schedule(), build_training_groups(), Float32Alignment, training_policy_table(), validate_training_policy(), resolve_training_cadence() (+5 more)

### Community 66 - "test_training_telemetry.py"
Cohesion: 0.16
Nodes (4): _SilentStream, test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(), test_widgets_update_in_place_and_restore_history_without_duplicate_points(), _widget_runtime()

### Community 68 - "test_context_tasks.py"
Cohesion: 0.09
Nodes (9): repair_preimage_coverage(), task(), test_eval_role_and_invisible_localization_target_rejected(), test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(), test_prepare_sweqa_instance_id_ignores_correct_answer(), test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline(), test_repair_requires_visible_preimage_without_refilling_context(), test_short_answer_scoring_rejects_unrequested_prose() (+1 more)

### Community 69 - "test_pretrained_probe_helpers.py"
Cohesion: 0.20
Nodes (5): run_length_grid(), test_length_grid_matches_baseline_budget_and_stops_after_failure(), probe(), test_notebook_stress_helpers_match_shared_library(), test_notebook_sweep_persists_results_per_case()

### Community 72 - "repository_corpus.py"
Cohesion: 0.15
Nodes (12): context_license_evidence(), patch_license_evidence(), path_license_evidence(), prepare_repository_corpus(), read_text_archive(), archive(), test_archive_commit_budget_and_static_license(), test_archive_download_retries_incomplete_transport() (+4 more)

### Community 74 - "hashlib"
Cohesion: 0.11
Nodes (17): map_codereview_to_bifocal(), map_instructcoder_to_bifocal(), map_nextcoder_to_bifocal(), sanitize_posix_path(), editing_record(), git_text_lines(), repository_editing_record(), recognize_license() (+9 more)

### Community 76 - "test_construction_blocks.py"
Cohesion: 0.22
Nodes (3): apply_word_embedding_policy(), EmbeddingPolicyResult, TestWordEmbeddingPolicy

### Community 77 - "training/offline_sft.py"
Cohesion: 0.22
Nodes (9): encoder_diagnostics(), stratified_subset(), cpu_tree(), evaluate_sft_loss(), file_digest(), restore_sft_checkpoint(), save_sft_checkpoint(), validation_batches() (+1 more)

### Community 79 - "test_editing_manifest.py"
Cohesion: 0.28
Nodes (8): editing_manifest(), repository_row(), test_conflicting_repository_hashes_are_rejected(), test_generated_preparation_consumes_frozen_raw_repository_pool(), test_legacy_paired_manifest_remains_byte_identical(), test_raw_and_normalized_repository_manifests_are_identical(), test_raw_repository_manifest_derives_hashes_without_mutation(), test_repository_eval_is_rejected()

### Community 80 - "test_dashboard_created_before_context_registers_all_displayed_models"
Cohesion: 0.32
Nodes (5): model_opens(), RecordingStream, test_dashboard_created_before_context_registers_all_displayed_models(), test_layout_and_wrappers_remain_stable_after_gc_and_updates(), widget_runtime()

### Community 82 - "wiggly_dashboard.py"
Cohesion: 0.20
Nodes (3): LastSeenWidget, phase_category(), TrainingLossLog

### Community 83 - "json"
Cohesion: 0.08
Nodes (32): archive_files(), audit_split_disjointness(), build_release(), digest(), fetch_collection(), main(), publish_release(), rows() (+24 more)

### Community 85 - "modeling_bifocalgemma.py"
Cohesion: 0.09
Nodes (8): BackboneProfile, CheckpointRef, _pair(), _ref(), CachedAttention, InferenceCache, clamped_triton_kernel_options(), test_v3_critical_attention_beta_and_triton_tile_clamping()

### Community 87 - "test_upload_recovery_receipts.py"
Cohesion: 0.09
Nodes (6): chunked_linear_cross_entropy(), checkpoint(), test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(), download(), test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only(), test_standalone_export_requires_matching_remote_contents()

### Community 91 - "test_benchmark_prompt_budget.py"
Cohesion: 0.20
Nodes (4): test_repair_record_type_has_same_prefix_as_explicit_repair_task(), test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(), generate(), TextTokenizer

### Community 92 - "run_matrix_training"
Cohesion: 0.05
Nodes (29): BenchmarkUploadQueue, benchmark_directory_name(), download_model(), RemoteHeartbeat, target_fingerprint(), _backfill_final_stage(), _final_stage_info(), _latest_checkpoint_matches() (+21 more)

### Community 93 - "TokenizedRepairExample"
Cohesion: 0.18
Nodes (6): TokenizedRepairExample, RepairCollator, __getattr__(), synthetic_stress_target(), TestDataAndLoss, test_exact_synthetic_lengths_preserve_shift_and_closing_turn()

### Community 95 - "remediation.py"
Cohesion: 0.12
Nodes (23): audit_rows(), build_remediated_pool(), event_key(), inspect_structure(), main(), merge_lineage(), preference(), refresh_patch_targets() (+15 more)

### Community 96 - "normalize_context_task"
Cohesion: 0.19
Nodes (11): is_permissive_license(), normalize_context_task(), generate_grounded_pool_tasks(), generate_grounded_tasks(), _render(), _sha(), validate_grounded_task(), parent() (+3 more)

### Community 101 - ".configure_stage1"
Cohesion: 0.19
Nodes (3): build_bifocalgemma_optimizer(), test_optimizer_momentum_preserved_stage1_to_stage2(), test_stage1_and_stage2_lora_configuration()

### Community 102 - "tests_e2e/conftest.py"
Cohesion: 0.15
Nodes (6): authoritative_empirical_results(), project_root(), pytest_configure(), sample_crosscode_row(), sample_mcq_row(), sample_repair_row()

### Community 103 - "test_v4_architecture.py"
Cohesion: 0.08
Nodes (14): critical_attention_beta(), test_compute_per_row_cross_query_has_no_host_sync_calls(), test_compute_per_row_cross_query_on_cuda_stays_on_device(), test_compute_per_row_cross_query_vectorized_parity(), test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence(), test_decoupled_upper_full_attention_cross_kv_topology(), test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(), spy_sdpa() (+6 more)

### Community 104 - "run_matrix_stage"
Cohesion: 0.29
Nodes (4): run_matrix_stage(), backward_window(), compute_example_loss_weights(), SharedSchedule

### Community 105 - "fast_pipeline.py"
Cohesion: 0.09
Nodes (8): ensure_assets_on_disk(), load_asset_json(), _check_transformers_version(), native_config(), gpu_memory(), optimizer_bytes(), profile_training_group(), tensor_bytes()

### Community 106 - "test_tier3_combinations.py"
Cohesion: 0.19
Nodes (4): validate_empirical_results_schema(), test_combination_crosscode_continuous_metrics_aggregation(), test_combination_eval_metrics_to_paper_data_serialization(), test_combination_mcq_scoring_across_benchmarks()

### Community 108 - "notebook_launcher.py"
Cohesion: 0.35
Nodes (6): test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(), _prepare(), _expected_bifocalgemma_path(), _normalized_bytes(), prepare_runtime(), _validate_loaded_modules()

### Community 110 - "patch_matrix_tokenization"
Cohesion: 0.17
Nodes (5): format_query_injected_context(), patch_matrix_tokenization(), safe_load_verified_tokenizer(), safe_tokenize_context_task(), test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract()

### Community 113 - "test_fast_pipeline.py"
Cohesion: 0.15
Nodes (5): DummyTrackingTokenizer, test_anti_forgetting_curriculum_and_stage1_pointer_loss(), test_checkpoint_save_and_resume_lora_scaling(), test_fast_training_pipeline_end_to_end_smoke(), test_jsonl_dataset_query_injection_into_encoder()

### Community 114 - "__main__.py"
Cohesion: 0.16
Nodes (7): ExperimentSpec, get_v2_experiment(), validate_experiment_matrix(), _initialization_plan(), main(), run_native_smoke(), TestConstructionEntrypoints

### Community 116 - "test_training_recipes.py"
Cohesion: 0.17
Nodes (5): comparison_recipes(), limit_threads(), test_fp32_alignment_on_bf16_backbone(), test_optimized_native_resume_matches_full_and_rejects_different_recipe(), model()

### Community 118 - "test_context_benchmark_mcq_budget_floor_execution"
Cohesion: 0.18
Nodes (3): FixtureGenerationModel, FixtureTokenizer, test_context_benchmark_mcq_budget_floor_execution()

### Community 121 - "GrokOrScaleLoRA"
Cohesion: 0.29
Nodes (3): GrokOrScaleLoRA, orthogonal_direction(), test_shared_numbers_and_orthogonal_zero_factor_bootstrap()

### Community 122 - "repository_sft.py"
Cohesion: 0.16
Nodes (13): curated_pool_digest(), load_curated_pool(), prepare_editing_records(), split_editing_records(), load_verified_tokenizer(), prepare_curated_training(), tokenize(), default_training_profiles() (+5 more)

### Community 123 - "fit_curated_sft"
Cohesion: 0.29
Nodes (3): run_training_memory_preflight(), fit_curated_sft(), test_curated_resume_rejects_runtime_drift()

### Community 130 - "test_tier1_features.py"
Cohesion: 0.12
Nodes (11): test_f10_methodological_autopsy_data_integrity(), test_f1_decoder_prompt_neutral_template(), test_f1_task_prompt_neutral_template(), test_f2_mcq_resilient_regex_parser(), test_f3_crosscodelong_continuous_metrics(), test_f4_reponiah_distractor_injection_and_retrieval(), test_f5_training_semantics_change_rejects_legacy_approval(), test_f6_repo_core_syntax_tests_executable() (+3 more)

### Community 134 - "tests_e2e/conftest.py"
Cohesion: 0.14
Nodes (7): reference_resilient_mcq_parser(), test_boundary_empty_and_whitespace_completions(), test_boundary_letter_extraction_alphabet(), test_boundary_multiline_conversational_preamble(), test_boundary_regex_meta_characters_in_generation(), test_boundary_token_budget_bounds_enforcement(), test_boundary_zero_identifier_f1_safeguard()

### Community 137 - "build_repository_corpus"
Cohesion: 0.16
Nodes (6): ArchiveSnapshotStore, build_repository_corpus(), RepositoryPreparationLedger, test_failed_download_removes_partial_bytes(), test_preparation_ledger_preserves_records_and_rejects_new_settings(), source_notices()

### Community 172 - "test_effective_benchmark_protocol.py"
Cohesion: 0.11
Nodes (10): effective_generation_budget(), test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold(), decode(), test_effective_allowance(), test_effective_spec_binds_gold_caps_and_runtime_spec(), test_invalid_allowance_fails_explicitly(), test_prepare_sampled_benchmark_collection_invalidates_stale_cache(), test_strict_primary_and_resilient_diagnostic_share_prediction() (+2 more)

### Community 180 - "FullCrossCachedAttention"
Cohesion: 0.10
Nodes (9): Bf16FullLinear, _compute_per_row_cross_query(), FullCrossCachedAttention, FullCrossTiledAttention, _norm_weight_dtype(), _owns_cross_kv(), _project_cross_kv(), _resolve_gate_init_tanh() (+1 more)

## Knowledge Gaps
- **1 isolated node(s):** `bifocalgemma`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 998 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **91 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BifocalGemmaForCausalLM` connect `BifocalGemmaForCausalLM` to `test_v4_architecture.py`, `fast_pipeline.py`, `SwiGLUDualGateBridge`, `FastTrainingPipeline`, `test_molab_remediation.py`, `bifocalgemma/__init__.py`, `FullCrossCachedAttention`, `modeling_bifocalgemma.py`, `make_experiment_spec`, `torch`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `stable_fingerprint()` (e.g. with `prepare_sampled_benchmark_collection()` and `run_batched_context_benchmark()`) actually correct?**
  _`stable_fingerprint()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `bifocalgemma` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `RepositorySnapshot` be split into smaller, more focused modules?**
  _Cohesion score 0.12315270935960591 - nodes in this community are weakly interconnected._
- **Why does `stable_fingerprint()` connect `stable_fingerprint` to `preflight.py`, `prepare_context_data.py`, `stages.py`, `repository_corpus.py`, `build_repository_corpus`, `hashlib`, `evaluation/context.py`, `curation.py`, `test_editing_manifest.py`, `json`, `json`, `make_experiment_spec`, `test_matrix_schedule_runtime.py`, `benchmark.py`, `repository_sft.py`, `run_matrix_training`, `remediation.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `pathlib` be split into smaller, more focused modules?**
  _Cohesion score 0.14624505928853754 - nodes in this community are weakly interconnected._
- **Why does `FastTrainingPipeline` connect `FastTrainingPipeline` to `.configure_stage1`, `fast_pipeline.py`, `test_molab_remediation.py`, `BifocalGemmaForCausalLM`, `test_fast_pipeline.py`, `__main__.py`, `JSONLRepairDataset`, `FullCrossCachedAttention`, `JSONLRepairDataset`, `.run`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._