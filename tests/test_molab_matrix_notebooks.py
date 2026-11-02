"""Comprehensive verification tests for the 10 Molab V3 matrix training notebooks and runner."""

import ast
import base64
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bifocalgemma.contracts import TokenizedRepairExample  # noqa: E402
from bifocalgemma.inference.generation import greedy_decode  # noqa: E402
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training  # noqa: E402
from bifocalgemma.models.native import NativeTinyModel  # noqa: E402
from bifocalgemma.training.hub import checked_prefix  # noqa: E402
from bifocalgemma.training.policy import configure_policy_model  # noqa: E402
from bifocalgemma.training.recipes import unified_recipe  # noqa: E402
from bifocalgemma.training.stages import StageLedger  # noqa: E402
from tools.molab_matrix_support import (  # noqa: E402
    ACCUMULATION,
    BATCH_SIZE,
    BENCHMARKS,
    CENTRALIZED_HF_REPO,
    COMPREHENSION_MIX_RATIO,
    EFFECTIVE_BATCH,
    GATE_INIT_TANH,
    HEAVY_BENCHMARK_SAMPLE_LIMIT,
    MATRIX_MODELS,
    STAGE1_STEPS,
    STAGE2_STEPS,
    TOTAL_STEPS,
    WARMUP_RATIO,
    StreamingPointerLoss,
    SwiGLUDualGateBridge,
    batched_greedy_decode,
    build_anti_forgetting_curriculum,
    build_bifocalgemma_v3_optimizer,
    clamped_triton_kernel_options,
    compute_diff_token_weights,
    configure_full_rank_cross_attention,
    configure_matrix_stage,
    critical_attention_beta,
    differential_diff_cross_entropy,
    format_query_injected_context,
    is_comprehension_example,
    is_cross_bridge_parameter,
    make_experiment_spec,
    patch_matrix_tokenization,
    prepare_sampled_benchmark_collection,
    prune_matrix_checkpoints,
    resolve_optimal_eval_batch_size,
    restore_matrix_checkpoint,
    run_matrix_stage,
    save_matrix_checkpoint,
)

EXPECTED_MODELS = {
    "2base-2it": {
        "encoder_id": "google/gemma-4-E2B",
        "encoder_rev": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "family": "symmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2base-2it",
    },
    "2it-2it": {
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "family": "symmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2it-2it",
    },
    "4base-4it": {
        "encoder_id": "google/gemma-4-E4B",
        "encoder_rev": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "family": "symmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4base-4it",
    },
    "4it-4it": {
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "family": "symmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4it-4it",
    },
    "4base-2it": {
        "encoder_id": "google/gemma-4-E4B",
        "encoder_rev": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "family": "asymmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4base-2it",
    },
    "4it-2it": {
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "family": "asymmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4it-2it",
    },
    "2base-4it": {
        "encoder_id": "google/gemma-4-E2B",
        "encoder_rev": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "family": "asymmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2base-4it",
    },
    "2it-4it": {
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "family": "asymmetric",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2it-4it",
    },
    "e2b-it": {
        "encoder_id": None,
        "encoder_rev": None,
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "family": "decoder_only",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "e2b-it",
    },
    "e4b-it": {
        "encoder_id": None,
        "encoder_rev": None,
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "family": "decoder_only",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "e4b-it",
    },
}


def test_matrix_models_definition_matches_all_ten_expected_models():
    assert CENTRALIZED_HF_REPO == "daruokta/bifocalgemma-2026-11-02"
    assert len(MATRIX_MODELS) == 10
    for key, expected in EXPECTED_MODELS.items():
        assert key in MATRIX_MODELS
        spec_def = MATRIX_MODELS[key]
        assert spec_def["encoder_id"] == expected["encoder_id"]
        assert spec_def["encoder_revision"] == expected["encoder_rev"]
        assert spec_def["decoder_id"] == expected["decoder_id"]
        assert spec_def["decoder_revision"] == expected["decoder_rev"]
        assert spec_def["family"] == expected["family"]
        assert spec_def["repo"] == expected["repo"]
        assert spec_def["subfolder"] == expected["subfolder"]


def test_make_experiment_spec_and_hub_prefix_for_all_ten_models():
    for key, expected in EXPECTED_MODELS.items():
        spec = make_experiment_spec(key)
        assert key in spec.name
        if expected["encoder_id"] is None:
            assert spec.encoder is None
            assert spec.attention_mode == "causal"
        else:
            assert spec.encoder is not None
            assert spec.encoder.model_id == expected["encoder_id"]
            assert spec.encoder.revision == expected["encoder_rev"]
            assert spec.attention_mode == "merged"
        assert spec.decoder.model_id == expected["decoder_id"]
        assert spec.decoder.revision == expected["decoder_rev"]
        prefix = f"{MATRIX_MODELS[key]['subfolder']}/{spec.name}/train"
        assert checked_prefix(prefix) == prefix


def test_matrix_curriculum_stages_and_batch_size():
    assert TOTAL_STEPS == 11102
    assert STAGE1_STEPS == 5551
    assert STAGE2_STEPS == 5551
    assert STAGE1_STEPS + STAGE2_STEPS == TOTAL_STEPS
    assert BATCH_SIZE == 2
    assert ACCUMULATION == 2
    assert BATCH_SIZE * ACCUMULATION == EFFECTIVE_BATCH == 4
    assert WARMUP_RATIO == 0.05


def test_benchmark_reordering_six_full_then_two_sampled_100():
    assert len(BENCHMARKS) == 8
    expected_order = [
        ("longcodeqa_gemma32k_dec8k_v1", None),
        ("humanevalfix_python_v2", None),
        ("sweqa_oracle_gemma32k_dec8k_v1", None),
        ("sweqa_noisy_oracle_gemma32k_dec8k_v1", None),
        ("gsm8k_zero_shot_v2", None),
        ("gsm8k_eight_shot_v2", None),
        ("crosscodelong_chunk_gemma32k_dec8k_v1", 100),
        ("crosscodelong_function_gemma32k_dec8k_v1", 100),
    ]
    assert HEAVY_BENCHMARK_SAMPLE_LIMIT == 100
    for idx, (expected_name, expected_limit) in enumerate(expected_order):
        item = BENCHMARKS[idx]
        assert item["collection"] == expected_name
        assert item.get("sample_limit") == expected_limit


@pytest.mark.parametrize("model_key", list(EXPECTED_MODELS.keys()))
def test_notebook_exists_and_is_valid_ast(model_key):
    filename = f"train_{model_key.replace('-', '_')}.py"
    notebook_path = ROOT / "notebooks" / filename
    assert notebook_path.is_file(), f"Notebook {filename} must exist"

    code = notebook_path.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(notebook_path))
    assert isinstance(tree, ast.Module)


@pytest.mark.parametrize("model_key", list(EXPECTED_MODELS.keys()))
def test_notebook_has_correct_metadata_and_centralized_hf_repo(model_key):
    filename = f"train_{model_key.replace('-', '_')}.py"
    notebook_path = ROOT / "notebooks" / filename
    code = notebook_path.read_text(encoding="utf-8")
    expected = EXPECTED_MODELS[model_key]

    assert f'MODEL_KEY = "{model_key}"' in code
    assert f'HF_REPO_ID = "{CENTRALIZED_HF_REPO}"' in code
    assert f'HF_SUBFOLDER = "{expected["subfolder"]}"' in code
    if expected["encoder_id"] is not None:
        assert expected["encoder_id"] in code
        assert expected["encoder_rev"] in code
    assert expected["decoder_id"] in code
    assert expected["decoder_rev"] in code


@pytest.mark.parametrize("model_key", list(EXPECTED_MODELS.keys()))
def test_embedded_bundle_can_be_extracted_and_contains_matrix_support(model_key):
    filename = f"train_{model_key.replace('-', '_')}.py"
    notebook_path = ROOT / "notebooks" / filename
    code = notebook_path.read_text(encoding="utf-8")

    tree = ast.parse(code)
    bundle_val = None
    bundle_sha = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "SOURCE_BUNDLE" and isinstance(node.value, ast.Constant):
                        bundle_val = node.value.value
                    elif target.id == "BUNDLE_SHA256" and isinstance(node.value, ast.Constant):
                        bundle_sha = node.value.value

    assert bundle_val is not None, "SOURCE_BUNDLE string constant must be present"
    assert bundle_sha is not None, "BUNDLE_SHA256 string constant must be present"

    raw_zip = base64.b85decode(bundle_val)
    computed_sha = hashlib.sha256(raw_zip).hexdigest()
    assert computed_sha == bundle_sha

    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        namelist = set(archive.namelist())
        assert "tools/molab_matrix_support.py" in namelist
        assert "tools/notebook_launcher.py" in namelist
        assert "tools/notebook_dependencies.py" in namelist
        assert "LICENSE" in namelist
        assert "src/bifocalgemma/__init__.py" in namelist
        assert "src/bifocalgemma/config.py" in namelist
        assert "src/bifocalgemma/models/memory_efficient.py" in namelist
        assert "src/bifocalgemma/training/wiggly_dashboard.py" in namelist


@pytest.mark.parametrize("model_key", ["2base-2it", "4base-2it", "2base-4it", "e2b-it"])
def test_configure_full_rank_cross_attention_v3_specs(model_key):
    spec = make_experiment_spec(model_key)
    family = MATRIX_MODELS[model_key]["family"]
    policy = unified_recipe(family)
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    info = configure_full_rank_cross_attention(model)
    if spec.encoder is None:
        assert info["mode"] == "decoder_only_causal_v3"
        assert info["total_cross_bridge_parameters"] == 0
    else:
        assert info["mode"] == "full_rank_cross_bf16"
        assert info["bridge_type"] == "SwiGLUDualGateBridge"
        assert math.isclose(info["gate_init_tanh"], GATE_INIT_TANH, rel_tol=1e-5)
        assert info["kv_owner_layers"] > 0
        assert info["total_cross_bridge_parameters"] > 0
        assert isinstance(model.encoder_projection, SwiGLUDualGateBridge)
        first_attn = model.decoder.layers[0].self_attn
        assert torch.allclose(
            torch.tanh(first_attn.cross_gate.detach()),
            torch.full_like(first_attn.cross_gate, GATE_INIT_TANH),
            atol=1e-5,
        )


def test_swiglu_dual_gate_bridge_symmetric_and_asymmetric_shapes_and_grads():
    sym_bridge = SwiGLUDualGateBridge(64, 64, init_rms_scale=1.25)
    assert isinstance(sym_bridge.w_res, torch.nn.Identity)
    assert sym_bridge.w_gate.out_features == 64  # d_mid == max(d_enc, d_dec)
    assert sym_bridge.gamma_calib.shape == (64,)  # 1D per-channel calibration gain vector
    x_sym = torch.randn(2, 16, 64, requires_grad=True)
    out_sym = sym_bridge(x_sym)
    assert out_sym.shape == (2, 16, 64)
    out_sym.sum().backward()
    assert x_sym.grad is not None and bool(torch.isfinite(x_sym.grad).all())
    assert sym_bridge.gamma_calib.grad is not None

    asym_bridge = SwiGLUDualGateBridge(96, 64)
    assert isinstance(asym_bridge.w_res, torch.nn.Linear)
    assert asym_bridge.w_gate.out_features == 96  # max(96, 64)
    assert asym_bridge.gamma_calib.shape == (64,)
    x_asym = torch.randn(2, 16, 96, requires_grad=True)
    out_asym = asym_bridge(x_asym)
    assert out_asym.shape == (2, 16, 64)
    out_asym.sum().backward()
    assert x_asym.grad is not None and bool(torch.isfinite(x_asym.grad).all())


def test_v3_critical_attention_beta_and_triton_tile_clamping():
    from tools.molab_matrix_support import _compute_per_row_cross_query

    assert critical_attention_beta(512) == 1.0
    assert critical_attention_beta(2048) == 1.0
    beta_32k = critical_attention_beta(32768)
    assert math.isclose(beta_32k, 0.1 * math.log(32768 / 2048) + 1.0, rel_tol=1e-6)
    assert beta_32k > 1.25

    # Per-row critical attention temperature divides cross-queries by beta_N (softening, not sharpening)
    q_cross = torch.ones((2, 4, 3, 8), dtype=torch.float32)
    enc_valid = torch.zeros((2, 4096), dtype=torch.bool)
    enc_valid[0, :4096] = True  # N_enc = 4096 -> beta > 1.0
    enc_valid[1, :512] = True  # N_enc = 512 -> beta == 1.0
    scaled_q = _compute_per_row_cross_query(q_cross, enc_valid, 4096)
    expected_beta_0 = critical_attention_beta(4096)
    assert torch.allclose(scaled_q[0], torch.full_like(scaled_q[0], 1.0 / expected_beta_0), atol=1e-6)
    assert torch.allclose(scaled_q[1], torch.ones_like(scaled_q[1]), atol=1e-6)

    clamp_256 = clamped_triton_kernel_options(256)
    assert clamp_256["bwd_BLOCK_M"] == 16
    assert clamp_256["bwd_BLOCK_N"] == 16
    assert clamp_256["bwd_num_stages"] == 1
    clamp_128 = clamped_triton_kernel_options(128)
    assert clamp_128["BLOCK_M"] == 64


def test_v3_streaming_pointer_loss_and_differential_diff_weighting():
    ptr_loss_fn = StreamingPointerLoss(d_dec=32, d_ptr=16, chunk_size=4)
    h_dec = torch.randn(2, 6, 32, requires_grad=True)
    z_enc = torch.randn(2, 8, 32, requires_grad=True)
    enc_ids = torch.tensor([[2, 10, 20, 30, 40, 50, 60, 70], [2, 11, 21, 31, 41, 51, 61, 71]])
    target_ids = torch.tensor([[10, 20, 99, -100, 40, 50], [11, 99, 31, 41, -100, -100]])
    enc_mask = torch.ones_like(enc_ids, dtype=torch.bool)

    loss = ptr_loss_fn(h_dec, z_enc, enc_ids, target_ids, enc_mask)
    assert loss.ndim == 0 and float(loss.item()) > 0.0
    loss.backward()
    assert h_dec.grad is not None and z_enc.grad is not None

    # Empty mask edge case returns 0.0 without NaN
    zero_loss = ptr_loss_fn(h_dec, z_enc, enc_ids, target_ids, torch.zeros_like(enc_mask))
    assert float(zero_loss.item()) == 0.0

    # Differential diff token weights normalize to mean 1.0 over valid labels
    labels = torch.tensor([[10, 20, 30, 40, -100]])
    weights = compute_diff_token_weights(
        labels, add_token_ids=(10,), del_token_ids=(20,), hdr_token_ids=(30,)
    )
    valid = labels != -100
    assert math.isclose(float(weights[valid].mean().item()), 1.0, rel_tol=1e-5)
    assert float(weights[0, 0]) > float(weights[0, 1]) > float(weights[0, 2]) > float(weights[0, 3])

    logits = torch.randn(1, 5, 64, requires_grad=True)
    ce = differential_diff_cross_entropy(logits, labels, weights)
    assert bool(torch.isfinite(ce))
    ce.backward()
    assert logits.grad is not None


def test_v3_partitioned_optimizer_weight_decay_invariants():
    spec = make_experiment_spec("4base-2it")
    policy = unified_recipe("asymmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    stage_groups, _ = configure_matrix_stage(model, policy, freeze_decoder=False, total_steps=100)
    stage_groups_by_name = {g["name"]: g for g in stage_groups}
    assert stage_groups_by_name["alignment_bridge_decay"]["weight_decay"] == 1e-4
    assert stage_groups_by_name["alignment"]["weight_decay"] == 0.0
    assert stage_groups_by_name["encoder"]["weight_decay"] == 0.0
    assert stage_groups_by_name["decoder"]["weight_decay"] == 0.0

    opt = build_bifocalgemma_v3_optimizer(model, weight_decay_bridge=1e-4)
    groups_by_name = {g["name"]: g for g in opt.param_groups}
    assert groups_by_name["bridge_linear_decay"]["weight_decay"] == 1e-4
    assert groups_by_name["bridge_norm_and_calib_nodecay"]["weight_decay"] == 0.0
    assert groups_by_name["decoupled_cross_attn_nodecay"]["weight_decay"] == 0.0
    assert groups_by_name["backbone_lora"]["weight_decay"] == 0.0


def test_anti_forgetting_curriculum_balances_comprehension_and_preserves_length():
    assert COMPREHENSION_MIX_RATIO == 0.20
    raw_examples = [
        TokenizedRepairExample(
            f"rep_{i}",
            (2, 10 + i, 20 + i, 30 + i, 40 + i, 50 + i),
            (2, 100 + i, 101 + i),
            (200 + i, 201 + i, 107),
            metadata={"source": "nebius/SWE-rebench-V2", "task_type": "repair"},
        )
        for i in range(40)
    ]
    curriculum_s1, stats_s1 = build_anti_forgetting_curriculum(
        raw_examples, stage_name="frozen_decoder", seed=17
    )
    assert len(curriculum_s1) == len(raw_examples) == 40
    assert stats_s1["comprehension_examples"] == 8
    assert math.isclose(stats_s1["comprehension_ratio"], 0.20, rel_tol=1e-5)
    # Stage 1 front-loads comprehension alignment examples at index 0
    assert is_comprehension_example(curriculum_s1[0])

    curriculum_s2, stats_s2 = build_anti_forgetting_curriculum(
        raw_examples, stage_name="joint", seed=17
    )
    assert len(curriculum_s2) == 40
    assert stats_s2["comprehension_examples"] == 8
    assert math.isclose(stats_s2["comprehension_ratio"], 0.20, rel_tol=1e-5)

    # Edge case: empty list
    empty_cur, empty_stats = build_anti_forgetting_curriculum([])
    assert empty_cur == [] and empty_stats["total_examples"] == 0


def test_prepare_sampled_benchmark_collection_deterministic_100_cases(tmp_path):
    src_dir = tmp_path / "crosscodelong_chunk_gemma32k_dec8k_v1"
    src_dir.mkdir(parents=True)
    with (src_dir / "inputs.jsonl").open("w", encoding="utf-8", newline="\n") as f_in, (
        src_dir / "hidden_gold.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as f_gold:
        for i in range(250):
            f_in.write(json.dumps({"instance_id": f"case_{i:04d}", "role": "eval", "context": f"x={i}"}) + "\n")
            f_gold.write(json.dumps({"instance_id": f"case_{i:04d}", "reference": f"x={i}"}) + "\n")

    # Full benchmark (sample_limit=None) returns original directory untouched
    assert prepare_sampled_benchmark_collection(src_dir, sample_limit=None) == src_dir.resolve()

    # Sampled benchmark (sample_limit=100) creates deterministic 100-case cohort
    sampled_1 = prepare_sampled_benchmark_collection(src_dir, sample_limit=100, sample_seed=17)
    sampled_2 = prepare_sampled_benchmark_collection(
        src_dir, sample_limit=100, sample_seed=17, output_dir=tmp_path / "second_check"
    )
    rows_1 = [json.loads(line) for line in (sampled_1 / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    gold_1 = [json.loads(line) for line in (sampled_1 / "hidden_gold.jsonl").read_text(encoding="utf-8").splitlines()]
    rows_2 = [json.loads(line) for line in (sampled_2 / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows_1) == 100
    assert [r["instance_id"] for r in rows_1] == [g["instance_id"] for g in gold_1]
    assert [r["instance_id"] for r in rows_1] == [r["instance_id"] for r in rows_2]


def test_resolve_optimal_eval_batch_size_scales_with_model_and_benchmark_weight():
    gsm8k_item = BENCHMARKS[4]  # light
    sweqa_item = BENCHMARKS[2]  # medium_long
    ccl_item = BENCHMARKS[6]  # heavy

    # Across benchmark weights for the same model: light >= medium_long >= heavy
    bs_light = resolve_optimal_eval_batch_size("2base-2it", gsm8k_item, free_vram_gib=64.0)
    bs_med = resolve_optimal_eval_batch_size("2base-2it", sweqa_item, free_vram_gib=64.0)
    bs_heavy = resolve_optimal_eval_batch_size("2base-2it", ccl_item, free_vram_gib=64.0)
    assert bs_light >= bs_med >= bs_heavy >= 1

    # Across model sizes for the same benchmark: e2b-it >= 2base-2it >= 4base-2it >= 4base-4it
    bs_e2b = resolve_optimal_eval_batch_size("e2b-it", sweqa_item, free_vram_gib=56.0)
    bs_2b2b = resolve_optimal_eval_batch_size("2base-2it", sweqa_item, free_vram_gib=56.0)
    bs_4b2b = resolve_optimal_eval_batch_size("4base-2it", sweqa_item, free_vram_gib=56.0)
    bs_4b4b = resolve_optimal_eval_batch_size("4base-4it", sweqa_item, free_vram_gib=56.0)
    assert bs_e2b >= bs_2b2b >= bs_4b2b >= bs_4b4b >= 1

    # Tight VRAM edge case falls back safely to batch_size = 1
    assert resolve_optimal_eval_batch_size("4base-4it", ccl_item, free_vram_gib=4.0) == 1


def test_configure_matrix_stage_parameter_freezing_paired_and_baseline():
    spec = make_experiment_spec("4base-2it")
    policy = unified_recipe("asymmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)

    # Stage 1: Frozen decoder on paired model
    groups1, summary1 = configure_matrix_stage(model, policy, freeze_decoder=True, total_steps=5551)
    assert summary1["freeze_decoder"] is True
    assert summary1["active_groups"] == ["encoder", "alignment"]
    assert [g["name"] for g in groups1] == ["encoder", "alignment_bridge_decay", "alignment"]
    assert all(g["warmup_ratio"] == 0.05 and g["warmup_steps"] == 278 for g in groups1)
    assert summary1["decoder_trainable_tensors"] == 0
    for name, param in model.named_parameters():
        if name.startswith("decoder.") and not is_cross_bridge_parameter(name):
            assert not param.requires_grad

    # Stage 2: Joint SFT on paired model
    groups2, summary2 = configure_matrix_stage(model, policy, freeze_decoder=False, total_steps=5551)
    assert summary2["freeze_decoder"] is False
    assert summary2["active_groups"] == ["encoder", "decoder", "alignment"]
    assert [g["name"] for g in groups2] == ["encoder", "decoder", "alignment_bridge_decay", "alignment"]
    assert summary2["decoder_trainable_tensors"] > 0

    # Baseline decoder-only model (`e2b-it`) keeps decoder LoRA active in both stages
    spec_base = make_experiment_spec("e2b-it")
    policy_base = unified_recipe("decoder_only")
    model_base = NativeTinyModel(spec_base, tiny=True)
    configure_policy_model(model_base, policy_base)
    configure_memory_efficient_training(model_base)
    configure_full_rank_cross_attention(model_base)
    _, sum_base = configure_matrix_stage(model_base, policy_base, freeze_decoder=True, total_steps=5551)
    assert sum_base["freeze_decoder"] is False
    assert sum_base["active_groups"] == ["decoder"]
    assert sum_base["decoder_trainable_tensors"] > 0


def test_save_and_restore_matrix_checkpoint_and_prune(tmp_path):
    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    groups, _ = configure_matrix_stage(model, policy, freeze_decoder=True, total_steps=5551)
    optimizer = torch.optim.AdamW(groups)

    p1 = save_matrix_checkpoint(
        model,
        optimizer,
        tmp_path / "checkpoint-005551",
        global_step=5551,
        stage_name="frozen_decoder",
        compatibility={"test": 1},
        training_state={"step": 5551},
    )
    assert (p1 / "READY.json").is_file()
    assert (p1 / "manifest.json").is_file()
    assert (p1 / "adapters.safetensors").is_file()
    assert (p1 / "trainer_state.pt").is_file()

    save_matrix_checkpoint(model, optimizer, tmp_path / "checkpoint-006000", 6000, "joint", {}, {})
    save_matrix_checkpoint(model, optimizer, tmp_path / "checkpoint-007000", 7000, "joint", {}, {})
    save_matrix_checkpoint(model, optimizer, tmp_path / "checkpoint-011102", 11102, "joint", {}, {})

    restored = restore_matrix_checkpoint(model, optimizer, p1)
    assert restored["global_step"] == 5551
    assert restored["stage"] == "frozen_decoder"
    assert restored["training_state"]["step"] == 5551

    kept = prune_matrix_checkpoints(tmp_path, keep_limit=2)
    assert kept == ["checkpoint-005551", "checkpoint-007000", "checkpoint-011102"]
    assert not (tmp_path / "checkpoint-006000").exists()


def test_run_matrix_stage_execution_batch2_and_frozen_decoder_invariance(tmp_path):
    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)

    examples = [
        TokenizedRepairExample(
            f"ex_{i}",
            (2, 10 + i, 20 + i),
            (2, 30 + i, 40 + i),
            (50 + i, 60 + i, 107),
            metadata={"source": "nebius/SWE-rebench-V2", "task_type": "repair"},
        )
        for i in range(12)
    ]
    decoder_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith("decoder.") and not is_cross_bridge_parameter(n)
    }

    rep1 = run_matrix_stage(
        model,
        examples,
        examples[:2],
        stage_name="frozen_decoder",
        policy=policy,
        max_steps=2,
        freeze_decoder=True,
        run_dir=tmp_path,
        compatibility={"model_key": "2base-2it"},
        start_global_step=0,
        batch_size=BATCH_SIZE,
        accumulation=ACCUMULATION,
    )
    assert rep1["completed_steps"] == 2
    assert rep1["end_global_step"] == 2
    assert rep1["curriculum"]["comprehension_examples"] > 0

    for n, p_before in decoder_before.items():
        p_after = model.get_parameter(n)
        torch.testing.assert_close(p_after, p_before, rtol=0, atol=0)

    rep2 = run_matrix_stage(
        model,
        examples,
        examples[:2],
        stage_name="joint",
        policy=policy,
        max_steps=2,
        freeze_decoder=False,
        run_dir=tmp_path,
        compatibility={"model_key": "2base-2it"},
        start_global_step=2,
        batch_size=BATCH_SIZE,
        accumulation=ACCUMULATION,
    )
    assert rep2["completed_steps"] == 2
    assert rep2["end_global_step"] == 4


def test_cached_uncached_and_batched_greedy_decode_equivalence():
    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)

    gen_uncached = greedy_decode(model, (2, 10, 20), (2, 30, 40), max_new_tokens=4, use_cache=False)
    gen_cached = greedy_decode(model, (2, 10, 20), (2, 30, 40), max_new_tokens=4, use_cache=True)
    assert gen_uncached.token_ids == gen_cached.token_ids

    # Heterogeneous encoder lengths, heterogeneous prefix lengths, and heterogeneous generation budgets
    het_enc = [(2, 10, 20), (2, 15, 25, 35, 45, 55), (2, 12)]
    het_pref = [(2, 30, 40, 50), (2, 35), (2, 41, 42)]
    het_budgets = [4, 3, 5]
    singles = [
        greedy_decode(model, enc_i, pref_i, max_new_tokens=b_i, use_cache=True)
        for enc_i, pref_i, b_i in zip(het_enc, het_pref, het_budgets, strict=True)
    ]

    encode_calls = []
    orig_encode = model.encode_context

    def spy_encode(input_ids, attention_mask):
        encode_calls.append(tuple(input_ids.shape))
        return orig_encode(input_ids, attention_mask)

    model.encode_context = spy_encode
    try:
        batched = batched_greedy_decode(
            model,
            het_enc,
            het_pref,
            max_new_tokens_list=het_budgets,
            use_cache=True,
        )
    finally:
        model.encode_context = orig_encode

    assert encode_calls == [(3, 6)]  # Single true batched encoder pass across all 3 heterogeneous items
    assert len(batched) == 3
    for b_idx in range(3):
        assert batched[b_idx].token_ids == singles[b_idx].token_ids
        assert batched[b_idx].encoder_tokens == len(het_enc[b_idx])
        assert batched[b_idx].decoder_prefix_tokens == len(het_pref[b_idx])

    # Decoder-only baseline (`e2b-it`) with heterogeneous prefix lengths
    spec_base = make_experiment_spec("e2b-it")
    policy_base = unified_recipe("decoder_only")
    model_base = NativeTinyModel(spec_base, tiny=True)
    configure_policy_model(model_base, policy_base)
    configure_memory_efficient_training(model_base)
    configure_full_rank_cross_attention(model_base)

    base_singles = [
        greedy_decode(model_base, (), pref_i, max_new_tokens=b_i, use_cache=True)
        for pref_i, b_i in zip(het_pref, het_budgets, strict=True)
    ]
    base_batched = batched_greedy_decode(
        model_base,
        [(), (), ()],
        het_pref,
        max_new_tokens_list=het_budgets,
        use_cache=True,
    )
    assert len(base_batched) == 3
    for b_idx in range(3):
        assert base_batched[b_idx].token_ids == base_singles[b_idx].token_ids


def test_synchronize_matrix_v3_retains_stage1_milestone_and_retries_412_contention(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    from tools.molab_matrix_support import synchronize_matrix_v3

    run_dir = tmp_path / "run"
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True)

    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    groups, _ = configure_matrix_stage(model, policy, freeze_decoder=True, total_steps=5551)
    optimizer = torch.optim.AdamW(groups)

    save_matrix_checkpoint(model, optimizer, train_dir / "checkpoint-005551", 5551, "frozen_decoder", {}, {})
    save_matrix_checkpoint(model, optimizer, train_dir / "checkpoint-006000", 6000, "joint", {}, {})
    save_matrix_checkpoint(model, optimizer, train_dir / "checkpoint-007000", 7000, "joint", {}, {})
    save_matrix_checkpoint(model, optimizer, train_dir / "checkpoint-011102", 11102, "joint", {}, {})

    prefix = f"2base-2it/{spec.name}/train"
    add_attempts = 0
    delete_attempts = 0
    remote_files: set[str] = {
        f"{prefix}/training/checkpoint-005551/READY.json",
        f"{prefix}/training/checkpoint-006000/READY.json",
    }

    def make_412():
        resp = SimpleNamespace(status_code=412, headers={}, request=None)
        return HfHubHTTPError("Precondition Failed", response=resp)

    class FakeHfApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, repo_id, repo_type="model", exist_ok=True):
            return None

        def model_info(self, repo_id):
            return SimpleNamespace(sha=f"head-{add_attempts}-{delete_attempts}")

        def create_commit(self, repo_id, operations, parent_commit, commit_message):
            nonlocal add_attempts, delete_attempts
            if "Save verified" in commit_message:
                add_attempts += 1
                if add_attempts < 3:
                    raise make_412()
                for op in operations:
                    remote_files.add(op.path_in_repo)
                return SimpleNamespace(oid="commit-main-ok")
            else:
                delete_attempts += 1
                if delete_attempts < 2:
                    raise make_412()
                for op in operations:
                    remote_files.discard(op.path_in_repo)
                return SimpleNamespace(oid="commit-prune-ok")

        def list_repo_files(self, repo_id, revision=None):
            return sorted(remote_files)

    def fake_download(repo_id, filename, revision=None, token=None):
        ckpt_name = filename.split("/")[-2]
        return str(train_dir / ckpt_name / "READY.json")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    res = synchronize_matrix_v3(
        CENTRALIZED_HF_REPO,
        run_dir,
        prefix,
        token="hf_test",
        max_retries=5,
        base_delay=0.0,
    )
    assert res["revision"] == "commit-prune-ok"
    assert res["retained_checkpoints"] == [
        "checkpoint-005551",
        "checkpoint-007000",
        "checkpoint-011102",
    ]
    assert add_attempts == 3
    assert delete_attempts == 2
    assert f"{prefix}/training/checkpoint-006000/READY.json" not in remote_files
    assert f"{prefix}/training/checkpoint-005551/READY.json" in remote_files


def test_pipeline_state_ledger_creation_for_remote_resumption(tmp_path):
    spec = make_experiment_spec("4base-4it")
    identity = {
        "format": "bifocalgemma_pipeline_v1",
        "spec": spec.to_dict(),
        "total_steps": TOTAL_STEPS,
        "stage1_steps": STAGE1_STEPS,
        "stage2_steps": STAGE2_STEPS,
    }
    ledger = StageLedger(tmp_path, identity)
    assert ledger.path == tmp_path / "pipeline_state.json"
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("dummy", encoding="utf-8")
    ledger.finish("stage1_frozen", [dummy_file], {"status": "ok"})
    state = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert state["format"] == "bifocalgemma_pipeline_v1"
    assert "stage1_frozen" in state["stages"]
    assert state["stages"]["stage1_frozen"]["status"] == "COMPLETED"
    assert state["stages"]["stage1_frozen"]["result"]["status"] == "ok"


def test_matrix_tokenization_patch_it_encoder_and_query_prefix_contract():
    """Verify IT encoder BOS prepending and optional query-prefix injection."""
    patch_matrix_tokenization()

    import bifocalgemma.data.context_tasks as ct_mod
    import bifocalgemma.evaluation.context as eval_ctx_mod

    assert format_query_injected_context("def foo(): pass", "Fix foo") == "[QUERY] Fix foo\ndef foo(): pass"

    class MockITEncoder:
        bos_token_id = 2

        def encode(self, text, add_special_tokens=True):
            return [101, 102, 103]

    class MockBaseEncoder:
        bos_token_id = 2

        def encode(self, text, add_special_tokens=True):
            return [2, 101, 102, 103]

    class MockDecoder:
        bos_token_id = 2

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ):
            if add_generation_prompt:
                return "<bos><start_of_turn>user\nrepair<end_of_turn>\n<start_of_turn>model\n"
            return (
                f"<bos><start_of_turn>user\nrepair<end_of_turn>\n<start_of_turn>model\n"
                f"{messages[-1]['content']}<end_of_turn>\n<turn|>"
            )

        def encode(self, text, add_special_tokens=False):
            if "<turn|>" in text:
                return (2, 10, 20, 30, 40, 107)
            return (2, 10, 20)

        def convert_tokens_to_ids(self, tok):
            return 107

    row = {
        "instance_id": "test_it_encoder_contract_1",
        "context": "def calculate(): return 42\n",
        "source": "nebius/SWE-rebench-V2",
        "task_type": "repair",
        "instruction": "Fix bug in calculate",
        "target_text": "diff --git a/calc.py b/calc.py\n",
    }

    spec_it = make_experiment_spec("4it-2it")
    res_it = ct_mod.tokenize_context_task(
        row, spec_it, MockITEncoder(), MockDecoder(), encoder_limit=1000, decoder_limit=1000
    )
    assert res_it.encoder_ids == (2, 101, 102, 103)
    assert res_it.decoder_prefix_ids[0] == 2
    assert res_it.target_ids[-1] == 107

    spec_base = make_experiment_spec("4base-2it")
    res_base = ct_mod.tokenize_context_task(
        row, spec_base, MockBaseEncoder(), MockDecoder(), encoder_limit=1000, decoder_limit=1000
    )
    assert res_base.encoder_ids == (2, 101, 102, 103)

    eval_row = {
        "role": "eval",
        "instruction": "Fix bug",
        "context": "def foo(): pass",
    }
    enc, _ = eval_ctx_mod.prompt_ids(
        eval_row, spec_it, MockITEncoder(), MockDecoder(), encoder_limit=1000
    )
    assert enc == (2, 101, 102, 103)
