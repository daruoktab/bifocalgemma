"""Offline v4 tests for 6-group partitioned optimizer, WSD/asymmetric schedule, and task-gated loss."""

from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from bifocalgemma.config import MATRIX_MODELS, make_experiment_spec
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.modeling_bifocalgemma import NativeTinyModel
from bifocalgemma.models.cross_attention import configure_full_rank_cross_attention
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.training.matrix import (
    _build_schedule_for_epochs,
    resolve_training_contract,
    run_matrix_stage,
)
from bifocalgemma.training.objectives import (
    DEFAULT_DIFF_ADD_TOKEN_IDS,
    DEFAULT_DIFF_DEL_TOKEN_IDS,
    DEFAULT_DIFF_HDR_TOKEN_IDS,
    DEFAULT_DIFF_HUNK_HDR_TOKEN_IDS,
    build_bifocalgemma_optimizer,
    chunked_differential_cross_entropy,
    compute_diff_token_weights,
    compute_example_loss_weights,
    differential_diff_cross_entropy,
)
from bifocalgemma.training.policy import (
    configure_policy_model,
    update_policy_learning_rates,
    validate_training_policy,
)
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.schedule import build_shared_schedule, canonical_schedule_records
from bifocalgemma.training.stage import (
    ALIGNMENT_GROUP_NAMES,
    CANONICAL_PARAM_GROUP_ORDER,
    build_bifocalgemma_adamw,
    clip_optimizer_group_grads,
    configure_matrix_stage,
)


def _tiny_model(
    key: str = "2base-2it", *, learnable_cross_scale: bool | None = None
) -> tuple[NativeTinyModel, dict[str, Any]]:
    torch.manual_seed(17)
    spec = make_experiment_spec(key)
    model = NativeTinyModel(spec, tiny=True)
    if learnable_cross_scale is not None:
        cast(Any, model).config = SimpleNamespace(learnable_cross_scale=learnable_cross_scale)
    policy = unified_recipe(MATRIX_MODELS[key]["family"])
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    return model, policy


TINY_DIFF_IDS: dict[str, tuple[int, ...]] = {
    "add_token_ids": (41,),
    "del_token_ids": (42,),
    "hdr_token_ids": (43,),
    "hunk_hdr_token_ids": (43,),
}


def _synthetic_cohort(
    key: str,
    *,
    count: int = 8,
    target_lengths: list[int] | None = None,
    tasks: list[str] | None = None,
) -> tuple[list[TokenizedRepairExample], Any, list[dict[str, Any]]]:
    paired = MATRIX_MODELS[key]["encoder_id"] is not None
    raw: list[dict[str, Any]] = []
    examples: list[TokenizedRepairExample] = []
    for i in range(count):
        task = tasks[i % len(tasks)] if tasks else ("repair" if i % 2 == 0 else "completion")
        t_len = target_lengths[i % len(target_lengths)] if target_lengths else 4
        if task == "repair":
            # Include real unified-diff tokens (hunk header + add + del + context) within tiny vocab
            base_tokens = [43, 42, 41]
            target_ids = tuple((base_tokens + [50 + (j % 30) for j in range(t_len)])[: max(3, t_len)])
        else:
            # Deliberately include '+' / '-' tokens in non-repair tasks to prove zero leakage
            target_ids = tuple(
                [41, 42] + [60 + ((i + j) % 30) for j in range(max(1, t_len - 2))]
            )[: max(2, t_len)]
        raw.append(
            {
                "instance_id": f"v4-item-{i}",
                "instruction": f"Task {task} #{i}",
                "context": f"def fn_{i}(x): return x + {i}",
                "target_text": f"out_{i}",
                "source": f"v4-src-{i % 2}",
                "task_type": task,
                "token_lengths": {"encoder": 16 + i},
            }
        )
        examples.append(
            TokenizedRepairExample(
                f"v4-item-{i}",
                (2, 10 + i, 20, 21) if paired else (),
                (2, 30, 31) if paired else (2, 30, 31, 10 + i, 20, 21),
                target_ids,
                metadata={"task_type": task, "source": f"v4-src-{i % 2}"},
            )
        )
    records = canonical_schedule_records(raw)
    schedule = build_shared_schedule(records, dataset_identity={"fixture": "v4_opt"})
    return examples, schedule, records


# ---------------------------------------------------------------------------
# 1 & 2. Consolidated 6-Group Optimizer, Argon #116 Bugfixes, Per-Group Clip
# ---------------------------------------------------------------------------


def test_six_group_partitioning_and_argon_116_invariants():
    model, policy = _tiny_model("4base-2it")
    validate_training_policy(policy, model)

    # Default learnable_cross_scale=False freezes cross_scale; Stage 2 activates all 6 groups.
    groups, summary = configure_matrix_stage(
        model, policy, freeze_decoder=False, total_steps=100, warmup_offset=6
    )
    by_name = {g["name"]: g for g in groups}
    assert tuple(by_name) == CANONICAL_PARAM_GROUP_ORDER
    assert summary["learnable_cross_scale"] is False

    # cross_gate_scalars: ONLY cross_gate (never cross_scale, never cross_gate_proj)
    assert by_name["cross_gate_scalars"]["lr"] == pytest.approx(2e-3)
    assert by_name["cross_gate_scalars"]["weight_decay"] == 0.0
    assert by_name["cross_gate_scalars"]["clip_norm"] is None
    assert all(n.endswith(".cross_gate") for n in by_name["cross_gate_scalars"]["param_names"])
    assert not any("cross_scale" in n for n in by_name["cross_gate_scalars"]["param_names"])
    assert not any("cross_gate_proj" in n for n in by_name["cross_gate_scalars"]["param_names"])

    # cross_gate_proj: OWN separate clip group, NOT lumped into bridge_mlp
    assert by_name["cross_gate_proj"]["lr"] == pytest.approx(5e-4)
    assert by_name["cross_gate_proj"]["weight_decay"] == 0.0
    assert by_name["cross_gate_proj"]["clip_norm"] == pytest.approx(1.0)
    assert all(n.endswith(".cross_gate_proj") for n in by_name["cross_gate_proj"]["param_names"])
    assert not any("cross_gate_proj" in n for n in by_name["bridge_mlp"]["param_names"])

    # bridge_mlp: 2D linears in encoder_projection ONLY, wd=1e-4
    assert by_name["bridge_mlp"]["lr"] == pytest.approx(4e-4)
    assert by_name["bridge_mlp"]["weight_decay"] == pytest.approx(1e-4)
    assert by_name["bridge_mlp"]["clip_norm"] == pytest.approx(1.0)
    assert all(n.startswith("encoder_projection.") for n in by_name["bridge_mlp"]["param_names"])
    assert all(p.ndim >= 2 for p in by_name["bridge_mlp"]["params"])

    # cross_kv_proj: cross_k/v_proj, cross_*_norm, bridge norms/gamma, wd=0.0
    assert by_name["cross_kv_proj"]["lr"] == pytest.approx(1.5e-4)
    assert by_name["cross_kv_proj"]["weight_decay"] == 0.0
    assert by_name["cross_kv_proj"]["clip_norm"] == pytest.approx(1.0)
    kv_names = by_name["cross_kv_proj"]["param_names"]
    assert any(".cross_k_proj." in n for n in kv_names)
    assert any(".cross_v_proj." in n for n in kv_names)
    assert any("encoder_projection.pre_norm" in n for n in kv_names)
    assert not any(n.endswith(".cross_scale") for n in kv_names)  # frozen by default

    # encoder_lora & decoder_lora
    assert by_name["encoder_lora"]["lr"] == pytest.approx(8e-5)
    assert by_name["encoder_lora"]["weight_decay"] == 0.0
    assert by_name["encoder_lora"]["clip_norm"] == pytest.approx(1.0)
    assert all(n.startswith("encoder.") for n in by_name["encoder_lora"]["param_names"])

    assert by_name["decoder_lora"]["lr"] == pytest.approx(4e-5)
    assert by_name["decoder_lora"]["weight_decay"] == 0.0
    assert by_name["decoder_lora"]["clip_norm"] == pytest.approx(1.0)
    assert all(n.startswith("decoder.") for n in by_name["decoder_lora"]["param_names"])

    # When cross_scale IS explicitly learnable, it goes to cross_kv_proj (1.5e-4), NEVER cross_gate_scalars!
    model_scale, policy_scale = _tiny_model("4base-2it", learnable_cross_scale=True)
    groups_scale, _ = configure_matrix_stage(
        model_scale, policy_scale, freeze_decoder=False, total_steps=100, warmup_offset=6
    )
    by_name_scale = {g["name"]: g for g in groups_scale}
    assert not any("cross_scale" in n for n in by_name_scale["cross_gate_scalars"]["param_names"])
    assert any(n.endswith(".cross_scale") for n in by_name_scale["cross_kv_proj"]["param_names"])


def test_consolidated_optimizer_builder_parity_and_decoder_only_apples_to_apples():
    # 1. objectives.build_bifocalgemma_optimizer and stage.configure_matrix_stage share ONE canonical partition
    model, policy = _tiny_model("2base-2it")
    stage_groups, _ = configure_matrix_stage(
        model, policy, freeze_decoder=False, total_steps=100, warmup_offset=6
    )
    stage_opt = build_bifocalgemma_adamw(stage_groups, policy=policy)
    obj_opt = build_bifocalgemma_optimizer(model)

    assert [g["name"] for g in obj_opt.param_groups] == [g["name"] for g in stage_opt.param_groups]
    for g_obj, g_stage in zip(obj_opt.param_groups, stage_opt.param_groups, strict=True):
        assert g_obj["param_names"] == g_stage["param_names"]
        assert g_obj["lr"] == pytest.approx(g_stage["lr"])
        assert g_obj["weight_decay"] == pytest.approx(g_stage["weight_decay"])
        assert g_obj["clip_norm"] == g_stage["clip_norm"]
        assert g_obj["betas"] == (0.9, 0.95)
        assert g_stage["betas"] == (0.9, 0.95)

    # 2. Override kwargs (fast_pipeline compatibility) apply cleanly to canonical groups
    override_opt = build_bifocalgemma_optimizer(
        model, lr_bridge=3e-4, lr_cross=1.2e-4, lr_lora=5e-5, weight_decay_bridge=2e-4
    )
    ov = {g["name"]: g for g in override_opt.param_groups}
    assert ov["bridge_mlp"]["lr"] == pytest.approx(3e-4)
    assert ov["bridge_mlp"]["weight_decay"] == pytest.approx(2e-4)
    assert ov["cross_kv_proj"]["lr"] == pytest.approx(1.2e-4)
    assert ov["encoder_lora"]["lr"] == pytest.approx(5e-5)
    assert ov["decoder_lora"]["lr"] == pytest.approx(5e-5)
    assert ov["cross_gate_scalars"]["lr"] == pytest.approx(2e-3)
    assert ov["cross_gate_proj"]["lr"] == pytest.approx(5e-4)

    # 3. 100% apples-to-apples decoder_lora parity between Bifocal and decoder_only baseline SFT
    sym_policy = unified_recipe("symmetric")
    asym_policy = unified_recipe("asymmetric")
    base_policy = unified_recipe("decoder_only")
    assert sym_policy["optimizer"] == asym_policy["optimizer"] == base_policy["optimizer"]
    assert base_policy["optimizer"]["betas"] == [0.9, 0.95]
    for key in ("lr", "weight_decay", "clip_norm", "warmup_ratio", "scheduler", "min_lr_ratio"):
        assert (
            sym_policy["groups"]["decoder_lora"][key]
            == asym_policy["groups"]["decoder_lora"][key]
            == base_policy["groups"]["decoder_lora"][key]
        )

    base_model, _ = _tiny_model("e2b-it")
    base_groups, _ = configure_matrix_stage(
        base_model, base_policy, freeze_decoder=False, total_steps=100
    )
    assert [g["name"] for g in base_groups] == ["decoder_lora"]
    assert base_groups[0]["lr"] == pytest.approx(4e-5)
    assert base_groups[0]["clip_norm"] == pytest.approx(1.0)


def test_per_group_grad_clipping_isolates_gate_scalars_and_gate_proj_from_bridge():
    model, policy = _tiny_model("2base-2it")
    groups, _ = configure_matrix_stage(
        model, policy, freeze_decoder=False, total_steps=100, warmup_offset=6
    )
    opt = build_bifocalgemma_adamw(groups, policy=policy)
    by_name = {g["name"]: g for g in opt.param_groups}

    # Assign huge gradients to bridge_mlp & cross_kv_proj, small gradients to cross_gate_proj,
    # and large (norm=5.0) gradients to unclipped cross_gate_scalars.
    def _set_group_norm(group: dict[str, Any], target_norm: float) -> None:
        total_elems = sum(p.numel() for p in group["params"])
        val = target_norm / math.sqrt(total_elems)
        for p in group["params"]:
            p.grad = torch.full_like(p, val)

    _set_group_norm(by_name["bridge_mlp"], 100.0)
    _set_group_norm(by_name["cross_kv_proj"], 50.0)
    _set_group_norm(by_name["cross_gate_proj"], 0.25)
    _set_group_norm(by_name["cross_gate_scalars"], 5.0)
    _set_group_norm(by_name["encoder_lora"], 0.5)
    _set_group_norm(by_name["decoder_lora"], 0.4)

    total_pre_norm, reported = clip_optimizer_group_grads(opt, default_clip_norm=1.0)
    assert reported["bridge_mlp"] == pytest.approx(100.0, rel=1e-4)
    assert reported["cross_gate_proj"] == pytest.approx(0.25, rel=1e-4)
    assert reported["cross_gate_scalars"] == pytest.approx(5.0, rel=1e-4)
    assert total_pre_norm == pytest.approx(
        math.sqrt(sum(v * v for v in reported.values())), rel=1e-5
    )

    def _actual_norm(group: dict[str, Any]) -> float:
        return math.sqrt(sum(float(p.grad.pow(2).sum().item()) for p in group["params"]))

    # bridge_mlp & cross_kv_proj clipped to 1.0
    assert _actual_norm(by_name["bridge_mlp"]) == pytest.approx(1.0, rel=1e-4)
    assert _actual_norm(by_name["cross_kv_proj"]) == pytest.approx(1.0, rel=1e-4)
    # cross_gate_proj UNTOUCHED (not starved by bridge_mlp's 100.0 norm!)
    assert _actual_norm(by_name["cross_gate_proj"]) == pytest.approx(0.25, rel=1e-4)
    # cross_gate_scalars UNCLIPPED (clip_norm=None -> norm stays 5.0!)
    assert _actual_norm(by_name["cross_gate_scalars"]) == pytest.approx(5.0, rel=1e-4)
    # LoRA groups untouched (below 1.0)
    assert _actual_norm(by_name["encoder_lora"]) == pytest.approx(0.5, rel=1e-4)
    assert _actual_norm(by_name["decoder_lora"]) == pytest.approx(0.4, rel=1e-4)


# ---------------------------------------------------------------------------
# 3. Short 6% Warm-Start, 0% Freeze Asymmetric Warmup, Stage-2 Carry-Over Fix, num_epochs
# ---------------------------------------------------------------------------


def test_training_contract_eliminates_50_50_freeze_and_supports_asymmetric_and_multi_epoch(
    tmp_path,
):
    paired_spec = make_experiment_spec("2base-2it")
    base_spec = make_experiment_spec("e2b-it")

    # Default paired contract: ~6% warm-start -> ~94% joint (NOT 50/50 = 5551!)
    default_contract = resolve_training_contract(paired_spec)
    assert default_contract["total_steps"] == 11102
    assert default_contract["stage1_steps"] == round(11102 * 0.06)  # 666
    assert default_contract["stage2_steps"] == 11102 - 666  # 10436
    assert default_contract["schedule_mode"] == "short_warmstart_joint"
    assert default_contract["num_epochs"] == 1

    # Explicit stage1_fraction=0.08
    c_8pct = resolve_training_contract(paired_spec, total_steps=1000, stage1_fraction=0.08)
    assert c_8pct["stage1_steps"] == 80
    assert c_8pct["stage2_steps"] == 920

    # 0% freeze asymmetric-warmup mode (via stage1_steps=0 or stage1_fraction=0.0)
    c_zero_steps = resolve_training_contract(paired_spec, total_steps=1000, stage1_steps=0)
    c_zero_frac = resolve_training_contract(paired_spec, total_steps=1000, stage1_fraction=0.0)
    assert c_zero_steps["stage1_steps"] == 0 and c_zero_steps["stage2_steps"] == 1000
    assert c_zero_steps["schedule_mode"] == "asymmetric_warmup_joint"
    assert c_zero_frac["stage1_steps"] == 0 and c_zero_frac["schedule_mode"] == "asymmetric_warmup_joint"

    # Baseline decoder_only always uses full horizon in stage1
    base_contract = resolve_training_contract(base_spec, total_steps=1000)
    assert base_contract["stage1_steps"] == 1000 and base_contract["stage2_steps"] == 0
    assert base_contract["schedule_mode"] == "decoder_only"

    # Multi-epoch support (1, 1.5, 2) on local pool
    pool = tmp_path / "pool"
    pool.mkdir()
    manifest: dict[str, Any] = {
        "format": "bifocalgemma_curated_pool_v1",
        "record_ids": {"train": [f"id-{i}" for i in range(40)]},
    }
    manifest["fingerprint"] = stable_fingerprint(manifest)
    (pool / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    c_ep1 = resolve_training_contract(paired_spec, pool_directory=pool, num_epochs=1)
    c_ep15 = resolve_training_contract(paired_spec, pool_directory=pool, num_epochs=1.5)
    c_ep2 = resolve_training_contract(paired_spec, pool_directory=pool, num_epochs=2)
    assert c_ep1["total_steps"] == 10  # 40 / 4
    assert c_ep15["total_steps"] == 15  # 60 / 4
    assert c_ep2["total_steps"] == 20  # 80 / 4
    assert c_ep15["num_epochs"] == 1.5
    assert c_ep2["num_epochs"] == 2


def test_stage2_alignment_groups_never_inherit_decayed_stage1_lr_wsd_and_cosine():
    model, policy = _tiny_model("2base-2it")

    # Test both 6% warm-start AND extreme 50% Stage-1 split to prove zero Stage-1 decay carry-over!
    for stage1_steps in (6, 50):
        # A. Default WSD schedule on alignment groups
        groups_s1, _ = configure_matrix_stage(
            model, policy, freeze_decoder=True, total_steps=100, warmup_offset=0
        )
        groups_s2, _ = configure_matrix_stage(
            model, policy, freeze_decoder=False, total_steps=100, warmup_offset=stage1_steps
        )
        opt1 = build_bifocalgemma_adamw(groups_s1, policy=policy)
        opt2 = build_bifocalgemma_adamw(groups_s2, policy=policy)

        r_s1_end = update_policy_learning_rates(opt1, stage1_steps - 1)
        r_s2_start = update_policy_learning_rates(opt2, stage1_steps)
        r_s2_next = update_policy_learning_rates(opt2, stage1_steps + 1)
        r_final = update_policy_learning_rates(opt2, 99)

        for name in ALIGNMENT_GROUP_NAMES:
            peak = policy["groups"][name]["lr"]
            # Flat-top at end of Stage 1 AND exact peak at start of Stage 2 (ZERO decay carry-over!)
            assert r_s1_end[name] == pytest.approx(peak)
            assert r_s2_start[name] == pytest.approx(peak)
            # Final step decays smoothly to min_lr_ratio * peak (0.1 * peak)
            assert r_final[name] == pytest.approx(peak * 0.1)

        # Decoder LoRA warms up cleanly from stage1_steps and ends at 0.1 * peak
        dec_peak = policy["groups"]["decoder_lora"]["lr"]
        assert 0.0 < r_s2_start["decoder_lora"] < r_s2_next["decoder_lora"]
        assert r_final["decoder_lora"] == pytest.approx(dec_peak * 0.1)

        # B. Explicit cosine schedule override on alignment groups -> clean Stage-2 cosine reset!
        cosine_policy = copy.deepcopy(policy)
        for name in ALIGNMENT_GROUP_NAMES:
            cosine_policy["groups"][name]["scheduler"] = "cosine"
        g_cos_s2, _ = configure_matrix_stage(
            model, cosine_policy, freeze_decoder=False, total_steps=100, warmup_offset=stage1_steps
        )
        opt_cos2 = build_bifocalgemma_adamw(g_cos_s2, policy=cosine_policy)
        r_cos_start = update_policy_learning_rates(opt_cos2, stage1_steps)
        r_cos_mid = update_policy_learning_rates(opt_cos2, (stage1_steps + 99) // 2)
        r_cos_end = update_policy_learning_rates(opt_cos2, 99)

        for name in ALIGNMENT_GROUP_NAMES:
            peak = cosine_policy["groups"][name]["lr"]
            assert r_cos_start[name] == pytest.approx(peak)  # Exact peak, NO Stage-1 decay!
            assert peak * 0.1 < r_cos_mid[name] < peak
            assert r_cos_end[name] == pytest.approx(peak * 0.1)


def test_zero_freeze_asymmetric_warmup_mode():
    model, policy = _tiny_model("2base-2it")
    groups, summary = configure_matrix_stage(
        model, policy, freeze_decoder=False, total_steps=100, warmup_offset=0
    )
    assert summary["freeze_decoder"] is False
    assert summary["asymmetric_warmup"] is True
    by_name = {g["name"]: g for g in groups}

    # Alignment groups warm up fast (3% = 3 steps); decoder_lora uses delayed 10% = 10 steps!
    for name in ALIGNMENT_GROUP_NAMES:
        assert by_name[name]["warmup_ratio"] == pytest.approx(0.03)
        assert by_name[name]["warmup_steps"] == 3
    assert by_name["encoder_lora"]["warmup_ratio"] == pytest.approx(0.05)
    assert by_name["encoder_lora"]["warmup_steps"] == 5
    assert by_name["decoder_lora"]["warmup_ratio"] == pytest.approx(0.10)
    assert by_name["decoder_lora"]["warmup_steps"] == 10

    opt = build_bifocalgemma_adamw(groups, policy=policy)
    r_step2 = update_policy_learning_rates(opt, 2)  # step 3 of 100 -> alignment at 100%, decoder at 30%
    r_step9 = update_policy_learning_rates(opt, 9)  # step 10 of 100 -> decoder reaches 100%
    for name in ALIGNMENT_GROUP_NAMES:
        assert r_step2[name] == pytest.approx(policy["groups"][name]["lr"])
    assert r_step2["decoder_lora"] == pytest.approx(policy["groups"]["decoder_lora"]["lr"] * 0.3)
    assert r_step9["decoder_lora"] == pytest.approx(policy["groups"]["decoder_lora"]["lr"])


# ---------------------------------------------------------------------------
# 4. Task-Gated diff_marker_v1 (Zero Leakage) & Per-Example / Length-Capped Loss
# ---------------------------------------------------------------------------


def test_diff_marker_v1_strictly_gated_to_repair_unified_diffs_only():
    add_id = DEFAULT_DIFF_ADD_TOKEN_IDS[0]
    del_id = DEFAULT_DIFF_DEL_TOKEN_IDS[0]
    hunk_id = DEFAULT_DIFF_HUNK_HDR_TOKEN_IDS[0]
    diff_word_id = DEFAULT_DIFF_HDR_TOKEN_IDS[0]  # "diff" word, NOT a hunk header
    ctx_id = 42

    # Batch with 6 diverse tasks, ALL containing '+' / '-' / '@@' tokens:
    # Row 0: repair + unified diff -> MUST get differential weights (add > del > hdr > ctx)
    # Row 1: completion -> MUST be strictly uniform 1.0
    # Row 2: localization -> MUST be strictly uniform 1.0
    # Row 3: qa -> MUST be strictly uniform 1.0
    # Row 4: math -> MUST be strictly uniform 1.0
    # Row 5: repair + full-function Python (has '+'/'-'/'diff' variable, NO hunk header) -> MUST be uniform 1.0
    labels = torch.tensor(
        [
            [hunk_id, add_id, del_id, ctx_id, -100],
            [hunk_id, add_id, del_id, ctx_id, -100],
            [hunk_id, add_id, del_id, ctx_id, -100],
            [hunk_id, add_id, del_id, ctx_id, -100],
            [hunk_id, add_id, del_id, ctx_id, -100],
            [diff_word_id, add_id, del_id, ctx_id, -100],
        ],
        dtype=torch.long,
    )
    task_types = ["repair", "completion", "localization", "qa", "math", "repair"]

    weights = compute_diff_token_weights(labels, task_types=task_types)

    # Row 0: true unified-diff repair -> differential weighting normalized to mean 1.0
    assert float(weights[0, :4].mean().item()) == pytest.approx(1.0, rel=1e-5)
    assert float(weights[0, 1]) > float(weights[0, 2]) > float(weights[0, 0]) > float(weights[0, 3])
    assert float(weights[0, 4]) == 0.0

    # Rows 1..4 (non-repair) AND Row 5 (non-diff full-function repair): exact 1.0 on all valid tokens!
    for row in range(1, 6):
        torch.testing.assert_close(
            weights[row, :4], torch.ones(4, dtype=torch.float32), rtol=0, atol=0
        )
        assert float(weights[row, 4]) == 0.0

    # Explicit is_unified_diff=False overrides even when repair tokens contain hunk markers
    weights_explicit_false = compute_diff_token_weights(
        labels[:1], task_types=["repair"], is_unified_diff=[False]
    )
    torch.testing.assert_close(
        weights_explicit_false[0, :4], torch.ones(4, dtype=torch.float32), rtol=0, atol=0
    )


def test_per_example_mean_and_length_capped_eliminate_short_target_starvation():
    # Row 0: short 10-token target; Row 1: long 500-token target (50:1 length ratio)
    torch.manual_seed(7)
    vocab = 32
    d_model = 8
    short_len, long_len = 10, 500
    labels = torch.full((2, long_len), -100, dtype=torch.long)
    labels[0, :short_len] = 1
    labels[1, :long_len] = 2

    # Construct logits where Row 0 has exact CE = 4.0/tok and Row 1 has exact CE = 1.0/tok
    # by testing with a tiny linear head where exact per-row mean CE is known.
    hidden = torch.randn(2, long_len, d_model, dtype=torch.float32)
    weight = torch.randn(vocab, d_model, dtype=torch.float32)
    logits = torch.nn.functional.linear(hidden, weight)

    row0_ce = float(
        torch.nn.functional.cross_entropy(logits[0, :short_len], labels[0, :short_len]).item()
    )
    row1_ce = float(
        torch.nn.functional.cross_entropy(logits[1, :long_len], labels[1, :long_len]).item()
    )
    assert abs(row0_ce - row1_ce) > 0.05

    # 1. per_example_mean -> exact 50/50 average of Row 0 and Row 1 (NO 50:1 starvation!)
    loss_per_ex = float(
        chunked_differential_cross_entropy(
            hidden, weight, labels, chunk_size=64, loss_reduction="per_example_mean"
        ).item()
    )
    loss_per_ex_dense = float(
        differential_diff_cross_entropy(
            logits, labels, loss_reduction="per_example_mean"
        ).item()
    )
    expected_per_ex = 0.5 * (row0_ce + row1_ce)
    assert loss_per_ex == pytest.approx(expected_per_ex, rel=1e-5)
    assert loss_per_ex_dense == pytest.approx(expected_per_ex, rel=1e-5)

    # 2. length_capped (cap=256) -> weights are 10 and 256 (NOT 10 and 500!)
    w_cap = compute_example_loss_weights(labels, reduction="length_capped", length_cap=256)
    assert w_cap.tolist() == [10.0, 256.0]
    loss_capped = float(
        chunked_differential_cross_entropy(
            hidden, weight, labels, chunk_size=64, loss_reduction="length_capped", length_cap=256
        ).item()
    )
    expected_capped = (10.0 * row0_ce + 256.0 * row1_ce) / (10.0 + 256.0)
    assert loss_capped == pytest.approx(expected_capped, rel=1e-5)

    # 3. legacy token_mean -> weights are 10 and 500
    loss_tok = float(
        chunked_differential_cross_entropy(
            hidden, weight, labels, chunk_size=64, loss_reduction="token_mean"
        ).item()
    )
    expected_tok = (10.0 * row0_ce + 500.0 * row1_ce) / 510.0
    assert loss_tok == pytest.approx(expected_tok, rel=1e-5)


@pytest.mark.parametrize("loss_reduction", ["per_example_mean", "length_capped"])
def test_matrix_stage_microbatch_invariance_under_heterogeneous_lengths(tmp_path, loss_reduction):
    """(batch_size=1, accum=4) and (batch_size=2, accum=2) produce identical updates on heterogeneous lengths."""
    examples, schedule, _ = _synthetic_cohort(
        "2base-2it",
        count=4,
        target_lengths=[3, 18, 4, 22],
        tasks=["repair", "completion", "qa", "math"],
    )
    model_a, policy = _tiny_model("2base-2it")
    model_b, _ = _tiny_model("2base-2it")
    model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))

    trace_a: list[dict[str, Any]] = []
    trace_b: list[dict[str, Any]] = []
    common: dict[str, Any] = dict(
        policy=policy,
        max_steps=1,
        freeze_decoder=False,
        start_global_step=0,
        compatibility={"model_key": "2base-2it"},
        shared_schedule=schedule,
        diff_token_ids=TINY_DIFF_IDS,
        loss_reduction=loss_reduction,
        length_cap=8,
    )
    run_matrix_stage(
        model_a,
        examples,
        [],
        "joint",
        batch_size=1,
        accumulation=4,
        run_dir=tmp_path / "bs1",
        progress=trace_a.append,
        **common,
    )
    run_matrix_stage(
        model_b,
        examples,
        [],
        "joint",
        batch_size=2,
        accumulation=2,
        run_dir=tmp_path / "bs2",
        progress=trace_b.append,
        **common,
    )

    assert trace_a[0]["train_loss"] == pytest.approx(trace_b[0]["train_loss"], rel=1e-5)
    assert set(trace_a[0]["group_gradient_norms"]) == set(CANONICAL_PARAM_GROUP_ORDER)
    for g_name in CANONICAL_PARAM_GROUP_ORDER:
        assert trace_a[0]["group_gradient_norms"][g_name] == pytest.approx(
            trace_b[0]["group_gradient_norms"][g_name], rel=1e-3, abs=1e-5
        )
    for name, param_a in model_a.state_dict().items():
        diff = (param_a.float() - model_b.state_dict()[name].float()).abs()
        # On Adam step 1, Δw = -lr * sign(g), so a single ~1e-7 rounding sign flip near zero
        # can differ by up to 2*lr; RMS diff across every tensor must remain tiny (< 1e-4).
        rms = float(diff.pow(2).mean().sqrt().item())
        assert rms < 1e-4, f"Micro-batch RMS drift too large on {name}: {rms}"


def test_matrix_stage_multi_epoch_and_zero_freeze_end_to_end(tmp_path):
    """End-to-end 1.5-epoch run in 0%-freeze asymmetric-warmup mode executes all 6 groups cleanly."""
    examples, _, raw_records = _synthetic_cohort("2base-2it", count=8)
    multi_sched = _build_schedule_for_epochs(
        raw_records,
        dataset_identity={"fixture": "v4_multi_epoch"},
        seed=17,
        effective_batch=4,
        num_epochs=1.5,
    )
    assert multi_sched.total_examples == 12
    assert multi_sched.total_updates == 3

    model, policy = _tiny_model("2base-2it")
    trace: list[dict[str, Any]] = []
    result = run_matrix_stage(
        model,
        examples,
        [],
        "joint",
        policy=policy,
        max_steps=3,
        freeze_decoder=False,
        run_dir=tmp_path / "multi_epoch_zero_freeze",
        compatibility={"model_key": "2base-2it"},
        start_global_step=0,
        batch_size=2,
        accumulation=2,
        shared_schedule=multi_sched,
        diff_token_ids=TINY_DIFF_IDS,
        progress=trace.append,
        total_steps=3,
        asymmetric_warmup=True,
    )
    assert result["completed_steps"] == 3
    assert len(trace) == 3
    for row in trace:
        assert set(row["group_gradient_norms"]) == set(CANONICAL_PARAM_GROUP_ORDER)
        assert set(row["learning_rates"]) == set(CANONICAL_PARAM_GROUP_ORDER)
