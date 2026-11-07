"""Tests for the E2B->E2B 1/4-budget Colab pilot builder and staged freeze logic."""

import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch

from bifocalgemma.config import get_v2_experiment
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.models.v2.native import build_native_tiny_model
from bifocalgemma.repository_study import RepositoryFile, RepositorySnapshot
from bifocalgemma.training.policy import configure_policy_model
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.v2_diagnostics import stratified_subset
from tools.notebook_bundle import source_bundle_bytes

ROOT = Path(__file__).resolve().parents[1]


class _DummyTokenizer:
    bos_token_id = 2

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        assert not tokenize and not enable_thinking
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = [2] if add_special_tokens or text.startswith("<user>") else []
        tokens.extend(3 + (ord(ch) % 90) for ch in text[:160])
        if not text.endswith("<assistant>"):
            tokens.append(106)
        return tokens

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return "decoded_" + "_".join(str(int(t)) for t in token_ids[:8])


@pytest.fixture
def pilot_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("prepare_colab_pilot")


def test_stratified_raw_subset_matches_tokenized_stratified_subset_order(pilot_module):
    raw_rows = [
        {
            "instance_id": f"case-{i}",
            "source": "swe_fixer" if i % 2 == 0 else "swe_rebench_v2",
            "task_type": ("repair", "localization", "completion")[i % 3],
        }
        for i in range(30)
    ]
    tokenized = [
        TokenizedRepairExample(
            row["instance_id"],
            (2, 5, 6),
            (2, 7, 8),
            (9, 106),
            metadata={"source": row["source"], "task_type": row["task_type"]},
        )
        for row in raw_rows
    ]
    raw_selected = pilot_module.stratified_raw_subset(raw_rows, 11, seed=17)
    tok_selected = stratified_subset(tokenized, 11, seed=17)
    assert [r["instance_id"] for r in raw_selected] == [x.instance_id for x in tok_selected]


def test_span_denoising_and_counterfactual_identifier_probes_prevent_leakage(pilot_module):
    sample_context = "\n".join(
        [
            "### File: pkg/solver.py",
            "def compute_checksum(records, multiplier):",
            "    running_accumulator = 0",
            "    for item_entry in records:",
            "        weighted_step = item_entry * multiplier",
            "        running_accumulator += weighted_step",
            "    final_digest = running_accumulator % 997",
            "    return final_digest",
        ]
    )
    rows = [
        {
            "instance_id": f"row-{idx}",
            "source": "swe_rebench_v2",
            "task_type": "repair",
            "instruction": "Fix modulo overflow in compute_checksum.",
            "context": sample_context,
            "target_patch": "diff --git a/pkg/solver.py b/pkg/solver.py\n",
        }
        for idx in range(4)
    ]
    extracted = pilot_module.extract_resolvable_span(sample_context, "row-0", seed=17)
    assert extracted is not None
    corrupted, target_span, start, stop = extracted
    assert corrupted.count("# <MISSING_CODE>") == 1
    assert target_span not in corrupted
    assert stop - start >= 2

    spec = get_v2_experiment("BifocalGemma-E2B-E2B")
    tok = _DummyTokenizer()
    ul2_examples, counts = pilot_module.build_span_denoising_examples(rows, spec, tok, tok, seed=17)
    assert len(ul2_examples) == 4
    assert counts["constructed_span_mask"] == 4
    assert all(ex.metadata["task_type"] == "span_denoising" for ex in ul2_examples)

    probes = pilot_module.build_source_copy_probes(rows, spec, tok, tok, count=2, seed=17)
    assert len(probes) == 2
    for probe in probes:
        renamed_id = probe["renamed_identifier"]
        orig_id = probe["original_identifier"]
        assert renamed_id.startswith("bf_id_")
        assert renamed_id not in probe["first_line"]
        assert renamed_id not in probe["last_line"]
        assert renamed_id in probe["renamed_target"]
        assert orig_id not in probe["renamed_target"]


def test_frozen_decoder_stage_trains_encoder_and_alignment_then_joint_unfreezes(
    pilot_module, tmp_path
):
    torch.manual_seed(23)
    spec = get_v2_experiment("BifocalGemma-E2B-E2B")
    model = build_native_tiny_model(spec)
    policy = unified_recipe("symmetric")
    policy.update(rank=2, alpha=4)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(
        model,
        {
            "query_chunk_size": 4,
            "loss_chunk_size": 4,
            "gradient_checkpointing": True,
        },
    )
    data = [
        TokenizedRepairExample(
            f"ex-{i}",
            (2, 11 + i, 12 + i, 13 + i),
            (2, 21, 22),
            (31 + (i % 3), 32, 106),
            metadata={"source": "swe_fixer", "task_type": "repair"},
        )
        for i in range(4)
    ]
    decoder_before = {
        n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("decoder.")
    }
    stage1 = pilot_module.run_pilot_stage(
        model,
        data,
        data[:2],
        tmp_path / "stage1",
        policy=copy.deepcopy(policy),
        max_steps=2,
        freeze_decoder=True,
        batch_size=1,
        accumulation=2,
        seed=17,
    )
    assert stage1["status"] == "COMPLETED"
    assert stage1["stage_summary"]["active_groups"] == ["encoder", "alignment"]
    assert stage1["stage_summary"]["decoder_trainable_tensors"] == 0
    assert stage1["max_decoder_lora_abs_delta"] == 0.0
    assert stage1["max_encoder_or_alignment_abs_delta"] > 0.0
    for name, before_tensor in decoder_before.items():
        torch.testing.assert_close(model.get_parameter(name), before_tensor, atol=0, rtol=0)

    stage2 = pilot_module.run_pilot_stage(
        model,
        data,
        data[:2],
        tmp_path / "stage2",
        policy=copy.deepcopy(policy),
        max_steps=2,
        freeze_decoder=False,
        batch_size=1,
        accumulation=2,
        seed=1017,
    )
    assert stage2["status"] == "COMPLETED"
    assert stage2["stage_summary"]["active_groups"] == ["encoder", "decoder", "alignment"]
    assert stage2["max_decoder_lora_abs_delta"] > 0.0
    assert stage2["max_encoder_or_alignment_abs_delta"] > 0.0


def test_normalize_and_recount_patch_rescues_plain_unified_diff_with_bad_hunk_count(pilot_module):
    snapshot = RepositorySnapshot(
        repo="humanevalfix",
        base_commit="c0",
        files=(
            RepositoryFile(
                "solution.py",
                "def add(a, b):\n    return a - b\n",
            ),
        ),
    )
    raw_output = "```diff\n--- a/solution.py\n+++ b/solution.py\n@@ -1,5 +1,5 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n```"
    canonical = pilot_module.normalize_and_recount_patch(raw_output, snapshot)
    assert canonical is not None
    assert canonical.startswith("diff --git a/solution.py b/solution.py\n")
    assert "+    return a + b\n" in canonical


def test_exported_pilot_script_contains_all_four_arms_and_no_hf_writes(pilot_module):
    manifest = json.loads((ROOT / "notebooks/runtime-manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source_bundle_bytes(ROOT)).hexdigest() != manifest["sha256"]:
        pytest.skip("Runtime bundle is stale after the V2/V3 split; republish before exporting.")
    rendered = pilot_module.render(steps_per_arm=120, humaneval_limit=8)
    tree = ast.parse(rendered)
    worker = next(
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_text"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)
        and "def pilot_e2b_e2b(" in n.args[0].value
    )
    ast.parse(worker)
    for required in (
        "direct_joint",
        "frozen_then_joint",
        "span_ul2_then_joint",
        "it_encoder_frozen_then_joint",
        "full_cross_bf16_frozen_then_joint",
        "configure_full_rank_cross_attention",
        "build_full_cross_cohorts",
        "build_span_denoising_examples",
        "build_source_copy_probes",
        "evaluate_humaneval_subset",
        "_steps = int(_env_steps) if _env_steps else 120",
        "pilot_e2b_e2b(root, steps_per_arm=_steps",
    ):
        assert required in worker
    for forbidden in ("create_commit(", "upload_file(", "upload_folder("):
        assert forbidden not in worker
    assert "bifocalgemma-test.zip" in rendered


def test_full_rank_cross_attention_stage_training_and_cached_equivalence(pilot_module, tmp_path):
    from bifocalgemma.inference.generation import greedy_decode

    torch.manual_seed(29)
    spec = get_v2_experiment("BifocalGemma-E2B-E2B")
    model = build_native_tiny_model(spec)
    policy = unified_recipe("symmetric")
    policy.update(rank=2, alpha=4)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(
        model,
        {
            "query_chunk_size": 4,
            "loss_chunk_size": 4,
            "gradient_checkpointing": True,
        },
    )
    info = pilot_module.configure_full_rank_cross_attention(model)
    assert info["mode"] == "full_rank_cross_bf16"
    assert info["kv_owner_layers"] == 15
    assert info["total_decoder_layers"] == 35

    data = [
        TokenizedRepairExample(
            f"ex-{i}",
            (2, 11 + i, 12 + i, 13 + i),
            (2, 21, 22),
            (31 + (i % 3), 32, 106),
            metadata={"source": "swe_fixer", "task_type": "repair"},
        )
        for i in range(4)
    ]
    causal_lora_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith("decoder.") and not pilot_module.is_cross_bridge_parameter(n)
    }
    cross_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if ".cross_k_proj." in n or ".cross_v_proj." in n or ".cross_gate" in n
    }
    assert len(cross_before) > 0

    stage1 = pilot_module.run_pilot_stage(
        model,
        data,
        data[:2],
        tmp_path / "stage1_full_cross",
        policy=copy.deepcopy(policy),
        max_steps=2,
        freeze_decoder=True,
        batch_size=1,
        accumulation=2,
        seed=17,
    )
    assert stage1["status"] == "COMPLETED"
    assert stage1["stage_summary"]["decoder_trainable_tensors"] == 0
    assert stage1["max_decoder_lora_abs_delta"] == 0.0
    for name, before_tensor in causal_lora_before.items():
        torch.testing.assert_close(model.get_parameter(name), before_tensor, atol=0, rtol=0)
    cross_deltas = [
        float((model.get_parameter(n).detach() - before_t).abs().max())
        for n, before_t in cross_before.items()
    ]
    assert max(cross_deltas) > 0.0

    stage2 = pilot_module.run_pilot_stage(
        model,
        data,
        data[:2],
        tmp_path / "stage2_full_cross",
        policy=copy.deepcopy(policy),
        max_steps=2,
        freeze_decoder=False,
        batch_size=1,
        accumulation=2,
        seed=1017,
    )
    assert stage2["status"] == "COMPLETED"
    assert stage2["max_decoder_lora_abs_delta"] > 0.0

    gen_cached = greedy_decode(
        model, (2, 15, 16, 17), (2, 21, 22), max_new_tokens=4, use_cache=True
    )
    gen_uncached = greedy_decode(
        model, (2, 15, 16, 17), (2, 21, 22), max_new_tokens=4, use_cache=False
    )
    assert gen_cached.token_ids == gen_uncached.token_ids
