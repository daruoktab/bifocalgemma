"""Comprehensive verification tests for the 8 Molab matrix training notebooks and runner."""

import ast
import base64
import hashlib
import io
import json
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
from bifocalgemma.training.policy import configure_policy_model  # noqa: E402
from bifocalgemma.training.recipes import unified_recipe  # noqa: E402
from bifocalgemma.training.stages import StageLedger  # noqa: E402
from tools.molab_matrix_support import (  # noqa: E402
    MATRIX_MODELS,
    STAGE1_STEPS,
    STAGE2_STEPS,
    TOTAL_STEPS,
    configure_full_rank_cross_attention,
    configure_matrix_stage,
    is_cross_bridge_parameter,
    make_experiment_spec,
    patch_matrix_tokenization,
    prune_matrix_checkpoints,
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
        "symmetric": True,
        "repo": "daruokta/bifocalgemma-v2-2base-2it",
    },
    "2it-2it": {
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "symmetric": True,
        "repo": "daruokta/bifocalgemma-v2-2it-2it",
    },
    "4base-4it": {
        "encoder_id": "google/gemma-4-E4B",
        "encoder_rev": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "symmetric": True,
        "repo": "daruokta/bifocalgemma-v2-4base-4it",
    },
    "4it-4it": {
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "symmetric": True,
        "repo": "daruokta/bifocalgemma-v2-4it-4it",
    },
    "4base-2it": {
        "encoder_id": "google/gemma-4-E4B",
        "encoder_rev": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "symmetric": False,
        "repo": "daruokta/bifocalgemma-v2-4base-2it",
    },
    "4it-2it": {
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "symmetric": False,
        "repo": "daruokta/bifocalgemma-v2-4it-2it",
    },
    "2base-4it": {
        "encoder_id": "google/gemma-4-E2B",
        "encoder_rev": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "symmetric": False,
        "repo": "daruokta/bifocalgemma-v2-2base-4it",
    },
    "2it-4it": {
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "symmetric": False,
        "repo": "daruokta/bifocalgemma-v2-2it-4it",
    },
}

FORBIDDEN_BASELINE_REPOS = [
    "daruokta/bifocalgemma-v2-e2b-it",
    "daruokta/bifocalgemma-v2-e4b-it",
]


def test_matrix_models_definition_matches_expected():
    assert len(MATRIX_MODELS) == 8
    for key, expected in EXPECTED_MODELS.items():
        assert key in MATRIX_MODELS
        spec_def = MATRIX_MODELS[key]
        assert spec_def["encoder_id"] == expected["encoder_id"]
        assert spec_def["encoder_revision"] == expected["encoder_rev"]
        assert spec_def["decoder_id"] == expected["decoder_id"]
        assert spec_def["decoder_revision"] == expected["decoder_rev"]
        assert (spec_def["family"] == "symmetric") == expected["symmetric"]
        assert spec_def["repo"] == expected["repo"]


def test_make_experiment_spec_validates_all_eight_models():
    for key, expected in EXPECTED_MODELS.items():
        spec = make_experiment_spec(key)
        assert key in spec.name
        assert spec.encoder.model_id == expected["encoder_id"]
        assert spec.encoder.revision == expected["encoder_rev"]
        assert spec.decoder.model_id == expected["decoder_id"]
        assert spec.decoder.revision == expected["decoder_rev"]


def test_matrix_curriculum_stages():
    assert TOTAL_STEPS == 11102
    assert STAGE1_STEPS == 5551
    assert STAGE2_STEPS == 5551
    assert STAGE1_STEPS + STAGE2_STEPS == TOTAL_STEPS


@pytest.mark.parametrize("model_key", list(EXPECTED_MODELS.keys()))
def test_notebook_exists_and_is_valid_ast(model_key):
    filename = f"train_{model_key.replace('-', '_')}.py"
    notebook_path = ROOT / "notebooks" / filename
    assert notebook_path.is_file(), f"Notebook {filename} must exist"

    code = notebook_path.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(notebook_path))
    assert isinstance(tree, ast.Module)


@pytest.mark.parametrize("model_key", list(EXPECTED_MODELS.keys()))
def test_notebook_has_correct_metadata_and_hf_repo(model_key):
    filename = f"train_{model_key.replace('-', '_')}.py"
    notebook_path = ROOT / "notebooks" / filename
    code = notebook_path.read_text(encoding="utf-8")
    expected = EXPECTED_MODELS[model_key]

    # Verify model key assignment
    assert f'MODEL_KEY = "{model_key}"' in code
    # Verify HF repo assignment
    assert f'HF_REPO_ID = "{expected["repo"]}"' in code
    # Verify backbone IDs
    assert expected["encoder_id"] in code
    assert expected["encoder_rev"] in code
    assert expected["decoder_id"] in code
    assert expected["decoder_rev"] in code

    # Verify no accidental targeting of baseline repos
    for forbidden in FORBIDDEN_BASELINE_REPOS:
        assert f'HF_REPO_ID = "{forbidden}"' not in code


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

    # Decode and check hash
    raw_zip = base64.b85decode(bundle_val)
    computed_sha = hashlib.sha256(raw_zip).hexdigest()
    assert computed_sha == bundle_sha

    # Verify zip content
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


@pytest.mark.parametrize("model_key", ["2base-2it", "4base-2it", "2base-4it"])
def test_configure_full_rank_cross_attention_specs(model_key):
    spec = make_experiment_spec(model_key)
    family = MATRIX_MODELS[model_key]["family"]
    policy = unified_recipe(family)
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    info = configure_full_rank_cross_attention(model)
    assert info["mode"] == "full_rank_cross_bf16"
    assert info["kv_owner_layers"] > 0
    assert info["total_cross_bridge_parameters"] > 0
    if spec.encoder.size != spec.decoder.size:
        assert hasattr(model, "encoder_projection")
        assert any("encoder_projection" in n for n, _ in model.named_parameters())


def test_configure_matrix_stage_parameter_freezing():
    spec = make_experiment_spec("4base-2it")
    policy = unified_recipe("asymmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)

    # Stage 1: Frozen decoder
    _, summary1 = configure_matrix_stage(model, policy, freeze_decoder=True, total_steps=5551)
    assert summary1["freeze_decoder"] is True
    assert summary1["active_groups"] == ["encoder", "alignment"]
    assert summary1["decoder_trainable_tensors"] == 0
    for name, param in model.named_parameters():
        if name.startswith("decoder.") and not is_cross_bridge_parameter(name):
            assert not param.requires_grad

    # Stage 2: Joint SFT
    _, summary2 = configure_matrix_stage(model, policy, freeze_decoder=False, total_steps=5551)
    assert summary2["freeze_decoder"] is False
    assert summary2["active_groups"] == ["encoder", "decoder", "alignment"]
    assert summary2["decoder_trainable_tensors"] > 0
    decoder_trainable = [
        name
        for name, p in model.named_parameters()
        if name.startswith("decoder.") and not is_cross_bridge_parameter(name) and p.requires_grad
    ]
    assert len(decoder_trainable) > 0


def test_save_and_restore_matrix_checkpoint_and_prune(tmp_path):
    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    groups, _ = configure_matrix_stage(model, policy, freeze_decoder=True, total_steps=5551)
    optimizer = torch.optim.AdamW(groups)

    # Save milestone 5551
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

    # Save additional checkpoints
    save_matrix_checkpoint(model, optimizer, tmp_path / "checkpoint-006000", 6000, "joint", {}, {})
    save_matrix_checkpoint(model, optimizer, tmp_path / "checkpoint-007000", 7000, "joint", {}, {})
    save_matrix_checkpoint(model, optimizer, tmp_path / "checkpoint-011102", 11102, "joint", {}, {})

    # Restore checkpoint-005551
    restored = restore_matrix_checkpoint(model, optimizer, p1)
    assert restored["global_step"] == 5551
    assert restored["stage"] == "frozen_decoder"
    assert restored["training_state"]["step"] == 5551

    # Prune checkpoints: keep milestone 5551 and 11102, plus last 2 (7000 and 11102)
    kept = prune_matrix_checkpoints(tmp_path, keep_limit=2)
    assert kept == ["checkpoint-005551", "checkpoint-007000", "checkpoint-011102"]
    assert not (tmp_path / "checkpoint-006000").exists()


def test_run_matrix_stage_execution_and_frozen_decoder_invariance(tmp_path):
    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)

    examples = [
        TokenizedRepairExample(f"ex_{i}", (10 + i, 20 + i), (30 + i, 40 + i), (50 + i, 60 + i))
        for i in range(8)
    ]
    decoder_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.startswith("decoder.") and not is_cross_bridge_parameter(n)
    }

    # Run Stage 1 (frozen decoder)
    rep1 = run_matrix_stage(
        model,
        examples,
        examples[:2],
        stage_name="frozen_decoder",
        policy=policy,
        max_steps=2,
        freeze_decoder=True,
        run_dir=tmp_path,
        compatibility={"v": 1},
        start_global_step=0,
        batch_size=1,
        accumulation=2,
    )
    assert rep1["completed_steps"] == 2
    assert rep1["end_global_step"] == 2

    # Verify frozen decoder parameters are bit-identical
    for n, p_before in decoder_before.items():
        p_after = model.get_parameter(n)
        torch.testing.assert_close(p_after, p_before, rtol=0, atol=0)

    # Run Stage 2 (joint SFT)
    rep2 = run_matrix_stage(
        model,
        examples,
        examples[:2],
        stage_name="joint",
        policy=policy,
        max_steps=2,
        freeze_decoder=False,
        run_dir=tmp_path,
        compatibility={"v": 1},
        start_global_step=2,
        batch_size=1,
        accumulation=2,
    )
    assert rep2["completed_steps"] == 2
    assert rep2["end_global_step"] == 4


def test_cached_vs_uncached_greedy_decode_equivalence():
    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)

    gen_uncached = greedy_decode(model, (10, 20), (30, 40), max_new_tokens=4, use_cache=False)
    gen_cached = greedy_decode(model, (10, 20), (30, 40), max_new_tokens=4, use_cache=True)
    assert gen_uncached.token_ids == gen_cached.token_ids


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


def test_matrix_tokenization_patch_it_encoder_contract():
    """Verify that IT encoders (which lack BOS in raw encode()) have BOS prepended properly."""
    patch_matrix_tokenization()

    import bifocalgemma.data.context_tasks as ct_mod
    import bifocalgemma.evaluation.context as eval_ctx_mod

    class MockITEncoder:
        bos_token_id = 2

        def encode(self, text, add_special_tokens=True):
            # IT tokenizers return raw tokens without BOS when encode() is called
            return [101, 102, 103]

    class MockBaseEncoder:
        bos_token_id = 2

        def encode(self, text, add_special_tokens=True):
            # Base tokenizers include BOS at index 0
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

    # 1. Test IT encoder (e.g. 4it-2it) - BOS must be prepended
    spec_it = make_experiment_spec("4it-2it")
    res_it = ct_mod.tokenize_context_task(
        row, spec_it, MockITEncoder(), MockDecoder(), encoder_limit=1000, decoder_limit=1000
    )
    assert res_it.encoder_ids[0] == 2
    assert res_it.encoder_ids == (2, 101, 102, 103)
    assert res_it.decoder_prefix_ids[0] == 2
    assert res_it.target_ids[-1] == 107

    # 2. Test Base encoder (e.g. 4base-2it) - BOS must not be duplicated
    spec_base = make_experiment_spec("4base-2it")
    res_base = ct_mod.tokenize_context_task(
        row, spec_base, MockBaseEncoder(), MockDecoder(), encoder_limit=1000, decoder_limit=1000
    )
    assert res_base.encoder_ids[0] == 2
    assert res_base.encoder_ids == (2, 101, 102, 103)

    # 3. Test prompt_ids in evaluation/context.py
    eval_row = {
        "role": "eval",
        "instruction": "Fix bug",
        "context": "def foo(): pass",
    }
    enc, prefix = eval_ctx_mod.prompt_ids(
        eval_row, spec_it, MockITEncoder(), MockDecoder(), encoder_limit=1000
    )
    assert enc[0] == 2
    assert enc == (2, 101, 102, 103)
