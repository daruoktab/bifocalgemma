"""Regression tests for v4 milestone pruning, weights_only=True, fast_pipeline resume, and data/patch/release fixes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bifocalgemma.config import (
    V4_COLLECTION,
    V4_DATASET_REVISION,
    V4_STAGE1_STEPS,
    V4_TOTAL_STEPS,
)
from bifocalgemma.data.patches import patch_paths
from bifocalgemma.data.releases import publish_release
from bifocalgemma.training.checkpoint import (
    prune_matrix_checkpoints,
    restore_matrix_checkpoint,
    save_matrix_checkpoint,
    synchronize_matrix,
)
from bifocalgemma.training.fast_pipeline import FastTrainingConfig, FastTrainingPipeline
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.stage import configure_matrix_stage


def _write_ready_ckpt(train_dir: Path, step: int) -> Path:
    ckpt = train_dir / f"checkpoint-{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "READY.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    (ckpt / "adapters.safetensors").write_bytes(b"")
    (ckpt / "trainer_state.pt").write_bytes(b"")
    (ckpt / "manifest.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    return ckpt


def test_v4_milestone_checkpoints_survive_pruning_and_hub_retention(tmp_path: Path, monkeypatch):
    """V4 stage1 (899) and total (4494) checkpoints must survive Stage 2 pruning and Hub sync."""
    train_dir = tmp_path / "training"
    for step in (899, 1000, 2000, 3000, 4494):
        _write_ready_ckpt(train_dir, step)

    kept = prune_matrix_checkpoints(
        train_dir,
        keep_limit=2,
        stage1_steps=V4_STAGE1_STEPS,
        total_steps=V4_TOTAL_STEPS,
    )
    assert kept == ["checkpoint-000899", "checkpoint-003000", "checkpoint-004494"]
    assert (train_dir / "checkpoint-000899").is_dir()
    assert not (train_dir / "checkpoint-001000").exists()
    assert not (train_dir / "checkpoint-002000").exists()
    assert (train_dir / "checkpoint-004494").is_dir()

    # Also verify synchronize_matrix retains 000899 & 004494 when given v4 horizons
    import huggingface_hub

    import bifocalgemma.training.checkpoint as ckpt_mod

    (tmp_path / "pipeline_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ckpt_mod, "validate_checkpoint_files", lambda p: None)

    uploaded_files: list[str] = []

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

        def model_info(self, repo_id):
            return SimpleNamespace(sha="deadbeef")

        def list_repo_files(self, *args, **kwargs):
            return list(uploaded_files)

        def create_commit(self, *args, **kwargs):
            for op in kwargs.get("operations", []):
                path_in_repo = getattr(op, "path_in_repo", None)
                if path_in_repo and path_in_repo not in uploaded_files:
                    uploaded_files.append(path_in_repo)
            return SimpleNamespace(oid="abc1234", commit_id="abc1234")

    def _fake_hf_hub_download(repo_id, filename, **kwargs):
        rel = filename.removeprefix("2base-2it/test/train/")
        return str(tmp_path / rel)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)
    res = synchronize_matrix(
        "daruokta/fake-repo",
        tmp_path,
        "2base-2it/test/train",
        token="hf_test",
        stage1_steps=V4_STAGE1_STEPS,
        total_steps=V4_TOTAL_STEPS,
    )
    assert res["retained_checkpoints"] == [
        "checkpoint-000899",
        "checkpoint-003000",
        "checkpoint-004494",
    ]


def test_matrix_checkpoint_roundtrip_weights_only_true(tmp_path: Path):
    """save_matrix_checkpoint + restore_matrix_checkpoint must roundtrip under weights_only=True."""
    from bifocalgemma.config import make_experiment_spec
    from bifocalgemma.modeling_bifocalgemma import NativeTinyModel
    from bifocalgemma.models.cross_attention import configure_full_rank_cross_attention
    from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
    from bifocalgemma.training.policy import configure_policy_model

    spec = make_experiment_spec("2base-2it")
    policy = unified_recipe("symmetric")
    model = NativeTinyModel(spec, tiny=True)
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    groups, _ = configure_matrix_stage(model, policy, freeze_decoder=True, total_steps=10)
    optimizer = torch.optim.AdamW(groups, lr=1e-4)

    # Run one step so AdamW state (exp_avg, exp_avg_sq, step) is populated
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad()

    ckpt_dir = tmp_path / "checkpoint-000001"
    save_matrix_checkpoint(
        model,
        optimizer,
        ckpt_dir,
        global_step=1,
        stage_name="frozen_decoder",
        compatibility={"recipe_version": "test"},
        training_state={"step": 1},
    )
    restored = restore_matrix_checkpoint(
        model,
        optimizer,
        ckpt_dir,
        expected_compatibility={"recipe_version": "test"},
    )
    assert restored["global_step"] == 1


def test_fast_pipeline_resume_identity_mismatch_and_corrupt_checkpoint(tmp_path: Path):
    """try_resume_checkpoint must reject mismatched resume_identity and corrupt local checkpoints."""
    out_dir = tmp_path / "fast_resume"
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        output_dir=str(out_dir),
        total_steps=10,
        stage1_steps=5,
        stage2_steps=5,
    )
    p1 = FastTrainingPipeline(cfg)
    p1.configure_stage1()
    ckpt_file = p1.save_checkpoint(3, "stage1_bridge_warmup")

    # Mismatched total_steps must raise unless force_fresh=True
    cfg_mismatch = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        output_dir=str(out_dir),
        total_steps=20,
        stage1_steps=10,
        stage2_steps=10,
    )
    with pytest.raises(ValueError, match="resume_identity mismatch"):
        FastTrainingPipeline(cfg_mismatch).try_resume_checkpoint()

    # force_fresh=True bypasses resume cleanly
    cfg_fresh = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        output_dir=str(out_dir),
        total_steps=20,
        stage1_steps=10,
        stage2_steps=10,
        force_fresh=True,
    )
    assert FastTrainingPipeline(cfg_fresh).try_resume_checkpoint() == 0

    # Corrupt local checkpoint must raise instead of silently returning 0
    ckpt_file.write_bytes(b"not a valid torch checkpoint")
    with pytest.raises(ValueError, match="Corrupt local checkpoint"):
        FastTrainingPipeline(cfg).try_resume_checkpoint()


def test_patch_paths_ignores_hunk_body_dash_and_plus_prefixes():
    """Deleted SQL/Lua comments ('-- foo' -> '--- foo') or YAML '--- ' inside hunks must not fail."""
    patch = (
        "diff --git a/src/query.sql b/src/query.sql\n"
        "--- a/src/query.sql\n"
        "+++ b/src/query.sql\n"
        "@@ -1,3 +1,3 @@\n"
        "--- old sql comment\n"
        "+++ new line\n"
        " SELECT 1;\n"
    )
    assert patch_paths(patch) == ("src/query.sql",)

    bad_header_patch = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/other.py\n"
        "+++ b/src/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    with pytest.raises(ValueError, match="patch_header_path_mismatch"):
        patch_paths(bad_header_patch)


def test_publish_release_replace_current_deletes_audits(tmp_path: Path, monkeypatch):
    """publish_release(..., replace_current=True) must include audits/** in delete_patterns."""
    import huggingface_hub

    import bifocalgemma.data.releases as rel_mod

    (tmp_path / "inventory.json").write_text(json.dumps(["release.json"]), encoding="utf-8")
    monkeypatch.setattr(rel_mod, "verify_release", lambda d: {"fingerprint": "fp123"})

    captured = {}

    class _FakeHfApi:
        def whoami(self):
            return {"name": "daruokta"}

        def repo_exists(self, *args, **kwargs):
            return True

        def dataset_info(self, repo_id, revision=None):
            return SimpleNamespace(private=True, sha="a" * 40)

        def upload_folder(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(oid="b" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    publish_release(tmp_path, "daruokta/test-ds", replace_current=True)
    assert "audits/**" in captured["delete_patterns"]
    assert "collections/**" in captured["delete_patterns"]
    assert "notices/**" in captured["delete_patterns"]


def test_dataset_release_lock_matches_v4():
    lock_path = Path(__file__).resolve().parents[1] / "configs/dataset_release.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["revision"] == V4_DATASET_REVISION
    assert lock["ready_training_collection"] == V4_COLLECTION


def test_matrix_resume_accepts_token_identical_prompt_source_reformat(tmp_path: Path, monkeypatch):
    """Formatting-only data_fingerprint shifts must resume only when targets and schedule match."""
    import bifocalgemma.training.matrix as mat

    src = Path(mat.__file__).read_text(encoding="utf-8")
    assert "def _compatibility_matches(" in src
    from typing import Any

    ns: dict[str, Any] = {"Any": Any}
    block_start = src.index("    def _compatibility_matches(")
    block_end = src.index("    resume_ckpt = latest_checkpoint(train_dir)", block_start)
    import textwrap

    exec(textwrap.dedent(src[block_start:block_end]), ns)
    matches = ns["_compatibility_matches"]

    expected = {
        "recipe_version": "bifocalgemma_shared_schedule_2026_11_12",
        "data_fingerprint": "a" * 64,
        "matched_target_fingerprints": {"train": "1" * 64, "dev": "2" * 64},
        "schedule_signature": "3" * 64,
        "training_semantics_sha256": "5" * 64,
    }
    assert matches(dict(expected), expected) is True
    assert matches({**expected, "data_fingerprint": "b" * 64}, expected) is True
    assert matches({**expected, "data_fingerprint": "tampered"}, expected) is False
    assert (
        matches(
            {
                **expected,
                "data_fingerprint": "b" * 64,
                "matched_target_fingerprints": {"train": "9" * 64, "dev": "2" * 64},
            },
            expected,
        )
        is False
    )
    assert (
        matches(
            {**expected, "data_fingerprint": "b" * 64, "schedule_signature": "9" * 64}, expected
        )
        is False
    )
