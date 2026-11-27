"""Cold recovery must accept the latest atomic checkpoint-only receipt."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from bifocalgemma.data.curation import file_digest
from bifocalgemma.training.hub import recover_remote_run, synchronize_run
from bifocalgemma.training.stages import atomic_json, retained_checkpoints


@pytest.mark.parametrize(
    ("remote_case", "expected"),
    [
        ("matching", True),
        ("index_only", False),
        ("zero_byte", False),
        ("corrupt_digest", False),
        ("missing_shard", False),
        ("missing_local", False),
        ("metadata_error", False),
    ],
)
def test_standalone_export_requires_matching_remote_contents(
    tmp_path, monkeypatch, remote_case, expected
):
    import torch
    from safetensors.torch import save_file

    from bifocalgemma.training.matrix import _standalone_export_exists

    export = tmp_path / "pretrained_model"
    export.mkdir()
    atomic_json(export / "config.json", {"model_type": "bifocalgemma"})
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for index, name in enumerate(shards):
        save_file({f"weight{index}": torch.tensor([index])}, str(export / name))
    atomic_json(
        export / "model.safetensors.index.json",
        {"weight_map": {f"weight{index}": name for index, name in enumerate(shards)}},
    )

    siblings = []
    for path in export.iterdir():
        data = path.read_bytes()
        # Exercise both Hub LFS SHA-256 and ordinary Git blob metadata.
        is_weight = path.suffix == ".safetensors"
        blob = hashlib.sha1(usedforsecurity=False)
        blob.update(f"blob {len(data)}\0".encode("ascii") + data)
        siblings.append(
            SimpleNamespace(
                rfilename=f"latest/pretrained_model/{path.name}",
                size=len(data),
                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if is_weight else None,
                blob_id=blob.hexdigest(),
            )
        )
    if remote_case == "index_only":
        siblings = [entry for entry in siblings if entry.rfilename.endswith(".index.json")]
    elif remote_case == "zero_byte":
        next(entry for entry in siblings if entry.rfilename.endswith(shards[0])).size = 0
    elif remote_case == "corrupt_digest":
        next(entry for entry in siblings if entry.rfilename.endswith(shards[0])).lfs.sha256 = (
            "0" * 64
        )
    elif remote_case == "missing_shard":
        siblings = [entry for entry in siblings if not entry.rfilename.endswith(shards[1])]
    elif remote_case == "missing_local":
        (export / shards[0]).unlink()

    class Api:
        def __init__(self, *, token):
            assert token == "fixture"

        def model_info(self, repo, *, files_metadata):
            assert repo == "fixture/model" and files_metadata is True
            if remote_case == "metadata_error":
                raise RuntimeError("metadata unavailable")
            return SimpleNamespace(siblings=siblings)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert (
        _standalone_export_exists(
            export, upload=True, token="fixture", repo="fixture/model", subfolder="latest"
        )
        is expected
    )


def checkpoint(root, step):
    path = root / "training" / f"checkpoint-{step:06}"
    path.mkdir(parents=True)
    (path / "adapters.safetensors").write_bytes(b"fixture-adapter-" + str(step).encode())
    (path / "trainer_state.pt").write_bytes(b"fixture-state-" + str(step).encode())
    atomic_json(
        path / "manifest.json", {"format": "bifocalgemma_notebook_sft_v2", "global_step": step}
    )
    atomic_json(
        path / "READY.json",
        {
            name: file_digest(path / name)
            for name in ("adapters.safetensors", "trainer_state.pt", "manifest.json")
        },
    )


def test_full_then_checkpoint_only_upload_refreshes_receipt_for_cold_resume(tmp_path, monkeypatch):
    remote = {}
    commits = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def repo_exists(self, *args, **kwargs):
            return True

        def create_repo(self, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(sha="a" * 40)

        def list_repo_files(self, *args, **kwargs):
            return list(remote)

        def create_commit(self, **kwargs):
            commits.append(kwargs["operations"])
            for op in kwargs["operations"]:
                if isinstance(op, huggingface_hub.CommitOperationDelete):
                    remote.pop(op.path_in_repo)
                else:
                    value = op.path_or_fileobj
                    remote[op.path_in_repo] = (
                        Path(value).read_bytes() if isinstance(value, str) else value.getvalue()
                    )
            return SimpleNamespace(oid="b" * 40)

    def download(repo, filename, **kwargs):
        path = tmp_path / "download" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    root = tmp_path / "original"
    checkpoint(root, 1)
    checkpoint(root, 2)
    atomic_json(root / "pipeline_state.json", {"stage": "running", "step": 2})
    history = root / "training/history.jsonl"
    history.write_text('{"step":2}\n', encoding="utf-8")
    prefix = "symmetric/BifocalGemma-E2B-E2B/fixture"
    synchronize_run("fixture/model", root, prefix, token="fixture")
    checkpoint(root, 3)
    retained_checkpoints(root / "training", limit=2)
    history.write_text('{"step":2}\n{"step":3}\n', encoding="utf-8")
    atomic_json(root / "pipeline_state.json", {"stage": "running", "step": 3})
    synchronize_run("fixture/model", root, prefix, token="fixture", checkpoint_only=True)
    receipt = json.loads(remote[prefix + "/upload_complete.json"])
    assert receipt["checkpoint_only"] is True
    assert receipt["files"]["training/history.jsonl"] == file_digest(history)
    assert not any("checkpoint-000001/" in name for name in remote)
    recovered = tmp_path / "recovered"
    recover_remote_run("fixture/model", recovered, prefix, token="fixture")
    assert (recovered / "training/history.jsonl").read_bytes() == history.read_bytes()
    assert json.loads((recovered / "pipeline_state.json").read_text())["step"] == 3
    assert len(commits) == 2


def test_recover_remote_run_tolerates_interrupted_recovery_and_rejects_local_only(
    tmp_path, monkeypatch
):
    remote = {}

    class Api:
        def __init__(self, token=None):
            pass

        def repo_exists(self, repo):
            return True

        def model_info(self, repo):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_files(self, repo, revision=None):
            return sorted(remote)

    def download(repo, filename, **kwargs):
        path = tmp_path / "dl" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    prefix = "latest/BifocalGemma-E2B-E2B-2it-2it/train"
    remote[f"{prefix}/benchmarks/bench_a/predictions.jsonl"] = b'{"id": 1}\n{"id": 2}\n'
    remote[f"{prefix}/last_seen.json"] = b'{"step": 10}\n'
    remote[f"{prefix}/pipeline_state.json"] = b'{"stage": "done"}\n'

    # Case 1: Interrupted legacy recovery (benchmarks/ + last_seen.json on disk, no pipeline_state.json)
    partial_old = tmp_path / "partial_old"
    (partial_old / "benchmarks/bench_a").mkdir(parents=True)
    (partial_old / "benchmarks/bench_a/predictions.jsonl").write_bytes(b'{"id": 1}\n')
    (partial_old / "last_seen.json").write_bytes(b'{"step": 9}\n')
    recover_remote_run("fixture/model", partial_old, prefix, token="fixture")
    assert (partial_old / "pipeline_state.json").read_bytes() == remote[
        f"{prefix}/pipeline_state.json"
    ]
    assert (partial_old / "benchmarks/bench_a/predictions.jsonl").read_bytes() == remote[
        f"{prefix}/benchmarks/bench_a/predictions.jsonl"
    ]
    assert not (partial_old / ".recovery_in_progress.json").exists()

    # Case 2: Interrupted atomic recovery with marker and temp artifact
    partial_new = tmp_path / "partial_new"
    (partial_new / "benchmarks/bench_a").mkdir(parents=True)
    (partial_new / ".recovery_in_progress.json").write_text('{"revision": "c"}', encoding="utf-8")
    (partial_new / "benchmarks/bench_a/.recover-abc123.tmp").write_bytes(b"partial")
    recover_remote_run("fixture/model", partial_new, prefix, token="fixture")
    assert (partial_new / "pipeline_state.json").exists()
    assert not (partial_new / ".recovery_in_progress.json").exists()
    assert not (partial_new / "benchmarks/bench_a/.recover-abc123.tmp").exists()

    # Case 3: Genuine unuploaded local-only artifact still raises ValueError
    local_only = tmp_path / "local_only"
    (local_only / "training/checkpoint-000999").mkdir(parents=True)
    (local_only / "training/checkpoint-000999/adapters.safetensors").write_bytes(b"local")
    with pytest.raises(ValueError, match="without a pipeline receipt"):
        recover_remote_run("fixture/model", local_only, prefix, token="fixture")

    # Case 4: Larger local predictions.jsonl without recovery marker still raises ValueError
    unuploaded_pred = tmp_path / "unuploaded_pred"
    (unuploaded_pred / "benchmarks/bench_a").mkdir(parents=True)
    (unuploaded_pred / "benchmarks/bench_a/predictions.jsonl").write_bytes(
        b'{"id": 1}\n{"id": 2}\n{"id": 3}\n'
    )
    with pytest.raises(ValueError, match="without a pipeline receipt"):
        recover_remote_run("fixture/model", unuploaded_pred, prefix, token="fixture")


def test_reusable_training_report_backfills_missing_final_stage_when_verified(tmp_path):
    import torch
    from safetensors.torch import save_file

    from bifocalgemma.training import matrix
    from bifocalgemma.training.schedule import (
        build_shared_schedule,
        canonical_schedule_records,
        persist_shared_schedule,
    )
    from bifocalgemma.training.source_identity import source_fingerprints
    from bifocalgemma.training.stages import StageLedger

    spec = matrix.make_experiment_spec("4it-4it")
    sem = source_fingerprints()["training_semantics_sha256"]
    resolved_total, resolved_stage1, resolved_stage2 = 11102, 5551, 5551
    schedule = build_shared_schedule(
        canonical_schedule_records(
            [
                {
                    "instance_id": "item-0",
                    "instruction": "Fix 0",
                    "context": "def f(): pass",
                    "target_text": "return 0",
                    "source": "fixture",
                    "task_type": "repair",
                    "token_lengths": {"encoder": 16},
                }
            ]
        ),
        dataset_identity={"fixture": "recovery"},
    )
    schedule_sig = schedule.signature
    repo = "daruokta/bifocalgemma-4it-4it"
    subfolder = "latest"

    def _build_run(
        name: str,
        *,
        ckpt_stage: str = "joint",
        ledger_failed: bool = False,
        export_mode: str = "valid",
        schedule_mode: str = "valid",
    ):
        root = tmp_path / name
        train_dir = root / "training"
        train_dir.mkdir(parents=True)
        if schedule_mode == "valid":
            persist_shared_schedule(train_dir / "shared_schedule.json", schedule)
        elif schedule_mode == "tampered_records":
            persist_shared_schedule(train_dir / "shared_schedule.json", schedule)
            sched_payload = json.loads(
                (train_dir / "shared_schedule.json").read_text(encoding="utf-8")
            )
            sched_payload["ordered_examples"][0]["instance_id"] = "tampered-item-0"
            atomic_json(train_dir / "shared_schedule.json", sched_payload)
        else:
            atomic_json(train_dir / "shared_schedule.json", {"signature": schedule_sig})
        ckpt_dir = train_dir / f"checkpoint-{resolved_total:06d}"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapters.safetensors").write_bytes(b"adapter-bytes")
        (ckpt_dir / "trainer_state.pt").write_bytes(b"state-bytes")
        atomic_json(
            ckpt_dir / "manifest.json",
            {
                "format": "bifocalgemma_full_matrix_v1",
                "recipe_version": matrix.CLEAN_RECIPE_VERSION,
                "global_step": resolved_total,
                "stage": ckpt_stage,
                "compatibility": {
                    "recipe_version": matrix.CLEAN_RECIPE_VERSION,
                    "experiment": spec.name,
                    "model_key": "4it-4it",
                    "model_spec": spec.to_dict(),
                    "training_semantics_sha256": sem,
                    "schedule_signature": schedule_sig,
                    "total_steps": resolved_total,
                    "stage1_steps": resolved_stage1,
                    "stage2_steps": resolved_stage2,
                },
            },
        )
        atomic_json(
            ckpt_dir / "READY.json",
            {
                file_name: file_digest(ckpt_dir / file_name)
                for file_name in ("adapters.safetensors", "trainer_state.pt", "manifest.json")
            },
        )
        export_dir = root / "pretrained_model"
        export_dir.mkdir(parents=True)
        if export_mode == "valid":
            atomic_json(export_dir / "config.json", {"model_type": "bifocalgemma", "d_enc": 32})
            save_file({"weight": torch.zeros(1)}, str(export_dir / "model.safetensors"))
        elif export_mode == "zero_byte":
            atomic_json(export_dir / "config.json", {"model_type": "bifocalgemma", "d_enc": 32})
            (export_dir / "model.safetensors").write_bytes(b"")
        else:
            (export_dir / "model.safetensors").write_bytes(b"weights")
        rep_file = train_dir / "training_report.json"
        atomic_json(
            rep_file,
            {
                "status": "COMPLETED",
                "export_error": None,
                "recipe_version": matrix.CLEAN_RECIPE_VERSION,
                "model_key": "4it-4it",
                "spec": spec.to_dict(),
                "repo": repo,
                "subfolder": subfolder,
                "global_step": resolved_total,
                "training_semantics_sha256": sem,
                "schedule_signature": schedule_sig,
                "stage2": None,
            },
        )
        identity = {"training_semantics_sha256": sem, "model_key": "4it-4it"}
        ledger = StageLedger(root, identity)
        if ledger_failed:
            ledger.fail("stage2_joint", RuntimeError("stage2 failed"))
        else:
            atomic_json(ledger.path, ledger.state)
        return root, train_dir, rep_file, ledger, identity

    def _check_reuse(run_tuple):
        run_root, run_train, run_rep, run_ledger, run_ident = run_tuple
        return matrix._reusable_training_report(
            root=run_root,
            train_dir=run_train,
            rep_file=run_rep,
            ledger=run_ledger,
            identity=run_ident,
            spec=spec,
            model_key="4it-4it",
            repo=repo,
            subfolder=subfolder,
            resolved_total=resolved_total,
            resolved_stage1=resolved_stage1,
            resolved_stage2=resolved_stage2,
            force_fresh=False,
            standalone_repo_id=None,
            upload=False,
            token=None,
        )

    # Missing stage2_joint receipt is backfilled and training report is reused
    run1 = _build_run("missing_stage2")
    assert not run1[3].completed("stage2_joint")
    res1 = _check_reuse(run1)
    assert res1 is not None and res1["status"] == "COMPLETED"
    assert run1[3].completed("stage2_joint")

    # Wrong checkpoint stage rejects reuse and does not backfill
    run2 = _build_run("wrong_stage", ckpt_stage="frozen_decoder")
    assert _check_reuse(run2) is None
    assert not run2[3].completed("stage2_joint")

    # Explicitly failed stage2_joint rejects reuse and does not backfill
    run3 = _build_run("failed_stage", ledger_failed=True)
    assert _check_reuse(run3) is None
    assert not run3[3].completed("stage2_joint")

    # Hardened checks reject stubbed/tampered schedule manifests and 0-byte/corrupt safetensors exports
    run4 = _build_run("bad_schedule", schedule_mode="stub")
    assert _check_reuse(run4) is None
    assert not run4[3].completed("stage2_joint")
    run5 = _build_run("tampered_schedule_records", schedule_mode="tampered_records")
    assert _check_reuse(run5) is None
    assert not run5[3].completed("stage2_joint")
    run6 = _build_run("bad_export", export_mode="corrupt")
    assert _check_reuse(run6) is None
    assert not run6[3].completed("stage2_joint")
    run7 = _build_run("zero_byte_export", export_mode="zero_byte")
    assert _check_reuse(run7) is None
    assert not run7[3].completed("stage2_joint")
