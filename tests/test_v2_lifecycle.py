import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx2
import huggingface_hub
import pytest
import torch

from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.models.v2.native import build_native_tiny_model
from bifocalgemma.training import v2
from bifocalgemma.training.benchmark_upload import BenchmarkUploadQueue
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.stages import atomic_json
from bifocalgemma.training.v2_diagnostics import stratified_subset
from bifocalgemma.training.v2_hub import RemoteHeartbeat, synchronize_v2
from tests.test_offline_sft_training import TinyLossModel, examples


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("key", list(v2.MODELS))
def test_one_model_training_stops_and_completed_resume_does_not_reload(tmp_path, monkeypatch, key):
    calls = []
    remote_events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])

    class Dashboard:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, event):
            pass

    monkeypatch.setattr(v2, "TrainingDashboard", Dashboard)

    class Heartbeat:
        def __init__(self, repo, *args, **kwargs):
            self.repo = repo

        def __call__(self, event, *, force=False):
            remote_events.append((self.repo, dict(event), force))

    monkeypatch.setattr(v2, "RemoteHeartbeat", Heartbeat)
    monkeypatch.setattr(v2, "recover_remote_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        v2,
        "resolve_shared_profile",
        lambda *args, **kwargs: {
            "batch_size": 1,
            "accumulation": 4,
            "evidence_sha256": "f" * 64,
            "memory_policy": {"query_chunk_size": 3, "loss_chunk_size": 2},
        },
    )
    monkeypatch.setattr(
        v2,
        "synchronize_v2",
        lambda *args, **kwargs: {
            "revision": "b" * 40,
            "retained_checkpoints": ["checkpoint-000001"],
        },
    )

    def factory(spec, *args):
        calls.append(spec.name)
        return build_native_tiny_model(spec), {"source_identity_verified": True}

    monkeypatch.setattr(v2, "download_model", factory)
    monkeypatch.setattr(v2, "fetch_collection", lambda *a, **kw: None)
    data = [
        replace(
            x, instance_id=f"fixture-{i}", encoder_ids=x.encoder_ids if "-it" not in key else ()
        )
        for i, x in enumerate(examples())
    ]
    monkeypatch.setattr(
        v2,
        "prepare_curated_training",
        lambda *a, **kw: (data, data[:2], {"fingerprint": "fixture"}),
    )

    def execution(model, *a, **kw):
        policy = configure_memory_efficient_training(
            model, {"query_chunk_size": 3, "loss_chunk_size": 2}
        )
        return {"memory_policy": policy, "batch_size": 1, "accumulation": 4, "candidates": []}

    monkeypatch.setattr(v2, "select_execution", execution)

    def forbidden_benchmark(*args, **kwargs):
        raise AssertionError("Training must never launch benchmarks.")

    monkeypatch.setattr(v2, "run_context_benchmark", forbidden_benchmark)
    result = v2.run_v2(
        key, mode="train", token="test", upload=True, output_root=tmp_path, cache=tmp_path / "cache"
    )
    assert result["global_step"] == 1 and result["benchmark_auto_started"] is False
    assert len(calls) == 1
    repo, event, forced = remote_events[-1]
    assert repo == "daruokta/bifocalgemma-v2-" + key
    assert forced and event["stage"] == "training_completed"
    assert event["final_synced"] and not event["upload_pending"]
    again = v2.run_v2(
        key, mode="train", token="test", upload=True, output_root=tmp_path, cache=tmp_path / "cache"
    )
    assert again["status"] == "COMPLETED" and len(calls) == 1
    assert remote_events[-1][1]["stage"] == "training_completed" and remote_events[-1][2]
    manifest = next(tmp_path.glob("*/train/training/checkpoint-*/manifest.json"))
    assert json.loads(manifest.read_text())["format"] == "bifocalgemma_normalized_sft_v3"


@pytest.mark.parametrize("corrupt", [False, True])
def test_snapshot_verifies_replacement_before_pruning(tmp_path, monkeypatch, corrupt):
    root = tmp_path / "run"
    fit_offline_sft(
        TinyLossModel(),
        examples(),
        examples()[:1],
        root / "training",
        {},
        max_steps=3,
        epochs=3,
        batch_size=1,
        accumulation=1,
        save_every=1,
        eval_every=10,
        event_callback=lambda e: None,
    )
    atomic_json(root / "pipeline_state.json", {"fixture": True})
    prefix = "symmetric/BifocalGemma-E2B-E2B/fixture"
    old = prefix + "/training/checkpoint-000001/READY.json"
    remote = {old: b"old"}
    events = []

    class Api:
        def __init__(self, **kw):
            pass

        def create_repo(self, *a, **kw):
            pass

        def model_info(self, *a, **kw):
            return SimpleNamespace(sha="a" * 40)

        def list_repo_files(self, *a, **kw):
            return list(remote)

        def create_commit(self, *a, **kw):
            ops = kw["operations"]
            for op in ops:
                if isinstance(op, huggingface_hub.CommitOperationDelete):
                    events.append("delete")
                    remote.pop(op.path_in_repo)
                else:
                    value = op.path_or_fileobj
                    remote[op.path_in_repo] = (
                        value if isinstance(value, bytes) else Path(value).read_bytes()
                    )
            return SimpleNamespace(oid="b" * 40)

    def download(repo, name, **kw):
        events.append("verified")
        target = tmp_path / "downloaded.json"
        target.write_bytes(b"wrong" if corrupt else remote[name])
        return str(target)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    if corrupt:
        with pytest.raises(ValueError, match="marker mismatch"):
            synchronize_v2("fixture/repo", root, prefix, token="test")
        assert old in remote and "delete" not in events
    else:
        result = synchronize_v2("fixture/repo", root, prefix, token="test")
        assert result["retained_checkpoints"] == ["checkpoint-000002", "checkpoint-000003"]
        assert events == ["verified", "verified", "delete"]
        receipt = json.loads(remote[prefix + "/upload_complete.json"])
        assert receipt["files"]["pipeline_state.json"]


def test_pilot_subset_is_deterministic_and_stratified():
    rows = [
        replace(
            examples()[0], instance_id=str(i), metadata={"source": "rare" if i < 2 else "common"}
        )
        for i in range(20)
    ]
    first = stratified_subset(rows, 6)
    assert [x.instance_id for x in first] == [x.instance_id for x in stratified_subset(rows, 6)]
    assert sum(x.metadata["source"] == "rare" for x in first) == 2


def test_atomic_status_retries_windows_sharing_violation_without_partial_write(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    atomic_json(path, {"step": 1})
    original_replace = Path.replace
    attempts = []

    def replace(source, target):
        attempts.append(source)
        if len(attempts) == 1:
            assert json.loads(path.read_text()) == {"step": 1}
            error = PermissionError("Temporary Windows sharing violation")
            error.winerror = 32
            raise error
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    atomic_json(path, {"step": 2})
    assert len(attempts) == 2 and json.loads(path.read_text()) == {"step": 2}


def test_checkpoint_directory_publish_retries_windows_sharing_violation(tmp_path, monkeypatch):
    original_rename = Path.rename
    attempts = []

    def rename(source, target):
        if source.name.startswith(".partial-"):
            attempts.append(source)
            if len(attempts) == 1:
                assert (source / "READY.json").is_file() and not target.exists()
                error = PermissionError("Temporary Windows directory sharing violation")
                error.winerror = 5
                raise error
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", rename)
    report = fit_offline_sft(
        TinyLossModel(),
        examples(),
        examples()[:1],
        tmp_path,
        {},
        max_steps=1,
        batch_size=1,
        accumulation=1,
        event_callback=lambda event: None,
    )
    assert len(attempts) == 2 and (Path(report["last_checkpoint"]) / "READY.json").is_file()


def test_six_disconnected_instances_preserve_local_results_and_retry_in_isolated_repos(
    tmp_path, monkeypatch
):
    now = [1000.0]
    online = [False]
    commits = {}

    class Api:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, repo, **kwargs):
            if not online[0]:
                raise httpx2.ConnectError("Network disconnected")

        def model_info(self, repo):
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, repo, **kwargs):
            commits.setdefault(repo, []).extend(op.path_in_repo for op in kwargs["operations"])

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr("bifocalgemma.training.v2_hub.time.time", lambda: now[0])
    instances = []
    for key in v2.MODELS:
        root = tmp_path / key
        root.mkdir()
        journal = root / "predictions.jsonl"
        journal.write_text('{"case_id":"fixture","prediction":"saved"}\n')
        heartbeat = RemoteHeartbeat("fixture/" + key, root, "family/model/train", token="test")
        queue = BenchmarkUploadQueue(root, backoff_path=root / "backoff.json", clock=lambda: now[0])

        def synchronize(repo=key):
            if not online[0]:
                raise httpx2.ConnectError("Network disconnected")
            return {"revision": repo}

        heartbeat({"stage": "training", "step": 1})
        assert queue.attempt(synchronize, final=True) is None
        assert queue.state["pending"] and not queue.state["final_synced"]
        assert heartbeat.last_success == 0 and journal.read_text().count("saved") == 1
        instances.append((heartbeat, queue, synchronize, journal))
    online[0] = True
    now[0] += 61
    for heartbeat, queue, upload_callback, journal in instances:
        heartbeat({"stage": "completed", "step": 2}, force=True)
        result = queue.attempt(upload_callback, final=True)
        assert result and not queue.state["pending"] and queue.state["final_synced"]
        assert heartbeat.last_success == now[0] and journal.read_text().count("saved") == 1
    assert set(commits) == {"fixture/" + key for key in v2.MODELS}
    assert all(paths == ["family/model/train/last_seen.json"] for paths in commits.values())
