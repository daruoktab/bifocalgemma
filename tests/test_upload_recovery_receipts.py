"""Cold recovery must accept the latest atomic checkpoint-only receipt."""

import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub

from bifocalgemma.data.curation import file_digest
from bifocalgemma.training.hub import recover_remote_run, synchronize_run
from bifocalgemma.training.stages import atomic_json, retained_checkpoints


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
