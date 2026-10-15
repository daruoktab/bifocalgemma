"""Atomic Hub retention deletes only obsolete checkpoints of the current run."""

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from bifocalgemma.training.hub import synchronize_run
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.stages import atomic_json
from tests.test_offline_sft_training import TinyLossModel, examples


@pytest.mark.parametrize("checkpoint_only", [False, True])
@pytest.mark.parametrize("recipe", [None, "optimized"])
def test_checkpoint_upload_and_retention_receipt_are_one_scoped_commit(
    tmp_path, monkeypatch, checkpoint_only, recipe
):
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
        eval_every=10,
        save_every=1,
    )
    atomic_json(root / "pipeline_state.json", {"fixture": True})
    prefix = "symmetric/BifocalGemma-E2B-E2B/fixture"
    if recipe:
        prefix += "/" + recipe
    foreign = "asymmetric/BifocalGemma-E2B-E4B/other/training/checkpoint-000001/READY.json"
    old = prefix + "/training/checkpoint-000001/READY.json"
    sibling = "symmetric/BifocalGemma-E2B-E2B/fixture/base/training/checkpoint-000001/READY.json"
    remote = {foreign: b"other-model", old: b"obsolete"}
    if recipe:
        remote[sibling] = b"other-recipe"
    commits = []

    class Api:
        def __init__(self, **kwargs):
            assert kwargs["token"] == "fixture-write-token"

        def create_repo(self, **kwargs):
            pass

        def model_info(self, repo):
            return SimpleNamespace(sha="a" * 40)

        def list_repo_files(self, *args, **kwargs):
            return list(remote)

        def create_commit(self, **kwargs):
            commits.append(kwargs)
            for operation in kwargs["operations"]:
                path = operation.path_in_repo
                if isinstance(operation, huggingface_hub.CommitOperationDelete):
                    assert path == old
                    del remote[path]
                else:
                    value = operation.path_or_fileobj
                    remote[path] = (
                        Path(value).read_bytes() if isinstance(value, str) else value.getvalue()
                    )
            return SimpleNamespace(oid="b" * 40)

    def download(repo, filename, **kwargs):
        path = tmp_path / "cached" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    receipt = synchronize_run(
        "fixture/model", root, prefix, token="fixture-write-token", checkpoint_only=checkpoint_only
    )
    assert len(commits) == 1 and old not in remote and remote[foreign] == b"other-model"
    if recipe:
        assert remote[sibling] == b"other-recipe"
    assert prefix + "/upload_complete.json" in remote
    assert receipt["retained_checkpoints"] == ["checkpoint-000002", "checkpoint-000003"]
    assert (root / "upload_complete.json").is_file()
    assert receipt["checkpoint_only"] is checkpoint_only
