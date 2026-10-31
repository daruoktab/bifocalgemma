"""Offline regression coverage for the standalone base-encoder control."""

import ast
import base64
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tools.generate_pilot_notebook import render
from tools.molab_pilot_support import Journal, PilotStorage, StageStore, journaled_loss


def test_standalone_bundle_has_fixed_control_and_no_colab_side_effects():
    text = render()
    tree = ast.parse(text)
    payload = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_encoded" for t in node.targets)
    )
    with zipfile.ZipFile(io.BytesIO(base64.b85decode(payload))) as archive:
        worker = archive.read("tools/prepare_colab_pilot.py").decode()
        support = archive.read("tools/molab_pilot_support.py").decode()
        compile(worker, "worker", "exec")
        assert "/content/" not in worker
        assert "ntfy.sh" not in worker
        assert "encoder_source='base'" in support or 'encoder_source="base"' in support
        assert "steps_per_arm=200" in support
        assert "full_cross_stage2" in worker
        assert "effective_gate_magnitude" in worker
        assert "checkpoint_store" in worker
    assert text == render()


def test_checkpoint_restores_optimizer_rng_and_skips_partial(tmp_path):
    torch.manual_seed(17)
    random.seed(17)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    def update():
        optimizer.zero_grad()
        model(torch.randn(2, 3)).square().sum().backward()
        optimizer.step()

    update()
    store = StageStore(tmp_path, "config")
    store.save(model, optimizer, {"step": 25, "cursor": 100, "order": [0, 1], "epoch": 0})
    random_expected = random.random()
    update()
    expected = {n: p.detach().clone() for n, p in model.named_parameters()}
    (tmp_path / "step-000050").mkdir()
    (tmp_path / "step-000050/state.pt").write_bytes(b"partial")
    saved = store.load(model, optimizer)
    assert saved is not None
    assert saved["step"] == 25 and saved["cursor"] == 100
    assert random.random() == random_expected
    update()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="different experiment"):
        StageStore(tmp_path, "other").load(model, optimizer)


def test_journal_integrity_and_phase_isolation(tmp_path):
    base = Journal(tmp_path / "base", "config")
    other = Journal(tmp_path / "it", "config")
    base.put("example", {"prediction": "a"})
    assert other.get("example") is None
    cached = base.get("example")
    assert cached is not None and cached["prediction"] == "a"
    path = base.path("example")
    data = json.loads(path.read_text())
    data["value"]["prediction"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        base.get("example")


def test_loss_journal_preserves_token_weighting_and_reuses_examples(tmp_path):
    from bifocalgemma.contracts import TokenizedRepairExample

    calls = []
    model = SimpleNamespace(
        pilot_journal=Journal(tmp_path, "config"), decoder=SimpleNamespace(layers=[])
    )
    examples = [TokenizedRepairExample(str(i), (2,), (2,), tuple([4] * i)) for i in (1, 3)]

    def original(model, items, **kwargs):
        calls.append(items[0].instance_id)
        count = len(items[0].target_ids)
        row = {"loss": float(count), "supervised_tokens": count, "examples": 1}
        return {**row, "by_source": {"x": row}, "by_task_type": {"repair": row}}

    evaluate = journaled_loss(original)
    first = evaluate(model, examples, batch_size=1, device="cpu")
    second = evaluate(model, examples, batch_size=1, device="cpu")
    assert first == second and first["loss"] == 2.5 and len(calls) == 2


def test_upload_retry_verification_and_base_prefix_only(tmp_path, monkeypatch):
    import huggingface_hub

    attempts = []

    class API:
        def upload_folder(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise ConnectionError("offline")
            return SimpleNamespace(oid="commit")

    storage = object.__new__(PilotStorage)
    storage.root, storage.token = tmp_path, "unused"
    monkeypatch.setattr(storage, "api", API(), raising=False)
    storage.identity, storage.remote_revision, storage.last_sync = "source", None, 0
    (tmp_path / "result.json").write_text("{}")
    monkeypatch.setattr("tools.molab_pilot_support.time.sleep", lambda _: None)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **kw: str(tmp_path / "remote.json")
    )
    storage.sync(final=True)
    assert len(attempts) == 2
    assert all(x["path_in_repo"] == "e2b-base" for x in attempts)
    assert storage.remote_revision == "commit"
    assert json.loads((tmp_path / "upload_status.json").read_text())["status"] == "VERIFIED"
    assert "cache" not in attempts[-1]["allow_patterns"]


def test_failed_final_upload_remains_pending(tmp_path, monkeypatch):
    storage = object.__new__(PilotStorage)
    storage.root, storage.token, storage.identity = tmp_path, "unused", "source"
    storage.remote_revision = "previous"
    storage.last_sync = 0

    def offline(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(storage, "api", SimpleNamespace(upload_folder=offline), raising=False)
    monkeypatch.setattr("tools.molab_pilot_support.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="pending"):
        storage.sync(final=True)
    status = json.loads((tmp_path / "upload_status.json").read_text())
    assert status["status"] == "PENDING_UPLOAD"
    assert status["last_verified_revision"] == "previous"


def test_completed_run_retries_upload_without_loading_model(tmp_path, monkeypatch):
    from tools import molab_pilot_support as support

    monkeypatch.setenv("HF_TOKEN", "test-only")

    monkeypatch.setattr(support.PilotStorage, "restore", lambda self: None)
    calls = []
    monkeypatch.setattr(support.PilotStorage, "sync", lambda self, final=False: calls.append(final))
    (tmp_path / "manifest.json").write_text(json.dumps({"source_identity": "source"}))
    (tmp_path / "pilot_e2b_e2b.json").write_text(json.dumps({"status": "COMPLETED"}))
    result = support.run(tmp_path, tmp_path, "test-only", "source")
    assert result["status"] == "COMPLETED" and calls == [True]
