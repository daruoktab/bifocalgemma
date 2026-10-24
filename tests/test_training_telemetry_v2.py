import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
import torch

from bifocalgemma.training import timing
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.source_identity import source_fingerprints
from bifocalgemma.training.wiggly_dashboard import WigglyDashboard
from tests.test_offline_sft_training import TinyLossModel, examples
from tools.runtime_bootstrap import load_runtime


def test_clock_counts_active_sessions_not_downtime_or_checkpoint_twice(tmp_path, monkeypatch):
    counter, wall = [0.0], [1000.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: counter[0])
    monkeypatch.setattr(timing.time, "time", lambda: wall[0])
    first = timing.DurableClock(tmp_path / "timing.json")
    counter[0] += 20
    first.persist(force=True)
    first.persist(force=True)
    assert first.elapsed() == 20
    # Hard kill: no close; the final seven seconds are not durably observed.
    counter[0] = 0
    wall[0] += 3600
    resumed = timing.DurableClock(tmp_path / "timing.json", checkpoint_seconds=15)
    assert resumed.elapsed() == 20 and resumed.gap == 3600
    counter[0] += 10
    resumed.close()
    assert resumed.elapsed() == 30
    counter[0] += 30
    resumed.persist(force=True)
    assert json.loads((tmp_path / "timing.json").read_text())["status"] == "COMPLETED"
    assert json.loads((tmp_path / "timing.json").read_text())["sessions"] == 2


def test_training_resume_keeps_elapsed_from_step_five_through_ten(tmp_path, monkeypatch):
    counter, wall = [0.0], [1000.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: counter[0])
    monkeypatch.setattr(timing.time, "time", lambda: wall[0])
    events, saved = [], []

    def event(row):
        events.append(row)
        if row["stage"] == "forward_started":
            counter[0] += 10
            wall[0] += 10

    def interrupt(path):
        saved.append(path)
        raise KeyboardInterrupt("Molab session closed")

    options = dict(
        max_steps=10,
        epochs=10,
        batch_size=1,
        accumulation=1,
        eval_every=10,
        save_every=5,
        event_callback=event,
    )
    with pytest.raises(KeyboardInterrupt):
        fit_offline_sft(
            TinyLossModel(),
            examples(),
            examples()[:1],
            tmp_path,
            {},
            checkpoint_callback=interrupt,
            **options,
        )
    checkpoint = torch.load(Path(saved[0]) / "trainer_state.pt", weights_only=False)
    assert checkpoint["training_state"]["accumulated_seconds"] == 50
    counter[0] = 0
    wall[0] += 3600
    events.clear()
    result = fit_offline_sft(
        TinyLossModel(), examples(), examples()[:1], tmp_path, {}, resume_from=saved[0], **options
    )
    assert events[0]["elapsed_seconds"] == 50
    assert result["timing"]["active_seconds"] == 100
    assert result["timing"]["unobserved_gap_seconds"] == 3600
    assert result["segment_seconds"] == 50


def test_semantic_identity_allows_comments_and_ui_but_rejects_model_changes(tmp_path):
    (tmp_path / "training").mkdir()
    model = tmp_path / "model.py"
    model.write_text('"""First docs."""\nx = 1\n')
    ui = tmp_path / "training/telemetry.py"
    ui.write_text("color = 'red'\n")
    first = source_fingerprints(tmp_path)
    model.write_text('"""Updated docs."""\n# A comment\nx=1\n')
    ui.write_text("color = 'blue'\n")
    updated = source_fingerprints(tmp_path)
    assert first["code_sha256"] != updated["code_sha256"]
    assert first["training_semantics_sha256"] == updated["training_semantics_sha256"]
    model.write_text("x = 2\n")
    assert (
        source_fingerprints(tmp_path)["training_semantics_sha256"]
        != first["training_semantics_sha256"]
    )


def test_widgets_update_in_place_and_restore_history_without_duplicate_points(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("wigglystuff.html._in_notebook", lambda: True)
    dashboard = WigglyDashboard(None, refresh_seconds=0)
    dashboard(
        {
            "stage": "run_started",
            "run_dir": str(tmp_path),
            "max_steps": 10,
            "history": [{"step": 1, "train_loss": 0.9}],
        }
    )
    identities = tuple(id(w) for w in (dashboard.loss, dashboard.cards, dashboard.progress_bar))
    dashboard({"stage": "optimizer_step_complete", "step": 2, "train_loss": 0.7})
    dashboard(
        {
            "stage": "validation_complete",
            "step": 2,
            "dev_loss": 0.6,
            "eval": {"by_source": {"fixture": {"loss": 0.6, "supervised_tokens": 5}}},
        }
    )
    dashboard({"stage": "checkpoint_complete", "step": 2, "max_steps": 10})
    assert identities == tuple(
        id(w) for w in (dashboard.loss, dashboard.cards, dashboard.progress_bar)
    )
    assert len(dashboard.loss.data) == 3
    assert dashboard.progress_bar.value == 2
    assert dashboard.ui_error is None
    json.dumps(dashboard.loss.spec, allow_nan=False)
    assert dashboard.table.data["children"]
    dashboard({"stage": "failed", "error": "<script>bad</script>", "error_type": "OOM"})
    assert "<script>bad</script>" not in dashboard.activity.html
    assert (tmp_path / "monitor.html").is_file()


@pytest.mark.parametrize("entry", ["src/bifocalgemma/__init__.py", "../../escape.py"])
def test_runtime_loader_pins_bytes_and_rejects_unsafe_paths(tmp_path, monkeypatch, entry):
    monkeypatch.chdir(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry, "# fixture\n")
    data = buffer.getvalue()
    requests = []

    def download(url, **kwargs):
        requests.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr("urllib.request.urlopen", download)
    digest = hashlib.sha256(data).hexdigest()
    if ".." in entry:
        with pytest.raises(ValueError, match="unsafe path"):
            load_runtime("fixture/assets", "runtime.zip", "a" * 40, digest)
        assert not (tmp_path / "escape.py").exists()
    else:
        root = load_runtime("fixture/assets", "runtime.zip", "a" * 40, digest)
        assert (root / entry).is_file() and "a" * 40 in requests[0]
        load_runtime("fixture/assets", "runtime.zip", "a" * 40, digest)
        assert len(requests) == 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_runtime("fixture/assets", "runtime.zip", "a" * 40, "0" * 64)
