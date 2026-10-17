import json
from datetime import datetime, timezone
from email.utils import format_datetime

import httpx
import pytest

from bifocalgemma.training.benchmark_upload import BenchmarkUploadQueue, retry_delay
from bifocalgemma.training.stages import atomic_json
from tools.inspect_benchmark_activity import inspect


def limited(headers=None, code=429):
    request = httpx.Request("POST", "https://fixture/commit")
    return httpx.HTTPStatusError(
        "fixture",
        request=request,
        response=httpx.Response(code, headers=headers or {}, request=request),
    )


def test_durable_backoff_shared_across_models_and_final_retries_without_sleep(tmp_path):
    now = [1000.0]
    root = tmp_path / "first"
    shared = tmp_path / "backoff.json"
    events = []
    calls = []
    queue = BenchmarkUploadQueue(
        root, backoff_path=shared, clock=lambda: now[0], progress=events.append
    )

    def fail():
        calls.append(1)
        raise limited({"Retry-After": "120", "RateLimit": '"api";r=0;t=180'})

    assert queue.attempt(fail) is None
    assert queue.state["pending"] and queue.state["next_retry_at"] == 1181
    assert queue.attempt(fail, final=True) is None and len(calls) == 1
    other = BenchmarkUploadQueue(tmp_path / "second", backoff_path=shared, clock=lambda: now[0])
    assert other.attempt(fail) is None and len(calls) == 1
    resumed = BenchmarkUploadQueue(root, backoff_path=shared, clock=lambda: now[0])
    now[0] = 1181
    result = resumed.attempt(lambda: {"revision": "verified"}, final=True)
    assert result == {"revision": "verified"}
    assert not resumed.state["pending"] and resumed.state["final_synced"]
    saved = json.loads(resumed.path.read_text())
    assert saved["status"] == "SYNCED" and saved["last_success_at"] == 1181
    assert events[1]["benchmark_backup_status"] == "DEFERRED"


def test_interval_throttles_periodic_upload_but_final_can_flush(tmp_path):
    now = [1000.0]
    queue = BenchmarkUploadQueue(tmp_path, backoff_path=tmp_path / "backoff", clock=lambda: now[0])
    assert queue.attempt(lambda: {"ok": True})
    queue.mark_pending()
    assert json.loads(queue.path.read_text())["status"] == "LOCAL_CHANGES"
    assert queue.state["pending"]
    now[0] += 1

    def unexpected_upload():
        raise AssertionError("must not upload yet")

    assert queue.attempt(unexpected_upload) is None
    assert queue.attempt(lambda: {"ok": True}, final=True)


@pytest.mark.parametrize(
    "error", [ValueError("checksum"), OSError("disk full"), limited(code=401), limited(code=403)]
)
def test_fatal_errors_not_hidden(tmp_path, error):
    queue = BenchmarkUploadQueue(tmp_path, backoff_path=tmp_path / "backoff")

    def fail():
        raise error

    with pytest.raises(type(error)):
        queue.attempt(fail)
    assert queue.state["status"] == "FAILED"


def test_retry_after_date_and_network_failures():
    stamp = format_datetime(datetime.fromtimestamp(1600, timezone.utc), usegmt=True)
    assert retry_delay(limited({"Retry-After": stamp}), 1, 1000) == 601
    assert retry_delay(httpx.ConnectError("offline"), 3, 1000) == 240
    assert retry_delay(limited(code=503), 1, 1000) == 60


def test_keyboard_interrupt_is_not_swallowed(tmp_path):
    queue = BenchmarkUploadQueue(tmp_path, backoff_path=tmp_path / "backoff")

    def stop():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        queue.attempt(stop)
    assert json.loads(queue.path.read_text())["pending"]


def test_external_activity_reader_does_not_need_running_kernel(tmp_path):
    atomic_json(tmp_path / "pipeline_state.json", {})
    atomic_json(
        tmp_path / "benchmarks/test/activity.json",
        {"updated_at": 1000, "stage": "benchmark_decoding", "generated_tokens": 10},
    )
    rows = inspect(tmp_path, now=1400)
    assert rows[0]["health"] == "STALE_REQUIRES_INSPECTION"
    assert rows[0]["detail"]["generated_tokens"] == 10


def test_parent_conflict_refreshes_but_keeps_retry_bounded(monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    import bifocalgemma.training.hub as hub

    calls = []

    def sync(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise HfHubHTTPError("branch moved", response=limited(code=412).response)
        return {"revision": "new-head"}

    monkeypatch.setattr(hub, "_synchronize_once", sync)
    assert (
        hub.synchronize_run("fixture/model", "root", "prefix", token="fixture")["revision"]
        == "new-head"
    )
    assert len(calls) == 3

    def always_conflict(*args, **kwargs):
        calls.append(kwargs)
        raise HfHubHTTPError("branch moved", response=limited(code=412).response)

    calls.clear()
    monkeypatch.setattr(hub, "_synchronize_once", always_conflict)
    with pytest.raises(HfHubHTTPError):
        hub.synchronize_run("fixture/model", "root", "prefix", token="fixture")
    assert len(calls) == 3
