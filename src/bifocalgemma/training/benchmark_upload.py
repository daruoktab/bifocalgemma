"""Durable deferred benchmark backups; transient Hub failures never discard predictions."""

import json
import math
import time
from collections.abc import Callable
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub.utils import parse_ratelimit_headers

from bifocalgemma.training.stages import atomic_json
from bifocalgemma.training.telemetry import append_event


def retry_delay(error, failures, now):
    """None means fatal; do not hide permissions, disk, or integrity errors."""
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    if code not in {408, 412, 429, 500, 502, 503, 504} and not isinstance(
        error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
    ):
        return None
    delay = min(3600, 60 * 2 ** min(max(0, failures - 1), 6))
    if response is not None:
        headers = response.headers
        rate = parse_ratelimit_headers(headers)
        if rate is not None:
            delay = max(delay, rate.reset_in_seconds + 1)
        value = headers.get("Retry-After")
        if value:
            try:
                wait = float(value)
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    wait = date.timestamp() - now
                except (ValueError, TypeError, OverflowError):
                    wait = 0
            if math.isfinite(wait):
                delay = max(delay, wait + 1)
    return float(delay)


class BenchmarkUploadQueue:
    """Retry at later case boundaries, without sleeping the evaluation loop.

    SDK-internal HTTP retries/timeouts still apply to an in-flight request.
    Shared backoff prevents subsequent model runs hammering the same Hub repo.
    """

    def __init__(self, root, *, backoff_path, progress=None, clock=time.time, interval=300):
        self.path = Path(root) / "benchmark_upload_status.json"
        self.backoff_path = Path(backoff_path)
        self.progress, self.clock, self.interval = progress, clock, interval
        self.state: dict[str, Any] = {"status": "NOT_STARTED", "pending": False, "failures": 0}
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))

    def emit(self):
        self.state["updated_at"] = self.clock()
        atomic_json(self.path, self.state)
        event = {
            "stage": "benchmark_upload_" + self.state["status"].lower(),
            "benchmark_backup_status": self.state["status"],
            "backup_pending": self.state["pending"],
            "backup_next_retry_at": self.state.get("next_retry_at"),
            "backup_last_success_at": self.state.get("last_success_at"),
            "message": "Benchmark backup: " + self.state["status"],
        }
        print(json.dumps(event), flush=True)
        append_event(self.path.with_name("benchmark_upload_events.jsonl"), event)
        if self.progress:
            self.progress(event)

    def mark_pending(self):
        if not self.state.get("pending"):
            self.state.update(
                pending=True, final_synced=False, status="LOCAL_CHANGES", updated_at=self.clock()
            )
            atomic_json(self.path, self.state)

    def attempt(self, synchronize: Callable[[], Any], *, final=False):
        now = self.clock()
        shared = {}
        if self.backoff_path.exists():
            shared = json.loads(self.backoff_path.read_text(encoding="utf-8"))
        retry_at = max(shared.get("next_retry_at", 0), self.state.get("next_retry_at", 0))
        if now < retry_at or (
            not final and now < self.state.get("last_success_at", -self.interval) + self.interval
        ):
            self.state.update(status="DEFERRED", pending=True, next_retry_at=retry_at)
            self.emit()
            return None
        self.state.update(status="UPLOADING", pending=True)
        self.state["final_synced"] = False
        self.emit()
        try:
            result = synchronize()
        except Exception as error:
            failures = max(self.state.get("failures", 0), shared.get("failures", 0)) + 1
            delay = retry_delay(error, failures, self.clock())
            if delay is None:
                self.state.update(status="FAILED", pending=True, error_type=type(error).__name__)
                self.emit()
                raise
            retry_at = self.clock() + delay
            self.state.update(
                status="DEFERRED",
                pending=True,
                failures=failures,
                next_retry_at=retry_at,
                error_type=type(error).__name__,
                http_status=getattr(getattr(error, "response", None), "status_code", None),
            )
            atomic_json(self.backoff_path, {"next_retry_at": retry_at, "failures": failures})
            self.emit()
            return None
        self.state.update(
            status="SYNCED",
            pending=False,
            failures=0,
            next_retry_at=0,
            last_success_at=self.clock(),
            final_synced=final,
        )
        atomic_json(self.backoff_path, {"next_retry_at": 0, "failures": 0})
        self.emit()
        return result
