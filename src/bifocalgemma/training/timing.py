"""Persist observed active time; never count an unobserved crash gap as training."""

import json
import time
from pathlib import Path

from bifocalgemma.training.stages import atomic_json


class DurableClock:
    def __init__(self, path, *, checkpoint_seconds=0.0, interval=5.0, started=None):
        self.path = Path(path)
        previous = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.started = time.perf_counter() if started is None else started
        self.interval = interval
        self.last_write = float("-inf")
        self.closed = False
        self.status = "RUNNING"
        self.base = max(float(previous.get("active_seconds", 0)), float(checkpoint_seconds))
        self.first_seen = previous.get("first_seen_unix", time.time())
        self.gap = float(previous.get("unobserved_gap_seconds", 0))
        if previous.get("last_seen_unix"):
            self.gap += max(0.0, time.time() - previous["last_seen_unix"])
        self.sessions = int(previous.get("sessions", 0)) + 1
        self.persist(force=True)

    def elapsed(self):
        return self.base if self.closed else self.base + max(0, time.perf_counter() - self.started)

    def summary(self):
        return {
            "format": "observed_active_time_v1",
            "active_seconds": self.elapsed(),
            "first_seen_unix": self.first_seen,
            "last_seen_unix": time.time(),
            "wall_span_seconds": max(0, time.time() - self.first_seen),
            "unobserved_gap_seconds": self.gap,
            "sessions": self.sessions,
            "coverage": "Observed up to the last durable event; hard-crash tail is unknown.",
        }

    def persist(self, *, force=False, status="RUNNING"):
        now = time.perf_counter()
        if force or now - self.last_write >= self.interval:
            atomic_json(
                self.path, {**self.summary(), "status": self.status if self.closed else status}
            )
            self.last_write = now

    def close(self, status="COMPLETED"):
        self.base = self.elapsed()
        self.closed = True
        self.status = status
        self.persist(force=True, status=status)
