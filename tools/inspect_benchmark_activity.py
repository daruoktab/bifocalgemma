"""Read durable benchmark heartbeats from a separate terminal, without using the kernel."""

import argparse
import json
import time
from pathlib import Path
from typing import Any


def inspect(root, *, stale_after=300, now=None) -> list[dict[str, Any]]:
    now = time.time() if now is None else now
    rows = []
    for state in Path(root).rglob("pipeline_state.json"):
        if any(part.startswith(".") for part in state.relative_to(root).parts):
            continue
        candidates = list((state.parent / "benchmarks").glob("*/activity.json"))
        backup = state.parent / "benchmark_upload_status.json"
        if backup.exists():
            candidates.append(backup)
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        payload = json.loads(latest.read_text(encoding="utf-8"))
        age = max(0, now - payload["updated_at"])
        stage = payload.get("stage", payload.get("status"))
        finished = stage == "benchmark_completed" or bool(
            payload.get("final_synced") and not payload.get("pending")
        )
        waiting = payload.get("status") == "DEFERRED" and now < payload.get("next_retry_at", 0)
        rows.append(
            {
                "run": str(state.parent),
                "activity": str(latest),
                "stage": stage,
                "seconds_since_activity": age,
                "health": "WAITING_RETRY"
                if waiting
                else (
                    "STALE_REQUIRES_INSPECTION"
                    if age > stale_after and not finished
                    else "RECENT_OR_IDLE"
                ),
                "detail": payload,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stale-after", type=float, default=300)
    args = parser.parse_args()
    print(json.dumps(inspect(args.root, stale_after=args.stale_after), indent=2))


if __name__ == "__main__":
    main()
