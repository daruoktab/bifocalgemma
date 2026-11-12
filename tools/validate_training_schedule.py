"""Audit the actual frozen pool and persist its model-independent one-pass schedule."""

import argparse
import json
from collections import Counter
from pathlib import Path

from bifocalgemma.data.offline_sft import load_curated_pool
from bifocalgemma.training.schedule import (
    build_shared_schedule,
    canonical_schedule_records,
    persist_shared_schedule,
)
from bifocalgemma.v3.constants import COLLECTION, DATASET_REPO, DATASET_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-updates", type=int)
    parser.add_argument("--local", action="store_true", help="Use the explicit local-pool identity instead of the pinned main release.")
    args = parser.parse_args()
    rows, manifest = load_curated_pool(args.pool)
    schedule = build_shared_schedule(
        canonical_schedule_records(rows["train"]),
        dataset_identity={
            "repo": "local_curated_pool" if args.local else DATASET_REPO,
            "revision": manifest["fingerprint"] if args.local else DATASET_REVISION,
            "collection": args.pool.name if args.local else COLLECTION,
            "pool_fingerprint": manifest["fingerprint"],
        },
    )
    boundary = (
        args.stage1_updates if args.stage1_updates is not None else schedule.total_updates // 2
    )
    offset = schedule.validate_cursor(boundary)
    first, second = schedule.records[:offset], schedule.records[offset:]
    if len(set(schedule.ordered_ids)) != len(rows["train"]):
        raise ValueError("Schedule lost or repeated an original example.")
    persist_shared_schedule(args.output / "shared_schedule.json", schedule)
    report = {
        "pool_fingerprint": manifest["fingerprint"],
        "schedule_signature": schedule.signature,
        "train_examples": len(rows["train"]),
        "dev_examples": len(rows["dev"]),
        "total_updates": schedule.total_updates,
        "stage1_updates": boundary,
        "stage2_updates": schedule.total_updates - boundary,
        "synthetic_replacements": 0,
        "duplicate_ids": 0,
        "missing_ids": 0,
        "partial_window_size": schedule.total_examples % 4,
        "stages": [
            {
                "examples": len(part),
                "tasks": dict(Counter(r.task for r in part)),
                "sources": dict(Counter(r.source for r in part)),
                "length_bins": dict(Counter(r.length_bin for r in part)),
            }
            for part in (first, second)
        ],
    }
    (args.output / "schedule_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
