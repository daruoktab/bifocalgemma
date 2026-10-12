"""Audit mixed task pools and every native/viewer record before dataset publication."""
# ruff: noqa: E402 -- local standalone tool bootstrap

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ["USE_TORCH"] = "0"  # Static audit; no model or tensor allocation.
sys.path.insert(0, str(ROOT / "src"))

from bifocalgemma.data.context import canonical_snapshot_hash
from bifocalgemma.data.context_tasks import normalize_context_task, repair_preimage_coverage
from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.patches import apply_patch_to_snapshot
from bifocalgemma.data.releases import audit_split_disjointness, verify_release
from bifocalgemma.data.repository_corpus import ArchiveSnapshotStore
from bifocalgemma.training.stages import atomic_json


def audit_pool(pool, archives):
    from types import SimpleNamespace

    records, manifest = load_curated_pool(pool)
    from bifocalgemma.data.remediation import inspect_structure

    foundation_rows, foundation = load_curated_pool(
        ROOT / "outputs/instruction_enc8k_dec2k_v2_ready"
    )
    if manifest["report"]["foundation_fingerprint"] != foundation["fingerprint"]:
        raise ValueError("Mixed pool is not based on the verified published foundation.")
    proof = foundation["report"]["git_patch_validation"]
    original_rows = foundation_rows["train"] + foundation_rows["dev"]
    if proof["retained_identity"] != stable_fingerprint(
        [(r["instance_id"], r["target_patch"]) for r in original_rows]
    ):
        raise ValueError("Published foundation patch proof identity changed.")
    foundation_by_id = {r["instance_id"]: r for r in original_rows}
    audit_split_disjointness(records)
    seen = set()
    task_counts = Counter()
    checks = Counter()
    scopes = Counter()
    token_totals = {split: Counter() for split in records}
    token_lengths = {split: {} for split in records}
    store = ArchiveSnapshotStore(archives, allow_network=False)
    last_key = None
    last_snapshot = None
    last_snapshot_hash = None
    for split, values in records.items():
        for index, row in enumerate(values, 1):
            signature = row.get("normalized_pair_hash", row.get("pair_hash"))
            if not signature or signature in seen:
                raise ValueError("mixed_pool_duplicate_target_pair")
            seen.add(signature)
            kind = row.get("task_type", "repair")
            task_counts[kind] += 1
            scopes[row.get("task_scope", "single_file_editing")] += 1
            for field, count in row["token_lengths"].items():
                if field in {"encoder", "pair_decoder", "baseline_decoder", "target"}:
                    token_totals[split][field] += count
                    token_lengths[split].setdefault(field, []).append(count)
            if row.get("record_type") == "context_sft_v1":
                normalize_context_task(row)
                key = (row["repos"], row["commit"])
                if key != last_key:
                    last_snapshot, _ = store.snapshot(
                        SimpleNamespace(repo=row["repos"], base_commit=row["commit"])
                    )
                    last_snapshot_hash = canonical_snapshot_hash(last_snapshot)
                    last_key = key
                snapshot = last_snapshot
                assert snapshot is not None
                if last_snapshot_hash != row["snapshot_sha256"]:
                    raise ValueError("context_source_snapshot_identity")
                if kind == "repair":
                    if not repair_preimage_coverage(row["target_text"], row["context_selection"]):
                        raise ValueError("repair_target_preimage_not_visible")
                    apply_patch_to_snapshot(snapshot, row["target_text"])
                    checks["repository_patch_applied"] += 1
                elif kind == "completion":
                    from bifocalgemma.data.python_quality import inspect_python_target

                    if inspect_python_target(
                        "def _sft_target_body_():\n" + row["target_text"],
                        filename=row["instance_id"],
                    ):
                        raise ValueError("completion_target_body_syntax_warning")
                    if (
                        row["target_text"] in row["context"]
                        or row["context"].count("# <MISSING_CODE>") != 1
                    ):
                        raise ValueError("completion_target_leak")
                    checks["completion_masking_and_snapshot_verified"] += 1
                else:
                    checks["localization_visibility_verified"] += 1
            else:
                original = foundation_by_id.get(row["instance_id"])
                if original is None or any(
                    row.get(key) != original.get(key)
                    for key in (
                        "instruction",
                        "before",
                        "after",
                        "path",
                        "target_patch",
                        "source",
                        "source_revision",
                        "commit",
                        "normalized_pair_hash",
                    )
                ):
                    raise ValueError(
                        "Mixed foundation input/target changed from its byte-verified patch proof."
                    )
                structure = inspect_structure(row["after"])
                if structure is None or structure["compile_error"] or structure["warnings"]:
                    raise ValueError(
                        "Mixed foundation has a noncompiling or warning-bearing target."
                    )
                checks["foundation_byte_unchanged_patch_proof_subset"] += 1
                checks["foundation_targets_fresh_compile_warning_free"] += 1
            if index % 1000 == 0:
                print(
                    json.dumps(
                        {"stage": "pool_audit", "split": split, "done": index, "total": len(values)}
                    ),
                    flush=True,
                )
    return {
        "pool_fingerprint": manifest["fingerprint"],
        "counts": {k: len(v) for k, v in records.items()},
        "tasks": dict(task_counts),
        "task_scopes": dict(scopes),
        "tokens": {split: dict(value) for split, value in token_totals.items()},
        "token_quantiles": {
            split: {
                field: {
                    name: sorted(lengths)[int((len(lengths) - 1) * q)]
                    for name, q in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1)]
                }
                for field, lengths in fields.items()
            }
            for split, fields in token_lengths.items()
        },
        "checks": dict(checks),
        "limits": manifest["report"]["tokens"]["limits"],
        "unique_pairs": len(seen),
        "foundation_fingerprint": foundation["fingerprint"],
        "foundation_git_patch_evidence": "byte-identical subset of published all-record Git apply/after-byte proof",
        "split_disjointness": True,
        "functional_quality_verified": False,
    }


def audit_views(release_directory):
    import zipfile

    import pyarrow.parquet as pq

    root = Path(release_directory)
    release = verify_release(root)
    results = {}
    for name, item in release["collections"].items():
        native = {}
        with zipfile.ZipFile(root / item["archives"]["public"]["path"]) as archive:
            for member in archive.namelist():
                if member.endswith(
                    ("train.jsonl", "dev.jsonl", "eval_inputs.jsonl", "inputs.jsonl")
                ):
                    for raw in archive.read(member).splitlines():
                        row = json.loads(raw)
                        if row["instance_id"] in native:
                            raise ValueError("release_duplicate_native_id")
                        native[row["instance_id"]] = row
        count = 0
        for file in sorted((root / "collections" / name / "view").glob("*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    record = json.loads(row["record_json"])
                    if record != native[row["instance_id"]]:
                        raise ValueError("parquet_native_row_mismatch")
                    if row["role"] != item["role"]:
                        raise ValueError("parquet_role_mismatch")
                    if item["role"] == "eval" and any(
                        k in record
                        for k in (
                            "answer",
                            "correct_answer",
                            "correct_letter",
                            "reference",
                            "gold_patch",
                            "test_script",
                        )
                    ):
                        raise ValueError("gold_exposed_in_eval_view")
                    count += 1
        if count != len(native) or count != sum(item["counts"].values()):
            raise ValueError("release_view_count_mismatch")
        results[name] = {
            "native_records": len(native),
            "parquet_records": count,
            "exact_row_parity": True,
        }
        print(json.dumps({"stage": "viewer_audit", "collection": name, "count": count}), flush=True)
    return {"release_fingerprint": release["fingerprint"], "collections": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path)
    parser.add_argument(
        "--archives", type=Path, default=ROOT / "outputs/context_sources/repository_corpus/archives"
    )
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "format": "bifocalgemma_context_release_audit_v1",
        "date": "2026-10-13",
    }
    if args.pool:
        result["pool"] = audit_pool(args.pool, args.archives)
    if args.release:
        result["release"] = audit_views(args.release)
    result["fingerprint"] = stable_fingerprint(result)
    atomic_json(args.output, result)
    print(json.dumps({"status": "PASS", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
