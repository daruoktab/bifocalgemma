"""Explicit evaluation reset, preserving training evidence and resume identity."""

import copy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from bifocalgemma.training.stages import atomic_json


def reset_state(state, reset_id):
    state = copy.deepcopy(state)
    state["stages"] = {
        name: value
        for name, value in state["stages"].items()
        if not name.startswith("benchmark:") and name != "final_upload"
    }
    state["benchmark_reset_id"] = reset_id
    return state


def reset_local_benchmarks(root, reset_id):
    """Archive old evaluation once per reset ID; never touch training checkpoints."""
    root = Path(root).resolve()
    state_path = root / "pipeline_state.json"
    if not reset_id or not state_path.exists():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("benchmark_reset_id") == reset_id:
        return
    archive = root / (".benchmark-reset-" + uuid4().hex)
    archive.mkdir()
    atomic_json(archive / "pipeline_state.json", state)
    for relative in ("benchmarks", "training/final_upload_receipt.json", "upload_complete.json"):
        source = root / relative
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError("Benchmark archive path escaped run directory.")
        if source.exists():
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    atomic_json(state_path, reset_state(state, reset_id))


def remote_reset_plan(prefix, names, state, receipt, reset_id):
    """Return exact deletions and replacement JSON bytes for a single Hub prefix."""
    from bifocalgemma.training.hub import checked_prefix

    checked_prefix(prefix)
    removed = sorted(
        name
        for name in names
        if name.startswith(prefix + "/benchmarks/")
        or name == prefix + "/training/final_upload_receipt.json"
    )
    replacements = {}
    payload = (
        json.dumps(reset_state(state, reset_id), ensure_ascii=False, indent=2) + "\n"
    ).encode()
    replacements[prefix + "/pipeline_state.json"] = payload
    if receipt is not None:
        receipt = copy.deepcopy(receipt)
        receipt["files"] = {
            name: digest
            for name, digest in receipt["files"].items()
            if prefix + "/" + name not in removed
        }
        receipt["files"]["pipeline_state.json"] = hashlib.sha256(payload).hexdigest()
        replacements[prefix + "/upload_complete.json"] = json.dumps(
            receipt, ensure_ascii=False, indent=2
        ).encode()
    return removed, replacements
