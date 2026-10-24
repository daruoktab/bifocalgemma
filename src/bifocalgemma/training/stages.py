"""Validated stage receipts for sequential, restartable notebook runs."""

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from bifocalgemma.data.curation import file_digest
from bifocalgemma.data.offline_sft import stable_fingerprint


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid4().hex)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    retry_windows_file_operation(lambda: temporary.replace(path))


def retry_windows_file_operation(operation):
    """Retry only transient Windows sharing/access failures; preserve atomicity."""
    for attempt in range(6):
        try:
            return operation()
        except PermissionError as error:
            # Windows readers/scanners can briefly deny an atomic replacement.
            # Preserve the old destination; never fall back to an in-place write.
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 5:
                raise
            time.sleep(0.02 * 2**attempt)


class StageLedger:
    """A completed stage is reusable only with the same signature and file hashes."""

    def __init__(self, root, identity):
        self.root = Path(root).resolve()
        self.path = self.root / "pipeline_state.json"
        self.signature = stable_fingerprint(identity)
        self.state: dict[str, Any] = {
            "format": "bifocalgemma_pipeline_v1",
            "signature": self.signature,
            "identity": identity,
            "stages": {},
        }
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("signature") != self.signature:
                raise ValueError("Pipeline identity changed; use a new RUN_NAME.")
            if self.state.get("format") != "bifocalgemma_pipeline_v1" or not isinstance(
                self.state.get("stages"), dict
            ):
                raise ValueError("Invalid pipeline receipt schema.")
            if stable_fingerprint(self.state.get("identity")) != self.signature:
                raise ValueError("Pipeline receipt identity is corrupt.")

    def completed(self, stage, *, settings=None):
        item = self.state["stages"].get(stage, {})
        if item.get("status") != "COMPLETED":
            return False
        if not isinstance(item.get("files"), dict) or not item["files"]:
            raise ValueError("Completed stage has no valid evidence inventory.")
        if settings is not None and item.get("settings_fingerprint") != stable_fingerprint(
            settings
        ):
            return False
        for name, digest in item.get("files", {}).items():
            path = self.root / name
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError("Stage receipt artifact missing: " + name)
            if file_digest(path) != digest:
                raise ValueError("Stage receipt artifact changed: " + name)
        return True

    def start(self, stage):
        self.state["stages"][stage] = {"status": "RUNNING"}
        atomic_json(self.path, self.state)

    def finish(self, stage, files, result=None, *, settings=None):
        inventory = {}
        for source in files:
            path = Path(source).resolve()
            if not path.is_relative_to(self.root) or path.is_symlink() or not path.is_file():
                raise ValueError("Stage receipt path escaped run: " + str(path))
            inventory[path.relative_to(self.root).as_posix()] = file_digest(path)
        if not inventory:
            raise ValueError("A completed stage requires materialized evidence.")
        self.state["stages"][stage] = {"status": "COMPLETED", "files": inventory, "result": result}
        if settings is not None:
            self.state["stages"][stage]["settings_fingerprint"] = stable_fingerprint(settings)
        atomic_json(self.path, self.state)

    def fail(self, stage, error):
        self.state["stages"][stage] = {
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_json(self.path, self.state)


def retained_checkpoints(root, *, limit=2):
    """Validate complete numeric checkpoints before pruning within this exact run."""
    import re
    import shutil

    if type(limit) is not int or limit < 1:
        raise ValueError("Checkpoint retention must be positive.")
    root = Path(root).resolve()
    valid = []
    for path in root.glob("checkpoint-*"):
        if not re.fullmatch(r"checkpoint-\d{6,}", path.name):
            continue
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Checkpoint path escaped run directory.")
        if not (path / "READY.json").is_file():
            continue
        validate_checkpoint_files(path)
        valid.append(path)
    valid.sort(key=lambda path: int(path.name.split("-")[1]))
    for path in valid[:-limit]:
        if path.resolve().parent != root:
            raise ValueError("Refusing checkpoint deletion outside run.")
        shutil.rmtree(path)
    return [str(path) for path in valid[-limit:]]


def validate_checkpoint_files(path):
    path = Path(path)
    required = {"adapters.safetensors", "trainer_state.pt", "manifest.json"}
    marker = json.loads((path / "READY.json").read_text(encoding="utf-8"))
    if set(marker) != required or any(
        not (path / name).is_file() or file_digest(path / name) != marker[name] for name in required
    ):
        raise ValueError("Incomplete/corrupt checkpoint: " + str(path))
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def latest_checkpoint(root):
    """Never treat a partial directory as a resumable training checkpoint."""
    root = Path(root).resolve()
    candidates = sorted(root.glob("checkpoint-[0-9]*"), reverse=True)
    for path in candidates:
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Checkpoint path escaped run directory.")
        if (path / "READY.json").exists():
            validate_checkpoint_files(path)
            return str(path)
    return None
