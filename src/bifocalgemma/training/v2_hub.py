"""Verified snapshots, followed by pruning; lightweight remote last-seen status."""

import json
import re
import time
from pathlib import Path

from bifocalgemma.data.curation import file_digest
from bifocalgemma.training.benchmark_upload import retry_delay
from bifocalgemma.training.hub import checked_prefix
from bifocalgemma.training.stages import atomic_json, validate_checkpoint_files


def synchronize_v2(repo_id, root, prefix, *, token, require_checkpoint=True):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    checked_prefix(prefix)
    root = Path(root).resolve()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    checkpoints = sorted(
        p for p in (root / "training").glob("checkpoint-*") if (p / "READY.json").exists()
    )
    for p in checkpoints:
        validate_checkpoint_files(p)
    retained = checkpoints[-2:]
    if not retained and require_checkpoint:
        raise ValueError("No complete checkpoint to snapshot.")
    keep = {p.name for p in retained}
    payloads = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        relative = p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("Snapshot escaped run directory.")
        if not (
            relative == "pipeline_state.json"
            or relative.startswith(("training/", "benchmarks/", "diagnostics/"))
        ):
            continue
        m = re.match(r"training/(checkpoint-\d+)/", relative)
        if m and m[1] not in keep:
            continue
        if p.suffix not in {".json", ".jsonl", ".html", ".pt", ".safetensors"}:
            continue
        payloads[relative] = p
    receipt = {
        "status": "COMPLETED",
        "repo_id": repo_id,
        "prefix": prefix,
        "checkpoint_only": False,
        "retained_checkpoints": sorted(keep),
        "files": {n: file_digest(p) for n, p in payloads.items()},
        "format": "verified_snapshot_v2",
    }
    operations = [
        CommitOperationAdd(path_in_repo=prefix + "/" + n, path_or_fileobj=str(p))
        for n, p in payloads.items()
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=prefix + "/upload_complete.json",
            path_or_fileobj=json.dumps(receipt, indent=2).encode(),
        )
    )
    for attempt in range(3):
        parent = api.model_info(repo_id).sha
        try:
            commit = api.create_commit(
                repo_id,
                operations=operations,
                parent_commit=parent,
                commit_message="Save verified v2 checkpoint and results snapshot",
            )
            break
        except HfHubHTTPError as error:
            if error.response.status_code != 412 or attempt == 2:
                raise
    else:
        raise RuntimeError("Snapshot commit retry exhausted.")
    visible = set(api.list_repo_files(repo_id, revision=commit.oid))
    if not all(prefix + "/" + n in visible for n in payloads):
        raise ValueError("Snapshot inventory not visible.")
    for p in retained:
        remote = hf_hub_download(
            repo_id,
            prefix + "/training/" + p.name + "/READY.json",
            revision=commit.oid,
            token=token,
        )
        if Path(remote).read_bytes() != (p / "READY.json").read_bytes():
            raise ValueError("Remote checkpoint marker mismatch.")
    # New checkpoints are committed AND verified before any old remote file is deleted.
    pattern = re.compile(re.escape(prefix) + r"/training/(checkpoint-\d+)/")
    obsolete = [n for n in visible if (m := pattern.match(n)) and m[1] not in keep]
    revision = commit.oid
    if obsolete:
        deleted = api.create_commit(
            repo_id,
            operations=[CommitOperationDelete(path_in_repo=n) for n in obsolete],
            parent_commit=commit.oid,
            commit_message="Prune older checkpoints after verifying replacement",
        )
        revision = deleted.oid
    atomic_json(root / "upload_complete.json", receipt)
    return {"revision": revision, "retained_checkpoints": sorted(keep), "verified_at": time.time()}


def upload_initial_results(repo_id, root, prefix, *, token):
    return synchronize_v2(repo_id, root, prefix, token=token, require_checkpoint=False)


def synchronize_preflight(repo_id, root, prefix, *, token):
    """Publish guard evidence before the first checkpoint, without a model receipt."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    checked_prefix(prefix)
    root = Path(root).resolve()
    names = ["pipeline_state.json"]
    names.extend(
        n
        for n in (
            "diagnostics/execution_preflight.json",
            "diagnostics/calibration.json",
            "diagnostics/shared_profile.json",
        )
        if (root / n).is_file()
    )
    names.extend(
        p.relative_to(root).as_posix()
        for p in sorted((root / "diagnostics/calibration").glob("*.json"))
    )
    if len(names) < 2:
        raise ValueError("No calibration or memory-guard evidence is available.")
    if not all((root / n).is_file() and not (root / n).is_symlink() for n in names):
        raise ValueError("Preflight evidence is missing or unsafe.")
    payloads = {n: (root / n).read_bytes() for n in names}
    import hashlib

    hashes = {n: hashlib.sha256(data).hexdigest() for n, data in payloads.items()}
    receipt = {"format": "verified_preflight_v1", "files": hashes}
    operations = [
        CommitOperationAdd(path_in_repo=prefix + "/" + n, path_or_fileobj=data)
        for n, data in payloads.items()
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=prefix + "/diagnostics/preflight_upload_complete.json",
            path_or_fileobj=json.dumps(receipt, indent=2).encode(),
        )
    )
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True)
    for attempt in range(3):
        try:
            commit = api.create_commit(
                repo_id,
                parent_commit=api.model_info(repo_id).sha,
                operations=operations,
                commit_message="Save resumable fixed memory-guard evidence",
            )
            break
        except HfHubHTTPError as error:
            if error.response.status_code != 412 or attempt == 2:
                raise
    else:
        raise RuntimeError("Preflight commit retry exhausted.")
    for name, expected in hashes.items():
        remote = hf_hub_download(repo_id, prefix + "/" + name, revision=commit.oid, token=token)
        if file_digest(remote) != expected:
            raise ValueError("Remote preflight evidence checksum mismatch.")
    return {"revision": commit.oid, "verified_at": time.time()}


class RemoteHeartbeat:
    def __init__(self, repo_id, root, prefix, *, token, interval=300):
        self.repo_id = repo_id
        self.root = Path(root)
        self.prefix = checked_prefix(prefix)
        self.token = token
        self.interval = interval
        self.last_success = 0.0
        self.retry_at = 0.0
        self.failures = 0
        self.snapshot = {}

    def __call__(self, event, *, force=False):
        from huggingface_hub import CommitOperationAdd, HfApi

        now = time.time()
        if event.get("stage") != self.snapshot.get("stage"):
            for key in ("items_done", "items_total"):
                self.snapshot.pop(key, None)
        # Keep counters across fine-grained events; never carry tokens or examples.
        for key in (
            "stage",
            "step",
            "max_steps",
            "items_done",
            "items_total",
            "last_checkpoint",
            "train_loss",
            "dev_loss",
            "error_type",
            "error",
            "last_remote_checkpoint",
            "remote_revision",
            "final_synced",
            "upload_pending",
            "batch_size",
            "accumulation",
            "attention_backend",
            "query_chunk_size",
            "loss_chunk_size",
            "preflight_phase",
            "gpu_allocated_gib",
            "gpu_reserved_gib",
            "gpu_peak_allocated_gib",
            "gpu_peak_reserved_gib",
            "gpu_free_gib",
        ):
            if key in event:
                self.snapshot[key] = event[key]
        payload = {
            **self.snapshot,
            "last_seen": now,
            "run": self.prefix,
            "liveness_note": "Snapshot only; stale last_seen does not establish whether Molab is alive.",
        }
        atomic_json(self.root / "last_seen.json", payload)
        if now < self.retry_at or (not force and now - self.last_success < self.interval):
            return
        try:
            api = HfApi(token=self.token)
            api.create_repo(self.repo_id, exist_ok=True)
            parent = api.model_info(self.repo_id).sha
            api.create_commit(
                self.repo_id,
                parent_commit=parent,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=self.prefix + "/last_seen.json",
                        path_or_fileobj=json.dumps(payload, indent=2).encode(),
                    )
                ],
                commit_message="Update v2 last-seen status",
            )
            self.last_success = now
            self.failures = 0
        except Exception as error:
            delay = retry_delay(error, self.failures + 1, now)
            self.failures += 1
            self.retry_at = now + (delay if delay is not None else 60.0)
            atomic_json(
                self.root / "heartbeat_upload_status.json",
                {
                    "status": "DEFERRED",
                    "next_retry_at": self.retry_at,
                    "error_type": type(error).__name__,
                },
            )
