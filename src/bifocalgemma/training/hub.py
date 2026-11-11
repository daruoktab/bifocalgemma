"""Scoped checkpoint synchronization and pinned remote recovery for sequential runs."""

import json
import logging
import re
from pathlib import Path, PurePosixPath

from bifocalgemma.data.curation import file_digest
from bifocalgemma.training.stages import atomic_json, validate_checkpoint_files

logger = logging.getLogger(__name__)


def checked_prefix(prefix):
    path = PurePosixPath(prefix)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != prefix
        or len(path.parts) not in {3, 4}
        or (len(path.parts) == 4 and path.parts[-1] not in {"base", "optimized"})
    ):
        raise ValueError("Expected family/experiment/run[/base|optimized] Hub prefix.")
    return prefix


def synchronize_run(repo_id, root, prefix, *, token, checkpoint_only=False):
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(3):
        try:
            return _synchronize_once(
                repo_id, root, prefix, token=token, checkpoint_only=checkpoint_only
            )
        except HfHubHTTPError as error:
            if error.response.status_code != 412 or attempt == 2:
                raise
            # Rebuild operations against the fresh HEAD, retaining the parent guard.
    raise AssertionError("unreachable")


def _synchronize_once(repo_id, root, prefix, *, token, checkpoint_only=False):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    checked_prefix(prefix)
    root = Path(root).resolve()
    if not token:
        raise ValueError("A write token is required for checkpoint synchronization.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    checkpoint_root = root / "training"
    retained = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if (path / "READY.json").exists():
            validate_checkpoint_files(path)
            retained.append(path.name)
    if len(retained) > 2:
        raise ValueError("Prune local checkpoints to two before upload.")
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Upload path escaped the run directory.")
        if any(part.startswith(".") for part in PurePosixPath(relative).parts):
            continue
        if relative in {"upload_complete.json", "failure.json"}:
            continue
        allowed = relative == "pipeline_state.json" or relative.startswith(
            ("training/", "benchmarks/")
        )
        if not allowed:
            continue
        if checkpoint_only and (
            relative.startswith("benchmarks/") or path.name in {"events.jsonl", "monitor.html"}
        ):
            continue
        if path.suffix not in {".json", ".jsonl", ".safetensors", ".pt", ".html"}:
            continue
        files.append((relative, path))
    if not files or not retained:
        raise ValueError("No complete training checkpoint to upload.")
    previous = api.model_info(repo_id, timeout=30).sha
    remote = api.list_repo_files(repo_id, revision=previous)
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=prefix + "/" + name, path_or_fileobj=str(path))
        for name, path in files
    ]
    pattern = re.compile(re.escape(prefix) + r"/training/(checkpoint-\d{6,})/(.+)")
    for name in remote:
        match = pattern.fullmatch(name)
        if match and match[1] not in retained:
            operations.append(CommitOperationDelete(path_in_repo=name))
    result = {
        "status": "COMPLETED",
        "repo_id": repo_id,
        "prefix": prefix,
        "checkpoint_only": checkpoint_only,
        "retained_checkpoints": sorted(retained),
        "files": {name: file_digest(path) for name, path in files},
    }
    import io

    # Refresh the receipt in every atomic commit, including checkpoint-only uploads.
    # Otherwise a previous full receipt can refer to replaced history/removed checkpoints.
    marker_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    operations.append(
        CommitOperationAdd(
            path_in_repo=prefix + "/upload_complete.json",
            path_or_fileobj=io.BytesIO(marker_bytes),
        )
    )
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        parent_commit=previous,
        commit_message="Save sequential run with two retained checkpoints",
    )
    uploaded_names = {prefix + "/" + name for name, _ in files}
    visible = set(api.list_repo_files(repo_id, revision=commit.oid))
    if not uploaded_names <= visible:
        raise ValueError("Run upload verification failed.")
    from huggingface_hub import hf_hub_download

    for name in retained:
        marker = hf_hub_download(
            repo_id, prefix + "/training/" + name + "/READY.json", revision=commit.oid, token=token
        )
        if Path(marker).read_bytes() != (checkpoint_root / name / "READY.json").read_bytes():
            raise ValueError("Remote checkpoint marker differs after upload.")
    atomic_json(root / "upload_complete.json", result)
    return {**result, "revision": commit.oid}


def recover_remote_run(repo_id, root, prefix, *, token):
    """Only recover this run, pinned to one remote revision; never download base weights."""
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(root).resolve()
    checked_prefix(prefix)
    if (root / "pipeline_state.json").exists():
        return None
    api = HfApi(token=token)
    if not api.repo_exists(repo_id):
        return None
    revision = api.model_info(repo_id).sha
    names = api.list_repo_files(repo_id, revision=revision)
    if prefix + "/pipeline_state.json" not in names:
        return None
    if root.exists() and any(root.iterdir()):
        non_trivial = [
            p
            for p in root.iterdir()
            if p.name not in {"monitor.html", ".gitkeep"}
            and not (p.is_dir() and not any(p.iterdir()))
        ]
        if non_trivial:
            raise ValueError(
                "Local run has artifacts without a pipeline receipt; inspect before recovery."
            )
    for name in names:
        if not name.startswith(prefix + "/"):
            continue
        relative = name[len(prefix) + 1 :]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or ".." in PurePosixPath(relative).parts:
            raise ValueError("Remote run path escaped recovery directory.")
        source = hf_hub_download(repo_id, name, revision=revision, token=token)
        path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copyfile(source, path)
    for path in (root / "training").glob("checkpoint-*"):
        try:
            validate_checkpoint_files(path)
        except Exception as _ckpt_v_exc:
            logger.warning("Downloaded checkpoint %s incomplete: %s", path, _ckpt_v_exc)
    receipt_path = root / "upload_complete.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for relative, digest in receipt.get("files", {}).items():
            target_p = root / relative
            if target_p.exists():
                actual_digest = file_digest(target_p)
                if actual_digest != digest:
                    if "checkpoint-" in relative:
                        raise ValueError(f"Remote run receipt hash mismatch for checkpoint: {relative}")
                    else:
                        logger.warning(
                            "Remote run receipt hash mismatch for non-checkpoint file %s (expected %s, got %s)",
                            relative,
                            digest,
                            actual_digest,
                        )
    return {"revision": revision, "prefix": prefix}


def retry_delay(error, failures, now):
    """None means fatal; do not hide permissions, disk, or integrity errors."""
    import math
    from datetime import timezone
    from email.utils import parsedate_to_datetime

    import httpx
    from huggingface_hub.utils import parse_ratelimit_headers

    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    if code not in {408, 409, 412, 429, 500, 502, 503, 504} and not isinstance(
        error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
    ):
        return None
    if code == 409:
        delay = min(60, 5 * 2 ** min(max(0, failures - 1), 4))
    else:
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


class RemoteHeartbeat:
    """Periodic telemetry and heartbeat publisher to Hugging Face Hub."""

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
        import time

        from huggingface_hub import CommitOperationAdd, HfApi

        now = time.time()
        if event.get("stage") != self.snapshot.get("stage"):
            for key in ("items_done", "items_total"):
                self.snapshot.pop(key, None)
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
            "liveness_note": "Snapshot only; stale last_seen does not establish whether training is alive.",
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
                commit_message="Update last-seen status",
            )
            self.last_success = now
            self.failures = 0
        except Exception as error:
            delay = retry_delay(error, self.failures + 1, now)
            self.failures += 1
            self.retry_at = now + (delay if delay is not None else 60.0)
            logger.warning("Remote heartbeat commit failed (%s): %s", type(error).__name__, error)
            atomic_json(
                self.root / "heartbeat_upload_status.json",
                {
                    "status": "DEFERRED",
                    "next_retry_at": self.retry_at,
                    "error_type": type(error).__name__,
                },
            )


def benchmark_directory_name(settings):
    """Reuse consistent Hub folder addressing for evaluation benchmarks."""
    from bifocalgemma.data.offline_sft import stable_fingerprint

    storage = {**settings, "protocol": "greedy_cached_normalized_merged_v2"}
    return (
        settings["snapshot"]
        + "-"
        + settings["benchmark"]["collection"]
        + "-"
        + stable_fingerprint(storage)[:12]
    )


def target_fingerprint(examples):
    """Compute dataset fingerprint across item target IDs."""
    import hashlib
    import struct

    digest = hashlib.sha256()
    for item in examples:
        identifier = item.instance_id.encode()
        digest.update(struct.pack("<I", len(identifier)))
        digest.update(identifier)
        digest.update(struct.pack("<I", len(item.target_ids)))
        digest.update(struct.pack("<" + "I" * len(item.target_ids), *item.target_ids))
    return digest.hexdigest()


def download_model(spec, cache, token, progress):
    """Download audited backbone assets and instantiate local native model."""
    import hashlib
    import shutil

    from huggingface_hub import hf_hub_download

    from bifocalgemma.assets import load_asset_json
    from bifocalgemma.data.tokenization import load_verified_tokenizer
    from bifocalgemma.models.v2.pretrained import load_local_native_model

    manifest = load_asset_json("tokenizer_manifest.json")
    weights = []
    provenance = []
    for ref in [spec.decoder] + ([spec.encoder] if spec.encoder else []):
        assets = manifest[ref.model_id]
        if assets["revision"] != ref.revision:
            raise ValueError("Backbone revision is not audited.")
        raw = Path(cache) / "tokenizers/raw" / ref.model_id.replace("/", "__")
        raw.mkdir(parents=True, exist_ok=True)
        for name, digest in assets["files"].items():
            path = Path(
                hf_hub_download(
                    ref.model_id,
                    name,
                    revision=ref.revision,
                    token=token,
                    cache_dir=str(Path(cache) / "hub"),
                )
            )
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Pinned asset hash mismatch: " + name)
            shutil.copyfile(path, raw / name)
        load_verified_tokenizer(ref, Path(cache) / "tokenizers")
        progress({"stage": "checkpoint_download", "message": ref.model_id + " @ " + ref.revision})
        path = hf_hub_download(
            ref.model_id,
            "model.safetensors",
            revision=ref.revision,
            token=token,
            cache_dir=str(Path(cache) / "hub"),
        )
        weights.append(path)
        provenance.append(
            {"model_id": ref.model_id, "revision": ref.revision, "assets": assets["files"]}
        )
    model, source = load_local_native_model(
        spec, weights[0], weights[1] if spec.encoder else None, seed=17, headroom_gib=8
    )
    source.update(source_identity_verified=True, verified_sources=provenance)
    return model, source

