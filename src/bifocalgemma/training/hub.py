"""Scoped checkpoint synchronization and pinned remote recovery for sequential runs."""

import json
import re
from pathlib import Path, PurePosixPath

from bifocalgemma.data.curation import file_digest
from bifocalgemma.training.stages import atomic_json, validate_checkpoint_files


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
    previous = api.model_info(repo_id).sha
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
        validate_checkpoint_files(path)
    receipt_path = root / "upload_complete.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for relative, digest in receipt["files"].items():
            if file_digest(root / relative) != digest:
                raise ValueError("Remote run receipt hash mismatch.")
    return {"revision": revision, "prefix": prefix}
