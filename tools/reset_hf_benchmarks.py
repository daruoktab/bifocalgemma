"""Reset selected stopped Hub runs' evaluation only; defaults to a read-only plan."""

import argparse
import io
import json
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

from bifocalgemma.training.benchmark_reset import remote_reset_plan
from bifocalgemma.training.hub import checked_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--reset-id", default="2026-10-26-exact-prompt-budget-v3")
    parser.add_argument("--archive", type=Path, default=Path("outputs/benchmark_reset"))
    parser.add_argument(
        "--apply", action="store_true", help="Requires all affected writers stopped."
    )
    args = parser.parse_args()
    api = HfApi()
    revision = api.model_info(args.repo).sha
    names = api.list_repo_files(args.repo, revision=revision)
    operations: list[CommitOperationAdd | CommitOperationDelete] = []
    metadata = {}
    new_metadata = {}
    deleted = []
    for prefix in args.prefix:
        checked_prefix(prefix)

        def read(relative):
            name = prefix + "/" + relative
            path = hf_hub_download(args.repo, name, revision=revision)
            value = Path(path).read_bytes()
            metadata[name] = value
            return json.loads(value)

        state = read("pipeline_state.json")
        # Evaluation reset is also safe for interrupted training: its cursor,
        # checkpoint tensors, optimizer and training receipt remain untouched.
        if not state.get("identity") or not isinstance(state.get("stages"), dict):
            raise ValueError("Refusing reset of an invalid pipeline receipt: " + prefix)
        receipt = (
            read("upload_complete.json") if prefix + "/upload_complete.json" in names else None
        )
        removed, replacements = remote_reset_plan(prefix, names, state, receipt, args.reset_id)
        if prefix + "/last_seen.json" in names:
            previous = read("last_seen.json")
            replacements[prefix + "/last_seen.json"] = json.dumps(
                {
                    "stage": "awaiting_resume",
                    "last_seen": previous.get("last_seen"),
                    "run": prefix,
                    "benchmark_reset_id": args.reset_id,
                    "status_updated_at": time.time(),
                    "liveness_note": "Benchmark results reset; last_seen is the previous notebook heartbeat, not reset time.",
                },
                indent=2,
            ).encode()
        new_metadata.update(replacements)
        deleted.extend(removed)
        operations.extend(CommitOperationDelete(path_in_repo=name) for name in removed)
        operations.extend(
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=io.BytesIO(value))
            for name, value in replacements.items()
        )
    print(
        json.dumps(
            {
                "source_revision": revision,
                "delete_files": deleted,
                "replace_metadata": list(new_metadata),
                "apply": args.apply,
            },
            indent=2,
        )
    )
    if not args.apply:
        return
    # Previous full evaluation files remain recoverable at this immutable Hub revision.
    archive = args.archive / revision
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "reset_plan.json").write_text(
        json.dumps(
            {"repo": args.repo, "revision": revision, "deleted": deleted, "prefixes": args.prefix},
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, value in metadata.items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="model",
        parent_commit=revision,
        operations=operations,
        commit_message="Reset evaluation artifacts; preserve trained checkpoints",
    )
    visible = set(api.list_repo_files(args.repo, revision=commit.oid))
    if any(name in visible for name in deleted):
        raise ValueError("Benchmark deletion verification failed.")
    for name, expected in new_metadata.items():
        remote = hf_hub_download(args.repo, name, revision=commit.oid)
        if Path(remote).read_bytes() != expected:
            raise ValueError("Reset metadata verification failed: " + name)

    # Compare Git blob identities without downloading any tensor/checkpoint bytes.
    def training_blobs(rev):
        return {
            entry.path: entry.blob_id
            for entry in api.list_repo_tree(args.repo, revision=rev, recursive=True)
            if hasattr(entry, "blob_id")
            and "/training/" in entry.path
            and not entry.path.endswith("/final_upload_receipt.json")
        }

    if training_blobs(revision) != training_blobs(commit.oid):
        raise ValueError("Training artifacts changed unexpectedly.")
    print(
        json.dumps(
            {
                "status": "RESET_VERIFIED",
                "revision": commit.oid,
                "deleted": len(deleted),
                "training_blobs_unchanged": True,
                "archive": str(archive),
            }
        )
    )


if __name__ == "__main__":
    main()
