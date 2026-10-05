"""D04 local contract validation; not a SWE-smith downloader or test harness.

SWE-smith patch is bug-inducing. This deliberately restricted normalizer handles
UTF-8 modifications of existing files, and verifies both patch directions with Git.
"""

import difflib
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bifocalgemma.contracts import RepairExample, RepositoryFile, RepositorySnapshot


def snapshot_hash(snapshot: RepositorySnapshot) -> str:
    payload = [(f.path, f.content) for f in sorted(snapshot.files, key=lambda f: f.path)]
    if len({p for p, _ in payload}) != len(payload):
        raise ValueError("Duplicate snapshot paths.")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SWESmithRecord:
    instance_id: str
    dataset_revision: str
    buggy_ref: str
    buggy_snapshot_sha256: str
    snapshot: RepositorySnapshot
    issue: str
    bug_patch: str


@dataclass(frozen=True)
class NormalizedSWESmithRecord:
    example: RepairExample
    dataset_revision: str
    buggy_ref: str
    buggy_snapshot_sha256: str
    fixed_snapshot_sha256: str
    bug_patch_sha256: str
    repair_patch_sha256: str
    validation: str = "patch_roundtrip_only_not_official_tests"


def _git_apply(directory: Path, patch: str, *, reverse: bool = False, check: bool = False):
    command = ["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn"]
    if reverse:
        command.append("--reverse")
    if check:
        command.append("--check")
    result = subprocess.run(
        command + ["-"], input=patch.encode("utf-8"), cwd=directory, capture_output=True, timeout=30
    )
    if result.returncode:
        raise ValueError("Patch verification failed: " + result.stderr.decode("utf-8", "replace"))


def normalize_swesmith_record(record: SWESmithRecord, *, split: str) -> NormalizedSWESmithRecord:
    """Reverse bug patch on a verified local buggy snapshot, then verify repair.

    Does not infer refs from instance IDs, expose evaluator tests, or execute code.
    Unsupported binary/add/delete/rename patches fail rather than silently normalize.
    """
    if not record.dataset_revision or not record.buggy_ref or not record.issue.strip():
        raise ValueError("Dataset revision, explicit buggy ref and issue are required.")
    if record.snapshot.base_commit != record.buggy_ref:
        raise ValueError("Snapshot base_commit must identify the buggy ref in this local contract.")
    if snapshot_hash(record.snapshot) != record.buggy_snapshot_sha256:
        raise ValueError("Buggy snapshot hash mismatch.")
    if not record.bug_patch.startswith("diff --git "):
        raise ValueError("Expected a Git unified diff.")
    snapshot_files = {f.path: f.content for f in record.snapshot.files}
    sections = re.split(r"(?m)(?=^diff --git )", record.bug_patch)
    changed = []
    for section in filter(None, sections):
        header = re.match(r"diff --git a/([^\s]+) b/([^\s]+)\n", section)
        if not header or header[1] != header[2]:
            raise ValueError("Only unquoted existing-file modifications are supported locally.")
        path = header[1]
        RepositoryFile(path, "")
        if any(part.startswith(".") for part in Path(path).parts):
            raise ValueError(
                "Hidden metadata paths are not supported by the local fixture adapter."
            )
        if path not in snapshot_files or path in changed:
            raise ValueError("Patch path must identify one existing snapshot file exactly once.")
        if f"\n--- a/{path}\n+++ b/{path}\n" not in section:
            raise ValueError("Add/delete/rename/binary patches require a production adapter.")
        if any(
            line.startswith(("old mode ", "new mode ", "rename ", "copy ", "GIT binary"))
            for line in section.splitlines()
        ):
            raise ValueError("Only text modifications are supported locally.")
        changed.append(path)
    if not changed:
        raise ValueError("Empty patch.")
    with tempfile.TemporaryDirectory(prefix="bifocalgemma-patch-") as temporary:
        directory = Path(temporary)
        # Write only patch paths; never write source checkout or execute tests here.
        for path in changed:
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(snapshot_files[path], encoding="utf-8", newline="")
        _git_apply(directory, record.bug_patch, reverse=True, check=True)
        _git_apply(directory, record.bug_patch, reverse=True)
        fixed_files = dict(snapshot_files)
        repair = []
        for path in changed:
            with (directory / path).open(encoding="utf-8", newline="") as stream:
                fixed_files[path] = stream.read()
            if not snapshot_files[path].endswith("\n") or not fixed_files[path].endswith("\n"):
                raise ValueError("Local fixtures require newline-terminated UTF-8 source files.")
            repair.append(f"diff --git a/{path} b/{path}\n")
            repair.extend(
                difflib.unified_diff(
                    snapshot_files[path].splitlines(keepends=True),
                    fixed_files[path].splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        repair_patch = "".join(repair)
        # Verify the generated repair starting from the actual buggy snapshot.
        for path in changed:
            (directory / path).write_text(snapshot_files[path], encoding="utf-8", newline="")
        _git_apply(directory, repair_patch, check=True)
        _git_apply(directory, repair_patch)
        for path in changed:
            with (directory / path).open(encoding="utf-8", newline="") as stream:
                if stream.read() != fixed_files[path]:
                    raise ValueError("Repair did not reconstruct expected fixed snapshot.")
        _git_apply(directory, record.bug_patch, check=True)
        _git_apply(directory, record.bug_patch)
        for path in changed:
            with (directory / path).open(encoding="utf-8", newline="") as stream:
                if stream.read() != snapshot_files[path]:
                    raise ValueError("Bug/repair patch roundtrip mismatch.")
    fixed = RepositorySnapshot(
        record.snapshot.repo,
        "derived-fixed",
        tuple(RepositoryFile(path, content) for path, content in sorted(fixed_files.items())),
    )
    return NormalizedSWESmithRecord(
        RepairExample(
            record.instance_id,
            "SWE-bench/SWE-smith",
            split,
            record.snapshot,
            record.issue,
            repair_patch,
        ),
        record.dataset_revision,
        record.buggy_ref,
        record.buggy_snapshot_sha256,
        snapshot_hash(fixed),
        hashlib.sha256(record.bug_patch.encode("utf-8")).hexdigest(),
        hashlib.sha256(repair_patch.encode("utf-8")).hexdigest(),
    )
