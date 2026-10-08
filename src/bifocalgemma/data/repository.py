"""Pinned source schemas and Git snapshot materialization for offline repo SFT."""

import hashlib
import json
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bifocalgemma.contracts import RepairExample, RepositoryFile, RepositorySnapshot
from bifocalgemma.data.context import canonical_snapshot_hash, select_repository_context
from bifocalgemma.data.patches import (
    apply_patch_to_snapshot,
    git_environment,
    patch_paths,
    safe_repo_path,
)

SOURCE_REVISIONS = {
    "fixture": ("bifocalgemma/synthetic-repository-fixture", "fixture-v1", "train"),
    "fixture_eval": ("bifocalgemma/synthetic-repository-fixture", "fixture-v1", "eval"),
    "swe_fixer": (
        "internlm/SWE-Fixer-Train-110K",
        "9e3f1366be2c9a1a230d243b4008354436685501",
        "train",
    ),
    "swe_rebench_v2": (
        "nebius/SWE-rebench-V2",
        "10483de0f50fe5da545942705a76c6150171af7f",
        "train",
    ),
    "resq": ("Qurrent/RES-Q", "c0e850b1c90d1c0eb7d3b2f3c9aa6ec09c2c04fa", "eval"),
}


@dataclass(frozen=True)
class RepositoryTask:
    instance_id: str
    source: str
    source_revision: str
    role: str
    repo: str
    base_commit: str
    issue: str
    target_patch: str
    license: str
    test_script: str | None = None
    python_environment: str | None = None
    requirements: str | None = None
    language: str = "python"


def repo_identity(repo: str, aliases: dict[str, str] | None = None) -> str:
    value = repo.removeprefix("https://github.com/").removesuffix(".git").strip("/").lower()
    if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", value) or ".." in value:
        raise ValueError("unsupported_github_repo")
    aliases = aliases or {}
    seen = set()
    while value in aliases:
        if value in seen:
            raise ValueError("repo_alias_cycle")
        seen.add(value)
        value = aliases[value].lower()
        if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", value):
            raise ValueError("invalid_repo_alias")
    return value


def repository_task_from_row(source: str, raw: dict) -> RepositoryTask:
    if source not in SOURCE_REVISIONS:
        raise ValueError("unknown_repository_source")
    dataset_id, revision, role = SOURCE_REVISIONS[source]
    is_resq = source == "resq"
    repo = raw.get("repo_url" if is_resq else "repo")
    issue = raw.get("instruction" if is_resq else "problem_statement")
    patch = raw.get("solution_patch" if is_resq else "patch")
    commit = raw.get("base_commit")
    identifier = raw.get("id" if is_resq else "instance_id")
    if not all(isinstance(v, str) and v.strip() for v in (repo, issue, patch, commit, identifier)):
        raise ValueError("missing_repository_task_fields")
    assert isinstance(commit, str)
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise ValueError("invalid_base_commit")
    assert isinstance(repo, str) and isinstance(issue, str) and isinstance(patch, str)
    assert isinstance(commit, str) and isinstance(identifier, str)
    patch_paths(patch)
    return RepositoryTask(
        identifier,
        dataset_id,
        revision,
        role,
        repo_identity(repo),
        commit.lower(),
        issue.strip(),
        patch,
        str(raw.get("license") or raw.get("license_name") or "unknown"),
        raw.get("test_script") if role == "eval" else None,
        raw.get("testbed_environment") if role == "eval" else None,
        raw.get("requirements_txt") if role == "eval" else None,
        str(raw.get("language") or "python").lower(),
    )


def read_source_rows(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("source_json_requires_array")
        yield from data
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        raise ValueError("unsupported_source_file_format")


class GitSnapshotStore:
    """One mirror per original URL, immutable blobs; no task installs or tests."""

    def __init__(self, root: str | Path, *, allow_network: bool = False):
        self.root = Path(root).resolve()
        self.allow_network = allow_network

    @staticmethod
    def _git(path: Path, *args: str) -> bytes:
        result = subprocess.run(
            ["git", "--git-dir", str(path), *args],
            env=git_environment(),
            capture_output=True,
            timeout=120,
        )
        if result.returncode:
            raise ValueError(
                "snapshot_git_failed: " + result.stderr.decode("utf-8", "replace")[:300]
            )
        return result.stdout

    def snapshot(
        self, task: RepositoryTask, *, max_files: int = 5000, max_bytes: int = 50_000_000
    ) -> RepositorySnapshot:
        identity = repo_identity(task.repo)
        mirror = self.root / (hashlib.sha256(identity.encode()).hexdigest() + ".git")
        if not mirror.resolve().is_relative_to(self.root):
            raise ValueError("mirror_path_escape")
        if not mirror.exists():
            if not self.allow_network:
                raise FileNotFoundError("Snapshot mirror missing; network fetching is disabled.")
            self.root.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--mirror", f"https://github.com/{identity}.git", str(mirror)],
                env=git_environment(),
                capture_output=True,
                timeout=300,
            )
            if result.returncode:
                raise ValueError(
                    "snapshot_clone_failed: " + result.stderr.decode("utf-8", "replace")[:300]
                )
        try:
            commit = (
                self._git(mirror, "rev-parse", "--verify", task.base_commit + "^{commit}")
                .decode()
                .strip()
            )
        except ValueError:
            if not self.allow_network:
                raise
            self._git(mirror, "fetch", "origin", task.base_commit)
            commit = (
                self._git(mirror, "rev-parse", "--verify", task.base_commit + "^{commit}")
                .decode()
                .strip()
            )
        entries = self._git(mirror, "ls-tree", "-rlz", commit).split(b"\0")
        files = []
        total = 0
        for entry in entries:
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            mode, kind, oid, size = metadata.split()
            if kind != b"blob" or mode not in {b"100644", b"100755"}:
                raise ValueError("snapshot_links_or_submodules_not_supported")
            filename = safe_repo_path(name.decode("utf-8"))
            total += int(size)
            if total > max_bytes or len(files) >= max_files:
                raise ValueError("snapshot_resource_budget_exceeded")
            content = self._git(mirror, "cat-file", "blob", oid.decode())
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                # Context/SFT uses text only; native execution needs a complete real checkout.
                continue
            if "\0" in text:
                continue
            files.append(RepositoryFile(filename, text))
        return RepositorySnapshot(task.repo, commit, tuple(files))


def prepare_repository_record(
    task: RepositoryTask,
    snapshot: RepositorySnapshot,
    tokenizer,
    *,
    blocked_repos: set[str],
    aliases: dict[str, str] | None = None,
    max_context_tokens: int = 4096,
) -> tuple[dict, dict | None]:
    """Training export and hidden evaluation spec are written into separate files."""
    if task.role not in {"train", "eval"}:
        raise ValueError("invalid_repository_task_role")
    if repo_identity(snapshot.repo) != repo_identity(task.repo):
        raise ValueError("snapshot_repo_mismatch")
    if not snapshot.base_commit.lower().startswith(task.base_commit):
        raise ValueError("snapshot_commit_mismatch")
    canonical = repo_identity(task.repo, aliases)
    blocked = {repo_identity(repo, aliases) for repo in blocked_repos}
    if task.role == "train" and canonical in blocked:
        raise ValueError("benchmark_repository_quarantine")
    before_hash = canonical_snapshot_hash(snapshot)
    context, selection = select_repository_context(
        snapshot, task.issue, tokenizer, max_tokens=max_context_tokens
    )
    # Gold affects validation and target only, never context selection.
    fixed = apply_patch_to_snapshot(snapshot, task.target_patch)
    patch_hash = hashlib.sha256(task.target_patch.encode("utf-8")).hexdigest()
    row = {
        "record_type": "repository_patch_v1",
        "instance_id": task.instance_id,
        "source": task.source,
        "source_revision": task.source_revision,
        "role": task.role,
        "language": task.language,
        "instruction": task.issue,
        "context": context,
        "snapshot_sha256": before_hash,
        "group": canonical,
        "commit": snapshot.base_commit,
        "repos": canonical,
        "license": task.license,
        "context_selection": selection,
        "validation": "static_patch_apply_not_functional_tests",
    }
    if task.role == "train":
        row["target_patch"] = task.target_patch
        row["fixed_snapshot_sha256"] = canonical_snapshot_hash(fixed)
        row["target_patch_sha256"] = patch_hash
        return row, None
    hidden = {
        "instance_id": task.instance_id,
        "source": task.source,
        "repo": task.repo,
        "base_commit": snapshot.base_commit,
        "snapshot_sha256": before_hash,
        "gold_patch": task.target_patch,
        "test_script": task.test_script,
        "python_environment": task.python_environment,
        "requirements": task.requirements,
    }
    return row, hidden


def example_from_record(row: dict, snapshot: RepositorySnapshot, *, split: str) -> RepairExample:
    if canonical_snapshot_hash(snapshot) != row["snapshot_sha256"]:
        raise ValueError("snapshot_hash_mismatch")
    return RepairExample(
        row["instance_id"],
        row["source"],
        split,
        snapshot,
        row["instruction"],
        row.get("target_patch"),
    )
