"""Strict UTF-8 Git patch application on isolated repository snapshots."""

import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot


def safe_repo_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or ".git" in path.parts
        or "\\" in value
        or ":" in value
        or any(char.isspace() or ord(char) < 32 for char in value)
    ):
        raise ValueError("unsupported_repository_path")
    return value


def patch_paths(patch: str) -> tuple[str, ...]:
    """Support text modifications/additions/deletions; reject renames/binary/links."""
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("empty_patch")
    if re.search(
        r"^(GIT binary patch|Binary files |rename |copy |similarity index|.*mode (120000|160000))",
        patch,
        re.M,
    ):
        raise ValueError("unsupported_patch_operation")
    headers = re.findall(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M)
    if not headers:
        raise ValueError("missing_git_diff_header")
    paths = []
    for old, new in headers:
        if old != new:
            raise ValueError("unsupported_rename")
        paths.append(safe_repo_path(old))
    for prefix, name in re.findall(r"^(---|\+\+\+) (.+)$", patch, re.M):
        if name == "/dev/null":
            continue
        expected = "a/" if prefix == "---" else "b/"
        if not name.startswith(expected) or safe_repo_path(name[2:]) not in paths:
            raise ValueError("patch_header_path_mismatch")
    if len(set(paths)) != len(paths) or len({p.casefold() for p in paths}) != len(paths):
        raise ValueError("duplicate_or_case_colliding_patch_path")
    if not re.search(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", patch, re.M):
        raise ValueError("missing_patch_hunk")
    return tuple(paths)


def git_environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
    return env


def apply_patch_to_snapshot(snapshot: RepositorySnapshot, patch: str) -> RepositorySnapshot:
    patch_paths(patch)
    paths = [safe_repo_path(f.path) for f in snapshot.files]
    if len(set(paths)) != len(paths) or len({p.casefold() for p in paths}) != len(paths):
        raise ValueError("duplicate_or_case_colliding_snapshot_path")
    with tempfile.TemporaryDirectory(prefix="bifocal-patch-") as name:
        root = Path(name).resolve()
        for file in snapshot.files:
            destination = (root / file.path).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("snapshot_path_escape")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.content.encode("utf-8"))
        command = ["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn"]
        for check in (True, False):
            result = subprocess.run(
                command + (["--check"] if check else []) + ["-"],
                input=patch.encode("utf-8"),
                cwd=root,
                env=git_environment(),
                capture_output=True,
                timeout=30,
            )
            if result.returncode:
                raise ValueError(
                    "patch_apply_failed: " + result.stderr.decode("utf-8", "replace")[:500]
                )
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("patch_created_symlink")
            if path.is_file():
                if not path.resolve().is_relative_to(root):
                    raise ValueError("patched_path_escape")
                files.append(
                    RepositoryFile(
                        path.relative_to(root).as_posix(), path.read_bytes().decode("utf-8")
                    )
                )
        # TemporaryDirectory owns this exact resolved target; no external paths are removed.
        if root != Path(name).resolve():
            raise ValueError("temporary_root_changed")
        return RepositorySnapshot(snapshot.repo, snapshot.base_commit, tuple(files))


def extract_unified_diff(text: str) -> str:
    """Only accept a raw diff or one complete diff/patch fence, not prose snippets."""
    text = text.strip("\r\n")
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:diff|patch)?\s*\n(.*?)\n```", text, re.S)
        if match is None:
            raise ValueError("invalid_patch_fence")
        text = match.group(1)
    if not text.startswith("diff --git "):
        raise ValueError("unexpected_patch_prose")
    patch = text if text.endswith("\n") else text + "\n"
    patch_paths(patch)
    return patch
