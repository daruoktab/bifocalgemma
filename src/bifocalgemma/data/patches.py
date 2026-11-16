"""Strict UTF-8 Git patch application on isolated repository snapshots."""

import difflib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot


@dataclass(frozen=True)
class PatchApplicationDiagnostic:
    """Non-destructive strict vs. hunk-recount / flexible header patch diagnostic."""

    strict_applied: bool
    recount_applied: bool
    diagnostic_mode: str
    strict_error: str = ""
    recount_error: str = ""
    normalized_patch: str | None = None
    snapshot: RepositorySnapshot | None = None


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


_BANNED_PATCH_RE = re.compile(
    r"^(GIT binary patch|Binary files |rename |copy |similarity index|.*mode (120000|160000))",
    re.M,
)


def patch_paths(patch: str) -> tuple[str, ...]:
    """Support text modifications/additions/deletions; reject renames/binary/links."""
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("empty_patch")
    if _BANNED_PATCH_RE.search(patch):
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


def _apply_validated_patch(
    snapshot: RepositorySnapshot, patch: str, *, recount: bool = False
) -> RepositorySnapshot:
    changed_paths = set(patch_paths(patch))
    paths = [safe_repo_path(f.path) for f in snapshot.files]
    if len(set(paths)) != len(paths) or len({p.casefold() for p in paths}) != len(paths):
        raise ValueError("duplicate_or_case_colliding_snapshot_path")
    with tempfile.TemporaryDirectory(prefix="bifocal-patch-") as name:
        root = Path(name).resolve()
        for file in snapshot.files:
            if file.path not in changed_paths:
                continue
            destination = (root / file.path).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("snapshot_path_escape")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.content.encode("utf-8"))
        command = ["git", "-c", "core.autocrlf=false", "apply"]
        if recount:
            command.append("--recount")
        command.append("--whitespace=nowarn")
        environment = git_environment()
        # Never discover a parent checkout when TMPDIR/TEMP is inside a repo:
        # git apply can otherwise silently skip paths outside its cwd prefix.
        environment["GIT_CEILING_DIRECTORIES"] = str(root.parent)
        for check in (True, False):
            result = subprocess.run(
                command + (["--check"] if check else []) + ["-"],
                input=patch.encode("utf-8"),
                cwd=root,
                env=environment,
                capture_output=True,
                timeout=30,
            )
            if result.returncode:
                raise ValueError(
                    "patch_apply_failed: " + result.stderr.decode("utf-8", "replace")[:500]
                )
        # Git apply only needs the files named by the already-validated patch.
        # Preserve every unrelated file byte-for-byte without materializing it.
        files = {file.path: file for file in snapshot.files if file.path not in changed_paths}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("patch_created_symlink")
            if path.is_file():
                if not path.resolve().is_relative_to(root):
                    raise ValueError("patched_path_escape")
                relative = path.relative_to(root).as_posix()
                if relative not in changed_paths:
                    raise ValueError("patch_created_unexpected_path")
                files[relative] = RepositoryFile(relative, path.read_bytes().decode("utf-8"))
        # TemporaryDirectory owns this exact resolved target; no external paths are removed.
        if root != Path(name).resolve():
            raise ValueError("temporary_root_changed")
        return RepositorySnapshot(
            snapshot.repo, snapshot.base_commit, tuple(files[k] for k in sorted(files))
        )


def apply_patch_to_snapshot(snapshot: RepositorySnapshot, patch: str) -> RepositorySnapshot:
    return _apply_validated_patch(snapshot, patch, recount=False)


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


def _git_text_lines(text: str) -> list[str]:
    pieces = text.split("\n")
    return [line + "\n" for line in pieces[:-1]] + ([pieces[-1]] if pieces[-1] else [])


def _build_canonical_patch(
    before_snapshot: RepositorySnapshot,
    after_snapshot: RepositorySnapshot,
    changed_paths: set[str],
) -> str:
    before_map = {file.path: file.content for file in before_snapshot.files}
    after_map = {file.path: file.content for file in after_snapshot.files}
    parts: list[str] = []
    for path in sorted(changed_paths):
        before = before_map.get(path)
        after = after_map.get(path)
        if before == after:
            continue
        lines = list(
            difflib.unified_diff(
                _git_text_lines(before or ""),
                _git_text_lines(after or ""),
                fromfile="/dev/null" if before is None else f"a/{path}",
                tofile="/dev/null" if after is None else f"b/{path}",
                lineterm="\n",
            )
        )
        if not lines:
            continue
        parts.append(f"diff --git a/{path} b/{path}\n")
        if before is None:
            parts.append("new file mode 100644\n")
        elif after is None:
            parts.append("deleted file mode 100644\n")
        for line in lines:
            parts.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(parts)


def _strip_diff_header_path(raw: str, prefix: str) -> str | None:
    token = raw.split("\t", 1)[0].strip()
    if token == "/dev/null":
        return None
    if token.startswith(prefix):
        token = token[len(prefix) :]
    return safe_repo_path(token)


def _is_file_header_pair(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    old_line, new_line = lines[index], lines[index + 1]
    if not (old_line.startswith("--- ") and new_line.startswith("+++ ")):
        return False
    if index == 0:
        return True
    if index + 2 >= len(lines) or not lines[index + 2].startswith("@@ "):
        return False
    return (old_line.startswith("--- a/") or old_line == "--- /dev/null") and (
        new_line.startswith("+++ b/") or new_line == "+++ /dev/null"
    )


def _add_missing_git_headers(body: str) -> str | None:
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    found_header = False
    while i < len(lines):
        if _is_file_header_pair(lines, i):
            old_path = _strip_diff_header_path(lines[i][4:], "a/")
            new_path = _strip_diff_header_path(lines[i + 1][4:], "b/")
            rel = new_path if old_path is None else old_path
            if rel is None or (old_path is not None and new_path is not None and old_path != new_path):
                raise ValueError("unsupported_rename")
            rel = safe_repo_path(rel)
            out.append(f"diff --git a/{rel} b/{rel}")
            if old_path is None:
                out.append("new file mode 100644")
            elif new_path is None:
                out.append("deleted file mode 100644")
            out.append(f"--- {'/dev/null' if old_path is None else 'a/' + rel}")
            out.append(f"+++ {'/dev/null' if new_path is None else 'b/' + rel}")
            found_header = True
            i += 2
            continue
        out.append(lines[i])
        i += 1
    if not found_header:
        return None
    return "\n".join(out) + "\n"


def _normalize_flexible_unified_diff(
    text: str, *, default_path: str | None = None
) -> tuple[str, bool]:
    """Return ``(candidate_patch, headers_repaired)`` without weakening path/op safety."""
    try:
        return extract_unified_diff(text), False
    except ValueError:
        pass
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty_patch")
    body = text.strip("\r\n")
    if body.startswith("```"):
        match = re.fullmatch(r"```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n```", body, re.S)
        if match is None:
            raise ValueError("invalid_patch_fence")
        body = match.group(1)
    body = body.replace("\r\n", "\n").lstrip("\n")
    if not body.strip():
        raise ValueError("empty_patch")
    if _BANNED_PATCH_RE.search(body):
        raise ValueError("unsupported_patch_operation")
    if body.startswith("diff --git "):
        candidate = body if body.endswith("\n") else body + "\n"
    elif body.startswith("--- "):
        candidate = _add_missing_git_headers(body)
        if candidate is None:
            raise ValueError("missing_git_diff_header")
    elif body.startswith("@@ ") and default_path is not None:
        rel = safe_repo_path(default_path)
        suffix = body if body.endswith("\n") else body + "\n"
        candidate = f"diff --git a/{rel} b/{rel}\n--- a/{rel}\n+++ b/{rel}\n{suffix}"
    else:
        raise ValueError("unexpected_patch_prose")
    patch_paths(candidate)
    return candidate, True


def apply_patch_with_recount(
    snapshot: RepositorySnapshot, text: str, *, default_path: str | None = None
) -> tuple[RepositorySnapshot, str]:
    """Diagnostic-only helper applying flexible unified diffs with ``git apply --recount``.

    Returns ``(patched_snapshot, canonical_patch)`` without modifying ``snapshot`` or
    weakening strict :func:`apply_patch_to_snapshot` semantics.
    """
    candidate, _ = _normalize_flexible_unified_diff(text, default_path=default_path)
    changed_paths = set(patch_paths(candidate))
    fixed = _apply_validated_patch(snapshot, candidate, recount=True)
    canonical = _build_canonical_patch(snapshot, fixed, changed_paths)
    return fixed, canonical or candidate


def diagnose_patch_application(
    snapshot: RepositorySnapshot, text: str, *, default_path: str | None = None
) -> PatchApplicationDiagnostic:
    """Compare strict patch application against non-destructive ``--recount`` diagnostics."""
    try:
        strict_patch = extract_unified_diff(text)
        fixed = apply_patch_to_snapshot(snapshot, strict_patch)
        return PatchApplicationDiagnostic(
            strict_applied=True,
            recount_applied=True,
            diagnostic_mode="strict",
            normalized_patch=strict_patch,
            snapshot=fixed,
        )
    except (ValueError, OSError, subprocess.TimeoutExpired) as strict_exc:
        strict_error = str(strict_exc)[:500]
    try:
        candidate, headers_repaired = _normalize_flexible_unified_diff(
            text, default_path=default_path
        )
        changed_paths = set(patch_paths(candidate))
        recounted = _apply_validated_patch(snapshot, candidate, recount=True)
        canonical = _build_canonical_patch(snapshot, recounted, changed_paths)
        return PatchApplicationDiagnostic(
            strict_applied=False,
            recount_applied=True,
            diagnostic_mode="normalized_recount" if headers_repaired else "recount",
            strict_error=strict_error,
            normalized_patch=canonical or candidate,
            snapshot=recounted,
        )
    except (ValueError, OSError, subprocess.TimeoutExpired) as recount_exc:
        return PatchApplicationDiagnostic(
            strict_applied=False,
            recount_applied=False,
            diagnostic_mode="failed",
            strict_error=strict_error,
            recount_error=str(recount_exc)[:500],
        )
