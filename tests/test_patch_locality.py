"""Patch application preserves the entire snapshot while writing only edited files."""

import pytest

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.patches import apply_patch_to_snapshot


def test_modify_add_delete_preserves_unrelated_unicode_bytes(tmp_path, monkeypatch):
    from pathlib import Path

    snapshot = RepositorySnapshot(
        "fixture/repo",
        "a" * 40,
        (
            RepositoryFile("a.py", "x=0\n"),
            RepositoryFile("old.py", "y=1\n"),
            RepositoryFile("pkg/untouched.py", "# café\r\nmessage = '你好'\r\n"),
        ),
    )
    writes = []
    write = Path.write_bytes

    def traced(path, data):
        writes.append(path.name)
        return write(path, data)

    monkeypatch.setattr(Path, "write_bytes", traced)
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x=0\n+x=2\n"
    patch += "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-y=1\n"
    patch += "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+z=3\n"
    fixed = apply_patch_to_snapshot(snapshot, patch)
    files = {f.path: f.content for f in fixed.files}
    assert files == {
        "a.py": "x=2\n",
        "new.py": "z=3\n",
        "pkg/untouched.py": snapshot.files[2].content,
    }
    assert writes == ["a.py", "old.py"]
    assert snapshot.files[0].content == "x=0\n"


def test_missing_old_file_is_still_rejected():
    snapshot = RepositorySnapshot("fixture/repo", "a" * 40, (RepositoryFile("other.py", "x=0\n"),))
    patch = "diff --git a/missing.py b/missing.py\n--- a/missing.py\n+++ b/missing.py\n@@ -1 +1 @@\n-x=0\n+x=2\n"
    with pytest.raises(ValueError, match="patch_apply_failed"):
        apply_patch_to_snapshot(snapshot, patch)
