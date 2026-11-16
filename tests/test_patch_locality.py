"""Patch application preserves the entire snapshot while writing only edited files."""

import pytest

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.patches import (
    apply_patch_to_snapshot,
    apply_patch_with_recount,
    diagnose_patch_application,
)


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


def test_diagnose_patch_application_rescues_recount_without_weakening_strict():
    snapshot = RepositorySnapshot("fixture/repo", "a" * 40, (RepositoryFile("a.py", "x=0\n"),))
    bad_count_patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,99 +1,99 @@\n-x=0\n+x=2\n"
    )
    fenced_headerless_patch = (
        "```patch\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,99 +1,99 @@\n-x=0\n+x=2\n```\n"
    )
    hunk_only_patch = "@@ -1,99 +1,99 @@\n-x=0\n+x=2\n"

    for candidate in (bad_count_patch, fenced_headerless_patch, hunk_only_patch):
        with pytest.raises(ValueError):
            apply_patch_to_snapshot(snapshot, candidate)

    diag_recount = diagnose_patch_application(snapshot, bad_count_patch)
    assert diag_recount.strict_applied is False
    assert diag_recount.recount_applied is True
    assert diag_recount.diagnostic_mode == "recount"
    assert diag_recount.snapshot is not None and diag_recount.snapshot.files[0].content == "x=2\n"
    assert diag_recount.normalized_patch is not None
    assert (
        apply_patch_to_snapshot(snapshot, diag_recount.normalized_patch).files[0].content == "x=2\n"
    )

    fixed_recount, canonical = apply_patch_with_recount(snapshot, bad_count_patch)
    assert fixed_recount.files[0].content == "x=2\n"
    assert apply_patch_to_snapshot(snapshot, canonical).files[0].content == "x=2\n"

    diag_fenced = diagnose_patch_application(snapshot, fenced_headerless_patch)
    assert diag_fenced.strict_applied is False
    assert diag_fenced.recount_applied is True
    assert diag_fenced.diagnostic_mode == "normalized_recount"
    assert diag_fenced.snapshot is not None and diag_fenced.snapshot.files[0].content == "x=2\n"
    fixed_fenced, fenced_canonical = apply_patch_with_recount(snapshot, fenced_headerless_patch)
    assert fixed_fenced.files[0].content == "x=2\n"
    assert apply_patch_to_snapshot(snapshot, fenced_canonical).files[0].content == "x=2\n"

    diag_hunk = diagnose_patch_application(snapshot, hunk_only_patch, default_path="a.py")
    assert diag_hunk.strict_applied is False
    assert diag_hunk.recount_applied is True
    assert diag_hunk.diagnostic_mode == "normalized_recount"
    fixed_hunk, _ = apply_patch_with_recount(snapshot, hunk_only_patch, default_path="a.py")
    assert fixed_hunk.files[0].content == "x=2\n"
    assert snapshot.files[0].content == "x=0\n"

    unsafe_patch = (
        "```diff\ndiff --git a/../escape.py b/../escape.py\n"
        "--- a/../escape.py\n+++ b/../escape.py\n@@ -1 +1 @@\n-x=0\n+x=2\n```\n"
    )
    rename_patch = (
        "diff --git a/a.py b/b.py\nrename from a.py\nrename to b.py\n"
        "--- a/a.py\n+++ b/b.py\n@@ -1 +1 @@\n-x=0\n+x=2\n"
    )
    for rejected in (unsafe_patch, rename_patch, "Here is prose instead of a diff."):
        with pytest.raises(ValueError):
            apply_patch_with_recount(snapshot, rejected)
        failed_diag = diagnose_patch_application(snapshot, rejected)
        assert failed_diag.strict_applied is False
        assert failed_diag.recount_applied is False
        assert failed_diag.diagnostic_mode == "failed"
