"""Archive identity, static licensing and read-only benchmark-index boundaries."""

import io
import tarfile

import pytest

from bifocalgemma.data.curation import CodeOverlapIndex
from bifocalgemma.data.repository_corpus import (
    patch_license_evidence,
    read_text_archive,
    recognize_license,
)

MIT = 'Permission is hereby granted, free of charge, to any person obtaining a copy. THE SOFTWARE IS PROVIDED "AS IS".'


def archive(path, contents, commit="a" * 40):
    with tarfile.open(path, "w:gz") as handle:
        for name, content in contents.items():
            item = tarfile.TarInfo("repo-" + commit + "/" + name)
            payload = content.encode()
            item.size = len(payload)
            handle.addfile(item, io.BytesIO(payload))


def test_archive_commit_budget_and_static_license(tmp_path):
    path = tmp_path / "snapshot.tar.gz"
    archive(path, {"LICENSE.rst": MIT, "pkg/a.py": "value = 0\n"})
    snapshot, coverage = read_text_archive(path, "owner/repo", "a" * 40)
    assert coverage["functional_checkout_complete"] is False
    assert snapshot.base_commit == "a" * 40
    patch = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-value = 0\n+value = 1\n"
    evidence = patch_license_evidence(snapshot, patch)
    assert evidence[0]["licenses"][0]["license"] == "mit"
    with pytest.raises(ValueError, match="commit_identity"):
        read_text_archive(path, "owner/repo", "b" * 40)
    with pytest.raises(ValueError, match="resource_budget"):
        read_text_archive(path, "owner/repo", "a" * 40, max_bytes=2)


def test_nested_licensing_overrides_root_and_traversal_rejected(tmp_path):
    path = tmp_path / "snapshot.tar.gz"
    archive(path, {"LICENSE": MIT, "vendor/LICENSE": "Proprietary.", "vendor/x.py": "x=0\n"})
    snapshot, _ = read_text_archive(path, "owner/repo", "a" * 40)
    patch = "diff --git a/vendor/x.py b/vendor/x.py\n--- a/vendor/x.py\n+++ b/vendor/x.py\n@@ -1 +1 @@\n-x=0\n+x=1\n"
    with pytest.raises(ValueError, match="license_not_admitted"):
        patch_license_evidence(snapshot, patch)
    archive(path, {"../../escape": "bad"})
    with pytest.raises(ValueError, match="path_escape"):
        read_text_archive(path, "owner/repo", "a" * 40)
    assert recognize_license("SPDX-License-Identifier: GPL-3.0") == "unknown"


def test_read_only_index_cannot_clear_or_write(tmp_path):
    path = tmp_path / "index.sqlite"
    index = CodeOverlapIndex(storage_path=path)
    code = "\n".join(f"value_{i} = {i}" for i in range(30))
    index.add("benchmark", code)
    index.close()
    reopened = CodeOverlapIndex(storage_path=path, read_only=True)
    assert reopened.match(code)["reference"] == "benchmark"
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        reopened.add("changed", code)
    reopened.close()


def test_license_document_include_is_resolved_inside_snapshot(tmp_path):
    path = tmp_path / "snapshot.tar.gz"
    archive(
        path,
        {
            "LICENSE.rst": MIT,
            "docs/license.rst": "MIT License\n.. include:: ../LICENSE.rst\n",
            "docs/example.py": "x=0\n",
        },
    )
    snapshot, _ = read_text_archive(path, "owner/repo", "a" * 40)
    patch = "diff --git a/docs/example.py b/docs/example.py\n--- a/docs/example.py\n+++ b/docs/example.py\n@@ -1 +1 @@\n-x=0\n+x=1\n"
    evidence = patch_license_evidence(snapshot, patch)
    assert evidence[0]["licenses"][0]["path"] == "LICENSE.rst"
    assert evidence[0]["licenses"][0]["license"] == "mit"


def test_preparation_ledger_preserves_records_and_rejects_new_settings(tmp_path):
    from bifocalgemma.data.repository_corpus import RepositoryPreparationLedger

    path = tmp_path / "ledger.sqlite"
    ledger = RepositoryPreparationLedger(path, {"source": "fixture", "budget": 64})
    ledger.save(0, {"exclusion": {"reason": "non_python"}})
    ledger.save(1, {"record": {"instance_id": "x"}, "lineage": "hash"})
    ledger.close()
    resumed = RepositoryPreparationLedger(path, {"source": "fixture", "budget": 64})
    assert list(resumed.rows()) == [
        (0, {"exclusion": {"reason": "non_python"}}),
        (1, {"record": {"instance_id": "x"}, "lineage": "hash"}),
    ]
    resumed.close()
    with pytest.raises(ValueError, match="settings changed"):
        RepositoryPreparationLedger(path, {"source": "fixture", "budget": 10000})


@pytest.mark.parametrize("reset", [False, True])
def test_archive_download_retries_incomplete_transport(tmp_path, monkeypatch, reset):
    import http.client
    from types import SimpleNamespace

    from bifocalgemma.data.repository_corpus import ArchiveSnapshotStore

    valid = tmp_path / "valid.tar.gz"
    archive(valid, {"LICENSE": MIT, "a.py": "x=0\n"})
    payload = valid.read_bytes()
    attempts = []

    class Response(io.BytesIO):
        def read(self, size=-1):
            if len(attempts) == 1:
                raise (
                    ConnectionResetError("reset")
                    if reset
                    else http.client.IncompleteRead(b"partial")
                )
            return super().read(size)

    def fetch(*args, **kwargs):
        attempts.append(1)
        return Response(payload)

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    task = SimpleNamespace(repo="owner/repo", base_commit="a" * 40)
    snapshot, _ = ArchiveSnapshotStore(tmp_path / "cache", allow_network=True).snapshot(task)
    assert len(attempts) == 2 and snapshot.base_commit == "a" * 40


def test_context_files_are_licensed_separately_from_patch(tmp_path):
    from bifocalgemma.data.repository_corpus import context_license_evidence

    path = tmp_path / "snapshot.tar.gz"
    archive(
        path,
        {
            "LICENSE": MIT,
            "pkg/a.py": "x=0\n",
            "vendor/LICENSE": "Proprietary.",
            "vendor/x.py": "x=0\n",
        },
    )
    snapshot, _ = read_text_archive(path, "owner/repo", "a" * 40)
    patch = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-x=0\n+x=1\n"
    assert patch_license_evidence(snapshot, patch)
    selection = {"gold_used": False, "windows": [{"path": "pkg/a.py"}, {"path": "vendor/x.py"}]}
    with pytest.raises(ValueError, match="license_not_admitted"):
        context_license_evidence(snapshot, selection)
    valid = context_license_evidence(
        snapshot, {"gold_used": False, "windows": [{"path": "pkg/a.py"}, {"path": "pkg/a.py"}]}
    )
    assert len(valid) == 1 and valid[0]["path"] == "pkg/a.py"
    with pytest.raises(ValueError, match="target_free"):
        context_license_evidence(snapshot, {**selection, "gold_used": True})
    with pytest.raises(ValueError, match="missing_selected"):
        context_license_evidence(
            snapshot, {"gold_used": False, "windows": [{"path": "missing.py"}]}
        )


def test_context_license_record_covers_all_visible_paths():
    from bifocalgemma.data.offline_sft import validate_context_license_record

    row = {
        "context_selection": {"windows": [{"path": "a.py"}]},
        "context_source_code_license_evidence": [
            {
                "path": "a.py",
                "licenses": [{"path": "LICENSE", "license": "mit", "sha256": "a" * 64}],
            }
        ],
    }
    validate_context_license_record(row)
    with pytest.raises(ValueError, match="coverage"):
        validate_context_license_record({**row, "context_source_code_license_evidence": []})
    row["context_source_code_license_evidence"][0]["licenses"][0]["license"] = "unknown"
    with pytest.raises(ValueError, match="not_admitted"):
        validate_context_license_record(row)
