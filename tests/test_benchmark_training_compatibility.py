"""A clean versionless baseline rejects legacy approvals and any unapproved model/training change."""

import json
import shutil
from pathlib import Path

from bifocalgemma.training.source_identity import source_fingerprints

LEGACY_BASELINES = {
    "fb253cb387c45c224c62232b3943bd87dd5682ccebfe358c8b2200aa2af8e5a7",
    "79b2a4e2de7774c22d7bd331f3b0e549c2bb24c7050b26789a3604fa770da8ff",
    "ebe51cc8e4050d46a1118846d4c7de6224b5cf8aeefd22dde8d0e173e85b021c",
}


def test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(tmp_path):
    project = Path(__file__).resolve().parents[1]
    approval_path = project / "configs/training_resume_compatibility.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    root = tmp_path / "project/src/bifocalgemma"
    shutil.copytree(
        project / "src/bifocalgemma", root, ignore=shutil.ignore_patterns("__pycache__")
    )
    config = root.parents[1] / "configs/training_resume_compatibility.json"
    config.parent.mkdir()
    config.write_text(json.dumps(approval), encoding="utf-8")
    actual = source_fingerprints(root)
    assert actual["source_semantics_sha256"] == approval["approved_source_semantics_sha256"]
    assert actual["training_semantics_sha256"] == approval["baseline_training_semantics_sha256"]
    assert actual["training_semantics_sha256"] == actual["source_semantics_sha256"]
    assert actual["training_semantics_sha256"] not in LEGACY_BASELINES
    for relative in ("modeling_bifocalgemma.py", "training/offline_sft.py", "training/matrix.py"):
        path = root / relative
        before = path.read_text(encoding="utf-8")
        path.write_text(before + "\nUNAPPROVED_CHANGE = 1\n", encoding="utf-8")
        mutated = source_fingerprints(root)
        assert mutated["training_semantics_sha256"] != actual["training_semantics_sha256"]
        assert mutated["source_semantics_sha256"] != actual["source_semantics_sha256"]
        path.write_text(before, encoding="utf-8")
