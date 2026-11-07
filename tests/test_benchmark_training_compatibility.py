"""A reviewed benchmark fix must resume old training without accepting model changes."""

import json
import shutil
from pathlib import Path

from bifocalgemma.training.source_identity import source_fingerprints


def test_exact_benchmark_release_preserves_training_identity_and_rejects_model_changes(tmp_path):
    project = Path(__file__).resolve().parents[1]
    approval_path = project / "configs/training_resume_compatibility.json"
    approval = json.loads(approval_path.read_text())
    root = tmp_path / "project/src/bifocalgemma"
    shutil.copytree(
        project / "src/bifocalgemma", root, ignore=shutil.ignore_patterns("__pycache__")
    )
    config = root.parents[1] / "configs/training_resume_compatibility.json"
    config.parent.mkdir()
    config.write_text(json.dumps(approval))
    actual = source_fingerprints(root)
    assert actual["source_semantics_sha256"] == approval["approved_source_semantics_sha256"]
    assert actual["training_semantics_sha256"] == approval["baseline_training_semantics_sha256"]
    assert actual["source_semantics_sha256"] != actual["training_semantics_sha256"]
    for relative in ("models/v2/native.py", "training/offline_sft.py", "training/v2.py"):
        path = root / relative
        before = path.read_text()
        path.write_text(before + "\nUNAPPROVED_CHANGE = 1\n")
        assert (
            source_fingerprints(root)["training_semantics_sha256"]
            != actual["training_semantics_sha256"]
        )
        path.write_text(before)
