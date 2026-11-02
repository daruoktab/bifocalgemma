import ast
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from tools.notebook_bundle import source_bundle_bytes


def test_six_notebooks_have_one_fixed_model_and_identical_verified_bundle():
    root = Path(__file__).resolve().parents[1]
    keys = {"e2b-e2b", "e4b-e2b", "e2b-e4b", "e4b-e4b", "e2b-it", "e4b-it"}
    digests = set()
    repos = set()
    notebook_root = Path(os.environ.get("BIFOCALGEMMA_NOTEBOOK_DIR", root / "notebooks/legacy/v2"))
    paths = [notebook_root / f"train_{k.replace('-', '_')}.py" for k in sorted(keys)]
    assert len(paths) == 6
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        literals = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    literals[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        key = literals["MODEL_KEY"]
        assert key in keys
        assert path.name == "train_" + key.replace("-", "_") + ".py"
        repos.add(literals["HF_REPO_ID"])
        manifest = literals["runtime_manifest"]
        assert manifest == json.loads((notebook_root / "runtime-manifest.json").read_text())
        local = notebook_root / "runtime" / manifest["filename"]
        packed = local.read_bytes() if local.exists() else source_bundle_bytes(root)
        assert hashlib.sha256(packed).hexdigest() == manifest["sha256"]
        assert len(manifest["revision"]) == 40 or (
            notebook_root != root / "notebooks" and manifest["revision"] == "LOCAL-PREVIEW"
        )
        assert "payload" not in literals and len(path.read_text().splitlines()) < 220
        digests.add(manifest["sha256"])
        with zipfile.ZipFile(io.BytesIO(packed)) as bundle:
            assert "src/bifocalgemma/training/v2.py" in bundle.namelist()
            assert bundle.read("src/bifocalgemma/training/v2.py") == (
                root / "src/bifocalgemma/training/v2.py"
            ).read_bytes().replace(b"\r\n", b"\n")
        calls = [
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        assert calls.count("run_training_and_benchmark") == 1
        assert calls.count("WigglyDashboard") == 1
        assert "run_v2" not in calls
        assert not any(
            isinstance(n, ast.Attribute) and n.attr == "dropdown" for n in ast.walk(tree)
        )
        assert "baseline_snapshot" not in path.read_text(encoding="utf-8")
        assert "run_recipe_comparison" not in calls and "run_sequential_pipeline" not in calls
    assert len(digests) == 1 and len(repos) == 6
