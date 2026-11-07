"""The exported checkpoint diagnostic stays read-only and executes the audit worker."""

import ast
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from tools.notebook_bundle import source_bundle_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_exported_worker_has_parity_context_and_exact_adapter_checks(monkeypatch):
    manifest = json.loads((ROOT / "notebooks/runtime-manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source_bundle_bytes(ROOT)).hexdigest() != manifest["sha256"]:
        pytest.skip("Runtime bundle is stale after the V2/V3 split; republish before exporting.")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    exporter = importlib.import_module("prepare_colab_audit")
    template = importlib.import_module("prepare_colab_probe")
    parent = template.render(worker=exporter.audit)
    tree = ast.parse(parent)
    # Extract the fresh-subprocess script, rather than merely parsing its wrapper.
    worker = next(
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_text"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)
        and "def audit(" in n.args[0].value
    )
    ast.parse(worker)
    for required in (
        "cache_boundary_parity",
        "adapters_loaded_exactly",
        "swapped",
        "empty",
        "audit(root)",
    ):
        assert required in worker
    for forbidden in ("fit_offline_sft(", "create_commit(", "upload_file(", "optimizer.step("):
        assert forbidden not in worker
    assert "secret.unlink()" in worker
    assert "bifocalgemma-test.zip" in parent
