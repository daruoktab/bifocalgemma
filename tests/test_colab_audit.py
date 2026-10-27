"""The exported checkpoint diagnostic stays read-only and executes the audit worker."""

import ast
import importlib
from pathlib import Path


def test_exported_worker_has_parity_context_and_exact_adapter_checks(monkeypatch):
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
