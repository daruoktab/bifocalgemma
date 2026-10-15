"""Bootstrap installs and restarts are simulated; never touch the test environment."""

import ast
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.notebook_dependencies import bootstrap_notebook_dependencies

PINS = {"torch": "2.11.0", "numpy": "2.5.3", "marimo": "0.25.1"}


def simulated_environment(monkeypatch):
    versions = {}
    calls = []
    monkeypatch.setattr(sys, "version_info", (3, 12, 6))
    monkeypatch.setattr(sys, "_bifocalgemma_dependency_session", "first-kernel", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    for name in ("torch", "numpy", "transformers", "pyarrow", "tokenizers"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "uv-test")

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    def run(command, **kwargs):
        calls.append(command)
        if "torch==2.11.0" in command:
            versions["torch"] = "2.11.0+cu130"
        else:
            versions.update(numpy="2.5.3", marimo="0.25.1", PyYAML="6.0.3", psutil="7.0.0")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(subprocess, "run", run)
    return versions, calls


def test_force_reinstall_targets_kernel_and_receipt_skips_next_install(monkeypatch, tmp_path):
    _, calls = simulated_environment(monkeypatch)
    result = bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert result["installed_now"] and len(calls) == 2
    assert all("--reinstall" in command for command in calls)
    assert all(command[command.index("--python") + 1] == sys.executable for command in calls)
    assert calls[0][-1] == "https://download.pytorch.org/whl/cu130"
    again = bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert not again["installed_now"] and len(calls) == 2


def test_preloaded_torch_requires_real_kernel_restart(monkeypatch, tmp_path):
    _, calls = simulated_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.10.0"))
    with pytest.raises(RuntimeError, match="Restart the kernel"):
        bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert len(calls) == 2 and len(list(tmp_path.glob("*.json"))) == 1
    with pytest.raises(RuntimeError, match="Restart the kernel"):
        bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert len(calls) == 2
    monkeypatch.setattr(sys, "_bifocalgemma_dependency_session", "second-kernel")
    monkeypatch.delitem(sys.modules, "torch")
    assert bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))["status"] == "READY"
    assert len(calls) == 2


def test_conda_base_is_rejected_before_any_install(monkeypatch, tmp_path):
    _, calls = simulated_environment(monkeypatch)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    with pytest.raises(RuntimeError, match="conda base"):
        bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert calls == [] and not list(tmp_path.glob("*.json"))


def test_failed_install_cannot_write_success_receipt(monkeypatch, tmp_path):
    simulated_environment(monkeypatch)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert not list(tmp_path.glob("*.json"))


def test_cpu_wheel_invalidates_matching_receipt(monkeypatch, tmp_path):
    versions, calls = simulated_environment(monkeypatch)
    bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    versions["torch"] = "2.11.0+cpu"
    assert bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))["installed_now"]
    assert len(calls) == 4


def test_reinstall_cell_is_first_and_native_imports_depend_on_it():
    root = Path(__file__).resolve().parents[1]
    for family in ("symmetric", "asymmetric", "baseline"):
        text = (root / f"notebooks/bifocalgemma_train_{family}.py").read_text(encoding="utf-8")
        cells = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)]
        first = ast.get_source_segment(text, cells[0])
        assert first is not None and "FORCE_REINSTALL = True" in first
        assert '"torch": "2.11.0"' in first
        assert "bootstrap_notebook_dependencies(" in first
        assert cells[1].args.args[0].arg == "dependency_runtime"
        native = next(
            node
            for node in cells
            if any(
                isinstance(item, ast.Import) and any(a.name == "torch" for a in item.names)
                for item in node.body
            )
        )
        assert "dependency_runtime" in [arg.arg for arg in native.args.args]
        assert 'torch.__version__ == "2.11.0+cu130"' in text
        assert '# requires-python = ">=3.12,<3.14"' in text
