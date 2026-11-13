"""Never accept a stale live widget stack as a ready notebook environment."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.notebook_dependencies import bootstrap_notebook_dependencies

UI_MODULES = ("marimo", "wigglystuff", "ipywidgets", "anywidget", "traitlets")
BASE_PINS = {"torch": "2.15.0", "torchvision": "0.29.2"}


@pytest.fixture
def dependency_state(monkeypatch, tmp_path):
    """Replace metadata and the installer; never mutate installed distributions."""
    versions = {
        "torch": "2.15.0+cu132",
        "torchvision": "0.29.2+cu132",
        "PyYAML": "6.0.3",
        "psutil": "7.2.2",
    }
    calls = []
    monkeypatch.setattr(sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "selected-environment"))
    monkeypatch.setattr(
        sys, "_bifocalgemma_dependency_session", "widget-test-kernel", raising=False
    )
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    for name in (
        *UI_MODULES,
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "transformers",
        "pyarrow",
        "tokenizers",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "mock-uv")

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    def run(command, **kwargs):
        calls.append(command)
        for argument in command:
            if "==" in argument:
                name, installed = argument.split("==", 1)
                if name in ("torch", "torchvision"):
                    installed += "+cu132"
                versions[name] = installed
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(subprocess, "run", run)
    return versions, calls


@pytest.mark.parametrize("module_name", UI_MODULES)
@pytest.mark.parametrize("changed_dependency", ["ui", "torch"])
def test_install_with_preloaded_pinned_ui_module_requires_restart(
    monkeypatch, tmp_path, dependency_state, module_name, changed_dependency
):
    versions, calls = dependency_state
    # Initially loaded code agrees with disk. The install changes only disk,
    # so testing the versions before installation cannot establish readiness.
    pins = {**BASE_PINS, module_name: "0.2.0"}
    loaded_version = "0.1.0" if changed_dependency == "ui" else pins[module_name]
    versions[module_name] = loaded_version
    if changed_dependency == "torch":
        versions["torch"] = "2.14.0+cu132"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__=loaded_version))
    with pytest.raises(RuntimeError, match="Restart"):
        bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert versions[module_name] == "0.2.0"
    assert len(calls) == 2
    receipts = list(tmp_path.glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["restart_required"] is True


@pytest.mark.parametrize("module_name", UI_MODULES)
@pytest.mark.parametrize("matching_receipt", [False, True])
def test_stale_loaded_ui_version_rejects_each_ready_fast_path(
    monkeypatch, tmp_path, dependency_state, module_name, matching_receipt
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, module_name: "0.2.0"}
    versions[module_name] = pins[module_name]
    if matching_receipt:
        # Seed a valid receipt with no UI module loaded; no real install runs.
        assert (
            bootstrap_notebook_dependencies(pins, state_directory=str(tmp_path))["status"]
            == "READY"
        )
    prior_calls = len(calls)
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.1.0"))
    with pytest.raises(RuntimeError, match="Restart"):
        bootstrap_notebook_dependencies(
            pins, reinstall=matching_receipt, state_directory=str(tmp_path)
        )
    assert len(calls) == prior_calls


@pytest.mark.parametrize("module_name", UI_MODULES)
@pytest.mark.parametrize("matching_receipt", [False, True])
def test_matching_loaded_and_installed_ui_version_remains_ready(
    monkeypatch, tmp_path, dependency_state, module_name, matching_receipt
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, module_name: "0.2.0"}
    versions[module_name] = pins[module_name]
    if matching_receipt:
        assert (
            bootstrap_notebook_dependencies(pins, state_directory=str(tmp_path))["status"]
            == "READY"
        )
    prior_calls = len(calls)
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.2.0"))
    result = bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert result["status"] == "READY"
    assert result["installed_now"] is False
    assert len(calls) == prior_calls
