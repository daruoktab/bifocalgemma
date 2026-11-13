"""Never accept a stale live widget stack as a ready notebook environment."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import marimo
import pytest

from tools.notebook_dependencies import bootstrap_notebook_dependencies
from tools.notebook_launcher import prepare_runtime

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
def test_install_with_preloaded_pinned_ui_module_requires_restart(
    monkeypatch, tmp_path, dependency_state, module_name
):
    versions, calls = dependency_state
    # Initially loaded code agrees with disk. The install changes only disk,
    # so testing the versions before installation cannot establish readiness.
    pins = {**BASE_PINS, module_name: "0.2.0"}
    loaded_version = "0.1.0"
    versions[module_name] = loaded_version
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__=loaded_version))
    with pytest.raises(RuntimeError, match="Restart"):
        bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert versions[module_name] == "0.2.0"
    assert len(calls) == 2
    receipts = list(tmp_path.glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["restart_required"] is True


@pytest.mark.parametrize("module_name", UI_MODULES)
@pytest.mark.parametrize("missing_dependency", ["torch", "psutil", "pytest"])
def test_unchanged_loaded_ui_does_not_require_restart_for_other_dependency_install(
    monkeypatch, tmp_path, dependency_state, module_name, missing_dependency
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, module_name: "0.2.0", "pytest": "9.2.0", "psutil": "7.2.2"}
    versions.update({module_name: "0.2.0", "pytest": "9.2.0"})
    versions.pop(missing_dependency, None)
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.2.0"))
    result = bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert result["status"] == "READY" and result["installed_now"] is True
    assert len(calls) == 2
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert receipt["restart_required"] is False
    # Run All again in the same kernel must neither reinstall nor demand restart.
    again = bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert again["status"] == "READY" and again["installed_now"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("marimo_metadata_available", [True, False])
def test_matrix_launcher_upload_and_repeated_setup_with_preloaded_marimo(
    monkeypatch, tmp_path, dependency_state, marimo_metadata_available
):
    versions, calls = dependency_state
    # Exercise the actual standalone launcher and its runpy-loaded helper with
    # full matrix pins, like an uploaded notebook whose marimo is already active.
    versions.update(marimo="0.25.2", wigglystuff="0.5.36")
    if not marimo_metadata_available:
        # Molab can inherit its live marimo module from the host while the
        # selected /tmp/uv-venv has no marimo distribution metadata yet.
        versions.pop("marimo")
    assert marimo.__version__ == "0.25.2"
    monkeypatch.setitem(sys.modules, "marimo", marimo)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    for name in list(sys.modules):
        if name == "bifocalgemma" or name.startswith("bifocalgemma."):
            monkeypatch.delitem(sys.modules, name)
    root = Path(__file__).resolve().parents[1]
    result = prepare_runtime(root)
    assert result["status"] == "READY" and result["installed_now"] is True
    again = prepare_runtime(root)
    assert again["status"] == "READY" and again["installed_now"] is False
    assert len(calls) == 2
    receipt = next((tmp_path / "outputs/dependency_setup").glob("*.json"))
    assert json.loads(receipt.read_text(encoding="utf-8"))["restart_required"] is False


@pytest.mark.parametrize("module_name", UI_MODULES)
def test_inherited_ui_without_local_metadata_can_install_the_same_loaded_version(
    monkeypatch, tmp_path, dependency_state, module_name
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, module_name: "0.2.0"}
    assert module_name not in versions
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.2.0"))
    result = bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert result["status"] == "READY" and result["installed_now"] is True
    assert versions[module_name] == "0.2.0"
    again = bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert again["status"] == "READY" and again["installed_now"] is False
    assert len(calls) == 2
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert receipt["restart_required"] is False
    assert receipt["loaded_ui"][module_name]["version"] == "0.2.0"


@pytest.mark.parametrize("module_name", UI_MODULES)
@pytest.mark.parametrize("loaded_version", ["0.1.0", None])
def test_inherited_ui_with_different_or_unknown_version_requires_diagnostic_restart(
    monkeypatch, tmp_path, dependency_state, module_name, loaded_version
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, module_name: "0.2.0"}
    origin = f"/host/site-packages/{module_name}/__init__.py"
    monkeypatch.setitem(
        sys.modules, module_name, SimpleNamespace(__version__=loaded_version, __file__=origin)
    )
    with pytest.raises(RuntimeError, match="Restart") as error:
        bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert f"{module_name}: loaded={loaded_version or 'unknown'}, installed=0.2.0" in str(
        error.value
    )
    assert origin in str(error.value) and len(calls) == 2
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert receipt["restart_required"] is True and receipt["restart_reasons"]
    assert receipt["loaded_ui"][module_name]["origin"] == origin


@pytest.mark.parametrize("module_name", UI_MODULES)
def test_changed_ui_install_becomes_ready_after_restart_without_reinstall(
    monkeypatch, tmp_path, dependency_state, module_name
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, module_name: "0.2.0"}
    versions[module_name] = "0.1.0"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.1.0"))
    with pytest.raises(RuntimeError, match="Restart"):
        bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert len(calls) == 2
    monkeypatch.setattr(sys, "_bifocalgemma_dependency_session", "restarted-kernel")
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.2.0"))
    result = bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert result["status"] == "READY" and result["installed_now"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("module_name", UI_MODULES)
def test_explicit_reinstall_of_loaded_ui_requires_restart_even_at_same_version(
    monkeypatch, tmp_path, dependency_state, module_name
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, module_name: "0.2.0"}
    versions[module_name] = "0.2.0"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.2.0"))
    with pytest.raises(RuntimeError, match="Restart"):
        bootstrap_notebook_dependencies(pins, reinstall=True, state_directory=str(tmp_path))
    assert len(calls) == 2
    assert all("--reinstall" in command for command in calls)


@pytest.mark.parametrize("module_name", UI_MODULES[1:])
def test_changed_transitive_loaded_ui_dependency_also_requires_restart(
    monkeypatch, tmp_path, dependency_state, module_name
):
    versions, calls = dependency_state
    pins = {**BASE_PINS, "marimo": "0.2.0", "pytest": "9.2.0"}
    versions.update({"marimo": "0.2.0", module_name: "0.1.0"})
    monkeypatch.setitem(sys.modules, "marimo", SimpleNamespace(__version__="0.2.0"))
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.1.0"))
    install = subprocess.run

    def install_with_transitive_upgrade(command, **kwargs):
        result = install(command, **kwargs)
        if "marimo==0.2.0" in command:
            versions[module_name] = "0.2.0"
        return result

    monkeypatch.setattr(subprocess, "run", install_with_transitive_upgrade)
    with pytest.raises(RuntimeError, match="Restart"):
        bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert len(calls) == 2
    monkeypatch.setattr(sys, "_bifocalgemma_dependency_session", "restarted-kernel")
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(__version__="0.2.0"))
    result = bootstrap_notebook_dependencies(pins, reinstall=False, state_directory=str(tmp_path))
    assert result["status"] == "READY" and result["installed_now"] is False
    assert len(calls) == 2


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
