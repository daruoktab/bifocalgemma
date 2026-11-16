"""Bootstrap installs and restarts are simulated; never touch the test environment."""

import ast
import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.notebook_dependencies import (
    activate_notebook_bundle,
    bootstrap_notebook_dependencies,
    import_native_gemma4,
)

PINS = {"torch": "2.14.1", "torchvision": "0.29.1", "numpy": "2.5.3", "marimo": "0.25.1"}


def simulated_environment(monkeypatch):
    versions = {}
    calls = []
    monkeypatch.setattr(sys, "version_info", (3, 12, 6))
    monkeypatch.setattr(sys, "_bifocalgemma_dependency_session", "first-kernel", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "transformers",
        "pyarrow",
        "tokenizers",
        "marimo",
        "wigglystuff",
        "ipywidgets",
        "anywidget",
        "traitlets",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "uv-test")

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    def run(command, **kwargs):
        calls.append(command)
        torch_pin = next((arg for arg in command if arg.startswith("torch==")), None)
        if torch_pin:
            versions["torch"] = torch_pin.removeprefix("torch==") + "+cu132"
            vision_pin = next((arg for arg in command if arg.startswith("torchvision==")), None)
            if vision_pin:
                versions["torchvision"] = vision_pin.removeprefix("torchvision==") + "+cu132"
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
    assert calls[0][-1] == "https://download.pytorch.org/whl/cu132"
    assert "torch==2.14.1" in calls[0] and "torchvision==0.29.1" in calls[0]
    assert not any(arg.startswith("torchvision==") for arg in calls[1])
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


def test_new_torch_pin_forces_reinstall_despite_old_receipt_or_matching_install(
    monkeypatch, tmp_path
):
    versions, calls = simulated_environment(monkeypatch)
    old_pins = {name: value for name, value in PINS.items() if name != "torchvision"}
    old_pins["torch"] = "2.11.0"
    bootstrap_notebook_dependencies(old_pins, state_directory=str(tmp_path))
    assert versions["torch"] == "2.11.0+cu132" and len(calls) == 2
    # The cloud image may already have the correct version, but no matching
    # reinstall receipt: FORCE_REINSTALL must still execute the installer.
    versions["torch"] = "2.14.1+cu130"
    versions["torchvision"] = PINS["torchvision"] + "+cu132"
    result = bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert result["installed_now"] and len(calls) == 4
    assert "torch==2.14.1" in calls[2] and "--reinstall" in calls[2]
    assert len(list(tmp_path.glob("*.json"))) == 2
    again = bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert not again["installed_now"] and len(calls) == 4


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
    versions["torch"] = "2.14.1+cpu"
    assert bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))["installed_now"]
    assert len(calls) == 4


def test_native_import_diagnostic_preserves_original_dependency_failure(monkeypatch, capsys):
    def broken(name):
        raise RuntimeError("operator torchvision::nms does not exist")

    monkeypatch.setattr(importlib, "import_module", broken)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    with pytest.raises(RuntimeError, match="Original traceback") as error:
        import_native_gemma4()
    assert "operator torchvision::nms does not exist" in str(error.value)
    assert '"torchvision": "test-version"' in str(error.value)
    assert "operator torchvision::nms does not exist" in capsys.readouterr().out


def test_native_import_loads_direct_modules_and_expected_symbols(monkeypatch):
    probes = []
    torch = SimpleNamespace(empty=lambda *a, **kw: object(), float32=object())
    vision = SimpleNamespace(
        ops=SimpleNamespace(nms=lambda *args: probes.append(args)),
        __version__=PINS["torchvision"] + "+cu132",
        __file__="/tmp/uv-venv/torchvision/__init__.py",
    )
    configuration = SimpleNamespace(Gemma4TextConfig=object())
    modeling = SimpleNamespace(
        **{
            name: object()
            for name in (
                "Gemma4TextModel",
                "Gemma4TextAttention",
                "Gemma4TextDecoderLayer",
                "apply_rotary_pos_emb",
                "repeat_kv",
            )
        }
    )
    modules = {
        "torch": torch,
        "torchvision": vision,
        "transformers.models.gemma4.configuration_gemma4": configuration,
        "transformers.models.gemma4.modeling_gemma4": modeling,
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    result = import_native_gemma4()
    assert result["Gemma4TextConfig"] is configuration.Gemma4TextConfig
    assert result["Gemma4TextModel"] is modeling.Gemma4TextModel
    assert result["torchvision_version"] == PINS["torchvision"] + "+cu132" and len(probes) == 1


@pytest.mark.parametrize("bad_vision", ["0.26.0", "0.29.1+cpu"])
def test_inherited_or_wrong_cuda_torchvision_invalidates_receipt(monkeypatch, tmp_path, bad_vision):
    versions, calls = simulated_environment(monkeypatch)
    bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    versions["torchvision"] = bad_vision
    result = bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))
    assert result["installed_now"] and len(calls) == 4
    assert versions["torchvision"] == PINS["torchvision"] + "+cu132"


def test_preloaded_torchvision_alone_requires_restart(monkeypatch, tmp_path):
    simulated_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(__version__="0.26.0"))
    with pytest.raises(RuntimeError, match="Restart the kernel"):
        bootstrap_notebook_dependencies(PINS, state_directory=str(tmp_path))


def test_source_bundle_checksum_failure_does_not_activate_source(tmp_path):
    import base64

    with pytest.raises(ValueError, match="checksum"):
        activate_notebook_bundle(base64.b85encode(b"invalid zip"), "0" * 64, directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_active_matrix_notebooks_pin_dependencies_and_gate_bifocalgemma_imports():
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted(p for p in (root / "notebooks").glob("train_*.py") if p.name != "train_bifocalgemma_fast.py")
    assert len(notebooks) == 10
    for path in notebooks:
        text = path.read_text(encoding="utf-8")
        assert '# requires-python = ">=3.12,<3.14"' in text
        assert '"torch==2.15.0"' in text
        assert '"transformers==5.22.1"' in text
        assert 'launcher["prepare_runtime"](runtime_root)' in text
        assert 'assert dependency_runtime["status"] == "READY"' in text
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("bifocalgemma"), path.name
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("bifocalgemma") for alias in node.names), path.name
