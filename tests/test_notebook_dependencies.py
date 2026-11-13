"""Bootstrap installs and restarts are simulated; never touch the test environment."""

import ast
import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def test_marimo_setup_token_order_precedes_native_imports():
    root = Path(__file__).resolve().parents[1]
    for family in ("symmetric", "asymmetric", "baseline"):
        text = (root / f"notebooks/legacy/bifocalgemma_train_{family}.py").read_text(
            encoding="utf-8"
        )
        cells = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)]
        assert cells[0].args.args == []
        first = ast.get_source_segment(text, cells[1])
        assert first is not None and "FORCE_REINSTALL = True" in first
        assert '"torch": "2.14.1"' in first
        assert '"torchvision": "0.29.1"' in first
        assert "bootstrap_notebook_dependencies(" in first
        assert "dependency_runtime" in [arg.arg for arg in cells[2].args.args]
        assert "dependency_runtime" in [arg.arg for arg in cells[3].args.args]
        assert "HfApi" not in [arg.arg for arg in cells[3].args.args]
        native = next(
            node
            for node in cells
            if any(
                isinstance(item, ast.Import) and any(a.name == "torch" for a in item.names)
                for item in node.body
            )
        )
        assert "dependency_runtime" in [arg.arg for arg in native.args.args]
        assert cells.index(native) == 5
        assert "auth_info" in [arg.arg for arg in native.args.args]
        assert 'torch.__version__ == "2.14.1+cu130"' in text
        assert '# requires-python = ">=3.12,<3.14"' in text
        assert "import_native_gemma4" in [arg.arg for arg in native.args.args]
        assert "_native_gemma4 = import_native_gemma4()" in text


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


def test_notebook_never_declares_bundled_project_as_external_package():
    root = Path(__file__).resolve().parents[1]
    for family in ("symmetric", "asymmetric", "baseline"):
        path = root / f"notebooks/legacy/bifocalgemma_train_{family}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("bifocalgemma")
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("bifocalgemma") for alias in node.names)
        first = [node for node in tree.body if isinstance(node, ast.FunctionDef)][1]
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "activate_notebook_bundle"
            for node in ast.walk(first)
        )


def test_one_uploaded_notebook_loads_source_without_repo_or_pythonpath(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks/legacy/bifocalgemma_train_symmetric.py"
    uploaded = tmp_path / "notebook.py"
    shutil.copyfile(source, uploaded)
    child = r"""
import ast
import importlib
import pathlib
import sys

uploaded = pathlib.Path(sys.argv[1])
tree = ast.parse(uploaded.read_text(encoding="utf-8"))
cell = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and any(isinstance(child, ast.FunctionDef) and child.name == "activate_notebook_bundle" for child in n.body))
helper = next(n for n in cell.body if isinstance(n, ast.FunctionDef) and n.name == "activate_notebook_bundle")
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), "uploaded-helper", "exec"), namespace)
values = {n.targets[0].id: ast.literal_eval(n.value) for n in cell.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"_payload", "_bundle_sha"}}
bundle = namespace["activate_notebook_bundle"](values["_payload"], values["_bundle_sha"])
project = importlib.import_module("bifocalgemma")
assert pathlib.Path(project.__file__).resolve().is_relative_to(bundle)
config = importlib.import_module("bifocalgemma.config")
assert config.get_experiment("BifocalGemma-E2B-E2B").name == "BifocalGemma-E2B-E2B"
assert not (uploaded.parent / "src").exists()
print("ONE_FILE_BUNDLE_OK")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", child, str(uploaded)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "ONE_FILE_BUNDLE_OK" in result.stdout


def test_source_bundle_checksum_failure_does_not_activate_source(tmp_path):
    import base64

    with pytest.raises(ValueError, match="checksum"):
        activate_notebook_bundle(base64.b85encode(b"invalid zip"), "0" * 64, directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_no_downstream_cell_can_run_before_authentication():
    root = Path(__file__).resolve().parents[1]
    for family in ("symmetric", "asymmetric", "baseline"):
        path = root / f"notebooks/legacy/bifocalgemma_train_{family}.py"
        cells = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
        ]
        for cell in cells[4:]:
            assert "auth_info" in [arg.arg for arg in cell.args.args]
            assert isinstance(cell.body[0], ast.Assert)
            gate = ast.Module(body=[cell.body[0]], type_ignores=[])
            with pytest.raises(AssertionError, match="Apply a valid HF token"):
                exec(compile(gate, "auth-gate", "exec"), {"auth_info": {"verified": False}})


@pytest.mark.parametrize("form_value", [None, {"token": ""}, {"token": "hf_fixture_write"}])
def test_apply_waits_for_submission_and_valid_token_without_native_imports(monkeypatch, form_value):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = []

    class FakeApi:
        def __init__(self, *, token):
            calls.append(token)

        def whoami(self):
            return {"name": "fixture", "auth": {"accessToken": {"role": "write"}}}

    class WaitingForToken(Exception):
        pass

    def stop(condition, message):
        if condition:
            raise WaitingForToken(message)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    path = Path(__file__).resolve().parents[1] / "notebooks/legacy/bifocalgemma_train_symmetric.py"
    cells = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    ]
    body = ast.Module(
        body=[node for node in cells[3].body if not isinstance(node, ast.Return)], type_ignores=[]
    )
    namespace: dict[str, Any] = {
        "dependency_runtime": {"status": "READY"},
        "hf_token_form": SimpleNamespace(value=form_value),
        "mo": SimpleNamespace(stop=stop, md=lambda value: value),
    }
    if not form_value or not form_value["token"]:
        with pytest.raises(WaitingForToken):
            exec(compile(body, "token-apply", "exec"), namespace)
        assert calls == [] and "auth_info" not in namespace
    else:
        exec(compile(body, "token-apply", "exec"), namespace)
        assert calls == ["hf_fixture_write"]
        assert namespace["auth_info"]["verified"] is True
        assert namespace["hf_download_token"] == "hf_fixture_write"
