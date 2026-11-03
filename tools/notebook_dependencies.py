"""Standard-library-only dependency bootstrap embedded in standalone notebooks."""


def activate_notebook_bundle(payload, expected_sha, directory="outputs/corpus_runtime"):
    """Activate source embedded in this one notebook, before native imports."""
    import base64
    import hashlib
    import importlib
    import io
    import sys
    import zipfile
    from pathlib import Path

    data = base64.b85decode(payload)
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Notebook source bundle checksum mismatch.")
    root = (Path(directory) / expected_sha[:16]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            destination = (root / entry.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Notebook source bundle path escape.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(entry))
    if "bifocalgemma" in sys.modules:
        loaded = sys.modules["bifocalgemma"].__file__
        if loaded is None:
            raise RuntimeError("Restart kernel: unexpected BifocalGemma module.")
        loaded_root = Path(loaded).resolve().parent
        for name, module in list(sys.modules.items()):
            if name != "bifocalgemma" and not name.startswith("bifocalgemma."):
                continue
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).suffix == ".py":
                path = Path(filename).resolve()
                if not path.is_relative_to(loaded_root):
                    raise RuntimeError("Restart kernel: mixed BifocalGemma modules.")
                expected = root / "src/bifocalgemma" / path.relative_to(loaded_root)
                if not expected.is_file() or expected.read_bytes() != path.read_bytes():
                    raise RuntimeError("Restart kernel to load the new notebook source bundle.")
    sys.path.insert(0, str(root / "src"))
    importlib.invalidate_caches()
    module = importlib.import_module("bifocalgemma")
    if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("BifocalGemma must load from this notebook's embedded source.")
    print(f"Embedded project source activated: {root} (no repository checkout needed).", flush=True)
    return root


def import_native_gemma4():
    """Import native modules directly and expose the complete failure in marimo.

    Lazy top-level Transformers imports can hide a binary/optional-dependency
    failure behind 'Could not import module Gemma4TextConfig'. Do not suppress
    the original exception or mutate optional packages without its evidence.
    """
    import importlib
    import importlib.metadata
    import json
    import sys
    import traceback

    try:
        torch = importlib.import_module("torch")
        torchvision = importlib.import_module("torchvision")
        # Prove the compiled extension loads, without allocating GPU memory.
        torchvision.ops.nms(
            torch.empty((0, 4), dtype=torch.float32, device="cpu"),
            torch.empty((0,), dtype=torch.float32, device="cpu"),
            0.5,
        )
        configuration = importlib.import_module("transformers.models.gemma4.configuration_gemma4")
        modeling = importlib.import_module("transformers.models.gemma4.modeling_gemma4")
        return {
            "Gemma4TextConfig": configuration.Gemma4TextConfig,
            "Gemma4TextModel": modeling.Gemma4TextModel,
            "Gemma4TextAttention": modeling.Gemma4TextAttention,
            "Gemma4TextDecoderLayer": modeling.Gemma4TextDecoderLayer,
            "apply_rotary_pos_emb": modeling.apply_rotary_pos_emb,
            "repeat_kv": modeling.repeat_kv,
            "torchvision_version": torchvision.__version__,
            "torchvision_file": torchvision.__file__,
        }
    except Exception:
        original = traceback.format_exc()
        versions = {}
        for name in (
            "torch",
            "torchvision",
            "torchaudio",
            "transformers",
            "huggingface-hub",
            "tokenizers",
            "numpy",
            "scipy",
            "scikit-learn",
            "librosa",
            "numba",
            "Pillow",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        message = (
            "Native Gemma4 import failed before model loading.\n"
            f"Python: {sys.version.split()[0]} ({sys.executable})\n"
            f"Installed packages: {json.dumps(versions, sort_keys=True)}\n"
            "Original traceback (send this entire block):\n" + original
        )
        print(message, flush=True)
        raise RuntimeError(message) from None


def bootstrap_notebook_dependencies(
    pins: dict[str, str], *, reinstall=True, state_directory="outputs/dependency_setup"
):
    """Reinstall once per matching environment; never install into conda base.

    A receipt avoids repeated reinstalls after a kernel restart. Critical modules
    already imported before installation require a new kernel, not another import.
    No installation is performed merely by importing this module.
    """
    import hashlib
    import importlib.metadata
    import json
    import os
    import shutil
    import subprocess
    import sys
    import urllib.request
    from pathlib import Path
    from uuid import uuid4

    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Select a Python 3.12 or 3.13 kernel before installing dependencies.")
    prefix = Path(sys.prefix).resolve()
    conda_prefix = os.environ.get("CONDA_PREFIX")
    base_selected = (
        os.environ.get("CONDA_DEFAULT_ENV") == "base"
        and conda_prefix is not None
        and Path(conda_prefix).resolve() == prefix
    )
    conda_base_layout = (prefix / "conda-meta").is_dir() and any(
        (prefix / name).exists() for name in ("condabin/conda", "condabin/conda.bat")
    )
    if base_selected or conda_base_layout:
        raise RuntimeError("Do not install into conda base. Ask which project environment to use.")
    if "torch" not in pins or not all(isinstance(v, str) and v for v in pins.values()):
        raise ValueError("Dependency pins must include an explicit Torch version.")

    identity = {
        "format": "bifocalgemma_dependency_setup_v1",
        "python": str(Path(sys.executable).resolve()),
        "python_version": list(sys.version_info[:2]),
        "pins": dict(pins),
        "extras": ["PyYAML>=6.0", "psutil>=7.0"],
        "torch_index": "https://download.pytorch.org/whl/cu132",
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(state_directory)
    receipt_path = root / f"{signature}.json"
    receipt = None
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            receipt = None

    session = getattr(sys, "_bifocalgemma_dependency_session", None)
    if session is None:
        session = uuid4().hex
        setattr(sys, "_bifocalgemma_dependency_session", session)

    def installed_versions():
        versions: dict[str, str | None] = {}
        for name in list(pins) + ["PyYAML", "psutil"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return versions

    def matches(versions: dict[str, str | None]):
        for name, expected in pins.items():
            actual = versions.get(name)
            if actual is None or actual.split("+")[0] != expected:
                return False
        for name in ("torch", "torchvision"):
            if name in pins and versions.get(name) != pins[name] + "+cu132":
                return False
        for name, minimum in (("PyYAML", 6), ("psutil", 7)):
            actual = versions.get(name)
            if actual is None or int(actual.split(".")[0]) < minimum:
                return False
        return True

    versions = installed_versions()
    valid_receipt = isinstance(receipt, dict) and receipt.get("identity") == identity
    if valid_receipt and receipt is not None and matches(versions):
        if receipt.get("restart_required") and receipt.get("session") == session:
            raise RuntimeError(
                "Dependencies were reinstalled in this kernel. Restart the kernel now."
            )
        return {"status": "READY", "installed_now": False, "versions": versions, **identity}

    if not reinstall and matches(versions):
        return {"status": "READY", "installed_now": False, "versions": versions, **identity}

    critical_loaded = any(
        name in sys.modules
        for name in (
            "torch",
            "torchvision",
            "torchaudio",
            "numpy",
            "transformers",
            "pyarrow",
            "tokenizers",
        )
    )
    uv = shutil.which("uv")
    if uv is None:
        candidate = Path.home() / ".local/bin" / ("uv.exe" if os.name == "nt" else "uv")
        if candidate.is_file():
            uv = str(candidate)
        elif os.name == "posix":
            print("Installing uv using the official standalone installer.", flush=True)
            with urllib.request.urlopen("https://astral.sh/uv/install.sh", timeout=60) as response:
                installer = response.read()
            subprocess.run(
                ["sh"],
                input=installer,
                check=True,
                env={**os.environ, "UV_NO_MODIFY_PATH": "1"},
            )
            if not candidate.is_file():
                raise RuntimeError("uv installer completed but its executable was not found.")
            uv = str(candidate)
        else:
            raise RuntimeError("Install uv in the chosen environment before running this cell.")

    common = [uv, "pip", "install", "--python", sys.executable]
    if reinstall:
        common.append("--reinstall")
    print(f"Dependency setup targets the current kernel: {sys.executable}", flush=True)
    cuda_packages = [name for name in ("torch", "torchvision") if name in pins]
    print(
        f"Reinstall={reinstall}; CUDA 13.0 packages: {', '.join(name + '==' + pins[name] for name in cuda_packages)}.",
        flush=True,
    )
    subprocess.run(
        common
        + [f"{name}=={pins[name]}" for name in cuda_packages]
        + ["--index-url", identity["torch_index"]],
        check=True,
    )
    subprocess.run(
        common
        + [f"{name}=={expected}" for name, expected in pins.items() if name not in cuda_packages]
        + identity["extras"]
        + ["--index-url", "https://pypi.org/simple"],
        check=True,
    )
    versions = installed_versions()
    if not matches(versions):
        raise RuntimeError("Installed dependency versions differ from the notebook pins.")
    if versions["torch"] != pins["torch"] + "+cu132":
        raise RuntimeError("Expected the CUDA 13.0 Torch wheel; verify the installed build.")
    root.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "identity": identity,
                "versions": versions,
                "session": session,
                "restart_required": critical_loaded,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    if critical_loaded:
        raise RuntimeError(
            "Dependencies installed. Restart the kernel before importing/training; "
            "the matching receipt will skip reinstall after restart."
        )
    print("Dependency setup verified. Continue with the next cell.", flush=True)
    return {"status": "READY", "installed_now": True, "versions": versions, **identity}
