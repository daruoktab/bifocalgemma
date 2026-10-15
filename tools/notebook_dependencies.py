"""Standard-library-only dependency bootstrap embedded in standalone notebooks."""


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
        "torch_index": "https://download.pytorch.org/whl/cu130",
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
        if versions.get("torch") != pins["torch"] + "+cu130":
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
        name in sys.modules for name in ("torch", "numpy", "transformers", "pyarrow", "tokenizers")
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
    print(f"Reinstall={reinstall}; Torch={pins['torch']} from CUDA 13.0 index.", flush=True)
    subprocess.run(
        common + [f"torch=={pins['torch']}", "--index-url", identity["torch_index"]],
        check=True,
    )
    subprocess.run(
        common
        + [f"{name}=={expected}" for name, expected in pins.items() if name != "torch"]
        + identity["extras"]
        + ["--index-url", "https://pypi.org/simple"],
        check=True,
    )
    versions = installed_versions()
    if not matches(versions):
        raise RuntimeError("Installed dependency versions differ from the notebook pins.")
    if versions["torch"] != pins["torch"] + "+cu130":
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
