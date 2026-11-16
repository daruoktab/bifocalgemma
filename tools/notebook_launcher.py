"""Verified runtime setup, kept outside the notebook's visible source."""

import importlib
import runpy
import sys
from pathlib import Path

_RESTART_ERROR = "Restart kernel to load this notebook's pinned runtime."


def _normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _expected_bifocalgemma_path(root: Path, loaded_path: Path) -> Path | None:
    parts = loaded_path.parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "src" and parts[i + 1] == "bifocalgemma":
            return root / "src" / Path(*parts[i + 1 :])
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "bifocalgemma":
            return root / "src" / Path(*parts[i:])
    return None


def _validate_loaded_modules(root: Path) -> None:
    resolved_root = root.resolve()
    for name, module in list(sys.modules.items()):
        is_pkg = name == "bifocalgemma" or name.startswith("bifocalgemma.")
        is_tools = name == "tools" or name.startswith("tools.")
        if not is_pkg and not is_tools:
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            pkg_paths = getattr(module, "__path__", None)
            if is_pkg and pkg_paths is not None:
                expected_pkg_dir = resolved_root / "src" / Path(*name.split("."))
                loaded_paths = [Path(p).resolve() for p in pkg_paths]
                if not loaded_paths or any(p != expected_pkg_dir for p in loaded_paths):
                    raise RuntimeError(_RESTART_ERROR)
            continue
        if Path(filename).suffix != ".py":
            continue
        loaded_path = Path(filename).resolve()
        if not loaded_path.is_file():
            raise RuntimeError(_RESTART_ERROR)

        if is_pkg:
            expected = _expected_bifocalgemma_path(resolved_root, loaded_path)
            if (
                expected is None
                or not expected.is_file()
                or _normalized_bytes(expected) != _normalized_bytes(loaded_path)
            ):
                raise RuntimeError(_RESTART_ERROR)
            continue

        parts = loaded_path.parts
        tools_idx = next(
            (i for i in range(len(parts) - 1, -1, -1) if parts[i] == "tools"),
            None,
        )
        if tools_idx is None:
            if "site-packages" not in parts and "dist-packages" not in parts:
                raise RuntimeError(_RESTART_ERROR)
            continue

        loaded_tools_root = Path(*parts[:tools_idx]) if tools_idx > 0 else Path(".")
        if loaded_tools_root == resolved_root:
            continue
        rel = Path(*parts[tools_idx + 1 :])
        expected = resolved_root / "tools" / rel
        if expected.is_file():
            if _normalized_bytes(expected) != _normalized_bytes(loaded_path):
                raise RuntimeError(_RESTART_ERROR)
        elif (
            "site-packages" not in parts
            and "dist-packages" not in parts
            and not (loaded_tools_root / "pyproject.toml").is_file()
        ):
            raise RuntimeError(_RESTART_ERROR)


def prepare_runtime(root):
    root = Path(root).resolve()
    helpers = runpy.run_path(str(root / "tools/notebook_dependencies.py"))
    pins = {
        "torch": "2.15.0",
        "torchvision": "0.29.2",
        "transformers": "5.22.1",
        "marimo": "0.25.2",
        "huggingface-hub": "2.4.0",
        "safetensors": "0.8.0",
        "tokenizers": "0.23.3",
        "pyarrow": "26.0.0",
        "numpy": "2.5.4",
        "pytest": "9.2.0",
        "psutil": "7.2.2",
        "wigglystuff": "0.5.36",
        **({"triton-windows": "3.9.0.post30"} if sys.platform == "win32" else {"triton": "3.9.0"}),
    }
    result = helpers["bootstrap_notebook_dependencies"](pins, reinstall=False)
    _validate_loaded_modules(root)
    for path_entry in (str(root / "src"), str(root)):
        if path_entry in sys.path:
            sys.path.remove(path_entry)
        sys.path.insert(0, path_entry)
    importlib.invalidate_caches()
    return result
