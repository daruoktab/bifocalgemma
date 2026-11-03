"""Verified runtime setup, kept outside the notebook's visible source."""

import importlib
import runpy
import sys
from pathlib import Path


def prepare_runtime(root):
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
    for name, module in list(sys.modules.items()):
        if name != "bifocalgemma" and not name.startswith("bifocalgemma."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename or Path(filename).suffix != ".py":
            continue
        parts = Path(filename).parts
        index = next(i for i, part in enumerate(parts) if part == "bifocalgemma")
        expected = root / "src" / Path(*parts[index:])
        if not expected.is_file() or expected.read_bytes().replace(b"\r\n", b"\n") != Path(
            filename
        ).read_bytes().replace(b"\r\n", b"\n"):
            raise RuntimeError("Restart kernel to load this notebook's pinned runtime.")
    sys.path.insert(0, str(root / "src"))
    importlib.invalidate_caches()
    return result
