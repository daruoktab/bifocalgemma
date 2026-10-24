"""Verified runtime setup, kept outside the notebook's visible source."""

import importlib
import runpy
import sys
from pathlib import Path


def prepare_runtime(root):
    helpers = runpy.run_path(str(root / "tools/notebook_dependencies.py"))
    pins = {
        "torch": "2.14.1",
        "torchvision": "0.29.1",
        "transformers": "5.20.2",
        "marimo": "0.25.2",
        "huggingface-hub": "2.2.0",
        "safetensors": "0.8.0",
        "tokenizers": "0.23.3",
        "pyarrow": "26.0.0",
        "numpy": "2.5.4",
        "pytest": "8.4.2",
        "psutil": "7.2.2",
        "wigglystuff": "0.5.36",
        **({"triton-windows": "3.8.0.post29"} if sys.platform == "win32" else {"triton": "3.8.0"}),
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
        if not expected.is_file() or expected.read_bytes() != Path(filename).read_bytes():
            raise RuntimeError("Restart kernel to load this notebook's pinned runtime.")
    sys.path.insert(0, str(root / "src"))
    importlib.invalidate_caches()
    return result
