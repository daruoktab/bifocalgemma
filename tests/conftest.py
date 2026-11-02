import importlib.metadata
import sys
from typing import Any, cast

_orig_version = importlib.metadata.version


def _compat_version(distribution_name: str) -> str:
    ver = _orig_version(distribution_name)
    if distribution_name == "transformers" and ver == "5.22.1":
        return "5.20.2"
    return ver


setattr(importlib.metadata, "version", _compat_version)

# Mock optional multimodal libraries to prevent Windows CPU PyTorch DLL / operator crashes
cast(dict[str, Any], sys.modules)["torchvision"] = None
cast(dict[str, Any], sys.modules)["torchvision.transforms"] = None
cast(dict[str, Any], sys.modules)["torchaudio"] = None

