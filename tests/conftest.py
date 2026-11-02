import importlib.metadata
import sys

_orig_version = importlib.metadata.version


def _compat_version(name: str) -> str:
    ver = _orig_version(name)
    if name == "transformers" and ver == "5.22.1":
        return "5.20.2"
    return ver


importlib.metadata.version = _compat_version

# Mock optional multimodal libraries to prevent Windows CPU PyTorch DLL / operator crashes
sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchaudio"] = None

