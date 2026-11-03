import sys
from typing import Any, cast

# Mock optional multimodal libraries to prevent Windows CPU PyTorch DLL / operator crashes
cast(dict[str, Any], sys.modules)["torchvision"] = None
cast(dict[str, Any], sys.modules)["torchvision.transforms"] = None
cast(dict[str, Any], sys.modules)["torchaudio"] = None
