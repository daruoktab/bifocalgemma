"""Import-time environment guards that must run before `transformers` is imported.

Mirrors the guards previously executed at the top of
`tools/molab_matrix_support.py`: CUDA allocator configuration plus optional
multimodal imports that crash the Windows CPU PyTorch build.
"""

from __future__ import annotations

import os
import sys
from typing import Any, cast

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if "torchvision" not in sys.modules or sys.modules["torchvision"] is not None:
    try:
        import torchvision

        torchvision.ops.nms(
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            0.5,
        )
    except Exception:
        cast(dict[str, Any], sys.modules)["torchvision"] = None
        cast(dict[str, Any], sys.modules)["torchvision.transforms"] = None

if "torchaudio" not in sys.modules or sys.modules["torchaudio"] is not None:
    try:
        import torchaudio  # noqa: F401
    except Exception:
        cast(dict[str, Any], sys.modules)["torchaudio"] = None
