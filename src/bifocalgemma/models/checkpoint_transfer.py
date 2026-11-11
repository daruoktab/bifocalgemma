"""Strict text-only safetensors transfer for the pretrained architecture probe.

Validates all keys/shapes before any copy. No network calls in this module.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def transfer_text_checkpoint(
    backbone: Any, path: Path, *, skip_words: bool = False, preflight_only: bool = False
) -> dict[str, Any]:
    prefix = "model.language_model."
    target = backbone.state_dict()
    expected = {
        key: value
        for key, value in target.items()
        if not (skip_words and key == "embed_tokens.weight") and "cross_" not in key
    }
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        available = set(reader.keys())
        prefix = "model.language_model."
        for cand_prefix in ("model.language_model.", "model.", "language_model.", ""):
            cand_test = [cand_prefix + k for k in ("embed_tokens.weight", "norm.weight")]
            if any(k in available for k in cand_test):
                prefix = cand_prefix
                break
        missing = [prefix + key for key in expected if prefix + key not in available]
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {missing[:5]}")
        for key, value in expected.items():
            shape = reader.get_slice(prefix + key).get_shape()
            if tuple(shape) != tuple(value.shape):
                raise ValueError(
                    f"Shape mismatch for {key}: source {shape}, target {tuple(value.shape)}"
                )
        if not preflight_only and any(value.is_meta for value in expected.values()):
            raise ValueError("Initialize destination storage before checkpoint transfer.")
        ignored_shared = []
        skipped = {prefix + "embed_tokens.weight"} if skip_words else set()
        for source_key in available:
            if not source_key.startswith(prefix) or source_key in skipped:
                continue
            key = source_key[len(prefix) :]
            if key in expected:
                continue
            if key in ("lm_head.weight", "lm_head.bias"):
                ignored_shared.append(key)
                continue
            match = re.fullmatch(
                r"layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm|v_norm)\.weight", key
            )
            layers = getattr(backbone, "layers", ())
            if (
                match
                and int(match[1]) < len(layers)
                and layers[int(match[1])].self_attn.is_kv_shared_layer
            ):
                ignored_shared.append(key)
            else:
                raise ValueError(f"Unmapped text tensor: {key}")
        if not preflight_only:
            with torch.no_grad():
                for key, destination in expected.items():
                    source = reader.get_tensor(prefix + key)
                    # Direct CPU-mmap -> destination copy avoids a second large GPU PLE tensor.
                    destination.copy_(source)
    return {
        "text_prefix": prefix,
        "tensors_copied": 0 if preflight_only else len(expected),
        "tensors_validated": len(expected),
        "preflight_only": preflight_only,
        "encoder_words_skipped": skip_words,
        "ignored_shared_kv_tensors": sorted(ignored_shared),
        "nontext_tensors_ignored": sum(not key.startswith(prefix) for key in available),
    }
