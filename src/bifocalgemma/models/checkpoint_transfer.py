"""Strict text-only safetensors transfer and local native/adapter loaders.

Validates all keys/shapes before any copy. No network calls in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file


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


def load_local_native_model(
    spec: Any,
    decoder_checkpoint: str | Path,
    encoder_checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    tiny: bool = False,
    headroom_gib: float = 8.0,
    seed: int = 17,
) -> tuple[Any, dict[str, Any]]:
    """Full models require CUDA/preflight; no downloads or CPU random fallback."""
    from bifocalgemma.modeling_bifocalgemma import build_native_model
    from bifocalgemma.training.memory_probe import tensor_bytes

    decoder_path = Path(decoder_checkpoint)
    encoder_path = Path(encoder_checkpoint) if encoder_checkpoint is not None else None
    if not decoder_path.is_file() or (
        spec.encoder and (encoder_path is None or not encoder_path.is_file())
    ):
        raise FileNotFoundError("Both required local text source checkpoints must exist.")
    if not spec.encoder and encoder_path is not None:
        raise ValueError("Baseline must not receive an encoder checkpoint.")
    if headroom_gib < 0:
        raise ValueError("Negative headroom.")
    if not tiny:
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Full checkpoint loading requires CUDA; no weights allocated.")
        if dtype != torch.bfloat16 or not torch.cuda.is_bf16_supported():
            raise ValueError("Full loader currently requires supported BF16 CUDA.")
    blueprint = build_native_model(spec, tiny=tiny, device="meta", dtype=dtype)
    transfer_text_checkpoint(blueprint.decoder, decoder_path, preflight_only=True)
    if spec.encoder:
        assert encoder_path is not None
        transfer_text_checkpoint(
            blueprint.encoder,
            encoder_path,
            skip_words=spec.tie_encoder_decoder_words,
            preflight_only=True,
        )
    weight_bytes = tensor_bytes(blueprint.parameters())
    del blueprint
    if not tiny:
        free, _ = torch.cuda.mem_get_info(device)
        if free < weight_bytes + int(headroom_gib * 2**30):
            raise ValueError(
                "Insufficient free VRAM for native weights plus headroom; no allocation."
            )
    torch.manual_seed(seed)
    model = build_native_model(spec, tiny=tiny, device=device, dtype=dtype)
    mapping = {"decoder": transfer_text_checkpoint(model.decoder, decoder_path)}
    if spec.encoder:
        assert encoder_path is not None
        mapping["encoder"] = transfer_text_checkpoint(
            model.encoder,
            encoder_path,
            skip_words=spec.tie_encoder_decoder_words,
        )
    model._tie_words()
    return model, {
        "experiment": spec.name,
        "expected_sources": spec.to_dict(),
        "mapping": mapping,
        "weights_loaded": True,
        "source_identity_verified": False,
        "source_identity_note": "Local files: shape/key validation does not prove upstream revision identity.",
        "tiny": tiny,
        "PLE_shared": False,
        "use_cache": False,
    }


def load_sft_adapters_for_inference(model: Any, checkpoint_dir: str | Path) -> dict[str, Any]:
    """Validate checksum, family/spec and every tensor before changing model parameters."""
    from bifocalgemma.modeling_bifocalgemma import build_native_model
    from bifocalgemma.training.policy import configure_policy_model

    root = Path(checkpoint_dir)
    marker = json.loads((root / "READY.json").read_text(encoding="utf-8"))
    for name in ("adapters.safetensors", "manifest.json"):
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if marker.get(name) != digest:
            raise ValueError("Adapter checkpoint checksum mismatch.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in {
        "bifocalgemma_notebook_sft_v2",
        "bifocalgemma_normalized_sft_v3",
    }:
        raise ValueError("Inference loader requires checkpoint v2.")
    compatibility = manifest["compatibility"]
    if compatibility.get("model_spec", model.spec.to_dict()) != model.spec.to_dict():
        raise ValueError("Adapter architecture/backbone revisions do not match model.")
    if compatibility.get("experiment") != model.spec.name:
        raise ValueError("Adapter experiment does not match model.")
    policy = compatibility.get("training_policy")
    if policy is None:
        raise ValueError("Checkpoint has no explicit training policy.")
    # Only adapt a fresh base; never silently reinitialize an existing adapter.
    if any(".lora_A" in name for name, _ in model.named_parameters()):
        raise ValueError("Load inference adapters on a fresh base model.")
    weights = load_file(str(root / "adapters.safetensors"), device="cpu")
    with torch.device("meta"):
        blueprint = build_native_model(
            model.spec, tiny=model.decoder.config.vocab_size == 128, device="meta"
        )
        configure_policy_model(blueprint, policy)
    expected = {name: tuple(p.shape) for name, p in blueprint.named_parameters() if p.requires_grad}
    if sorted(expected) != manifest.get("trainable_names") or set(weights) != set(expected):
        raise ValueError("Adapter parameter names mismatch.")
    if any(tuple(weights[name].shape) != shape for name, shape in expected.items()):
        raise ValueError("Adapter parameter shape mismatch.")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in weights.values()):
        raise ValueError("Nonfinite adapter weights.")
    del blueprint
    configure_policy_model(model, policy)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in weights.items():
            parameters[name].copy_(tensor.to(parameters[name].device))
    model._tie_words()
    model.eval()
    return {
        "format": manifest["format"],
        "global_step": manifest["global_step"],
        "training_policy": policy,
    }
