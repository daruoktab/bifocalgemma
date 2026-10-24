"""Explicit native checkpoint loading and adapter-only inference restoration."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from bifocalgemma.models.checkpoint_transfer import transfer_text_checkpoint
from bifocalgemma.models.native import build_native_model
from bifocalgemma.training.memory_probe import tensor_bytes
from bifocalgemma.training.policy import configure_policy_model


def load_local_native_model(
    spec,
    decoder_checkpoint,
    encoder_checkpoint=None,
    *,
    device="cuda",
    dtype=torch.bfloat16,
    tiny=False,
    headroom_gib=8.0,
    seed=17,
):
    """Full models require CUDA/preflight; no downloads or CPU random fallback."""
    decoder_checkpoint = Path(decoder_checkpoint)
    encoder_checkpoint = Path(encoder_checkpoint) if encoder_checkpoint is not None else None
    if not decoder_checkpoint.is_file() or (
        spec.encoder and (encoder_checkpoint is None or not encoder_checkpoint.is_file())
    ):
        raise FileNotFoundError("Both required local text source checkpoints must exist.")
    if not spec.encoder and encoder_checkpoint is not None:
        raise ValueError("Baseline must not receive an encoder checkpoint.")
    if headroom_gib < 0:
        raise ValueError("Negative headroom.")
    if not tiny:
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Full checkpoint loading requires CUDA; no weights allocated.")
        if dtype != torch.bfloat16 or not torch.cuda.is_bf16_supported():
            raise ValueError("Full loader currently requires supported BF16 CUDA.")
    blueprint = build_native_model(spec, tiny=tiny, device="meta", dtype=dtype)
    transfer_text_checkpoint(blueprint.decoder, decoder_checkpoint, preflight_only=True)
    if spec.encoder:
        assert encoder_checkpoint is not None
        transfer_text_checkpoint(
            blueprint.encoder,
            encoder_checkpoint,
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
    mapping = {"decoder": transfer_text_checkpoint(model.decoder, decoder_checkpoint)}
    if spec.encoder:
        assert encoder_checkpoint is not None
        mapping["encoder"] = transfer_text_checkpoint(
            model.encoder,
            encoder_checkpoint,
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


def load_sft_adapters_for_inference(model, checkpoint_dir) -> dict:
    """Validate checksum, family/spec and every tensor before changing model parameters."""
    root = Path(checkpoint_dir)
    marker = json.loads((root / "READY.json").read_text(encoding="utf-8"))
    for name in ("adapters.safetensors", "manifest.json"):
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if marker.get(name) != digest:
            raise ValueError("Adapter checkpoint checksum mismatch.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in {"bifocalgemma_notebook_sft_v2", "bifocalgemma_normalized_sft_v3"}:
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
