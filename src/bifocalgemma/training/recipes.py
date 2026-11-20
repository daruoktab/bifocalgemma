"""Shared experimental recipes and training-length-relative I/O cadence."""

from __future__ import annotations

import copy
import math
from typing import Any

import bifocalgemma.config as _config
from bifocalgemma.config import WARMUP_RATIO
from bifocalgemma.training.policy import default_training_profiles


def v4_optimizer_defaults() -> dict[str, Any]:
    """Canonical AdamW hyperparameters shared identically by Bifocal and decoder_only SFT."""
    raw_betas = getattr(_config, "ADAM_BETAS", (0.9, 0.95))
    betas = [float(raw_betas[0]), float(raw_betas[1])]
    return {
        "name": "adamw",
        "betas": betas,
        "eps": float(getattr(_config, "ADAM_EPS", 1e-8)),
        "weight_decay": 0.0,
    }


def v4_group_defaults() -> dict[str, dict[str, Any]]:
    """Canonical 6-group v4 LR, weight-decay, grad-clip, and warmup defaults.

    Incorporates the Argon #116 fixes:
    - `cross_gate_scalars` holds ONLY `cross_gate` (`wd=0.0`, `lr=2e-3`, unclipped by default).
    - `cross_gate_proj` has its own clip group (`wd=0.0`, `lr=5e-4`, `clip_norm=1.0`), separate from `bridge_mlp`.
    - `bridge_mlp` holds SwiGLU bridge 2D linears (`wd=1e-4`, `lr=4e-4`, `clip_norm=1.0`).
    - `cross_kv_proj` holds cross K/V projections, norms, bridge norms/gamma, and optional `cross_scale` (`wd=0.0`, `lr=1.5e-4`).
    - `encoder_lora` (`wd=0.0`, `lr=8e-5`) and `decoder_lora` (`wd=0.0`, `lr=4e-5`) each have own clip group `1.0`.
    """
    align_warmup = float(getattr(_config, "ALIGNMENT_WARMUP_RATIO", 0.03))
    lora_warmup = float(getattr(_config, "LORA_WARMUP_RATIO", WARMUP_RATIO))
    asym_dec_warmup = float(getattr(_config, "ASYMMETRIC_DECODER_WARMUP_RATIO", 0.10))
    min_lr_ratio = float(getattr(_config, "MIN_LR_RATIO", 0.1))
    wsd_decay_fraction = float(getattr(_config, "WSD_DECAY_FRACTION", 0.20))
    default_clip = float(getattr(_config, "DEFAULT_GRAD_CLIP_NORM", 1.0))
    gate_scalar_clip = getattr(_config, "CROSS_GATE_SCALARS_CLIP_NORM", None)
    if gate_scalar_clip is not None:
        gate_scalar_clip = float(gate_scalar_clip)

    return {
        "cross_gate_scalars": {
            "lr": float(getattr(_config, "LR_CROSS_GATE_SCALARS", 2e-3)),
            "weight_decay": 0.0,
            "clip_norm": gate_scalar_clip,
            "warmup_steps": 0,
            "warmup_ratio": align_warmup,
            "scheduler": "wsd",
            "wsd_decay_fraction": wsd_decay_fraction,
            "min_lr_ratio": min_lr_ratio,
        },
        "cross_gate_proj": {
            "lr": float(getattr(_config, "LR_CROSS_GATE_PROJ", 5e-4)),
            "weight_decay": 0.0,
            "clip_norm": default_clip,
            "warmup_steps": 0,
            "warmup_ratio": align_warmup,
            "scheduler": "wsd",
            "wsd_decay_fraction": wsd_decay_fraction,
            "min_lr_ratio": min_lr_ratio,
        },
        "bridge_mlp": {
            "lr": float(getattr(_config, "LR_BRIDGE_MLP", 4e-4)),
            "weight_decay": float(getattr(_config, "WEIGHT_DECAY_BRIDGE_MLP", 1e-4)),
            "clip_norm": default_clip,
            "warmup_steps": 0,
            "warmup_ratio": align_warmup,
            "scheduler": "wsd",
            "wsd_decay_fraction": wsd_decay_fraction,
            "min_lr_ratio": min_lr_ratio,
        },
        "cross_kv_proj": {
            "lr": float(getattr(_config, "LR_CROSS_KV_PROJ", 1.5e-4)),
            "weight_decay": 0.0,
            "clip_norm": default_clip,
            "warmup_steps": 0,
            "warmup_ratio": align_warmup,
            "scheduler": "wsd",
            "wsd_decay_fraction": wsd_decay_fraction,
            "min_lr_ratio": min_lr_ratio,
        },
        "encoder_lora": {
            "lr": float(getattr(_config, "LR_ENCODER_LORA", 8e-5)),
            "weight_decay": 0.0,
            "clip_norm": default_clip,
            "warmup_steps": 0,
            "warmup_ratio": lora_warmup,
            "scheduler": "cosine",
            "min_lr_ratio": min_lr_ratio,
        },
        "decoder_lora": {
            "lr": float(getattr(_config, "LR_DECODER_LORA", 4e-5)),
            "weight_decay": 0.0,
            "clip_norm": default_clip,
            "warmup_steps": 0,
            "warmup_ratio": lora_warmup,
            "asymmetric_warmup_ratio": asym_dec_warmup,
            "scheduler": "cosine",
            "min_lr_ratio": min_lr_ratio,
        },
    }


def unified_recipe(family: str) -> dict[str, Any]:
    """One partitioned v4 AdamW policy for all matrix models with apples-to-apples decoder_lora."""
    policy = default_training_profiles()[family]
    defaults = v4_group_defaults()
    policy.update(
        version="unified_v4",
        train_alignment=family != "decoder_only",
        alignment_fp32=True,
    )
    if family == "decoder_only":
        policy["groups"] = {"decoder_lora": copy.deepcopy(defaults["decoder_lora"])}
    else:
        policy["groups"] = copy.deepcopy(defaults)
    policy["optimizer"] = v4_optimizer_defaults()
    return policy


def comparison_recipes(family):
    """Same data/LoRA/objective/LR numbers in every architecture and recipe.

    These are conservative experimental defaults, not a claim of an optimum.
    The combined optimized arm cannot isolate each optimizer mechanism.
    """
    policy = default_training_profiles()[family]
    policy["alignment_fp32"] = True
    for settings in policy["groups"].values():
        settings.update(
            lr=2e-5, warmup_steps=0, warmup_ratio=WARMUP_RATIO, scheduler="cosine", min_lr_ratio=0.1
        )
    baseline = copy.deepcopy(policy)
    baseline["optimizer"] = {
        "name": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    optimized = copy.deepcopy(policy)
    optimized["optimizer"] = {
        "name": "grok_orscale_lora_v1",
        "weight_decay": 0.0,
        "grok_decay": 0.98,
        "grok_amplification": 0.5,
        "momentum": 0.95,
        "ns_steps": 5,
        "trust_min": 0.5,
        "trust_max": 2.0,
    }
    return {"base": baseline, "optimized": optimized}


def resolve_training_cadence(
    total_steps, *, eval_every=None, save_every=None, eval_fraction=0.25, save_fraction=0.10
):
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("Training horizon must be a positive integer.")
    result = {"total_steps": total_steps}
    for name, explicit, fraction in (
        ("eval_every", eval_every, eval_fraction),
        ("save_every", save_every, save_fraction),
    ):
        if (
            not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 < fraction <= 1
        ):
            raise ValueError("Cadence fraction must be in (0, 1].")
        if explicit is not None and (type(explicit) is not int or explicit < 1):
            raise ValueError("Explicit cadence must be a positive integer.")
        result[name] = (
            explicit if explicit is not None else max(1, math.ceil(total_steps * fraction))
        )
    return result
