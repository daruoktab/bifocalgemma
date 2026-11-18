"""Two-stage optimizer, freezing, and continuous cosine warmup configuration."""

from __future__ import annotations

import math
from typing import Any

from bifocalgemma.config import WARMUP_RATIO
from bifocalgemma.training.memory_probe import ProbeLoRALinear
from bifocalgemma.training.objectives import (
    is_cross_bridge_parameter,
    is_zero_weight_decay_bridge_parameter,
)


def configure_matrix_stage(
    model: Any,
    policy: dict[str, Any],
    *,
    freeze_decoder: bool,
    total_steps: int,
    warmup_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Configure trainable groups for Stage 1 (frozen decoder on paired models) or Stage 2 (joint SFT).

    Enforces partitioned weight decay: 1e-4 on bridge linear weights, 0.0 on RMSNorm/calibration/cross-attn/LoRA.
    When `warmup_offset > 0` on a paired Stage 2 run, `encoder` and `alignment` continue their
    cosine schedule from step 0 across `total_steps`, while the newly unfrozen `decoder` group
    uses `warmup_offset` so its warmup and cosine decay span the Stage 2 horizon without an LR spike.
    """
    if total_steps < 1:
        raise ValueError("Stage horizon must be a positive integer.")
    if warmup_offset < 0 or warmup_offset >= total_steps:
        raise ValueError("warmup_offset must lie in [0, total_steps).")
    default_scaling = policy["alpha"] / policy["rank"]
    has_encoder = getattr(model, "encoder", None) is not None
    effective_freeze_decoder = freeze_decoder and has_encoder
    learnable_cross_scale = getattr(getattr(model, "config", None), "learnable_cross_scale", True)

    for name, module in model.named_modules():
        if isinstance(module, ProbeLoRALinear):
            if name.startswith("decoder."):
                module.scaling = 0.0 if effective_freeze_decoder else default_scaling
            else:
                module.scaling = default_scaling

    for name, param in model.named_parameters():
        if has_encoder and is_cross_bridge_parameter(name):
            param.requires_grad_(learnable_cross_scale or not name.endswith(".cross_scale"))
        elif has_encoder and name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(True)
        elif name.startswith("decoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(not effective_freeze_decoder)
        else:
            param.requires_grad_(False)

    if not has_encoder:
        active_names: tuple[str, ...] = ("decoder",)
    else:
        active_names = (
            ("encoder", "alignment")
            if effective_freeze_decoder
            else ("encoder", "decoder", "alignment")
        )
    entries: dict[str, list[tuple[str, Any]]] = {name: [] for name in active_names}
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        if has_encoder and is_cross_bridge_parameter(name):
            entries["alignment"].append((name, param))
        elif has_encoder and name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            entries["encoder"].append((name, param))
        elif (
            not effective_freeze_decoder
            and name.startswith("decoder.")
            and name.endswith((".lora_A", ".lora_B"))
        ):
            entries["decoder"].append((name, param))
        else:
            raise ValueError(f"Unexpected trainable parameter in matrix stage: {name}")

    groups: list[dict[str, Any]] = []
    for group_name in active_names:
        if not entries[group_name]:
            raise ValueError(f"Empty trainable group in matrix stage: {group_name}")
        settings = policy["groups"][group_name]
        warmup_ratio = float(settings.get("warmup_ratio", WARMUP_RATIO))
        group_offset = (
            warmup_offset if (group_name == "decoder" and has_encoder and warmup_offset > 0) else 0
        )
        effective_horizon = max(1, total_steps - group_offset)
        warmup_steps = min(
            effective_horizon,
            max(settings.get("warmup_steps", 0), math.ceil(effective_horizon * warmup_ratio)),
        )
        if group_name == "alignment":
            bridge_decay_entries = [
                (n, p)
                for n, p in entries["alignment"]
                if n.startswith("encoder_projection.")
                and not is_zero_weight_decay_bridge_parameter(n)
                and p.ndim >= 2
            ]
            nodecay_entries = [
                (n, p)
                for n, p in entries["alignment"]
                if not (
                    n.startswith("encoder_projection.")
                    and not is_zero_weight_decay_bridge_parameter(n)
                    and p.ndim >= 2
                )
            ]
            if bridge_decay_entries:
                groups.append(
                    {
                        "name": "alignment_bridge_decay",
                        "params": [p for _, p in bridge_decay_entries],
                        "param_names": [n for n, _ in bridge_decay_entries],
                        "lr": settings["lr"],
                        "base_lr": settings["lr"],
                        "warmup_steps": warmup_steps,
                        "warmup_offset": group_offset,
                        "scheduler": settings.get("scheduler", "cosine"),
                        "total_steps": total_steps,
                        "warmup_ratio": warmup_ratio,
                        "min_lr_ratio": settings.get("min_lr_ratio", 0.1),
                        "weight_decay": 1e-4,
                    }
                )
            if nodecay_entries:
                groups.append(
                    {
                        "name": "alignment",
                        "params": [p for _, p in nodecay_entries],
                        "param_names": [n for n, _ in nodecay_entries],
                        "lr": settings["lr"],
                        "base_lr": settings["lr"],
                        "warmup_steps": warmup_steps,
                        "warmup_offset": group_offset,
                        "scheduler": settings.get("scheduler", "cosine"),
                        "total_steps": total_steps,
                        "warmup_ratio": warmup_ratio,
                        "min_lr_ratio": settings.get("min_lr_ratio", 0.1),
                        "weight_decay": 0.0,
                    }
                )
        else:
            groups.append(
                {
                    "name": group_name,
                    "params": [p for _, p in entries[group_name]],
                    "param_names": [n for n, _ in entries[group_name]],
                    "lr": settings["lr"],
                    "base_lr": settings["lr"],
                    "warmup_steps": warmup_steps,
                    "warmup_offset": group_offset,
                    "scheduler": settings.get("scheduler", "cosine"),
                    "total_steps": total_steps,
                    "warmup_ratio": warmup_ratio,
                    "min_lr_ratio": settings.get("min_lr_ratio", 0.1),
                    "weight_decay": 0.0,
                }
            )
    summary = {
        "freeze_decoder": effective_freeze_decoder,
        "active_groups": list(active_names),
        "trainable_parameters": {
            name: sum(int(p.numel()) for _, p in entries[name]) for name in active_names
        },
        "decoder_trainable_tensors": sum(
            1
            for n, p in model.named_parameters()
            if n.startswith("decoder.") and not is_cross_bridge_parameter(n) and p.requires_grad
        ),
    }
    return groups, summary


configure_training_stage = configure_matrix_stage
