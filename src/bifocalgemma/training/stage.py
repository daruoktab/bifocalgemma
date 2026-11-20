"""Canonical 6-group partitioned optimizer, stage freezing, per-group clipping, and WSD/cosine schedules."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch

import bifocalgemma.config as _config
from bifocalgemma.config import WARMUP_RATIO
from bifocalgemma.training.memory_probe import ProbeLoRALinear
from bifocalgemma.training.objectives import (
    is_cross_bridge_parameter,
    is_zero_weight_decay_bridge_parameter,
)
from bifocalgemma.training.recipes import v4_group_defaults, v4_optimizer_defaults

CANONICAL_PARAM_GROUP_ORDER: tuple[str, ...] = (
    "cross_gate_scalars",
    "cross_gate_proj",
    "bridge_mlp",
    "cross_kv_proj",
    "encoder_lora",
    "decoder_lora",
)

ALIGNMENT_GROUP_NAMES: tuple[str, ...] = (
    "cross_gate_scalars",
    "cross_gate_proj",
    "bridge_mlp",
    "cross_kv_proj",
)

_LEGACY_POLICY_GROUP_MAP: dict[str, str] = {
    "cross_gate_scalars": "alignment",
    "cross_gate_proj": "alignment",
    "bridge_mlp": "alignment",
    "cross_kv_proj": "alignment",
    "encoder_lora": "encoder",
    "decoder_lora": "decoder",
}


def classify_bifocalgemma_parameter(name: str, param: torch.Tensor) -> str:
    """Classify a trainable parameter into one of the 6 canonical v4 optimizer/clip groups.

    Strict invariants (Argon #116):
    - `cross_gate_scalars`: ONLY `*.cross_gate` (never `cross_scale`, never `cross_gate_proj`).
    - `cross_gate_proj`: ONLY `*.cross_gate_proj` / `*.cross_gate_proj.*` (own clip group, NOT `bridge_mlp`).
    - `bridge_mlp`: 2D+ linear weights inside `encoder_projection.*` (strict `wd=1e-4`).
    - `cross_kv_proj`: `cross_k/v_proj`, `cross_*_norm`, bridge norms/gamma, optional `cross_scale` / `cross_q_delta`.
    - `encoder_lora`: `encoder.*.lora_A/B`.
    - `decoder_lora`: `decoder.*.lora_A/B` (or bare `*.lora_A/B` on decoder-only models).
    """
    if name.endswith(".cross_gate_proj") or ".cross_gate_proj." in name:
        return "cross_gate_proj"
    if name.endswith(".cross_gate"):
        return "cross_gate_scalars"
    if name.startswith("encoder_projection."):
        if not is_zero_weight_decay_bridge_parameter(name) and param.ndim >= 2:
            return "bridge_mlp"
        return "cross_kv_proj"
    if is_cross_bridge_parameter(name):
        return "cross_kv_proj"
    if (name.startswith("encoder.") or ".encoder." in name) and name.endswith(
        (".lora_A", ".lora_B")
    ):
        return "encoder_lora"
    if (name.startswith("decoder.") or ".decoder." in name) and name.endswith(
        (".lora_A", ".lora_B")
    ):
        return "decoder_lora"
    if name.endswith((".lora_A", ".lora_B")):
        return "decoder_lora"
    raise ValueError(f"Unexpected trainable parameter in BifocalGemma optimizer: {name}")


def _resolve_group_settings(
    policy: dict[str, Any] | None,
    group_name: str,
    group_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve effective settings for one canonical group from v4 defaults, policy, and overrides."""
    defaults = v4_group_defaults()[group_name]
    resolved = copy.deepcopy(defaults)
    if policy is not None:
        policy_groups = policy.get("groups", {})
        if group_name in policy_groups:
            resolved.update(policy_groups[group_name])
        else:
            legacy_key = _LEGACY_POLICY_GROUP_MAP.get(group_name)
            if legacy_key and legacy_key in policy_groups:
                legacy_cfg = policy_groups[legacy_key]
                for k, v in legacy_cfg.items():
                    resolved[k] = v
                # Preserve canonical WD and clip boundaries even when given a legacy policy dict.
                if "weight_decay" not in legacy_cfg:
                    resolved["weight_decay"] = defaults["weight_decay"]
                if "clip_norm" not in legacy_cfg:
                    resolved["clip_norm"] = defaults["clip_norm"]
    if group_overrides and group_name in group_overrides:
        resolved.update(group_overrides[group_name])
    return resolved


def build_partitioned_param_groups(
    model: Any,
    *,
    policy: dict[str, Any] | None = None,
    total_steps: int | None = None,
    warmup_offset: int = 0,
    freeze_decoder: bool = False,
    asymmetric_warmup: bool | None = None,
    group_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the single canonical 6-group partitioned AdamW param groups for BifocalGemma.

    Shared by `configure_matrix_stage` (matrix.py) and `build_bifocalgemma_optimizer`
    (objectives.py / fast_pipeline.py) so optimizer partitioning can never diverge.
    """
    if total_steps is not None:
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("Stage horizon must be a positive integer.")
        if type(warmup_offset) is not int or warmup_offset < 0 or warmup_offset >= total_steps:
            raise ValueError("warmup_offset must lie in [0, total_steps).")
    elif warmup_offset < 0:
        raise ValueError("warmup_offset must be nonnegative.")

    has_encoder = getattr(model, "encoder", None) is not None
    effective_freeze_decoder = freeze_decoder and has_encoder
    use_asymmetric_warmup = (
        asymmetric_warmup
        if asymmetric_warmup is not None
        else (has_encoder and not effective_freeze_decoder and warmup_offset == 0)
    )

    entries: dict[str, list[tuple[str, Any]]] = {name: [] for name in CANONICAL_PARAM_GROUP_ORDER}
    seen_ids: set[int] = set()
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        group_name = classify_bifocalgemma_parameter(name, param)
        if group_name == "decoder_lora" and effective_freeze_decoder:
            raise ValueError(f"Frozen decoder parameter unexpectedly trainable: {name}")
        entries[group_name].append((name, param))

    groups: list[dict[str, Any]] = []
    for group_name in CANONICAL_PARAM_GROUP_ORDER:
        bucket = entries[group_name]
        if not bucket:
            continue
        settings = _resolve_group_settings(policy, group_name, group_overrides)
        lr = float(settings["lr"])
        weight_decay = float(settings.get("weight_decay", 0.0))
        clip_norm = settings.get("clip_norm", 1.0)
        if clip_norm is not None:
            clip_norm = float(clip_norm)

        group_dict: dict[str, Any] = {
            "name": group_name,
            "params": [p for _, p in bucket],
            "param_names": [n for n, _ in bucket],
            "lr": lr,
            "base_lr": lr,
            "weight_decay": weight_decay,
            "clip_norm": clip_norm,
        }

        if total_steps is not None:
            scheduler = str(settings.get("scheduler", "cosine"))
            min_lr_ratio = float(settings.get("min_lr_ratio", 0.1))
            base_warmup_ratio = float(settings.get("warmup_ratio", WARMUP_RATIO))
            explicit_warmup_steps = int(settings.get("warmup_steps", 0))
            is_alignment = group_name in ALIGNMENT_GROUP_NAMES

            if group_name == "decoder_lora" and has_encoder and warmup_offset > 0:
                group_offset = warmup_offset
                warmup_ratio = base_warmup_ratio
                effective_horizon = max(1, total_steps - group_offset)
                warmup_steps = min(
                    effective_horizon,
                    max(explicit_warmup_steps, math.ceil(effective_horizon * warmup_ratio)),
                )
                group_dict.update(
                    warmup_steps=warmup_steps,
                    warmup_offset=group_offset,
                    scheduler=scheduler,
                    total_steps=total_steps,
                    warmup_ratio=warmup_ratio,
                    min_lr_ratio=min_lr_ratio,
                )
            elif group_name == "decoder_lora" and use_asymmetric_warmup:
                # 0%-freeze asymmetric-warmup mode: decoder warms up slowly (default 0.10)
                # while bridge/cross-attn warm up fast (0.03).
                group_offset = 0
                warmup_ratio = float(settings.get("asymmetric_warmup_ratio", 0.10))
                effective_horizon = total_steps
                warmup_steps = min(
                    effective_horizon,
                    max(explicit_warmup_steps, math.ceil(effective_horizon * warmup_ratio)),
                )
                group_dict.update(
                    warmup_steps=warmup_steps,
                    warmup_offset=group_offset,
                    scheduler=scheduler,
                    total_steps=total_steps,
                    warmup_ratio=warmup_ratio,
                    min_lr_ratio=min_lr_ratio,
                )
            elif is_alignment:
                # Alignment groups NEVER inherit decayed Stage-1 cosine LR when Stage 2 begins:
                # - Under WSD (default): hold flat-top during Stage 1 AND at Stage-2 entry, then decay.
                # - Under cosine: reset cosine cleanly across Stage 2 starting at exact peak LR.
                warmup_ratio = base_warmup_ratio
                if scheduler == "wsd":
                    group_offset = 0
                    warmup_steps = min(
                        total_steps,
                        max(explicit_warmup_steps, math.ceil(total_steps * warmup_ratio)),
                    )
                    wsd_decay_fraction = float(settings.get("wsd_decay_fraction", 0.20))
                    if effective_freeze_decoder:
                        # Hold flat-top throughout Stage 1 warm-start; never decay before Stage 2.
                        stable_until_step: int | None = total_steps
                    elif warmup_offset > 0:
                        # Stage 2 begins at full peak LR (step == warmup_offset -> 1.0 * base_lr)
                        # and decays smoothly over the tail of Stage 2.
                        stable_until_step = (
                            min(total_steps - 1, warmup_offset + 1)
                            if total_steps > warmup_offset + 1
                            else warmup_offset
                        )
                    else:
                        stable_until_step = None
                    group_dict.update(
                        warmup_steps=warmup_steps,
                        warmup_offset=group_offset,
                        scheduler="wsd",
                        total_steps=total_steps,
                        warmup_ratio=warmup_ratio,
                        min_lr_ratio=min_lr_ratio,
                        wsd_decay_fraction=wsd_decay_fraction,
                        stable_until_step=stable_until_step,
                    )
                elif scheduler == "cosine" and warmup_offset > 0:
                    # Clean Stage-2 cosine reset: warmup_steps=1 at offset=warmup_offset yields
                    # exact peak LR (1.0 * base_lr) on first Stage-2 step and smooth cosine decay
                    # to min_lr_ratio on the final step (total_steps - 1).
                    group_dict.update(
                        warmup_steps=1,
                        warmup_offset=warmup_offset,
                        scheduler="cosine",
                        total_steps=total_steps,
                        warmup_ratio=warmup_ratio,
                        min_lr_ratio=min_lr_ratio,
                    )
                elif scheduler == "cosine" and effective_freeze_decoder:
                    # Stage 1 warm-start with cosine policy: warm up fast and hold peak during
                    # Stage 1 so Stage 2 cosine reset transitions smoothly from peak -> peak.
                    warmup_steps = min(
                        total_steps,
                        max(explicit_warmup_steps, math.ceil(total_steps * warmup_ratio)),
                    )
                    group_dict.update(
                        warmup_steps=warmup_steps,
                        warmup_offset=0,
                        scheduler="wsd",
                        total_steps=total_steps,
                        warmup_ratio=warmup_ratio,
                        min_lr_ratio=min_lr_ratio,
                        wsd_decay_fraction=float(settings.get("wsd_decay_fraction", 0.20)),
                        stable_until_step=total_steps,
                    )
                else:
                    warmup_steps = min(
                        total_steps,
                        max(explicit_warmup_steps, math.ceil(total_steps * warmup_ratio)),
                    )
                    group_dict.update(
                        warmup_steps=warmup_steps,
                        warmup_offset=0,
                        scheduler=scheduler,
                        total_steps=total_steps,
                        warmup_ratio=warmup_ratio,
                        min_lr_ratio=min_lr_ratio,
                    )
            else:
                # encoder_lora or decoder_only baseline decoder_lora
                group_offset = 0
                warmup_ratio = base_warmup_ratio
                warmup_steps = min(
                    total_steps,
                    max(explicit_warmup_steps, math.ceil(total_steps * warmup_ratio)),
                )
                group_dict.update(
                    warmup_steps=warmup_steps,
                    warmup_offset=group_offset,
                    scheduler=scheduler,
                    total_steps=total_steps,
                    warmup_ratio=warmup_ratio,
                    min_lr_ratio=min_lr_ratio,
                )

        groups.append(group_dict)

    active_names = [g["name"] for g in groups]
    summary = {
        "freeze_decoder": effective_freeze_decoder,
        "asymmetric_warmup": use_asymmetric_warmup,
        "active_groups": active_names,
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


def build_bifocalgemma_adamw(
    groups: list[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
    betas: tuple[float, float] | None = None,
    eps: float | None = None,
) -> torch.optim.AdamW:
    """Instantiate canonical AdamW optimizer with v4 default betas=(0.9, 0.95) and foreach=False."""
    default_opt = v4_optimizer_defaults()
    opt_cfg = (policy or {}).get("optimizer", default_opt)
    resolved_betas = (
        tuple(float(b) for b in betas)
        if betas is not None
        else tuple(float(b) for b in opt_cfg.get("betas", default_opt["betas"]))
    )
    resolved_eps = float(eps if eps is not None else opt_cfg.get("eps", default_opt["eps"]))
    resolved_wd = float(opt_cfg.get("weight_decay", 0.0))
    return torch.optim.AdamW(
        groups,
        betas=(resolved_betas[0], resolved_betas[1]),
        eps=resolved_eps,
        weight_decay=resolved_wd,
        foreach=False,
    )


def clip_optimizer_group_grads(
    optimizer: torch.optim.Optimizer,
    *,
    default_clip_norm: float | None = 1.0,
) -> tuple[float, dict[str, float]]:
    """Clip gradients independently per param group, respecting per-group `clip_norm`.

    - Each group (e.g. `cross_gate_proj`, `bridge_mlp`, `cross_kv_proj`, `encoder_lora`,
      `decoder_lora`) is clipped in its own isolated L2 norm bucket so large-norm bridge
      or KV gradients cannot starve small gate/LoRA projections.
    - Groups with `clip_norm=None` (default for `cross_gate_scalars`) are left unclipped
      while still reporting their exact L2 gradient norm for telemetry.
    """
    group_grad_norms: dict[str, float] = {}
    for group in optimizer.param_groups:
        group_params = [p for p in group["params"] if p.grad is not None]
        if not group_params:
            continue
        group_name = str(group.get("name", "default"))
        max_norm = group["clip_norm"] if "clip_norm" in group else default_clip_norm
        if max_norm is None:
            sq_norm = sum(
                float(p.grad.detach().float().pow(2).sum().item()) for p in group_params
            )
            group_grad_norms[group_name] = math.sqrt(sq_norm)
        else:
            group_grad_norms[group_name] = float(
                torch.nn.utils.clip_grad_norm_(group_params, float(max_norm), foreach=False)
            )
    total_norm = math.sqrt(sum(v * v for v in group_grad_norms.values()))
    return total_norm, group_grad_norms


def configure_matrix_stage(
    model: Any,
    policy: dict[str, Any],
    *,
    freeze_decoder: bool,
    total_steps: int,
    warmup_offset: int = 0,
    asymmetric_warmup: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Configure trainable groups for Stage 1 (short warm-start) or Stage 2 / Step-0 joint SFT.

    Uses the canonical 6-group partitioned optimizer builder:
    - `cross_gate_scalars` (ONLY `cross_gate`, `wd=0.0`, `lr=2e-3`, unclipped by default)
    - `cross_gate_proj` (`wd=0.0`, `lr=5e-4`, own clip group `1.0`)
    - `bridge_mlp` (`encoder_projection` 2D linears, `wd=1e-4`, `lr=4e-4`, own clip group `1.0`)
    - `cross_kv_proj` (`cross_k/v_proj`, norms, bridge norms/gamma, optional `cross_scale`, `wd=0.0`, `lr=1.5e-4`)
    - `encoder_lora` (`wd=0.0`, `lr=8e-5`, own clip group `1.0`)
    - `decoder_lora` (`wd=0.0`, `lr=4e-5`, own clip group `1.0`, identical on Bifocal and `decoder_only`)

    When `warmup_offset > 0` on a paired Stage-2 run, alignment groups NEVER inherit decayed
    Stage-1 cosine LR (WSD flat-top or clean Stage-2 cosine reset), while `decoder_lora` warms up
    over Stage 2. When `warmup_offset == 0` and `freeze_decoder=False` on a paired model (0%-freeze
    asymmetric-warmup mode), `decoder_lora` uses delayed warmup (`0.10`) while alignment uses `0.03`.
    """
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("Stage horizon must be a positive integer.")
    if type(warmup_offset) is not int or warmup_offset < 0 or warmup_offset >= total_steps:
        raise ValueError("warmup_offset must lie in [0, total_steps).")

    default_scaling = policy["alpha"] / policy["rank"]
    has_encoder = getattr(model, "encoder", None) is not None
    effective_freeze_decoder = freeze_decoder and has_encoder
    default_learnable_cross_scale = bool(getattr(_config, "LEARNABLE_CROSS_SCALE", False))
    learnable_cross_scale = bool(
        getattr(getattr(model, "config", None), "learnable_cross_scale", default_learnable_cross_scale)
    )

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

    groups, summary = build_partitioned_param_groups(
        model,
        policy=policy,
        total_steps=total_steps,
        warmup_offset=warmup_offset,
        freeze_decoder=effective_freeze_decoder,
        asymmetric_warmup=asymmetric_warmup,
    )

    group_names = {g["name"] for g in groups}
    if not has_encoder:
        if "decoder_lora" not in group_names:
            raise ValueError("Empty trainable group in matrix stage: decoder_lora")
    else:
        if "encoder_lora" not in group_names:
            raise ValueError("Empty trainable group in matrix stage: encoder_lora")
        if not any(name in group_names for name in ALIGNMENT_GROUP_NAMES):
            raise ValueError("Empty alignment trainable groups in matrix stage.")
        if not effective_freeze_decoder and "decoder_lora" not in group_names:
            raise ValueError("Empty trainable group in matrix stage: decoder_lora")

    summary["learnable_cross_scale"] = learnable_cross_scale
    return groups, summary


configure_training_stage = configure_matrix_stage
