"""Explicit family policy and deterministic optimizer groups for notebook SFT."""

import copy
import math
from typing import Any

import torch.nn as nn

from bifocalgemma.training.memory_probe import ProbeLoRALinear, configure_probe_training


class Float32Alignment(nn.Linear):
    """Keep small learned alignment in FP32 while returning backbone dtype."""

    def forward(self, input):
        return nn.functional.linear(input.float(), self.weight, self.bias).to(input.dtype)


def default_training_profiles():
    common = {
        "version": "attention_lora_policy_v1",
        "rank": 16,
        "alpha": 32,
        "freeze_words": True,
        "freeze_ple": True,
        "phase": "joint",
    }
    profiles = {}
    for family in ("symmetric", "asymmetric"):
        groups = {name: {"lr": 2e-5, "warmup_steps": 5} for name in ("encoder", "decoder")}
        if family == "asymmetric":
            groups["alignment"] = {"lr": 2e-5, "warmup_steps": 5}
        profiles[family] = {
            **copy.deepcopy(common),
            "family": family,
            "train_alignment": family == "asymmetric",
            "groups": groups,
        }
    profiles["decoder_only"] = {
        **copy.deepcopy(common),
        "family": "decoder_only",
        "train_alignment": False,
        "groups": {"decoder": {"lr": 2e-5, "warmup_steps": 5}},
    }
    return profiles


def validate_training_policy(policy, model=None):
    if "optimizer" in policy:
        from bifocalgemma.training.optimizers import validate_optimizer_config

        validate_optimizer_config(policy["optimizer"])
    if type(policy.get("alignment_fp32", False)) is not bool:
        raise ValueError("alignment_fp32 must be boolean.")
    family = policy.get("family")
    if family not in {"symmetric", "asymmetric", "decoder_only"}:
        raise ValueError("Unknown training-policy family.")
    v2 = policy.get("version") == "unified_v2"
    v4 = policy.get("version") == "unified_v4"
    v2_like = v2 or v4
    if (
        policy.get("version") not in {"attention_lora_policy_v1", "unified_v2", "unified_v4"}
        or policy.get("phase") != "joint"
    ):
        raise ValueError(
            "Only joint attention LoRA policy v1, unified_v2, or unified_v4 is supported."
        )
    if policy.get("freeze_words") is not True or policy.get("freeze_ple") is not True:
        raise ValueError("Word/PLE training needs a separately validated policy.")
    if type(policy.get("rank")) is not int or policy["rank"] < 1:
        raise ValueError("LoRA rank must be a positive integer.")
    if (
        not isinstance(policy.get("alpha"), (int, float))
        or not math.isfinite(policy["alpha"])
        or policy["alpha"] <= 0
    ):
        raise ValueError("LoRA alpha must be finite and positive.")
    if v4:
        expected = (
            {"decoder_lora"}
            if family == "decoder_only"
            else {
                "cross_gate_scalars",
                "cross_gate_proj",
                "bridge_mlp",
                "cross_kv_proj",
                "encoder_lora",
                "decoder_lora",
            }
        )
        expected_train_alignment = family != "decoder_only"
    else:
        expected = ({"decoder"} if family == "decoder_only" else {"encoder", "decoder"}) | (
            {"alignment"} if family == "asymmetric" or (v2 and family == "symmetric") else set()
        )
        expected_train_alignment = family == "asymmetric" or (v2 and family == "symmetric")
    if (
        set(policy.get("groups", {})) != expected
        or policy.get("train_alignment") != expected_train_alignment
    ):
        raise ValueError("Policy groups/alignment do not match family.")
    for name, settings in policy["groups"].items():
        lr, warmup = settings.get("lr"), settings.get("warmup_steps")
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Invalid LR for {name}.")
        if type(warmup) is not int or warmup < 0:
            raise ValueError(f"Invalid warmup for {name}.")
        if settings.get("scheduler", "constant") not in {"constant", "cosine", "wsd"}:
            raise ValueError(f"Invalid scheduler for {name}.")
        for key in ("warmup_ratio", "min_lr_ratio", "asymmetric_warmup_ratio"):
            if key == "asymmetric_warmup_ratio" and key not in settings:
                continue
            value = settings.get(key, 0.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f"Invalid {key} for {name}.")
        if "wsd_decay_fraction" in settings:
            decay_frac = settings["wsd_decay_fraction"]
            if (
                not isinstance(decay_frac, (int, float))
                or isinstance(decay_frac, bool)
                or not math.isfinite(decay_frac)
                or not 0 < decay_frac <= 1
            ):
                raise ValueError(f"Invalid wsd_decay_fraction for {name}.")
        if "weight_decay" in settings:
            wd = settings["weight_decay"]
            if (
                not isinstance(wd, (int, float))
                or isinstance(wd, bool)
                or not math.isfinite(wd)
                or wd < 0
            ):
                raise ValueError(f"Invalid weight_decay for {name}.")
        if "clip_norm" in settings and settings["clip_norm"] is not None:
            clip = settings["clip_norm"]
            if (
                not isinstance(clip, (int, float))
                or isinstance(clip, bool)
                or not math.isfinite(clip)
                or clip <= 0
            ):
                raise ValueError(f"Invalid clip_norm for {name}.")
    if model is not None:
        if v2_like and model.spec.architecture_version != "normalized_merged_v2":
            raise ValueError(
                "unified_v2/unified_v4 requires a v2 model; legacy checkpoints cannot resume this architecture."
            )
        if family == "decoder_only":
            if model.encoder is not None or model.encoder_projection is not None:
                raise ValueError("Decoder-only policy requires a baseline model.")
            return
        if getattr(model, "encoder", None) is None or model.spec.tie_encoder_decoder_words != (
            family == "symmetric" and not v2_like
        ):
            raise ValueError("Model architecture does not match policy family.")
        projection = model.encoder_projection
        if v2_like:
            from bifocalgemma.models.bridge import NormalizedAlignment, SwiGLUDualGateBridge

            if model.spec.architecture_version != "normalized_merged_v2" or not isinstance(
                projection, (NormalizedAlignment, SwiGLUDualGateBridge)
            ):
                raise ValueError("unified_v2/unified_v4 requires normalized_merged_v2 alignment.")
            return
        if family == "symmetric" and not isinstance(projection, nn.Identity):
            raise ValueError("Symmetric policy expects Identity alignment.")
        if family == "asymmetric" and not isinstance(projection, nn.Linear):
            raise ValueError("Asymmetric policy expects a learned Linear alignment.")


def configure_policy_model(model, policy):
    validate_training_policy(policy, model)
    scope = configure_probe_training(model, policy="lora", rank=policy["rank"])
    if policy["version"] in {"unified_v2", "unified_v4"} and model.encoder_projection is not None:
        model.encoder_projection.float().requires_grad_(True)
    if policy.get("alignment_fp32") and isinstance(model.encoder_projection, nn.Linear):
        model.encoder_projection.__class__ = Float32Alignment
        model.encoder_projection.float()
    for module in model.modules():
        if isinstance(module, ProbeLoRALinear):
            module.scaling = policy["alpha"] / policy["rank"]
    scope["trainable_bytes"] = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    return scope


def _build_group_schedule(settings: dict[str, Any], total_steps: int | None) -> dict[str, Any]:
    scheduler = settings.get("scheduler", "constant")
    if scheduler not in {"cosine", "wsd"}:
        return {}
    if total_steps is not None and (type(total_steps) is not int or total_steps < 1):
        raise ValueError("Schedule horizon must be a positive integer.")
    schedule: dict[str, Any] = {
        "scheduler": scheduler,
        "total_steps": total_steps,
        "warmup_ratio": settings.get("warmup_ratio", 0.0),
        "min_lr_ratio": settings.get("min_lr_ratio", 0.0),
    }
    if scheduler == "wsd" and "wsd_decay_fraction" in settings:
        schedule["wsd_decay_fraction"] = settings["wsd_decay_fraction"]
    if total_steps is not None:
        schedule["warmup_steps"] = min(
            total_steps,
            max(settings["warmup_steps"], math.ceil(total_steps * schedule["warmup_ratio"])),
        )
    return schedule


def build_training_groups(model, policy, total_steps=None) -> "list[dict[str, Any]]":
    validate_training_policy(policy, model)
    is_v4 = policy.get("version") == "unified_v4"
    seen = set()
    if is_v4:
        from bifocalgemma.training.stage import (
            CANONICAL_PARAM_GROUP_ORDER,
            classify_bifocalgemma_parameter,
        )

        entries: dict[str, list[Any]] = {name: [] for name in CANONICAL_PARAM_GROUP_ORDER}
        # named_parameters deduplicates shared Parameter objects, including aliases.
        for name, parameter in sorted(model.named_parameters()):
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError("A trainable Parameter was enumerated twice.")
            seen.add(id(parameter))
            group = classify_bifocalgemma_parameter(name, parameter)
            if group not in policy["groups"]:
                raise ValueError(f"Unexpected trainable parameter for policy: {name}")
            entries[group].append((name, parameter))
        groups = []
        for name in CANONICAL_PARAM_GROUP_ORDER:
            if name not in policy["groups"]:
                continue
            if not entries[name]:
                if policy["family"] == "decoder_only":
                    raise ValueError(f"Empty trainable group: {name}")
                continue
            settings = policy["groups"][name]
            schedule = _build_group_schedule(settings, total_steps)
            extra: dict[str, Any] = {}
            if "weight_decay" in settings:
                extra["weight_decay"] = float(settings["weight_decay"])
            if "clip_norm" in settings:
                extra["clip_norm"] = (
                    None if settings["clip_norm"] is None else float(settings["clip_norm"])
                )
            groups.append(
                {
                    "name": name,
                    "params": [p for _, p in entries[name]],
                    "param_names": [n for n, _ in entries[name]],
                    "lr": settings["lr"],
                    "base_lr": settings["lr"],
                    "warmup_steps": settings["warmup_steps"],
                    **extra,
                    **schedule,
                }
            )
        if not groups:
            raise ValueError("No trainable parameters matched unified_v4 policy groups.")
        if seen != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError("Optimizer groups do not cover all trainable parameters.")
        return groups

    entries = {name: [] for name in ("encoder", "decoder", "alignment")}
    # named_parameters deduplicates shared Parameter objects, including aliases.
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("A trainable Parameter was enumerated twice.")
        seen.add(id(parameter))
        if name.startswith("encoder_projection.") and policy["train_alignment"]:
            group = "alignment"
        elif name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            group = "encoder"
        elif name.startswith("decoder.") and name.endswith((".lora_A", ".lora_B")):
            group = "decoder"
        else:
            raise ValueError(f"Unsupported trainable parameter: {name}; configure policy first.")
        entries[group].append((name, parameter))
    groups = []
    for name in ("encoder", "decoder", "alignment"):
        if name not in policy["groups"]:
            if entries[name]:
                raise ValueError("Unexpected alignment parameters.")
            continue
        if not entries[name]:
            raise ValueError(f"Empty trainable group: {name}")
        settings = policy["groups"][name]
        schedule = _build_group_schedule(settings, total_steps)
        groups.append(
            {
                "name": name,
                "params": [p for _, p in entries[name]],
                "param_names": [n for n, _ in entries[name]],
                "lr": settings["lr"],
                "base_lr": settings["lr"],
                "warmup_steps": settings["warmup_steps"],
                **schedule,
            }
        )
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Optimizer groups do not cover all trainable parameters.")
    return groups


def training_policy_table(groups, policy):
    return [
        {
            "group": group["name"],
            "base_lr": group["base_lr"],
            "warmup_steps": group["warmup_steps"],
            "scheduler": group.get("scheduler", "constant"),
            "warmup_ratio": group.get("warmup_ratio", 0.0),
            "min_lr_ratio": group.get("min_lr_ratio", 1.0),
            "parameters": sum(p.numel() for p in group["params"]),
            "dtypes": sorted({str(p.dtype) for p in group["params"]}),
            "words_frozen": policy["freeze_words"],
            "PLE_frozen": policy["freeze_ple"],
            "word_tying": "shared_IT_anchor"
            if policy["family"] == "symmetric"
            and policy["version"] not in {"unified_v2", "unified_v4"}
            else "separate",
            "optimizer": policy.get("optimizer", {}).get("name", "adamw"),
            "alignment_fp32": policy.get("alignment_fp32", False),
        }
        for group in groups
    ]


def optimizer_group_manifest(groups):
    return [
        {
            "name": group["name"],
            "param_names": list(group["param_names"]),
            "base_lr": group["base_lr"],
            "warmup_steps": group["warmup_steps"],
            **(
                {
                    key: group[key]
                    for key in ("scheduler", "total_steps", "warmup_ratio", "min_lr_ratio")
                }
                if "scheduler" in group
                else {}
            ),
        }
        for group in groups
    ]


def update_policy_learning_rates(optimizer, step):
    if step < 0:
        raise ValueError("Optimizer step must be nonnegative.")
    rates = {}
    for group in optimizer.param_groups:
        offset = int(group.get("warmup_offset", 0))
        effective_step = step - offset
        if effective_step < 0:
            fraction = 0.0
        else:
            warmup = group["warmup_steps"]
            fraction = min(1.0, (effective_step + 1) / warmup) if warmup else 1.0
            scheduler = group.get("scheduler", "constant")
            if scheduler == "cosine":
                total = group["total_steps"]
                if total is None:
                    raise ValueError("Cosine schedule needs a resolved training horizon.")
                effective_total = max(1, total - offset)
                if effective_step + 1 > warmup:
                    # First post-warmup update starts decay; final update reaches the floor.
                    progress = min(
                        1.0, (effective_step + 1 - warmup) / max(1, effective_total - warmup)
                    )
                    floor = group["min_lr_ratio"]
                    fraction = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
            elif scheduler == "wsd":
                total = group["total_steps"]
                if total is None:
                    raise ValueError("WSD schedule needs a resolved training horizon.")
                effective_total = max(1, total - offset)
                decay_steps = int(group.get("decay_steps", 0))
                if decay_steps <= 0:
                    decay_fraction = float(group.get("wsd_decay_fraction", 0.20))
                    decay_steps = max(1, math.ceil(effective_total * decay_fraction))
                stable_end = max(warmup, effective_total - decay_steps)
                stable_until = group.get("stable_until_step")
                if stable_until is not None:
                    stable_end = max(stable_end, int(stable_until) - offset)
                stable_end = min(stable_end, effective_total)
                if effective_step + 1 <= warmup:
                    fraction = min(1.0, (effective_step + 1) / warmup) if warmup else 1.0
                elif effective_step + 1 <= stable_end:
                    fraction = 1.0
                else:
                    remaining = max(1, effective_total - stable_end)
                    progress = min(1.0, (effective_step + 1 - stable_end) / remaining)
                    floor = group["min_lr_ratio"]
                    fraction = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        group["lr"] = group["base_lr"] * fraction
        rates[group["name"]] = group["lr"]
    return rates
