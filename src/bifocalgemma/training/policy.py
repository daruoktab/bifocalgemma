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
    if (
        policy.get("version") not in {"attention_lora_policy_v1", "unified_v2"}
        or policy.get("phase") != "joint"
    ):
        raise ValueError("Only joint attention LoRA policy v1 is supported.")
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
    expected = ({"decoder"} if family == "decoder_only" else {"encoder", "decoder"}) | (
        {"alignment"} if family == "asymmetric" or (v2 and family == "symmetric") else set()
    )
    if set(policy.get("groups", {})) != expected or policy.get("train_alignment") != (
        family == "asymmetric" or (v2 and family == "symmetric")
    ):
        raise ValueError("Policy groups/alignment do not match family.")
    for name, settings in policy["groups"].items():
        lr, warmup = settings.get("lr"), settings.get("warmup_steps")
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Invalid LR for {name}.")
        if type(warmup) is not int or warmup < 0:
            raise ValueError(f"Invalid warmup for {name}.")
        if settings.get("scheduler", "constant") not in {"constant", "cosine"}:
            raise ValueError(f"Invalid scheduler for {name}.")
        for key in ("warmup_ratio", "min_lr_ratio"):
            value = settings.get(key, 0.0)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f"Invalid {key} for {name}.")
    if model is not None:
        if v2 and model.spec.architecture_version != "normalized_merged_v2":
            raise ValueError(
                "unified_v2 requires a v2 model; legacy checkpoints cannot resume this architecture."
            )
        if family == "decoder_only":
            if model.encoder is not None or model.encoder_projection is not None:
                raise ValueError("Decoder-only policy requires a baseline model.")
            return
        if getattr(model, "encoder", None) is None or model.spec.tie_encoder_decoder_words != (
            family == "symmetric" and not v2
        ):
            raise ValueError("Model architecture does not match policy family.")
        projection = model.encoder_projection
        if v2:
            from bifocalgemma.models.v2.alignment import NormalizedAlignment

            if model.spec.architecture_version != "normalized_merged_v2" or not isinstance(
                projection, NormalizedAlignment
            ):
                raise ValueError("unified_v2 requires normalized_merged_v2 alignment.")
            return
        if family == "symmetric" and not isinstance(projection, nn.Identity):
            raise ValueError("Symmetric policy expects Identity alignment.")
        if family == "asymmetric" and not isinstance(projection, nn.Linear):
            raise ValueError("Asymmetric policy expects a learned Linear alignment.")


def configure_policy_model(model, policy):
    validate_training_policy(policy, model)
    scope = configure_probe_training(model, policy="lora", rank=policy["rank"])
    if policy["version"] == "unified_v2" and model.encoder_projection is not None:
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


def build_training_groups(model, policy, total_steps=None) -> "list[dict[str, Any]]":
    validate_training_policy(policy, model)
    entries = {name: [] for name in ("encoder", "decoder", "alignment")}
    seen = set()
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
        schedule = {}
        if settings.get("scheduler", "constant") == "cosine":
            if total_steps is not None and (type(total_steps) is not int or total_steps < 1):
                raise ValueError("Schedule horizon must be a positive integer.")
            schedule = {
                "scheduler": "cosine",
                "total_steps": total_steps,
                "warmup_ratio": settings.get("warmup_ratio", 0.0),
                "min_lr_ratio": settings.get("min_lr_ratio", 0.0),
            }
            if total_steps is not None:
                schedule["warmup_steps"] = min(
                    total_steps,
                    max(
                        settings["warmup_steps"], math.ceil(total_steps * schedule["warmup_ratio"])
                    ),
                )
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
            if policy["family"] == "symmetric" and policy["version"] != "unified_v2"
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
        warmup = group["warmup_steps"]
        fraction = min(1.0, (step + 1) / warmup) if warmup else 1.0
        if group.get("scheduler") == "cosine":
            total = group["total_steps"]
            if total is None:
                raise ValueError("Cosine schedule needs a resolved training horizon.")
            if step + 1 > warmup:
                # First post-warmup update starts decay; final update reaches the floor.
                progress = min(1.0, (step + 1 - warmup) / max(1, total - warmup))
                floor = group["min_lr_ratio"]
                fraction = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        group["lr"] = group["base_lr"] * fraction
        rates[group["name"]] = group["lr"]
    return rates
