"""Explicit family policy and deterministic optimizer groups for notebook SFT."""

import copy
import math

import torch.nn as nn

from bifocalgemma.training.memory_probe import ProbeLoRALinear, configure_probe_training


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
    return profiles


def validate_training_policy(policy, model=None):
    family = policy.get("family")
    if family not in {"symmetric", "asymmetric"}:
        raise ValueError("Unknown training-policy family.")
    if policy.get("version") != "attention_lora_policy_v1" or policy.get("phase") != "joint":
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
    expected = {"encoder", "decoder"} | ({"alignment"} if family == "asymmetric" else set())
    if set(policy.get("groups", {})) != expected or policy.get("train_alignment") != (
        family == "asymmetric"
    ):
        raise ValueError("Policy groups/alignment do not match family.")
    for name, settings in policy["groups"].items():
        lr, warmup = settings.get("lr"), settings.get("warmup_steps")
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Invalid LR for {name}.")
        if type(warmup) is not int or warmup < 0:
            raise ValueError(f"Invalid warmup for {name}.")
    if model is not None:
        if getattr(model, "encoder", None) is None or model.spec.tie_encoder_decoder_words != (
            family == "symmetric"
        ):
            raise ValueError("Model architecture does not match policy family.")
        projection = model.encoder_projection
        if family == "symmetric" and not isinstance(projection, nn.Identity):
            raise ValueError("Symmetric policy expects Identity alignment.")
        if family == "asymmetric" and not isinstance(projection, nn.Linear):
            raise ValueError("Asymmetric policy expects a learned Linear alignment.")


def configure_policy_model(model, policy):
    validate_training_policy(policy, model)
    scope = configure_probe_training(model, policy="lora", rank=policy["rank"])
    for module in model.modules():
        if isinstance(module, ProbeLoRALinear):
            module.scaling = policy["alpha"] / policy["rank"]
    return scope


def build_training_groups(model, policy):
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
        groups.append(
            {
                "name": name,
                "params": [p for _, p in entries[name]],
                "param_names": [n for n, _ in entries[name]],
                "lr": settings["lr"],
                "base_lr": settings["lr"],
                "warmup_steps": settings["warmup_steps"],
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
            "parameters": sum(p.numel() for p in group["params"]),
            "dtypes": sorted({str(p.dtype) for p in group["params"]}),
            "words_frozen": policy["freeze_words"],
            "PLE_frozen": policy["freeze_ple"],
            "word_tying": "shared_IT_anchor" if policy["family"] == "symmetric" else "separate",
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
        group["lr"] = group["base_lr"] * fraction
        rates[group["name"]] = group["lr"]
    return rates
