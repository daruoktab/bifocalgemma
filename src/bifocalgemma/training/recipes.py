"""Shared experimental recipes and training-length-relative I/O cadence."""

import copy
import math

from bifocalgemma.config import WARMUP_RATIO
from bifocalgemma.training.policy import default_training_profiles


def unified_recipe(family):
    """One AdamW policy for all matrix models, including paired alignment gains."""
    policy = default_training_profiles()[family]
    policy.update(
        version="unified_v2", train_alignment=family != "decoder_only", alignment_fp32=True
    )
    if family != "decoder_only":
        policy["groups"]["alignment"] = {"lr": 1e-4, "warmup_steps": 0}
    for name, settings in policy["groups"].items():
        settings.update(
            lr=1e-4 if name == "alignment" else 2e-5,
            warmup_steps=0,
            warmup_ratio=WARMUP_RATIO,
            scheduler="cosine",
            min_lr_ratio=0.1,
        )
    policy["optimizer"] = {"name": "adamw", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}
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
