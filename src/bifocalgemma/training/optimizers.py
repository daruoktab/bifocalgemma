"""V8-inspired LoRA optimizer experiment; not an official OrScale reproduction.

The orthogonal update acts on A/B factors separately, not on their product BA.
All moments and Newton--Schulz work stay FP32. No extra model copy is retained.
"""

import math
from collections.abc import Callable
from typing import overload

import torch


def validate_optimizer_config(config):
    name = config.get("name")
    allowed = {"name", "weight_decay"}
    if name == "adamw":
        allowed |= {"betas", "eps"}
        betas = config.get("betas", [0.9, 0.999])
        if len(betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError("Invalid AdamW betas.")
        if not math.isfinite(config.get("eps", 1e-8)) or config.get("eps", 1e-8) <= 0:
            raise ValueError("Invalid AdamW epsilon.")
    elif name == "grok_orscale_lora_v1":
        allowed |= {
            "grok_decay",
            "grok_amplification",
            "momentum",
            "ns_steps",
            "trust_min",
            "trust_max",
        }
        for key, default in (("grok_decay", 0.98), ("momentum", 0.95)):
            value = config.get(key, default)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"Invalid {key}.")
        amp = config.get("grok_amplification", 0.5)
        if not math.isfinite(amp) or amp < 0:
            raise ValueError("Invalid Grokfast amplification.")
        if type(config.get("ns_steps", 5)) is not int or not 1 <= config.get("ns_steps", 5) <= 20:
            raise ValueError("Invalid Newton--Schulz iteration count.")
        lower, upper = config.get("trust_min", 0.5), config.get("trust_max", 2.0)
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= 1 <= upper):
            raise ValueError("Trust bounds must contain one.")
    else:
        raise ValueError("Unsupported optimizer recipe.")
    decay = config.get("weight_decay", 0.0)
    if not math.isfinite(decay) or decay < 0 or set(config) - allowed:
        raise ValueError("Invalid optimizer weight decay or unknown option.")


def orthogonal_direction(matrix, steps):
    """FP32 polynomial polar approximation, forming the smaller Gram matrix."""
    x = matrix.float()
    x = x / x.norm().clamp_min(1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.775 * gram + 2.0315 * (gram @ gram)) @ x
    return x.T if transposed else x


class GrokOrScaleLoRA(torch.optim.Optimizer):
    """Matrix-only Grokfast + calibrated OrScale-LM-style update.

    Zero-initialized LoRA B bootstraps with trust=1. Calibration waits until
    both the parameter and update have nonzero norm. No dormant 1-D branch.
    """

    def __init__(self, groups, config):
        validate_optimizer_config(config)
        if any(p.ndim != 2 or p.dtype != torch.float32 for g in groups for p in g["params"]):
            raise ValueError("GrokOrScaleLoRA requires FP32 matrix trainables.")
        defaults = dict(
            weight_decay=config.get("weight_decay", 0.0),
            grok_decay=config.get("grok_decay", 0.98),
            grok_amplification=config.get("grok_amplification", 0.5),
            momentum=config.get("momentum", 0.95),
            ns_steps=config.get("ns_steps", 5),
            trust_min=config.get("trust_min", 0.5),
            trust_max=config.get("trust_max", 2.0),
        )
        super().__init__(groups, defaults)
        self.last_diagnostics = {}

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        diagnostics = {}
        for group in self.param_groups:
            ratios, update_sq, parameter_sq, bootstraps = [], 0.0, 0.0, 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or not bool(torch.isfinite(p.grad).all()):
                    raise RuntimeError("Optimizer requires finite dense gradients.")
                state = self.state[p]
                if not state:
                    state.update(
                        slow=torch.zeros_like(p),
                        momentum_buffer=torch.zeros_like(p),
                        calibration=None,
                    )
                state["slow"].mul_(group["grok_decay"]).add_(p.grad, alpha=1 - group["grok_decay"])
                filtered = p.grad + group["grok_amplification"] * state["slow"]
                # The trainer clips the raw global gradient. The EMA-filtered
                # direction gets the same unit per-matrix guard used by V8.
                filtered = filtered / filtered.norm().clamp_min(1.0)
                state["momentum_buffer"].mul_(group["momentum"]).add_(filtered)
                lookahead = filtered + group["momentum"] * state["momentum_buffer"]
                q = orthogonal_direction(lookahead, group["ns_steps"])
                direction = group["weight_decay"] * p + 0.2 * math.sqrt(max(p.shape)) * q
                weight_norm, direction_norm = float(p.norm()), float(direction.norm())
                if state["calibration"] is None and weight_norm > 0 and direction_norm > 0:
                    state["calibration"] = weight_norm / direction_norm
                if state["calibration"] is None or direction_norm == 0:
                    ratio = 1.0
                    bootstraps += 1
                else:
                    ratio = min(
                        group["trust_max"],
                        max(
                            group["trust_min"],
                            weight_norm / max(state["calibration"] * direction_norm, 1e-30),
                        ),
                    )
                if not bool(torch.isfinite(direction).all()) or not math.isfinite(ratio):
                    raise RuntimeError("Nonfinite orthogonal update.")
                update_norm = group["lr"] * ratio * direction_norm
                update_sq += update_norm**2
                parameter_sq += weight_norm**2
                ratios.append(ratio)
                p.add_(direction, alpha=-group["lr"] * ratio)
            diagnostics[group["name"]] = {
                "trust_min": min(ratios) if ratios else None,
                "trust_max": max(ratios) if ratios else None,
                "trust_mean": sum(ratios) / len(ratios) if ratios else None,
                "update_norm": math.sqrt(update_sq),
                "parameter_norm_before": math.sqrt(parameter_sq),
                "bootstrap_matrices": bootstraps,
            }
        self.last_diagnostics = diagnostics
        return loss


def build_policy_optimizer(groups, policy):
    config = policy.get("optimizer", {"name": "adamw"})
    validate_optimizer_config(config)
    if config["name"] == "adamw":
        return torch.optim.AdamW(
            groups,
            weight_decay=config.get("weight_decay", 0.0),
            betas=tuple(config.get("betas", [0.9, 0.999])),
            eps=config.get("eps", 1e-8),
            foreach=False,
        )
    return GrokOrScaleLoRA(groups, config)
