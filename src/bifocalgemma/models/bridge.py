"""SwiGLU dual-gate encoder-to-decoder bridge and normalized linear alignment."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F


class NormalizedAlignment(nn.Module):
    """FP32 linear alignment followed by RMSNorm for normalized merged models."""

    def __init__(self, input_width: int, output_width: int, seed: int = 17) -> None:
        super().__init__()
        self.linear = nn.Linear(input_width, output_width, bias=False, dtype=torch.float32)
        self.norm = nn.RMSNorm(output_width, eps=1e-6, dtype=torch.float32)
        if not self.linear.weight.is_meta:
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                # Initialize on CPU to make initialization independent of GPU selection.
                generator = torch.Generator(device="cpu").manual_seed(seed)
                weight = torch.empty(output_width, input_width, device="cpu", dtype=torch.float32)
                if input_width == output_width:
                    nn.init.eye_(weight)
                else:
                    nn.init.orthogonal_(weight, generator=generator)
                with torch.no_grad():
                    self.linear.weight.copy_(weight)
                    self.norm.weight.fill_(1)

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> NormalizedAlignment:
        super()._apply(fn, recurse=recurse)
        # Device conversions must never downcast trainable alignment or its gain.
        self.linear.float()
        self.norm.float()
        return self

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # nn.Module.forward is a writable Callable.
        projected = F.linear(hidden.float(), self.linear.weight)
        normalized = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * self.norm.weight).to(hidden.dtype)


class SwiGLUDualGateBridge(torch.nn.Module):
    """Non-linear manifold warping bridge for BifocalGemma.

    Maps bidirectional encoder representations H_enc in R^{d_enc} into the
    causal decoder manifold Z_enc in R^{d_dec} with Pre/Post RMSNorm, a trainable
    linear residual path, and explicit per-channel scale calibration.
    """

    def __init__(
        self,
        d_enc: int,
        d_dec: int,
        d_mid: int | None = None,
        init_rms_scale: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_enc = d_enc
        self.d_dec = d_dec
        self.d_mid = d_mid if d_mid is not None else max(d_enc, d_dec)

        self.pre_norm = torch.nn.RMSNorm(self.d_enc, eps=eps)
        self.w_gate = torch.nn.Linear(self.d_enc, self.d_mid, bias=False)
        self.w_up = torch.nn.Linear(self.d_enc, self.d_mid, bias=False)
        self.w_down = torch.nn.Linear(self.d_mid, self.d_dec, bias=False)
        self.w_res = torch.nn.Linear(self.d_enc, self.d_dec, bias=False)

        self.post_norm = torch.nn.RMSNorm(self.d_dec, eps=eps)
        self.gamma_calib = torch.nn.Parameter(
            torch.full((self.d_dec,), float(init_rms_scale), dtype=torch.float32)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.w_res.weight.is_meta:
            return
        with torch.no_grad():
            if self.d_enc == self.d_dec:
                torch.nn.init.eye_(self.w_res.weight)
            else:
                torch.nn.init.orthogonal_(self.w_res.weight)
            torch.nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
            torch.nn.init.xavier_uniform_(self.w_up.weight, gain=0.5)
            torch.nn.init.normal_(self.w_down.weight, mean=0.0, std=0.05 / math.sqrt(self.d_mid))

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> SwiGLUDualGateBridge:
        super()._apply(fn, recurse=recurse)
        for p in self.parameters(recurse=recurse):
            if p.dtype in (torch.bfloat16, torch.float16):
                p.data = p.data.float()
                if p.grad is not None and p.grad.dtype in (torch.bfloat16, torch.float16):
                    p.grad.data = p.grad.data.float()
        return self

    def forward(self, h_enc: torch.Tensor) -> torch.Tensor:
        orig_dtype = h_enc.dtype
        compute_dtype = (
            torch.bfloat16
            if (h_enc.device.type == "cuda" or orig_dtype == torch.bfloat16)
            else orig_dtype
        )
        x = h_enc.to(compute_dtype)
        h_tilde = self.pre_norm(x.to(self.pre_norm.weight.dtype)).to(compute_dtype)
        gated = F.silu(F.linear(h_tilde, self.w_gate.weight.to(compute_dtype))) * F.linear(
            h_tilde, self.w_up.weight.to(compute_dtype)
        )
        warped = F.linear(gated, self.w_down.weight.to(compute_dtype))
        res = F.linear(h_tilde, self.w_res.weight.to(compute_dtype))
        z = res + warped
        normed = self.post_norm(z.to(self.post_norm.weight.dtype)).to(orig_dtype)
        return normed * self.gamma_calib.to(orig_dtype)


__all__ = ["NormalizedAlignment", "SwiGLUDualGateBridge"]
