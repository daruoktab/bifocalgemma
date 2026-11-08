"""SwiGLU dual-gate encoder-to-decoder bridge."""

from __future__ import annotations

import math

import torch
import torch.utils.checkpoint
from torch.nn import functional as F


class SwiGLUDualGateBridge(torch.nn.Module):
    """Non-linear manifold warping bridge for BifocalGemma V3.

    Maps bidirectional encoder representations H_enc in R^{d_enc} into the
    causal decoder manifold Z_enc in R^{d_dec} with Pre/Post RMSNorm and
    explicit per-channel scale calibration.
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
        self.d_enc = int(d_enc)
        self.d_dec = int(d_dec)
        self.d_mid = int(d_mid) if d_mid is not None else int(max(d_enc, d_dec))

        self.pre_norm = torch.nn.RMSNorm(self.d_enc, eps=eps)
        self.w_gate = torch.nn.Linear(self.d_enc, self.d_mid, bias=False)
        self.w_up = torch.nn.Linear(self.d_enc, self.d_mid, bias=False)
        self.w_down = torch.nn.Linear(self.d_mid, self.d_dec, bias=False)

        if self.d_enc != self.d_dec:
            self.w_res: torch.nn.Module = torch.nn.Linear(self.d_enc, self.d_dec, bias=False)
        else:
            self.w_res = torch.nn.Identity()

        self.post_norm = torch.nn.RMSNorm(self.d_dec, eps=eps)
        self.gamma_calib = torch.nn.Parameter(
            torch.full((self.d_dec,), float(init_rms_scale), dtype=torch.float32)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            if isinstance(self.w_res, torch.nn.Linear):
                torch.nn.init.orthogonal_(self.w_res.weight)
            torch.nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
            torch.nn.init.xavier_uniform_(self.w_up.weight, gain=0.5)
            torch.nn.init.normal_(self.w_down.weight, mean=0.0, std=0.01 / math.sqrt(self.d_mid))

    def forward(self, h_enc: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
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
        if isinstance(self.w_res, torch.nn.Linear):
            res = F.linear(h_tilde, self.w_res.weight.to(compute_dtype))
        else:
            res = h_tilde
        z = res + warped
        normed = self.post_norm(z.to(self.post_norm.weight.dtype)).to(orig_dtype)
        return normed * self.gamma_calib.to(orig_dtype)
