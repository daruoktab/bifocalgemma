"""FP32 linear alignment followed by RMSNorm for all paired v2 models."""

import torch
from torch import nn
from torch.nn import functional as F


class NormalizedAlignment(nn.Module):
    def __init__(self, input_width: int, output_width: int, seed: int = 17):
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

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Device conversions must never downcast trainable alignment or its gain.
        self.linear.float()
        self.norm.float()
        return self

    def forward(self, hidden):  # nn.Module.forward is a writable Callable.
        projected = F.linear(hidden.float(), self.linear.weight)
        normalized = projected * torch.rsqrt(projected.square().mean(-1, keepdim=True) + 1e-6)
        return (normalized * self.norm.weight).to(hidden.dtype)
