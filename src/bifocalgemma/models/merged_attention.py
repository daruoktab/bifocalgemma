"""M01: numerical merged-attention reference, not a Gemma layer replacement.

Joint softmax and shared K/V projections are implemented. Gemma RoPE, PLE,
native KV-sharing/caches and checkpoint normalization mapping remain M04/M05/M06.
See docs/ARCHITECTURE.md for the exact production target and limitations.
"""

import torch
import torch.nn as nn


def build_joint_visibility(
    encoder_valid: torch.Tensor,
    decoder_valid: torch.Tensor,
    self_window: int | None = None,
) -> torch.Tensor:
    """Keys are [decoder, encoder]; decoder causal/windowed, encoder globally visible."""
    if encoder_valid.ndim != 2 or decoder_valid.ndim != 2:
        raise ValueError("Validity masks must have [batch, sequence] shape.")
    if encoder_valid.shape[0] != decoder_valid.shape[0]:
        raise ValueError("Encoder and decoder batch sizes differ.")
    if self_window is not None and self_window <= 0:
        raise ValueError("Self-attention window must be positive.")
    length = decoder_valid.shape[1]
    positions = torch.arange(length, device=decoder_valid.device)
    visibility = positions[None, :] <= positions[:, None]
    if self_window is not None:
        visibility &= positions[None, :] > positions[:, None] - self_window
    self_visible = visibility[None] & decoder_valid.bool()[:, None, :]
    cross_visible = encoder_valid.bool()[:, None, :].expand(-1, length, -1)
    return (
        torch.cat((self_visible, cross_visible), dim=-1)[:, None]
        & decoder_valid.bool()[:, None, :, None]
    )


class ReferenceMergedAttention(nn.Module):
    """Small reference block with GQA and learned asymmetric hidden projection."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 8,
        self_window: int | None = None,
    ):
        super().__init__()
        if min(encoder_dim, decoder_dim, num_heads, num_kv_heads, head_dim) <= 0:
            raise ValueError("Attention dimensions must be positive.")
        if num_heads % num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.self_window = self_window
        self.encoder_projection = (
            nn.Identity()
            if encoder_dim == decoder_dim
            else nn.Linear(encoder_dim, decoder_dim, bias=False)
        )
        self.q_proj = nn.Linear(decoder_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(decoder_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(decoder_dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, decoder_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        # Unit RMS for values, with no trainable scale in this reference.
        self.scale = 1.0

    def forward(  # ty: ignore[invalid-mutable-override]  # nn.Module.forward is annotated Callable.
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_valid: torch.Tensor,
        decoder_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, dec_len, _ = decoder_states.shape
        if (
            encoder_states.shape[0] != batch
            or decoder_states.shape[-1] != self.decoder_dim
            or encoder_states.shape[-1] != self.encoder_dim
        ):
            raise ValueError("State shapes do not match attention dimensions.")
        if tuple(encoder_valid.shape) != tuple(encoder_states.shape[:2]) or tuple(
            decoder_valid.shape
        ) != tuple(decoder_states.shape[:2]):
            raise ValueError("Validity masks must match the supplied state lengths.")
        if dec_len == 0:
            raise ValueError("Decoder sequence must contain at least one token.")
        context = self.encoder_projection(encoder_states)
        all_states = torch.cat((decoder_states, context), dim=1)
        q = (
            self.q_proj(decoder_states)
            .view(batch, dec_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_len = all_states.shape[1]
        k = (
            self.k_proj(all_states)
            .view(batch, key_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(all_states)
            .view(batch, key_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = (v.float() * torch.rsqrt(v.float().square().mean(-1, keepdim=True) + 1e-6)).to(v.dtype)
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        allowed = build_joint_visibility(encoder_valid, decoder_valid, self.self_window)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~allowed, float("-inf"))
        # Avoid NaN forward AND backward for entirely masked padded query rows.
        scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        probabilities = scores.softmax(-1).masked_fill(~allowed, 0).to(v.dtype)
        attended = torch.matmul(probabilities, v)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, dec_len, self.num_heads * self.head_dim)
        )
        return self.out_proj(attended)
