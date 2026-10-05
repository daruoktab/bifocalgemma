"""Optional separated-attention reference for ablations, not the default architecture.

The direct-SFT target uses merged attention from initialization (see docs/ARCHITECTURE.md).

Initializes g = atanh(0.05), scaling the cross-attention residual by 0.05.
The input residual is unchanged; this coefficient is not a percentage of preserved
decoder behavior or encoder gradients. Real Gemma integration is not implemented.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization used by this bridge; not a verified Gemma layer replacement."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class GatedCrossAttentionBridge(nn.Module):
    """Gated Cross-Attention Bridge connecting Bidirectional Encoder and Causal Decoder.

    Args:
        decoder_dim: Hidden dimension of the causal decoder (e.g., Gemma 4 E4B).
        encoder_dim: Hidden dimension of the proposed encoder (e.g., Gemma 4 E2B).
        num_heads: Number of attention heads.
        head_dim: Dimension per head. If None, defaults to decoder_dim // num_heads.
        dropout: Attention dropout probability.
        initial_gate_val: Initial tanh gate factor. Default is 0.05 (atanh(0.05) ~ 0.05004),
            scaling the added residual and allowing gradients into nonzero bridge projections.
            This does not guarantee a fixed percentage of preserved behavior or gradients.
        use_qk_norm: Whether to apply RMSNorm on Q and K before dot-product.
    """

    def __init__(
        self,
        decoder_dim: int,
        encoder_dim: int,
        num_heads: int = 16,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        initial_gate_val: float = 0.05,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else decoder_dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_qk_norm = use_qk_norm

        # Projections
        self.q_proj = nn.Linear(decoder_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(encoder_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(encoder_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, decoder_dim, bias=False)

        # QK-Norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        # Gate parameter: initialized such that tanh(gate) = initial_gate_val
        # atanh(x) = 0.5 * ln((1 + x) / (1 - x))
        init_g = 0.5 * math.log((1.0 + initial_gate_val) / (1.0 - initial_gate_val))
        self.gate = nn.Parameter(torch.tensor([init_g], dtype=torch.float32))

    def forward(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for cross-attention bridge.

        Args:
            decoder_states: Tensor of shape (batch, dec_len, decoder_dim).
            encoder_states: Tensor of shape (batch, enc_len, encoder_dim).
            encoder_mask: Optional boolean or additive mask of shape (batch, 1, 1, enc_len)
                or (batch, enc_len) where True/1 indicates valid tokens.

        Returns:
            Gated updated decoder states: shape (batch, dec_len, decoder_dim).
        """
        b, dec_len, _ = decoder_states.shape
        _, enc_len, _ = encoder_states.shape

        q = self.q_proj(decoder_states)
        k = self.k_proj(encoder_states)
        v = self.v_proj(encoder_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(b, dec_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, enc_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if encoder_mask is not None:
            if encoder_mask.dim() == 2:
                # Shape (batch, enc_len) -> (batch, 1, 1, enc_len)
                encoder_mask = encoder_mask[:, None, None, :]
            if encoder_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~encoder_mask, float("-inf"))
            else:
                attn_scores = attn_scores + encoder_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: (batch, num_heads, dec_len, head_dim)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back to (batch, dec_len, inner_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, dec_len, self.inner_dim)
        projected_out = self.out_proj(attn_out)

        # Gated residual connection: z = X + tanh(gate) * CrossAttn(X, H)
        gate_coeff = torch.tanh(self.gate).to(decoder_states.dtype)
        return decoder_states + gate_coeff * projected_out
