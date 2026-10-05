"""Context partitioning and an illustrative decoder self-KV estimator for BifocalGemma.

These utilities do not load a Gemma backbone or implement a repository cache.
Self-attention estimates exclude encoder states, cross-attention, and model weights.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class RoutedRepoBatch:
    """Represents a repository-level batch prepared for asymmetric encoder-decoder execution."""

    encoder_input_ids: torch.Tensor
    encoder_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    encoder_ast_types: Optional[torch.Tensor] = None
    encoder_ast_depths: Optional[torch.Tensor] = None


class RepoContextRouter:
    """Partitions supplied token tensors for proposed encoder-decoder execution."""

    @staticmethod
    def calculate_decoder_self_kv_savings(
        prompt_len: int = 16384,
        gen_len: int = 512,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 256,
        bytes_per_param: int = 2,
    ) -> Dict[str, float]:
        """Compare decoder self-attention KV only, assuming uniform full-attention layers.

        Excludes encoder hidden states, cross-attention K/V, weights, activations,
        and allocator overhead. Gemma's hybrid attention and KV sharing are not
        modeled. The defaults are illustrative, not a Gemma configuration.

        Args:
            prompt_len: Context sequence length (e.g. 16k or 32k repository tokens).
            gen_len: Length of generated code diff / patch (e.g. 512 tokens).
            num_layers: Number of transformer decoder layers.
            num_kv_heads: Number of key-value heads.
            head_dim: Hidden dimension per head.
            bytes_per_param: Bytes per KV element (2 for BF16/FP16).

        Returns:
            Self-KV sizes in MiB (keys use the suffix ``_mb``), and the self-KV
            reduction percentage. This is not total VRAM saved. Two empty
            sequences return zero sizes and zero reduction.
        """
        if prompt_len < 0 or gen_len < 0:
            raise ValueError("Sequence lengths must be nonnegative.")
        if min(num_layers, num_kv_heads, head_dim, bytes_per_param) <= 0:
            raise ValueError(
                "Layer count, KV heads, head dimension, and element size must be positive."
            )

        bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_param

        # Uniform full-attention baseline: prompt and generated tokens in self-KV.
        decoder_only_bytes = (prompt_len + gen_len) * num_layers * bytes_per_token_per_layer
        decoder_only_mb = decoder_only_bytes / (1024 * 1024)

        # Proposed split: generated tokens only in decoder self-KV; context is elsewhere.
        bifocalgemma_dec_bytes = gen_len * num_layers * bytes_per_token_per_layer
        bifocalgemma_dec_mb = bifocalgemma_dec_bytes / (1024 * 1024)

        saved_mb = decoder_only_mb - bifocalgemma_dec_mb
        reduction_pct = (saved_mb / decoder_only_mb) * 100.0 if decoder_only_mb else 0.0

        return {
            "decoder_only_self_kv_mb": decoder_only_mb,
            "bifocalgemma_decoder_self_kv_mb": bifocalgemma_dec_mb,
            "self_kv_saved_mb": saved_mb,
            "self_kv_reduction_percentage": reduction_pct,
        }

    @staticmethod
    def create_bidirectional_repo_mask(
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Mark all supplied repository tokens valid, with shape (1, seq_len).

        This is a padding/validity mask, not a pairwise attention mask. Its name
        describes the intended encoder use; it does not disable causal masking
        or sliding windows inside a Gemma backbone.
        """
        return torch.ones((1, seq_len), dtype=torch.bool, device=device)

    @staticmethod
    def partition_context_and_target(
        full_ids: torch.Tensor,
        split_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits full sequence into encoder context and decoder generation target."""
        encoder_ids = full_ids[:, :split_idx]
        decoder_ids = full_ids[:, split_idx:]
        return encoder_ids, decoder_ids
