"""Repository Context & Long-Context Token Routing Protocol for JANUS-4.

Ensures the Golden Rule for Long-Context Software Engineering:
100% of the long repository context (file trees, dependency graphs, AST tokens,
and issue logs) is ingested by the Bidirectional Encoder, eliminating 90%+ of the
autoregressive KV-cache footprint in the Causal Decoder during patch generation.
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
    """Routes long-context code repository tokens to optimal sub-networks."""

    @staticmethod
    def calculate_long_context_kv_savings(
        prompt_len: int = 16384,
        gen_len: int = 512,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 256,
        bytes_per_param: int = 2,
    ) -> Dict[str, float]:
        """Calculates dynamic autoregressive KV-cache memory saved by JANUS-4 vs Decoder-Only.

        Args:
            prompt_len: Context sequence length (e.g. 16k or 32k repository tokens).
            gen_len: Length of generated code diff / patch (e.g. 512 tokens).
            num_layers: Number of transformer decoder layers.
            num_kv_heads: Number of key-value heads.
            head_dim: Hidden dimension per head.
            bytes_per_param: Bytes per weight element (2 for BF16/FP16).

        Returns:
            Dictionary with memory usage (MB) and percentage savings.
        """
        bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_param

        # Decoder-Only must carry entire (prompt + generated) tokens in dynamic KV cache
        decoder_only_bytes = (prompt_len + gen_len) * num_layers * bytes_per_token_per_layer
        decoder_only_mb = decoder_only_bytes / (1024 * 1024)

        # In JANUS-4, the Decoder KV cache ONLY retains generated patch tokens
        janus_dec_bytes = gen_len * num_layers * bytes_per_token_per_layer
        janus_dec_mb = janus_dec_bytes / (1024 * 1024)

        saved_mb = decoder_only_mb - janus_dec_mb
        reduction_pct = (saved_mb / decoder_only_mb) * 100.0

        return {
            "decoder_only_kv_mb": decoder_only_mb,
            "janus_decoder_kv_mb": janus_dec_mb,
            "saved_mb": saved_mb,
            "reduction_percentage": reduction_pct,
        }

    @staticmethod
    def create_bidirectional_repo_mask(
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Builds a full bidirectional attention mask for repository context."""
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
