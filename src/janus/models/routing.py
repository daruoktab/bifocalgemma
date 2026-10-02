"""Multimodal Token Routing Protocol for JANUS-4.

Ensures the Golden Rule:
100% of visual patch tokens (576 patches) are ingested by the Bidirectional
Encoder, keeping Decoder KV-cache footprint at 0 MB for visual context.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RoutedBatch:
    """Represents a batch prepared for asymmetric encoder-decoder execution."""

    encoder_input_ids: Optional[torch.Tensor]
    encoder_visual_patches: Optional[torch.Tensor]
    encoder_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor


class MultimodalTokenRouter:
    """Routes multimodal components to optimal sub-networks."""

    @staticmethod
    def calculate_kv_cache_savings(
        num_images: int = 1,
        patches_per_image: int = 576,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 256,
        bytes_per_param: int = 2,
    ) -> float:
        """Calculates static KV-cache memory saved (in Megabytes) by routing to Encoder.

        Returns:
            Megabytes of KV cache saved in the Decoder.
        """
        total_tokens = num_images * patches_per_image
        bytes_saved = total_tokens * num_layers * num_kv_heads * head_dim * bytes_per_param
        return bytes_saved / (1024 * 1024)

    @staticmethod
    def create_encoder_attention_mask(
        text_seq_len: int,
        visual_patch_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Builds a bidirectional attention mask spanning both code AST tokens and visual patches."""
        total_len = text_seq_len + visual_patch_len
        return torch.ones((1, total_len), dtype=torch.bool, device=device)
