"""Linear Patch Projection module for Encoder-Free Multimodal Gemma 4.

Projects raw image patches directly into LLM embedding space with 2D coordinate
positional embeddings, replacing heavy vision encoders (SigLIP/ViT).
"""

from typing import Tuple

import torch
import torch.nn as nn


class LinearPatchProjector(nn.Module):
    """Linear Patch Projector for Gemma 4 12B encoder-free vision input.

    Args:
        patch_size: Square patch spatial resolution (default: 48).
        in_channels: Number of image channels (default: 3).
        embed_dim: Output LLM embedding dimension (default: 3840 for Gemma 4 12B).
        max_patches_h: Maximum number of vertical patches (default: 24).
        max_patches_w: Maximum number of horizontal patches (default: 24).
    """

    def __init__(
        self,
        patch_size: int = 48,
        in_channels: int = 3,
        embed_dim: int = 3840,
        max_patches_h: int = 24,
        max_patches_w: int = 24,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size * patch_size * in_channels
        self.embed_dim = embed_dim

        # Direct linear projection (~35M params when patch_dim=6912, embed_dim=3840)
        self.projection = nn.Linear(self.patch_dim, self.embed_dim, bias=False)

        # 2D coordinate positional embeddings
        self.pos_emb_h = nn.Embedding(max_patches_h, self.embed_dim)
        self.pos_emb_w = nn.Embedding(max_patches_w, self.embed_dim)

        # Final normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        grid_shape: Tuple[int, int] = (24, 24),
    ) -> torch.Tensor:
        """Projects patch tensor and adds 2D positional embeddings.

        Args:
            patch_tokens: Tensor of shape (batch, num_patches, patch_dim).
            grid_shape: (H_patches, W_patches) such that H_patches * W_patches == num_patches.

        Returns:
            Projected patch embeddings of shape (batch, num_patches, embed_dim).
        """
        b, num_patches, _ = patch_tokens.shape
        h_grid, w_grid = grid_shape
        if h_grid * w_grid != num_patches:
            raise ValueError(f"grid_shape {grid_shape} does not match total patches {num_patches}.")

        # Linear projection
        x = self.projection(patch_tokens)

        # 2D Positional embeddings
        device = patch_tokens.device
        y_coords = (
            torch.arange(h_grid, device=device).unsqueeze(1).expand(h_grid, w_grid).reshape(-1)
        )
        x_coords = (
            torch.arange(w_grid, device=device).unsqueeze(0).expand(h_grid, w_grid).reshape(-1)
        )

        pos_2d = self.pos_emb_h(y_coords) + self.pos_emb_w(x_coords)
        x = x + pos_2d.unsqueeze(0)

        return self.layer_norm(x)
