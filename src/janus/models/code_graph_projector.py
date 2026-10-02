"""AST & Code Structure Projection Module for Long-Context Code Comprehension.

Projects hierarchical code structural features (AST node types, syntactic depth,
and scope coordinates) directly into the LLM embedding space to empower deep
bidirectional repository reasoning.
"""

from typing import Optional

import torch
import torch.nn as nn


class ASTGraphProjector(nn.Module):
    """Projects hierarchical AST syntactic features and code graph coordinates into LLM embedding space.

    Args:
        num_node_types: Number of distinct AST node types (from tree-sitter vocabulary).
        max_depth: Maximum syntactic tree depth to embed.
        embed_dim: Target LLM embedding dimension (e.g., 2048 for E2B, 2560 for E4B).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_node_types: int = 256,
        max_depth: int = 64,
        embed_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_node_types = num_node_types
        self.max_depth = max_depth
        self.embed_dim = embed_dim

        # Categorical AST node type embedding
        self.node_type_embed = nn.Embedding(num_node_types, embed_dim)

        # Hierarchical depth embedding (scope/nesting level)
        self.depth_embed = nn.Embedding(max_depth, embed_dim)

        # Linear projection combining syntax and scope features
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_types: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes structural AST embeddings for repository code tokens.

        Args:
            node_types: Tensor of node type IDs of shape (batch, seq_len).
            depths: Optional tensor of nesting depth IDs of shape (batch, seq_len).

        Returns:
            Projected structural embedding tensor of shape (batch, seq_len, embed_dim).
        """
        features = self.node_type_embed(node_types)

        if depths is not None:
            clamped_depths = torch.clamp(depths, 0, self.max_depth - 1)
            features = features + self.depth_embed(clamped_depths)

        projected = self.proj(features)
        projected = self.layer_norm(projected)
        return self.dropout(projected)
