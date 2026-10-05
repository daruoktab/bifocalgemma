"""BifocalGemma architectural modules, currently tested with synthetic tensors."""

from bifocalgemma.models.bridge import GatedCrossAttentionBridge
from bifocalgemma.models.code_graph_projector import ASTGraphProjector
from bifocalgemma.models.embeddings import apply_word_embedding_policy
from bifocalgemma.models.merged_attention import ReferenceMergedAttention
from bifocalgemma.models.routing import RepoContextRouter, RoutedRepoBatch

__all__ = [
    "GatedCrossAttentionBridge",
    "ReferenceMergedAttention",
    "apply_word_embedding_policy",
    "ASTGraphProjector",
    "RepoContextRouter",
    "RoutedRepoBatch",
]
