"""BifocalGemma architectural modules, currently tested with synthetic tensors."""

from bifocalgemma.models.bridge import GatedCrossAttentionBridge
from bifocalgemma.models.code_graph_projector import ASTGraphProjector
from bifocalgemma.models.embeddings import apply_word_embedding_policy
from bifocalgemma.models.merged_attention import ReferenceMergedAttention
from bifocalgemma.models.native import build_native_tiny_model
from bifocalgemma.models.routing import RepoContextRouter, RoutedRepoBatch

__all__ = [
    "build_native_tiny_model",
    "GatedCrossAttentionBridge",
    "ReferenceMergedAttention",
    "apply_word_embedding_policy",
    "ASTGraphProjector",
    "RepoContextRouter",
    "RoutedRepoBatch",
]
