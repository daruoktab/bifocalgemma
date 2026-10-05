"""D05: feature extraction is optional until a matched ablation is defined."""

from dataclasses import dataclass

from bifocalgemma.contracts import RepositorySnapshot
from bifocalgemma.scaffold import FeatureNotReadyError


@dataclass(frozen=True)
class TokenASTFeatures:
    node_types: tuple[int, ...]
    depths: tuple[int, ...]


def extract_token_ast_features(
    snapshot: RepositorySnapshot, original_token_offsets: tuple[tuple[int, int], ...]
) -> TokenASTFeatures:
    """Future: Tree-sitter byte ranges -> original tokenizer character offsets.

    Return one feature per encoder token including explicit handling of BOS/pad.
    Existing ASTGraphProjector consumes IDs; it is not an extractor or graph GNN.
    """
    raise FeatureNotReadyError(
        "D05", "Implement language/version-pinned parsing and offset alignment."
    )
