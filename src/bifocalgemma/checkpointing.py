"""S01: shared parameter serialization and full initialization provenance."""

from pathlib import Path

from bifocalgemma.models.model import BifocalGemmaModel
from bifocalgemma.scaffold import FeatureNotReadyError


def save_checkpoint(model: BifocalGemmaModel, output_dir: Path) -> None:
    """Future: config/source revisions/tokenizer hashes, adapters and alias metadata."""
    raise FeatureNotReadyError("S01", "Implement alias-safe state saving and deterministic resume.")


def restore_checkpoint(checkpoint_dir: Path) -> BifocalGemmaModel:
    """Future: rebuild architecture and ties BEFORE optimizer/state restoration."""
    raise FeatureNotReadyError(
        "S01", "Validate shape, revision, word aliases and untied PLE on reload."
    )
