"""I01: real autoregressive patch generation; not a synthetic score producer."""

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.contracts import RepairExample, RepairPrediction
from bifocalgemma.scaffold import FeatureNotReadyError


def generate_patch(
    spec: ExperimentSpec, example: RepairExample, max_new_tokens: int
) -> RepairPrediction:
    """Future: IT Jinja prefix, EOS [1,106,50], cache, parser and output boundaries.

    Tool-response token 50 is a handoff boundary, not an executable tool result.
    Encoder input contains pre-fix context; target_patch is never consumed here.
    """
    raise FeatureNotReadyError("I01", "Generation/cache and patch extraction are not integrated.")
