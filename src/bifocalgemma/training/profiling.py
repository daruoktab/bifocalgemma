"""T03: measured hardware profile. No VRAM estimate is treated as a measured result."""

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.scaffold import FeatureNotReadyError


def profile_experiment(spec: ExperimentSpec, context_tokens: int, target_tokens: int) -> dict:
    """Future: weights/optimizer/activations/self-KV/cross-KV, cold/warm latency.

    Report hardware, precision, token lengths and allocator peaks separately from
    effective parameters. Two GPU capacities are not assumed to be pooled.
    """
    raise FeatureNotReadyError("T03", "Native model integration is required for actual profiling.")
