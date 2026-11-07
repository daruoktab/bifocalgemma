"""V2 (legacy) BifocalGemma components: merged attention, linear alignment, router.

Everything in this package belongs to the V2 research line
(`docs/BIFOCALGEMMA_V2.md`). The production training path is V3 and lives in
`bifocalgemma.v3`; V3 only touches this package to build the native Gemma 4
skeleton (`native.NativeTinyModel`) before replacing its attention.

Import `pretrained`, `checkpoint_transfer`, `model`, `cache`, or
`initialization` explicitly; they are intentionally not re-exported here to
keep package import light.
"""

from bifocalgemma.models.v2.alignment import NormalizedAlignment
from bifocalgemma.models.v2.bridge import GatedCrossAttentionBridge
from bifocalgemma.models.v2.code_graph_projector import ASTGraphProjector
from bifocalgemma.models.v2.merged_attention import (
    ReferenceMergedAttention,
    build_joint_visibility,
)
from bifocalgemma.models.v2.native import (
    NativeMergedAttention,
    NativeTinyModel,
    build_native_model,
    build_native_tiny_model,
    native_config,
    run_native_smoke,
)
from bifocalgemma.models.v2.routing import RepoContextRouter, RoutedRepoBatch

__all__ = [
    "ASTGraphProjector",
    "GatedCrossAttentionBridge",
    "NativeMergedAttention",
    "NativeTinyModel",
    "NormalizedAlignment",
    "ReferenceMergedAttention",
    "RepoContextRouter",
    "RoutedRepoBatch",
    "build_joint_visibility",
    "build_native_model",
    "build_native_tiny_model",
    "native_config",
    "run_native_smoke",
]
