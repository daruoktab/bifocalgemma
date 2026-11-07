"""BifocalGemma: Decoupled cross-attention and SwiGLU manifold warping for long-context code intelligence."""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

_EXPORTS = {
    "BifocalGemmaConfig": "bifocalgemma.configuration_bifocalgemma",
    "BifocalGemmaForCausalLM": "bifocalgemma.modeling_bifocalgemma",
    "BifocalGemmaModel": "bifocalgemma.modeling_bifocalgemma",
    "BifocalGemmaPreTrainedModel": "bifocalgemma.modeling_bifocalgemma",
    "export_matrix_pretrained_model": "bifocalgemma.modeling_bifocalgemma",
}

if TYPE_CHECKING:
    from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
    from bifocalgemma.modeling_bifocalgemma import (
        BifocalGemmaForCausalLM,
        BifocalGemmaModel,
        BifocalGemmaPreTrainedModel,
        export_matrix_pretrained_model,
    )

__all__ = [
    "BifocalGemmaConfig",
    "BifocalGemmaForCausalLM",
    "BifocalGemmaModel",
    "BifocalGemmaPreTrainedModel",
    "export_matrix_pretrained_model",
]


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name = _EXPORTS[name]
        module = __import__(module_name, fromlist=[name])
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
