"""M04: checkpoint provenance and future native weight transfer."""

from dataclasses import dataclass

import torch.nn as nn

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.scaffold import FeatureNotReadyError


@dataclass(frozen=True)
class InitializationPlan:
    experiment: str
    encoder_body_source: str | None
    encoder_words_source: str | None
    encoder_ple_source: str | None
    decoder_source: str
    shared_word_anchor: str | None
    encoder_projection: str
    attention_target: str


@dataclass
class InitializedComponents:
    """Future native modules plus explicit provenance, consumed by factory M05."""

    encoder: nn.Module | None
    decoder: nn.Module
    encoder_projection: nn.Module | None
    provenance: InitializationPlan


def describe_initialization(spec: ExperimentSpec) -> InitializationPlan:
    encoder_source = spec.encoder.model_id if spec.encoder else None
    decoder_source = spec.decoder.model_id
    asymmetric = spec.encoder is not None and spec.encoder.size != spec.decoder.size
    return InitializationPlan(
        spec.name,
        encoder_source,
        decoder_source if spec.tie_encoder_decoder_words else encoder_source,
        encoder_source,
        decoder_source,
        decoder_source if spec.tie_encoder_decoder_words else None,
        "not_applicable"
        if spec.encoder is None
        else (
            "normalized_linear_rmsnorm"
            if spec.architecture_version == "normalized_merged_v2"
            else "new_learned_projection"
            if asymmetric
            else "identity"
        ),
        spec.attention_mode,
    )


def initialize_pretrained_components(spec: ExperimentSpec) -> InitializedComponents:
    """Future: explicit local/opt-in weight loading, native Q/K/V and alias mapping.

    Preserve Gemma residual/norm/RoPE/PLE/KV-sharing contracts. Transfer decoder IT
    projections into the final merged architecture directly; no teacher conversion.
    Record all newly initialized parameters and shared word provenance.
    """
    raise FeatureNotReadyError("M04", "Native checkpoint transfer has not been verified.")
