"""M02: cross-component word tying; PLE stays separate. See docs/ARCHITECTURE.md."""

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class EmbeddingPolicyResult:
    tied_encoder_decoder: bool
    saved_encoder_parameters: int
    shared_source: str


def apply_word_embedding_policy(
    encoder: nn.Embedding,
    decoder: nn.Embedding,
    decoder_lm_head: nn.Linear,
    *,
    tie_encoder_decoder: bool,
) -> EmbeddingPolicyResult:
    """Tie parameter objects before constructing the optimizer.

    Decoder is the anchor. This does not tie PLE or implement checkpoint alias
    restoration. Caller must follow ExperimentSpec's symmetric-only policy.
    """
    if decoder_lm_head.weight.shape != decoder.weight.shape:
        raise ValueError("Decoder LM head must have [vocabulary, decoder width] shape.")
    if decoder_lm_head.bias is not None:
        raise ValueError("Reference LM head must be bias-free.")
    if tie_encoder_decoder:
        if encoder.weight.shape != decoder.weight.shape:
            raise ValueError("Direct word tying requires equal vocabulary and hidden width.")
        if (
            encoder.weight.device != decoder.weight.device
            or encoder.weight.dtype != decoder.weight.dtype
        ):
            raise ValueError("Shared embeddings must use the same device and dtype.")
        saved = encoder.weight.numel() if encoder.weight is not decoder.weight else 0
        encoder.weight = decoder.weight
    else:
        saved = 0
        if encoder.weight is decoder.weight:
            raise ValueError("Untied policy received an already shared encoder/decoder parameter.")
    decoder_lm_head.weight = decoder.weight
    return EmbeddingPolicyResult(tie_encoder_decoder, saved, "decoder_it")
