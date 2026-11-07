"""M05: production factory slot. Explicit native tiny implementation is M07/native.py."""

import torch
import torch.nn as nn

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.contracts import ModelOutput, Seq2SeqBatch
from bifocalgemma.scaffold import FeatureNotReadyError


class BifocalGemmaModel(nn.Module):
    def __init__(self, spec: ExperimentSpec):
        super().__init__()
        self.spec = spec

    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Future native Henc [B,Lenc,Denc]; no gold target or dynamic encoder cache."""
        raise FeatureNotReadyError("M05", "Wire the native bidirectional encoder.")

    def decode_logits(
        self,
        input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
        encoder_states: torch.Tensor | None,
        encoder_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Future uncached merged decoder [B,Ldec,V]; incremental cache belongs to M06."""
        raise FeatureNotReadyError("M05", "Wire native merged attention, PLE and LM head.")

    def forward(self, batch: Seq2SeqBatch) -> ModelOutput:  # ty: ignore[invalid-mutable-override]  # nn.Module.forward is annotated Callable.
        """Future: native bidirectional encoder -> adapted H -> merged decoder -> LM head."""
        raise FeatureNotReadyError("M05", "Full Gemma decoder layer wiring is still required.")


def build_pretrained_model(spec: ExperimentSpec) -> BifocalGemmaModel:
    """Future factory. No from_pretrained call or hidden download exists here."""
    raise FeatureNotReadyError(
        "M04", "Implement and verify native initialization before model assembly."
    )
