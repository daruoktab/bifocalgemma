"""M06: production caches, distinct from the old illustrative self-KV estimator."""

from dataclasses import dataclass

import torch

from bifocalgemma.scaffold import FeatureNotReadyError


@dataclass
class EncoderMemory:
    context_fingerprint: str
    encoder_checkpoint_revision: str
    decoder_checkpoint_revision: str
    hidden_states: torch.Tensor


class RepositoryCache:
    def get_or_encode(self, context_fingerprint: str) -> EncoderMemory:
        """Future inference only: key covers issue/snapshot/tokenizer/weights/AST policy.

        Re-encode the entire context after edits. Training cache across optimizer
        steps and one-hop invalidation are not enabled by this contract.
        """
        raise FeatureNotReadyError("M06", "Implement invalidation and self/cross KV ownership.")
