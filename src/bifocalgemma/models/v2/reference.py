"""M03: tiny random-weight assembly for testing pieces, not a Gemma backbone."""

import torch
import torch.nn as nn

from bifocalgemma.config import PROFILES, ExperimentSpec
from bifocalgemma.contracts import ModelOutput, Seq2SeqBatch, TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.v2.embeddings import apply_word_embedding_policy
from bifocalgemma.models.v2.merged_attention import ReferenceMergedAttention
from bifocalgemma.training.objectives import patch_cross_entropy


class TinyReferenceModel(nn.Module):
    def __init__(self, spec: ExperimentSpec, vocab_size: int = 128):
        super().__init__()
        if spec.encoder is None:
            raise ValueError("Tiny assembly supports the four encoder-decoder variants.")
        self.spec = spec
        widths = {"E2B": 32, "E4B": 48}
        encoder_dim = widths[spec.encoder.size]
        decoder_dim = widths[spec.decoder.size]
        self.encoder_words = nn.Embedding(vocab_size, encoder_dim, padding_idx=0)
        self.decoder_words = nn.Embedding(vocab_size, decoder_dim, padding_idx=0)
        self.lm_head = nn.Linear(decoder_dim, vocab_size, bias=False)
        self.embedding_policy = apply_word_embedding_policy(
            self.encoder_words,
            self.decoder_words,
            self.lm_head,
            tie_encoder_decoder=spec.tie_encoder_decoder_words,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            encoder_dim, 4, dim_feedforward=4 * encoder_dim, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 1, enable_nested_tensor=False)
        self.merged_attention = ReferenceMergedAttention(
            encoder_dim,
            decoder_dim,
            num_heads=8,
            num_kv_heads=PROFILES[spec.decoder.size].kv_heads,
            head_dim=8,
        )
        self.output_norm = nn.RMSNorm(decoder_dim)

    def forward(self, batch: Seq2SeqBatch) -> ModelOutput:  # nn.Module.forward is annotated Callable.
        encoded = self.encoder(
            self.encoder_words(batch.encoder_input_ids),
            src_key_padding_mask=~batch.encoder_attention_mask,
        )
        decoder_states = self.decoder_words(batch.decoder_input_ids)
        attended = self.merged_attention(
            decoder_states,
            encoded,
            batch.encoder_attention_mask,
            batch.decoder_attention_mask,
        )
        logits = self.lm_head(self.output_norm(decoder_states + attended))
        return ModelOutput(logits, patch_cross_entropy(logits, batch.labels))


def run_reference_smoke(spec: ExperimentSpec, device: str = "cpu") -> dict:
    """One synthetic forward/backward, with no pretrained files, data or scores."""
    torch.manual_seed(17)
    model = TinyReferenceModel(spec).to(device)
    examples = [
        TokenizedRepairExample("synthetic-1", (2, 12, 13, 14), (2, 105, 7), (18, 19, 106)),
        TokenizedRepairExample("synthetic-2", (2, 15, 16), (2, 105, 8), (20, 106)),
    ]
    batch = RepairCollator()(examples).to(device)
    output = model(batch)
    assert output.loss is not None
    output.loss.backward()
    parameters = list(model.parameters())
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not bool(torch.isfinite(output.loss)) or not all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    ):
        raise RuntimeError("Reference computation produced nonfinite loss/gradients.")
    return {
        "experiment": spec.name,
        "mode": "synthetic_reference",
        "device": device,
        "logits_shape": list(output.logits.shape),
        "synthetic_loss": output.loss.item(),
        "word_embeddings_shared": model.encoder_words.weight is model.decoder_words.weight,
        "decoder_lm_head_shared": model.lm_head.weight is model.decoder_words.weight,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "pretrained_weights_loaded": False,
        "benchmark_executed": False,
    }
