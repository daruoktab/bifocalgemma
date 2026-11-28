"""Regression tests for mixed encoder/bypass batches and Stage-1 frozen-decoder safety."""

from __future__ import annotations

import torch

from bifocalgemma.config import make_experiment_spec
from bifocalgemma.data.collator import RepairCollator, TokenizedRepairExample
from bifocalgemma.modeling_bifocalgemma import NativeTinyModel
from bifocalgemma.models.cross_attention import configure_full_rank_cross_attention
from bifocalgemma.training.objectives import compute_step_loss


def _build_tiny_bifocal_model() -> NativeTinyModel:
    torch.manual_seed(7)
    spec = make_experiment_spec("2it-2it")
    model = NativeTinyModel(spec, tiny=True)
    configure_full_rank_cross_attention(model)
    model.eval()
    return model


def test_mixed_encoder_bypass_batch_matches_standalone_logits_and_finite_grads():
    model = _build_tiny_bifocal_model()
    collator = RepairCollator()

    short_ex = TokenizedRepairExample("short", (), (2, 105, 11), (18, 106))
    long_ex = TokenizedRepairExample("long", (2, 12, 13, 14), (2, 105, 11), (18, 106))

    standalone_batch = collator([short_ex])
    mixed_batch = collator([short_ex, long_ex])

    with torch.no_grad():
        standalone_enc = model.encode_context(
            standalone_batch.encoder_input_ids,
            standalone_batch.encoder_attention_mask,
        )
        standalone_hidden = model._decode_hidden(
            standalone_batch.decoder_input_ids,
            standalone_batch.decoder_attention_mask,
            standalone_enc,
            standalone_batch.encoder_attention_mask,
        )
        standalone_logits = model.lm_head(standalone_hidden)

        mixed_enc = model.encode_context(
            mixed_batch.encoder_input_ids,
            mixed_batch.encoder_attention_mask,
        )
        mixed_hidden = model._decode_hidden(
            mixed_batch.decoder_input_ids,
            mixed_batch.decoder_attention_mask,
            mixed_enc,
            mixed_batch.encoder_attention_mask,
        )
        mixed_logits = model.lm_head(mixed_hidden)

    torch.testing.assert_close(mixed_logits[0], standalone_logits[0], rtol=1e-4, atol=1e-4)

    model.train()
    model.zero_grad(set_to_none=True)
    loss = compute_step_loss(model, mixed_batch, lambda_ptr=0.15)
    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(g).all()) for g in grads)


def test_all_bypass_stage1_frozen_decoder_zero_grad_and_encoder_skip():
    model = _build_tiny_bifocal_model()

    # Simulate Stage 1 freeze_decoder=True: freeze non-cross-bridge decoder params
    for name, param in model.named_parameters():
        if name.startswith("decoder.") and "cross" not in name:
            param.requires_grad_(False)

    def _fail_if_encoder_called(*args, **kwargs):
        raise AssertionError("self.encoder should not be invoked when all encoder masks are False")

    assert model.encoder is not None
    setattr(model.encoder, "forward", _fail_if_encoder_called)

    collator = RepairCollator()
    all_bypass_batch = collator(
        [
            TokenizedRepairExample("short_1", (), (2, 105, 11), (18, 106)),
            TokenizedRepairExample("short_2", (), (2, 105, 12), (19, 106)),
        ]
    )

    enc_states = model.encode_context(
        all_bypass_batch.encoder_input_ids,
        all_bypass_batch.encoder_attention_mask,
    )
    assert enc_states.shape == (2, 0, model.decoder.config.hidden_size)
    assert float(enc_states.abs().sum().item()) == 0.0

    model.train()
    model.zero_grad(set_to_none=True)
    loss = compute_step_loss(model, all_bypass_batch, lambda_ptr=0.15)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()

    # In an all-bypass Stage-1 microbatch, no encoder/cross-bridge gradient should be NaN,
    # and the effective gradient norm of trainable parameters is 0.0.
    trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = (
        sum(float(g.detach().pow(2).sum().item()) for g in trainable_grads) ** 0.5
        if trainable_grads
        else 0.0
    )
    assert grad_norm == 0.0
