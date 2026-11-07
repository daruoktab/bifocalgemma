"""Native integration invariants; no network or pretrained weights."""

import copy
import io
import unittest
from typing import cast

import torch
from transformers import Gemma4TextModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.models.v2.native import (
    NativeMergedAttention,
    build_native_tiny_model,
    native_config,
    run_native_smoke,
)


class TestNativeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_six_forward_backward(self):
        for spec in EXPERIMENT_SPECS:
            with self.subTest(spec=spec.name):
                report = run_native_smoke(spec)
                self.assertTrue(report["gradients_finite"])
                self.assertGreater(report["parameters_with_nonzero_gradient"], 0)
                self.assertEqual(report["word_embeddings_shared"], spec.tie_encoder_decoder_words)
                self.assertTrue(report["decoder_lm_head_shared"])
                self.assertFalse(report["pretrained_weights_loaded"])

    def test_no_context_parity_native_decoder_and_long_window(self):
        for spec in EXPERIMENT_SPECS[:2]:
            model = build_native_tiny_model(spec).eval()
            native = Gemma4TextModel(copy.deepcopy(model.decoder.config)).eval()
            native.load_state_dict(model.decoder.state_dict(), strict=True)
            for length in (7, 513):
                with self.subTest(spec=spec.name, length=length), torch.no_grad():
                    ids = torch.randint(3, 100, (1, length))
                    mask = torch.ones_like(ids, dtype=torch.bool)
                    mask[:, :2] = False
                    positions = model._positions(mask)
                    expected = model.lm_head(
                        native(
                            ids, attention_mask=mask, position_ids=positions, use_cache=False
                        ).last_hidden_state
                    )
                    cap = model.decoder.config.final_logit_softcapping
                    assert cap is not None
                    expected = torch.tanh(expected / cap) * cap
                    actual = model.decode_logits(ids, mask)
                    torch.testing.assert_close(actual[mask], expected[mask], atol=2e-5, rtol=2e-5)

    def test_future_decoder_isolation_encoder_influence_and_padding(self):
        model = build_native_tiny_model(EXPERIMENT_SPECS[2]).eval()
        dec = torch.tensor([[2, 8, 9, 10]])
        enc = torch.tensor([[2, 3, 4, 5]])
        valid = torch.ones_like(enc, dtype=torch.bool)
        with torch.no_grad():
            hidden = model.encode_context(enc, valid)
            original = model.decode_logits(dec, valid, hidden, valid)
            changed_dec = dec.clone()
            changed_dec[0, -1] = 12
            changed = model.decode_logits(changed_dec, valid, hidden, valid)
            torch.testing.assert_close(original[:, :-1], changed[:, :-1], atol=1e-6, rtol=0)
            changed_enc = enc.clone()
            changed_enc[0, -1] = 13
            hidden_b = model.encode_context(changed_enc, valid)
            self.assertFalse(torch.equal(hidden[:, 0], hidden_b[:, 0]))
            self.assertFalse(
                torch.equal(original, model.decode_logits(dec, valid, hidden_b, valid))
            )
            padded_enc = torch.cat((torch.zeros(1, 2, dtype=torch.long), enc), 1)
            padded_valid = padded_enc != 0
            padded_hidden = model.encode_context(padded_enc, padded_valid)
            torch.testing.assert_close(hidden, padded_hidden[:, 2:], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(
                original,
                model.decode_logits(dec, valid, padded_hidden, padded_valid),
                atol=2e-5,
                rtol=2e-5,
            )

    def test_all_masked_encoder_and_decoder_rows_finite_backward(self):
        model = build_native_tiny_model(EXPERIMENT_SPECS[0])
        ids = torch.tensor([[2, 3, 4], [2, 6, 7]])
        mask = torch.tensor([[False, False, False], [True, True, True]])
        hidden = model.encode_context(ids, torch.zeros_like(mask))
        logits = model.decode_logits(ids, mask, hidden, torch.zeros_like(mask))
        empty_logits = model.decode_logits(ids, mask)
        torch.testing.assert_close(logits[mask], empty_logits[mask], atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )

    def test_shared_kv_uses_owner_without_extra_projections(self):
        for spec in EXPERIMENT_SPECS[:2]:
            model = build_native_tiny_model(spec)
            owners, reused, hooks = {}, [], []

            def inspect(module, args, kwargs, output):
                states = kwargs["shared_kv_states"]
                if module.store_full_length_kv:
                    owners[module.layer_type] = states[module.layer_type]
                if module.is_kv_shared_layer:
                    self.assertIs(states[module.layer_type][0], owners[module.layer_type][0])
                    self.assertEqual(states[module.layer_type][0].shape[2], 7)
                    self.assertFalse(hasattr(module, "k_proj"))
                    self.assertFalse(hasattr(module, "v_proj"))
                    reused.append(module.layer_idx)

            for layer in model.decoder.layers:
                layer = cast(Gemma4TextDecoderLayer, layer)
                hooks.append(layer.self_attn.register_forward_hook(inspect, with_kwargs=True))
            try:
                enc = torch.tensor([[2, 3, 4, 5]])
                dec = torch.tensor([[2, 6, 7]])
                hidden = model.encode_context(enc, enc != 0)
                model.decode_logits(dec, dec != 0, hidden, enc != 0).sum().backward()
                self.assertEqual(len(reused), model.decoder.config.num_kv_shared_layers)
                self.assertEqual(set(owners), {"sliding_attention", "full_attention"})
            finally:
                for hook in hooks:
                    hook.remove()

    def test_local_window_does_not_hide_cross_context(self):
        config = native_config("E2B")
        attn = NativeMergedAttention(config, 0).eval()
        states = torch.randn(1, 513, 32)
        context = torch.randn(1, 2, 32)
        valid = torch.ones(1, 513, dtype=torch.bool)
        cvalid = torch.ones(1, 2, dtype=torch.bool)
        # Identity RoPE isolates mask behavior, actual RoPE is covered by parity.
        positions = (torch.ones(1, 513, 8), torch.zeros(1, 513, 8))

        def run(x, h):
            return attn(
                x,
                positions,
                None,
                {},
                encoder_hidden_states=h,
                encoder_valid=cvalid,
                decoder_valid=valid,
            )[0]

        with torch.no_grad():
            original = run(states, context)
            changed = states.clone()
            changed[:, 0] += 10
            torch.testing.assert_close(original[:, -1], run(changed, context)[:, -1])
            changed_context = context.clone()
            changed_context[:, 0] += 10
            self.assertFalse(torch.equal(original[:, -1], run(states, changed_context)[:, -1]))

    def test_word_alias_reload_and_dtype_move_ple_separate(self):
        for spec in EXPERIMENT_SPECS[:4]:
            model = build_native_tiny_model(spec).eval().to(dtype=torch.float64)
            enc = model.encoder
            assert enc is not None
            self.assertEqual(
                enc.embed_tokens.weight is model.decoder.embed_tokens.weight,
                spec.tie_encoder_decoder_words,
            )
            self.assertIs(model.lm_head.weight, model.decoder.embed_tokens.weight)
            self.assertIsNot(
                enc.embed_tokens_per_layer.weight, model.decoder.embed_tokens_per_layer.weight
            )
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = build_native_tiny_model(spec).eval().to(dtype=torch.float64)
            restored.load_state_dict(torch.load(buffer, weights_only=True), assign=True)
            assert restored.encoder is not None
            self.assertEqual(
                restored.encoder.embed_tokens.weight is restored.decoder.embed_tokens.weight,
                spec.tie_encoder_decoder_words,
            )
            self.assertIs(restored.lm_head.weight, restored.decoder.embed_tokens.weight)
            ids = torch.tensor([[2, 3, 4]])
            torch.testing.assert_close(
                model.decode_logits(ids, ids != 0), restored.decode_logits(ids, ids != 0)
            )

    def test_temporal_cache_explicitly_rejected(self):
        model = build_native_tiny_model(EXPERIMENT_SPECS[0])
        with self.assertRaisesRegex(ValueError, "temporal cache"):
            model.decoder(
                torch.tensor([[2, 3]]),
                use_cache=True,
                encoder_hidden_states=torch.empty(1, 0, 32),
                encoder_valid=torch.empty(1, 0, dtype=torch.bool),
                decoder_valid=torch.ones(1, 2, dtype=torch.bool),
            )
