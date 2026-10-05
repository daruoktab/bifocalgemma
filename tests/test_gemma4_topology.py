"""Offline checks of Gemma 4 Edge topology using config snapshots and random weights.

Exact checkpoint dimensions are instantiated on meta only. Runtime tests keep
the real layer schedule, KV-head counts, KV sharing, PLE, and MLP variants while
reducing tensor widths and vocabulary. No pretrained weights are downloaded.
"""

import copy
import json
import unittest
from pathlib import Path
from typing import cast

import torch
from transformers import Gemma4TextConfig, Gemma4TextModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

from bifocalgemma.models import GatedCrossAttentionBridge

FIXTURES = Path(__file__).parent / "fixtures"
PROFILES = {
    "e2b": {"width": 1536, "layers": 35, "local": 28, "kv_heads": 1, "shared": 20},
    "e4b": {"width": 2560, "layers": 42, "local": 35, "kv_heads": 2, "shared": 18},
}


class TestGemma4EdgeTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, profile: str, *, thin: bool = False, bidirectional: bool = False):
        snapshot = json.loads((FIXTURES / f"gemma4_{profile}_text_config.json").read_text())
        values = copy.deepcopy(snapshot["text_config"])
        if thin:
            width = 32 if profile == "e2b" else 48
            values.update(
                vocab_size=128,
                vocab_size_per_layer_input=128,
                hidden_size=width,
                intermediate_size=4 * width,
                hidden_size_per_layer_input=4,
                head_dim=8,
                global_head_dim=16,
            )
        values["use_bidirectional_attention"] = "all" if bidirectional else None
        values["use_cache"] = False
        config = Gemma4TextConfig.from_dict(values)
        config._attn_implementation = "eager"
        return config

    def test_exact_edge_parameter_shapes_on_meta(self):
        for name, expected in PROFILES.items():
            with self.subTest(profile=name):
                config = self.config(name)
                with torch.device("meta"):
                    model = Gemma4TextModel(config)
                self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
                self.assertEqual(len(model.layers), expected["layers"])
                self.assertEqual(config.layer_types.count("sliding_attention"), expected["local"])
                self.assertEqual(config.layer_types.count("full_attention"), 7)
                self.assertEqual(config.num_key_value_heads, expected["kv_heads"])
                attention_layers = [
                    cast(Gemma4TextAttention, layer.self_attn) for layer in model.layers
                ]
                self.assertEqual(
                    sum(bool(attention.is_kv_shared_layer) for attention in attention_layers),
                    expected["shared"],
                )
                self.assertEqual(model.embed_tokens.weight.shape, (262144, expected["width"]))
                self.assertEqual(
                    model.embed_tokens_per_layer.weight.shape,
                    (262144, expected["layers"] * 256),
                )
                local_projection = attention_layers[0].q_proj
                assert isinstance(local_projection, torch.nn.Linear)
                self.assertEqual(local_projection.weight.shape, (2048, expected["width"]))
                global_projection = attention_layers[
                    config.layer_types.index("full_attention")
                ].q_proj
                assert isinstance(global_projection, torch.nn.Linear)
                self.assertEqual(global_projection.weight.shape, (4096, expected["width"]))

    def test_full_depth_future_token_visibility(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                torch.manual_seed(7)
                causal = Gemma4TextModel(self.config(name, thin=True)).eval()
                bidirectional = Gemma4TextModel(
                    self.config(name, thin=True, bidirectional=True)
                ).eval()
                bidirectional.load_state_dict(causal.state_dict())
                self.assertLess(sum(p.numel() for p in causal.parameters()), 2_000_000)
                tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
                changed = tokens.clone()
                changed[0, -1] = 19
                with torch.no_grad():
                    causal_a = causal(tokens).last_hidden_state
                    causal_b = causal(changed).last_hidden_state
                    bidi_a = bidirectional(tokens).last_hidden_state
                    bidi_b = bidirectional(changed).last_hidden_state
                torch.testing.assert_close(causal_a[:, 0], causal_b[:, 0], atol=1e-6, rtol=0)
                self.assertGreater((bidi_a[:, 0] - bidi_b[:, 0]).abs().max().item(), 1e-5)

    def test_causal_shared_kv_incremental_matches_full(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                torch.manual_seed(11)
                model = Gemma4TextModel(self.config(name, thin=True)).eval()
                tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
                with torch.no_grad():
                    full = model(tokens, use_cache=False).last_hidden_state
                    prefix = model(tokens[:, :-1], use_cache=True)
                    incremental = model(
                        tokens[:, -1:], past_key_values=prefix.past_key_values, use_cache=True
                    ).last_hidden_state
                torch.testing.assert_close(incremental[:, -1], full[:, -1], atol=2e-5, rtol=2e-5)

    def test_all_edge_pairs_bridge_backward_at_full_depth(self):
        for encoder_name in PROFILES:
            for decoder_name in PROFILES:
                with self.subTest(encoder=encoder_name, decoder=decoder_name):
                    torch.manual_seed(17)
                    encoder_config = self.config(encoder_name, thin=True, bidirectional=True)
                    decoder_config = self.config(decoder_name, thin=True)
                    encoder = Gemma4TextModel(encoder_config)
                    decoder = Gemma4TextModel(decoder_config)
                    bridge = GatedCrossAttentionBridge(
                        encoder_dim=encoder_config.hidden_size,
                        decoder_dim=decoder_config.hidden_size,
                        num_heads=8,
                        head_dim=8,
                    )
                    encoder_hidden = encoder(torch.tensor([[2, 3, 4, 5]])).last_hidden_state
                    decoder_hidden = decoder(torch.tensor([[2, 6, 7]])).last_hidden_state
                    output = bridge(decoder_hidden, encoder_hidden)
                    self.assertEqual(output.shape, (1, 3, decoder_config.hidden_size))
                    self.assertTrue(torch.isfinite(output).all())
                    output.square().mean().backward()
                    for parameter in (
                        encoder.embed_tokens.weight,
                        decoder.embed_tokens.weight,
                        bridge.gate,
                    ):
                        gradient = parameter.grad
                        self.assertIsNotNone(gradient)
                        assert gradient is not None
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
