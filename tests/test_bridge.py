"""Synthetic module tests for BifocalGemma; no Gemma weights or GPU are required."""

import sys
import unittest
from pathlib import Path

# Ensure src/ is on sys.path for direct test discovery without requiring external PYTHONPATH
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch  # noqa: E402

from bifocalgemma.models.bridge import SwiGLUDualGateBridge  # noqa: E402


class TestBifocalGemmaArchitecture(unittest.TestCase):
    def _run_bridge_test(self, enc_dim: int, dec_dim: int) -> None:
        batch_size = 2
        enc_len = 64

        bridge = SwiGLUDualGateBridge(d_enc=enc_dim, d_dec=dec_dim)

        enc_states = torch.randn(batch_size, enc_len, enc_dim, requires_grad=True)
        out = bridge(enc_states)
        self.assertEqual(out.shape, (batch_size, enc_len, dec_dim))

        loss = out.sum()
        loss.backward()

        # Verify gradients flow back to encoder states and bridge parameters
        self.assertIsNotNone(enc_states.grad)
        assert enc_states.grad is not None
        self.assertGreater(torch.abs(enc_states.grad).sum().item(), 0.0)

        for param in (bridge.w_res.weight, bridge.w_down.weight, bridge.gamma_calib):
            self.assertIsNotNone(param.grad)
            assert param.grad is not None
            self.assertGreater(torch.abs(param.grad).sum().item(), 0.0)

    def test_symmetric_e2b_e2b(self):
        """Synthetic equal-width bridge with exact Gemma 4 E2B hidden dimension (1536)."""
        self._run_bridge_test(enc_dim=1536, dec_dim=1536)

    def test_symmetric_e4b_e4b(self):
        """Synthetic equal-width bridge with exact Gemma 4 E4B hidden dimension (2560)."""
        self._run_bridge_test(enc_dim=2560, dec_dim=2560)

    def test_asymmetric_e4b_enc_e2b_dec(self):
        """Synthetic bridge: Gemma 4 E4B encoder (2560) to E2B decoder (1536)."""
        self._run_bridge_test(enc_dim=2560, dec_dim=1536)

    def test_asymmetric_e2b_enc_e4b_dec(self):
        """Synthetic bridge: Gemma 4 E2B encoder (1536) to E4B decoder (2560)."""
        self._run_bridge_test(enc_dim=1536, dec_dim=2560)

    def test_cuda_bridge_execution(self):
        """Verify SwiGLU bridge execution on GPU when CUDA is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda:0")
        bridge = SwiGLUDualGateBridge(d_enc=1536, d_dec=2560).to(device)
        enc_states = torch.randn(1, 64, 1536, device=device, requires_grad=True)
        out = bridge(enc_states)
        self.assertEqual(out.shape, (1, 64, 2560))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(enc_states.grad)
        self.assertIsNotNone(bridge.w_res.weight.grad)
        self.assertIsNotNone(bridge.w_down.weight.grad)
        self.assertIsNotNone(bridge.gamma_calib.grad)

    def test_gemma4_structural_integration(self):
        """Structural integration test using Gemma4TextModel with synthetic weights.

        Verifies that actual Gemma 4 model output hidden states with E2B and E4B
        dimensions seamlessly pass through SwiGLUDualGateBridge and backpropagate gradients.
        """
        try:
            from transformers import Gemma4TextConfig, Gemma4TextModel
        except ImportError:
            self.skipTest("transformers with Gemma4 support not available")

        # Fast synthetic configuration with 2 layers and compact vocab for unit test speed
        # The encoder explicitly enables bidirectional attention across all repository tokens
        enc_config = Gemma4TextConfig(
            vocab_size=1000,
            vocab_size_per_layer_input=1000,
            hidden_size_per_layer_input=16,
            use_cache=False,
            hidden_size=1536,
            intermediate_size=6144,
            use_double_wide_mlp=True,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=256,
            use_bidirectional_attention="all",
        )
        dec_config = Gemma4TextConfig(
            vocab_size=1000,
            vocab_size_per_layer_input=1000,
            hidden_size_per_layer_input=16,
            use_cache=False,
            hidden_size=2560,
            intermediate_size=10240,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
        )

        enc_model = Gemma4TextModel(enc_config)
        dec_model = Gemma4TextModel(dec_config)
        bridge = SwiGLUDualGateBridge(d_enc=1536, d_dec=2560)

        input_enc = torch.randint(0, 1000, (1, 16))
        input_dec = torch.randint(0, 1000, (1, 8))

        enc_hidden = enc_model(input_enc).last_hidden_state
        dec_hidden = dec_model(input_dec).last_hidden_state

        self.assertEqual(enc_hidden.shape, (1, 16, 1536))
        self.assertEqual(dec_hidden.shape, (1, 8, 2560))

        bridged = bridge(enc_hidden)
        self.assertEqual(bridged.shape, (1, 16, 2560))

        loss = (dec_hidden @ bridged.transpose(-1, -2)).sum()
        loss.backward()

        enc_param = next(enc_model.parameters())
        dec_param = next(dec_model.parameters())
        self.assertIsNotNone(enc_param.grad)
        self.assertIsNotNone(dec_param.grad)
        self.assertIsNotNone(bridge.w_res.weight.grad)
        self.assertIsNotNone(bridge.w_down.weight.grad)
        self.assertIsNotNone(bridge.gamma_calib.grad)

    def test_gemma4_bidirectional_attention_mechanism(self):
        """Verify that Gemma4TextModel with use_bidirectional_attention='all' allows
        later repository tokens to influence earlier token representations (bidirectional flow),
        unlike causal generation where token 0 is strictly invariant to future tokens.
        """
        try:
            from transformers import Gemma4TextConfig, Gemma4TextModel
        except ImportError:
            self.skipTest("transformers with Gemma4 support not available")

        # Causal baseline configuration
        causal_config = Gemma4TextConfig.from_dict(
            dict(
                vocab_size=100,
                vocab_size_per_layer_input=100,
                hidden_size_per_layer_input=8,
                global_head_dim=64,
                use_cache=False,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                use_bidirectional_attention=None,
            )
        )
        causal_model = Gemma4TextModel(causal_config)
        causal_model.eval()

        # Bidirectional encoder configuration
        bidi_config = Gemma4TextConfig.from_dict(
            dict(
                vocab_size=100,
                vocab_size_per_layer_input=100,
                hidden_size_per_layer_input=8,
                global_head_dim=64,
                use_cache=False,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                use_bidirectional_attention="all",
            )
        )
        bidi_model = Gemma4TextModel(bidi_config)
        bidi_model.eval()

        tokens_a = torch.tensor([[1, 2, 3, 4, 5]])
        tokens_b = torch.tensor([[1, 2, 3, 4, 99]])  # only last token is altered

        with torch.no_grad():
            # In causal model, token 0 hidden state must be invariant to subsequent tokens
            causal_out_a = causal_model(tokens_a).last_hidden_state
            causal_out_b = causal_model(tokens_b).last_hidden_state
            causal_token0_diff = (causal_out_a[0, 0] - causal_out_b[0, 0]).abs().sum().item()
            self.assertAlmostEqual(causal_token0_diff, 0.0, places=4)

            # In bidirectional model, token 0 hidden state must change due to full bidirectional attention
            bidi_out_a = bidi_model(tokens_a).last_hidden_state
            bidi_out_b = bidi_model(tokens_b).last_hidden_state
            bidi_token0_diff = (bidi_out_a[0, 0] - bidi_out_b[0, 0]).abs().sum().item()
            self.assertGreater(bidi_token0_diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
