"""Unit tests for JANUS-4 architectural modules and 2x2 model matrix for long-context code."""

import unittest

import torch

from janus.models.bridge import GatedCrossAttentionBridge
from janus.models.code_graph_projector import ASTGraphProjector
from janus.models.routing import RepoContextRouter


class TestJanusArchitecture(unittest.TestCase):
    def _run_bridge_test(self, enc_dim: int, dec_dim: int):
        batch_size = 2
        dec_len = 16
        enc_len = 64

        bridge = GatedCrossAttentionBridge(
            decoder_dim=dec_dim,
            encoder_dim=enc_dim,
            num_heads=4,
            head_dim=64,
            initial_gate_val=0.05,
            use_qk_norm=True,
        )

        dec_states = torch.randn(batch_size, dec_len, dec_dim, requires_grad=True)
        enc_states = torch.randn(batch_size, enc_len, enc_dim, requires_grad=True)

        out = bridge(dec_states, enc_states)
        self.assertEqual(out.shape, (batch_size, dec_len, dec_dim))

        loss = out.sum()
        loss.backward()

        # Verify gradients flow back to both inputs
        self.assertIsNotNone(dec_states.grad)
        self.assertIsNotNone(enc_states.grad)
        assert enc_states.grad is not None
        self.assertGreater(torch.abs(enc_states.grad).sum().item(), 0.0)

        # Verify gate parameter receives non-zero gradient (no gradient starvation)
        self.assertIsNotNone(bridge.gate.grad)
        assert bridge.gate.grad is not None
        self.assertGreater(torch.abs(bridge.gate.grad).item(), 0.0)

    def test_symmetric_e2b_e2b(self):
        """Symmetric E2B Encoder + E2B Decoder (~4.6B Effective)."""
        self._run_bridge_test(enc_dim=256, dec_dim=256)

    def test_symmetric_e4b_e4b(self):
        """Symmetric E4B Encoder + E4B Decoder (~9.0B Effective)."""
        self._run_bridge_test(enc_dim=384, dec_dim=384)

    def test_asymmetric_e4b_enc_e2b_dec(self):
        """Asymmetric E4B Encoder + E2B Decoder (~6.8B: Heavy Reader, Agile Coder)."""
        self._run_bridge_test(enc_dim=384, dec_dim=256)

    def test_asymmetric_e2b_enc_e4b_dec(self):
        """Asymmetric E2B Encoder + E4B Decoder (~6.8B: Agile Reader, Deep Reasoner)."""
        self._run_bridge_test(enc_dim=256, dec_dim=384)

    def test_ast_graph_projector(self):
        """Tests AST node type and syntax depth embedding projection."""
        batch_size = 2
        seq_len = 128
        embed_dim = 256

        projector = ASTGraphProjector(
            num_node_types=100,
            max_depth=32,
            embed_dim=embed_dim,
        )

        dummy_node_types = torch.randint(0, 100, (batch_size, seq_len))
        dummy_depths = torch.randint(0, 32, (batch_size, seq_len))

        embeddings = projector(dummy_node_types, depths=dummy_depths)
        self.assertEqual(embeddings.shape, (batch_size, seq_len, embed_dim))

    def test_long_context_kv_savings(self):
        """Tests KV-cache memory calculation for 16k context vs 512 gen tokens."""
        savings = RepoContextRouter.calculate_long_context_kv_savings(
            prompt_len=16384,
            gen_len=512,
            num_layers=32,
            num_kv_heads=8,
            head_dim=256,
            bytes_per_param=2,
        )
        self.assertGreater(savings["saved_mb"], 1000.0)
        self.assertGreater(savings["reduction_percentage"], 95.0)

    def test_bidirectional_repo_mask(self):
        """Tests bidirectional mask construction for repository context."""
        mask = RepoContextRouter.create_bidirectional_repo_mask(
            seq_len=2048,
            device=torch.device("cpu"),
        )
        self.assertEqual(mask.shape, (1, 2048))
        self.assertTrue(torch.all(mask))


if __name__ == "__main__":
    unittest.main()
