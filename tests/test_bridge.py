"""Unit tests for JANUS-4 architectural modules and 2x2 model matrix."""

import unittest

import torch

from janus.models.bridge import GatedCrossAttentionBridge
from janus.models.patch_projector import LinearPatchProjector
from janus.models.routing import MultimodalTokenRouter


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
        """Asymmetric E4B Encoder + E2B Decoder (~6.8B: Heavy Reader, Agile Writer)."""
        self._run_bridge_test(enc_dim=384, dec_dim=256)

    def test_asymmetric_e2b_enc_e4b_dec(self):
        """Asymmetric E2B Encoder + E4B Decoder (~6.8B: Agile Reader, Deep Reasoner)."""
        self._run_bridge_test(enc_dim=256, dec_dim=384)

    def test_linear_patch_projector(self):
        batch_size = 2
        num_patches = 576  # 24 x 24 grid
        patch_size = 48
        in_channels = 3
        patch_dim = patch_size * patch_size * in_channels
        embed_dim = 512

        projector = LinearPatchProjector(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            max_patches_h=24,
            max_patches_w=24,
        )

        dummy_patches = torch.randn(batch_size, num_patches, patch_dim)
        embeddings = projector(dummy_patches, grid_shape=(24, 24))

        self.assertEqual(embeddings.shape, (batch_size, num_patches, embed_dim))

    def test_token_router_kv_savings(self):
        savings = MultimodalTokenRouter.calculate_kv_cache_savings(
            num_images=3,
            patches_per_image=576,
            num_layers=32,
            num_kv_heads=8,
            head_dim=256,
            bytes_per_param=2,
        )
        self.assertGreater(savings, 200.0)


if __name__ == "__main__":
    unittest.main()
