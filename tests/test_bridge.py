"""Unit tests for JANUS-4 architectural modules."""

import unittest

import torch

from janus.models.bridge import GatedCrossAttentionBridge
from janus.models.patch_projector import LinearPatchProjector
from janus.models.routing import MultimodalTokenRouter


class TestJanusArchitecture(unittest.TestCase):
    def test_gated_cross_attention_forward_backward(self):
        batch_size = 2
        dec_len = 16
        enc_len = 64
        decoder_dim = 256  # Scaled down for unit testing
        encoder_dim = 384

        bridge = GatedCrossAttentionBridge(
            decoder_dim=decoder_dim,
            encoder_dim=encoder_dim,
            num_heads=4,
            head_dim=64,
            initial_gate_val=0.05,
            use_qk_norm=True,
        )

        dec_states = torch.randn(batch_size, dec_len, decoder_dim, requires_grad=True)
        enc_states = torch.randn(batch_size, enc_len, encoder_dim, requires_grad=True)

        out = bridge(dec_states, enc_states)
        self.assertEqual(out.shape, (batch_size, dec_len, decoder_dim))

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
