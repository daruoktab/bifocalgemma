"""Small local fixtures: real checkpoint/model runs happen on the user's Molab GPU."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.checkpoint_transfer import transfer_text_checkpoint
from bifocalgemma.models.native import build_native_tiny_model
from bifocalgemma.training.memory_probe import (
    ProbeLoRALinear,
    configure_probe_training,
    profile_training_group,
)


class TestCheckpointTransfer(unittest.TestCase):
    def fixture(self, model, value):
        return {
            "model.language_model." + key: torch.full_like(tensor, value)
            for key, tensor in model.state_dict().items()
        }

    def test_decoder_anchor_base_body_ple_and_shared_kv_extras(self):
        model = build_native_tiny_model(EXPERIMENT_SPECS[0])
        assert model.encoder is not None
        decoder = self.fixture(model.decoder, 0.3)
        encoder = self.fixture(model.encoder, 0.7)
        shared = model.decoder.config.num_hidden_layers - model.decoder.config.num_kv_shared_layers
        decoder[f"model.language_model.layers.{shared}.self_attn.k_proj.weight"] = torch.zeros(1)
        decoder["model.vision_tower.unused.weight"] = torch.zeros(2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.safetensors"
            save_file(decoder, path)
            result = transfer_text_checkpoint(model.decoder, path)
            self.assertEqual(len(result["ignored_shared_kv_tensors"]), 1)
            self.assertEqual(result["nontext_tensors_ignored"], 1)
            save_file(encoder, path)
            transfer_text_checkpoint(model.encoder, path, skip_words=True)
        self.assertIs(model.encoder.embed_tokens.weight, model.decoder.embed_tokens.weight)
        torch.testing.assert_close(
            model.encoder.embed_tokens.weight,
            torch.full_like(model.encoder.embed_tokens.weight, 0.3),
        )
        torch.testing.assert_close(
            model.encoder.embed_tokens_per_layer.weight,
            torch.full_like(model.encoder.embed_tokens_per_layer.weight, 0.7),
        )
        self.assertIsNot(
            model.encoder.embed_tokens_per_layer.weight, model.decoder.embed_tokens_per_layer.weight
        )

    def test_missing_shape_or_unknown_text_key_fails_before_copy(self):
        model = build_native_tiny_model(EXPERIMENT_SPECS[4])
        original = {k: v.clone() for k, v in model.decoder.state_dict().items()}
        valid = self.fixture(model.decoder, 1)
        variants = []
        missing = dict(valid)
        missing.pop(next(iter(missing)))
        variants.append(missing)
        wrong = dict(valid)
        wrong[next(iter(wrong))] = torch.zeros(1)
        variants.append(wrong)
        unknown = dict(valid)
        unknown["model.language_model.unmapped.weight"] = torch.zeros(1)
        variants.append(unknown)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.safetensors"
            for state in variants:
                save_file(state, path)
                with self.assertRaises(ValueError):
                    transfer_text_checkpoint(model.decoder, path)
                for key, tensor in model.decoder.state_dict().items():
                    torch.testing.assert_close(tensor, original[key])


class TestMemoryProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def batch(self):
        return RepairCollator()(
            [TokenizedRepairExample("synthetic", (2, 3, 4), (2, 105, 7), (18, 106))]
        )

    def test_lora_zero_initialization_and_trainable_scope(self):
        model = build_native_tiny_model(EXPERIMENT_SPECS[2])
        scope = configure_probe_training(model, policy="lora", rank=2)
        self.assertGreater(scope["trainable_parameters"], 0)
        assert model.encoder is not None
        self.assertFalse(model.encoder.embed_tokens.weight.requires_grad)
        self.assertFalse(model.decoder.embed_tokens_per_layer.weight.requires_grad)
        self.assertIs(model.lm_head.weight, model.decoder.embed_tokens.weight)
        assert isinstance(model.encoder_projection, torch.nn.Linear)
        self.assertTrue(model.encoder_projection.weight.requires_grad)
        adapters = [m for m in model.modules() if isinstance(m, ProbeLoRALinear)]
        self.assertTrue(adapters)
        module = adapters[0]
        values = torch.randn(1, 2, module.base.in_features)
        torch.testing.assert_close(module(values), module.base(values))

    def test_optimizer_states_and_overlap_graphs_are_measured(self):
        models = [build_native_tiny_model(EXPERIMENT_SPECS[0]) for _ in range(2)]
        for model in models:
            configure_probe_training(model, rank=2)
        before = [
            next(m.lora_B for m in model.modules() if isinstance(m, ProbeLoRALinear)).clone()
            for model in models
        ]
        for mode in ("round_robin", "overlap_graphs"):
            result = profile_training_group(
                models, [self.batch(), self.batch()], steps=2, mode=mode
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["model_count"], 2)
            self.assertGreater(result["optimizer_state_bytes"], 0)
            self.assertEqual(result["optimizer_gpu_state_bytes"], 0)
            self.assertIsNone(result["after"]["peak_allocated_bytes"])
            self.assertFalse(result["concurrent_cuda_kernels_claimed"])
            self.assertEqual(len(result["losses"]), 4)
            self.assertTrue(all(p.grad is None for m in models for p in m.parameters()))
        for model, old in zip(models, before, strict=True):
            changed = next(m.lora_B for m in model.modules() if isinstance(m, ProbeLoRALinear))
            self.assertFalse(torch.equal(changed, old))

    def test_shared_training_model_rejected(self):
        model = build_native_tiny_model(EXPERIMENT_SPECS[0])
        configure_probe_training(model, rank=2)
        with self.assertRaisesRegex(ValueError, "independent"):
            profile_training_group([model, model], [self.batch(), self.batch()])
