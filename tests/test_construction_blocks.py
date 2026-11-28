"""Semantic tests for offline construction blocks; no pretrained downloads."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from bifocalgemma.__main__ import main
from bifocalgemma.config import EXPERIMENT_SPECS, get_experiment, validate_experiment_matrix
from bifocalgemma.configuration_bifocalgemma import native_config
from bifocalgemma.contracts import (
    RepairExample,
    RepairPrediction,
    RepositoryFile,
    RepositorySnapshot,
    TokenizedRepairExample,
)
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.tokenization import format_encoder_context
from bifocalgemma.evaluation.predictions import write_swebench_predictions
from bifocalgemma.modeling_bifocalgemma import NativeMergedAttention, run_native_smoke
from bifocalgemma.models.embeddings import apply_word_embedding_policy
from bifocalgemma.models.memory_efficient import build_joint_visibility
from bifocalgemma.training.objectives import patch_cross_entropy

ROOT = Path(__file__).resolve().parent.parent


class TestWordEmbeddingPolicy(unittest.TestCase):
    def test_symmetric_alias_preserves_decoder_and_accumulates_both_gradients(self):
        encoder, decoder = nn.Embedding(8, 4), nn.Embedding(8, 4)
        head = nn.Linear(4, 8, bias=False)
        before = decoder.weight.detach().clone()
        result = apply_word_embedding_policy(encoder, decoder, head, tie_encoder_decoder=True)
        self.assertIs(encoder.weight, decoder.weight)
        self.assertIs(head.weight, decoder.weight)
        torch.testing.assert_close(decoder.weight, before)
        self.assertEqual(result.saved_encoder_parameters, 32)
        holder = nn.ModuleDict({"encoder": encoder, "decoder": decoder, "head": head})
        self.assertEqual(len(list(holder.parameters())), 1)
        (encoder(torch.tensor([2])).sum() + decoder(torch.tensor([2])).sum()).backward()
        assert decoder.weight.grad is not None
        torch.testing.assert_close(decoder.weight.grad[2], torch.full((4,), 2.0))

    def test_asymmetric_keeps_encoder_separate_but_ties_decoder_head(self):
        encoder, decoder = nn.Embedding(8, 4), nn.Embedding(8, 6)
        head = nn.Linear(6, 8, bias=False)
        result = apply_word_embedding_policy(encoder, decoder, head, tie_encoder_decoder=False)
        self.assertIsNot(encoder.weight, decoder.weight)
        self.assertIs(head.weight, decoder.weight)
        self.assertFalse(result.tied_encoder_decoder)
        with self.assertRaises(ValueError):
            apply_word_embedding_policy(encoder, decoder, head, tie_encoder_decoder=True)


class TestMergedAttentionReference(unittest.TestCase):
    def _attention(self):
        return NativeMergedAttention(native_config("E2B"), 0).eval()

    def test_joint_normalization_competes_across_self_and_encoder_keys(self):
        model = self._attention()
        with torch.no_grad():
            assert model.v_proj is not None
            model.q_proj.weight.zero_()
            model.k_proj.weight.zero_()
            model.v_proj.weight.zero_()
            model.v_proj.weight[:8, :8].copy_(torch.eye(8))
            model.o_proj.weight.zero_()
            model.o_proj.weight[:8, :8].copy_(torch.eye(8))
        dec = torch.zeros(1, 1, 32)
        enc = torch.zeros(1, 1, 32)
        dec[0, 0, 0] = 1.0
        enc[0, 0, 1] = 1.0
        positions = (torch.ones(1, 1, 8), torch.zeros(1, 1, 8))
        output = model(
            dec,
            positions,
            None,
            {},
            encoder_hidden_states=enc,
            encoder_valid=torch.ones(1, 1, dtype=torch.bool),
            decoder_valid=torch.ones(1, 1, dtype=torch.bool),
        )[0]
        # RMSNorm(with_scale=False) scales single unit entry by sqrt(head_dim=8), and
        # uniform 2-key softmax yields 0.5 weight on self and encoder.
        expected_head = torch.zeros(8)
        expected_head[0] = 0.5 * (8**0.5)
        expected_head[1] = 0.5 * (8**0.5)
        torch.testing.assert_close(output[0, 0, :8], expected_head)
        torch.testing.assert_close(output[0, 0, 8:], torch.zeros(24))

    def test_future_decoder_tokens_hidden_but_encoder_tokens_visible(self):
        torch.manual_seed(19)
        model = self._attention()
        decoder, encoder = torch.randn(1, 4, 32), torch.randn(1, 3, 32)
        enc_valid = torch.ones(1, 3, dtype=torch.bool)
        dec_valid = torch.ones(1, 4, dtype=torch.bool)
        positions = (torch.ones(1, 4, 8), torch.zeros(1, 4, 8))
        original = model(
            decoder,
            positions,
            None,
            {},
            encoder_hidden_states=encoder,
            encoder_valid=enc_valid,
            decoder_valid=dec_valid,
        )[0]
        changed_decoder = decoder.clone()
        changed_decoder[:, -1] = torch.randn(1, 32) * 5
        changed = model(
            changed_decoder,
            positions,
            None,
            {},
            encoder_hidden_states=encoder,
            encoder_valid=enc_valid,
            decoder_valid=dec_valid,
        )[0]
        torch.testing.assert_close(original[:, :3], changed[:, :3])
        changed_encoder = encoder.clone()
        changed_encoder[:, -1] = torch.randn(1, 32) * 5
        encoded_change = model(
            decoder,
            positions,
            None,
            {},
            encoder_hidden_states=changed_encoder,
            encoder_valid=enc_valid,
            decoder_valid=dec_valid,
        )[0]
        self.assertGreater((original[:, 0] - encoded_change[:, 0]).abs().max().item(), 1e-5)

    def test_local_self_window_does_not_truncate_encoder_context(self):
        mask = build_joint_visibility(
            torch.tensor([[True, False]]), torch.ones(1, 4, dtype=torch.bool), self_window=2
        )
        self.assertEqual(mask[0, 0, 3].tolist(), [False, False, True, True, True, False])

    def test_fully_masked_queries_have_finite_zero_outputs_and_gradients(self):
        model = self._attention()
        decoder = torch.randn(1, 2, 32, requires_grad=True)
        encoder = torch.randn(1, 2, 32, requires_grad=True)
        invalid = torch.zeros(1, 2, dtype=torch.bool)
        positions = (torch.ones(1, 2, 8), torch.zeros(1, 2, 8))
        output = model(
            decoder,
            positions,
            None,
            {},
            encoder_hidden_states=encoder,
            encoder_valid=invalid,
            decoder_valid=invalid,
        )[0]
        torch.testing.assert_close(output, torch.zeros_like(output))
        output.sum().backward()
        assert decoder.grad is not None and encoder.grad is not None
        self.assertTrue(torch.isfinite(decoder.grad).all())
        self.assertTrue(torch.isfinite(encoder.grad).all())


class TestDataAndLoss(unittest.TestCase):
    def test_teacher_forcing_shifts_once_and_masks_prefix_and_padding(self):
        batch = RepairCollator()(
            [
                TokenizedRepairExample("a", (2, 10, 11), (2, 105, 7), (8, 9, 106)),
                TokenizedRepairExample("b", (2, 12), (2, 105), (13, 106)),
            ]
        )
        self.assertEqual(batch.decoder_input_ids[0].tolist(), [2, 105, 7, 8, 9])
        self.assertEqual(batch.labels[0].tolist(), [-100, -100, 8, 9, 106])
        self.assertEqual(batch.labels[1].tolist(), [-100, -100, -100, 13, 106])
        self.assertEqual(batch.decoder_position_ids[1].tolist(), [0, 0, 0, 1, 2])
        logits = torch.randn(2, 5, 128, requires_grad=True)
        loss = patch_cross_entropy(logits, batch.labels)
        loss.backward()
        assert logits.grad is not None
        self.assertEqual(logits.grad[0, :2].abs().sum().item(), 0.0)
        self.assertGreater(logits.grad[0, 2:].abs().sum().item(), 0.0)

    def test_decoder_only_batch_has_empty_encoder_and_masked_context_prefix(self):
        batch = RepairCollator()([TokenizedRepairExample("baseline", (), (2, 105, 7), (18, 106))])
        self.assertEqual(tuple(batch.encoder_input_ids.shape), (1, 0))
        self.assertEqual(tuple(batch.encoder_attention_mask.shape), (1, 0))
        self.assertEqual(batch.labels[0].tolist(), [-100, -100, 18, 106])

    def test_mixed_encoder_presence_left_pads_safely(self):
        batch = RepairCollator()(
            [
                TokenizedRepairExample("short_bypass", (), (2, 105), (18, 106)),
                TokenizedRepairExample("pair_long", (2, 12, 13), (2, 105), (18, 106)),
            ]
        )
        self.assertEqual(tuple(batch.encoder_input_ids.shape), (2, 3))
        self.assertEqual(batch.encoder_attention_mask[0].tolist(), [False, False, False])
        self.assertEqual(batch.encoder_attention_mask[1].tolist(), [True, True, True])

    def test_encoder_context_excludes_gold_patch(self):
        example = RepairExample(
            "a",
            "synthetic",
            "train",
            RepositorySnapshot("local/toy", "synthetic", (RepositoryFile("main.py", "return 0"),)),
            "Fix the return value.",
            target_patch="DO_NOT_LEAK_GOLD",
        )
        context = format_encoder_context(example)
        self.assertIn("main.py", context)
        self.assertNotIn("DO_NOT_LEAK_GOLD", context)

    def test_all_ignored_loss_is_differentiable_zero(self):
        logits = torch.randn(1, 2, 8, requires_grad=True)
        loss = patch_cross_entropy(logits, torch.full((1, 2), -100))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        assert logits.grad is not None
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_prediction_wire_format_has_no_training_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            write_swebench_predictions([RepairPrediction("a", "model", "diff...")], path)
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            row, {"instance_id": "a", "model_name_or_path": "model", "model_patch": "diff..."}
        )


class TestConstructionEntrypoints(unittest.TestCase):
    def test_checked_in_config_matches_six_model_registry(self):
        self.assertEqual(
            validate_experiment_matrix(ROOT / "configs/experiments.json"), EXPERIMENT_SPECS
        )
        self.assertTrue(
            all(not spec.distillation and not spec.share_ple for spec in EXPERIMENT_SPECS)
        )

    def test_native_backward_matches_embedding_policy_for_all_four_pairs(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(min(previous, 4))
        try:
            for spec in EXPERIMENT_SPECS[:4]:
                with self.subTest(experiment=spec.name):
                    report = run_native_smoke(spec)
                    self.assertEqual(
                        report["word_embeddings_shared"], spec.tie_encoder_decoder_words
                    )
                    self.assertTrue(report["decoder_lm_head_shared"])
                    self.assertFalse(report["pretrained_weights_loaded"])
                    self.assertFalse(report["benchmark_executed"])
                    self.assertGreater(report["parameters_with_nonzero_gradient"], 0)
        finally:
            torch.set_num_threads(previous)

    def test_train_placeholder_fails_with_named_documented_block(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as raised:
            main(["train", "--experiment", get_experiment("BifocalGemma-E2B-E2B").name])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("T02", output.getvalue())
        self.assertIn("docs/BUILD_GUIDE.md#t02", output.getvalue())


if __name__ == "__main__":
    unittest.main()
