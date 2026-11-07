"""Small local fixtures: real checkpoint/model runs happen on the user's Molab GPU."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.v2.checkpoint_transfer import transfer_text_checkpoint
from bifocalgemma.models.v2.native import build_native_tiny_model
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


def test_exact_synthetic_lengths_preserve_shift_and_closing_turn():
    from bifocalgemma.training.memory_probe import synthetic_stress_target

    prefix = (2, 7, 8)
    target = (9, 10, 106)
    for length in (2048, 4096, 8192, 16896):
        filled = synthetic_stress_target(len(prefix), target, length, (11, 12))
        assert filled[:2] == target[:2] and filled[-1] == 106
        batch = RepairCollator()([TokenizedRepairExample("stress", (), prefix, filled)])
        assert batch.decoder_input_ids.shape == (1, length)
        assert batch.labels[0, -1].item() == 106
        assert (batch.labels[0, : len(prefix) - 1] == -100).all()
    assert synthetic_stress_target(3, target, 5, (11,)) == target
    for args in ((3, target, 4, (11,)), (3, (9,), 8, (11,)), (3, target, 8, ())):
        try:
            synthetic_stress_target(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid/truncating stress configuration was accepted")


def test_length_grid_matches_baseline_budget_and_stops_after_failure():
    from bifocalgemma.training.memory_probe import run_length_grid

    calls = []

    def probe(name, encoder, decoder):
        calls.append((name, encoder, decoder))
        return {"status": "OOM" if name.startswith("BifocalGemma") and encoder == 4096 else "PASS"}

    grid = ((2048, 2048), (4096, 4096), (8192, 8192))
    progress = []
    result = run_length_grid(
        ("BifocalGemma-E2B-E2B", "Gemma-4-E4B-IT"), grid, probe, progress=progress.append
    )
    assert progress == result["rows"]
    assert len(calls) == 5
    assert calls[-1] == ("Gemma-4-E4B-IT", 8192, 16896)
    same_encoder = run_length_grid(("Gemma-4-E2B-IT",), ((2048, 2048), (2048, 4096)), probe)
    assert same_encoder["all_configurations_passed_maximum"]
    assert result["rows"][1]["larger_boundaries_not_attempted"] == [(8192, 8192)]
    assert not result["all_configurations_passed_maximum"]
    assert result["one_resident_model_at_a_time"]
    for invalid in (((4096, 4096), (2048, 2048)), ((1, 1),), ()):
        try:
            run_length_grid(("Gemma-4-E4B-IT",), invalid, probe)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid length grid was accepted")
    assert run_length_grid(("Gemma-4-E2B-IT",), ((2048, 2048),), probe)[
        "all_configurations_passed_maximum"
    ]


def test_notebook_stress_helpers_match_shared_library():
    import ast

    def definitions(path):
        nodes = ast.walk(ast.parse(Path(path).read_text(encoding="utf-8")))
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in nodes
            if isinstance(node, ast.FunctionDef)
            and node.name in {"synthetic_stress_target", "run_length_grid"}
        }

    library = definitions("src/bifocalgemma/training/memory_probe.py")
    assert library == definitions("notebooks/bifocalgemma_pretrained_probe.py")


def test_notebook_sweep_persists_results_per_case(tmp_path):
    import ast
    import copy
    import json
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from bifocalgemma.training.memory_probe import run_length_grid

    tree = ast.parse(Path("notebooks/bifocalgemma_pretrained_probe.py").read_text(encoding="utf-8"))
    cell = copy.deepcopy(
        next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and any(a.arg == "RUN_LENGTH_SWEEP" for a in n.args.args)
        )
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), "probe_cell", "exec"), namespace)
    observed = []

    def probe(*args, **kwargs):
        assert kwargs["stress_decoder"]
        files = list(tmp_path.glob("length_sweep_*.json"))
        if files:
            observed.append(json.loads(files[0].read_text(encoding="utf-8")))
        return {"status": "PASS", "data": {"decoder_tokens_actual": args[6]}}

    mo = SimpleNamespace(
        md=lambda text: text,
        vstack=lambda rows: rows,
        ui=SimpleNamespace(table=lambda rows, **kwargs: rows),
    )
    result, path = namespace["_"](
        RUN_PRETRAINED=True,
        RUN_LENGTH_SWEEP=True,
        LENGTH_EXPERIMENTS=("Gemma-4-E2B-IT",),
        LENGTH_GRID=((2048, 2048), (4096, 4096)),
        TRAINING_POLICY="lora",
        LORA_RANK=16,
        STEPS=2,
        CACHE_DIR="cache",
        BATCH_SIZE=1,
        HEADROOM_GIB=8,
        SEED=17,
        device="cuda",
        run_single_pretrained=probe,
        run_length_grid=run_length_grid,
        single_result={},
        mo=mo,
        Path=Path,
        OUTPUT_DIR=tmp_path,
        datetime=datetime,
        timezone=timezone,
        json=json,
    )
    assert observed[0]["status"] == "IN_PROGRESS" and len(observed[0]["rows"]) == 1
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["status"] == "FINISHED" and len(final["rows"]) == 2
    assert result["all_configurations_passed_maximum"]
