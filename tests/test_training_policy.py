"""Policy group coverage, scheduling and atomic v2 resume validation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.models.native import build_native_tiny_model
from bifocalgemma.training.offline_sft import file_digest, fit_offline_sft
from bifocalgemma.training.policy import (
    build_training_groups,
    configure_policy_model,
    default_training_profiles,
    optimizer_group_manifest,
    update_policy_learning_rates,
)
from tests.test_offline_sft_training import examples


class TestTrainingPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def policy(self, index):
        family = "symmetric" if index == 0 else "asymmetric"
        policy = default_training_profiles()[family]
        policy.update(rank=2, alpha=4)
        policy["groups"]["encoder"].update(lr=1e-3, warmup_steps=2)
        policy["groups"]["decoder"].update(lr=2e-3, warmup_steps=4)
        if family == "asymmetric":
            policy["groups"]["alignment"].update(lr=5e-4, warmup_steps=0)
        return policy

    def model(self, index, policy):
        torch.manual_seed(19)
        model = build_native_tiny_model(EXPERIMENT_SPECS[index])
        configure_policy_model(model, policy)
        return model

    def test_coverage_frozen_tables_aliases_and_family_alignment(self):
        for index in (0, 2):
            policy = self.policy(index)
            model = self.model(index, policy)
            # Extra alias does not create a second optimizer entry.
            attention = model.decoder.layers[0].self_attn.q_proj
            attention.alias_A = attention.lora_A
            groups = build_training_groups(model, policy)
            ids = [id(p) for group in groups for p in group["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
            self.assertEqual(
                [g["name"] for g in groups],
                ["encoder", "decoder"] if index == 0 else ["encoder", "decoder", "alignment"],
            )
            assert model.encoder is not None
            self.assertFalse(model.encoder.embed_tokens.weight.requires_grad)
            self.assertFalse(model.decoder.embed_tokens_per_layer.weight.requires_grad)
            self.assertTrue(all(p.requires_grad for g in groups for p in g["params"]))

    def test_per_group_lr_and_warmup_are_independent(self):
        policy = self.policy(2)
        model = self.model(2, policy)
        groups = build_training_groups(model, policy)
        optimizer = torch.optim.AdamW(groups)
        signature = optimizer_group_manifest(groups)
        first = update_policy_learning_rates(optimizer, 0)
        self.assertEqual(first, {"encoder": 5e-4, "decoder": 5e-4, "alignment": 5e-4})
        last = update_policy_learning_rates(optimizer, 3)
        self.assertEqual(last, {"encoder": 1e-3, "decoder": 2e-3, "alignment": 5e-4})
        self.assertEqual(optimizer_group_manifest(optimizer.param_groups), signature)

    def test_v2_resume_matches_uninterrupted_for_both_families(self):
        for index in (0, 2):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                policy = self.policy(index)
                full_model = self.model(index, policy)
                staged = self.model(index, policy)
                common = dict(
                    epochs=3,
                    batch_size=1,
                    accumulation=2,
                    eval_every=10,
                    save_every=10,
                    training_policy=policy,
                )
                full = fit_offline_sft(
                    full_model,
                    examples(),
                    examples()[:1],
                    Path(temp) / "full",
                    {},
                    max_steps=4,
                    **common,
                )
                half = fit_offline_sft(
                    staged,
                    examples(),
                    examples()[:1],
                    Path(temp) / "staged",
                    {},
                    max_steps=2,
                    **common,
                )
                restored = self.model(index, policy)
                resumed = fit_offline_sft(
                    restored,
                    examples(),
                    examples()[:1],
                    Path(temp) / "staged",
                    {},
                    max_steps=4,
                    resume_from=half["checkpoints"][-1],
                    **common,
                )
                self.assertEqual(resumed["checkpoint_format"], "bifocalgemma_notebook_sft_v2")
                self.assertNotIn("learning_rate", resumed["history"][0])
                self.assertEqual(resumed["history"][0]["learning_rates"]["encoder"], 5e-4)
                self.assertEqual(resumed["history"][3]["learning_rates"]["decoder"], 2e-3)

                # Wall-clock diagnostics vary; every training-semantic field must match.
                def semantic(rows):
                    return [
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "step_seconds",
                                "eval_seconds",
                                "forward_seconds",
                                "backward_seconds",
                                "optimizer_seconds",
                                "target_tokens_per_second",
                                "eta_train_seconds",
                            }
                            and not k.startswith(("ram_", "gpu_"))
                        }
                        for row in rows
                    ]

                self.assertEqual(semantic(full["history"]), semantic(resumed["history"]))
                for key, tensor in full_model.state_dict().items():
                    torch.testing.assert_close(
                        tensor, restored.state_dict()[key], atol=1e-6, rtol=1e-6
                    )

    def test_legacy_v1_and_changed_policy_rejected_before_mutation(self):
        policy = self.policy(0)
        with tempfile.TemporaryDirectory() as temp:
            model = self.model(0, policy)
            legacy = fit_offline_sft(
                model, examples(), examples()[:1], Path(temp) / "legacy", {}, max_steps=1
            )
            restored = self.model(0, policy)
            original = {k: v.clone() for k, v in restored.state_dict().items()}
            config = (Path(temp) / "legacy/run_config.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "requires v2"):
                fit_offline_sft(
                    restored,
                    examples(),
                    examples()[:1],
                    Path(temp) / "legacy",
                    {},
                    max_steps=2,
                    resume_from=legacy["checkpoints"][-1],
                    training_policy=policy,
                )
            self.assertEqual((Path(temp) / "legacy/run_config.json").read_bytes(), config)
            for key, tensor in original.items():
                torch.testing.assert_close(tensor, restored.state_dict()[key])
            v2 = fit_offline_sft(
                restored,
                examples(),
                examples()[:1],
                Path(temp) / "v2",
                {},
                max_steps=1,
                training_policy=policy,
            )
            changed = copy.deepcopy(policy)
            changed["groups"]["decoder"]["lr"] *= 2
            with self.assertRaisesRegex(ValueError, "mismatch"):
                fit_offline_sft(
                    self.model(0, changed),
                    examples(),
                    examples()[:1],
                    Path(temp) / "v2",
                    {},
                    max_steps=2,
                    resume_from=v2["checkpoints"][-1],
                    training_policy=changed,
                )

    def test_changed_group_order_rejected_before_copy(self):
        policy = self.policy(0)
        with tempfile.TemporaryDirectory() as temp:
            result = fit_offline_sft(
                self.model(0, policy),
                examples(),
                examples()[:1],
                Path(temp) / "run",
                {},
                max_steps=1,
                training_policy=policy,
            )
            checkpoint = Path(result["checkpoints"][-1])
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            manifest["optimizer_groups"].reverse()
            (checkpoint / "manifest.json").write_text(json.dumps(manifest))
            marker = json.loads((checkpoint / "READY.json").read_text())
            marker["manifest.json"] = file_digest(checkpoint / "manifest.json")
            (checkpoint / "READY.json").write_text(json.dumps(marker))
            model = self.model(0, policy)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "group order"):
                fit_offline_sft(
                    model,
                    examples(),
                    examples()[:1],
                    Path(temp) / "run",
                    {},
                    max_steps=2,
                    resume_from=checkpoint,
                    training_policy=policy,
                )
            for key, tensor in before.items():
                torch.testing.assert_close(tensor, model.state_dict()[key])
