"""Meaningful local checks for matched notebook data, accumulation and resume."""

import ast
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import ModelOutput, TokenizedRepairExample
from bifocalgemma.data.offline_sft import (
    editing_record,
    prepare_editing_records,
    split_editing_records,
)
from bifocalgemma.models.v2.native import build_native_tiny_model
from bifocalgemma.training.memory_probe import configure_probe_training
from bifocalgemma.training.objectives import patch_cross_entropy
from bifocalgemma.training.offline_sft import fit_offline_sft, restore_sft_checkpoint


class TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.words = torch.nn.Embedding(128, 8)
        self.head = torch.nn.Linear(8, 128, bias=False)

    def forward(self, batch):  # ty: ignore[invalid-mutable-override]  # nn.Module.forward is annotated Callable.
        logits = self.head(self.words(batch.decoder_input_ids))
        return ModelOutput(logits, patch_cross_entropy(logits, batch.labels))


def examples():
    return [
        TokenizedRepairExample(
            str(i), (2, 3, 4), (2, 105, 7), (18, 106) if i % 2 == 0 else (19, 20, 21, 22, 106)
        )
        for i in range(4)
    ]


class TestOfflineEditing(unittest.TestCase):
    def test_nonterminated_source_diff_applies_exactly(self):
        row = editing_record(
            dict(
                old_file="a.py",
                new_file="a.py",
                old_contents="x = 0",
                new_contents="x = 1",
                instruction="Fix x",
                commit="c1",
            ),
            source="editpackft",
        )
        self.assertIn("No newline at end of file", row["target_patch"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.py"
            path.write_text(row["before"], encoding="utf-8")
            result = subprocess.run(
                ["git", "apply", "-"],
                cwd=temp,
                input=row["target_patch"].encode(),
                capture_output=True,
                env={**os.environ, "GIT_CEILING_DIRECTORIES": str(Path(temp).parent)},
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(path.read_text(), row["after"])

    def test_duplicate_pairs_and_commit_holdout(self):
        rows = [
            dict(
                old_file="a.py",
                new_file="a.py",
                old_contents=f"x={i}\n",
                new_contents=f"x={i + 10}\n",
                instruction="Fix x",
                commit=f"c{i // 2}",
            )
            for i in range(8)
        ]
        rows.append(dict(rows[0]))
        prepared, filtered = prepare_editing_records(rows, source="editpackft")
        self.assertEqual(filtered["duplicate_before_after"], 1)
        train, dev = split_editing_records(prepared, seed=17)
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in dev})


class TestOfflineSFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_token_weighted_accumulation_matches_large_batch(self):
        torch.manual_seed(7)
        first = TinyLossModel()
        second = copy.deepcopy(first)
        with tempfile.TemporaryDirectory() as temp:
            common = dict(
                max_steps=1,
                epochs=1,
                learning_rate=1e-3,
                warmup_steps=0,
                clip_norm=1e9,
                eval_every=10,
                save_every=10,
            )
            a = fit_offline_sft(
                first,
                examples()[:2],
                examples()[:1],
                Path(temp) / "a",
                {},
                batch_size=2,
                accumulation=1,
                **common,
            )
            b = fit_offline_sft(
                second,
                examples()[:2],
                examples()[:1],
                Path(temp) / "b",
                {},
                batch_size=1,
                accumulation=2,
                **common,
            )
        self.assertAlmostEqual(
            a["history"][0]["train_loss"], b["history"][0]["train_loss"], places=6
        )
        for pa, pb in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(pa, pb, atol=2e-6, rtol=2e-6)

    def model(self, index):
        torch.manual_seed(19)
        model = build_native_tiny_model(EXPERIMENT_SPECS[index])
        configure_probe_training(model, rank=2)
        return model

    def test_native_lora_resume_matches_uninterrupted_both_families(self):
        for index in (0, 2):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                continuous = self.model(index)
                staged = self.model(index)
                compat = {"model": EXPERIMENT_SPECS[index].name, "rank": 2, "data": "fixture-v1"}
                common = dict(
                    epochs=3,
                    batch_size=1,
                    accumulation=2,
                    learning_rate=1e-3,
                    warmup_steps=0,
                    eval_every=10,
                    save_every=10,
                )
                full = fit_offline_sft(
                    continuous,
                    examples(),
                    examples()[:1],
                    Path(temp) / "full",
                    compat,
                    max_steps=4,
                    **common,
                )
                half = fit_offline_sft(
                    staged,
                    examples(),
                    examples()[:1],
                    Path(temp) / "staged",
                    compat,
                    max_steps=2,
                    **common,
                )
                restored = self.model(index)
                resumed = fit_offline_sft(
                    restored,
                    examples(),
                    examples()[:1],
                    Path(temp) / "staged",
                    compat,
                    max_steps=4,
                    resume_from=half["checkpoints"][-1],
                    **common,
                )
                self.assertEqual(resumed["global_step"], full["global_step"])
                for key, value in continuous.state_dict().items():
                    torch.testing.assert_close(
                        value, restored.state_dict()[key], atol=1e-6, rtol=1e-6
                    )
                assert restored.encoder is not None
                self.assertEqual(
                    restored.encoder.embed_tokens.weight is restored.decoder.embed_tokens.weight,
                    index == 0,
                )
                self.assertIsNot(
                    restored.encoder.embed_tokens_per_layer.weight,
                    restored.decoder.embed_tokens_per_layer.weight,
                )

    def test_incompatible_resume_and_run_overwrite_rejected(self):
        model = TinyLossModel()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            result = fit_offline_sft(
                model, examples(), examples()[:1], root, {"data": "a"}, max_steps=1
            )
            with self.assertRaises(FileExistsError):
                fit_offline_sft(model, examples(), examples()[:1], root, {"data": "a"}, max_steps=1)
            optimizer = torch.optim.AdamW(model.parameters())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                restore_sft_checkpoint(
                    model, optimizer, result["checkpoints"][-1], {"data": "b"}, len(examples())
                )
            marker = Path(result["checkpoints"][-1]) / "READY.json"
            metadata = json.loads(marker.read_text())
            metadata["manifest.json"] = "invalid"
            marker.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore_sft_checkpoint(
                    model, optimizer, result["checkpoints"][-1], {"data": "a"}, len(examples())
                )


class TestNotebookCellParity(unittest.TestCase):
    def test_cell_count_order_and_contents_are_matched(self):
        root = Path(__file__).resolve().parents[1]
        texts = [
            (root / f"notebooks/legacy/bifocalgemma_train_{family}.py").read_text(encoding="utf-8")
            for family in ("symmetric", "asymmetric", "baseline")
        ]
        cells = [
            [
                ast.get_source_segment(text, node)
                for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef)
            ]
            for text in texts
        ]
        self.assertEqual([len(items) for items in cells], [21, 21, 21])
        self.assertEqual(
            [i for i, group in enumerate(zip(*cells, strict=True)) if len(set(group)) != 1],
            [4],
        )

    def test_hub_upload_controls_and_path(self):
        root = Path(__file__).resolve().parents[1]
        for family in ("symmetric", "asymmetric"):
            text = (root / f"notebooks/legacy/bifocalgemma_train_{family}.py").read_text(encoding="utf-8")
            self.assertIn('HF_REPO_ID = "daruokta/bifocalgemma-sft"', text)
            self.assertIn("HF_UPLOAD_ENABLED = True", text)
            self.assertIn('path_in_repo = f"{FAMILY}/{EXPERIMENT}/{RUN_NAME}"', text)
            self.assertIn("run_recipe_comparison", text)
            self.assertIn("CHECKPOINT_UPLOAD_ENABLED = True", text)
            self.assertIn("SAVE_TOTAL_LIMIT = 2", text)

    def test_notebooks_marimo_graph_has_no_duplicate_definitions(self):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        for family in ("symmetric", "asymmetric"):
            path = root / f"notebooks/legacy/bifocalgemma_train_{family}.py"
            spec = importlib.util.spec_from_file_location(f"notebook_{family}", path)
            self.assertIsNotNone(spec)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            app = getattr(module, "app")
            # Ensure marimo app initializes DAG without MultipleDefinitionError or cycles
            app._maybe_initialize()

    def test_token_role_detection_handles_classic_and_fine_grained_and_nulls(self):
        def parse_role(identity):
            access_token = (identity.get("auth") or {}).get("accessToken") or {}
            role = access_token.get("role")
            if not role:
                fine_grained = access_token.get("fineGrained") or {}
                scoped = fine_grained.get("scoped") or []
                has_write = any(
                    any("write" in str(perm).lower() for perm in scope_item.get("permissions", []))
                    for scope_item in scoped
                    if isinstance(scope_item, dict)
                )
                if has_write:
                    role = "write"
                elif fine_grained:
                    role = "fine-grained"
                else:
                    role = "unknown"
            return role

        self.assertEqual(parse_role({"auth": {"accessToken": {"role": "write"}}}), "write")
        self.assertEqual(parse_role({"auth": {"accessToken": {"role": "read"}}}), "read")
        self.assertEqual(parse_role({"auth": {"accessToken": None}}), "unknown")
        self.assertEqual(parse_role({"auth": None}), "unknown")
        self.assertEqual(parse_role({}), "unknown")
        self.assertEqual(
            parse_role(
                {
                    "auth": {
                        "accessToken": {
                            "fineGrained": {
                                "scoped": [{"permissions": ["repo.write", "repo.read"]}]
                            }
                        }
                    }
                }
            ),
            "write",
        )
        self.assertEqual(
            parse_role(
                {
                    "auth": {
                        "accessToken": {"fineGrained": {"scoped": [{"permissions": ["repo.read"]}]}}
                    }
                }
            ),
            "fine-grained",
        )

    def test_instruction_changes_data_identity(self):
        raw = dict(
            old_file="a.py",
            new_file="a.py",
            old_contents="x = 0\n",
            new_contents="x = 1\n",
            instruction="Fix x",
            commit="same",
        )
        first = editing_record(raw, source="editpackft")
        raw["instruction"] = "Change x to one"
        second = editing_record(raw, source="editpackft")
        self.assertEqual(first["pair_hash"], second["pair_hash"])
        self.assertNotEqual(first["instruction_hash"], second["instruction_hash"])
        self.assertNotEqual(first["instance_id"], second["instance_id"])
