"""Real cached tokenizers (opt-in path) and portable synthetic patch fixtures."""

import copy
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import RepairExample, RepositoryFile, RepositorySnapshot
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.swesmith import SWESmithRecord, normalize_swesmith_record
from bifocalgemma.data.tokenization import (
    format_encoder_context,
    load_verified_tokenizer,
    tokenize_repair_example,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_records():
    for row in json.loads((FIXTURES / "swesmith_local.synthetic.json").read_text(encoding="utf-8")):
        yield SWESmithRecord(
            row["instance_id"],
            row["dataset_revision"],
            row["buggy_ref"],
            row["buggy_snapshot_sha256"],
            RepositorySnapshot(
                row["repo"], row["buggy_ref"], tuple(RepositoryFile(**f) for f in row["files"])
            ),
            row["issue"],
            row["bug_patch"],
        )


class TestSWESmithContract(unittest.TestCase):
    def test_one_and_multiple_files_reverse_bug_patch(self):
        for record in fixture_records():
            result = normalize_swesmith_record(record, split="train")
            self.assertNotEqual(result.example.target_patch, record.bug_patch)
            self.assertNotEqual(result.buggy_snapshot_sha256, result.fixed_snapshot_sha256)
            self.assertIn("not_official_tests", result.validation)
            assert result.example.target_patch is not None
            context = format_encoder_context(result.example)
            self.assertNotIn("diff --git", context)
            if record.instance_id.endswith("single"):
                self.assertIn("-    return a - b", result.example.target_patch)
                self.assertIn("+    return a + b", result.example.target_patch)
            else:
                self.assertEqual(result.example.target_patch.count("diff --git"), 2)

    def test_wrong_snapshot_ref_hash_and_patch_rejected(self):
        record = next(fixture_records())
        for changed in [
            replace(record, buggy_ref="wrong"),
            replace(record, buggy_snapshot_sha256="wrong"),
            replace(record, bug_patch=record.bug_patch.replace("a - b", "a / b")),
            replace(record, bug_patch=record.bug_patch.replace("maths.py", "../outside.py")),
        ]:
            with self.subTest(record=changed.instance_id), self.assertRaises(ValueError):
                normalize_swesmith_record(changed, split="train")

    def test_does_not_mutate_input_record(self):
        record = next(fixture_records())
        original = copy.deepcopy(record)
        normalize_swesmith_record(record, split="train")
        self.assertEqual(record, original)


class TestTokenizerVerification(unittest.TestCase):
    def test_missing_and_tampered_asset_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = EXPERIMENT_SPECS[0].decoder
            with self.assertRaises(FileNotFoundError):
                load_verified_tokenizer(ref, root)
            folder = root / "raw" / ref.model_id.replace("/", "__")
            folder.mkdir(parents=True)
            (folder / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_verified_tokenizer(ref, root)
            with self.assertRaisesRegex(ValueError, "revision"):
                load_verified_tokenizer(replace(ref, revision="unverified"), root)


@unittest.skipUnless(
    os.environ.get("BIFOCALGEMMA_TOKENIZER_CACHE"),
    "Set BIFOCALGEMMA_TOKENIZER_CACHE to the audited offline cache.",
)
class TestRealOfflineTokenizers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = Path(os.environ["BIFOCALGEMMA_TOKENIZER_CACHE"])
        cls.example = RepairExample(
            "unicode",
            "synthetic",
            "train",
            RepositorySnapshot(
                "local/toy",
                "buggy",
                (RepositoryFile("main.py", "def greet():\n\treturn 'selamat pagi ☀'\n"),),
            ),
            "Fix the greeting.",
            "diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py\n@@ -1,2 +1,2 @@\n def greet():\n-\treturn 'selamat pagi ☀'\n+\treturn 'halo 世界'\n",
        )

    def test_six_models_roundtrip_boundaries_no_leak_and_shift(self):
        for spec in EXPERIMENT_SPECS:
            with self.subTest(spec=spec.name):
                item = tokenize_repair_example(self.example, spec, self.cache)
                decoder, _ = load_verified_tokenizer(spec.decoder, self.cache)
                target = decoder.decode(item.target_ids, skip_special_tokens=False)
                assert self.example.target_patch is not None
                self.assertIn(self.example.target_patch.rstrip(), target)
                self.assertEqual(item.target_ids[-1], 106)
                self.assertEqual(item.decoder_prefix_ids.count(2), 1)
                self.assertNotIn("halo 世界", decoder.decode(item.decoder_prefix_ids))
                self.assertNotIn("halo 世界", decoder.decode(item.encoder_ids))
                if spec.encoder:
                    self.assertEqual(item.encoder_ids.count(2), 1)
                    self.assertIn("\treturn 'selamat pagi ☀'", decoder.decode(item.encoder_ids))
                else:
                    self.assertEqual(item.encoder_ids, ())
                batch = RepairCollator()([item])
                start = len(item.decoder_prefix_ids) - 1
                self.assertEqual(batch.labels[0, start].item(), item.target_ids[0])
                self.assertTrue((batch.labels[0, :start] == -100).all())
                self.assertGreater(max(item.decoder_prefix_ids + item.target_ids), 127)
                self.assertFalse(item.metadata["truncated"])

    def test_overflow_rejected_and_target_required(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            tokenize_repair_example(
                self.example, EXPERIMENT_SPECS[0], self.cache, max_encoder_tokens=2
            )
        with self.assertRaisesRegex(ValueError, "budget"):
            tokenize_repair_example(
                self.example, EXPERIMENT_SPECS[4], self.cache, max_decoder_tokens=2
            )
        with self.assertRaisesRegex(ValueError, "target"):
            tokenize_repair_example(
                replace(self.example, target_patch=None), EXPERIMENT_SPECS[0], self.cache
            )
