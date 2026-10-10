"""Manifest schema boundary exercised by the real standalone preparation function."""

import ast
import copy
import hashlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.curation import export_pool
from bifocalgemma.data.offline_sft import (
    editing_manifest,
    editing_record,
    load_curated_pool,
    repository_editing_record,
    stable_fingerprint,
)


def repository_row(index=0):
    return {
        "record_type": "repository_patch_v1",
        "role": "train",
        "instance_id": f"repository-{index}",
        "source": "nebius/SWE-rebench-V2",
        "source_revision": "a" * 40,
        "group": f"owner/repo{index}",
        "commit": f"commit{index}",
        "repos": f"owner/repo{index}",
        "instruction": "  Fix the greeting.  ",
        "context": "### File: helper.py\ndef greet():\n    return '☀'\n",
        "snapshot_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
        "context_selection": {"gold_used": False},
        "license": "mit",
        "target_patch": "diff --git a/helper.py b/helper.py\n",
        "before": "public context",
        "after": "",
        "path": "repository_context",
        "pool_split": "train" if index == 0 else "dev",
    }


def notebook_function(name):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (root / "notebooks/bifocalgemma_train_symmetric.py").read_text(encoding="utf-8")
    )
    return next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_raw_repository_manifest_derives_hashes_without_mutation():
    row = repository_row()
    original = copy.deepcopy(row)
    result = editing_manifest([row])[0]
    assert (
        result["pair_hash"]
        == hashlib.sha256(
            (row["snapshot_sha256"] + "\0" + row["target_patch"]).encode()
        ).hexdigest()
    )
    assert (
        result["instruction_hash"]
        == hashlib.sha256(row["instruction"].strip().encode()).hexdigest()
    )
    assert result["instance_id"] == row["instance_id"]
    assert row == original
    assert "pair_hash" not in row and "instruction_hash" not in row


def test_raw_and_normalized_repository_manifests_are_identical():
    row = repository_row()
    assert editing_manifest([row]) == editing_manifest([repository_editing_record(row)])
    for changed in (
        {**row, "target_patch": row["target_patch"] + "+hello\n"},
        {**row, "snapshot_sha256": "b" * 64},
    ):
        assert stable_fingerprint(editing_manifest([row])) != stable_fingerprint(
            editing_manifest([changed])
        )


@pytest.mark.parametrize("key", ["pair_hash", "instruction_hash"])
def test_conflicting_repository_hashes_are_rejected(key):
    with pytest.raises(ValueError, match="repository_manifest_hash_mismatch:" + key):
        editing_manifest([{**repository_row(), key: "invalid"}])


def test_repository_eval_is_rejected():
    with pytest.raises(ValueError, match="evaluation_record"):
        editing_manifest([{**repository_row(), "role": "eval"}])


def test_legacy_paired_manifest_remains_byte_identical():
    row = editing_record(
        {
            "old_file": "a.py",
            "new_file": "a.py",
            "old_contents": "x = 0\n",
            "new_contents": "x = 1\n",
            "instruction": "Fix x",
            "commit": "abc",
            "repos": "owner/repo",
            "license": "mit",
        },
        source="editpackft",
    )
    fields = (
        "instance_id",
        "group",
        "pair_hash",
        "instruction_hash",
        "path",
        "source",
        "commit",
        "repos",
        "license",
    )
    expected = [{key: row[key] for key in fields}]
    assert editing_manifest([row]) == expected
    assert stable_fingerprint(editing_manifest([row])) == stable_fingerprint(expected)


@pytest.mark.parametrize("index", range(4))
def test_generated_preparation_consumes_frozen_raw_repository_pool(tmp_path, index):
    records = [repository_row(0), repository_row(1)]
    root = tmp_path / "pool"
    pool = export_pool(records, root, {"split": {"policy": "frozen_fixture_split"}})
    original_bytes = {p.name: p.read_bytes() for p in root.iterdir()}
    spec = EXPERIMENT_SPECS[index]
    # Tokenization is a separate tested block. Here the actual preparation function
    # executes all loading/retention/manifest steps using lightweight token fixtures.
    namespace: dict[str, Any] = {
        "Counter": Counter,
        "load_curated_pool": load_curated_pool,
        "repository_editing_record": repository_editing_record,
        "stable_fingerprint": stable_fingerprint,
        "DATA_SOURCES": {},
        "CHECKPOINTS": {
            ref.size: {"encoder": asdict(ref), "decoder": asdict(ref)}
            for ref in (spec.encoder, spec.decoder)
            if ref is not None
        },
        "verified_assets": lambda *args: {"config.json": tmp_path / "config.json"},
        "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        "tokenize_editing_row": lambda row, *args: TokenizedRepairExample(
            row["instance_id"], (2, 3), (2, 4), (5, 106)
        ),
    }
    for name in ("editing_manifest", "prepare_training_data"):
        exec(
            compile(
                ast.Module(body=[notebook_function(name)], type_ignores=[]),
                "<generated-data-helper>",
                "exec",
            ),
            namespace,
        )
    train, dev, manifest = namespace["prepare_training_data"](
        spec, "curated_pool", root, tmp_path, 2, 1, 1, 0.1, 8192, 2048, 17
    )
    assert [row.instance_id for row in train] == pool["record_ids"]["train"]
    assert [row.instance_id for row in dev] == pool["record_ids"]["dev"]
    assert manifest["curated_pool_fingerprint"] == pool["fingerprint"]
    assert manifest["train"][0]["pair_hash"]
    assert manifest["dev"][0]["instruction_hash"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == original_bytes
