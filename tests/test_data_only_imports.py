"""Static data processing must not initialize the tensor runtime."""

import os
import subprocess
import sys
from pathlib import Path


def test_static_helpers_and_official_tokenizer_class_do_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from bifocalgemma.contracts import RepositorySnapshot
from bifocalgemma.data.repository_corpus import ArchiveSnapshotStore
from bifocalgemma.data.releases import build_release
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.training.repository_sft import tokenize_repository_record
from bifocalgemma.training.curated_sft import prepare_curated_training
from bifocalgemma.data.packed_tokens import build_packed_cache
from bifocalgemma.evaluation.context import prompt_ids
assert 'torch' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "USE_TORCH": "0"},
        check=True,
        capture_output=True,
        text=True,
    )


def test_packed_cache_can_be_built_without_tensor_runtime(tmp_path):
    from bifocalgemma.contracts import TokenizedRepairExample
    from bifocalgemma.data.packed_tokens import build_packed_cache, open_packed_cache

    expected = {
        "train": [TokenizedRepairExample("train", (2, 262143), (2, 1001), (99999, 106))],
        "dev": [TokenizedRepairExample("dev", (2, 262000), (2, 1002), (88888, 106))],
    }
    by_id = {x.instance_id: x for items in expected.values() for x in items}
    rows = {
        split: [{"instance_id": x.instance_id} for x in items] for split, items in expected.items()
    }
    datasets, receipt = build_packed_cache(
        tmp_path / "cache", {"exact": True}, rows, lambda row: by_id[row["instance_id"]]
    )
    assert receipt["token_ids_changed"] is False
    for split, dataset in datasets.items():
        assert list(dataset) == expected[split]
        dataset.close()
    datasets, _ = open_packed_cache(tmp_path / "cache", {"exact": True})
    for split, dataset in datasets.items():
        assert list(dataset) == expected[split]
        dataset.close()
