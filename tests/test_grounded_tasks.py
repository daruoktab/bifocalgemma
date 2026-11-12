"""Grounded proxy labels must be reconstructible from training-only evidence."""

import copy

import pytest

from bifocalgemma.data.context_tasks import normalize_context_task
from bifocalgemma.data.grounded_tasks import TASKS, generate_grounded_tasks, validate_grounded_task


def parent():
    return {
        "role": "train",
        "record_type": "context_sft_v1",
        "pool_split": "train",
        "instance_id": "fixture-parent",
        "group": "fixture-repo",
        "task_type": "repair",
        "instruction": "Fix the code",
        "context": "def alpha(value):\n    return value + 1\n\ndef beta(value):\n    return value * 2\n\ndef gamma(value):\n    return value - 1\n",
        "target_text": "original patch must never be used",
        "source": "nebius/SWE-rebench-V2",
        "source_key": "swe_rebench_v2",
        "source_revision": "pinned",
        "snapshot_sha256": "a" * 64,
        "commit": "pinned-commit",
        "repos": "example/repo",
        "context_selection": {
            "gold_used": False,
            "windows": [{"path": "a.py", "start_line": 1, "end_line": 8}],
        },
        "license": "mit",
        "context_source_code_license_evidence": [
            {"path": "a.py", "licenses": [{"license": "mit", "sha256": "b" * 64}]}
        ],
    }


def test_grounded_generation_answerability_provenance_and_target_independence():
    raw = parent()
    before = copy.deepcopy(raw)
    children = generate_grounded_tasks(raw)
    assert {row["task_type"] for row in children} == set(TASKS)
    assert raw == before
    assert children == generate_grounded_tasks(raw)
    raw["target_text"] = "A completely different patch"
    assert generate_grounded_tasks(raw) == children
    for row in children:
        normalize_context_task(row)
        assert row["pool_split"] == "train" and row["group"] == raw["group"]
        assert row["instruction"] in row["context"]  # Baseline sees the same question.
        assert row["derivation"]["parent_id"] == raw["instance_id"]
        if row["task_type"] == "source_reconstruction":
            assert row["target_text"] not in row["context"]


@pytest.mark.parametrize("mutation", ["target_text", "context", "instruction", "instance_id"])
def test_grounded_mutation_rejected(mutation):
    child = generate_grounded_tasks(parent())[0]
    child[mutation] += " changed"
    with pytest.raises(ValueError, match="grounded_"):
        validate_grounded_task(child)


def test_evaluation_and_unlicensed_sources_rejected():
    row = parent()
    row["role"] = "eval"
    with pytest.raises(ValueError, match="not_training"):
        generate_grounded_tasks(row)
    row = parent()
    row["context_source_code_license_evidence"][0]["licenses"][0]["license"] = "unknown"
    with pytest.raises(ValueError, match="not_admitted"):
        generate_grounded_tasks(row)
