"""Shared schedules bind semantic data and never rebuild or wrap at stage boundaries."""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from bifocalgemma.training.schedule import (
    build_shared_schedule,
    canonical_schedule_records,
    persist_shared_schedule,
)

IDENTITY = {"dataset": "fixture", "revision": "pinned", "split": "train"}


def rows(count: int = 20) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": f"row-{i:05}",
            "source": f"source-{i % 3}",
            "task_type": "repair" if i % 2 else "completion",
            "instruction": f"Implement behavior {i}",
            "context": f"def helper_{i}(): return {i}\n# <MISSING_CODE>",
            "target_text": f"return helper_{i}()",
            "token_lengths": {"encoder": 1000 + (i % 4) * 8000, "target": 8},
        }
        for i in range(count)
    ]


def schedule(raw=None, **kwargs):
    return build_shared_schedule(
        canonical_schedule_records(rows() if raw is None else raw),
        dataset_identity=kwargs.pop("dataset_identity", IDENTITY),
        **kwargs,
    )


def test_deterministic_model_independent_schedule_and_mapping():
    raw = rows(44)
    first = schedule(raw)
    assert first == schedule(raw)
    reversed_schedule = schedule(list(reversed(raw)))
    assert first.ordered_ids == reversed_schedule.ordered_ids
    assert first.signature == reversed_schedule.signature
    assert [raw[i]["instance_id"] for i in first.ordered_indices] == list(first.ordered_ids)
    baseline = copy.deepcopy(raw)
    for item in baseline:
        item["token_lengths"]["baseline_decoder"] = 50000
        item["model_family"] = "decoder_only"
    assert schedule(baseline).signature == first.signature
    assert canonical_schedule_records(baseline) == canonical_schedule_records(raw)
    assert schedule(raw, seed=18).signature != first.signature


def test_all_originals_once_with_disjoint_stage_windows():
    shared = schedule(rows(44408))
    assert shared.total_updates == 11102
    first_half = tuple(index for u in range(5551) for index in shared.window(u))
    second_half = tuple(index for u in range(5551, 11102) for index in shared.window(u))
    assert len(first_half) == len(second_half) == 22204
    assert set(first_half).isdisjoint(second_half)
    assert sorted(first_half + second_half) == list(range(44408))
    assert shared.validate_cursor(5551, example_cursor=22204) == 22204
    assert shared.validate_cursor(11102, example_cursor=44408) == 44408
    with pytest.raises(ValueError, match="wrapping"):
        shared.window(11102)


def test_partial_window_and_cursor_validation():
    shared = schedule(rows(9))
    assert shared.total_updates == 3
    assert tuple(map(len, (shared.window(0), shared.window(1), shared.window(2)))) == (4, 4, 1)
    assert shared.to_manifest()["partial_window_size"] == 1
    assert shared.validate_cursor(3, example_cursor=9) == 9
    invalid_values: tuple[Any, ...] = (-1, 4, 1.0, True)
    for invalid in invalid_values:
        with pytest.raises(ValueError):
            shared.validate_cursor(invalid)
    with pytest.raises(ValueError, match="disagrees"):
        shared.validate_cursor(1, example_cursor=5)


@pytest.mark.parametrize("field", ["instruction", "context", "target_text", "source", "task_type"])
def test_each_semantic_mutation_changes_signature(field):
    raw = rows()
    first = schedule(raw)
    raw[0][field] += " changed"
    assert schedule(raw).signature != first.signature


def test_foundation_context_and_target_patch_are_bound():
    row = {
        "instance_id": "edit",
        "source": "source",
        "instruction": "Fix this",
        "path": "a.py",
        "before": "old",
        "target_patch": "--- a\n+++ b\n",
        "token_lengths": {"encoder": 100},
    }
    first = schedule([row])
    row["path"] = "b.py"
    assert schedule([row]).signature != first.signature
    row["before"] = "new old"
    assert schedule([row]).signature != first.signature


@pytest.mark.parametrize(
    "field", ["instance_id", "instruction", "context", "target_text", "source"]
)
def test_empty_required_fields_rejected(field):
    raw = rows()
    raw[0][field] = " "
    with pytest.raises(ValueError, match="nonempty"):
        canonical_schedule_records(raw)


def test_invalid_cohorts_and_descriptor_lengths_rejected():
    raw = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        canonical_schedule_records(raw + [raw[0]])
    descriptors = canonical_schedule_records(raw)
    with pytest.raises(ValueError, match="Duplicate"):
        build_shared_schedule(descriptors + [descriptors[0]], dataset_identity=IDENTITY)
    with pytest.raises(ValueError, match="empty training"):
        schedule([])
    with pytest.raises(ValueError, match="identity"):
        schedule(dataset_identity={})
    del raw[0]["token_lengths"]["encoder"]
    with pytest.raises(ValueError, match="encoder tokens"):
        canonical_schedule_records(raw)
    descriptors[0]["length_bin"] = "incorrect"
    with pytest.raises(ValueError, match="length bin"):
        build_shared_schedule(descriptors, dataset_identity=IDENTITY)


def test_schedule_is_immutable_and_manifest_contains_no_source_text():
    shared = schedule()
    with pytest.raises(FrozenInstanceError):
        setattr(shared, "seed", 19)
    with pytest.raises(FrozenInstanceError):
        setattr(shared.records[0], "source", "changed")
    manifest = shared.to_manifest()
    manifest["ordered_examples"][0]["instance_id"] = "mutated"
    manifest["dataset_identity"]["revision"] = "changed"
    assert shared.to_manifest()["dataset_identity"] == IDENTITY
    serialized = json.dumps(shared.to_manifest())
    assert "def helper_" not in serialized
    assert "Implement behavior" not in serialized
    assert "return helper_" not in serialized


def test_atomic_manifest_reuse_and_fail_closed_mismatch(tmp_path):
    path = tmp_path / "nested" / "schedule.json"
    shared = schedule()
    persist_shared_schedule(path, shared)
    original = path.read_bytes()
    persist_shared_schedule(path, shared)
    assert path.read_bytes() == original
    for changed in (
        schedule(seed=19),
        schedule(effective_batch=2),
        schedule(dataset_identity={**IDENTITY, "revision": "new"}),
    ):
        with pytest.raises(ValueError, match="does not match"):
            persist_shared_schedule(path, changed)
        assert path.read_bytes() == original
    assert [item.name for item in path.parent.iterdir()] == ["schedule.json"]


@pytest.mark.parametrize("change", ["target", "signature", "schema", "order", "invalid_json"])
def test_manifest_tampering_rejected_without_rewriting(tmp_path, change):
    path = tmp_path / "schedule.json"
    shared = schedule()
    manifest = shared.to_manifest()
    if change == "target":
        manifest["ordered_examples"][0]["target_sha256"] = "0" * 64
    elif change == "signature":
        manifest["signature"] = "0" * 64
    elif change == "schema":
        manifest["schema_version"] = "future"
    elif change == "order":
        manifest["ordered_examples"].reverse()
    path.write_text("{" if change == "invalid_json" else json.dumps(manifest), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError):
        persist_shared_schedule(path, shared)
    assert path.read_bytes() == original
