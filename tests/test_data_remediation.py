"""V2 semantic dedup, structural split and provenance boundaries."""

import copy
import json

import pytest

from bifocalgemma.data.curation import code_signature, split_connected_records
from bifocalgemma.data.offline_sft import editing_record
from bifocalgemma.data.remediation import audit_rows, inspect_structure, remediate_records


def record(i, before=None, after=None, source="editpackft", commit=None, repo=None):
    before = before or f"def fn_{i}():\n    return {i}\n"
    after = after or f"def fn_{i}():\n    return {i + 10}\n"
    row = editing_record(
        {
            "old_contents": before,
            "new_contents": after,
            "old_file": "a.py",
            "new_file": "a.py",
            "instruction": "Repair.",
            "commit": commit or f"commit-{i}",
            "repos": repo or f"owner/repo-{i}",
            "license": "mit",
        },
        source="editpackft",
    )
    row.update(
        instance_id=f"id-{i}",
        source_key=source,
        source=source,
        role="train",
        canonical_repos=[repo or f"owner/repo-{i}"],
        code_license="mit",
        dataset_license="mit",
    )
    return row


def test_indentation_changes_are_not_identical():
    before = "def f(x):\n    if x:\n        x += 1\n    return x\n"
    after = "def f(x):\n    if x:\n        x += 1\n        return x\n"
    assert code_signature(before) != code_signature(after)
    assert inspect_structure(before)["core"] != inspect_structure(after)["core"]
    assert code_signature(before.replace("    ", "  ")) == code_signature(before)


def test_unparseable_fallback_preserves_whitespace():
    assert code_signature("def f():\n    (") != code_signature("def f():\n        (")


def test_event_variants_choose_documented_export_and_merge_lineage():
    a = record(1, commit="same")
    b = record(
        2,
        '"""Upstream license header."""\n' + a["before"],
        '"""Upstream license header."""\n' + a["after"],
        source="commitpackft_python",
        commit="same",
    )
    original = copy.deepcopy([a, b])
    selected, ledger, counts = remediate_records([a, b])
    assert len(selected) == 1 and selected[0]["instance_id"] == "id-2"
    assert selected[0]["canonical_repos"] == ["owner/repo-1", "owner/repo-2"]
    assert selected[0]["duplicate_source_records"][0]["instance_id"] == "id-1"
    assert counts == {"duplicate_git_or_seed_event": 1}
    assert ledger[0]["retained_id"] == "id-2"
    assert [a, b] == original


def test_conflicting_event_variants_are_all_quarantined():
    selected, _, counts = remediate_records([record(1, commit="same"), record(2, commit="same")])
    assert not selected and counts == {"ambiguous_git_event_variants": 2}


def test_same_input_different_targets_is_not_silently_chosen():
    a = record(1)
    b = record(2, a["before"], a["after"].replace("11", "999"))
    selected, _, counts = remediate_records([a, b])
    assert not selected and counts == {"ambiguous_visible_input": 2}


def test_doc_only_and_warning_targets_are_removed():
    a = record(1, "x=1\n", '"""New docs."""\nx=1\n')
    b = record(2, 'value = ""\n', 'value = "\\q"\n')
    selected, _, counts = remediate_records([a, b])
    assert not selected
    assert counts == {"documentation_or_format_only": 1, "after_syntax_warning": 1}


def test_synthetic_seed_is_not_claimed_as_git_or_code_license():
    row = remediate_records([record(1, source="nextcoder")])[0][0]
    assert row["git_commit"] is None and row["seed_program_hash"] == row["commit"]
    assert row["code_license"] == row["license"] == "unknown"
    assert row["dataset_license"] == "mit" and not row["functional_quality_verified"]


def test_structure_before_after_and_discarded_aliases_are_indivisible():
    a = record(1, "x=1\n", "x=2\n")
    b = record(2, "x=2\n", "x=3\n")
    c = record(3, "y=4\n", "y=5\n")
    selected, _, _ = remediate_records([a, b, c])
    train, dev, _ = split_connected_records(selected)
    groups = {r["instance_id"]: r["pool_split"] for r in train + dev}
    assert groups["id-1"] == groups["id-2"]
    assert audit_rows({"train": train, "dev": dev})["unique_ids"] == 3


def test_heldout_role_rejected():
    row = record(1)
    row["role"] = "eval"
    with pytest.raises(ValueError, match="evaluation_record"):
        remediate_records([row])


def test_view_record_json_is_lossless_and_gsm_shots_are_explicit(tmp_path):
    import pyarrow.parquet as pq

    from bifocalgemma.data.releases import write_view

    row = {
        "instance_id": "g1",
        "role": "eval",
        "benchmark": "gsm8k",
        "question": "Unicode café: 1 + 1?",
        "context": "Shared demonstrations + question",
    }
    path = tmp_path / "view.parquet"
    write_view(path, [row], training=False, shots=8, protocol={"tools": False})
    view = pq.read_table(path).to_pylist()[0]
    assert json.loads(view["record_json"]) == row
    assert view["question"] == row["question"] and view["shots"] == "8"
    assert json.loads(view["protocol_json"]) == {"tools": False}
    assert "target_patch" not in view and "answer" not in json.loads(view["record_json"])


def test_index_with_legacy_normalization_is_rejected(tmp_path):
    import sqlite3

    from bifocalgemma.data.curation import CodeOverlapIndex

    path = tmp_path / "legacy.sqlite"
    writer = CodeOverlapIndex(storage_path=path)
    writer.add("one", "x=1\n")
    writer.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE metadata")
    with pytest.raises(ValueError, match="normalization_mismatch"):
        CodeOverlapIndex(storage_path=path, read_only=True)


def test_canonical_git_patch_batch_including_missing_final_newline():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "release_auditor", Path(__file__).parents[1] / "tools/audit_dataset_release.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    examples = [
        record(1, "x=1", "x=2"),
        record(2, "x=1\n", "x=2"),
        record(3, "x=1\ry=2\n", "x=1\ny=3\n"),
        record(4, "# copyright \u2028 header\nx=1\n", "# copyright \u2028 header\nx=2\n"),
        record(5, "def f():\n\f    return 1\n", "def f():\n\f    return 2\n"),
    ]
    assert module.validate_all_patches(examples, batch_size=2) == 5


def test_ast_valid_but_bytecode_invalid_or_warning_target_is_quarantined():
    invalid = record(1, "x=1\n", "return 1\n")
    warning = record(2, "x=1\n", "x = 1 is 1\n")
    selected, _, counts = remediate_records([invalid, warning])
    assert not selected and counts == {"after_compile_error": 1, "after_syntax_warning": 1}
