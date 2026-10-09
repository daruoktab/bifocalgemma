"""Curation boundaries, connected splits and prepared-pool integrity."""

import copy
import json

import pytest

from bifocalgemma.data.curation import (
    CodeOverlapIndex,
    code_signature,
    curate_paired_sources,
    export_pool,
    nextcoder_row,
    split_connected_records,
)


def raw(i, repo="owner/repo", license="mit"):
    return {
        "old_file": "a.py",
        "new_file": "a.py",
        "old_contents": f"def function_{i}():\n    return {i}\n",
        "new_contents": f"def function_{i}():\n    return {i + 100}\n",
        "instruction": "Change return value.",
        "subject": "Change return value.",
        "commit": f"commit{i}",
        "repos": repo,
        "license": license,
    }


def test_license_repo_and_benchmark_gates():
    index = CodeOverlapIndex()
    index.add("heldout", raw(3)["new_contents"])
    records, report = curate_paired_sources(
        [
            (
                "editpackft",
                [
                    raw(0, license="unknown"),
                    raw(1, repo="team/fork"),
                    raw(2, repo="bench-project"),
                    raw(3),
                    raw(4),
                ],
            )
        ],
        index,
        blocked_repos={"real/project"},
        aliases={"team/fork": "real/project"},
        blocked_repo_slugs={"bench-project"},
    )
    assert len(records) == 1 and records[0]["upstream_row"] == 4
    assert len(report["rejected"]) == 4
    assert records[0]["source_revision"] and records[0]["code_license"] == "mit"


def test_cross_source_duplicates_and_nextcoder_variants():
    first = raw(1)
    second = {**first, "instruction": "Another wording."}
    records, report = curate_paired_sources(
        [("editpackft", [first]), ("commitpackft_python", [second])],
        CodeOverlapIndex(),
        blocked_repos=set(),
    )
    assert len(records) == 1
    assert report["rejected"][0]["reason"].startswith("duplicate_normalized_edit")
    row = nextcoder_row(
        {
            "prompt": "Rewrite the given python program as per the following instruction.\nChange x.\nWrite the entire code and no other text in the response.\n```python\nx = 0\n```",
            "completion": "```python\nx = 1\n```",
        },
        offset=7,
    )
    assert row["old_contents"] == "x = 0\n" and row["instruction"] == "Change x."
    assert row["repos"] == ""
    with pytest.raises(ValueError, match="non_python"):
        nextcoder_row({"prompt": "Rewrite the given c program.", "completion": "x"}, offset=0)


def test_near_matching_and_short_fragment_policy():
    code = "\n".join(f"value_{i} = {i}" for i in range(40))
    index = CodeOverlapIndex(threshold=0.8)
    index.add("reference", code)
    assert (
        index.match(code.replace("value_0 = 0", "value_0 = 999"))["kind"] == "token_shingle_jaccard"
    )
    assert index.match("value_2 = 2") is None


def test_split_connects_repo_commit_and_seed_then_export(tmp_path):
    rows = []
    for i in range(12):
        row = raw(i)
        row.update(
            instance_id=str(i),
            before=row["old_contents"],
            after=row["new_contents"],
            path="a.py",
            canonical_repos=[f"owner/r{i // 2}"],
            before_signature=code_signature(row["old_contents"]),
        )
        rows.append(row)
    rows[2]["commit"] = rows[0]["commit"]
    rows[4]["before_signature"] = rows[2]["before_signature"]
    original = copy.deepcopy(rows)
    train, dev, report = split_connected_records(rows)
    split = {r["instance_id"]: r["pool_split"] for r in train + dev}
    assert split["0"] == split["1"] == split["2"] == split["3"] == split["4"] == split["5"]
    assert {r for row in train for r in row["canonical_repos"]}.isdisjoint(
        {r for row in dev for r in row["canonical_repos"]}
    )
    other = split_connected_records(original)
    assert {r["instance_id"] for r in dev} == {r["instance_id"] for r in other[1]}
    manifest = export_pool(train + dev, tmp_path / "pool", report)
    assert (
        manifest["files"]
        and json.loads((tmp_path / "pool/manifest.json").read_text(encoding="utf-8"))["fingerprint"]
        == manifest["fingerprint"]
    )
    with pytest.raises(FileExistsError):
        export_pool(train + dev, tmp_path / "pool", report)


def test_frozen_pool_verifies_checksums_before_use(tmp_path):
    from bifocalgemma.data.curation import SOURCE_REFS
    from bifocalgemma.data.offline_sft import load_curated_pool

    rows = []
    for i, split in enumerate(("train", "dev")):
        row = raw(i, repo=f"owner/r{i}")
        row.update(
            instance_id=str(i),
            before=row["old_contents"],
            after=row["new_contents"],
            path="a.py",
            pool_split=split,
            role="train",
            target_patch="diff --git a/a.py b/a.py\n",
            source=SOURCE_REFS["editpackft"]["id"],
            source_revision=SOURCE_REFS["editpackft"]["revision"],
        )
        rows.append(row)
    export_pool(rows, tmp_path / "pool", {"split": {"policy": "connected"}})
    frozen, manifest = load_curated_pool(tmp_path / "pool")
    assert len(frozen["train"]) == len(frozen["dev"]) == 1
    with (tmp_path / "pool/train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="file_integrity"):
        load_curated_pool(tmp_path / "pool")


def test_disk_index_matches_memory_and_syntax_repairs_are_admitted(tmp_path):
    code = "\n".join(f"value_{i} = {i}" for i in range(40))
    memory = CodeOverlapIndex()
    disk = CodeOverlapIndex(storage_path=tmp_path / "benchmarks.sqlite")
    for index in (memory, disk):
        index.add("benchmark", code)
    for query in (code, code.replace("value_0 = 0", "value_0 = 100"), "x = 0"):
        assert memory.match(query) == disk.match(query)
    disk.close()
    item = raw(9)
    item["old_contents"] = "def broken(:\n    pass\n"
    records, _ = curate_paired_sources(
        [("editpackft", [item])],
        CodeOverlapIndex(),
        blocked_repos=set(),
        near_index_path=tmp_path / "near.sqlite",
    )
    assert len(records) == 1


@pytest.mark.parametrize("index", [0, 2, 4])
def test_complete_epoch_training_budget_all_families(index, tmp_path):
    import torch

    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.contracts import TokenizedRepairExample
    from bifocalgemma.models.native import build_native_tiny_model
    from bifocalgemma.training.curated_sft import fit_curated_sft

    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        spec = EXPERIMENT_SPECS[index]
        model = build_native_tiny_model(spec)
        train = [
            TokenizedRepairExample(str(i), (2, 3) if spec.encoder else (), (2, 4), (5, 6))
            for i in range(4)
        ]
        dev = [TokenizedRepairExample("heldout", (2, 3) if spec.encoder else (), (2, 4), (5, 6))]
        result = fit_curated_sft(
            model,
            train,
            dev,
            tmp_path / "run",
            {"fingerprint": "data-v1", "pool_fingerprint": "pool-v1"},
            epochs=2,
            accumulation=2,
            eval_every=2,
            save_every=2,
        )
        assert result["status"] == "COMPLETED"
        assert result["global_step"] == result["planned_full_epoch_updates"] == 4
        assert result["explicit_max_steps"] is None
        assert all(row["step_seconds"] > 0 for row in result["history"])
        assert result["timing"]["segment_wall_seconds"] > 0
        assert result["held_out_eval"]["examples"] == 1
        assert (tmp_path / "run/checkpoint-000004/READY.json").exists()
    finally:
        torch.set_num_threads(previous)


def test_curated_resume_rejects_runtime_drift(tmp_path, monkeypatch):
    import importlib.metadata

    import torch

    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.contracts import TokenizedRepairExample
    from bifocalgemma.models.native import build_native_tiny_model
    from bifocalgemma.training.curated_sft import fit_curated_sft

    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        spec = EXPERIMENT_SPECS[4]
        examples = [TokenizedRepairExample(str(i), (), (2, 3), (4, 5)) for i in range(2)]
        manifest = {"fingerprint": "data", "pool_fingerprint": "pool"}
        first = fit_curated_sft(
            build_native_tiny_model(spec),
            examples,
            examples[:1],
            tmp_path / "run",
            manifest,
            max_steps=1,
            accumulation=1,
            save_every=1,
            eval_every=1,
        )
        restored = build_native_tiny_model(spec)
        version = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: "5.20.0" if name == "transformers" else version(name),
        )
        with pytest.raises(ValueError, match="config mismatch"):
            fit_curated_sft(
                restored,
                examples,
                examples[:1],
                tmp_path / "run",
                manifest,
                max_steps=2,
                accumulation=1,
                resume_from=first["checkpoints"][-1],
            )
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("encoder,decoder", [(2048, 2048), (4096, 8192)])
def test_frozen_budget_matches_all_six_controls(encoder, decoder):
    from bifocalgemma.data.offline_sft import validate_curated_pool_limits

    limits = {
        "encoder": encoder,
        "pair_decoder": decoder,
        "baseline_decoder": encoder + decoder + 512,
    }
    manifest = {"report": {"tokens": {"limits": limits}}}
    original = copy.deepcopy(manifest)
    actual = validate_curated_pool_limits(manifest, encoder_limit=encoder, decoder_limit=decoder)
    assert actual == limits and manifest == original
    actual["encoder"] = 1
    assert manifest == original
    for key in limits:
        changed = copy.deepcopy(manifest)
        changed["report"]["tokens"]["limits"][key] += 1
        with pytest.raises(ValueError, match="Pool token limits differ"):
            validate_curated_pool_limits(changed, encoder_limit=encoder, decoder_limit=decoder)
    with pytest.raises(ValueError, match="Pool token limits differ"):
        validate_curated_pool_limits(manifest, encoder_limit=encoder, decoder_limit=decoder * 2)
    with pytest.raises(ValueError, match="Pool token limits differ"):
        validate_curated_pool_limits({}, encoder_limit=encoder, decoder_limit=decoder)


@pytest.mark.parametrize("invalid", [0, -1, 131073, True, 2048.0, "2048"])
def test_invalid_requested_pool_budget_is_rejected(invalid):
    from bifocalgemma.data.offline_sft import validate_curated_pool_limits

    with pytest.raises(ValueError, match="Invalid requested"):
        validate_curated_pool_limits({}, encoder_limit=invalid, decoder_limit=2048)


def test_repository_pool_requires_context_license_policy_and_coverage(tmp_path):
    from bifocalgemma.data.offline_sft import load_curated_pool

    rows = []
    for split in ("train", "dev"):
        rows.append(
            {
                "instance_id": split,
                "pool_split": split,
                "role": "train",
                "record_type": "repository_patch_v1",
                "source": "synthetic-fixture",
                "before": "### File: a.py\nx=0\n",
                "after": "",
                "path": "repository_context",
                "source_revision": "fixture-v1",
                "group": "fixture/" + split,
                "commit": "a" * 40,
                "repos": "fixture/" + split,
                "instruction": "Fix x",
                "context": "### File: a.py\nx=0\n",
                "target_patch": "diff --git a/a.py b/a.py\n",
                "snapshot_sha256": "b" * 64,
                "context_selection": {"gold_used": False, "windows": [{"path": "a.py"}]},
                "context_source_code_license_evidence": [
                    {
                        "path": "a.py",
                        "licenses": [{"path": "LICENSE", "license": "mit", "sha256": "c" * 64}],
                    }
                ],
            }
        )
    old = {"snapshot_policy": "immutable text archive"}
    export_pool(rows, tmp_path / "old", old)
    with pytest.raises(ValueError, match="context_license_audit_required"):
        load_curated_pool(tmp_path / "old")
    current = {**old, "context_source_license_policy": "nearest_license_for_each_visible_file_v1"}
    export_pool(rows, tmp_path / "current", current)
    assert len(load_curated_pool(tmp_path / "current")[0]["train"]) == 1
    broken = copy.deepcopy(rows)
    broken[0]["context_source_code_license_evidence"] = []
    export_pool(broken, tmp_path / "broken", current)
    with pytest.raises(ValueError, match="context_license_coverage"):
        load_curated_pool(tmp_path / "broken")
