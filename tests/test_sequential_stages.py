"""Checkpoint retention, stage integrity, and a complete local six-model tiny sequence."""

import gc
import json
import weakref
from pathlib import Path

import pytest
import torch

from bifocalgemma.models.v2.native import build_native_tiny_model
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.policy import default_training_profiles
from bifocalgemma.training.stages import StageLedger, latest_checkpoint
from tests.test_offline_sft_training import TinyLossModel, examples


def test_only_latest_two_complete_checkpoints_remain_and_partial_is_ignored(tmp_path):
    root = tmp_path / "run"
    result = fit_offline_sft(
        TinyLossModel(),
        examples(),
        examples()[:1],
        root,
        {},
        max_steps=4,
        epochs=3,
        batch_size=1,
        accumulation=1,
        eval_every=10,
        save_every=1,
        save_total_limit=2,
    )
    assert [p.name for p in sorted(root.glob("checkpoint-*"))] == [
        "checkpoint-000003",
        "checkpoint-000004",
    ]
    assert len(result["checkpoints"]) == 2
    (root / "checkpoint-999999").mkdir()
    latest = latest_checkpoint(root)
    assert latest is not None
    assert Path(latest).name == "checkpoint-000004"
    marker = root / "checkpoint-000004/READY.json"
    value = json.loads(marker.read_text())
    value["manifest.json"] = "0" * 64
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="corrupt"):
        latest_checkpoint(root)


def test_stage_receipt_cannot_skip_changed_or_missing_evidence(tmp_path):
    ledger = StageLedger(tmp_path, {"dataset": "revision-one"})
    evidence = tmp_path / "result.json"
    evidence.write_text('{"status":"COMPLETED"}')
    ledger.start("training")
    ledger.finish("training", [evidence])
    assert StageLedger(tmp_path, {"dataset": "revision-one"}).completed("training")
    with pytest.raises(ValueError, match="identity changed"):
        StageLedger(tmp_path, {"dataset": "revision-two"})
    evidence.write_text('{"status":"FAILED"}')
    with pytest.raises(ValueError, match="changed"):
        ledger.completed("training")
    evidence.unlink()
    with pytest.raises(ValueError, match="missing"):
        ledger.completed("training")


@pytest.mark.parametrize("family", ["symmetric", "asymmetric", "decoder_only"])
def test_each_family_sequence_runs_both_tiny_models_then_skips_completed_training(
    tmp_path, monkeypatch, family
):
    import bifocalgemma.training.sequential as pipeline

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(
        pipeline, "load_curated_pool", lambda p: ({}, {"fingerprint": "fixture-pool"})
    )

    def prepare(spec, *args, **kwargs):
        rows = examples()
        if spec.encoder is None:
            from dataclasses import replace

            rows = [replace(r, encoder_ids=()) for r in rows]
        return rows, rows[:1], {"fingerprint": "tiny-fixture", "pool_fingerprint": "fixture-pool"}

    monkeypatch.setattr(pipeline, "prepare_curated_training", prepare)
    monkeypatch.setattr(
        pipeline, "run_training_memory_preflight", lambda *a, **kw: {"status": "SYNTHETIC_ONLY"}
    )
    references = []
    loaded = []

    def factory(spec, progress):
        gc.collect()
        assert all(ref() is None for ref in references), "Earlier model still resident"
        torch.manual_seed(17)
        model = build_native_tiny_model(spec)
        loaded.append(spec.name)
        references.append(weakref.ref(model))
        return model, {"fixture": "native-tiny-not-pretrained"}

    policy = default_training_profiles()[family]
    policy.update(rank=2, alpha=4)
    settings = dict(
        family=family,
        experiments=pipeline.FAMILY_SEQUENCES[family],
        output_root=tmp_path,
        run_name="fixture",
        dataset_repo="fixture/data",
        dataset_revision="a" * 40,
        collection="fixture",
        pool_directory=tmp_path,
        tokenizer_cache=tmp_path,
        benchmark_collections=[],
        benchmark_cache=tmp_path,
        model_factory=factory,
        training_policy=policy,
        memory_policy=None,
        environment={"device": "cpu", "transformers": "5.19.0"},
        model_repo="fixture/model",
        token=False,
        max_steps=1,
        accumulation=1,
        eval_every=10,
        save_every=10,
        upload=False,
        run_benchmarks=False,
    )
    try:
        first = pipeline.run_sequential_pipeline(**settings)
        assert first["status"] == "COMPLETED" and len(first["models"]) == 2
        assert loaded == list(pipeline.FAMILY_SEQUENCES[family])
        second = pipeline.run_sequential_pipeline(**settings)
        assert len(loaded) == 2 and second["status"] == "COMPLETED"
    finally:
        torch.set_num_threads(threads)
