"""Fair initialization, optimizer state resume, and sparse validation/save cadence."""

import copy
import gc
import weakref

import pytest
import torch

from bifocalgemma.config import get_experiment
from bifocalgemma.models.native import build_native_tiny_model
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.optimizers import GrokOrScaleLoRA, orthogonal_direction
from bifocalgemma.training.policy import configure_policy_model
from bifocalgemma.training.recipes import (
    comparison_recipes,
    resolve_training_cadence,
    run_recipe_comparison,
)
from tests.test_offline_sft_training import TinyLossModel, examples


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_shared_numbers_and_orthogonal_zero_factor_bootstrap():
    common = None
    for family in ("symmetric", "asymmetric", "decoder_only"):
        recipes = comparison_recipes(family)
        for policy in recipes.values():
            numerical = (policy["rank"], policy["alpha"], policy["groups"]["decoder"])
            common = numerical if common is None else common
            assert numerical == common
            assert all(
                settings == policy["groups"]["decoder"] for settings in policy["groups"].values()
            )
    p = torch.nn.Parameter(torch.zeros(16, 2))
    config = comparison_recipes("symmetric")["optimized"]["optimizer"]
    optimizer = GrokOrScaleLoRA([{"params": [p], "name": "decoder", "lr": 2e-5}], config)
    p.grad = torch.ones_like(p)
    optimizer.step()
    assert p.count_nonzero() == p.numel() and torch.isfinite(p).all()
    assert optimizer.state[p]["calibration"] is None
    optimizer.step()
    assert optimizer.state[p]["calibration"] > 0
    assert all(
        t.dtype == torch.float32 for t in optimizer.state[p].values() if isinstance(t, torch.Tensor)
    )
    stats = optimizer.last_diagnostics["decoder"]
    assert stats["trust_min"] is not None and stats["trust_max"] is not None
    assert 0.5 <= stats["trust_min"] <= stats["trust_max"] <= 2
    assert torch.equal(orthogonal_direction(torch.zeros(2, 100), 5), torch.zeros(2, 100))
    g = torch.randn(2, 100)
    torch.testing.assert_close(orthogonal_direction(g.T, 5), orthogonal_direction(g, 5).T)


def test_cadence_uses_actual_updates_and_final_validation_is_not_duplicated(tmp_path):
    plan = resolve_training_cadence(11102)
    assert plan == {"total_steps": 11102, "eval_every": 2776, "save_every": 1111}
    events = []
    result = fit_offline_sft(
        TinyLossModel(),
        examples(),
        examples()[:1],
        tmp_path,
        {},
        max_steps=8,
        epochs=8,
        batch_size=4,
        accumulation=1,
        **{k: v for k, v in resolve_training_cadence(8).items() if k != "total_steps"},
        event_callback=events.append,
    )
    assert [r["step"] for r in events if r["stage"] == "validation_started"] == [2, 4, 6, 8]
    assert len(result["checkpoints"]) == 2
    assert len(list(tmp_path.glob("checkpoint-*"))) == 2
    with pytest.raises(ValueError):
        resolve_training_cadence(0)
    with pytest.raises(ValueError):
        resolve_training_cadence(10, save_fraction=0)


@pytest.mark.parametrize(
    "family,name",
    [
        ("symmetric", "BifocalGemma-E2B-E2B"),
        ("asymmetric", "BifocalGemma-E2B-E4B"),
        ("decoder_only", "Gemma-4-E2B-IT"),
    ],
)
def test_optimized_native_resume_matches_full_and_rejects_different_recipe(tmp_path, family, name):
    policy = comparison_recipes(family)["optimized"]
    policy.update(rank=2, alpha=4)
    rows = examples()
    if family == "decoder_only":
        from dataclasses import replace

        rows = [replace(r, encoder_ids=()) for r in rows]

    def model():
        torch.manual_seed(17)
        value = build_native_tiny_model(get_experiment(name))
        configure_policy_model(value, policy)
        return value

    full, staged = model(), model()
    settings = dict(
        max_steps=4,
        epochs=4,
        batch_size=2,
        accumulation=1,
        training_policy=policy,
        eval_every=2,
        save_every=2,
        event_callback=lambda e: None,
    )
    report = fit_offline_sft(full, rows, rows[:1], tmp_path / "full", {}, **settings)
    saved = []

    def stop(path):
        saved.append(path)
        raise KeyboardInterrupt("Simulated interruption")

    with pytest.raises(KeyboardInterrupt):
        fit_offline_sft(
            staged, rows, rows[:1], tmp_path / "resume", {}, checkpoint_callback=stop, **settings
        )
    resumed_model = model()
    original = {k: p.clone() for k, p in resumed_model.state_dict().items()}
    changed = copy.deepcopy(policy)
    changed["optimizer"] = comparison_recipes(family)["base"]["optimizer"]
    config_before = (tmp_path / "resume/run_config.json").read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        fit_offline_sft(
            resumed_model,
            rows,
            rows[:1],
            tmp_path / "resume",
            {},
            resume_from=saved[0],
            **{**settings, "training_policy": changed},
        )
    assert config_before == (tmp_path / "resume/run_config.json").read_bytes()
    for key, tensor in original.items():
        torch.testing.assert_close(tensor, resumed_model.state_dict()[key], rtol=0, atol=0)
    resumed = fit_offline_sft(
        resumed_model, rows, rows[:1], tmp_path / "resume", {}, resume_from=saved[0], **settings
    )
    for key, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, resumed_model.state_dict()[key], rtol=0, atol=0)
    assert report["held_out_eval"] == resumed["held_out_eval"]
    assert [r["optimizer_diagnostics"] for r in report["history"]] == [
        r["optimizer_diagnostics"] for r in resumed["history"]
    ]


@pytest.mark.parametrize("family", ["symmetric", "asymmetric", "decoder_only"])
def test_four_fresh_runs_per_notebook_and_independent_resume(tmp_path, monkeypatch, family):
    import bifocalgemma.training.sequential as pipeline

    monkeypatch.setattr(pipeline, "load_curated_pool", lambda p: ({}, {"fingerprint": "fixture"}))

    def prepare(spec, *args, **kwargs):
        from dataclasses import replace

        rows = examples()
        if spec.encoder is None:
            rows = [replace(row, encoder_ids=()) for row in rows]
        return rows, rows[:1], {"fingerprint": "fixture"}

    monkeypatch.setattr(pipeline, "prepare_curated_training", prepare)
    monkeypatch.setattr(pipeline, "run_training_memory_preflight", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "fetch_collection", lambda *a, **kw: None)
    evaluated = []

    def benchmark(model, spec, directory, output, tokenizer_cache, **kwargs):
        from bifocalgemma.training.stages import atomic_json

        # Real tiny training/checkpoints and inference adapter reload, fake scoring.
        assert len(list(tmp_path.glob("*/comparison/*/training/training_report.json"))) == 4
        assert not model.training
        assert any("lora_A" in name for name, _ in model.named_parameters())
        evaluated.append(spec.name)
        for filename in ("benchmark_report.json", "predictions.jsonl", "evaluation_state.json"):
            atomic_json(output / filename, {"status": "COMPLETED"})
        return {"status": "COMPLETED"}

    monkeypatch.setattr(pipeline, "run_context_benchmark", benchmark)
    references, loaded, initial = [], [], {}

    def factory(spec, progress):
        gc.collect()
        assert all(ref() is None for ref in references)
        torch.manual_seed(17)
        value = build_native_tiny_model(spec)
        weights = value.state_dict()["decoder.layers.0.self_attn.q_proj.weight"].clone()
        if spec.name in initial:
            torch.testing.assert_close(initial[spec.name], weights, rtol=0, atol=0)
        initial[spec.name] = weights
        loaded.append(spec.name)
        references.append(weakref.ref(value))
        return value, {"fixture": "native tiny"}

    recipes = comparison_recipes(family)
    for policy in recipes.values():
        policy.update(rank=2, alpha=4)
    settings = dict(
        recipes=recipes,
        experiments=pipeline.FAMILY_SEQUENCES[family],
        run_name="comparison",
        family=family,
        output_root=tmp_path,
        dataset_repo="fixture/data",
        dataset_revision="a" * 40,
        collection="fixture",
        pool_directory=tmp_path,
        tokenizer_cache=tmp_path,
        benchmark_collections=[{"collection": "fixture-eval"}],
        benchmark_cache=tmp_path,
        model_factory=factory,
        memory_policy=None,
        environment={"device": "cpu", "transformers": "5.19.0"},
        model_repo="fixture/model",
        token=False,
        max_steps=1,
        accumulation=1,
        upload=False,
        run_benchmarks=True,
    )
    outcome = run_recipe_comparison(**settings)
    assert len(outcome["models"]) == 4
    assert [r["recipe"] for r in outcome["models"]] == ["base", "optimized", "base", "optimized"]
    assert len({(r["experiment"], r["run_dir"]) for r in outcome["models"]}) == 4
    order = [name for name in pipeline.FAMILY_SEQUENCES[family] for _ in range(2)]
    assert loaded == order * 2
    assert evaluated == order
    again = run_recipe_comparison(**settings)
    assert len(again["models"]) == 4 and len(loaded) == 8 and len(evaluated) == 4


def test_fp32_alignment_on_bf16_backbone():
    policy = comparison_recipes("asymmetric")["base"]
    policy.update(rank=2, alpha=4)
    model = build_native_tiny_model(get_experiment("BifocalGemma-E2B-E4B")).to(torch.bfloat16)
    configure_policy_model(model, policy)
    assert isinstance(model.encoder_projection, torch.nn.Linear)
    assert model.encoder is not None
    assert model.encoder_projection.weight.dtype == torch.float32
    assert model.decoder.embed_tokens.weight.dtype == torch.bfloat16
    values = torch.randn(1, 3, model.encoder.config.hidden_size, dtype=torch.bfloat16)
    result = model.encoder_projection(values)
    assert result.dtype == torch.bfloat16
    result.float().sum().backward()
    assert model.encoder_projection.weight.grad is not None
    assert model.encoder_projection.weight.grad.dtype == torch.float32


@pytest.mark.parametrize("family", ["symmetric", "asymmetric", "decoder_only"])
def test_all_training_precedes_benchmark_and_failure_stops_barrier(monkeypatch, family):
    import bifocalgemma.training.sequential as pipeline

    calls = []
    failed = False

    def run(**kwargs):
        calls.append(kwargs)
        if failed and len(calls) == 3:
            raise RuntimeError("training upload failed")
        return {"models": [{"experiment": kwargs["experiments"][0], "status": "COMPLETED"}]}

    monkeypatch.setattr(pipeline, "run_sequential_pipeline", run)
    settings = dict(
        family=family,
        recipes=comparison_recipes(family),
        experiments=pipeline.FAMILY_SEQUENCES[family],
        run_name="same-existing-run",
        run_training=True,
        run_benchmarks=True,
        auto_resume=False,
    )
    result = run_recipe_comparison(**settings)
    assert len(result["models"]) == 4
    assert [c["run_benchmarks"] for c in calls] == [False] * 4 + [True] * 4
    assert [c["run_training"] for c in calls] == [True] * 4 + [False] * 4
    assert [c["finalize"] for c in calls] == [False] * 4 + [True] * 4
    assert [c["auto_resume"] for c in calls] == [False] * 4 + [True] * 4
    assert [(c["experiments"], c["run_name"]) for c in calls[:4]] == [
        (c["experiments"], c["run_name"]) for c in calls[4:]
    ]
    calls.clear()
    failed = True
    with pytest.raises(RuntimeError, match="upload failed"):
        run_recipe_comparison(**settings)
    assert all(not c["run_benchmarks"] for c in calls)


@pytest.mark.parametrize("fail_final_ledger_commit", [False, True])
def test_completed_benchmark_with_pending_upload_resumes_without_model_load(
    tmp_path, monkeypatch, fail_final_ledger_commit
):
    import json

    import bifocalgemma.training.sequential as pipeline
    from bifocalgemma.training.stages import atomic_json
    from tests.test_benchmark_upload import limited

    monkeypatch.setattr(pipeline, "load_curated_pool", lambda p: ({}, {"fingerprint": "fixture"}))
    monkeypatch.setattr(
        pipeline,
        "prepare_curated_training",
        lambda *a, **kw: (examples(), examples()[:1], {"fingerprint": "fixture"}),
    )
    monkeypatch.setattr(pipeline, "run_training_memory_preflight", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "fetch_collection", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "recover_remote_run", lambda *a, **kw: None)
    blocked = [True]
    uploads = []

    def upload(repo, root, prefix, **kwargs):
        stages = json.loads((root / "pipeline_state.json").read_text())["stages"]
        uploads.append(list(stages))
        failing_stage = (
            stages.get("final_upload", {}).get("status") == "COMPLETED"
            if fail_final_ledger_commit
            else any(k.startswith("benchmark:") for k in stages)
        )
        if blocked[0] and failing_stage:
            raise limited()
        return {"revision": "fixture-verified"}

    monkeypatch.setattr(pipeline, "synchronize_run", upload)
    evaluated = []

    def benchmark(model, spec, directory, output, tokenizer_cache, **kwargs):
        kwargs["prediction_callback"]()
        evaluated.append(spec.name)
        for filename in ("benchmark_report.json", "predictions.jsonl", "evaluation_state.json"):
            atomic_json(output / filename, {"status": "COMPLETED"})
        return {"status": "COMPLETED"}

    monkeypatch.setattr(pipeline, "run_context_benchmark", benchmark)
    loaded = []

    def factory(spec, progress):
        loaded.append(spec.name)
        return build_native_tiny_model(spec), {}

    policy = comparison_recipes("symmetric")["base"]
    policy.update(rank=2, alpha=4)
    settings = dict(
        family="symmetric",
        recipes={"base": policy},
        experiments=("BifocalGemma-E2B-E2B",),
        run_name="fixture",
        output_root=tmp_path,
        dataset_repo="fixture/data",
        dataset_revision="a" * 40,
        collection="fixture",
        pool_directory=tmp_path,
        tokenizer_cache=tmp_path,
        benchmark_collections=[{"collection": "eval"}],
        benchmark_cache=tmp_path,
        model_factory=factory,
        memory_policy=None,
        environment={"device": "cpu", "transformers": "5.19.0"},
        model_repo="fixture/model",
        token="fixture",
        max_steps=1,
        accumulation=1,
        checkpoint_upload=False,
        upload=True,
    )
    first = run_recipe_comparison(**settings)
    assert first["status"] == "PENDING_UPLOAD" and len(evaluated) == 1
    assert first["models"][0]["stages"]["benchmark:eval"]["status"] == "COMPLETED"
    if fail_final_ledger_commit:
        assert first["models"][0]["stages"]["final_upload"]["status"] == "COMPLETED"
    assert len(loaded) == 2
    # Simulate expiry of the persisted backoff; completed scores are not regenerated.
    blocked[0] = False
    for path in [
        tmp_path / ".benchmark-upload-backoff.json",
        tmp_path / "BifocalGemma-E2B-E2B/fixture/base/benchmark_upload_status.json",
    ]:
        value = json.loads(path.read_text())
        value["next_retry_at"] = 0
        atomic_json(path, value)
    second = run_recipe_comparison(**settings)
    assert second["status"] == "COMPLETED" and len(loaded) == 2 and len(evaluated) == 1
    assert second["models"][0]["stages"]["final_upload"]["status"] == "COMPLETED"
