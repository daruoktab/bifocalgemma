"""Exercise shared windows through real tiny matrix stages, checkpoints and loss."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from typing import Any

import pytest
import torch

from bifocalgemma.config import MATRIX_MODELS, make_experiment_spec
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.modeling_bifocalgemma import NativeTinyModel
from bifocalgemma.models.cross_attention import configure_full_rank_cross_attention
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.training.checkpoint import restore_matrix_checkpoint
from bifocalgemma.training.matrix import run_matrix_stage
from bifocalgemma.training.policy import configure_policy_model, update_policy_learning_rates
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.schedule import build_shared_schedule, canonical_schedule_records
from bifocalgemma.training.stage import configure_matrix_stage


def fixture_data(key, count=12):
    paired = MATRIX_MODELS[key]["encoder_id"] is not None
    raw: list[dict[str, Any]] = [
        {
            "instance_id": f"item-{i}",
            "instruction": f"Implement {i}",
            "context": f"def func_{i}(): pass",
            "target_text": f"return {i}",
            "source": f"fixture-{i % 2}",
            "task_type": "repair" if i % 2 else "completion",
            "token_lengths": {"encoder": 8 + i},
        }
        for i in range(count)
    ]
    examples = [
        TokenizedRepairExample(
            row["instance_id"],
            (2, 10 + i, 20) if paired else (),
            (2, 30) if paired else (2, 30, 10 + i, 20),
            (50 + i, 60, 107) if i % 2 else (50 + i, 107),
            metadata={"task_type": row["task_type"], "source": row["source"]},
        )
        for i, row in enumerate(raw)
    ]
    schedule = build_shared_schedule(
        canonical_schedule_records(raw), dataset_identity={"fixture": "runtime"}
    )
    return examples, schedule


def fixture_model(key):
    torch.manual_seed(111)
    spec = make_experiment_spec(key)
    model = NativeTinyModel(spec, tiny=True)
    policy = unified_recipe(MATRIX_MODELS[key]["family"])
    configure_policy_model(model, policy)
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    return model, policy


@pytest.mark.parametrize("key", list(MATRIX_MODELS))
def test_all_ten_real_stage_loaders_consume_identical_windows(key, tmp_path):
    examples, schedule = fixture_data(key)
    model, policy = fixture_model(key)
    trace = []
    common: dict[str, Any] = dict(
        policy=policy,
        run_dir=tmp_path,
        compatibility={"model_key": key},
        shared_schedule=schedule,
        progress=trace.append,
        save_every_steps=1,
    )
    if model.encoder is None:
        run_matrix_stage(
            model,
            examples,
            [],
            "full_decoder_sft",
            max_steps=3,
            freeze_decoder=False,
            start_global_step=0,
            **common,
        )
    else:
        first = run_matrix_stage(
            model,
            examples,
            [],
            "frozen_decoder",
            max_steps=1,
            freeze_decoder=True,
            start_global_step=0,
            **common,
        )
        run_matrix_stage(
            model,
            examples,
            [],
            "joint",
            max_steps=2,
            freeze_decoder=False,
            start_global_step=1,
            resume_from_checkpoint=first["last_checkpoint"],
            **common,
        )
    assert [identifier for step in trace for identifier in step["instance_ids"]] == list(
        schedule.ordered_ids
    )
    assert [step["example_cursor"] for step in trace] == [4, 8, 12]
    assert sum(step["target_tokens"] for step in trace) == sum(len(x.target_ids) for x in examples)
    for step in trace:
        assert sum(step["target_tokens_by_task"].values()) == step["target_tokens"]
    batch = RepairCollator()(examples[:2])
    for i, example in enumerate(examples[:2]):
        start = batch.labels.shape[1] - len(example.target_ids)
        assert batch.labels[i, :start].tolist() == [-100] * start
        assert batch.labels[i, start : start + len(example.target_ids)].tolist() == list(
            example.target_ids
        )


def test_interrupted_optimizer_boundary_resume_matches_uninterrupted(tmp_path):
    key = "e2b-it"
    examples, schedule = fixture_data(key)
    model, policy = fixture_model(key)
    initial = copy.deepcopy(model.state_dict())
    common: dict[str, Any] = dict(
        policy=policy,
        max_steps=3,
        freeze_decoder=False,
        start_global_step=0,
        compatibility={"model_key": key},
        shared_schedule=schedule,
        save_every_steps=1,
    )
    run_matrix_stage(model, examples, [], "full_decoder_sft", run_dir=tmp_path / "full", **common)
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(initial)

    class Interrupted(Exception):
        pass

    def stop_after_checkpoint(path):
        raise Interrupted

    with pytest.raises(Interrupted):
        run_matrix_stage(
            model,
            examples,
            [],
            "full_decoder_sft",
            run_dir=tmp_path / "resume",
            checkpoint_callback=stop_after_checkpoint,
            **common,
        )
    model.load_state_dict(initial)
    trace = []
    run_matrix_stage(
        model,
        examples,
        [],
        "full_decoder_sft",
        run_dir=tmp_path / "resume",
        resume_from_checkpoint=tmp_path / "resume/training/checkpoint-000001",
        progress=trace.append,
        **common,
    )
    assert [row["step"] for row in trace] == [2, 3]
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="compatibility changed"):
        restore_matrix_checkpoint(
            model,
            None,
            tmp_path / "resume/training/checkpoint-000003",
            expected_compatibility={"model_key": "e4b-it"},
        )
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, before[name], rtol=0, atol=0)


def test_partial_backward_oom_replays_entire_window_without_double_gradient(tmp_path, monkeypatch):
    import bifocalgemma.training.matrix as matrix

    examples, schedule = fixture_data("e2b-it", 4)
    model, policy = fixture_model("e2b-it")
    initial = copy.deepcopy(model.state_dict())
    common: dict[str, Any] = dict(
        policy=policy,
        max_steps=1,
        freeze_decoder=False,
        start_global_step=0,
        compatibility={"model_key": "e2b-it"},
        shared_schedule=schedule,
    )
    run_matrix_stage(
        model,
        examples,
        [],
        "full_decoder_sft",
        batch_size=1,
        accumulation=4,
        run_dir=tmp_path / "single",
        **common,
    )
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(initial)
    original = matrix.compute_step_loss
    calls = 0

    def fail_second_microbatch(*args, **kwargs):
        nonlocal calls
        calls += 1
        loss = original(*args, **kwargs)
        if calls == 2:
            (loss * 0.3).backward()
            raise torch.cuda.OutOfMemoryError("injected after partial backward")
        return loss

    monkeypatch.setattr(matrix, "compute_step_loss", fail_second_microbatch)
    run_matrix_stage(
        model,
        examples,
        [],
        "full_decoder_sft",
        batch_size=2,
        accumulation=2,
        run_dir=tmp_path / "oom",
        **common,
    )
    assert calls == 6
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=1e-5, atol=1e-7)
    payload = torch.load(
        tmp_path / "oom/training/checkpoint-000001/trainer_state.pt", weights_only=False
    )
    assert payload["training_state"]["example_cursor"] == 4
    assert (
        json.loads((tmp_path / "oom/training/shared_schedule.json").read_text())["signature"]
        == schedule.signature
    )


@pytest.mark.parametrize("key", ["e2b-it", "2base-2it", "4base-2it"])
def test_trained_adapters_export_reload_logit_parity(key, tmp_path):
    from bifocalgemma.modeling_bifocalgemma import (
        BifocalGemmaForCausalLM,
        export_matrix_pretrained_model,
    )

    examples, schedule = fixture_data(key, 4)
    model, policy = fixture_model(key)
    run_matrix_stage(
        model,
        examples,
        [],
        "joint",
        policy=policy,
        max_steps=1,
        freeze_decoder=False,
        start_global_step=0,
        compatibility={"model_key": key},
        shared_schedule=schedule,
        run_dir=tmp_path / "training-run",
    )
    # The trained LoRA and bridge weights must survive consolidation, not only shapes.
    assert any(
        bool(p.detach().abs().sum() > 0)
        for name, p in model.named_parameters()
        if name.endswith(".lora_B")
    )
    batch = RepairCollator()(examples[:2])
    model.eval()
    with torch.no_grad():
        states = (
            model.encode_context(batch.encoder_input_ids, batch.encoder_attention_mask)
            if model.encoder is not None
            else None
        )
        expected = model.decode_logits(
            batch.decoder_input_ids,
            batch.decoder_attention_mask,
            states,
            batch.encoder_attention_mask if states is not None else None,
        )
    export_matrix_pretrained_model(model, make_experiment_spec(key), tmp_path / "export")
    reloaded = BifocalGemmaForCausalLM.from_pretrained(tmp_path / "export").eval()
    if reloaded.encoder is not None:
        assert reloaded.encoder.embed_tokens.weight is not reloaded.decoder.embed_tokens.weight
    with torch.no_grad():
        actual = reloaded(
            input_ids=batch.decoder_input_ids,
            attention_mask=batch.decoder_attention_mask,
            encoder_input_ids=batch.encoder_input_ids if model.encoder is not None else None,
            encoder_attention_mask=batch.encoder_attention_mask
            if model.encoder is not None
            else None,
        )
    # Padding queries are intentionally masked differently by tiled/eager kernels.
    # Every visible decoder token, including the next-token query, must agree.
    torch.testing.assert_close(
        actual.logits[batch.decoder_attention_mask],
        expected[batch.decoder_attention_mask],
        rtol=1e-4,
        atol=1e-5,
    )


def test_matrix_entrypoint_local_pool_completed_resume_exports_trained_weights(
    tmp_path, monkeypatch
):
    from safetensors.torch import load_file

    import bifocalgemma.data.tokenization as tokenization
    import bifocalgemma.modeling_bifocalgemma as pretrained
    import bifocalgemma.training.matrix as matrix
    from bifocalgemma.data.offline_sft import stable_fingerprint

    key = "2base-2it"
    examples, schedule = fixture_data(key)
    pool = tmp_path / "local-supplement"
    pool.mkdir()
    pool_manifest: dict[str, Any] = {
        "format": "bifocalgemma_curated_pool_v1",
        "record_ids": {"train": [ex.instance_id for ex in examples]},
    }
    pool_manifest["fingerprint"] = stable_fingerprint(pool_manifest)
    (pool / "manifest.json").write_text(json.dumps(pool_manifest))
    by_id = {record.instance_id: asdict(record) for record in schedule.records}

    def prepare(*args, **kwargs):
        return (
            examples,
            [],
            {
                "fingerprint": "tokenized-fixture",
                "pool_fingerprint": pool_manifest["fingerprint"],
                "schedule_records": [by_id[ex.instance_id] for ex in examples],
            },
        )

    def download(spec, *args, **kwargs):
        torch.manual_seed(111)
        return NativeTinyModel(spec, tiny=True), {"fixture": "fixed-initialization"}

    def no_tokenizer(*args, **kwargs):
        raise FileNotFoundError("tiny fixture uses integer IDs")

    export = pretrained.export_matrix_pretrained_model

    def local_export(**kwargs: Any):
        kwargs["tokenizer_cache"] = None
        return export(**kwargs)

    monkeypatch.setattr(matrix, "download_model", download)
    monkeypatch.setattr(matrix, "prepare_curated_training", prepare)
    monkeypatch.setattr(tokenization, "load_verified_tokenizer", no_tokenizer)
    monkeypatch.setattr(pretrained, "export_matrix_pretrained_model", local_export)
    trace = []
    kwargs: dict[str, Any] = dict(
        pool_directory=pool,
        output_root=tmp_path / "runs",
        cache=tmp_path / "cache",
        upload=False,
        monitor=trace.append,
    )

    # Monitor needs a run_dir slot, like the notebook dashboard.
    class Monitor:
        run_dir = None

        def __call__(self, event):
            trace.append(event)

    kwargs["monitor"] = Monitor()
    first = matrix.run_matrix_training(key, **kwargs)
    assert first["status"] == "COMPLETED"
    assert first["global_step"] == 3 and first["schedule_total_examples"] == 12
    assert first["realized_example_cursor"] == 12
    assert first["realized_id_order_sha256"] == stable_fingerprint(schedule.ordered_ids)
    assert not first["explicit_prefix_pilot"]
    weights = load_file(str(tmp_path / "runs" / key / "train/pretrained_model/model.safetensors"))
    trace.clear()
    second = matrix.run_matrix_training(key, **kwargs)
    assert second["status"] == "COMPLETED"
    assert second["stage1"] == first["stage1"]
    assert second["stage2"] == first["stage2"]
    assert not any(event.get("stage") == "matrix_training_step" for event in trace)
    restored_weights = load_file(
        str(tmp_path / "runs" / key / "train/pretrained_model/model.safetensors")
    )
    for name, value in weights.items():
        torch.testing.assert_close(restored_weights[name], value, rtol=0, atol=0)
    ledger = tmp_path / "runs" / key / "train/pipeline_state.json"
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        matrix.run_matrix_training(key, total_steps=2, **kwargs)
    assert ledger.read_bytes() == before
    contract = matrix.resolve_training_contract(
        make_experiment_spec(key), pool_directory=pool, stage1_steps=0
    )
    assert contract["stage1_steps"] == 0 and contract["stage2_steps"] == 3


def test_recovered_completed_training_and_benchmark_resume_skip_completed_work(
    monkeypatch, tmp_path
):
    """Completed training (with HF export) and completed benchmarks are reused across resumes."""
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace

    import huggingface_hub

    import bifocalgemma.data.tokenization as tokenization
    import bifocalgemma.modeling_bifocalgemma as pretrained
    import bifocalgemma.training.matrix as matrix
    from bifocalgemma.data.offline_sft import stable_fingerprint

    key = "2base-2it"
    examples, schedule = fixture_data(key)
    pool = tmp_path / "local-supplement"
    pool.mkdir()
    pool_manifest: dict[str, Any] = {
        "format": "bifocalgemma_curated_pool_v1",
        "record_ids": {"train": [ex.instance_id for ex in examples]},
    }
    pool_manifest["fingerprint"] = stable_fingerprint(pool_manifest)
    (pool / "manifest.json").write_text(json.dumps(pool_manifest))
    by_id = {record.instance_id: asdict(record) for record in schedule.records}

    monkeypatch.setattr(
        matrix,
        "prepare_curated_training",
        lambda *a, **k: (
            examples,
            [],
            {
                "fingerprint": "tokenized-fixture",
                "pool_fingerprint": pool_manifest["fingerprint"],
                "schedule_records": [by_id[ex.instance_id] for ex in examples],
            },
        ),
    )

    def download(spec, *args, **kwargs):
        torch.manual_seed(111)
        return NativeTinyModel(spec, tiny=True), {"fixture": "fixed-initialization"}

    monkeypatch.setattr(matrix, "download_model", download)
    monkeypatch.setattr(
        tokenization,
        "load_verified_tokenizer",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("tiny fixture")),
    )
    orig_export = pretrained.export_matrix_pretrained_model

    def local_export(**kw: Any) -> Any:
        kw["tokenizer_cache"] = None
        return orig_export(**kw)

    monkeypatch.setattr(pretrained, "export_matrix_pretrained_model", local_export)

    class Monitor:
        run_dir = None

        def __call__(self, event):
            pass

    output_root = tmp_path / "runs"
    common: dict[str, Any] = dict(
        pool_directory=pool,
        output_root=output_root,
        cache=tmp_path / "cache",
        monitor=Monitor(),
    )
    first = matrix.run_matrix_training(key, upload=False, **common)
    assert first["status"] == "COMPLETED"

    # Reuse requires validated local export bytes matching the uploaded snapshot.
    # Remote filenames alone must never stand in for export validation.
    run_root = output_root / key / "train"
    subfolder = first["subfolder"]
    remote_siblings = []
    for filename in ("config.json", "model.safetensors"):
        content = (run_root / "pretrained_model" / filename).read_bytes()
        remote_siblings.append(
            SimpleNamespace(
                rfilename=f"{subfolder}/pretrained_model/{filename}",
                size=len(content),
                lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()),
                blob_id=None,
            )
        )

    class FakeHfApi:
        def __init__(self, token=None):
            pass

        def repo_exists(self, repo_id):
            return True

        def model_info(self, repo_id, **kwargs):
            return SimpleNamespace(sha="d" * 40, siblings=remote_siblings)

        def list_repo_files(self, repo_id, revision=None):
            return []

        def create_repo(self, *args, **kwargs):
            return None

        def create_commit(self, *args, **kwargs):
            return SimpleNamespace(oid="e" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(matrix, "recover_remote_run", lambda *a, **k: None)
    monkeypatch.setattr(
        matrix,
        "download_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("download_model must not run")),
    )
    monkeypatch.setattr(
        matrix,
        "prepare_curated_training",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("prepare_curated_training must not run")
        ),
    )
    monkeypatch.setattr(
        pretrained,
        "export_matrix_pretrained_model",
        lambda **kw: (_ for _ in ()).throw(AssertionError("export must not re-run")),
    )

    resumed_train = matrix.run_matrix_training(key, token="fixture-tok", upload=True, **common)
    assert resumed_train["status"] == "COMPLETED"
    assert resumed_train["stage1"] == first["stage1"]

    # Now mark first 2 benchmarks completed and verify resume only runs the remaining 1 + final sync.
    test_benchmarks = [
        {"collection": "bench_one", "encoder_limit": 32, "decoder_limit": 32, "max_new_tokens": 8},
        {"collection": "bench_two", "encoder_limit": 32, "decoder_limit": 32, "max_new_tokens": 8},
        {
            "collection": "bench_three",
            "encoder_limit": 32,
            "decoder_limit": 32,
            "max_new_tokens": 8,
        },
    ]
    monkeypatch.setattr(matrix, "BENCHMARKS", test_benchmarks)
    monkeypatch.setattr(matrix, "download_model", download)
    monkeypatch.setattr(matrix, "configure_policy_model", lambda *a, **k: None)
    monkeypatch.setattr(matrix, "configure_memory_efficient_training", lambda *a, **k: None)
    monkeypatch.setattr(matrix, "configure_full_rank_cross_attention", lambda *a, **k: None)
    monkeypatch.setattr(matrix, "load_matrix_adapters_for_inference", lambda *a, **k: None)
    monkeypatch.setattr(matrix, "fetch_collection", lambda *a, **k: None)
    monkeypatch.setattr(matrix, "resolve_optimal_eval_batch_size", lambda *a, **k: 2)

    sync_calls: list[str] = []

    def fake_sync(repo, root, prefix, *, token, require_checkpoint=True):
        sync_calls.append("sync")
        return {"revision": f"rev-{len(sync_calls)}", "retained_checkpoints": []}

    monkeypatch.setattr(matrix, "synchronize_matrix", fake_sync)

    ran_collections: list[str] = []

    def fake_run_bench(model, spec, collection_dir, destination, *args, **kwargs):
        item = kwargs["benchmark_item"]
        col = item["collection"]
        ran_collections.append(col)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        report = {"collection": col, "accuracy": 0.5, "total": 10}
        (destination / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
        (destination / "predictions.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
        (destination / "evaluation_state.json").write_text('{"done": true}\n', encoding="utf-8")
        return report

    monkeypatch.setattr(matrix, "run_batched_context_benchmark", fake_run_bench)

    # First run completes only bench_one & bench_two by stopping before bench_three.
    orig_bench = fake_run_bench

    def stop_on_third(*args, **kwargs):
        if kwargs["benchmark_item"]["collection"] == "bench_three":
            raise RuntimeError("simulated preemption during bench_three")
        return orig_bench(*args, **kwargs)

    monkeypatch.setattr(matrix, "run_batched_context_benchmark", stop_on_third)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        matrix.run_matrix_benchmark(key, token="fixture-tok", upload=True, **common)
    assert ran_collections == ["bench_one", "bench_two"]
    # Even with interval=1800, each completed benchmark triggered immediate final sync!
    assert len(sync_calls) == 2

    # Resume with all 3 allowed: bench_one and bench_two must be skipped!
    ran_collections.clear()
    sync_calls.clear()
    monkeypatch.setattr(matrix, "run_batched_context_benchmark", fake_run_bench)
    bench_res = matrix.run_matrix_benchmark(key, token="fixture-tok", upload=True, **common)
    assert bench_res["status"] == "COMPLETED"
    assert ran_collections == ["bench_three"]
    assert [r["collection"] for r in bench_res["benchmarks"]] == [
        "bench_one",
        "bench_two",
        "bench_three",
    ]
    # 1 final sync on bench_three + 1 end-of-suite sync; ensure immediate sync worked.
    assert len(sync_calls) >= 1

    # Third call when all 3 are already COMPLETED must not even download the model!
    ran_collections.clear()
    monkeypatch.setattr(
        matrix,
        "download_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not load when all done")),
    )
    all_cached = matrix.run_matrix_benchmark(key, token="fixture-tok", upload=True, **common)
    assert all_cached["status"] == "COMPLETED"
    assert ran_collections == []
    assert len(all_cached["benchmarks"]) == 3

    # A different valid checkpoint at the same step must invalidate every receipt.
    from safetensors.torch import load_file, save_file

    from bifocalgemma.data.curation import file_digest

    checkpoint_path = matrix.latest_checkpoint(run_root / "training")
    assert checkpoint_path is not None
    checkpoint = Path(checkpoint_path)
    weights = load_file(str(checkpoint / "adapters.safetensors"))
    first_key = next(iter(weights))
    weights[first_key] = weights[first_key] + 1
    save_file(weights, str(checkpoint / "adapters.safetensors"))
    ready = json.loads((checkpoint / "READY.json").read_text(encoding="utf-8"))
    ready["adapters.safetensors"] = file_digest(checkpoint / "adapters.safetensors")
    (checkpoint / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    monkeypatch.setattr(matrix, "download_model", download)
    monkeypatch.setattr(matrix, "synchronize_matrix", fake_sync)
    replaced = matrix.run_matrix_benchmark(key, token="fixture-tok", upload=True, **common)
    assert replaced["status"] == "COMPLETED"
    assert ran_collections == [item["collection"] for item in test_benchmarks]

    # A cached suite is still pending until its final synchronization succeeds.
    import httpx2

    def failed_sync(*args, **kwargs):
        raise httpx2.ConnectError("simulated final upload failure")

    ran_collections.clear()
    monkeypatch.setattr(matrix, "synchronize_matrix", failed_sync)
    monkeypatch.setattr(
        matrix,
        "download_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cached retry must not load model")),
    )
    pending = matrix.run_matrix_benchmark(key, token="fixture-tok", upload=True, **common)
    assert pending["status"] == "PENDING_UPLOAD"
    assert ran_collections == []


def test_stage1_to_stage2_continuous_warmup_offset_and_per_group_grad_norms(tmp_path):
    """Stage 2 keeps alignment at peak on entry (WSD flat-top / reset) while decoder warms up via warmup_offset."""
    key = "2base-2it"
    model, base_policy = fixture_model(key)
    policy = copy.deepcopy(base_policy)
    for cfg in policy["groups"].values():
        cfg["warmup_ratio"] = 0.03

    # Verify warmup_ratio=0.03 is honored verbatim without silent override.
    groups_s1, _ = configure_matrix_stage(model, policy, freeze_decoder=True, total_steps=100)
    assert all(g["warmup_steps"] == 3 and g["warmup_offset"] == 0 for g in groups_s1)

    groups_s2, _ = configure_matrix_stage(
        model, policy, freeze_decoder=False, total_steps=100, warmup_offset=50
    )
    by_name = {g["name"]: g for g in groups_s2}
    assert by_name["encoder_lora"]["warmup_offset"] == 0
    assert by_name["bridge_mlp"]["warmup_offset"] == 0
    assert by_name["cross_gate_scalars"]["warmup_offset"] == 0
    assert by_name["cross_gate_proj"]["warmup_offset"] == 0
    assert by_name["cross_kv_proj"]["warmup_offset"] == 0
    assert by_name["decoder_lora"]["warmup_offset"] == 50

    opt1 = torch.optim.AdamW(groups_s1)
    opt2 = torch.optim.AdamW(groups_s2)
    r49 = update_policy_learning_rates(opt1, 49)
    r50 = update_policy_learning_rates(opt2, 50)
    r51 = update_policy_learning_rates(opt2, 51)
    r99 = update_policy_learning_rates(opt2, 99)

    # Alignment groups hold peak at Stage 2 entry (fixing Stage-2 alignment LR carry-over starvation)
    assert r50["cross_gate_scalars"] == pytest.approx(r49["cross_gate_scalars"])
    assert r50["bridge_mlp"] == pytest.approx(r49["bridge_mlp"])
    assert 0.0 < r50["decoder_lora"] < r51["decoder_lora"]
    assert r99["encoder_lora"] == pytest.approx(by_name["encoder_lora"]["base_lr"] * 0.1)
    assert r99["decoder_lora"] == pytest.approx(by_name["decoder_lora"]["base_lr"] * 0.1)

    # Verify real Stage 1 -> Stage 2 run_matrix_stage emits per-group clipped norms and continuous LRs.
    examples, schedule = fixture_data(key, count=16)
    fresh_model, fresh_policy = fixture_model(key)
    trace: list[dict[str, Any]] = []
    common: dict[str, Any] = dict(
        policy=fresh_policy,
        run_dir=tmp_path / "continuous",
        compatibility={"model_key": key},
        shared_schedule=schedule,
        progress=trace.append,
        total_steps=4,
    )
    first = run_matrix_stage(
        fresh_model,
        examples,
        [],
        "frozen_decoder",
        max_steps=2,
        freeze_decoder=True,
        start_global_step=0,
        **common,
    )
    run_matrix_stage(
        fresh_model,
        examples,
        [],
        "joint",
        max_steps=2,
        freeze_decoder=False,
        start_global_step=2,
        resume_from_checkpoint=first["last_checkpoint"],
        **common,
    )
    assert [row["step"] for row in trace] == [1, 2, 3, 4]
    for row in trace:
        group_norms = row["group_gradient_norms"]
        assert row["gradient_norm"] == pytest.approx(
            math.sqrt(sum(v * v for v in group_norms.values()))
        )
    assert set(trace[2]["group_gradient_norms"]) == {
        "cross_gate_scalars",
        "cross_gate_proj",
        "bridge_mlp",
        "cross_kv_proj",
        "encoder_lora",
        "decoder_lora",
    }
    assert trace[2]["learning_rates"]["cross_gate_scalars"] == pytest.approx(
        trace[1]["learning_rates"]["cross_gate_scalars"]
    )
    assert trace[2]["learning_rates"]["bridge_mlp"] == pytest.approx(
        trace[1]["learning_rates"]["bridge_mlp"]
    )


def test_length_grouped_microbatch_packing_preserves_gradients_and_backoff(tmp_path, monkeypatch):
    import bifocalgemma.training.matrix as matrix
    from bifocalgemma.training.matrix import pack_window_microbatches

    # Build 8 varied-length examples (4 short + 2 medium + 2 long) in a single effective_batch=8 window.
    lengths = [6, 40, 8, 20, 7, 50, 9, 22]
    raw: list[dict[str, Any]] = [
        {
            "instance_id": f"len-{i}",
            "instruction": f"Task {i}",
            "context": "x " * ln,
            "target_text": f"out {i}",
            "source": "fixture",
            "task_type": "repair" if i % 2 else "completion",
            "token_lengths": {"encoder": ln},
        }
        for i, ln in enumerate(lengths)
    ]
    examples = [
        TokenizedRepairExample(
            row["instance_id"],
            (2, *(10 + (j % 20) for j in range(ln))),
            (2, 30, 31),
            (50 + i, 60, 107),
            metadata={"task_type": row["task_type"], "source": row["source"]},
        )
        for i, (row, ln) in enumerate(zip(raw, lengths, strict=True))
    ]
    schedule = build_shared_schedule(
        canonical_schedule_records(raw),
        dataset_identity={"fixture": "length_pack"},
        effective_batch=8,
    )
    window_items = [examples[i] for i in schedule.window(0)]
    window_records = list(schedule.records[:8])

    item_groups, record_groups = pack_window_microbatches(
        window_items,
        window_records,
        max_microbatch_size=4,
        max_microbatch_tokens=64,
    )
    # 4 short items pack into one 4-item micro-batch; medium pair packs together; long items go solo.
    assert [len(g) for g in item_groups] == [4, 2, 1, 1]
    assert {item.instance_id for g in item_groups for item in g} == {
        item.instance_id for item in window_items
    }
    assert len(record_groups) == len(item_groups)

    # Verify exact mathematical equivalence of loss, grad norm, and weights vs batch_size=1 accumulation=8.
    model, policy = fixture_model("2base-2it")
    initial = copy.deepcopy(model.state_dict())
    bs1_trace: list[dict[str, Any]] = []
    run_matrix_stage(
        model,
        examples,
        [],
        "joint",
        policy=copy.deepcopy(policy),
        max_steps=1,
        freeze_decoder=False,
        start_global_step=0,
        batch_size=1,
        accumulation=8,
        run_dir=tmp_path / "bs1",
        compatibility={"model_key": "2base-2it"},
        shared_schedule=schedule,
        progress=bs1_trace.append,
    )
    expected = copy.deepcopy(model.state_dict())

    model.load_state_dict(initial)
    trace: list[dict[str, Any]] = []
    run_matrix_stage(
        model,
        examples,
        [],
        "joint",
        policy=copy.deepcopy(policy),
        max_steps=1,
        freeze_decoder=False,
        start_global_step=0,
        batch_size=2,
        accumulation=4,
        max_microbatch_size=4,
        max_microbatch_tokens=64,
        run_dir=tmp_path / "packed",
        compatibility={"model_key": "2base-2it"},
        shared_schedule=schedule,
        progress=trace.append,
    )
    assert trace[0]["instance_ids"] == [r.instance_id for r in window_records]
    assert trace[0]["micro_batches"] == 4
    assert trace[0]["max_microbatch_size"] == 4
    assert trace[0]["train_loss"] == pytest.approx(bs1_trace[0]["train_loss"], rel=1e-5, abs=1e-6)
    assert trace[0]["gradient_norm"] == pytest.approx(
        bs1_trace[0]["gradient_norm"], rel=1e-4, abs=1e-5
    )
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=1e-3, atol=5e-4)

    # Verify gradual OOM backoff (4 -> 2 -> ok without dropping all the way to 1) preserves weights.
    model.load_state_dict(initial)
    original_loss = matrix.compute_step_loss
    oom_injected = [False]

    def fail_on_bs4(*args, **kwargs):
        batch = args[1]
        if not oom_injected[0] and batch.labels.shape[0] >= 4:
            oom_injected[0] = True
            raise torch.cuda.OutOfMemoryError("simulated OOM on bs=4")
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(matrix, "compute_step_loss", fail_on_bs4)
    oom_trace: list[dict[str, Any]] = []
    run_matrix_stage(
        model,
        examples,
        [],
        "joint",
        policy=copy.deepcopy(policy),
        max_steps=1,
        freeze_decoder=False,
        start_global_step=0,
        batch_size=2,
        accumulation=4,
        max_microbatch_size=4,
        max_microbatch_tokens=64,
        run_dir=tmp_path / "oom_backoff",
        compatibility={"model_key": "2base-2it"},
        shared_schedule=schedule,
        progress=oom_trace.append,
    )
    assert oom_injected[0]
    assert oom_trace[0]["max_microbatch_size"] == 2
    assert oom_trace[0]["micro_batches"] == 6
    assert oom_trace[0]["train_loss"] == pytest.approx(
        bs1_trace[0]["train_loss"], rel=1e-5, abs=1e-6
    )
    assert oom_trace[0]["gradient_norm"] == pytest.approx(
        bs1_trace[0]["gradient_norm"], rel=1e-4, abs=1e-5
    )
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=1e-3, atol=5e-4)
