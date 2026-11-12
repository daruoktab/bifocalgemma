"""Exercise shared windows through real tiny matrix stages, checkpoints and loss."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import Any

import pytest
import torch

from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.models.v2.native import NativeTinyModel
from bifocalgemma.training.policy import configure_policy_model
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.schedule import build_shared_schedule, canonical_schedule_records
from bifocalgemma.v3.attention import configure_full_rank_cross_attention
from bifocalgemma.v3.checkpoint import restore_matrix_checkpoint
from bifocalgemma.v3.constants import MATRIX_MODELS
from bifocalgemma.v3.matrix import make_experiment_spec, run_matrix_stage


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
    import bifocalgemma.v3.matrix as matrix

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
    original = matrix.compute_v3_step_loss
    calls = 0

    def fail_second_microbatch(*args, **kwargs):
        nonlocal calls
        calls += 1
        loss = original(*args, **kwargs)
        if calls == 2:
            (loss * 0.3).backward()
            raise torch.cuda.OutOfMemoryError("injected after partial backward")
        return loss

    monkeypatch.setattr(matrix, "compute_v3_step_loss", fail_second_microbatch)
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
    import bifocalgemma.v3.matrix as matrix
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
