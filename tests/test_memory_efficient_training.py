"""Conservative memory paths preserve visibility, loss, gradients, and resume."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.memory_efficient import (
    configure_memory_efficient_training,
    tiled_attention,
)
from bifocalgemma.models.native import build_native_tiny_model
from bifocalgemma.training.objectives import chunked_linear_cross_entropy, patch_cross_entropy
from bifocalgemma.training.offline_sft import evaluate_sft_loss, fit_offline_sft
from bifocalgemma.training.policy import configure_policy_model, default_training_profiles
from bifocalgemma.training.telemetry import TrainingDashboard, render_dashboard


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(index):
    torch.manual_seed(17)
    model = build_native_tiny_model(EXPERIMENT_SPECS[index])
    family = "decoder_only" if index >= 4 else "symmetric" if index < 2 else "asymmetric"
    policy = default_training_profiles()[family]
    policy["rank"] = 2
    configure_policy_model(model, policy)
    # Nonzero B: test encoder/input/A gradients, not only zero-initialized B.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.normal_(std=0.003)
    return model, policy


def examples(index):
    return [
        TokenizedRepairExample(
            "one", (2, 3, 4, 5) if index < 4 else (), (2, 7), (8, 9, 10), {"source": "a"}
        ),
        TokenizedRepairExample(
            "two", (2, 5) if index < 4 else (), (2, 7), (9, 10), {"source": "b"}
        ),
    ]


@pytest.mark.parametrize("index", range(6))
def test_full_stack_loss_logits_and_adapter_gradients_match_eager(index):
    eager, _ = make_model(index)
    optimized = copy.deepcopy(eager)
    configure_memory_efficient_training(optimized, {"query_chunk_size": 3, "loss_chunk_size": 2})
    batch = RepairCollator()(examples(index))
    eager.train()
    optimized.train()
    reference = eager(batch)
    actual_logits = optimized(batch)
    assert reference.logits is not None and actual_logits.logits is not None
    torch.testing.assert_close(actual_logits.logits, reference.logits, atol=2e-5, rtol=2e-5)
    loss_only = optimized.forward_loss(batch)
    assert reference.loss is not None and loss_only.loss is not None
    assert loss_only.logits is None
    torch.testing.assert_close(loss_only.loss, reference.loss, atol=2e-6, rtol=2e-6)
    reference.loss.backward()
    loss_only.loss.backward()
    ref_params, opt_params = dict(eager.named_parameters()), dict(optimized.named_parameters())
    assert ref_params.keys() == opt_params.keys()
    for name, parameter in ref_params.items():
        expected, actual = parameter.grad, opt_params[name].grad
        assert (expected is None) == (actual is None), name
        if expected is not None:
            assert actual is not None
            assert torch.isfinite(actual).all()
            # Tiling changes floating-point summation, notably near cancellations.
            # Bound both vector relative error and worst element relative to gradient scale.
            difference = actual - expected
            assert difference.norm() <= 3e-4 * expected.norm() + 2e-6, name
            assert difference.abs().max() <= 3e-4 * expected.abs().max() + 2e-6, name
    assert optimized.lm_head.weight is optimized.decoder.embed_tokens.weight
    if optimized.encoder is not None:
        assert (optimized.encoder.embed_tokens.weight is optimized.decoder.embed_tokens.weight) == (
            index < 2
        )
        assert (
            optimized.encoder.embed_tokens_per_layer.weight
            is not optimized.decoder.embed_tokens_per_layer.weight
        )


@pytest.mark.parametrize("head_dim", [256, 512])
def test_tiles_keep_full_key_softmax_and_gradients_with_gqa(head_dim):
    torch.manual_seed(3)
    q = (torch.randn(1, 8, 9, head_dim, dtype=torch.float64) * 0.1).requires_grad_()
    k = (torch.randn(1, 2, 15, head_dim, dtype=torch.float64) * 0.1).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    mask = torch.zeros(1, 1, 9, 15, dtype=torch.float64)
    mask[:, :, :, :2] = float("-inf")
    keys, values = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
    scores = q @ keys.transpose(-1, -2) + mask
    reference = scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ values
    tiled = tiled_attention(q, k, v, chunk_size=3, attention_mask=mask)
    torch.testing.assert_close(tiled, reference, atol=1e-7, rtol=1e-7)
    direction = torch.randn_like(reference)
    left = torch.autograd.grad((reference * direction).sum(), (q, k, v), retain_graph=True)
    right = torch.autograd.grad((tiled * direction).sum(), (q, k, v))
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_head_softcap_and_ignored_labels_preserve_loss_and_gradients(dtype):
    torch.manual_seed(7)
    h = torch.randn(2, 13, 32, dtype=dtype, requires_grad=True)
    w = torch.randn(128, 32, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 128, (2, 13))
    labels[:, :4] = -100
    labels[0, -2:] = -100
    logits = 30 * torch.tanh(torch.nn.functional.linear(h, w) / 30)
    expected = patch_cross_entropy(logits, labels)
    actual = chunked_linear_cross_entropy(h, w, labels, chunk_size=3, softcap=30)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    a = torch.autograd.grad(expected, (h, w), retain_graph=True)
    b = torch.autograd.grad(actual, (h, w))
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(
            left,
            right,
            atol=0.002 if dtype == torch.bfloat16 else 2e-6,
            rtol=0.03 if dtype == torch.bfloat16 else 2e-5,
        )
    ignored = chunked_linear_cross_entropy(
        h, w, torch.full_like(labels, -100), chunk_size=3, softcap=30
    )
    assert ignored.item() == 0 and torch.isfinite(ignored)


def test_optimized_long_window_causality_context_and_all_masked_backward():
    model, _ = make_model(2)
    configure_memory_efficient_training(model, {"query_chunk_size": 64, "loss_chunk_size": 32})
    model.eval()
    ids = torch.randint(3, 100, (1, 513))
    valid = torch.ones_like(ids, dtype=torch.bool)
    enc = torch.tensor([[2, 3, 4]])
    enc_valid = enc != 0
    with torch.no_grad():
        hidden = model.encode_context(enc, enc_valid)
        original = model.decode_logits(ids, valid, hidden, enc_valid)
        changed = ids.clone()
        changed[:, -1] = 99
        torch.testing.assert_close(
            original[:, :-1], model.decode_logits(changed, valid, hidden, enc_valid)[:, :-1]
        )
        changed_context = model.encode_context(torch.tensor([[2, 3, 9]]), enc_valid)
        assert not torch.equal(
            original, model.decode_logits(ids, valid, changed_context, enc_valid)
        )
    model.train()
    empty = model.encode_context(enc, torch.zeros_like(enc_valid))
    masked = torch.zeros_like(valid)
    result = model.decode_logits(ids, masked, empty, torch.zeros_like(enc_valid))
    assert torch.isfinite(result).all()
    result.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("index", [0, 2])
def test_optimized_resume_and_validation_loss_only(index, tmp_path):
    first, policy = make_model(index)
    second, _ = make_model(index)
    for model in (first, second):
        configure_memory_efficient_training(model, {"query_chunk_size": 3, "loss_chunk_size": 2})
    data = examples(index)
    events = []
    opts = dict(
        epochs=3,
        batch_size=1,
        accumulation=2,
        training_policy=policy,
        eval_every=2,
        save_every=2,
        event_callback=events.append,
    )
    full = fit_offline_sft(first, data, data, tmp_path / "full", {}, max_steps=2, **opts)
    half = fit_offline_sft(second, data, data, tmp_path / "staged", {}, max_steps=1, **opts)
    restored, _ = make_model(index)
    configure_memory_efficient_training(restored, {"query_chunk_size": 3, "loss_chunk_size": 2})
    resumed = fit_offline_sft(
        restored,
        data,
        data,
        tmp_path / "staged",
        {},
        max_steps=2,
        resume_from=half["last_checkpoint"],
        **opts,
    )
    assert resumed["global_step"] == full["global_step"]
    for name, p in first.named_parameters():
        torch.testing.assert_close(p, dict(restored.named_parameters())[name], atol=1e-6, rtol=1e-6)
    stages = {r["stage"] for r in events}
    assert {
        "encoder_forward",
        "decoder_forward",
        "loss_forward",
        "backward_started",
        "validation_started",
        "validation_progress",
        "validation_complete",
        "checkpoint_complete",
        "completed",
    } <= stages
    assert (tmp_path / "full" / "events.jsonl").exists()
    assert (
        json.loads((tmp_path / "full" / "training_report.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "COMPLETED"
    )
    # Mismatched memory policy fails before mutating model or run metadata.
    wrong, _ = make_model(index)
    configure_memory_efficient_training(wrong, {"query_chunk_size": 4, "loss_chunk_size": 2})
    before = (tmp_path / "staged" / "run_config.json").read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        fit_offline_sft(
            wrong,
            data,
            data,
            tmp_path / "staged",
            {},
            max_steps=3,
            resume_from=half["last_checkpoint"],
            **opts,
        )
    assert (tmp_path / "staged" / "run_config.json").read_bytes() == before
    # Validation must not request dense logits.
    restored.decode_logits = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("dense logits"))
    evaluation = evaluate_sft_loss(restored, data, batch_size=1, device="cpu")
    assert evaluation["examples"] == 2 and set(evaluation["by_source"]) == {"a", "b"}


def test_failure_records_stage_lengths_ids_and_no_completion(tmp_path):
    model, _ = make_model(0)
    configure_memory_efficient_training(model)

    def fail(batch):
        raise RuntimeError("simulated allocation failure")

    model.forward_loss = fail
    events = []
    with pytest.raises(RuntimeError, match="allocation failure"):
        fit_offline_sft(
            model, examples(0), examples(0), tmp_path / "failed", {}, event_callback=events.append
        )
    failure = json.loads((tmp_path / "failed" / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED" and failure["global_step"] == 0
    assert failure["last_batch"]["instance_ids"]
    assert failure["last_batch"]["encoder_tokens"] > 0
    assert failure["last_checkpoint"] is None
    assert not any(e["stage"] == "completed" for e in events)
    assert not (tmp_path / "failed" / "training_report.json").exists()


def test_monitor_escapes_errors_updates_validation_and_exports_html(tmp_path):
    outputs = []
    mo = SimpleNamespace(Html=lambda value: value, output=SimpleNamespace(replace=outputs.append))
    dashboard = TrainingDashboard(mo, experiment="test", refresh_seconds=0)
    dashboard({"stage": "run_started", "run_dir": str(tmp_path), "step": 0, "max_steps": 2})
    dashboard({"stage": "optimizer_step_complete", "step": 1, "train_loss": 0.8})
    dashboard(
        {
            "stage": "validation_started",
            "step": 1,
            "eval_examples_done": 0,
            "eval_examples_total": 7,
        }
    )
    dashboard(
        {
            "stage": "validation_progress",
            "step": 1,
            "eval_examples_done": 3,
            "eval_examples_total": 7,
            "eval_running_loss": 0.7,
        }
    )
    assert "3 / 7" in outputs[-1]
    dashboard({"stage": "validation_complete", "step": 1, "dev_loss": 0.6})
    assert dashboard.history[0]["eval"]["loss"] == 0.6
    dashboard({"stage": "failed", "error_type": "OOM", "error": "<script>bad</script>"})
    content = (tmp_path / "monitor.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in content and "<script>bad" not in content
    assert (
        "loss" not in render_dashboard({}, [], []).lower()
        or "kurva" in render_dashboard({}, [], []).lower()
    )


def test_preflight_changes_neither_weights_rng_nor_dataset():
    from bifocalgemma.models.memory_efficient import run_training_memory_preflight

    model, _ = make_model(2)
    configure_memory_efficient_training(model, {"query_chunk_size": 3, "loss_chunk_size": 2})
    data = examples(2)
    model.eval()
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    events = []
    result = run_training_memory_preflight(model, data, progress=events.append)
    assert result["status"] == "PASS" and result["optimizer_updates"] == 0
    assert result["rows"][-1]["shape_only"] is True
    assert model.training is False and len(data) == 2
    assert torch.equal(torch.get_rng_state(), rng)
    for name, p in model.named_parameters():
        assert torch.equal(p, weights[name]) and p.grad is None


@pytest.mark.parametrize("index", range(6))
def test_mixed_precision_backbone_fp32_adapters_match_native(index):
    eager, _ = make_model(index)
    eager.to(torch.bfloat16)
    for name, parameter in eager.named_parameters():
        if name.endswith((".lora_A", ".lora_B")):
            parameter.data = parameter.data.float()
    optimized = copy.deepcopy(eager)
    configure_memory_efficient_training(optimized, {"query_chunk_size": 3, "loss_chunk_size": 2})
    batch = RepairCollator()(examples(index))
    reference = eager(batch)
    actual = optimized.forward_loss(batch)
    assert reference.loss is not None and actual.loss is not None
    torch.testing.assert_close(actual.loss, reference.loss, atol=2e-6, rtol=2e-6)
    reference.loss.backward()
    actual.loss.backward()
    squared_error = squared_reference = 0.0
    for name, p in eager.named_parameters():
        q = dict(optimized.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert q.grad is not None
            assert torch.isfinite(q.grad).all()
            squared_error += float((p.grad.float() - q.grad.float()).square().sum())
            squared_reference += float(p.grad.float().square().sum())
    assert squared_error <= 1e-4 * max(squared_reference, 1e-12)


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("head_dim", [256, 512])
def test_bf16_many_query_tiles_do_not_accumulate_bf16_kv_rounding(kv_heads, head_dim):
    torch.manual_seed(23)
    q = (torch.randn(1, 8, 65, head_dim, dtype=torch.bfloat16) * 0.1).requires_grad_()
    k = (torch.randn(1, kv_heads, 97, head_dim, dtype=torch.bfloat16) * 0.1).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    groups = 8 // kv_heads
    keys, values = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    expected = (q @ keys.transpose(-1, -2)).softmax(-1, dtype=torch.float32).to(q.dtype) @ values
    actual = tiled_attention(q, k, v, chunk_size=7)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    direction = torch.randn_like(expected)
    left = torch.autograd.grad((expected * direction).sum(), (q, k, v), retain_graph=True)
    right = torch.autograd.grad((actual * direction).sum(), (q, k, v))
    for reference, result in zip(left, right, strict=True):
        relative = (reference.float() - result.float()).norm() / reference.float().norm().clamp_min(
            1e-8
        )
        assert relative < 0.005
