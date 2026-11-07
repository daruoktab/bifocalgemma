import copy
import json
from typing import cast

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

from bifocalgemma.config import EXPERIMENT_NAMES, get_v2_experiment
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.inference_cache import InferenceCache, cached_logits
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.models.v2.alignment import NormalizedAlignment
from bifocalgemma.models.v2.native import build_native_tiny_model
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.policy import build_training_groups, configure_policy_model
from bifocalgemma.training.recipes import unified_recipe
from tests.test_memory_efficient_training import examples


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("index", range(6))
def test_v2_alignment_groups_masks_and_gradients(index):
    model = build_native_tiny_model(get_v2_experiment(EXPERIMENT_NAMES[index]))
    policy = unified_recipe(
        "decoder_only" if index >= 4 else "symmetric" if index < 2 else "asymmetric"
    )
    configure_policy_model(model, policy)
    groups = build_training_groups(model, policy, total_steps=200)
    assert len(groups) == (1 if index >= 4 else 3)
    assert model.lm_head.weight is model.decoder.embed_tokens.weight
    if index < 4:
        assert model.encoder is not None
        assert isinstance(model.encoder_projection, NormalizedAlignment)
        assert model.encoder.embed_tokens.weight is not model.decoder.embed_tokens.weight
        p = model.encoder_projection.linear.weight.detach()
        if p.shape[0] == p.shape[1]:
            torch.testing.assert_close(p, torch.eye(p.shape[0]))
        else:
            gram = p @ p.T if p.shape[0] < p.shape[1] else p.T @ p
            torch.testing.assert_close(gram, torch.eye(gram.shape[0]), atol=1e-5, rtol=1e-5)
        assert not model.encoder.embed_tokens.weight.requires_grad
    model.train()
    reference = copy.deepcopy(model)
    configure_memory_efficient_training(model, {"query_chunk_size": 3, "loss_chunk_size": 2})
    batch = RepairCollator()(examples(index))
    eager = reference(batch)
    actual = model(batch)
    valid = batch.decoder_attention_mask.bool()
    torch.testing.assert_close(actual.logits[valid], eager.logits[valid], atol=2e-5, rtol=2e-5)
    actual.loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    if index < 4:
        assert isinstance(model.encoder_projection, NormalizedAlignment)
        assert model.encoder_projection.norm.weight.grad is not None


@pytest.mark.parametrize("index", range(6))
def test_cached_logits_match_full_prefix_and_cross_window(index):
    model = build_native_tiny_model(get_v2_experiment(EXPERIMENT_NAMES[index])).eval()
    # Exercise eviction at tiny lengths while retaining native global/local pattern.
    for layer in model.decoder.layers:
        attention = cast(Gemma4TextAttention, layer.self_attn)
        if attention.sliding_window:
            attention.sliding_window = 3
    configure_memory_efficient_training(model, {"query_chunk_size": 3})
    ids = torch.tensor([[2, 7, 8, 9]])
    enc = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        states = model.encode_context(enc, torch.ones_like(enc)) if index < 4 else None
        mask = torch.ones_like(enc) if index < 4 else None
        cache = InferenceCache()
        for j in range(5):
            full = model.decode_next_logits(ids, torch.ones_like(ids), states, mask)
            actual = cached_logits(model, ids if j == 0 else ids[:, -1:], cache, states, mask)
            torch.testing.assert_close(actual, full, atol=3e-5, rtol=3e-5)
            ids = torch.cat((ids, torch.tensor([[10 + j]])), dim=1)
        assert all(
            v[0].shape[2] <= 3
            for i, v in cache.layers.items()
            if model.decoder.layers[i].self_attn.sliding_window
        )


def test_alignment_stays_fp32_after_bf16_conversion():
    model = build_native_tiny_model(get_v2_experiment(EXPERIMENT_NAMES[2])).to(torch.bfloat16)
    assert isinstance(model.encoder_projection, NormalizedAlignment)
    assert all(p.dtype == torch.float32 for p in model.encoder_projection.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Direct CUDA construction regression")
@pytest.mark.parametrize("input_width,output_width", [(8, 8), (8, 12), (12, 8)])
def test_alignment_initialization_with_cuda_default_device(input_width, output_width):
    reference = NormalizedAlignment(input_width, output_width)
    with torch.device("cuda"):
        alignment = NormalizedAlignment(input_width, output_width)
    assert all(p.device.type == "cuda" and p.dtype == torch.float32 for p in alignment.parameters())
    torch.testing.assert_close(alignment.linear.weight.cpu(), reference.linear.weight, rtol=0, atol=0)
    hidden = torch.randn(1, 3, input_width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    result = alignment(hidden)
    assert result.dtype == torch.bfloat16
    result.float().square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert alignment.norm.weight.grad is not None


def test_encoder_bidirectional_window_boundary_matches_native():
    model = build_native_tiny_model(get_v2_experiment(EXPERIMENT_NAMES[0])).eval()
    ids = torch.randint(2, 100, (1, 260))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        expected = model.encode_context(ids, mask)
        configure_memory_efficient_training(model, {"query_chunk_size": 64})
        actual = model.encode_context(ids, mask)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("index", range(4))
def test_empty_and_fully_masked_context_matches_self_only(index):
    model = build_native_tiny_model(get_v2_experiment(EXPERIMENT_NAMES[index])).eval()
    configure_memory_efficient_training(model)
    ids = torch.tensor([[2, 5, 6]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        hidden = model.encode_context(
            torch.tensor([[2, 3, 4]]), torch.zeros((1, 3), dtype=torch.bool)
        )
        full = model.decode_next_logits(ids, mask, hidden, torch.zeros((1, 3), dtype=torch.bool))
        empty = model.decode_next_logits(ids, mask)
        torch.testing.assert_close(full, empty, rtol=1e-5, atol=1e-5)
        cached = cached_logits(
            model, ids, InferenceCache(), hidden, torch.zeros((1, 3), dtype=torch.bool)
        )
        torch.testing.assert_close(cached, empty, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("index", [0, 2, 4])
def test_v3_wall_time_checkpoint_resume_is_exact(tmp_path, index):
    family = "decoder_only" if index == 4 else "symmetric" if index == 0 else "asymmetric"
    policy = unified_recipe(family)
    policy.update(rank=2, alpha=4)

    def make():
        torch.manual_seed(71)
        model = build_native_tiny_model(get_v2_experiment(EXPERIMENT_NAMES[index]))
        configure_policy_model(model, policy)
        return model

    full, interrupted = make(), make()
    data = examples(index)
    options = dict(
        max_steps=2,
        epochs=2,
        batch_size=1,
        accumulation=1,
        eval_batch_size=1,
        training_policy=policy,
        save_every=1000,
        save_interval_seconds=1e-9,
        eval_every=1000,
        event_callback=lambda e: None,
    )
    fit_offline_sft(full, data, data, tmp_path / "full", {}, **options)
    saved = []

    def stop(path):
        saved.append(path)
        raise KeyboardInterrupt("Molab session interruption fixture")

    with pytest.raises(KeyboardInterrupt):
        fit_offline_sft(
            interrupted,
            data,
            data,
            tmp_path / "interrupted",
            {},
            checkpoint_callback=stop,
            **options,
        )
    assert (
        json.loads((__import__("pathlib").Path(saved[0]) / "manifest.json").read_text())["format"]
        == "bifocalgemma_normalized_sft_v3"
    )
    restored = make()
    fit_offline_sft(
        restored, data, data, tmp_path / "interrupted", {}, resume_from=saved[0], **options
    )
    for name, expected in full.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, atol=0, rtol=0)
