"""Offline unit and parity tests for BifocalGemma v4 architecture upgrades."""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import bifocalgemma.config as config_mod
from bifocalgemma.config import make_experiment_spec
from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
from bifocalgemma.modeling_bifocalgemma import BifocalGemmaForCausalLM, NativeTinyModel
from bifocalgemma.models.bridge import SwiGLUDualGateBridge
from bifocalgemma.models.cross_attention import (
    Bf16FullLinear,
    FullCrossCachedAttention,
    FullCrossTiledAttention,
    _compute_per_row_cross_query,
    configure_full_rank_cross_attention,
)
from bifocalgemma.models.inference_cache import InferenceCache
from bifocalgemma.models.memory_efficient import (
    TiledEncoderAttention,
    configure_memory_efficient_training,
    sdpa_attention,
    sdpa_cross_attention,
    tiled_attention,
)
from bifocalgemma.training.runtime import critical_attention_beta

# ---------------------------------------------------------------------------
# (a) Zero .tolist() / Vectorized per-row beta_N parity
# ---------------------------------------------------------------------------


def test_compute_per_row_cross_query_has_no_host_sync_calls():
    """AST check: _compute_per_row_cross_query must not call .tolist(), .item(), or torch.tensor()."""
    source = inspect.getsource(_compute_per_row_cross_query)
    tree = ast.parse(source)
    forbidden_attrs = {"tolist", "item"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attrs, (
                    f"Forbidden host-sync call .{node.func.attr}() in _compute_per_row_cross_query"
                )
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "tensor"
                ):
                    raise AssertionError(
                        "Forbidden torch.tensor(...) H2D copy in _compute_per_row_cross_query"
                    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compute_per_row_cross_query_vectorized_parity(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
):
    """Vectorized on-device beta_N must match scalar critical_attention_beta without calling .tolist()."""
    # Guard at runtime that Tensor.tolist is never triggered
    def _forbid_tolist(self: torch.Tensor) -> Any:
        raise AssertionError("Tensor.tolist() must not be called in _compute_per_row_cross_query")

    monkeypatch.setattr(torch.Tensor, "tolist", _forbid_tolist)

    torch.manual_seed(7)
    # 4 rows: short (512 <= 2048), threshold (2048), medium (4096), zero-valid (fallback=8192)
    q = torch.randn(4, 4, 3, 16, dtype=dtype)
    enc_valid = torch.zeros(4, 5000, dtype=torch.bool)
    enc_valid[0, :512] = True
    enc_valid[1, :2048] = True
    enc_valid[2, :4096] = True
    # Row 3 stays all False -> falls back to fallback_n_enc=8192

    fallback_n = 8192
    actual = _compute_per_row_cross_query(q, enc_valid, fallback_n)
    assert actual.dtype == dtype
    assert actual.shape == q.shape

    expected_betas = [
        critical_attention_beta(512),
        critical_attention_beta(2048),
        critical_attention_beta(4096),
        critical_attention_beta(fallback_n),
    ]
    expected_scale = torch.tensor(expected_betas, dtype=torch.float32).view(4, 1, 1, 1)
    expected = (q.float() * expected_scale).to(dtype)
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    # Fast path: short sequence (<= 2048) returns input directly
    short_valid = torch.ones(4, 128, dtype=torch.bool)
    short_out = _compute_per_row_cross_query(q, short_valid, 128)
    assert short_out is q


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device test")
def test_compute_per_row_cross_query_on_cuda_stays_on_device():
    q = torch.randn(2, 4, 5, 16, device="cuda", dtype=torch.bfloat16)
    enc_valid = torch.zeros(2, 4096, device="cuda", dtype=torch.bool)
    enc_valid[0, :1024] = True
    enc_valid[1, :4096] = True
    out = _compute_per_row_cross_query(q, enc_valid, 4096)
    assert out.device.type == "cuda"
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out[0], q[0], atol=1e-3, rtol=1e-3)
    expected_b1 = critical_attention_beta(4096)
    torch.testing.assert_close(out[1], (q[1].float() * expected_b1).to(torch.bfloat16), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# (b) SDPA vs eager numerical parity & safe fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_cross_attention_matches_eager_and_handles_all_masked_rows(
    head_dim: int, kv_heads: int, dtype: torch.dtype
):
    torch.manual_seed(11)
    q = torch.randn(3, 8, 19, head_dim, dtype=dtype, requires_grad=True)
    k = torch.randn(3, kv_heads, 37, head_dim, dtype=dtype, requires_grad=True)
    v = torch.randn(3, kv_heads, 37, head_dim, dtype=dtype, requires_grad=True)

    enc_valid = torch.ones(3, 37, dtype=torch.bool)
    enc_valid[0, 20:] = False
    enc_valid[1, :] = False  # wholly masked encoder row
    enc_valid[2, :5] = False

    dec_valid = torch.ones(3, 19, dtype=torch.bool)
    dec_valid[0, :4] = False
    dec_valid[2, :] = False  # wholly masked decoder row

    scale = head_dim**-0.5
    sdpa_out = sdpa_cross_attention(q, k, v, enc_valid, dec_valid, scale, chunk_size=8, use_sdpa=True)
    eager_out = sdpa_cross_attention(q, k, v, enc_valid, dec_valid, scale, chunk_size=8, use_sdpa=False)

    assert torch.isfinite(sdpa_out).all()
    assert torch.all(sdpa_out[1] == 0.0)
    assert torch.all(sdpa_out[2] == 0.0)
    assert torch.all(sdpa_out[0, :, :4, :] == 0.0)

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(sdpa_out, eager_out, atol=tol, rtol=tol)

    # Backward parity & finite gradients even with wholly masked rows
    sdpa_grads = torch.autograd.grad(sdpa_out.float().square().sum(), (q, k, v), retain_graph=True)
    eager_grads = torch.autograd.grad(eager_out.float().square().sum(), (q, k, v))
    for g_sdpa, g_eager in zip(sdpa_grads, eager_grads, strict=True):
        assert torch.isfinite(g_sdpa).all()
        if dtype == torch.float32:
            torch.testing.assert_close(g_sdpa, g_eager, atol=1e-4, rtol=1e-4)
        else:
            rel_err = (g_sdpa.float() - g_eager.float()).norm() / g_eager.float().norm().clamp_min(1e-8)
            assert float(rel_err) < 0.03


@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_attention_structured_masks_match_tiled_eager(
    bidirectional: bool, window: int | None, dtype: torch.dtype
):
    torch.manual_seed(17)
    q = torch.randn(2, 8, 29, 64, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 29, 64, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 2, 29, 64, dtype=dtype, requires_grad=True)
    valid = torch.ones(2, 29, dtype=torch.bool)
    valid[0, :3] = False
    valid[0, -2:] = False
    valid[1, :] = False  # wholly masked row
    mask = {"valid": valid, "bidirectional": bidirectional}
    scale = 64**-0.5

    ref = tiled_attention(
        q, k, v, chunk_size=7, scaling=scale, attention_mask=mask, window=window, backend="tiled_eager_fp32"
    )
    actual = tiled_attention(
        q, k, v, chunk_size=7, scaling=scale, attention_mask=mask, window=window, backend="sdpa"
    )
    assert torch.isfinite(actual).all()
    assert torch.all(actual[1] == 0.0)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, ref, atol=tol, rtol=tol)


def test_sdpa_attention_safe_fallback_on_float64_and_custom_float_mask():
    torch.manual_seed(23)
    # 1. float64 fallback
    q64 = torch.randn(1, 4, 8, 16, dtype=torch.float64)
    k64 = torch.randn(1, 2, 8, 16, dtype=torch.float64)
    v64 = torch.randn(1, 2, 8, 16, dtype=torch.float64)
    mask_dict = {"valid": torch.ones(1, 8, dtype=torch.bool), "bidirectional": True}
    out64 = sdpa_attention(q64, k64, v64, chunk_size=4, scaling=16**-0.5, attention_mask=mask_dict)
    ref64 = tiled_attention(
        q64, k64, v64, chunk_size=4, scaling=16**-0.5, attention_mask=mask_dict, backend="tiled_eager_fp32"
    )
    torch.testing.assert_close(out64, ref64, atol=1e-7, rtol=1e-7)

    # 2. Custom float additive mask fallback
    q32 = torch.randn(1, 4, 8, 16, dtype=torch.float32)
    k32 = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    v32 = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    additive_mask = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    additive_mask[:, :, :, :2] = float("-inf")
    out32 = sdpa_attention(
        q32, k32, v32, chunk_size=4, scaling=16**-0.5, attention_mask=additive_mask
    )
    ref32 = tiled_attention(
        q32, k32, v32, chunk_size=4, scaling=16**-0.5, attention_mask=additive_mask, backend="tiled_eager_fp32"
    )
    torch.testing.assert_close(out32, ref32, atol=1e-6, rtol=1e-6)


def test_eval_tiled_encoder_attention_uses_sdpa_and_matches_eager(monkeypatch: pytest.MonkeyPatch):
    """Eval TiledEncoderAttention must automatically dispatch to SDPA and match tiled_eager_fp32."""
    import bifocalgemma.models.memory_efficient as mem_mod

    torch.manual_seed(31)
    spec = make_experiment_spec("2base-2it")
    model = NativeTinyModel(spec, tiny=True)
    configure_memory_efficient_training(model, {"query_chunk_size": 4, "loss_chunk_size": 4})
    assert model.encoder is not None
    assert isinstance(model.encoder.layers[0].self_attn, TiledEncoderAttention)
    # Default memory policy leaves attention_backend='tiled_eager_fp32'; eval must upgrade to sdpa.
    assert model.encoder.layers[0].self_attn.attention_backend == "tiled_eager_fp32"

    model.eval()
    enc_ids = torch.randint(3, 100, (2, 12))
    enc_mask = torch.ones_like(enc_ids, dtype=torch.bool)
    enc_mask[0, :3] = False

    sdpa_calls = 0
    orig_sdpa = mem_mod.sdpa_attention

    def spy_sdpa(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal sdpa_calls
        sdpa_calls += 1
        return orig_sdpa(*args, **kwargs)

    monkeypatch.setattr(mem_mod, "sdpa_attention", spy_sdpa)
    with torch.no_grad():
        sdpa_hidden = model.encode_context(enc_ids, enc_mask)
    assert sdpa_calls == len(model.encoder.layers)

    # Compare against forced eager eval across all 35 encoder layers
    for layer in model.encoder.layers:
        setattr(layer.self_attn, "_force_eager_eval", True)
    with torch.no_grad():
        eager_hidden = model.encode_context(enc_ids, enc_mask)
    torch.testing.assert_close(sdpa_hidden, eager_hidden, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# (c) Decoupled upper full_attention cross-KV & Tiled vs Cached parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("encoder_size", "decoder_size", "expected_decoupled_indices"),
    [
        ("E2B", "E2B", [19, 24, 29, 34]),
        ("E4B", "E4B", [29, 35, 41]),
    ],
)
def test_decoupled_upper_full_attention_cross_kv_topology(
    encoder_size: str, decoder_size: str, expected_decoupled_indices: list[int]
):
    cfg = BifocalGemmaConfig(
        encoder_size=encoder_size,
        decoder_size=decoder_size,
        tiny=True,
        decouple_full_attn_cross_kv=True,
    )
    model = BifocalGemmaForCausalLM(cfg)
    layers = cast(Any, model.decoder).layers

    full_owner = None
    actual_decoupled: list[int] = []
    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        assert isinstance(attn, FullCrossTiledAttention)
        if not attn.is_kv_shared_layer:
            assert attn.owns_cross_kv is True
            assert isinstance(attn.cross_k_proj, Bf16FullLinear)
            assert isinstance(attn.cross_v_proj, Bf16FullLinear)
            if attn.layer_type == "full_attention":
                full_owner = attn
        elif attn.layer_type == "full_attention":
            actual_decoupled.append(idx)
            assert attn.owns_cross_kv is True
            assert isinstance(attn.cross_k_proj, Bf16FullLinear)
            assert isinstance(attn.cross_v_proj, Bf16FullLinear)
            assert hasattr(attn, "cross_k_norm")
            assert hasattr(attn, "cross_v_norm")
            assert full_owner is not None
            # Must be independent parameters, NOT aliased to the owner layer
            assert attn.cross_k_proj.weight is not full_owner.cross_k_proj.weight
            assert attn.cross_v_proj.weight is not full_owner.cross_v_proj.weight
        else:
            # Sliding KV-shared layers remain cross-KV shared
            assert attn.owns_cross_kv is False
            assert not hasattr(attn, "cross_k_proj")
            assert not hasattr(attn, "cross_v_proj")

    assert actual_decoupled == expected_decoupled_indices

    # Verify decouple_full_attn_cross_kv=False cleanly restores v3 shared behavior
    info_disabled = configure_full_rank_cross_attention(model, decouple_full_attn_cross_kv=False)
    assert info_disabled["decoupled_full_attn_cross_kv_layers"] == 0
    for idx in expected_decoupled_indices:
        attn = layers[idx].self_attn
        assert attn.owns_cross_kv is False
        assert not hasattr(attn, "cross_k_proj")


def test_decoupled_full_attn_cross_kv_tiled_vs_cached_parity_and_independence():
    """FullCrossTiledAttention and FullCrossCachedAttention must match with distinct decoupled upper full_attn weights."""
    torch.manual_seed(41)
    cfg = BifocalGemmaConfig(
        encoder_size="E2B",
        decoder_size="E2B",
        tiny=True,
        gate_init_tanh=0.25,
        decouple_full_attn_cross_kv=True,
    )
    model = BifocalGemmaForCausalLM(cfg).eval()
    dec_layers = cast(Any, model.decoder).layers

    # Give each decoupled upper full_attention layer distinct weights & non-zero cross_gate_proj
    with torch.no_grad():
        for idx in (19, 24, 29, 34):
            attn = dec_layers[idx].self_attn
            attn.cross_k_proj.weight.add_(torch.randn_like(attn.cross_k_proj.weight) * 0.5)
            attn.cross_v_proj.weight.add_(torch.randn_like(attn.cross_v_proj.weight) * 0.5)
            attn.cross_gate.fill_(math.atanh(0.4 + 0.05 * (idx // 10)))
            attn.cross_gate_proj.normal_(std=0.2)

    enc_ids = torch.randint(5, 100, (2, 10))
    enc_mask = torch.ones_like(enc_ids, dtype=torch.bool)
    enc_mask[0, 6:] = False

    # Left-padded batch of 2 sequences
    prompt_ids = torch.tensor([[0, 0, 2, 11, 12], [2, 13, 14, 15, 16]])
    prompt_mask = torch.tensor([[False, False, True, True, True], [True, True, True, True, True]])

    with torch.no_grad():
        enc_states = model.encode_context(enc_ids, enc_mask)

        # 1. Verify perturbing decoupled L24 cross_k_proj direction actually changes logits (not sharing L14!)
        baseline_logits = model.decode_logits(prompt_ids, prompt_mask, enc_states, enc_mask)
        saved_l24_k = dec_layers[24].self_attn.cross_k_proj.weight.clone()
        dec_layers[24].self_attn.cross_k_proj.weight.add_(
            torch.randn_like(dec_layers[24].self_attn.cross_k_proj.weight) * 2.0
        )
        perturbed_logits = model.decode_logits(prompt_ids, prompt_mask, enc_states, enc_mask)
        assert not torch.allclose(baseline_logits, perturbed_logits, atol=1e-4)
        dec_layers[24].self_attn.cross_k_proj.weight.copy_(saved_l24_k)

        # 1b. Verify SDPA cross-attention matches forced eager cross-attention on full model
        for layer in dec_layers:
            setattr(layer.self_attn, "_force_eager_cross", True)
        eager_cross_logits = model.decode_logits(prompt_ids, prompt_mask, enc_states, enc_mask)
        for layer in dec_layers:
            setattr(layer.self_attn, "_force_eager_cross", False)
        torch.testing.assert_close(baseline_logits, eager_cross_logits, atol=1e-4, rtol=1e-4)

        # 2. Verify prefill + 3-step decode logits parity between FullCrossTiledAttention and FullCrossCachedAttention
        full_ids = torch.cat([prompt_ids, torch.tensor([[20, 21, 22], [30, 31, 32]])], dim=1)
        full_mask = torch.cat([prompt_mask, torch.ones((2, 3), dtype=torch.bool)], dim=1)
        tiled_all_logits = model.decode_logits(full_ids, full_mask, enc_states, enc_mask)

        cache = InferenceCache()
        orig_classes = [layer.self_attn.__class__ for layer in dec_layers]
        try:
            for layer in dec_layers:
                object.__setattr__(layer.self_attn, "__class__", FullCrossCachedAttention)

            # Prefill step
            setattr(cache, "step_valid_mask", prompt_mask)
            prefill_pos = model._positions(prompt_mask)
            dec_out = model.decoder(
                input_ids=prompt_ids,
                position_ids=prefill_pos,
                attention_mask={"full_attention": None, "sliding_attention": None},
                use_cache=False,
                inference_cache=cache,
                encoder_hidden_states=enc_states,
                encoder_valid=enc_mask,
            )
            cached_prefill = model.lm_head(dec_out.last_hidden_state)
            cap = model.decoder.config.final_logit_softcapping
            if cap is not None:
                cached_prefill = torch.tanh(cached_prefill / cap) * cap
            cache.offset += prompt_ids.shape[1]

            # Verify decoupled upper full_attention layers cached their own distinct cross KV
            l14_cross = cache.layers[14][3]
            for idx in (19, 24, 29, 34):
                assert idx in cache.layers
                layer_cross = cache.layers[idx][3]
                assert layer_cross is not None
                assert not torch.allclose(layer_cross[0], l14_cross[0], atol=1e-3)

            torch.testing.assert_close(
                cached_prefill[prompt_mask],
                tiled_all_logits[:, :5][prompt_mask],
                atol=1e-4,
                rtol=1e-4,
            )

            # 3 incremental decode steps
            cur_mask = prompt_mask
            for step in range(3):
                step_token = full_ids[:, 5 + step : 6 + step]
                cur_mask = torch.cat([cur_mask, torch.ones((2, 1), dtype=torch.bool)], dim=1)
                step_pos = (cur_mask.long().sum(dim=-1, keepdim=True) - 1).clamp_min(0)
                setattr(cache, "step_valid_mask", torch.ones((2, 1), dtype=torch.bool))
                step_out = model.decoder(
                    input_ids=step_token,
                    position_ids=step_pos,
                    attention_mask={"full_attention": None, "sliding_attention": None},
                    use_cache=False,
                    inference_cache=cache,
                    encoder_hidden_states=enc_states,
                    encoder_valid=enc_mask,
                )
                step_logits = model.lm_head(step_out.last_hidden_state)
                if cap is not None:
                    step_logits = torch.tanh(step_logits / cap) * cap
                cache.offset += 1
                torch.testing.assert_close(
                    step_logits[:, 0],
                    tiled_all_logits[:, 5 + step],
                    atol=1e-4,
                    rtol=1e-4,
                )
        finally:
            for layer, cls in zip(dec_layers, orig_classes, strict=True):
                object.__setattr__(layer.self_attn, "__class__", cls)

        # 3. End-to-end model.generate parity (cached vs uncached) with decoupled weights
        gen_cached = model.generate(
            input_ids=prompt_ids,
            encoder_input_ids=enc_ids,
            encoder_attention_mask=enc_mask,
            attention_mask=prompt_mask,
            max_new_tokens=4,
            use_cache=True,
        )
        gen_uncached = model.generate(
            input_ids=prompt_ids,
            encoder_input_ids=enc_ids,
            encoder_attention_mask=enc_mask,
            attention_mask=prompt_mask,
            max_new_tokens=4,
            use_cache=False,
        )
        assert torch.equal(gen_cached, gen_uncached)


def test_decoupled_full_attn_cross_kv_fp32_master_weights_and_gradients():
    """Decoupled upper full_attention cross-KV projections must stay FP32 under BF16 cast and receive grads."""
    torch.manual_seed(43)
    cfg = BifocalGemmaConfig(
        encoder_size="E2B",
        decoder_size="E2B",
        tiny=True,
        gate_init_tanh=0.15,
        decouple_full_attn_cross_kv=True,
    )
    model = BifocalGemmaForCausalLM(cfg)
    cast(Any, model).to(dtype=torch.bfloat16)

    dec_layers = cast(Any, model.decoder).layers
    for idx in (19, 24, 29, 34):
        attn = dec_layers[idx].self_attn
        assert attn.cross_k_proj.weight.dtype == torch.float32
        assert attn.cross_v_proj.weight.dtype == torch.float32
        assert attn.cross_k_norm.weight.dtype == torch.float32

    enc_ids = torch.randint(5, 50, (2, 6))
    dec_ids = torch.randint(5, 50, (2, 5))
    out = model(input_ids=dec_ids, encoder_input_ids=enc_ids, labels=dec_ids)
    assert out.loss is not None
    out.loss.backward()

    for idx in (19, 24, 29, 34):
        attn = dec_layers[idx].self_attn
        assert attn.cross_k_proj.weight.grad is not None
        assert attn.cross_v_proj.weight.grad is not None
        assert attn.cross_k_proj.weight.grad.dtype == torch.float32
        assert attn.cross_k_proj.weight.grad.abs().sum().item() > 0.0
        assert attn.cross_v_proj.weight.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# (d) Non-redundant SwiGLUDualGateBridge gain + legacy v3 checkpoint compat
# ---------------------------------------------------------------------------


def test_swiglu_bridge_non_redundant_single_post_gain():
    bridge = SwiGLUDualGateBridge(d_enc=64, d_dec=64, init_rms_scale=1.35)
    # post_norm must have elementwise_affine=False (no redundant unit-init post_norm.weight)
    assert bridge.post_norm.elementwise_affine is False
    assert bridge.post_norm.weight is None
    assert "post_norm.weight" not in bridge.state_dict()
    assert "post_norm.weight" not in dict(bridge.named_parameters())
    assert "gamma_calib" in bridge.state_dict()
    assert bridge.gamma_calib.dtype == torch.float32
    torch.testing.assert_close(
        bridge.gamma_calib.detach(), torch.full((64,), 1.35, dtype=torch.float32)
    )

    # Verify exact single-gain scaling behavior and gradient flow
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = bridge(x)
    assert y.shape == (2, 8, 64)
    # Output RMS along last dim should equal init_rms_scale (1.35) since post_norm normalizes RMS to ~1
    rms = y.float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.full_like(rms, 1.35), atol=1e-2, rtol=1e-2)
    y.sum().backward()
    assert bridge.gamma_calib.grad is not None
    assert bridge.gamma_calib.grad.abs().sum().item() > 0.0


def test_swiglu_bridge_loads_legacy_v3_checkpoint_folding_post_norm_weight():
    """Legacy v3 state_dicts with both post_norm.weight and gamma_calib must fold cleanly."""
    torch.manual_seed(51)
    bridge = SwiGLUDualGateBridge(d_enc=32, d_dec=32, init_rms_scale=1.0)
    legacy_sd = {k: v.clone() for k, v in bridge.state_dict().items()}
    post_w = torch.linspace(0.8, 1.2, 32, dtype=torch.float32)
    gamma_w = torch.linspace(1.1, 0.9, 32, dtype=torch.float32)
    legacy_sd["post_norm.weight"] = post_w
    legacy_sd["gamma_calib"] = gamma_w

    reloaded = SwiGLUDualGateBridge(d_enc=32, d_dec=32, init_rms_scale=1.0)
    incompatible = reloaded.load_state_dict(legacy_sd, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(reloaded.gamma_calib.detach(), gamma_w * post_w, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# (e) Frozen cross_scale by default & clean GATE_INIT_TANH resolution
# ---------------------------------------------------------------------------


def test_cross_scale_frozen_by_default_and_configurable():
    spec = make_experiment_spec("2base-2it")
    model = NativeTinyModel(spec, tiny=True)
    info = configure_full_rank_cross_attention(model)
    assert info["learnable_cross_scale"] is False

    scales = [p for n, p in model.named_parameters() if n.endswith(".cross_scale")]
    assert len(scales) == len(model.decoder.layers)
    for scale in scales:
        assert scale.requires_grad is False
        assert scale.dtype == torch.float32
        torch.testing.assert_close(scale.detach(), torch.ones_like(scale))

    # Forward + backward: cross_scale must NOT accumulate grads, while cross_gate & cross_gate_proj DO
    enc_ids = torch.randint(5, 50, (1, 6))
    dec_ids = torch.randint(5, 50, (1, 4))
    enc_h = model.encode_context(enc_ids, torch.ones_like(enc_ids, dtype=torch.bool))
    logits = model.decode_logits(
        dec_ids, torch.ones_like(dec_ids, dtype=torch.bool), enc_h, torch.ones_like(enc_ids, dtype=torch.bool)
    )
    logits.square().mean().backward()
    for layer in model.decoder.layers:
        attn = cast(Any, layer.self_attn)
        assert attn.cross_scale.grad is None
        assert attn.cross_gate.grad is not None and attn.cross_gate.grad.abs().sum().item() > 0.0
        assert attn.cross_gate_proj.grad is not None

    # Explicit opt-in unfreezes cross_scale
    info_opt_in = configure_full_rank_cross_attention(model, learnable_cross_scale=True)
    assert info_opt_in["learnable_cross_scale"] is True
    assert all(
        p.requires_grad for n, p in model.named_parameters() if n.endswith(".cross_scale")
    )


def test_gate_init_tanh_resolution_dynamic_and_v4_fallback(monkeypatch: pytest.MonkeyPatch):
    spec = make_experiment_spec("2base-2it")

    # 1. Dynamically reads config.GATE_INIT_TANH when updated
    monkeypatch.setattr(config_mod, "GATE_INIT_TANH", 0.15)
    model_dyn = NativeTinyModel(spec, tiny=True)
    info_dyn = configure_full_rank_cross_attention(model_dyn)
    assert math.isclose(info_dyn["gate_init_tanh"], 0.15, rel_tol=1e-6)
    first_gate = cast(Any, model_dyn.decoder.layers[0].self_attn).cross_gate
    torch.testing.assert_close(
        torch.tanh(first_gate.detach()), torch.full_like(first_gate, 0.15), atol=1e-5, rtol=1e-5
    )

    # 2. Falls back to 0.15 when v4 is enabled even if legacy GATE_INIT_TANH is 0.05
    monkeypatch.setattr(config_mod, "GATE_INIT_TANH", 0.05)
    v4_spec = replace(spec, architecture_version="normalized_merged_v4")
    model_v4 = NativeTinyModel(v4_spec, tiny=True)
    info_v4 = configure_full_rank_cross_attention(model_v4)
    assert math.isclose(info_v4["gate_init_tanh"], 0.15, rel_tol=1e-6)

    # 3. Respects explicit config.gate_init_tanh override
    model_cfg = NativeTinyModel(spec, tiny=True)
    cast(Any, model_cfg).config = SimpleNamespace(gate_init_tanh=0.20, learnable_cross_scale=False)
    info_cfg = configure_full_rank_cross_attention(model_cfg)
    assert math.isclose(info_cfg["gate_init_tanh"], 0.20, rel_tol=1e-6)
    assert info_cfg["learnable_cross_scale"] is False


# ---------------------------------------------------------------------------
# (f) Audit regressions: head_dim=512 training OOM guard & FullCrossCachedAttention
# ---------------------------------------------------------------------------


def test_head_dim_512_routes_training_to_tiled_recompute_and_keeps_eval_and_256_on_sdpa(
    monkeypatch: pytest.MonkeyPatch,
):
    """head_dim > 256 in training must route to _TiledAttention (no O(N^2) Math SDPA tile retention),
    while eval (no_grad) and head_dim <= 256 keep the SDPA fast path.
    """
    import bifocalgemma.models.memory_efficient as mem_mod

    sdpa_calls = 0
    tiled_calls = 0
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    orig_tiled_apply = mem_mod._TiledAttention.apply

    def spy_sdpa(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal sdpa_calls
        sdpa_calls += 1
        return orig_sdpa(*args, **kwargs)

    def spy_tiled_apply(*args: Any, **kwargs: Any) -> Any:
        nonlocal tiled_calls
        tiled_calls += 1
        return orig_tiled_apply(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy_sdpa)
    monkeypatch.setattr(mem_mod._TiledAttention, "apply", staticmethod(spy_tiled_apply))

    torch.manual_seed(61)
    # 1. Training with head_dim=512 -> routes to _TiledAttention (0 Math SDPA calls)
    q512 = torch.randn(2, 4, 18, 512, dtype=torch.float32, requires_grad=True)
    k512 = torch.randn(2, 2, 18, 512, dtype=torch.float32, requires_grad=True)
    v512 = torch.randn(2, 2, 18, 512, dtype=torch.float32, requires_grad=True)
    valid = torch.ones(2, 18, dtype=torch.bool)
    valid[0, :3] = False
    valid[1, :] = False  # wholly masked row
    mask_bi = {"valid": valid, "bidirectional": True}
    mask_ca = {"valid": valid, "bidirectional": False}
    scale512 = 512**-0.5

    sdpa_calls = tiled_calls = 0
    out_bi = sdpa_attention(q512, k512, v512, chunk_size=6, scaling=scale512, attention_mask=mask_bi)
    out_ca = sdpa_attention(q512, k512, v512, chunk_size=6, scaling=scale512, attention_mask=mask_ca)
    assert sdpa_calls == 0
    assert tiled_calls == 2
    assert torch.isfinite(out_bi).all() and torch.all(out_bi[1] == 0.0)
    assert torch.isfinite(out_ca).all() and torch.all(out_ca[1] == 0.0)
    grads = torch.autograd.grad((out_bi + out_ca).square().sum(), (q512, k512, v512))
    assert all(torch.isfinite(g).all() for g in grads)

    # Cross-attention training with head_dim=512 -> routes to _TiledAttention (0 Math SDPA calls)
    kc512 = torch.randn(2, 2, 25, 512, dtype=torch.float32, requires_grad=True)
    vc512 = torch.randn(2, 2, 25, 512, dtype=torch.float32, requires_grad=True)
    enc_valid = torch.ones(2, 25, dtype=torch.bool)
    enc_valid[1, :] = False
    sdpa_calls = tiled_calls = 0
    out_cross = sdpa_cross_attention(
        q512, kc512, vc512, enc_valid, valid, scale512, chunk_size=6, use_sdpa=True
    )
    assert sdpa_calls == 0
    assert tiled_calls == 1
    assert torch.isfinite(out_cross).all() and torch.all(out_cross[1] == 0.0)
    cross_grads = torch.autograd.grad(out_cross.square().sum(), (q512, kc512, vc512))
    assert all(torch.isfinite(g).all() for g in cross_grads)

    # 2. Eval (no_grad) with head_dim=512 -> keeps SDPA fast path!
    sdpa_calls = tiled_calls = 0
    with torch.no_grad():
        eval_bi = sdpa_attention(
            q512, k512, v512, chunk_size=6, scaling=scale512, attention_mask=mask_bi
        )
        eval_cross = sdpa_cross_attention(
            q512, kc512, vc512, enc_valid, valid, scale512, chunk_size=6, use_sdpa=True
        )
    assert sdpa_calls >= 2
    assert tiled_calls == 0
    torch.testing.assert_close(eval_bi, out_bi.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(eval_cross, out_cross.detach(), atol=1e-5, rtol=1e-5)

    # 3. Training with head_dim=256 -> keeps SDPA fast path!
    q256 = torch.randn(2, 4, 10, 256, dtype=torch.float32, requires_grad=True)
    k256 = torch.randn(2, 2, 10, 256, dtype=torch.float32, requires_grad=True)
    v256 = torch.randn(2, 2, 10, 256, dtype=torch.float32, requires_grad=True)
    valid256 = torch.ones(2, 10, dtype=torch.bool)
    sdpa_calls = tiled_calls = 0
    out256 = sdpa_attention(
        q256,
        k256,
        v256,
        chunk_size=16,
        scaling=256**-0.5,
        attention_mask={"valid": valid256, "bidirectional": True},
    )
    out256_cross = sdpa_cross_attention(
        q256, k256, v256, valid256, valid256, 256**-0.5, chunk_size=16, use_sdpa=True
    )
    assert sdpa_calls == 2
    assert tiled_calls == 0
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad((out256 + out256_cross).square().sum(), (q256, k256, v256))
    )

    # 4. BifocalGemmaForCausalLM.enable_memory_efficient_training routes head_dim > 256 safely
    cfg = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    model = BifocalGemmaForCausalLM(cfg)
    assert model.encoder is not None
    # Simulate a full-attention layer with head_dim=512 alongside sliding layers with head_dim <= 256
    full_layer_enc = cast(Any, model.encoder).layers[4].self_attn
    full_layer_dec = cast(Any, model.decoder).layers[4].self_attn
    slide_layer_enc = cast(Any, model.encoder).layers[0].self_attn
    object.__setattr__(full_layer_enc, "head_dim", 512)
    object.__setattr__(full_layer_dec, "head_dim", 512)
    model.enable_memory_efficient_training(gradient_checkpointing=False, attention_backend="sdpa")
    assert cast(Any, model.encoder).layers[4].self_attn.attention_backend == "tiled_eager_fp32"
    assert cast(Any, model.decoder).layers[4].self_attn.attention_backend == "tiled_eager_fp32"
    assert slide_layer_enc.attention_backend == "sdpa"


def test_full_cross_cached_attention_sdpa_guard_fallback_and_non_left_padded_step_valid(
    monkeypatch: pytest.MonkeyPatch,
):
    """FullCrossCachedAttention must guard SDPA with _sdpa_supported/fallback, avoid hardcoded
    enable_gqa=True, and honor non-left-padded step_valid masks identically to FullCrossTiledAttention.
    """
    torch.manual_seed(67)
    cfg = BifocalGemmaConfig(
        encoder_size="E2B",
        decoder_size="E2B",
        tiny=True,
        gate_init_tanh=0.25,
        decouple_full_attn_cross_kv=True,
    )
    model = BifocalGemmaForCausalLM(cfg).eval()
    dec_layers = cast(Any, model.decoder).layers

    enc_ids = torch.randint(5, 80, (2, 8))
    enc_mask = torch.ones_like(enc_ids, dtype=torch.bool)
    enc_mask[0, 5:] = False

    # Non-left-padded mask (middle/trailing invalid positions!)
    dec_ids = torch.randint(5, 80, (2, 5))
    non_left_mask = torch.tensor(
        [[True, False, True, False, True], [False, True, True, False, True]],
        dtype=torch.bool,
    )

    def _run_cached(force_sdpa_error: bool = False) -> tuple[torch.Tensor, list[bool]]:
        seen_enable_gqa: list[bool] = []
        orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def _spy_sdpa(*args: Any, **kwargs: Any) -> torch.Tensor:
            if "enable_gqa" in kwargs:
                seen_enable_gqa.append(bool(kwargs["enable_gqa"]))
            if force_sdpa_error:
                raise TypeError("Simulated unsupported enable_gqa")
            return orig_sdpa(*args, **kwargs)

        with monkeypatch.context() as mp:
            mp.setattr(torch.nn.functional, "scaled_dot_product_attention", _spy_sdpa)
            cache = InferenceCache()
            orig_classes = [layer.self_attn.__class__ for layer in dec_layers]
            try:
                for layer in dec_layers:
                    object.__setattr__(layer.self_attn, "__class__", FullCrossCachedAttention)
                setattr(cache, "step_valid_mask", non_left_mask)
                pos_ids = model._positions(non_left_mask)
                with torch.no_grad():
                    enc_states = model.encode_context(enc_ids, enc_mask)
                    dec_out = model.decoder(
                        input_ids=dec_ids,
                        position_ids=pos_ids,
                        attention_mask={"full_attention": None, "sliding_attention": None},
                        use_cache=False,
                        inference_cache=cache,
                        encoder_hidden_states=enc_states,
                        encoder_valid=enc_mask,
                    )
                    logits = model.lm_head(dec_out.last_hidden_state)
                    cap = model.decoder.config.final_logit_softcapping
                    if cap is not None:
                        logits = torch.tanh(logits / cap) * cap
                return logits, seen_enable_gqa
            finally:
                for layer, cls in zip(dec_layers, orig_classes, strict=True):
                    object.__setattr__(layer.self_attn, "__class__", cls)

    with torch.no_grad():
        enc_states_ref = model.encode_context(enc_ids, enc_mask)
        tiled_logits = model.decode_logits(dec_ids, non_left_mask, enc_states_ref, enc_mask)

    # 1. SDPA path matches FullCrossTiledAttention on ALL positions (including non-left-padded invalid slots!)
    cached_sdpa_logits, gqa_flags = _run_cached(force_sdpa_error=False)
    assert len(gqa_flags) > 0
    torch.testing.assert_close(cached_sdpa_logits, tiled_logits, atol=1e-4, rtol=1e-4)

    # 2. Eager fallback when SDPA raises TypeError matches FullCrossTiledAttention identically
    cached_fallback_logits, _ = _run_cached(force_sdpa_error=True)
    torch.testing.assert_close(cached_fallback_logits, tiled_logits, atol=1e-4, rtol=1e-4)

    # 3. Direct float64 check: _sdpa_supported returns False -> eager fallback succeeds without error
    layer0 = dec_layers[0].self_attn
    orig_cls = layer0.__class__
    try:
        object.__setattr__(layer0, "__class__", FullCrossCachedAttention)
        layer0_64 = layer0.double()
        cache64 = InferenceCache()
        setattr(cache64, "step_valid_mask", non_left_mask)
        hidden64 = torch.randn(2, 5, model.decoder.config.hidden_size, dtype=torch.float64)
        cos64 = torch.ones(2, 5, layer0_64.head_dim, dtype=torch.float64)
        sin64 = torch.zeros(2, 5, layer0_64.head_dim, dtype=torch.float64)
        with torch.no_grad():
            out64, _ = layer0_64(
                hidden64,
                (cos64, sin64),
                None,
                {},
                inference_cache=cache64,
                encoder_hidden_states=None,
                encoder_valid=None,
            )
        assert out64.dtype == torch.float64
        assert torch.isfinite(out64).all()
        # Invalid query positions under non_left_mask must produce zero self-attn output (o_proj has no bias)
        torch.testing.assert_close(
            out64[~non_left_mask], torch.zeros_like(out64[~non_left_mask]), atol=1e-12, rtol=0
        )
    finally:
        layer0.float()
        object.__setattr__(layer0, "__class__", orig_cls)
