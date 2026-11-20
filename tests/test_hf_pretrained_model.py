"""Tests for BifocalGemma HF PreTrainedModel FP32 bridge & cross-attention invariants."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
from bifocalgemma.contracts import Seq2SeqBatch
from bifocalgemma.modeling_bifocalgemma import BifocalGemmaForCausalLM
from bifocalgemma.models.bridge import SwiGLUDualGateBridge
from bifocalgemma.models.cross_attention import Bf16FullLinear, FullCrossTiledAttention
from bifocalgemma.training.memory_probe import ProbeLoRALinear
from bifocalgemma.training.objectives import is_cross_bridge_parameter


def test_model_to_bfloat16_preserves_fp32_master_weights():
    """Calling model.to(dtype=torch.bfloat16) must keep bridge, cross-attn, and LoRA in FP32."""
    gate_tanh = 0.05
    config = BifocalGemmaConfig(
        encoder_size="E2B",
        decoder_size="E2B",
        tiny=True,
        init_rms_scale=1.1,
        gate_init_tanh=gate_tanh,
    )
    model = BifocalGemmaForCausalLM(config)

    assert model.decoder is not None
    assert model.encoder is not None
    decoder = cast(Any, model.decoder)
    encoder = cast(Any, model.encoder)

    # Attach a ProbeLoRALinear to verify LoRA parameters stay float32 as well
    first_layer = decoder.layers[0]
    base_linear = first_layer.mlp.gate_proj
    lora_module = ProbeLoRALinear(base_linear, rank=4)
    first_layer.mlp.gate_proj = lora_module

    # Cast whole model to bfloat16
    cast(Any, model).to(dtype=torch.bfloat16)

    # 1. Decoder and encoder backbone weights should be bfloat16
    assert decoder.embed_tokens.weight.dtype == torch.bfloat16
    assert decoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert encoder.embed_tokens.weight.dtype == torch.bfloat16

    # 2. Bridge parameters must be preserved in float32
    bridge = model.encoder_projection
    assert isinstance(bridge, SwiGLUDualGateBridge)
    assert bridge.pre_norm.weight.dtype == torch.float32
    assert bridge.w_gate.weight.dtype == torch.float32
    assert bridge.w_up.weight.dtype == torch.float32
    assert bridge.w_down.weight.dtype == torch.float32
    assert bridge.post_norm.weight is None  # v4: gamma_calib is the single gain
    assert bridge.gamma_calib.dtype == torch.float32

    # 3. Cross attention parameters must be preserved in float32
    for layer in decoder.layers:
        attn = layer.self_attn
        assert isinstance(attn, FullCrossTiledAttention)
        cross_gate = cast(torch.Tensor, attn.cross_gate)
        cross_gate_proj = cast(torch.Tensor, attn.cross_gate_proj)
        assert cross_gate.dtype == torch.float32
        assert cross_gate_proj.dtype == torch.float32
        assert attn.cross_scale.dtype == torch.float32
        cross_q_norm = cast(Any, attn.cross_q_norm)
        assert cross_q_norm.weight.dtype == torch.float32

        expected_gate_init = math.atanh(gate_tanh)
        assert torch.allclose(
            cross_gate,
            torch.full_like(cross_gate, expected_gate_init),
            atol=1e-4,
        )
        assert torch.all(cross_gate_proj == 0.0)

        if not attn.is_kv_shared_layer:
            cross_k_proj = cast(Bf16FullLinear, attn.cross_k_proj)
            cross_v_proj = cast(Bf16FullLinear, attn.cross_v_proj)
            cross_k_norm = cast(Any, attn.cross_k_norm)
            cross_v_norm = cast(Any, attn.cross_v_norm)
            assert isinstance(cross_k_proj, Bf16FullLinear)
            assert isinstance(cross_v_proj, Bf16FullLinear)
            assert cross_k_proj.weight.dtype == torch.float32
            assert cross_v_proj.weight.dtype == torch.float32
            assert cross_k_norm.weight.dtype == torch.float32
            if hasattr(cross_v_norm, "weight") and cross_v_norm.weight is not None:
                assert cross_v_norm.weight.dtype == torch.float32

    # 4. LoRA parameters must be preserved in float32
    assert lora_module.lora_A.dtype == torch.float32
    assert lora_module.lora_B.dtype == torch.float32

    # 5. All parameters matching is_cross_bridge_parameter must be float32
    for name, param in model.named_parameters():
        if is_cross_bridge_parameter(name) or name.endswith((".lora_A", ".lora_B")):
            assert param.dtype == torch.float32, (
                f"Parameter {name} expected float32, got {param.dtype}"
            )


def test_bfloat16_forward_backward_gradient_flow_and_optimizer_step():
    """BF16 forward + backward must produce valid gradients and updates on FP32 master parameters."""
    config = BifocalGemmaConfig(
        encoder_size="E2B",
        decoder_size="E2B",
        tiny=True,
        init_rms_scale=1.0,
        gate_init_tanh=0.05,
    )
    model = BifocalGemmaForCausalLM(config)
    cast(Any, model).to(dtype=torch.bfloat16)

    assert model.decoder is not None
    assert model.encoder is not None
    assert model.encoder_projection is not None
    bridge = cast(SwiGLUDualGateBridge, model.encoder_projection)
    decoder = cast(Any, model.decoder)
    first_attn = cast(FullCrossTiledAttention, decoder.layers[0].self_attn)
    cross_gate = cast(torch.Tensor, first_attn.cross_gate)
    cross_k_proj = cast(Bf16FullLinear, first_attn.cross_k_proj)

    cross_gate_before = cross_gate.detach().clone()
    cross_k_weight_before = cross_k_proj.weight.detach().clone()
    gamma_calib_before = bridge.gamma_calib.detach().clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 2
    enc_len = 6
    dec_len = 4
    encoder_input_ids = torch.randint(5, 50, (batch_size, enc_len))
    decoder_input_ids = torch.randint(5, 50, (batch_size, dec_len))
    labels = decoder_input_ids.clone()

    output = model(
        input_ids=decoder_input_ids,
        encoder_input_ids=encoder_input_ids,
        labels=labels,
    )
    assert not isinstance(output, tuple)
    loss = output.loss
    assert loss is not None
    assert loss.item() > 0.0

    loss.backward()

    # Verify gradients are non-zero and float32
    assert cross_gate.grad is not None
    assert cross_gate.grad.dtype == torch.float32
    assert cross_gate.grad.abs().sum().item() > 0.0

    assert cross_k_proj.weight.grad is not None
    assert cross_k_proj.weight.grad.dtype == torch.float32
    assert cross_k_proj.weight.grad.abs().sum().item() > 0.0

    assert bridge.gamma_calib.grad is not None
    assert bridge.gamma_calib.grad.dtype == torch.float32
    assert bridge.gamma_calib.grad.abs().sum().item() > 0.0

    cross_gate_proj = cast(torch.Tensor, first_attn.cross_gate_proj)
    assert cross_gate_proj.grad is not None
    assert cross_gate_proj.grad.dtype == torch.float32
    assert cross_gate_proj.grad.abs().sum().item() > 0.0

    cross_gate_proj_before = cross_gate_proj.detach().clone()

    # Optimizer step produces parameter updates
    optimizer.step()

    assert not torch.equal(cross_gate, cross_gate_before)
    assert not torch.equal(cross_gate_proj, cross_gate_proj_before)
    assert not torch.equal(cross_k_proj.weight, cross_k_weight_before)
    assert not torch.equal(bridge.gamma_calib, gamma_calib_before)

    # Preserved in float32 after step
    assert cross_gate.dtype == torch.float32
    assert cross_k_proj.weight.dtype == torch.float32
    assert bridge.gamma_calib.dtype == torch.float32


def test_bridge_init_preserved_across_post_init():
    """SwiGLUDualGateBridge eye_/orthogonal_/w_down init and zeros cross_gate_proj survive post_init()."""
    torch.manual_seed(7)
    sym = BifocalGemmaForCausalLM(
        BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    )
    sym_bridge = cast(SwiGLUDualGateBridge, sym.encoder_projection)
    assert isinstance(sym_bridge.w_res, nn.Linear)
    torch.testing.assert_close(
        sym_bridge.w_res.weight.detach(),
        torch.eye(sym_bridge.d_enc, dtype=sym_bridge.w_res.weight.dtype),
        atol=1e-6,
        rtol=0,
    )
    expected_std = 0.05 / math.sqrt(sym_bridge.d_mid)
    assert abs(sym_bridge.w_down.weight.std().item() - expected_std) < 0.005

    asym = BifocalGemmaForCausalLM(
        BifocalGemmaConfig(encoder_size="E4B", decoder_size="E2B", tiny=True)
    )
    asym_bridge = cast(SwiGLUDualGateBridge, asym.encoder_projection)
    assert isinstance(asym_bridge.w_res, nn.Linear)
    w = asym_bridge.w_res.weight.detach()
    gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    torch.testing.assert_close(gram, torch.eye(gram.shape[0], dtype=gram.dtype), atol=1e-5, rtol=0)
    asym_expected_std = 0.05 / math.sqrt(asym_bridge.d_mid)
    assert abs(asym_bridge.w_down.weight.std().item() - asym_expected_std) < 0.005
    first_attn = cast(FullCrossTiledAttention, cast(Any, asym.decoder).layers[0].self_attn)
    assert torch.all(cast(torch.Tensor, first_attn.cross_gate_proj) == 0.0)


def test_from_pretrained_raises_file_not_found_when_weights_missing(tmp_path: Path):
    cfg = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    empty_dir = tmp_path / "missing_weights"
    empty_dir.mkdir()
    cfg.to_json_file(empty_dir / "config.json")
    with pytest.raises(FileNotFoundError):
        BifocalGemmaForCausalLM.from_pretrained(empty_dir)

    sharded_missing = tmp_path / "missing_shard"
    sharded_missing.mkdir()
    cfg.to_json_file(sharded_missing / "config.json")
    (sharded_missing / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        BifocalGemmaForCausalLM.from_pretrained(sharded_missing)


def test_forward_prompt_masked_labels_shift_and_noncontiguous_forward_loss():
    torch.manual_seed(13)
    cfg = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    model = BifocalGemmaForCausalLM(cfg).eval()
    enc_ids = torch.randint(5, 50, (2, 4))
    dec_ids = torch.randint(5, 50, (2, 5))
    labels = dec_ids.clone()
    labels[:, :2] = -100

    out = model(input_ids=dec_ids, encoder_input_ids=enc_ids, labels=labels)
    expected = F.cross_entropy(
        out.logits[:, :-1].float().reshape(-1, cfg.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(out.loss, expected, atol=1e-5, rtol=1e-5)

    noncontig_labels = labels[:, 1:]
    assert not noncontig_labels.is_contiguous()
    batch = Seq2SeqBatch(
        encoder_input_ids=enc_ids,
        encoder_attention_mask=torch.ones_like(enc_ids, dtype=torch.bool),
        decoder_input_ids=dec_ids[:, :-1],
        decoder_attention_mask=torch.ones_like(dec_ids[:, :-1], dtype=torch.bool),
        labels=noncontig_labels,
    )
    loss_out = model.forward_loss(batch)
    torch.testing.assert_close(loss_out.loss, expected, atol=1e-5, rtol=1e-5)


def test_generate_multi_eos_per_row_stopping_and_cache_parity(monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(29)
    cfg = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    model = BifocalGemmaForCausalLM(cfg).eval()
    enc_ids = torch.randint(5, 50, (2, 4))
    input_ids = torch.tensor([[0, 0, 2, 10], [2, 11, 12, 13]])
    attn_mask = torch.tensor([[False, False, True, True], [True, True, True, True]])

    cached = model.generate(
        input_ids=input_ids,
        encoder_input_ids=enc_ids,
        attention_mask=attn_mask,
        max_new_tokens=5,
        use_cache=True,
    )
    uncached = model.generate(
        input_ids=input_ids,
        encoder_input_ids=enc_ids,
        attention_mask=attn_mask,
        max_new_tokens=5,
        use_cache=False,
    )
    assert torch.equal(cached, uncached)

    step_counter = [0]
    orig_forward = model.lm_head.forward

    def scripted_head(hidden: torch.Tensor) -> torch.Tensor:
        logits = orig_forward(hidden).zero_()
        step = step_counter[0]
        step_counter[0] += 1
        if step == 0:
            logits[0, -1, 106] = 10.0
            logits[1, -1, 25] = 10.0
        elif step == 1:
            logits[0, -1, 99] = 10.0
            logits[1, -1, 50] = 10.0
        else:
            logits[:, -1, 77] = 10.0
        return logits

    monkeypatch.setattr(model.lm_head, "forward", scripted_head)
    gen_cached = model.generate(
        input_ids=input_ids,
        encoder_input_ids=enc_ids,
        attention_mask=attn_mask,
        max_new_tokens=5,
        pad_token_id=0,
        eos_token_id=[1, 106, 50],
        use_cache=True,
    )
    assert gen_cached[:, 4:].tolist() == [[106, 0], [25, 50]]

    step_counter[0] = 0
    gen_uncached = model.generate(
        input_ids=input_ids,
        encoder_input_ids=enc_ids,
        attention_mask=attn_mask,
        max_new_tokens=5,
        pad_token_id=0,
        eos_token_id=[1, 106, 50],
        use_cache=False,
    )
    assert torch.equal(gen_cached, gen_uncached)
