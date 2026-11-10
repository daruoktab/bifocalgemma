"""Tests for BifocalGemma HF PreTrainedModel V3 FP32 bridge & cross-attention invariants."""

from __future__ import annotations

import importlib.metadata
import math
from typing import Any, cast

import torch

_real_version = importlib.metadata.version


def _compat_version(distribution_name: str) -> str:
    if distribution_name == "transformers":
        return "5.22.1"
    return _real_version(distribution_name)


setattr(importlib.metadata, "version", _compat_version)

import bifocalgemma.models.v2.native  # noqa: E402

setattr(bifocalgemma.models.v2.native, "version", _compat_version)

from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig  # noqa: E402
from bifocalgemma.modeling_bifocalgemma import BifocalGemmaForCausalLM  # noqa: E402
from bifocalgemma.training.memory_probe import ProbeLoRALinear  # noqa: E402
from bifocalgemma.v3.attention import Bf16FullLinear, FullCrossTiledAttention  # noqa: E402
from bifocalgemma.v3.bridge import SwiGLUDualGateBridge  # noqa: E402
from bifocalgemma.v3.objectives import is_cross_bridge_parameter  # noqa: E402


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
    assert bridge.post_norm.weight.dtype == torch.float32
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
        assert not bool(torch.all(cross_gate_proj == 0.0))

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

    # Optimizer step produces parameter updates
    optimizer.step()

    assert not torch.equal(cross_gate, cross_gate_before)
    assert not torch.equal(cross_k_proj.weight, cross_k_weight_before)
    assert not torch.equal(bridge.gamma_calib, gamma_calib_before)

    # Preserved in float32 after step
    assert cross_gate.dtype == torch.float32
    assert cross_k_proj.weight.dtype == torch.float32
    assert bridge.gamma_calib.dtype == torch.float32
