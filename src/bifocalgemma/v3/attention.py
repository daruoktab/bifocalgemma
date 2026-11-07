"""Decoupled full-rank cross-attention for the V3 decoder."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.utils.checkpoint
from torch.nn import functional as F
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from bifocalgemma.models import inference_cache as inference_cache_mod
from bifocalgemma.models.memory_efficient import (
    TiledMergedAttention,
    attention_tile,
    tiled_attention,
)
from bifocalgemma.v3.bridge import SwiGLUDualGateBridge
from bifocalgemma.v3.constants import (
    GATE_INIT_ATANH,
    GATE_INIT_TANH,
)
from bifocalgemma.v3.objectives import (
    is_cross_bridge_parameter,
)
from bifocalgemma.v3.runtime import (
    clamped_triton_kernel_options,
    critical_attention_beta,
    ensure_hardware_memory_guards,
)


def _compute_per_row_cross_query(
    query_cross: torch.Tensor,
    encoder_valid: torch.Tensor,
    fallback_n_enc: int,
) -> torch.Tensor:
    """Divide cross-queries by per-row critical attention length temperature beta_N (V3 §5.4 & §8.2)."""
    bsz = query_cross.shape[0]
    valid_counts = encoder_valid.bool().sum(dim=-1).tolist()
    betas = [
        critical_attention_beta(int(cnt) if int(cnt) > 0 else int(fallback_n_enc))
        for cnt in valid_counts
    ]
    if all(b == 1.0 for b in betas):
        return query_cross
    inv_beta = torch.tensor(
        [1.0 / b for b in betas],
        dtype=torch.float32,
        device=query_cross.device,
    ).view(bsz, 1, 1, 1)
    return (query_cross.float() * inv_beta).to(query_cross.dtype)


_ORIGINAL_CACHED_ATTENTION = getattr(
    inference_cache_mod, "_orig_cached_attention", inference_cache_mod.CachedAttention
)
setattr(inference_cache_mod, "_orig_cached_attention", _ORIGINAL_CACHED_ATTENTION)


class FullCrossCachedAttention(_ORIGINAL_CACHED_ATTENTION):
    """Cached inference attention matching FullCrossTiledAttention with static cross-KV cache and batched left-pad support."""

    def forward(
        self: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        shared_kv_states: dict[str, Any],
        past_key_values: Any = None,
        *,
        inference_cache: Any = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_valid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        step_valid = (
            getattr(inference_cache, "step_valid_mask", None)
            if inference_cache is not None
            else None
        )
        has_full_cross = bool(getattr(self, "full_rank_cross_attention", False))
        if not has_full_cross and step_valid is None and hidden_states.shape[0] == 1:
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                shared_kv_states,
                past_key_values=past_key_values,
                inference_cache=inference_cache,
                encoder_hidden_states=encoder_hidden_states,
                encoder_valid=encoder_valid,
                **kwargs,
            )
        if self.training or torch.is_grad_enabled():
            raise ValueError("Inference cache requires eval and inference_mode/no_grad.")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        raw_q = self.q_proj(hidden_states).view(shape)
        query_self = apply_rotary_pos_emb(self.q_norm(raw_q), cos, sin, unsqueeze_dim=2).transpose(
            1, 2
        )
        query_cross = (
            self.cross_q_norm(raw_q).to(raw_q.dtype).transpose(1, 2)
            if has_full_cross and hasattr(self, "cross_q_norm")
            else None
        )
        cache = inference_cache
        if cache is None:
            raise ValueError("Explicit inference cache required.")
        positions = torch.arange(cache.offset, cache.offset + length, device=query_self.device)
        if self.is_kv_shared_layer:
            key_self, value_self, cross, key_valid = shared_kv_states[self.layer_type]
            key_positions, self_length = cache.shared_positions[self.layer_type]
        else:
            raw_key = self.k_proj(hidden_states).view(shape)
            raw_value = (
                self.v_proj(hidden_states).view(shape) if self.v_proj is not None else raw_key
            )
            key_self = apply_rotary_pos_emb(
                self.k_norm(raw_key), cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            value_self = self.v_norm(raw_value).transpose(1, 2)
            previous = cache.layers.get(self.layer_idx)
            key_positions = positions
            key_valid = step_valid
            if previous is not None:
                key_self = torch.cat((previous[0], key_self), dim=2)
                value_self = torch.cat((previous[1], value_self), dim=2)
                key_positions = torch.cat((previous[2], positions))
                prev_valid = previous[4] if len(previous) > 4 else None
                if prev_valid is not None and step_valid is not None:
                    key_valid = torch.cat((prev_valid, step_valid), dim=1)
            keep = self.sliding_window if self.sliding_window else key_self.shape[2]
            stored_key, stored_value, stored_positions = (
                key_self[:, :, -keep:],
                value_self[:, :, -keep:],
                key_positions[-keep:],
            )
            stored_valid = key_valid[:, -keep:] if key_valid is not None else None
            cross = previous[3] if previous is not None else None
            if (
                has_full_cross
                and encoder_hidden_states is not None
                and encoder_hidden_states.shape[1] > 0
                and cross is None
            ):
                cross_shape = (
                    batch,
                    encoder_hidden_states.shape[1],
                    self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                    self.head_dim,
                )
                cross = (
                    self.cross_k_norm(self.cross_k_proj(encoder_hidden_states).view(cross_shape))
                    .to(query_self.dtype)
                    .transpose(1, 2),
                    self.cross_v_norm(self.cross_v_proj(encoder_hidden_states).view(cross_shape))
                    .to(query_self.dtype)
                    .transpose(1, 2),
                )
            cache.layers[self.layer_idx] = (
                stored_key,
                stored_value,
                stored_positions,
                cross,
                stored_valid,
            )
            self_length = key_self.shape[2]
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (key_self, value_self, cross, key_valid)
            cache.shared_positions[self.layer_type] = (key_positions, self_length)

        self_pieces = []
        for start in range(0, length, 128):
            stop = min(length, start + 128)
            visible = key_positions[None, :] <= positions[start:stop, None]
            if self.sliding_window:
                visible &= (
                    key_positions[None, :] > positions[start:stop, None] - self.sliding_window
                )
            allowed = visible[None, None].expand(batch, 1, -1, -1)
            if key_valid is not None:
                allowed = allowed & key_valid[:, None, None, :].bool()
                has_any_key = allowed.any(dim=-1, keepdim=True)
                safe_allowed = allowed | ~has_any_key
                tile_out = F.scaled_dot_product_attention(
                    query_self[:, :, start:stop],
                    key_self,
                    value_self,
                    attn_mask=safe_allowed,
                    dropout_p=0.0,
                    scale=self.scaling,
                    enable_gqa=True,
                )
                tile_out = tile_out.masked_fill(~has_any_key, 0.0)
            else:
                tile_out = F.scaled_dot_product_attention(
                    query_self[:, :, start:stop],
                    key_self,
                    value_self,
                    attn_mask=allowed,
                    dropout_p=0.0,
                    scale=self.scaling,
                    enable_gqa=True,
                )
            self_pieces.append(tile_out)
        self_out = torch.cat(self_pieces, dim=2)

        if (
            has_full_cross
            and query_cross is not None
            and cross is not None
            and encoder_valid is not None
            and bool(encoder_valid.any())
        ):
            n_enc = int(cross[0].shape[2])
            scaled_query_cross = _compute_per_row_cross_query(query_cross, encoder_valid, n_enc)
            if hasattr(self, "cross_scale"):
                scaled_query_cross = scaled_query_cross * torch.clamp(
                    self.cross_scale.to(scaled_query_cross.dtype), min=0.1, max=5.0
                )
            cross_scale = float(self.scaling)
            cross_pieces = []
            for start in range(0, length, 128):
                stop = min(length, start + 128)
                cross_mask = (
                    encoder_valid[:, None, None, :].bool().expand(batch, 1, stop - start, -1)
                )
                if step_valid is not None:
                    cross_mask = cross_mask & step_valid[:, None, start:stop, None].bool()
                cross_pieces.append(
                    attention_tile(
                        scaled_query_cross[:, :, start:stop],
                        cross[0],
                        cross[1],
                        cross_mask,
                        cross_scale,
                        0.0,
                        True,
                    )
                )
            cross_out = torch.cat(cross_pieces, dim=2)
            gate_logits = torch.einsum(
                "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
            ) * (4.0 * (self.head_dim**-0.5))
            query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
            head_gate = torch.tanh(self.cross_gate.float()).view(1, -1, 1, 1)
            effective_gate = (head_gate * query_gate).to(self_out.dtype)
            result = self_out + effective_gate * cross_out
        else:
            result = self_out

        result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(result), None


class Bf16FullLinear(torch.nn.Module):
    """Full-rank linear projection with 16-bit bfloat16 compute and FP32 master weight."""

    def __init__(self, base_layer: torch.nn.Module) -> None:
        super().__init__()
        base: Any = getattr(base_layer, "base", base_layer)
        w = base.weight.detach().float().clone()
        self.in_features = int(w.shape[1])
        self.out_features = int(w.shape[0])
        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        compute_dtype = (
            torch.bfloat16 if (x.device.type == "cuda" or x.dtype == torch.bfloat16) else x.dtype
        )
        return F.linear(x.to(compute_dtype), self.weight.to(compute_dtype)).to(x.dtype)


class FullCrossTiledAttention(TiledMergedAttention):
    """Decoupled causal self-attention and stationary unrotated full-rank 16-bit gated cross-attention."""

    def forward(  # ty: ignore[invalid-method-override]
        self: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any,
        shared_kv_states: dict[str, Any],
        past_key_values: Any = None,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_valid: torch.Tensor | None = None,
        decoder_valid: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if past_key_values is not None:
            raise ValueError("Tiled training attention does not support temporal cache.")
        if encoder_hidden_states is None or encoder_valid is None or decoder_valid is None:
            raise ValueError("Merged attention requires encoder states and validity masks.")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        raw_q = self.q_proj(hidden_states).view(shape)
        query_self = apply_rotary_pos_emb(self.q_norm(raw_q), cos, sin, unsqueeze_dim=2).transpose(
            1, 2
        )
        # Stationary unrotated cross-query coordinates (zero RoPE frequency spinning)
        query_cross = self.cross_q_norm(raw_q).to(raw_q.dtype).transpose(1, 2)

        has_encoder_tokens = encoder_hidden_states.shape[1] > 0 and bool(encoder_valid.any())
        if self.is_kv_shared_layer:
            key_self, value_self, cross_key, cross_value = shared_kv_states[self.layer_type]
        else:
            if self.v_proj is None:
                raise ValueError("Merged branch requires separate value projection.")
            key_self = self.k_norm(self.k_proj(hidden_states).view(shape))
            key_self = apply_rotary_pos_emb(key_self, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value_self = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
            if has_encoder_tokens:
                cross_shape = (
                    batch,
                    encoder_hidden_states.shape[1],
                    self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                    self.head_dim,
                )
                cross_key = (
                    self.cross_k_norm(self.cross_k_proj(encoder_hidden_states).view(cross_shape))
                    .to(query_self.dtype)
                    .transpose(1, 2)
                )
                cross_value = (
                    self.cross_v_norm(self.cross_v_proj(encoder_hidden_states).view(cross_shape))
                    .to(query_self.dtype)
                    .transpose(1, 2)
                )
            else:
                cross_key, cross_value = None, None
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (
                key_self,
                value_self,
                cross_key,
                cross_value,
            )

        chunk_sz = getattr(self, "query_chunk_size", 128)
        backend_name = getattr(self, "attention_backend", "tiled_eager_fp32")
        self_mask = {"valid": decoder_valid, "bidirectional": False}
        self_out = tiled_attention(
            query_self,
            key_self,
            value_self,
            chunk_size=chunk_sz,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            attention_mask=self_mask,
            window=self.sliding_window,
            backend=backend_name,
        )

        if has_encoder_tokens and cross_key is not None and cross_value is not None:
            n_enc = int(encoder_hidden_states.shape[1])
            scaled_query_cross = _compute_per_row_cross_query(query_cross, encoder_valid, n_enc)
            if hasattr(self, "cross_scale"):
                scaled_query_cross = scaled_query_cross * torch.clamp(
                    self.cross_scale.to(scaled_query_cross.dtype), min=0.1, max=5.0
                )
            cross_scale = float(self.scaling)
            pieces = []
            for start in range(0, length, chunk_sz):
                stop = min(length, start + chunk_sz)
                cross_mask = (
                    encoder_valid[:, None, None, :].bool()
                    & decoder_valid[:, None, start:stop, None].bool()
                )
                pieces.append(
                    attention_tile(
                        scaled_query_cross[:, :, start:stop],
                        cross_key,
                        cross_value,
                        cross_mask,
                        cross_scale,
                        0.0,
                        True,
                    )
                )
            cross_out = torch.cat(pieces, dim=2)
            gate_logits = torch.einsum(
                "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
            ) * (4.0 * (self.head_dim**-0.5))
            query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
            head_gate = torch.tanh(self.cross_gate.float()).view(1, -1, 1, 1)
            effective_gate = (head_gate * query_gate).to(self_out.dtype)
            result = self_out + effective_gate * cross_out
        else:
            result = self_out

        result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(result), None


def configure_full_rank_cross_attention(model: Any) -> dict[str, Any]:
    """Attach SwiGLUDualGateBridge, full-rank BF16 cross-attention projections, unrotated cross-query norm, and atanh(0.05) gates."""
    ensure_hardware_memory_guards()
    setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)

    if getattr(model, "encoder", None) is None:
        return {
            "mode": "decoder_only_causal_v3",
            "bridge_type": None,
            "kv_owner_layers": 0,
            "total_decoder_layers": len(model.decoder.layers),
            "total_cross_bridge_parameters": 0,
        }

    # Install SwiGLUDualGateBridge as model.encoder_projection
    enc_dim = int(model.encoder.config.hidden_size)
    dec_dim = int(model.decoder.config.hidden_size)
    dev = next(model.decoder.parameters()).device
    model.encoder_projection = SwiGLUDualGateBridge(enc_dim, dec_dim).to(dev)

    kv_owner_layers = 0
    total_cross_params = 0
    for layer in model.decoder.layers:
        attn = layer.self_attn
        q_base = getattr(attn.q_proj, "base", attn.q_proj)
        dev = q_base.weight.device
        num_heads = int(q_base.weight.shape[0] // attn.head_dim)
        attn.cross_q_norm = copy.deepcopy(attn.q_norm).float().to(dev)
        attn.cross_gate = torch.nn.Parameter(
            torch.full((num_heads,), float(GATE_INIT_ATANH), dtype=torch.float32, device=dev)
        )
        attn.cross_gate_proj = torch.nn.Parameter(
            torch.zeros((num_heads, attn.head_dim), dtype=torch.float32, device=dev)
        )
        attn.cross_scale = torch.nn.Parameter(
            torch.ones((num_heads, 1, 1), dtype=torch.float32, device=dev)
        )
        if not attn.is_kv_shared_layer:
            attn.cross_k_proj = Bf16FullLinear(attn.k_proj).to(dev)
            attn.cross_v_proj = Bf16FullLinear(attn.v_proj).to(dev)
            attn.cross_k_norm = copy.deepcopy(attn.k_norm).float().to(dev)
            attn.cross_v_norm = copy.deepcopy(attn.v_norm).float().to(dev)
            kv_owner_layers += 1
        attn.query_chunk_size = getattr(attn, "query_chunk_size", 128)
        attn.attention_backend = getattr(attn, "attention_backend", "tiled_eager_fp32")
        attn.triton_kernel_options = clamped_triton_kernel_options(attn.head_dim)
        attn.full_rank_cross_attention = True
        attn.__class__ = FullCrossTiledAttention

    for n, p in model.named_parameters():
        if is_cross_bridge_parameter(n):
            total_cross_params += int(p.numel())
    return {
        "mode": "full_rank_cross_bf16",
        "bridge_type": "SwiGLUDualGateBridge",
        "gate_init_tanh": GATE_INIT_TANH,
        "kv_owner_layers": kv_owner_layers,
        "total_decoder_layers": len(model.decoder.layers),
        "total_cross_bridge_parameters": total_cross_params,
    }
