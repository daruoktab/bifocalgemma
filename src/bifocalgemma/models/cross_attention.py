# mypy: disable-error-code="override"
"""Decoupled full-rank cross-attention for the BifocalGemma decoder."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.utils.checkpoint
from torch.nn import functional as F
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

import bifocalgemma.config as _config_mod
from bifocalgemma.models import inference_cache as inference_cache_mod
from bifocalgemma.models.bridge import SwiGLUDualGateBridge
from bifocalgemma.models.memory_efficient import (
    TiledMergedAttention,
    sdpa_cross_attention,
    tiled_attention,
)
from bifocalgemma.training.objectives import is_cross_bridge_parameter
from bifocalgemma.training.runtime import (
    clamped_triton_kernel_options,
    ensure_hardware_memory_guards,
)

_CRITICAL_ATTENTION_N0 = 2048
_V4_DEFAULT_GATE_INIT_TANH = 0.15


def _compute_per_row_cross_query(
    query_cross: torch.Tensor,
    encoder_valid: torch.Tensor,
    fallback_n_enc: int,
    n_0: int = _CRITICAL_ATTENTION_N0,
) -> torch.Tensor:
    """Multiply cross-queries by per-row critical attention length temperature beta_N on-device."""
    if n_0 <= 0:
        return query_cross
    # Fast host-only shape check avoids device-to-host sync whenever N_enc <= n_0.
    if max(int(encoder_valid.shape[-1]), int(fallback_n_enc)) <= n_0:
        return query_cross
    bsz = query_cross.shape[0]
    counts = encoder_valid.to(device=query_cross.device).bool().sum(dim=-1)
    effective_counts = torch.where(
        counts > 0,
        counts,
        counts.new_full((), int(fallback_n_enc)),
    ).to(dtype=torch.float32)
    ratio = (effective_counts / float(n_0)).clamp_min(1.0)
    beta_scale = (0.1 * torch.log(ratio) + 1.0).view(bsz, 1, 1, 1)
    return (query_cross.float() * beta_scale).to(query_cross.dtype)


def _owns_cross_kv(attn: Any) -> bool:
    """Return True if this attention layer owns independent cross-KV projections."""
    return (
        bool(getattr(attn, "owns_cross_kv", not getattr(attn, "is_kv_shared_layer", False)))
        and hasattr(attn, "cross_k_proj")
        and hasattr(attn, "cross_v_proj")
    )


def _project_cross_kv(
    attn: Any,
    encoder_hidden_states: torch.Tensor,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project encoder hidden states into stationary unrotated cross-attention K/V tensors."""
    batch = encoder_hidden_states.shape[0]
    enc_len = encoder_hidden_states.shape[1]
    cross_shape = (
        batch,
        enc_len,
        attn.config.per_layer_config[attn.layer_idx].num_key_value_heads,
        attn.head_dim,
    )
    cross_k = (
        attn.cross_k_norm(
            attn.cross_k_proj(encoder_hidden_states)
            .view(cross_shape)
            .to(_norm_weight_dtype(attn.cross_k_norm, target_dtype))
        )
        .to(target_dtype)
        .transpose(1, 2)
    )
    cross_v = (
        attn.cross_v_norm(
            attn.cross_v_proj(encoder_hidden_states)
            .view(cross_shape)
            .to(_norm_weight_dtype(attn.cross_v_norm, target_dtype))
        )
        .to(target_dtype)
        .transpose(1, 2)
    )
    return cross_k, cross_v


if TYPE_CHECKING:
    _ORIGINAL_CACHED_ATTENTION = inference_cache_mod.CachedAttention
else:
    _ORIGINAL_CACHED_ATTENTION = getattr(
        inference_cache_mod, "_orig_cached_attention", inference_cache_mod.CachedAttention
    )
setattr(inference_cache_mod, "_orig_cached_attention", _ORIGINAL_CACHED_ATTENTION)


def _norm_weight_dtype(norm: Any, default: torch.dtype) -> torch.dtype:
    """Return norm.weight.dtype if weight parameter exists, else default."""
    w = getattr(norm, "weight", None)
    return w.dtype if isinstance(w, torch.Tensor) else default


class FullCrossCachedAttention(_ORIGINAL_CACHED_ATTENTION):
    """Cached inference attention matching FullCrossTiledAttention with static cross-KV cache and batched left-pad support."""

    def _apply(self: Any, fn: Any, recurse: bool = True) -> Any:
        super()._apply(fn, recurse=recurse)
        for name, param in self.named_parameters():
            if any(
                tok in name
                for tok in (
                    "cross_gate",
                    "cross_scale",
                    "cross_k_norm",
                    "cross_v_norm",
                    "cross_q_norm",
                )
            ) and param.dtype in (torch.bfloat16, torch.float16):
                param.data = param.data.float()
                if param.grad is not None and param.grad.dtype in (torch.bfloat16, torch.float16):
                    param.grad.data = param.grad.data.float()
        return self

    def forward(  # ty: ignore[incompatible-method-override]  # type: ignore[override]
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
            self.cross_q_norm(raw_q.to(_norm_weight_dtype(self.cross_q_norm, raw_q.dtype)))
            .to(raw_q.dtype)
            .transpose(1, 2)
            if has_full_cross and hasattr(self, "cross_q_norm")
            else None
        )
        cache = inference_cache
        if cache is None:
            raise ValueError("Explicit inference cache required.")
        positions = torch.arange(cache.offset, cache.offset + length, device=query_self.device)
        owns_cross_kv = has_full_cross and _owns_cross_kv(self)
        if self.is_kv_shared_layer:
            key_self, value_self, shared_cross, key_valid = shared_kv_states[self.layer_type]
            key_positions, self_length = cache.shared_positions[self.layer_type]
            if owns_cross_kv:
                previous = cache.layers.get(self.layer_idx)
                cross = previous[3] if previous is not None else None
                if (
                    encoder_hidden_states is not None
                    and encoder_hidden_states.shape[1] > 0
                    and cross is None
                ):
                    cross = _project_cross_kv(self, encoder_hidden_states, query_self.dtype)
                    cache.layers[self.layer_idx] = (None, None, None, cross, None)
            else:
                cross = shared_cross
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
                owns_cross_kv
                and encoder_hidden_states is not None
                and encoder_hidden_states.shape[1] > 0
                and cross is None
            ):
                cross = _project_cross_kv(self, encoder_hidden_states, query_self.dtype)
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
            and encoder_valid.shape[1] > 0
        ):
            n_enc = int(cross[0].shape[2])
            scaled_query_cross = _compute_per_row_cross_query(query_cross, encoder_valid, n_enc)
            if hasattr(self, "cross_scale"):
                scaled_query_cross = scaled_query_cross * torch.clamp(
                    self.cross_scale.to(scaled_query_cross.dtype), min=0.1, max=5.0
                )
            cross_scale = float(self.scaling)
            cross_out = sdpa_cross_attention(
                scaled_query_cross,
                cross[0],
                cross[1],
                encoder_valid,
                step_valid,
                cross_scale,
                chunk_size=getattr(self, "query_chunk_size", 128),
                use_sdpa=not getattr(self, "_force_eager_cross", False),
            )
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

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> Bf16FullLinear:
        super()._apply(fn, recurse=recurse)
        if self.weight.dtype in (torch.bfloat16, torch.float16):
            self.weight.data = self.weight.data.float()
            if self.weight.grad is not None and self.weight.grad.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                self.weight.grad.data = self.weight.grad.data.float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = (
            torch.bfloat16 if (x.device.type == "cuda" or x.dtype == torch.bfloat16) else x.dtype
        )
        return F.linear(x.to(compute_dtype), self.weight.to(compute_dtype)).to(x.dtype)


class FullCrossTiledAttention(TiledMergedAttention):
    """Decoupled causal self-attention and stationary unrotated full-rank 16-bit gated cross-attention."""

    def _apply(self: Any, fn: Any, recurse: bool = True) -> Any:
        super()._apply(fn, recurse=recurse)
        for name, param in self.named_parameters():
            if any(
                tok in name
                for tok in (
                    "cross_gate",
                    "cross_scale",
                    "cross_k_norm",
                    "cross_v_norm",
                    "cross_q_norm",
                )
            ) and param.dtype in (torch.bfloat16, torch.float16):
                param.data = param.data.float()
                if param.grad is not None and param.grad.dtype in (torch.bfloat16, torch.float16):
                    param.grad.data = param.grad.data.float()
        return self

    def forward(  # ty: ignore[incompatible-method-override]  # type: ignore[override]
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
        query_cross = (
            self.cross_q_norm(raw_q.to(_norm_weight_dtype(self.cross_q_norm, raw_q.dtype)))
            .to(raw_q.dtype)
            .transpose(1, 2)
        )

        has_encoder_tokens = encoder_hidden_states.shape[1] > 0
        owns_cross_kv = _owns_cross_kv(self)
        if self.is_kv_shared_layer:
            key_self, value_self, shared_cross_key, shared_cross_value = shared_kv_states[
                self.layer_type
            ]
            if owns_cross_kv:
                if has_encoder_tokens:
                    cross_key, cross_value = _project_cross_kv(
                        self, encoder_hidden_states, query_self.dtype
                    )
                else:
                    cross_key, cross_value = None, None
            else:
                cross_key, cross_value = shared_cross_key, shared_cross_value
        else:
            if self.v_proj is None:
                raise ValueError("Merged branch requires separate value projection.")
            key_self = self.k_norm(self.k_proj(hidden_states).view(shape))
            key_self = apply_rotary_pos_emb(key_self, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value_self = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
            if has_encoder_tokens and owns_cross_kv:
                cross_key, cross_value = _project_cross_kv(
                    self, encoder_hidden_states, query_self.dtype
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
        if (
            backend_name == "tiled_eager_fp32"
            and not self.training
            and not torch.is_grad_enabled()
            and not getattr(self, "_force_eager_eval", False)
        ):
            backend_name = "sdpa"
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
            n_enc = encoder_hidden_states.shape[1]
            scaled_query_cross = _compute_per_row_cross_query(query_cross, encoder_valid, n_enc)
            if hasattr(self, "cross_scale"):
                scaled_query_cross = scaled_query_cross * torch.clamp(
                    self.cross_scale.to(scaled_query_cross.dtype), min=0.1, max=5.0
                )
            cross_scale = float(self.scaling)
            cross_out = sdpa_cross_attention(
                scaled_query_cross,
                cross_key,
                cross_value,
                encoder_valid,
                decoder_valid,
                cross_scale,
                chunk_size=chunk_sz,
                use_sdpa=not getattr(self, "_force_eager_cross", False),
            )
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


def _resolve_gate_init_tanh(model: Any, explicit: float | None = None) -> float:
    """Resolve gate_init_tanh from explicit kwarg, model config, v4 fallback (0.15), or config.GATE_INIT_TANH."""
    if explicit is not None:
        return float(explicit)
    cfg = getattr(model, "config", None)
    cfg_val = getattr(cfg, "gate_init_tanh", None) if cfg is not None else None
    if cfg_val is not None:
        return float(cfg_val)
    spec = getattr(model, "spec", None)
    arch_ver = str(getattr(spec, "architecture_version", "") or "").lower()
    v4_enabled = bool(
        getattr(cfg, "v4_enabled", False)
        or getattr(_config_mod, "V4_ENABLED", False)
        or "v4" in arch_ver
    )
    if v4_enabled:
        return float(getattr(_config_mod, "V4_GATE_INIT_TANH", _V4_DEFAULT_GATE_INIT_TANH))
    return float(getattr(_config_mod, "GATE_INIT_TANH", _V4_DEFAULT_GATE_INIT_TANH))


def configure_full_rank_cross_attention(
    model: Any,
    *,
    decouple_full_attn_cross_kv: bool | None = None,
    learnable_cross_scale: bool | None = None,
    gate_init_tanh: float | None = None,
) -> dict[str, Any]:
    """Attach SwiGLUDualGateBridge, full-rank BF16 cross-attention projections, decoupled upper full-attn cross-KV, and gated cross-attn."""
    ensure_hardware_memory_guards()
    setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)

    if getattr(model, "encoder", None) is None:
        return {
            "mode": "decoder_only_causal_v3",
            "bridge_type": None,
            "kv_owner_layers": 0,
            "decoupled_full_attn_cross_kv_layers": 0,
            "total_decoder_layers": len(model.decoder.layers),
            "total_cross_bridge_parameters": 0,
        }

    # Install SwiGLUDualGateBridge as model.encoder_projection, respecting config parameters
    enc_dim = int(model.encoder.config.hidden_size)
    dec_dim = int(model.decoder.config.hidden_size)
    dev = next(model.decoder.parameters()).device
    cfg = getattr(model, "config", None)
    spec = getattr(model, "spec", None)
    d_mid = getattr(cfg, "d_mid", None)
    init_rms = getattr(cfg, "init_rms_scale", 1.0)
    existing_proj = getattr(model, "encoder_projection", None)
    if (
        isinstance(existing_proj, SwiGLUDualGateBridge)
        and existing_proj.d_enc == enc_dim
        and existing_proj.d_dec == dec_dim
    ):
        model.encoder_projection = existing_proj.to(dev)
    else:
        model.encoder_projection = SwiGLUDualGateBridge(
            d_enc=enc_dim,
            d_dec=dec_dim,
            d_mid=d_mid,
            init_rms_scale=init_rms,
        ).to(dev)

    resolved_gate_init_tanh = _resolve_gate_init_tanh(model, explicit=gate_init_tanh)
    resolved_gate_init_tanh = max(-0.999, min(0.999, resolved_gate_init_tanh))
    gate_init_atanh = math.atanh(resolved_gate_init_tanh)

    if decouple_full_attn_cross_kv is not None:
        resolved_decouple_full_attn = bool(decouple_full_attn_cross_kv)
    elif cfg is not None and hasattr(cfg, "decouple_full_attn_cross_kv"):
        resolved_decouple_full_attn = bool(getattr(cfg, "decouple_full_attn_cross_kv"))
    elif spec is not None and hasattr(spec, "decouple_full_attn_cross_kv"):
        resolved_decouple_full_attn = bool(getattr(spec, "decouple_full_attn_cross_kv"))
    else:
        resolved_decouple_full_attn = bool(
            getattr(_config_mod, "DECOUPLE_FULL_ATTN_CROSS_KV", True)
        )

    if learnable_cross_scale is not None:
        resolved_learnable_scale = bool(learnable_cross_scale)
    elif cfg is not None and hasattr(cfg, "learnable_cross_scale"):
        resolved_learnable_scale = bool(getattr(cfg, "learnable_cross_scale"))
    else:
        resolved_learnable_scale = bool(getattr(_config_mod, "LEARNABLE_CROSS_SCALE", False))

    kv_owner_layers = 0
    decoupled_full_attn_layers = 0
    total_cross_params = 0
    full_attn_kv_owner: Any = None
    for layer in model.decoder.layers:
        attn = layer.self_attn
        q_base = getattr(attn.q_proj, "base", attn.q_proj)
        dev = q_base.weight.device
        num_heads = int(q_base.weight.shape[0] // attn.head_dim)
        attn.cross_q_norm = copy.deepcopy(attn.q_norm).float().to(dev)
        attn.cross_gate = torch.nn.Parameter(
            torch.full((num_heads,), float(gate_init_atanh), dtype=torch.float32, device=dev)
        )
        attn.cross_gate_proj = torch.nn.Parameter(
            torch.empty((num_heads, attn.head_dim), dtype=torch.float32, device=dev)
        )
        torch.nn.init.zeros_(attn.cross_gate_proj)
        attn.cross_scale = torch.nn.Parameter(
            torch.ones((num_heads, 1, 1), dtype=torch.float32, device=dev),
            requires_grad=resolved_learnable_scale,
        )
        is_full_attn = getattr(attn, "layer_type", None) == "full_attention"
        if not attn.is_kv_shared_layer:
            if is_full_attn:
                full_attn_kv_owner = attn
            v_source = attn.v_proj if getattr(attn, "v_proj", None) is not None else attn.k_proj
            attn.cross_k_proj = Bf16FullLinear(attn.k_proj).to(dev)
            attn.cross_v_proj = Bf16FullLinear(v_source).to(dev)
            attn.cross_k_norm = copy.deepcopy(attn.k_norm).float().to(dev)
            attn.cross_v_norm = copy.deepcopy(attn.v_norm).float().to(dev)
            attn.owns_cross_kv = True
            kv_owner_layers += 1
        elif resolved_decouple_full_attn and is_full_attn and full_attn_kv_owner is not None:
            v_source = (
                full_attn_kv_owner.v_proj
                if getattr(full_attn_kv_owner, "v_proj", None) is not None
                else full_attn_kv_owner.k_proj
            )
            attn.cross_k_proj = Bf16FullLinear(full_attn_kv_owner.k_proj).to(dev)
            attn.cross_v_proj = Bf16FullLinear(v_source).to(dev)
            attn.cross_k_norm = copy.deepcopy(full_attn_kv_owner.k_norm).float().to(dev)
            attn.cross_v_norm = copy.deepcopy(full_attn_kv_owner.v_norm).float().to(dev)
            attn.owns_cross_kv = True
            kv_owner_layers += 1
            decoupled_full_attn_layers += 1
        else:
            attn.owns_cross_kv = False
            for stale_attr in ("cross_k_proj", "cross_v_proj", "cross_k_norm", "cross_v_norm"):
                if hasattr(attn, stale_attr):
                    delattr(attn, stale_attr)
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
        "gate_init_tanh": resolved_gate_init_tanh,
        "kv_owner_layers": kv_owner_layers,
        "decoupled_full_attn_cross_kv_layers": decoupled_full_attn_layers,
        "decouple_full_attn_cross_kv": resolved_decouple_full_attn,
        "learnable_cross_scale": resolved_learnable_scale,
        "total_decoder_layers": len(model.decoder.layers),
        "total_cross_bridge_parameters": total_cross_params,
    }


configure_cross_attention = configure_full_rank_cross_attention
