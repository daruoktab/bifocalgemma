"""Per-generation self KV and static cross KV, respecting native shared owners."""

import torch
from torch.nn import functional as F
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, apply_rotary_pos_emb


class InferenceCache:
    def __init__(self):
        self.offset = 0
        self.layers = {}
        self.shared_positions = {}


class CachedAttention(Gemma4TextAttention):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        past_key_values=None,
        *,
        inference_cache=None,
        encoder_hidden_states=None,
        encoder_valid=None,
        **kwargs,
    ):
        if self.training or torch.is_grad_enabled():
            raise ValueError("Inference cache requires eval and inference_mode/no_grad.")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        cache = inference_cache
        if cache is None:
            raise ValueError("Explicit inference cache required.")
        positions = torch.arange(cache.offset, cache.offset + length, device=query.device)
        if self.is_kv_shared_layer:
            key, value = shared_kv_states[self.layer_type]
            key_positions, self_length = cache.shared_positions[self.layer_type]
        else:
            raw_key = self.k_proj(hidden_states).view(shape)
            raw_value = (
                self.v_proj(hidden_states).view(shape) if self.v_proj is not None else raw_key
            )
            key = apply_rotary_pos_emb(self.k_norm(raw_key), cos, sin, unsqueeze_dim=2).transpose(
                1, 2
            )
            value = self.v_norm(raw_value).transpose(1, 2)
            previous = cache.layers.get(self.layer_idx)
            key_positions = positions
            if previous is not None:
                key = torch.cat((previous[0], key), dim=2)
                value = torch.cat((previous[1], value), dim=2)
                key_positions = torch.cat((previous[2], positions))
            # Keep the full current prefill for queries, but bound future local self KV.
            keep = self.sliding_window if self.sliding_window else key.shape[2]
            stored_key, stored_value, stored_positions = (
                key[:, :, -keep:],
                value[:, :, -keep:],
                key_positions[-keep:],
            )
            cross = previous[3] if previous is not None else None
            if (
                encoder_hidden_states is not None
                and encoder_hidden_states.shape[1]
                and cross is None
            ):
                assert self.v_proj is not None
                cross_shape = (batch, encoder_hidden_states.shape[1], -1, self.head_dim)
                cross = (
                    self.k_norm(self.k_proj(encoder_hidden_states).view(cross_shape)).transpose(
                        1, 2
                    ),
                    self.v_norm(self.v_proj(encoder_hidden_states).view(cross_shape)).transpose(
                        1, 2
                    ),
                )
            cache.layers[self.layer_idx] = (stored_key, stored_value, stored_positions, cross)
            self_length = key.shape[2]
            if cross is not None:
                key = torch.cat((key, cross[0]), dim=2)
                value = torch.cat((value, cross[1]), dim=2)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (key, value)
            cache.shared_positions[self.layer_type] = (key_positions, self_length)
        pieces = []
        for start in range(0, length, 128):
            stop = min(length, start + 128)
            visible = key_positions[None, :] <= positions[start:stop, None]
            if self.sliding_window:
                visible &= (
                    key_positions[None, :] > positions[start:stop, None] - self.sliding_window
                )
            allowed = visible[None, None].expand(batch, 1, -1, -1)
            if key.shape[2] > self_length:
                assert encoder_valid is not None
                cross_visible = (
                    encoder_valid[:, None, None, :].bool().expand(batch, 1, stop - start, -1)
                )
                allowed = torch.cat((allowed, cross_visible), dim=-1)
            pieces.append(
                F.scaled_dot_product_attention(
                    query[:, :, start:stop],
                    key,
                    value,
                    attn_mask=allowed,
                    dropout_p=0.0,
                    scale=self.scaling,
                    enable_gqa=True,
                )
            )
        result = torch.cat(pieces, dim=2).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(result), None


def cached_logits(model, ids, cache, encoder_states=None, encoder_mask=None):
    """Single unpadded sequence; cache is scoped to this model/context generation."""
    if ids.shape[0] != 1:
        raise ValueError("Cached decoding currently requires one unpadded sequence.")
    if model.training or torch.is_grad_enabled():
        raise ValueError("Cached decoding requires eval and inference_mode/no_grad.")
    original = [layer.self_attn.__class__ for layer in model.decoder.layers]
    try:
        for layer in model.decoder.layers:
            layer.self_attn.__class__ = CachedAttention
        positions = torch.arange(cache.offset, cache.offset + ids.shape[1], device=ids.device)[None]
        hidden = model.decoder(
            input_ids=ids,
            position_ids=positions,
            attention_mask={"full_attention": None, "sliding_attention": None},
            use_cache=False,
            inference_cache=cache,
            encoder_hidden_states=encoder_states,
            encoder_valid=encoder_mask,
        ).last_hidden_state[:, -1:]
        logits = model.lm_head(hidden)
        cap = model.decoder.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        cache.offset += ids.shape[1]
        return logits
    finally:
        for layer, cls in zip(model.decoder.layers, original, strict=True):
            layer.self_attn.__class__ = cls
