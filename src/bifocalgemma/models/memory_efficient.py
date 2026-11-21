"""Bounded-memory exact attention: query tiles, FP32 softmax, joint normalization.

No quantization, context truncation, independent cross softmax or detached encoder.
Eager remains the reference. Recompute trades throughput for activation memory.
"""

import copy

import torch
import torch.utils.checkpoint
from torch.nn import functional as F
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, apply_rotary_pos_emb

DEFAULT_MEMORY_POLICY = {
    "version": "tiled_recompute_v1",
    "attention_backend": "tiled_eager_fp32",
    "query_chunk_size": 128,
    "loss_chunk_size": 128,
    "gradient_checkpointing": True,
    "ple_offload": False,
    "kv_gradient_accumulation": "float32",
}


def _kv_slice_bounds(
    attention_mask,
    encoder_valid,
    start: int,
    stop: int,
    k_len: int,
    window: int | None,
) -> tuple[int, int]:
    """Return exact `[k_start, k_stop)` key bounds outside which visibility is guaranteed False."""
    if encoder_valid is not None or not isinstance(attention_mask, dict) or attention_mask.get("cross"):
        return 0, k_len
    bidirectional = bool(attention_mask.get("bidirectional", False))
    if window:
        if bidirectional:
            return max(0, start - window), min(k_len, stop + window)
        return max(0, start - window + 1), min(k_len, stop)
    if not bidirectional:
        return 0, min(k_len, stop)
    return 0, k_len


def mask_tile(
    attention_mask,
    encoder_valid,
    decoder_valid,
    start,
    stop,
    window,
    *,
    k_start: int = 0,
    k_stop: int | None = None,
):
    if encoder_valid is not None:
        return joint_visibility_tile(encoder_valid, decoder_valid, start, stop, window)
    if isinstance(attention_mask, dict):
        if attention_mask.get("cross"):
            enc_valid = attention_mask["encoder_valid"].bool()
            q_valid = attention_mask.get("query_valid")
            mask = enc_valid[:, None, None, :].expand(enc_valid.shape[0], 1, stop - start, -1)
            if q_valid is not None:
                mask = mask & q_valid[:, None, start:stop, None].bool()
            return mask
        valid = attention_mask["valid"]
        eff_k_stop = valid.shape[1] if k_stop is None else k_stop
        q_pos = torch.arange(start, stop, device=valid.device)[:, None]
        k_pos = torch.arange(k_start, eff_k_stop, device=valid.device)[None, :]
        visible = torch.ones((stop - start, eff_k_stop - k_start), device=valid.device, dtype=torch.bool)
        if not attention_mask["bidirectional"]:
            visible &= k_pos <= q_pos
        if window:
            distance = (k_pos - q_pos).abs()
            visible &= distance <= window if attention_mask["bidirectional"] else distance < window
        return (
            visible[None, None]
            & valid[:, None, None, k_start:eff_k_stop].bool()
            & valid[:, None, start:stop, None].bool()
        )
    return attention_mask[..., start:stop, :] if attention_mask is not None else None


def build_joint_visibility(
    encoder_valid: torch.Tensor,
    decoder_valid: torch.Tensor,
    self_window: int | None = None,
) -> torch.Tensor:
    """Keys are [decoder, encoder]; decoder causal/windowed, encoder globally visible."""
    if encoder_valid.ndim != 2 or decoder_valid.ndim != 2:
        raise ValueError("Validity masks must have [batch, sequence] shape.")
    if encoder_valid.shape[0] != decoder_valid.shape[0]:
        raise ValueError("Encoder and decoder batch sizes differ.")
    if self_window is not None and self_window <= 0:
        raise ValueError("Self-attention window must be positive.")
    length = decoder_valid.shape[1]
    positions = torch.arange(length, device=decoder_valid.device)
    visibility = positions[None, :] <= positions[:, None]
    if self_window is not None:
        visibility &= positions[None, :] > positions[:, None] - self_window
    self_visible = visibility[None] & decoder_valid.bool()[:, None, :]
    cross_visible = encoder_valid.bool()[:, None, :].expand(-1, length, -1)
    return (
        torch.cat((self_visible, cross_visible), dim=-1)[:, None]
        & decoder_valid.bool()[:, None, :, None]
    )


def joint_visibility_tile(encoder_valid, decoder_valid, start, stop, window):
    positions = torch.arange(decoder_valid.shape[1], device=decoder_valid.device)
    queries = positions[start:stop, None]
    visible = positions[None, :] <= queries
    if window is not None:
        visible &= positions[None, :] > queries - window
    self_visible = visible[None] & decoder_valid[:, None, :].bool()
    cross_visible = encoder_valid[:, None, :].bool().expand(-1, stop - start, -1)
    return (
        torch.cat((self_visible, cross_visible), -1)[:, None]
        & decoder_valid[:, None, start:stop, None].bool()
    )


def attention_tile(query, key, value, mask, scaling, dropout, boolean_mask):
    batch, heads, length, width = query.shape
    kv_heads = key.shape[1]
    if heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads.")
    groups = heads // kv_heads
    # Preserve native head/group layout. Backend matmul workspace is transient.
    key = key.to(query.dtype)
    value = value.to(query.dtype)
    grouped_query = query.reshape(batch, kv_heads, groups, length, width)
    scores = torch.matmul(grouped_query, key[:, :, None].transpose(-1, -2))
    scores = scores.reshape(batch, heads, length, key.shape[2]) * scaling
    if mask is not None:
        if boolean_mask:
            scores = scores.masked_fill(~mask, float("-inf"))
            scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        else:
            scores = scores + mask
    probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
    if mask is not None and boolean_mask:
        probabilities = probabilities.masked_fill(~mask, 0)
    probabilities = torch.nn.functional.dropout(probabilities, p=dropout, training=dropout > 0)
    grouped_probabilities = probabilities.reshape(batch, kv_heads, groups, length, key.shape[2])
    attended = torch.matmul(grouped_probabilities, value[:, :, None])
    return attended.reshape(batch, heads, length, value.shape[-1])


class _TiledAttention(torch.autograd.Function):
    """First-order tiled eager attention; FP32 KV accumulation before one cast.

    Forward keeps native score/probability dtypes. Backward recomputes softmax
    tiles; it does not sum BF16 partial KV gradients from separate graph nodes.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attention_mask,
        encoder_valid,
        decoder_valid,
        chunk_size,
        scaling,
        window,
    ):
        ctx.save_for_backward(query, key, value)
        ctx.attention_mask, ctx.encoder_valid, ctx.decoder_valid = (
            attention_mask,
            encoder_valid,
            decoder_valid,
        )
        ctx.chunk_size, ctx.scaling, ctx.window = chunk_size, scaling, window
        pieces = []
        is_boolean_mask = (
            encoder_valid is not None
            or isinstance(attention_mask, dict)
            or (isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.bool)
        )
        k_len = key.shape[2]
        for start in range(0, query.shape[2], chunk_size):
            stop = min(start + chunk_size, query.shape[2])
            k_start, k_stop = _kv_slice_bounds(
                attention_mask, encoder_valid, start, stop, k_len, window
            )
            mask = mask_tile(
                attention_mask,
                encoder_valid,
                decoder_valid,
                start,
                stop,
                window,
                k_start=k_start,
                k_stop=k_stop,
            )
            if isinstance(mask, torch.Tensor) and mask.device != query.device:
                mask = mask.to(query.device)
            pieces.append(
                attention_tile(
                    query[:, :, start:stop],
                    key[:, :, k_start:k_stop],
                    value[:, :, k_start:k_stop],
                    mask,
                    scaling,
                    0.0,
                    is_boolean_mask,
                )
            )
        return torch.cat(pieces, dim=2)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        query, key, value = ctx.saved_tensors
        batch, heads, length, width = query.shape
        groups = heads // key.shape[1]
        # Match native repeat_kv head order, including its final head-group reduction.
        repeated_key = key.to(query.dtype).repeat_interleave(groups, dim=1)
        repeated_value = value.to(query.dtype).repeat_interleave(groups, dim=1)
        accumulation_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        dk = torch.zeros(repeated_key.shape, dtype=accumulation_dtype, device=key.device)
        dv = torch.zeros(repeated_value.shape, dtype=accumulation_dtype, device=value.device)
        dq = torch.zeros_like(query)
        merged = (
            ctx.encoder_valid is not None
            or isinstance(ctx.attention_mask, dict)
            or (
                isinstance(ctx.attention_mask, torch.Tensor)
                and ctx.attention_mask.dtype == torch.bool
            )
        )
        k_len = key.shape[2]
        for start in range(0, length, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, length)
            q = query[:, :, start:stop]
            grad = grad_output[:, :, start:stop]
            k_start, k_stop = _kv_slice_bounds(
                ctx.attention_mask, ctx.encoder_valid, start, stop, k_len, ctx.window
            )
            k_slice = repeated_key[:, :, k_start:k_stop]
            v_slice = repeated_value[:, :, k_start:k_stop]
            mask = mask_tile(
                ctx.attention_mask,
                ctx.encoder_valid,
                ctx.decoder_valid,
                start,
                stop,
                ctx.window,
                k_start=k_start,
                k_stop=k_stop,
            )
            if isinstance(mask, torch.Tensor) and mask.device != query.device:
                mask = mask.to(query.device)
            scores = torch.matmul(q, k_slice.transpose(-1, -2)) * ctx.scaling
            if mask is not None:
                if merged:
                    scores = scores.masked_fill(~mask, float("-inf"))
                    scores = torch.where(
                        mask.any(-1, keepdim=True), scores, torch.zeros_like(scores)
                    )
                else:
                    scores = scores + mask
            raw_probabilities = scores.softmax(-1, dtype=torch.float32)
            probabilities = raw_probabilities.to(query.dtype)
            if mask is not None and merged:
                probabilities = probabilities.masked_fill(~mask, 0)
            # dP has native matmul dtype; softmax backward uses native FP32 precision.
            dp = torch.matmul(grad, v_slice.transpose(-1, -2))
            if mask is not None and merged:
                dp = dp.masked_fill(~mask, 0)
            ds = torch.ops.aten._softmax_backward_data(
                dp.float(), raw_probabilities, -1, torch.float32
            )
            ds = ds.to(query.dtype)
            if mask is not None and merged:
                ds = ds.masked_fill(~mask, 0)
            ds = ds * ctx.scaling
            dq[:, :, start:stop] = torch.matmul(ds, k_slice)
            # Sum partial query reductions in FP32 instead of repeated BF16 adds.
            dk[:, :, k_start:k_stop].add_(
                torch.matmul(ds.to(accumulation_dtype).transpose(-1, -2), q.to(accumulation_dtype))
            )
            dv[:, :, k_start:k_stop].add_(
                torch.matmul(
                    probabilities.to(accumulation_dtype).transpose(-1, -2),
                    grad.to(accumulation_dtype),
                )
            )
        dk = dk.to(key.dtype).reshape(batch, key.shape[1], groups, key.shape[2], width).sum(dim=2)
        dv = (
            dv.to(value.dtype)
            .reshape(batch, value.shape[1], groups, value.shape[2], value.shape[-1])
            .sum(dim=2)
        )
        return dq, dk, dv, None, None, None, None, None, None


def _sdpa_supported(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    """Return True when F.scaled_dot_product_attention can safely execute this Q/K/V layout."""
    if not hasattr(F, "scaled_dot_product_attention"):
        return False
    if query.dtype == torch.float64 or key.dtype == torch.float64 or value.dtype == torch.float64:
        return False
    heads = query.shape[1]
    kv_heads = key.shape[1]
    return kv_heads > 0 and (heads % kv_heads == 0)


def sdpa_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    encoder_valid: torch.Tensor,
    query_valid: torch.Tensor | None,
    scaling: float,
    *,
    chunk_size: int = 128,
    dropout: float = 0.0,
    use_sdpa: bool = True,
) -> torch.Tensor:
    """Fused SDPA cross-attention with broadcast encoder mask and safe eager fallback.

    Preserves exact mask semantics (including wholly masked encoder/query rows)
    without materializing a dense [B, 1, T_dec, N_enc] mask tensor. During training
    (`torch.is_grad_enabled()`), `head_dim > 256` routes to recomputed `_TiledAttention`
    so Math SDPA cannot retain O(T_dec * N_enc) softmax activations across tiles.
    """
    batch, heads, length, width = query.shape
    if key.shape[2] == 0:
        return query.new_zeros((batch, heads, length, value.shape[-1]))
    eff_chunk = max(1, chunk_size)
    max_dim = max(width, value.shape[-1])
    enc_mask_bool = encoder_valid.to(device=query.device, dtype=torch.bool)
    q_valid_bool = (
        query_valid.to(device=query.device, dtype=torch.bool) if query_valid is not None else None
    )
    cross_mask_spec = {
        "cross": True,
        "encoder_valid": enc_mask_bool,
        "query_valid": q_valid_bool,
    }

    # On CUDA, Flash/MemEfficient SDPA require head_dim <= 256. For head_dim > 256
    # during training, tiling Math SDPA in Python retains every tile's softmax tensor
    # simultaneously (O(T_dec * N_enc) VRAM). Route to recomputed _TiledAttention instead.
    if torch.is_grad_enabled() and max_dim > 256 and dropout == 0.0:
        return _TiledAttention.apply(
            query, key, value, cross_mask_spec, None, None, eff_chunk, scaling, None
        )

    if use_sdpa and _sdpa_supported(query, key, value):
        try:
            k_cast = key.to(query.dtype)
            v_cast = value.to(query.dtype)
            enc_mask = enc_mask_bool[:, None, None, :]
            has_enc = enc_mask.any(dim=-1, keepdim=True)
            safe_enc_mask = enc_mask | ~has_enc
            enable_gqa = heads != k_cast.shape[1]
            # Single-shot when short or Flash/MemEfficient-eligible (head_dim <= 256 on CUDA);
            # tile over query length otherwise to keep transient workspace bounded for head_dim=512.
            if length <= eff_chunk or (query.device.type == "cuda" and max_dim <= 256):
                out = F.scaled_dot_product_attention(
                    query,
                    k_cast,
                    v_cast,
                    attn_mask=safe_enc_mask,
                    dropout_p=dropout,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                )
            else:
                checkpoint_tiles = torch.is_grad_enabled() and max_dim > 256

                def _sdpa_tile(
                    q_slice: torch.Tensor,
                    k_tensor: torch.Tensor,
                    v_tensor: torch.Tensor,
                    mask_tensor: torch.Tensor,
                ) -> torch.Tensor:
                    return F.scaled_dot_product_attention(
                        q_slice,
                        k_tensor,
                        v_tensor,
                        attn_mask=mask_tensor,
                        dropout_p=dropout,
                        scale=scaling,
                        enable_gqa=enable_gqa,
                    )

                pieces = []
                for start in range(0, length, eff_chunk):
                    stop = min(length, start + eff_chunk)
                    q_slice = query[:, :, start:stop]
                    if checkpoint_tiles:
                        tile_out = torch.utils.checkpoint.checkpoint(
                            _sdpa_tile,
                            q_slice,
                            k_cast,
                            v_cast,
                            safe_enc_mask,
                            use_reentrant=False,
                        )
                    else:
                        tile_out = _sdpa_tile(q_slice, k_cast, v_cast, safe_enc_mask)
                    pieces.append(tile_out)
                out = torch.cat(pieces, dim=2)
            out_valid = has_enc
            if q_valid_bool is not None:
                out_valid = out_valid & q_valid_bool[:, None, :, None]
            return out.masked_fill(~out_valid, 0.0)
        except (RuntimeError, TypeError):
            pass

    if dropout == 0.0:
        return _TiledAttention.apply(
            query, key, value, cross_mask_spec, None, None, eff_chunk, scaling, None
        )

    pieces = []
    for start in range(0, length, eff_chunk):
        stop = min(length, start + eff_chunk)
        cross_mask = enc_mask_bool[:, None, None, :].expand(batch, 1, stop - start, -1)
        if q_valid_bool is not None:
            cross_mask = cross_mask & q_valid_bool[:, None, start:stop, None]
        pieces.append(
            attention_tile(
                query[:, :, start:stop],
                key,
                value,
                cross_mask,
                scaling,
                dropout,
                True,
            )
        )
    return torch.cat(pieces, dim=2)


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    chunk_size: int = 128,
    scaling: float = 1.0,
    dropout: float = 0.0,
    attention_mask=None,
    encoder_valid=None,
    decoder_valid=None,
    window=None,
) -> torch.Tensor:
    """SDPA / FlashAttention fast path with window-sliced KV tiles and safe eager fallback.

    During training (`torch.is_grad_enabled()`), `head_dim > 256` routes to recomputed
    `_TiledAttention` so Math SDPA cannot retain O(N^2) softmax activations across tiles.
    """
    if not _sdpa_supported(query, key, value):
        return _TiledAttention.apply(
            query, key, value, attention_mask, encoder_valid, decoder_valid, chunk_size, scaling, window
        )
    # Non-boolean custom float additive masks fall back cleanly to tiled eager FP32.
    if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype != torch.bool:
        return _TiledAttention.apply(
            query, key, value, attention_mask, encoder_valid, decoder_valid, chunk_size, scaling, window
        )

    batch, heads, q_len, width = query.shape
    max_dim = max(width, value.shape[-1])
    if torch.is_grad_enabled() and max_dim > 256 and dropout == 0.0:
        return _TiledAttention.apply(
            query, key, value, attention_mask, encoder_valid, decoder_valid, chunk_size, scaling, window
        )
    k_len = key.shape[2]
    k_cast = key.to(query.dtype)
    v_cast = value.to(query.dtype)
    enable_gqa = heads != k_cast.shape[1]
    eff_chunk = max(chunk_size, 256) if not torch.is_grad_enabled() else max(1, chunk_size)
    checkpoint_tiles = torch.is_grad_enabled() and max_dim > 256

    def _run_sdpa_tile(
        q_slice: torch.Tensor,
        k_slice: torch.Tensor,
        v_slice: torch.Tensor,
        attn_mask_slice: torch.Tensor | None,
    ) -> torch.Tensor:
        if checkpoint_tiles:
            if attn_mask_slice is None:
                return torch.utils.checkpoint.checkpoint(
                    lambda q_, k_, v_: F.scaled_dot_product_attention(
                        q_,
                        k_,
                        v_,
                        attn_mask=None,
                        dropout_p=dropout,
                        scale=scaling,
                        enable_gqa=enable_gqa,
                    ),
                    q_slice,
                    k_slice,
                    v_slice,
                    use_reentrant=False,
                )
            return torch.utils.checkpoint.checkpoint(
                lambda q_, k_, v_, m_: F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=m_,
                    dropout_p=dropout,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                ),
                q_slice,
                k_slice,
                v_slice,
                attn_mask_slice,
                use_reentrant=False,
            )
        return F.scaled_dot_product_attention(
            q_slice,
            k_slice,
            v_slice,
            attn_mask=attn_mask_slice,
            dropout_p=dropout,
            scale=scaling,
            enable_gqa=enable_gqa,
        )

    try:
        if encoder_valid is not None:
            pieces = []
            for start in range(0, q_len, eff_chunk):
                stop = min(q_len, start + eff_chunk)
                mask = joint_visibility_tile(encoder_valid, decoder_valid, start, stop, window)
                has_any = mask.any(dim=-1, keepdim=True)
                safe_mask = mask | ~has_any
                tile_out = _run_sdpa_tile(query[:, :, start:stop], k_cast, v_cast, safe_mask)
                pieces.append(tile_out.masked_fill(~has_any, 0.0))
            return torch.cat(pieces, dim=2)

        if isinstance(attention_mask, dict):
            valid = attention_mask["valid"].to(device=query.device, dtype=torch.bool)
            bidirectional = bool(attention_mask["bidirectional"])
            positions = torch.arange(k_len, device=query.device)

            if bidirectional and not window:
                k_mask = valid[:, None, None, :]
                has_any = k_mask.any(dim=-1, keepdim=True)
                safe_mask = k_mask | ~has_any
                q_valid = valid[:, None, :, None]
                if q_len <= eff_chunk or (query.device.type == "cuda" and max_dim <= 256):
                    out = F.scaled_dot_product_attention(
                        query,
                        k_cast,
                        v_cast,
                        attn_mask=safe_mask,
                        dropout_p=dropout,
                        scale=scaling,
                        enable_gqa=enable_gqa,
                    )
                else:
                    pieces = []
                    for start in range(0, q_len, eff_chunk):
                        stop = min(q_len, start + eff_chunk)
                        pieces.append(
                            _run_sdpa_tile(query[:, :, start:stop], k_cast, v_cast, safe_mask)
                        )
                    out = torch.cat(pieces, dim=2)
                return out.masked_fill(~(has_any & q_valid), 0.0)

            if (
                not bidirectional
                and not window
                and q_len == k_len
                and query.device.type == "cuda"
                and max_dim <= 256
                and bool(valid.all())
            ):
                return F.scaled_dot_product_attention(
                    query,
                    k_cast,
                    v_cast,
                    attn_mask=None,
                    dropout_p=dropout,
                    is_causal=True,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                )

            # Use large SDPA tiles for Flash/MemEfficient-eligible head_dim <= 256 so 32K sliding
            # windows execute in 16-32 fused CUDA launches instead of 256 Python loop iterations.
            sdpa_tile_chunk = (
                max(eff_chunk, 2048 if not torch.is_grad_enabled() else 1024)
                if max_dim <= 256
                else eff_chunk
            )
            pieces = []
            for start in range(0, q_len, sdpa_tile_chunk):
                stop = min(q_len, start + sdpa_tile_chunk)
                q_pos = positions[start:stop, None]
                if window:
                    if bidirectional:
                        k_start = max(0, start - window)
                        k_stop = min(k_len, stop + window)
                        k_pos = positions[None, k_start:k_stop]
                        visible = (k_pos - q_pos).abs() <= window
                    else:
                        k_start = max(0, start - window + 1)
                        k_stop = min(k_len, stop)
                        k_pos = positions[None, k_start:k_stop]
                        visible = (k_pos <= q_pos) & ((q_pos - k_pos) < window)
                else:
                    # Full causal self-attention: only keys [:stop] can be visible.
                    k_start = 0
                    k_stop = min(k_len, stop)
                    k_pos = positions[None, k_start:k_stop]
                    visible = k_pos <= q_pos

                tile_mask = visible[None, None] & valid[:, None, None, k_start:k_stop]
                has_any = tile_mask.any(dim=-1, keepdim=True)
                safe_mask = tile_mask | ~has_any
                tile_out = _run_sdpa_tile(
                    query[:, :, start:stop],
                    k_cast[:, :, k_start:k_stop],
                    v_cast[:, :, k_start:k_stop],
                    safe_mask,
                )
                pieces.append(
                    tile_out.masked_fill(~(has_any & valid[:, None, start:stop, None]), 0.0)
                )
            return torch.cat(pieces, dim=2)

        if attention_mask is None:
            if q_len <= eff_chunk or (query.device.type == "cuda" and max_dim <= 256):
                return F.scaled_dot_product_attention(
                    query,
                    k_cast,
                    v_cast,
                    attn_mask=None,
                    dropout_p=dropout,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                )
            pieces = []
            for start in range(0, q_len, eff_chunk):
                stop = min(q_len, start + eff_chunk)
                pieces.append(_run_sdpa_tile(query[:, :, start:stop], k_cast, v_cast, None))
            return torch.cat(pieces, dim=2)

        # Boolean tensor mask
        pieces = []
        for start in range(0, q_len, eff_chunk):
            stop = min(q_len, start + eff_chunk)
            tile_mask = attention_mask[..., start:stop, :].to(device=query.device, dtype=torch.bool)
            has_any = tile_mask.any(dim=-1, keepdim=True)
            safe_mask = tile_mask | ~has_any
            tile_out = _run_sdpa_tile(query[:, :, start:stop], k_cast, v_cast, safe_mask)
            pieces.append(tile_out.masked_fill(~has_any, 0.0))
        return torch.cat(pieces, dim=2)
    except (RuntimeError, TypeError):
        return _TiledAttention.apply(
            query, key, value, attention_mask, encoder_valid, decoder_valid, chunk_size, scaling, window
        )


def tiled_attention(
    query,
    key,
    value,
    *,
    chunk_size,
    scaling=1.0,
    dropout=0.0,
    attention_mask=None,
    encoder_valid=None,
    decoder_valid=None,
    window=None,
    backend="tiled_eager_fp32",
):
    if chunk_size < 1 or query.shape[2] < 1:
        raise ValueError("Attention chunk size and query length must be positive.")
    if dropout != 0.0:
        raise ValueError("Conservative tiled attention is validated for native Gemma dropout=0.")
    if encoder_valid is not None and decoder_valid is None:
        raise ValueError("Merged attention requires decoder validity.")
    if backend == "sdpa":
        return sdpa_attention(
            query,
            key,
            value,
            chunk_size=chunk_size,
            scaling=scaling,
            dropout=dropout,
            attention_mask=attention_mask,
            encoder_valid=encoder_valid,
            decoder_valid=decoder_valid,
            window=window,
        )
    if backend == "flex_attention" or (backend == "hybrid_flex" and query.shape[-1] <= 256):
        from bifocalgemma.models.flex_attention import fused_attention

        return fused_attention(
            query,
            key,
            value,
            scaling=scaling,
            attention_mask=attention_mask,
            encoder_valid=encoder_valid,
            decoder_valid=decoder_valid,
            window=window,
        )
    return _TiledAttention.apply(
        query, key, value, attention_mask, encoder_valid, decoder_valid, chunk_size, scaling, window
    )


class TiledEncoderAttention(Gemma4TextAttention):
    """Native self attention, including RoPE/KV owners, with bounded score tiles."""

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            raise ValueError("Tiled training attention does not support temporal cache.")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        if self.is_kv_shared_layer:
            key, value = shared_kv_states[self.layer_type]
            key, value = key.to(query.device), value.to(query.device)
        else:
            key = self.k_proj(hidden_states).view(shape)
            value = self.v_proj(hidden_states).view(shape) if self.v_proj is not None else key
            key = self.k_norm(key)
            key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value = self.v_norm(value).transpose(1, 2)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (key, value)
        valid = kwargs.get("self_valid") if getattr(self, "structured_masks", False) else None
        if valid is not None:
            attention_mask = {"valid": valid, "bidirectional": not self.is_causal}
        backend = getattr(self, "attention_backend", "tiled_eager_fp32")
        if (
            backend == "tiled_eager_fp32"
            and not self.training
            and not torch.is_grad_enabled()
            and not getattr(self, "_force_eager_eval", False)
        ):
            backend = "sdpa"
        result = tiled_attention(
            query,
            key,
            value,
            chunk_size=self.query_chunk_size,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            attention_mask=attention_mask,
            window=self.sliding_window,
            backend=backend,
        )
        result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(result), None


class TiledMergedAttention(TiledEncoderAttention):
    """A single joint softmax over causal/windowed self keys and all encoder keys."""

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        past_key_values=None,
        *,
        encoder_hidden_states=None,
        encoder_valid=None,
        decoder_valid=None,
        **kwargs,
    ):
        if past_key_values is not None:
            raise ValueError("Tiled training attention does not support temporal cache.")
        if encoder_hidden_states is None or encoder_valid is None or decoder_valid is None:
            raise ValueError("Merged attention requires encoder states and validity masks.")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        if self.is_kv_shared_layer:
            key, value = shared_kv_states[self.layer_type]
        else:
            if self.v_proj is None:
                raise ValueError("Merged branch requires separate value projection.")
            key = self.k_norm(self.k_proj(hidden_states).view(shape))
            key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
            cross_shape = (
                batch,
                encoder_hidden_states.shape[1],
                self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                self.head_dim,
            )
            cross_key = self.k_norm(self.k_proj(encoder_hidden_states).view(cross_shape)).transpose(
                1, 2
            )
            cross_value = self.v_norm(
                self.v_proj(encoder_hidden_states).view(cross_shape)
            ).transpose(1, 2)
            key = torch.cat((key, cross_key), dim=2)
            value = torch.cat((value, cross_value), dim=2)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (key, value)
        if getattr(self, "collect_encoder_mass", False):
            from bifocalgemma.training.diagnostics import sample_encoder_mass

            sample_encoder_mass(self, query, key, encoder_valid, decoder_valid)
        result = tiled_attention(
            query,
            key,
            value,
            chunk_size=self.query_chunk_size,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            encoder_valid=encoder_valid,
            decoder_valid=decoder_valid,
            window=self.sliding_window,
            backend=getattr(self, "attention_backend", "tiled_eager_fp32"),
        )
        result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(result), None


def configure_memory_efficient_training(model, policy=None):
    options = {**DEFAULT_MEMORY_POLICY, **(policy or {})}
    if set(options) != set(DEFAULT_MEMORY_POLICY) or options["version"] != "tiled_recompute_v1":
        raise ValueError("Unknown memory policy.")
    if (
        options["attention_backend"]
        not in {"tiled_eager_fp32", "sdpa", "flex_attention", "hybrid_flex"}
        or options["ple_offload"] is not False
    ):
        raise ValueError(
            "Only conservative tiled FP32 attention with GPU-resident PLE is supported."
        )
    if options["kv_gradient_accumulation"] != "float32":
        raise ValueError("KV gradients must accumulate in FP32 for tiled training.")
    for name in ("query_chunk_size", "loss_chunk_size"):
        value = options[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Invalid {name}.")
    if type(options["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean.")
    for role in ("encoder", "decoder"):
        stack = getattr(model, role, None)
        if stack is None:
            continue
        for layer in stack.layers:
            source = layer.self_attn
            target = copy.copy(source)
            target.__class__ = (
                TiledMergedAttention
                if role == "decoder" and model.encoder is not None
                else TiledEncoderAttention
            )
            target.query_chunk_size = options["query_chunk_size"]
            target.attention_backend = options["attention_backend"]
            target.structured_masks = model.spec.architecture_version == "normalized_merged_v2"
            layer.self_attn = target
        if options["gradient_checkpointing"]:
            stack.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            stack.gradient_checkpointing_disable()
        if stack.config.attention_dropout != 0.0:
            raise ValueError("Tiled memory policy requires native attention dropout=0.")
        stack.config.use_cache = False
    model.loss_chunk_size = options["loss_chunk_size"]
    model.memory_policy = options
    return copy.deepcopy(options)


def run_training_memory_preflight(
    model, examples, *, progress=None, batch_size=1, min_headroom_gib=0
):
    """Probe actual maxima plus a combined shape stress, without any optimizer update.

    Restores RNG/mode and discards probe gradients. Combined context/target is a
    shape-only stress fixture, never inserted into the training cohort.
    """
    from bifocalgemma.contracts import TokenizedRepairExample
    from bifocalgemma.data.collator import RepairCollator
    from bifocalgemma.training.telemetry import resource_snapshot

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Preflight batch size must be a positive integer.")
    if not examples or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Preflight requires fresh parameters and nonempty examples.")

    def cost(item):
        e = len(item.encoder_ids)
        d = len(item.decoder_prefix_ids) + len(item.target_ids) - 1
        return e * e + d * (e + d)

    if hasattr(examples, "extreme_examples"):
        longest_encoder, longest_decoder, hardest_joint = examples.extreme_examples()
    else:
        longest_encoder = max(examples, key=lambda x: len(x.encoder_ids))
        longest_decoder = max(examples, key=lambda x: len(x.decoder_prefix_ids) + len(x.target_ids))
        hardest_joint = max(examples, key=cost)
    items = {x.instance_id: x for x in (longest_encoder, longest_decoder, hardest_joint)}
    combined = TokenizedRepairExample(
        "shape-only-combined-maxima",
        longest_encoder.encoder_ids,
        longest_decoder.decoder_prefix_ids,
        longest_decoder.target_ids,
        {"source": "shape_stress_only"},
    )
    probes = [*items.values(), combined]
    device = next(model.parameters()).device
    old_mode, old_rng = model.training, torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    results = []
    try:
        model.train()
        for item in probes:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            batch = RepairCollator()([item] * batch_size).to(device)
            row = {
                "stage": "memory_preflight_started",
                "batch_size": batch_size,
                "instance_ids": [item.instance_id],
                "encoder_tokens": batch.encoder_input_ids.shape[1],
                "decoder_tokens": batch.decoder_input_ids.shape[1],
                "batch_target_tokens": int((batch.labels != -100).sum()),
                "shape_only": item.instance_id == combined.instance_id,
            }
            if progress:
                progress({**row, **resource_snapshot(device)})
            model.zero_grad(set_to_none=True)
            output = model.forward_loss(batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError("Nonfinite memory-preflight loss.")
            output.loss.backward()
            gradients = [
                p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
            ]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError("Invalid memory-preflight gradients.")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row.update(
                stage="memory_preflight_complete",
                loss=output.loss.detach().item(),
                **resource_snapshot(device),
            )
            results.append(row)
            if device.type == "cuda" and min_headroom_gib:
                free, total = torch.cuda.mem_get_info(device)
                external = total - free - torch.cuda.memory_reserved(device)
                remaining = (total - external - torch.cuda.max_memory_allocated(device)) / 2**30
                row["estimated_peak_headroom_gib"] = remaining
                if remaining < min_headroom_gib:
                    raise RuntimeError(
                        f"Batch {batch_size} preflight leaves only {remaining:.2f} GiB; "
                        f"requires {min_headroom_gib} GiB for optimizer/runtime headroom. "
                        "Reduce batch size and preserve effective batch with accumulation."
                    )
            if progress:
                progress(row)
            del output, batch, gradients
            model.zero_grad(set_to_none=True)
        return {
            "status": "PASS",
            "scope": "actual_length_maxima_and_combined_shape_not_quality",
            "optimizer_updates": 0,
            "rows": results,
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(old_mode)
        torch.set_rng_state(old_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
