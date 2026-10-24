"""Structured masks for fused joint attention; selected only after GPU validation."""

from typing import Any

import torch

_compiled_flex = None
_compiled_mask = None


def flex_kernel_options(head_dim, dtype):
    """Bound backward shared-memory use for head256 as well as head512.

    The old qlen>=32/head256 branch left backward tiles to compiler defaults;
    those requested 112 KiB on the Molab runs, above its reported 99 KiB limit.
    This conservative experimental kernel still requires target-GPU validation.
    """
    options: dict[str, Any] = {
        "fwd_BLOCK_M": 16 if head_dim > 256 else 32,
        "fwd_BLOCK_N": 16 if head_dim > 256 else 64,
        "bwd_BLOCK_M1": 16,
        "bwd_BLOCK_N1": 16,
        "bwd_BLOCK_M2": 16,
        "bwd_BLOCK_N2": 16,
        "bwd_num_stages": 1,
    }
    if dtype == torch.float32:
        options.update(
            FLOAT32_PRECISION="'tf32x3'", fwd_num_stages=1, fwd_BLOCK_M=16, fwd_BLOCK_N=16
        )
    return options


def fused_attention(
    query,
    key,
    value,
    *,
    scaling,
    attention_mask=None,
    encoder_valid=None,
    decoder_valid=None,
    window=None,
):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if query.device.type != "cuda":
        raise ValueError("Production FlexAttention requires CUDA.")
    global _compiled_flex, _compiled_mask
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=True)
        _compiled_mask = torch.compile(create_block_mask, dynamic=True)
    assert _compiled_flex is not None and _compiled_mask is not None
    qlen, klen = query.shape[2], key.shape[2]
    if encoder_valid is not None:
        assert decoder_valid is not None
        dec_valid = decoder_valid
        enc_valid = encoder_valid
        if enc_valid.shape[1] == 0:
            enc_valid = torch.zeros((query.shape[0], 1), device=query.device, dtype=torch.bool)

        def visibility(b, h, q, k):
            qi = q.clamp(max=qlen - 1)
            self_index = k.clamp(max=qlen - 1)
            # Padded block indices must not index outside validity tensors.
            enc_index = (k - qlen).clamp(min=0, max=enc_valid.shape[1] - 1)
            self_ok = (k < qlen) & (k <= q)
            if window:
                self_ok = self_ok & (k > q - window)
            self_ok = self_ok & dec_valid[b, self_index].bool()
            cross_ok = (k >= qlen) & enc_valid[b, enc_index].bool()
            return (q < qlen) & (k < klen) & dec_valid[b, qi].bool() & (self_ok | cross_ok)
    elif isinstance(attention_mask, dict):
        valid = attention_mask["valid"]
        bidirectional = attention_mask["bidirectional"]

        def visibility(b, h, q, k):
            visible = (q < qlen) & (k < klen)
            if not bidirectional:
                visible = visible & (k <= q)
            if window:
                visible = visible & (
                    ((k - q).abs() <= window) if bidirectional else ((k - q).abs() < window)
                )
            return (
                visible
                & valid[b, q.clamp(max=qlen - 1)].bool()
                & valid[b, k.clamp(max=klen - 1)].bool()
            )
    else:
        raise ValueError("FlexAttention requires a structured visibility mask.")
    # Rebuild for each batch validity: never reuse a mask from another example.
    block_mask = _compiled_mask(
        visibility,
        B=query.shape[0],
        H=None,
        Q_LEN=qlen,
        KV_LEN=klen,
        device=str(query.device),
    )
    options = flex_kernel_options(query.shape[-1], query.dtype)
    from typing import cast

    from torch.nn.attention.flex_attention import FlexKernelOptions

    return _compiled_flex(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=True,
        kernel_options=cast(FlexKernelOptions, options),
    )
