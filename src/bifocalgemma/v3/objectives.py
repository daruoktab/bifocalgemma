"""Differential diff weighting, streaming pointer loss, and V3 optimizer groups."""

from __future__ import annotations

import math
import string
from typing import Any

import torch
import torch.utils.checkpoint
from torch.nn import functional as F

DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,  # <pad>, <eos>, <bos>, <unk>, <mask>, [multimodal]
    107,
    108,
    109,
    110,  # '\n', '\n\n', '\n\n\n', ...
    138,
    140,
    236743,  # '  ', '    ', ' ' (space)
    255968,  # '\t'
    236761,
    603,  # '.', ' .'
    236764,
    708,  # ',', ' ,'
    236787,
    794,  # ':', ' :'
    236793,
    1782,  # ';', ' ;'
    236769,
    833,  # '(', ' ('
    236768,
    1017,  # ')', ' )'
    236840,
    1673,  # '[', ' ['
    236842,
    1681,  # ']', ' ]'
    236782,
    1836,  # '{', ' {'
    236783,
    1888,  # '}', ' }'
    236784,
    808,  # '=', ' ='
    236862,
    900,  # '+', ' +'
    236772,
    753,  # '-', ' -'
    236829,
    1374,  # '*', ' *'
    236786,
    1195,  # '/', ' /'
)


def extract_trivial_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Extract all whitespace, punctuation, and special token IDs from tokenizer vocabulary."""
    if tokenizer is None:
        return DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS
    vocab = getattr(tokenizer, "vocab", None) or (
        tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    )
    if not vocab and hasattr(tokenizer, "tokenizer"):
        vocab = tokenizer.tokenizer.get_vocab()
    if not vocab:
        return DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS
    trivial = set()
    for tok, idx in vocab.items():
        cleaned = tok.replace("\u2581", " ").replace(" ", " ").strip()
        if not cleaned or all(ch in string.punctuation or ch.isspace() for ch in cleaned):
            trivial.add(int(idx))
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            trivial.add(int(val))
    return tuple(sorted(trivial))


class StreamingPointerLoss(torch.nn.Module):
    """Zero-materialization streaming LogSumExp pointer NLL for exact span grounding."""

    def __init__(
        self,
        d_dec: int,
        d_ptr: int = 256,
        chunk_size: int = 512,
        trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.d_ptr = d_ptr
        self.chunk_size = chunk_size
        self.scale = 1.0 / math.sqrt(float(self.d_ptr))
        self.w_ptr_q = torch.nn.Linear(d_dec, self.d_ptr, bias=False)
        self.w_ptr_k = torch.nn.Linear(d_dec, self.d_ptr, bias=False)
        self.trivial_token_ids = trivial_token_ids
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        h_dec: torch.Tensor,
        z_enc: torch.Tensor,
        enc_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        enc_valid_mask: torch.Tensor,
        trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_enc.shape[1] == 0 or not bool(enc_valid_mask.any()):
            return h_dec.sum() * 0.0
        q_ptr = self.w_ptr_q(h_dec.float())
        k_ptr = self.w_ptr_k(z_enc.float())
        effective_trivial = (
            trivial_token_ids if trivial_token_ids is not None else self.trivial_token_ids
        )
        return _streaming_lse_pointer_nll(
            q_ptr,
            k_ptr,
            enc_token_ids,
            target_token_ids,
            enc_valid_mask,
            scale=self.scale,
            chunk_size=self.chunk_size,
            trivial_token_ids=effective_trivial,
            use_checkpoint=self.use_checkpoint,
        )


def _pointer_chunk_forward(
    q_chunk: torch.Tensor,
    k_ptr: torch.Tensor,
    t_chunk: torch.Tensor,
    enc_token_ids: torch.Tensor,
    enc_valid_mask: torch.Tensor,
    trivial_tensor: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pointer loss sum and valid count for one decoder chunk."""
    valid_t = t_chunk != -100
    if trivial_tensor is not None and trivial_tensor.numel() > 0:
        valid_t = valid_t & ~torch.isin(t_chunk, trivial_tensor)

    scores = torch.bmm(q_chunk, k_ptr.transpose(1, 2)) * scale
    scores = scores.masked_fill(~enc_valid_mask[:, None, :].bool(), float("-inf"))
    log_denom = torch.logsumexp(scores, dim=-1)

    match_mask = (
        (t_chunk[:, :, None] == enc_token_ids[:, None, :])
        & enc_valid_mask[:, None, :].bool()
        & valid_t[:, :, None]
    )
    has_copy_target = match_mask.any(dim=-1)
    if not bool(has_copy_target.any()):
        return q_chunk.sum() * 0.0, torch.tensor(0, dtype=torch.int64, device=q_chunk.device)

    pos_scores = scores.masked_fill(~match_mask, float("-inf"))
    log_num = torch.logsumexp(pos_scores, dim=-1)
    token_nll = -(log_num - log_denom)
    valid_nll = token_nll[has_copy_target]
    chunk_loss = valid_nll.sum()
    chunk_valid = torch.tensor(valid_nll.numel(), dtype=torch.int64, device=q_chunk.device)
    return chunk_loss, chunk_valid


def _streaming_lse_pointer_nll(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    enc_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    enc_valid_mask: torch.Tensor,
    *,
    scale: float,
    chunk_size: int = 256,
    trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Compute streaming LogSumExp pointer NLL with per-chunk checkpointing and trivial token filtering."""
    if k_ptr.shape[1] == 0 or not bool(enc_valid_mask.any()):
        return q_ptr.sum() * 0.0

    if trivial_token_ids is None:
        trivial_token_ids = DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS

    if trivial_token_ids is not None and len(trivial_token_ids) > 0:
        if isinstance(trivial_token_ids, torch.Tensor):
            trivial_tensor = trivial_token_ids.to(
                device=target_token_ids.device, dtype=target_token_ids.dtype
            )
        else:
            trivial_tensor = torch.tensor(
                tuple(trivial_token_ids),
                device=target_token_ids.device,
                dtype=target_token_ids.dtype,
            )
    else:
        trivial_tensor = None

    _, seq_len, _ = q_ptr.shape
    total_loss = q_ptr.new_zeros((), dtype=torch.float32)
    total_valid = 0

    do_checkpoint = (
        use_checkpoint and torch.is_grad_enabled() and (q_ptr.requires_grad or k_ptr.requires_grad)
    )

    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        q_chunk = q_ptr[:, start:end, :]
        t_chunk = target_token_ids[:, start:end]

        valid_t = t_chunk != -100
        if trivial_tensor is not None and trivial_tensor.numel() > 0:
            valid_t = valid_t & ~torch.isin(t_chunk, trivial_tensor)
        if not bool(valid_t.any()):
            continue

        match_mask = (
            (t_chunk[:, :, None] == enc_token_ids[:, None, :])
            & enc_valid_mask[:, None, :].bool()
            & valid_t[:, :, None]
        )
        if not bool(match_mask.any()):
            continue

        if do_checkpoint:
            chunk_loss, chunk_valid = torch.utils.checkpoint.checkpoint(
                _pointer_chunk_forward,
                q_chunk,
                k_ptr,
                t_chunk,
                enc_token_ids,
                enc_valid_mask,
                trivial_tensor,
                scale,
                use_reentrant=False,
            )
        else:
            chunk_loss, chunk_valid = _pointer_chunk_forward(
                q_chunk,
                k_ptr,
                t_chunk,
                enc_token_ids,
                enc_valid_mask,
                trivial_tensor,
                scale,
            )

        n = int(chunk_valid.item())
        if n > 0:
            total_loss = total_loss + chunk_loss
            total_valid += n

    if total_valid == 0:
        return q_ptr.sum() * 0.0
    return total_loss / float(total_valid)


def streaming_pointer_auxiliary_loss(
    h_dec: torch.Tensor,
    z_enc: torch.Tensor,
    enc_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    enc_valid_mask: torch.Tensor,
    *,
    chunk_size: int = 256,
    trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Parameter-free zero-materialization streaming pointer NLL aligning decoder hidden states to bridge states."""
    if z_enc is None or z_enc.shape[1] == 0 or not bool(enc_valid_mask.any()):
        return h_dec.sum() * 0.0
    d_model = max(1, h_dec.shape[-1])
    q_norm = F.normalize(h_dec.float(), dim=-1) * math.sqrt(float(d_model))
    k_norm = F.normalize(z_enc.float(), dim=-1) * math.sqrt(float(d_model))
    scale = 1.0 / math.sqrt(float(d_model))
    return _streaming_lse_pointer_nll(
        q_norm,
        k_norm,
        enc_token_ids,
        target_token_ids,
        enc_valid_mask,
        scale=scale,
        chunk_size=chunk_size,
        trivial_token_ids=trivial_token_ids,
        use_checkpoint=use_checkpoint,
    )


# ==============================================================================
# Verified Gemma-4 Differential Diff Token Identifiers
# ==============================================================================
DEFAULT_DIFF_ADD_TOKEN_IDS: tuple[int, ...] = (236862, 900)
DEFAULT_DIFF_DEL_TOKEN_IDS: tuple[int, ...] = (236772, 753)
DEFAULT_DIFF_HDR_TOKEN_IDS: tuple[int, ...] = (16417, 7243, 24523, 10379, 15338, 2617, 13705)


def resolve_diff_token_ids(tokenizer: Any = None) -> dict[str, tuple[int, ...]]:
    """Resolve diff marker token IDs from a decoder tokenizer with fallback to Gemma-4 defaults."""
    if tokenizer is None:
        return {
            "add_token_ids": DEFAULT_DIFF_ADD_TOKEN_IDS,
            "del_token_ids": DEFAULT_DIFF_DEL_TOKEN_IDS,
            "hdr_token_ids": DEFAULT_DIFF_HDR_TOKEN_IDS,
        }

    def _enc(s: str) -> int | None:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    add_ids = tuple(filter(None, [_enc("+"), _enc(" +")]))
    del_ids = tuple(filter(None, [_enc("-"), _enc(" -")]))
    hdr_candidates = ["diff", "---", "+++", "@@", " @@", " --", "git"]
    hdr_ids = tuple(filter(None, [_enc(c) for c in hdr_candidates]))

    return {
        "add_token_ids": add_ids or DEFAULT_DIFF_ADD_TOKEN_IDS,
        "del_token_ids": del_ids or DEFAULT_DIFF_DEL_TOKEN_IDS,
        "hdr_token_ids": hdr_ids or DEFAULT_DIFF_HDR_TOKEN_IDS,
    }


def compute_diff_token_weights(
    labels: torch.Tensor,
    *,
    add_weight: float = 1.4,
    del_weight: float = 1.2,
    hdr_weight: float = 1.1,
    ctx_weight: float = 0.8,
    add_token_ids: tuple[int, ...] = DEFAULT_DIFF_ADD_TOKEN_IDS,
    del_token_ids: tuple[int, ...] = DEFAULT_DIFF_DEL_TOKEN_IDS,
    hdr_token_ids: tuple[int, ...] = DEFAULT_DIFF_HDR_TOKEN_IDS,
) -> torch.Tensor:
    """Compute normalized differential token weights so mean weight over valid labels is 1.0."""
    valid = labels != -100
    weights = torch.zeros_like(labels, dtype=torch.float32)
    if not bool(valid.any()):
        return weights
    weights[valid] = float(ctx_weight)
    if hdr_token_ids:
        hdr_mask = valid & torch.isin(
            labels, torch.tensor(hdr_token_ids, device=labels.device, dtype=labels.dtype)
        )
        weights[hdr_mask] = float(hdr_weight)
    if del_token_ids:
        del_mask = valid & torch.isin(
            labels, torch.tensor(del_token_ids, device=labels.device, dtype=labels.dtype)
        )
        weights[del_mask] = float(del_weight)
    if add_token_ids:
        add_mask = valid & torch.isin(
            labels, torch.tensor(add_token_ids, device=labels.device, dtype=labels.dtype)
        )
        weights[add_mask] = float(add_weight)
    if weights.ndim == 2:
        for b in range(weights.shape[0]):
            row_valid = valid[b]
            if bool(row_valid.any()):
                row_mean = weights[b, row_valid].mean().clamp_min(1e-6)
                weights[b, row_valid] = weights[b, row_valid] / row_mean
    else:
        mean_w = weights[valid].mean().clamp_min(1e-6)
        weights[valid] = weights[valid] / mean_w
    return weights


def differential_diff_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute normalized weighted cross-entropy loss across aligned target tokens [B, L]."""
    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("Expected aligned logits [B, L, V] and labels [B, L].")
    valid = labels != -100
    if not bool(valid.any()):
        return logits.sum() * 0.0
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    if token_weights is None:
        weights = torch.ones_like(per_token, dtype=torch.float32)
    else:
        weights = token_weights.float().clone()
        mean_w = weights[valid].mean().clamp_min(1e-6)
        weights[valid] = weights[valid] / mean_w
    return (per_token * weights)[valid].sum() / valid.float().sum().clamp_min(1.0)


def chunked_differential_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    *,
    chunk_size: int = 128,
    softcap: float | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Memory-bounded chunked LM-head cross-entropy with normalized differential token weights."""
    from torch.utils.checkpoint import checkpoint

    if chunk_size < 1 or hidden.shape[:2] != labels.shape:
        raise ValueError("Invalid loss chunk size or aligned label shape.")
    valid = labels.reshape(-1) != ignore_index
    if not bool(valid.any()):
        return hidden.sum() * 0.0 + weight.reshape(-1)[0] * 0.0
    states = hidden.reshape(-1, hidden.shape[-1])[valid]
    targets = labels.reshape(-1)[valid]
    if token_weights is None:
        flat_weights = torch.ones_like(targets, dtype=torch.float32)
    else:
        flat_weights = token_weights.reshape(-1)[valid].float()
        mean_w = flat_weights.mean().clamp_min(1e-6)
        flat_weights = flat_weights / mean_w

    def compute(
        values: torch.Tensor, words: torch.Tensor, target: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        logits = F.linear(values, words)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        ce = F.cross_entropy(logits.float(), target, reduction="none")
        return (ce * w).sum()

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        stop = start + chunk_size
        args = (states[start:stop], weight, targets[start:stop], flat_weights[start:stop])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            value = checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            value = compute(*args)
        total = total + value
    return total / float(targets.numel())


def compute_v3_step_loss(
    model: Any,
    batch: Any,
    *,
    stage_name: str = "frozen_decoder",
    lambda_ptr: float | None = None,
    diff_token_ids: dict[str, tuple[int, ...]] | None = None,
    trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute BifocalGemma V3 training loss combining chunked differential CE and streaming pointer loss."""
    if not hasattr(model, "_decode_hidden") or not hasattr(model, "lm_head"):
        output = model.forward_loss(batch) if hasattr(model, "forward_loss") else model(batch)
        return output.loss

    has_encoder = getattr(model, "encoder", None) is not None
    if has_encoder:
        encoder_states = model.encode_context(batch.encoder_input_ids, batch.encoder_attention_mask)
        encoder_mask = batch.encoder_attention_mask
    else:
        encoder_states, encoder_mask = None, None

    decoder_hidden = model._decode_hidden(
        batch.decoder_input_ids,
        batch.decoder_attention_mask,
        encoder_states,
        encoder_mask,
    )
    diff_ids = getattr(model, "diff_token_ids", None) or diff_token_ids or {}
    token_weights = compute_diff_token_weights(
        batch.labels,
        add_token_ids=diff_ids.get("add_token_ids", DEFAULT_DIFF_ADD_TOKEN_IDS),
        del_token_ids=diff_ids.get("del_token_ids", DEFAULT_DIFF_DEL_TOKEN_IDS),
        hdr_token_ids=diff_ids.get("hdr_token_ids", DEFAULT_DIFF_HDR_TOKEN_IDS),
    )
    chunk_sz = getattr(model, "loss_chunk_size", None) or 128
    softcap = getattr(getattr(model, "decoder", None), "config", None)
    softcap_val = getattr(softcap, "final_logit_softcapping", None) if softcap is not None else None
    ce_loss = chunked_differential_cross_entropy(
        decoder_hidden,
        model.lm_head.weight,
        batch.labels,
        token_weights=token_weights,
        chunk_size=int(chunk_sz),
        softcap=softcap_val,
    )
    if not has_encoder or encoder_states is None or encoder_mask is None:
        return ce_loss

    DEFAULT_LAMBDA_PTR = 0.0
    ptr_weight = float(lambda_ptr) if lambda_ptr is not None else DEFAULT_LAMBDA_PTR
    if ptr_weight <= 0.0:
        return ce_loss
    trivial_ids = getattr(model, "trivial_token_ids", None) or trivial_token_ids
    ptr_loss = streaming_pointer_auxiliary_loss(
        decoder_hidden,
        encoder_states,
        batch.encoder_input_ids,
        batch.labels,
        encoder_mask,
        trivial_token_ids=trivial_ids,
    )
    return ce_loss + ptr_weight * ptr_loss


compute_bifocalgemma_v3_loss = compute_v3_step_loss


def is_cross_bridge_parameter(name: str) -> bool:
    """Return True if parameter belongs to encoder_projection or the full-rank cross-attention bridge."""
    return name.startswith("encoder_projection.") or any(
        tok in name
        for tok in (
            ".cross_k_proj.",
            ".cross_v_proj.",
            ".cross_k_norm.",
            ".cross_v_norm.",
            ".cross_q_norm.",
            ".cross_gate",
            ".cross_scale",
        )
    )


def is_zero_weight_decay_bridge_parameter(name: str) -> bool:
    """Return True if an alignment parameter must have strict 0.0 weight decay (RMSNorm, gates, cross-attn)."""
    if name.startswith("encoder_projection."):
        return any(tok in name for tok in ("pre_norm", "post_norm", "gamma_calib", "norm"))
    return True


def build_bifocalgemma_v3_optimizer(
    model: Any,
    *,
    lr_bridge: float = 1e-4,
    lr_cross: float = 1e-4,
    lr_lora: float = 2e-5,
    weight_decay_bridge: float = 1e-4,
) -> torch.optim.AdamW:
    """Build partitioned AdamW optimizer enforcing strict 0.0 weight decay on RMSNorm, cross-attn, and LoRA."""
    bridge_decay = []
    bridge_decay_names = []
    bridge_nodecay = []
    bridge_nodecay_names = []
    cross_attn = []
    cross_attn_names = []
    backbone_lora = []
    backbone_lora_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder_projection."):
            if is_zero_weight_decay_bridge_parameter(name) or param.ndim < 2:
                bridge_nodecay.append(param)
                bridge_nodecay_names.append(name)
            else:
                bridge_decay.append(param)
                bridge_decay_names.append(name)
        elif is_cross_bridge_parameter(name):
            cross_attn.append(param)
            cross_attn_names.append(name)
        else:
            backbone_lora.append(param)
            backbone_lora_names.append(name)

    param_groups = [
        {
            "name": "bridge_linear_decay",
            "params": bridge_decay,
            "param_names": bridge_decay_names,
            "lr": lr_bridge,
            "base_lr": lr_bridge,
            "weight_decay": weight_decay_bridge,
        },
        {
            "name": "bridge_norm_and_calib_nodecay",
            "params": bridge_nodecay,
            "param_names": bridge_nodecay_names,
            "lr": lr_bridge,
            "base_lr": lr_bridge,
            "weight_decay": 0.0,
        },
        {
            "name": "decoupled_cross_attn_nodecay",
            "params": cross_attn,
            "param_names": cross_attn_names,
            "lr": lr_cross,
            "base_lr": lr_cross,
            "weight_decay": 0.0,
        },
        {
            "name": "backbone_lora",
            "params": backbone_lora,
            "param_names": backbone_lora_names,
            "lr": lr_lora,
            "base_lr": lr_lora,
            "weight_decay": 0.0,
        },
    ]
    active_groups = [g for g in param_groups if g["params"]]
    return torch.optim.AdamW(
        active_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
