"""Aligned cross-entropy, differential diff weighting, streaming pointer loss, and optimizer groups."""

from __future__ import annotations

import math
import string
from collections.abc import Sequence
from typing import Any

import torch
import torch.utils.checkpoint
from torch.nn import functional as F

import bifocalgemma.config as _config


def patch_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute cross-entropy over already-aligned logits [B,L,V] and labels [B,L]."""
    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("Expected logits [B,L,V] and already aligned labels [B,L].")
    if not bool((labels != ignore_index).any()):
        return logits.sum() * 0
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index
    )


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 128,
    softcap: float | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Loss-only head: identical softcap/dtype/shift and token weighting, bounded logits.

    Each chunk is recomputed in backward. Chunking an already materialized logits
    tensor would not save LM-head memory. Prefix hidden states remain in the model.
    """
    from torch.utils.checkpoint import checkpoint

    if chunk_size < 1 or hidden.shape[:2] != labels.shape:
        raise ValueError("Invalid loss chunk size or aligned label shape.")
    valid = labels.reshape(-1) != ignore_index
    if not bool(valid.any()):
        return hidden.sum() * 0 + weight.reshape(-1)[0] * 0
    states = hidden.reshape(-1, hidden.shape[-1])[valid]
    targets = labels.reshape(-1)[valid]

    def compute(values: torch.Tensor, words: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = F.linear(values, words)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(logits.float(), target, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        args = (states[start : start + chunk_size], weight, targets[start : start + chunk_size])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            value = checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            value = compute(*args)
        total = total + value
    return total / targets.numel()


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
# Strict unified-diff hunk/file headers only: "---", "+++", "@@", " @@"
DEFAULT_DIFF_HUNK_HDR_TOKEN_IDS: tuple[int, ...] = (7243, 24523, 10379, 15338)

VALID_LOSS_REDUCTIONS: tuple[str, ...] = ("per_example_mean", "length_capped", "token_mean")


def resolve_diff_token_ids(tokenizer: Any = None) -> dict[str, tuple[int, ...]]:
    """Resolve diff marker token IDs from a decoder tokenizer with fallback to Gemma-4 defaults."""
    if tokenizer is None:
        return {
            "add_token_ids": DEFAULT_DIFF_ADD_TOKEN_IDS,
            "del_token_ids": DEFAULT_DIFF_DEL_TOKEN_IDS,
            "hdr_token_ids": DEFAULT_DIFF_HDR_TOKEN_IDS,
            "hunk_hdr_token_ids": DEFAULT_DIFF_HUNK_HDR_TOKEN_IDS,
        }

    def _enc(s: str) -> int | None:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    add_ids = tuple(x for x in [_enc("+"), _enc(" +")] if x is not None)
    del_ids = tuple(x for x in [_enc("-"), _enc(" -")] if x is not None)
    hdr_candidates = ["diff", "---", "+++", "@@", " @@", " --", "git"]
    hdr_ids = tuple(x for x in (_enc(c) for c in hdr_candidates) if x is not None)
    hunk_hdr_candidates = ["---", "+++", "@@", " @@"]
    hunk_hdr_ids = tuple(x for x in (_enc(c) for c in hunk_hdr_candidates) if x is not None)

    return {
        "add_token_ids": add_ids or DEFAULT_DIFF_ADD_TOKEN_IDS,
        "del_token_ids": del_ids or DEFAULT_DIFF_DEL_TOKEN_IDS,
        "hdr_token_ids": hdr_ids or DEFAULT_DIFF_HDR_TOKEN_IDS,
        "hunk_hdr_token_ids": hunk_hdr_ids or DEFAULT_DIFF_HUNK_HDR_TOKEN_IDS,
    }


def is_unified_diff_target_tokens(
    row_labels: torch.Tensor,
    *,
    add_token_ids: tuple[int, ...] = DEFAULT_DIFF_ADD_TOKEN_IDS,
    del_token_ids: tuple[int, ...] = DEFAULT_DIFF_DEL_TOKEN_IDS,
    hdr_token_ids: tuple[int, ...] = DEFAULT_DIFF_HDR_TOKEN_IDS,
    hunk_hdr_token_ids: tuple[int, ...] | None = None,
    ignore_index: int = -100,
) -> bool:
    """Return True only when valid target tokens exhibit unified-diff markers (hunk/file header + add/del)."""
    valid_tokens = row_labels[row_labels != ignore_index]
    if valid_tokens.numel() == 0:
        return False
    strict_hdr_ids = (
        hunk_hdr_token_ids
        if hunk_hdr_token_ids is not None
        else (
            DEFAULT_DIFF_HUNK_HDR_TOKEN_IDS
            if tuple(hdr_token_ids) == DEFAULT_DIFF_HDR_TOKEN_IDS
            else hdr_token_ids
        )
    )
    if not strict_hdr_ids:
        return False
    has_hdr = bool(
        torch.isin(
            valid_tokens,
            torch.tensor(strict_hdr_ids, device=row_labels.device, dtype=row_labels.dtype),
        ).any()
    )
    if not has_hdr:
        return False
    edit_ids = tuple(add_token_ids) + tuple(del_token_ids)
    if not edit_ids:
        return False
    return bool(
        torch.isin(
            valid_tokens,
            torch.tensor(edit_ids, device=row_labels.device, dtype=row_labels.dtype),
        ).any()
    )


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
    hunk_hdr_token_ids: tuple[int, ...] | None = None,
    task_types: Sequence[str | None] | None = None,
    is_unified_diff: Sequence[bool | None] | None = None,
    require_unified_diff: bool = True,
) -> torch.Tensor:
    """Compute per-row normalized differential token weights (mean=1.0 over valid labels).

    Diff weighting (`1.4x/1.2x/1.1x/0.8x`) is gated strictly to `task_type == "repair"`
    with unified-diff targets; all other tasks (`completion`, `localization`, `qa`, `mcq`,
    `math`, full-function `repair`) receive exact uniform `1.0` weights over valid labels.
    """
    valid = labels != -100
    weights = torch.zeros_like(labels, dtype=torch.float32)
    if not bool(valid.any()):
        return weights

    def _compute_row_weights(
        row_labels: torch.Tensor,
        row_valid: torch.Tensor,
        row_task: str | None,
        row_is_diff: bool | None,
    ) -> torch.Tensor:
        out = torch.zeros_like(row_labels, dtype=torch.float32)
        if not bool(row_valid.any()):
            return out
        apply_diff = True
        if row_task is not None and str(row_task).strip().lower() != "repair":
            apply_diff = False
        if apply_diff:
            if row_is_diff is not None:
                apply_diff = bool(row_is_diff)
            elif require_unified_diff:
                apply_diff = is_unified_diff_target_tokens(
                    row_labels,
                    add_token_ids=add_token_ids,
                    del_token_ids=del_token_ids,
                    hdr_token_ids=hdr_token_ids,
                    hunk_hdr_token_ids=hunk_hdr_token_ids,
                )
        if not apply_diff:
            out[row_valid] = 1.0
            return out

        out[row_valid] = float(ctx_weight)
        if hdr_token_ids:
            hdr_mask = row_valid & torch.isin(
                row_labels,
                torch.tensor(hdr_token_ids, device=row_labels.device, dtype=row_labels.dtype),
            )
            out[hdr_mask] = float(hdr_weight)
        if del_token_ids:
            del_mask = row_valid & torch.isin(
                row_labels,
                torch.tensor(del_token_ids, device=row_labels.device, dtype=row_labels.dtype),
            )
            out[del_mask] = float(del_weight)
        if add_token_ids:
            add_mask = row_valid & torch.isin(
                row_labels,
                torch.tensor(add_token_ids, device=row_labels.device, dtype=row_labels.dtype),
            )
            out[add_mask] = float(add_weight)
        row_mean = out[row_valid].mean().clamp_min(1e-6)
        out[row_valid] = out[row_valid] / row_mean
        return out

    if labels.ndim == 2:
        batch_size = int(labels.shape[0])
        if task_types is not None and len(task_types) != batch_size:
            raise ValueError(f"Expected {batch_size} task_types, got {len(task_types)}.")
        if is_unified_diff is not None and len(is_unified_diff) != batch_size:
            raise ValueError(
                f"Expected {batch_size} is_unified_diff flags, got {len(is_unified_diff)}."
            )
        for b in range(batch_size):
            row_task = task_types[b] if task_types is not None else None
            row_is_diff = is_unified_diff[b] if is_unified_diff is not None else None
            weights[b] = _compute_row_weights(labels[b], valid[b], row_task, row_is_diff)
    else:
        if task_types is not None and len(task_types) != 1:
            raise ValueError(f"Expected 1 task_type for 1D labels, got {len(task_types)}.")
        if is_unified_diff is not None and len(is_unified_diff) != 1:
            raise ValueError(
                f"Expected 1 is_unified_diff flag for 1D labels, got {len(is_unified_diff)}."
            )
        row_task = task_types[0] if task_types is not None else None
        row_is_diff = is_unified_diff[0] if is_unified_diff is not None else None
        weights = _compute_row_weights(labels, valid, row_task, row_is_diff)
    return weights


def compute_example_loss_weights(
    labels: torch.Tensor,
    *,
    reduction: str = "per_example_mean",
    length_cap: int = 256,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute per-example loss weights [B] preventing long-target token-count starvation.

    Supported reductions:
    - `per_example_mean`: every valid example receives equal weight `1.0`.
    - `length_capped`: example weight is `min(N_tgt, length_cap)` (default cap `256`).
    - `token_mean`: legacy raw token-count weight `N_tgt`.
    """
    if reduction not in VALID_LOSS_REDUCTIONS:
        raise ValueError(
            f"Unknown loss_reduction {reduction!r}; expected one of {VALID_LOSS_REDUCTIONS}."
        )
    if type(length_cap) is not int or length_cap < 1:
        raise ValueError("length_cap must be a positive integer.")
    if labels.ndim == 1:
        counts = (labels != ignore_index).sum().to(dtype=torch.float32).unsqueeze(0)
    elif labels.ndim == 2:
        counts = (labels != ignore_index).sum(dim=-1).to(dtype=torch.float32)
    else:
        raise ValueError(f"Expected 1D or 2D labels tensor, got shape {tuple(labels.shape)}.")
    has_tokens = counts > 0
    if reduction == "per_example_mean":
        return torch.where(has_tokens, torch.ones_like(counts), torch.zeros_like(counts))
    if reduction == "length_capped":
        return torch.where(
            has_tokens, counts.clamp(max=float(length_cap)), torch.zeros_like(counts)
        )
    return counts


def _normalized_token_coefficients(
    labels: torch.Tensor,
    token_weights: torch.Tensor | None,
    *,
    loss_reduction: str,
    length_cap: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute normalized per-token coefficients [B, L] summing to 1.0 across valid tokens.

    Guarantees exact mathematical micro-batch invariance across arbitrary `(batch_size, accumulation)`
    splits when combined with micro-batch weight scaling `W(micro) / W(effective_batch)`.
    """
    valid = labels != ignore_index
    coeffs = torch.zeros_like(labels, dtype=torch.float32)
    if not bool(valid.any()):
        return coeffs
    example_weights = compute_example_loss_weights(
        labels, reduction=loss_reduction, length_cap=length_cap, ignore_index=ignore_index
    )
    total_example_weight = example_weights.sum().clamp_min(1e-12)
    batch_size = int(labels.shape[0])
    for b in range(batch_size):
        row_valid = valid[b]
        n_valid = int(row_valid.sum().item())
        if n_valid == 0:
            continue
        if token_weights is None:
            row_w = torch.ones((n_valid,), device=labels.device, dtype=torch.float32)
        else:
            row_w = token_weights[b, row_valid].float()
            row_mean = row_w.mean().clamp_min(1e-6)
            row_w = row_w / row_mean
        row_share = example_weights[b] / total_example_weight
        coeffs[b, row_valid] = row_w * (row_share / float(n_valid))
    return coeffs


def differential_diff_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    *,
    loss_reduction: str = "per_example_mean",
    length_cap: int = 256,
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
    coeffs = _normalized_token_coefficients(
        labels,
        token_weights,
        loss_reduction=loss_reduction,
        length_cap=length_cap,
        ignore_index=-100,
    )
    return (per_token * coeffs)[valid].sum()


def chunked_differential_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    *,
    chunk_size: int = 128,
    softcap: float | None = None,
    ignore_index: int = -100,
    loss_reduction: str = "per_example_mean",
    length_cap: int = 256,
) -> torch.Tensor:
    """Memory-bounded chunked LM-head cross-entropy with per-example/length-capped reduction."""
    from torch.utils.checkpoint import checkpoint

    if chunk_size < 1 or hidden.shape[:2] != labels.shape:
        raise ValueError("Invalid loss chunk size or aligned label shape.")
    valid = labels.reshape(-1) != ignore_index
    if not bool(valid.any()):
        return hidden.sum() * 0.0 + weight.reshape(-1)[0] * 0.0
    states = hidden.reshape(-1, hidden.shape[-1])[valid]
    targets = labels.reshape(-1)[valid]
    coeffs = _normalized_token_coefficients(
        labels,
        token_weights,
        loss_reduction=loss_reduction,
        length_cap=length_cap,
        ignore_index=ignore_index,
    )
    flat_coeffs = coeffs.reshape(-1)[valid].float()

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
        args = (states[start:stop], weight, targets[start:stop], flat_coeffs[start:stop])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            value = checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            value = compute(*args)
        total = total + value
    return total


def compute_step_loss(
    model: Any,
    batch: Any,
    *,
    stage_name: str = "frozen_decoder",
    lambda_ptr: float | None = None,
    diff_token_ids: dict[str, tuple[int, ...]] | None = None,
    trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
    token_weighting: str = "diff_marker_v1",
    task_types: Sequence[str | None] | None = None,
    is_unified_diff: Sequence[bool | None] | None = None,
    loss_reduction: str | None = None,
    length_cap: int | None = None,
) -> torch.Tensor:
    """Compute BifocalGemma training loss combining task-gated diff CE and optional pointer loss."""
    if token_weighting not in {"uniform_target_v1", "diff_marker_v1"}:
        raise ValueError(f"Unknown token weighting: {token_weighting}")
    resolved_reduction = (
        loss_reduction
        if loss_reduction is not None
        else str(getattr(_config, "LOSS_REDUCTION", "per_example_mean"))
    )
    resolved_cap = (
        length_cap
        if length_cap is not None
        else int(getattr(_config, "LOSS_LENGTH_CAP", 256))
    )
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
    effective_task_types = (
        task_types if task_types is not None else getattr(batch, "task_types", None)
    )
    effective_is_diff = (
        is_unified_diff
        if is_unified_diff is not None
        else getattr(batch, "is_unified_diff", None)
    )
    token_weights = (
        None
        if token_weighting == "uniform_target_v1"
        else compute_diff_token_weights(
            batch.labels,
            add_token_ids=diff_ids.get("add_token_ids", DEFAULT_DIFF_ADD_TOKEN_IDS),
            del_token_ids=diff_ids.get("del_token_ids", DEFAULT_DIFF_DEL_TOKEN_IDS),
            hdr_token_ids=diff_ids.get("hdr_token_ids", DEFAULT_DIFF_HDR_TOKEN_IDS),
            hunk_hdr_token_ids=diff_ids.get("hunk_hdr_token_ids"),
            task_types=effective_task_types,
            is_unified_diff=effective_is_diff,
            require_unified_diff=True,
        )
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
        loss_reduction=resolved_reduction,
        length_cap=resolved_cap,
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


compute_v3_step_loss = compute_step_loss
compute_bifocalgemma_loss = compute_step_loss
compute_bifocalgemma_v3_loss = compute_step_loss


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
            ".cross_q_delta",
            ".cross_gate",
            ".cross_scale",
        )
    )


def is_zero_weight_decay_bridge_parameter(name: str) -> bool:
    """Return True if an alignment parameter must have strict 0.0 weight decay (RMSNorm, gates, cross-attn)."""
    if name.startswith("encoder_projection."):
        return any(tok in name for tok in ("pre_norm", "post_norm", "gamma_calib", "norm"))
    return True


def build_bifocalgemma_optimizer(
    model: Any,
    *,
    policy: dict[str, Any] | None = None,
    lr_cross_gate_scalars: float | None = None,
    lr_cross_gate_proj: float | None = None,
    lr_bridge: float | None = None,
    lr_cross: float | None = None,
    lr_encoder_lora: float | None = None,
    lr_decoder_lora: float | None = None,
    lr_lora: float | None = None,
    weight_decay_bridge: float | None = None,
    betas: tuple[float, float] | None = None,
    eps: float | None = None,
    total_steps: int | None = None,
    warmup_offset: int = 0,
    freeze_decoder: bool = False,
) -> torch.optim.AdamW:
    """Build canonical 6-group partitioned AdamW optimizer via `stage.build_partitioned_param_groups`.

    Delegates directly to the single canonical partitioned param-group builder shared with
    `stage.py` and `matrix.py` so `fast_pipeline.py` and `matrix.py` cannot diverge.
    """
    from bifocalgemma.training.stage import (
        build_bifocalgemma_adamw,
        build_partitioned_param_groups,
    )

    overrides: dict[str, dict[str, Any]] = {}
    if lr_cross_gate_scalars is not None:
        overrides.setdefault("cross_gate_scalars", {})["lr"] = float(lr_cross_gate_scalars)
    if lr_cross_gate_proj is not None:
        overrides.setdefault("cross_gate_proj", {})["lr"] = float(lr_cross_gate_proj)
    if lr_bridge is not None:
        overrides.setdefault("bridge_mlp", {})["lr"] = float(lr_bridge)
    if weight_decay_bridge is not None:
        overrides.setdefault("bridge_mlp", {})["weight_decay"] = float(weight_decay_bridge)
    if lr_cross is not None:
        overrides.setdefault("cross_kv_proj", {})["lr"] = float(lr_cross)
    enc_lora_lr = lr_encoder_lora if lr_encoder_lora is not None else lr_lora
    if enc_lora_lr is not None:
        overrides.setdefault("encoder_lora", {})["lr"] = float(enc_lora_lr)
    dec_lora_lr = lr_decoder_lora if lr_decoder_lora is not None else lr_lora
    if dec_lora_lr is not None:
        overrides.setdefault("decoder_lora", {})["lr"] = float(dec_lora_lr)

    groups, _ = build_partitioned_param_groups(
        model,
        policy=policy,
        total_steps=total_steps,
        warmup_offset=warmup_offset,
        freeze_decoder=freeze_decoder,
        group_overrides=overrides or None,
    )
    return build_bifocalgemma_adamw(groups, policy=policy, betas=betas, eps=eps)


build_bifocalgemma_v3_optimizer = build_bifocalgemma_optimizer
