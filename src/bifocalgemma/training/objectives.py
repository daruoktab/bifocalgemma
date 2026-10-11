"""T01: aligned patch-only CE. Collator shifts once; loss must not shift again."""

import torch
import torch.nn.functional as F


def patch_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("Expected logits [B,L,V] and already aligned labels [B,L].")
    if not bool((labels != ignore_index).any()):
        return logits.sum() * 0
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index
    )


def chunked_linear_cross_entropy(
    hidden, weight, labels, *, chunk_size=128, softcap=None, ignore_index=-100
):
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

    def compute(values, words, target):
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
