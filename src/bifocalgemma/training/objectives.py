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
