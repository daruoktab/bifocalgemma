"""T04: optional later experiment; not imported or required by direct SFT."""

import torch

from bifocalgemma.scaffold import FeatureNotReadyError


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Future: explicit alignment/masks/temperature and teacher compute accounting."""
    raise FeatureNotReadyError(
        "T04", "Define a justified distillation experiment before implementing teacher inference."
    )
