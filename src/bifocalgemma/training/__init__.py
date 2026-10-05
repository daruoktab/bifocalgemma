"""Direct-SFT interfaces; full training remains a documented integration block."""

from bifocalgemma.training.objectives import patch_cross_entropy

__all__ = ["patch_cross_entropy"]
