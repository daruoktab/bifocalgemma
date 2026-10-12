"""Direct-SFT interfaces; tensor objectives load only when requested."""

__all__ = ["patch_cross_entropy"]


def __getattr__(name):
    if name == "patch_cross_entropy":
        from bifocalgemma.training.objectives import patch_cross_entropy

        return patch_cross_entropy
    raise AttributeError(name)
