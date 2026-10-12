"""Repository data utilities; tensor collators load only when requested."""

__all__ = ["RepairCollator"]


def __getattr__(name):
    if name == "RepairCollator":
        from bifocalgemma.data.collator import RepairCollator

        return RepairCollator
    raise AttributeError(name)
