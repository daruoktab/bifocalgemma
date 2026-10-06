"""D04: adapters reconstruct pre-fix snapshots. See docs/DATA_PIPELINE.md."""

from collections.abc import Iterator
from pathlib import Path

from bifocalgemma.contracts import RepairExample
from bifocalgemma.scaffold import FeatureNotReadyError


class DatasetAdapter:
    source = "unassigned"

    def iter_examples(self, manifest: Path, split: str) -> Iterator[RepairExample]:
        """Future: pinned source -> canonical RepairExample, with overlap manifest.

        Do not treat commit messages as verified bugs or benchmark targets as
        training inputs. Actual fetching/reconstruction/filtering is not wired.
        """
        raise FeatureNotReadyError(
            "D04", f"Implement {self.source} schema mapping and snapshot reconstruction."
        )


class SWESmithAdapter(DatasetAdapter):
    # Local direction/hash validation is D06 in swesmith.py; fetching remains D04.
    source = "SWE-bench/SWE-smith"


class CommitPackFTAdapter(DatasetAdapter):
    source = "bigcode/commitpackft"


class SWEGymAdapter(DatasetAdapter):
    source = "SWE-Gym/SWE-Gym"


class EditPackFTAdapter(DatasetAdapter):
    """Offline old_contents/instruction -> before-to-after patch slot (D04).

    Do not use content/ndiff as inputs: they can carry the answer. No execution,
    reward or VM is part of this data adapter. Production mapping is still D04.
    """

    source = "nuprl/EditPackFT"
