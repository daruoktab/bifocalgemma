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
    source = "SWE-bench/SWE-smith"


class CommitPackFTAdapter(DatasetAdapter):
    source = "bigcode/commitpackft"


class SWEGymAdapter(DatasetAdapter):
    source = "SWE-Gym/SWE-Gym"
