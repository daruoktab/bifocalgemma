"""E01: actual repair grading interface. See docs/EVALUATION.md."""

from pathlib import Path

from bifocalgemma.contracts import EvaluationReport
from bifocalgemma.scaffold import FeatureNotReadyError


class SWEBenchHarness:
    def evaluate(
        self, predictions_path: Path, dataset_revision: str, run_id: str
    ) -> EvaluationReport:
        """Future Docker runner: FAIL_TO_PASS, PASS_TO_PASS and infrastructure errors.

        Dataset/run manifests must record selected tasks and benchmark denominator.
        A serializer or printed comparison table is not a successful evaluation.
        """
        raise FeatureNotReadyError(
            "E01", "Docker harness execution and result normalization are not wired."
        )
