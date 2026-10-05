"""A01: Gemma-specific tool parsing and recorded action loops."""

from bifocalgemma.contracts import RepairExample, RepairPrediction
from bifocalgemma.scaffold import FeatureNotReadyError


class DeveloperAgent:
    def run(self, example: RepairExample, max_steps: int) -> RepairPrediction:
        """Future: read/search/edit/test actions, fixed budgets and context refresh.

        Tool grammar, timeouts and workspace-scoped execution must be implemented.
        All six comparisons use the same allowed tools and context selection.
        """
        raise FeatureNotReadyError("A01", "Connect generation, tool parser, execution and traces.")
