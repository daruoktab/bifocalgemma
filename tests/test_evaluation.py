"""Unit tests for evaluation reporting utilities."""

import sys
import unittest
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bifocalgemma.evaluation.eval_code import (  # noqa: E402
    REFERENCE_BASELINES,
    BaselineScores,
    _format_score,
    print_comparison_table,
    print_repair_comparison_table,
)


class TestEvaluationReporting(unittest.TestCase):
    def test_format_score(self):
        self.assertEqual(_format_score(None), "N/A")
        self.assertEqual(_format_score(44.0), "44.0")
        self.assertEqual(_format_score(52.123), "52.1")

    def test_reference_baselines_contents(self):
        self.assertIn("Gemma-4-E2B-IT", REFERENCE_BASELINES)
        self.assertIn("Gemma-4-E4B-IT", REFERENCE_BASELINES)
        e2b = REFERENCE_BASELINES["Gemma-4-E2B-IT"]
        self.assertIsInstance(e2b, BaselineScores)
        self.assertEqual(e2b.livecodebench_v6, 44.0)

    def test_print_comparison_table_runs_without_error(self):
        results = {"humaneval": 65.0, "mbpp": 70.0, "livecodebench_v6": 48.0}
        # Verify function runs cleanly with both real and illustrative flags
        print_comparison_table(results, illustrative=False)
        print_comparison_table(results, illustrative=True)

    def test_print_repair_comparison_table_runs_without_error(self):
        repair_results = {
            "swe_bench_lite": 25.0,
            "swe_bench_verified": 22.0,
            "crosscodeeval": 45.0,
        }
        print_repair_comparison_table(repair_results, illustrative=False)
        print_repair_comparison_table(repair_results, illustrative=True)


if __name__ == "__main__":
    unittest.main()
