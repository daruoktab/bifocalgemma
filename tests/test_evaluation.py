"""Unit tests for evaluation reporting utilities."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bifocalgemma.evaluation.eval_code import (  # noqa: E402
    EXPERIMENT_MODELS,
    REFERENCE_BASELINES,
    BaselineScores,
    _format_score,
    print_comparison_table,
    print_repair_comparison_table,
    print_repair_experiment_table,
)


class TestEvaluationReporting(unittest.TestCase):
    def test_format_score(self):
        self.assertEqual(_format_score(None), "N/A")
        self.assertEqual(_format_score(44.0), "44.0")
        self.assertEqual(_format_score(52.123), "52.1")

    def test_reference_baselines_contents(self):
        self.assertEqual(set(REFERENCE_BASELINES), {"Gemma-4-E2B-IT", "Gemma-4-E4B-IT"})
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

    def test_six_model_repair_table_includes_every_variant_and_baseline(self):
        expected = {
            "BifocalGemma-E2B-E2B",
            "BifocalGemma-E4B-E4B",
            "BifocalGemma-E2B-E4B",
            "BifocalGemma-E4B-E2B",
            "Gemma-4-E2B-IT",
            "Gemma-4-E4B-IT",
        }
        self.assertEqual(set(EXPERIMENT_MODELS), expected)
        output = io.StringIO()
        with redirect_stdout(output):
            print_repair_experiment_table(
                {"BifocalGemma-E2B-E2B": {"swe_bench_lite": 0.0}}, illustrative=True
            )
        table = output.getvalue()
        rows = [
            line for line in table.splitlines() if any(line.startswith(name) for name in expected)
        ]
        self.assertEqual(len(rows), 6)
        for name in expected:
            self.assertEqual(sum(line.startswith(name) for line in rows), 1)
        self.assertIn("ILLUSTRATIVE SCORES (NOT MEASURED)", table)
        self.assertIn("0.0", table)
        self.assertIn("N/A", table)
        self.assertNotIn("T5Gemma", table)
        self.assertNotIn("Gemma-3", table)

    def test_six_model_repair_table_rejects_other_model_families(self):
        with self.assertRaises(ValueError):
            print_repair_experiment_table({"T5Gemma-2-4B-4B": {"swe_bench_lite": 10.0}})


if __name__ == "__main__":
    unittest.main()
