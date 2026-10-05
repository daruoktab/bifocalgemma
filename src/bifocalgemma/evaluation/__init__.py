"""Reference score reporting for BifocalGemma; benchmark execution is not implemented."""

from bifocalgemma.evaluation.eval_code import (
    REFERENCE_BASELINES,
    print_comparison_table,
    print_repair_comparison_table,
)

__all__ = ["REFERENCE_BASELINES", "print_comparison_table", "print_repair_comparison_table"]
