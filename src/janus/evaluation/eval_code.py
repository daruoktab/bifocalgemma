"""Lightweight Code Evaluation Pipeline for JANUS-Edge.

Provides turnkey evaluation for:
1. HumanEval & HumanEval+ (via EvalPlus)
2. MBPP & MBPP+ (via EvalPlus)
3. Direct comparison against official Google Gemma 4 & T5Gemma 2 baselines.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BaselineScores:
    """Official baseline scores from Google Technical Reports."""

    model_name: str
    humaneval_pass1: float
    mbpp_pass1: float
    livecodebench_v6: Optional[float] = None


OFFICIAL_BASELINES: Dict[str, BaselineScores] = {
    # Gemma 4 Technical Report (Table 5) & Gemma 3 (Table 18)
    "Gemma-4-E2B-IT": BaselineScores(
        "Gemma-4-E2B-IT", humaneval_pass1=44.0, mbpp_pass1=38.0, livecodebench_v6=44.0
    ),
    "Gemma-4-E4B-IT": BaselineScores(
        "Gemma-4-E4B-IT", humaneval_pass1=54.5, mbpp_pass1=44.0, livecodebench_v6=52.0
    ),
    "Gemma-3-1B-IT": BaselineScores("Gemma-3-1B-IT", humaneval_pass1=41.5, mbpp_pass1=35.2),
    "Gemma-3-4B-IT": BaselineScores("Gemma-3-4B-IT", humaneval_pass1=71.3, mbpp_pass1=63.2),
    # T5Gemma 2 Technical Report (Table 5)
    "T5Gemma-2-1B-1B": BaselineScores("T5Gemma-2-1B-1B", humaneval_pass1=61.6, mbpp_pass1=54.8),
    "T5Gemma-2-4B-4B": BaselineScores("T5Gemma-2-4B-4B", humaneval_pass1=76.8, mbpp_pass1=66.8),
}


def print_comparison_table(
    our_results: Dict[str, float],
    model_name: str = "JANUS-E2B-E2B",
) -> None:
    """Formats and prints an ASCII comparison table against official Google baselines."""
    print("=" * 72)
    print(f"  BENCHMARK EVALUATION RESULTS: {model_name}")
    print("=" * 72)
    print(f"{'Model':<22} | {'HumanEval (pass@1)':<20} | {'MBPP (pass@1)':<15}")
    print("-" * 72)

    for name, base in OFFICIAL_BASELINES.items():
        print(f"{name:<22} | {base.humaneval_pass1:<20.1f} | {base.mbpp_pass1:<15.1f}")

    print("-" * 72)
    our_he = our_results.get("humaneval", 0.0)
    our_mbpp = our_results.get("mbpp", 0.0)
    print(f"⭐ {model_name:<19} | {our_he:<20.1f} | {our_mbpp:<15.1f}")
    print("=" * 72)
