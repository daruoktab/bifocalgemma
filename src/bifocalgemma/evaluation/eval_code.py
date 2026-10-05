"""Report supplied BifocalGemma scores alongside separately sourced references.

This module does not execute HumanEval, MBPP, EvalPlus, or repository repair tasks.
Published scores are context only; controlled comparisons require rerunning
baselines with the same dataset, harness, sampling, and token/tool budget.
"""

from dataclasses import dataclass


@dataclass
class BaselineScores:
    """Reference scores; None means unavailable or not verified from a source."""

    model_name: str
    humaneval_pass1: float | None = None
    mbpp_pass1: float | None = None
    livecodebench_v6: float | None = None
    source_url: str | None = None


GEMMA4_MODEL_CARD = "https://ai.google.dev/gemma/docs/core/model_card_4"

# Only these LiveCodeBench v6 scores were verified in the model card.
# The previous HumanEval/MBPP numbers had no verified attribution.
REFERENCE_BASELINES: dict[str, BaselineScores] = {
    "Gemma-4-E2B-IT": BaselineScores(
        "Gemma-4-E2B-IT", livecodebench_v6=44.0, source_url=GEMMA4_MODEL_CARD
    ),
    "Gemma-4-E4B-IT": BaselineScores(
        "Gemma-4-E4B-IT", livecodebench_v6=52.0, source_url=GEMMA4_MODEL_CARD
    ),
    "Gemma-3-1B-IT": BaselineScores("Gemma-3-1B-IT"),
    "Gemma-3-4B-IT": BaselineScores("Gemma-3-4B-IT"),
    "T5Gemma-2-1B-1B": BaselineScores("T5Gemma-2-1B-1B"),
    "T5Gemma-2-4B-4B": BaselineScores("T5Gemma-2-4B-4B"),
}


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def print_comparison_table(
    our_results: dict[str, float],
    model_name: str = "BifocalGemma-E2B-E2B",
    *,
    illustrative: bool = False,
) -> None:
    """Print references and supplied scores; absent metrics display as N/A.

    Set illustrative=True for example numbers rather than measured results.
    Supported result keys are humaneval, mbpp, and livecodebench_v6.
    """
    width = 98
    print("=" * width)
    label = "ILLUSTRATIVE SCORES (NOT MEASURED)" if illustrative else "SUPPLIED SCORES"
    print(f"  {label}: {model_name}")
    print("  Reference protocols may differ; this table does not establish superiority.")
    print("=" * width)
    print(f"{'Model':<28} | {'HumanEval (pass@1)':<20} | {'MBPP (pass@1)':<15} | LCB v6 (%)")
    print("-" * width)

    for name, base in REFERENCE_BASELINES.items():
        print(
            f"{name:<28} | {_format_score(base.humaneval_pass1):<20} | "
            f"{_format_score(base.mbpp_pass1):<15} | {_format_score(base.livecodebench_v6)}"
        )

    print("-" * width)
    print(
        f"{model_name:<28} | {_format_score(our_results.get('humaneval')):<20} | "
        f"{_format_score(our_results.get('mbpp')):<15} | "
        f"{_format_score(our_results.get('livecodebench_v6'))}"
    )
    print("=" * width)
    print(f"Gemma 4 LiveCodeBench v6 reference: {GEMMA4_MODEL_CARD}")
