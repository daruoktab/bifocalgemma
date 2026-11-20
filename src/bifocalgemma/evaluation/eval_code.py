"""Report supplied BifocalGemma scores alongside separately sourced references.

This module does not execute HumanEval, MBPP, EvalPlus, or repository repair tasks.
Published scores are context only; controlled comparisons require rerunning
baselines with the same dataset, harness, sampling, and token/tool budget.
"""

from dataclasses import dataclass
from typing import Optional

from bifocalgemma.config import EXPERIMENT_NAMES, EXTERNAL_COMPARISON_MODELS


@dataclass
class BaselineScores:
    """Reference scores; None means unavailable or not verified from a source."""

    model_name: str
    humaneval_pass1: Optional[float] = None
    mbpp_pass1: Optional[float] = None
    livecodebench_v6: Optional[float] = None
    swe_bench_lite: Optional[float] = None
    swe_bench_verified: Optional[float] = None
    crosscodeeval: Optional[float] = None
    source_url: Optional[str] = None


GEMMA4_MODEL_CARD = "https://ai.google.dev/gemma/docs/core/model_card_4"
SWEBENCH_LEADERBOARD = "https://www.swebench.com/"
CROSSCODEEVAL_REPO = "https://github.com/amazon-science/cceval"

# Only verified scores are included; unverified baseline numbers are None/N/A
REFERENCE_BASELINES: dict[str, BaselineScores] = {
    "Gemma-4-E2B-IT": BaselineScores(
        "Gemma-4-E2B-IT", livecodebench_v6=44.0, source_url=GEMMA4_MODEL_CARD
    ),
    "Gemma-4-E4B-IT": BaselineScores(
        "Gemma-4-E4B-IT", livecodebench_v6=52.0, source_url=GEMMA4_MODEL_CARD
    ),
}


# Canonical registry shared with configuration and the CLI.
EXPERIMENT_MODELS = EXPERIMENT_NAMES
EXTERNAL_BASELINE_MODELS = tuple(EXTERNAL_COMPARISON_MODELS)


def _format_score(value: Optional[float]) -> str:
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


def print_repair_comparison_table(
    our_results: dict[str, float],
    model_name: str = "BifocalGemma-E2B-E2B",
    *,
    illustrative: bool = False,
) -> None:
    """Print repair scores alongside a separately labeled completion diagnostic.

    Supported keys: swe_bench_lite and swe_bench_verified are resolved rates;
    crosscodeeval is completion exact match, not a repair success rate.
    """
    width = 98
    print("=" * width)
    label = "ILLUSTRATIVE SCORES (NOT MEASURED)" if illustrative else "SUPPLIED SCORES"
    print(f"  {label} (REPAIR / COMPLETION DIAGNOSTIC): {model_name}")
    print("  Reference protocols may differ; this table does not establish superiority.")
    print("=" * width)
    print(
        f"{'Model':<28} | {'SWE-bench Lite (%)':<20} | {'SWE-bench Ver. (%)':<20} | CrossCodeEval EM (%)"
    )
    print("-" * width)

    for name, base in REFERENCE_BASELINES.items():
        print(
            f"{name:<28} | {_format_score(base.swe_bench_lite):<20} | "
            f"{_format_score(base.swe_bench_verified):<20} | {_format_score(base.crosscodeeval)}"
        )

    print("-" * width)
    print(
        f"{model_name:<28} | {_format_score(our_results.get('swe_bench_lite')):<20} | "
        f"{_format_score(our_results.get('swe_bench_verified')):<20} | "
        f"{_format_score(our_results.get('crosscodeeval'))}"
    )
    print("=" * width)
    print(f"SWE-bench reference: {SWEBENCH_LEADERBOARD}")
    print(f"CrossCodeEval completion diagnostic: {CROSSCODEEVAL_REPO}")


def print_repair_experiment_table(
    results_by_model: dict[str, dict[str, float]],
    *,
    illustrative: bool = False,
) -> None:
    """Print the fixed four-variant/two-baseline repair study.

    Models use the canonical IDs in EXPERIMENT_MODELS. Values use swe_bench_lite,
    swe_bench_verified, and crosscodeeval (completion exact-match percentage).
    Absent metrics/models display N/A; references are not substituted for measurements.
    """
    unknown_models = set(results_by_model) - set(EXPERIMENT_MODELS)
    if unknown_models:
        raise ValueError(f"Models outside the six-model study: {sorted(unknown_models)}")
    label = "ILLUSTRATIVE SCORES (NOT MEASURED)" if illustrative else "SUPPLIED SCORES"
    print("=" * 104)
    print(f"  {label}: FOUR BASE-ENCODER / IT-DECODER PAIRS AND TWO IT BASELINES")
    print("  Matched protocols are required; this report does not execute benchmarks.")
    print(
        f"{'Model':<28} | {'SWE-bench Lite (%)':<20} | "
        f"{'SWE-bench Ver. (%)':<20} | CrossCodeEval EM (%)"
    )
    print("-" * 104)
    for model_name in EXPERIMENT_MODELS:
        metrics = results_by_model.get(model_name, {})
        print(
            f"{model_name:<28} | {_format_score(metrics.get('swe_bench_lite')):<20} | "
            f"{_format_score(metrics.get('swe_bench_verified')):<20} | "
            f"{_format_score(metrics.get('crosscodeeval'))}"
        )
    print("=" * 104)


def print_v4_parity_comparison_table(
    results_by_model: dict[str, dict[str, float]],
    *,
    title: str = "V4 APPLES-TO-APPLES PARITY COMPARISON (INTERNAL + EXTERNAL)",
) -> None:
    """Print controlled v4 parity scores across internal 3-way controls and external models."""
    width = 122
    print("=" * width)
    print(f"  {title}")
    print("  Same cohorts, greedy decoding, enable_thinking=False; N/A indicates unmeasured.")
    print("=" * width)
    print(
        f"{'Model / Mode':<34} | {'LongCodeQA':<11} | {'RepoNIAH':<10} | "
        f"{'HEFix pass@1':<12} | {'GSM8K Res.':<11} | {'CCL NormSim':<11}"
    )
    print("-" * width)
    for model_name, metrics in results_by_model.items():
        print(
            f"{model_name:<34} | {_format_score(metrics.get('longcodeqa')):<11} | "
            f"{_format_score(metrics.get('reponiah')):<10} | "
            f"{_format_score(metrics.get('humanevalfix_pass1')):<12} | "
            f"{_format_score(metrics.get('gsm8k_resilient')):<11} | "
            f"{_format_score(metrics.get('crosscodelong_norm_sim')):<11}"
        )
    print("=" * width)
