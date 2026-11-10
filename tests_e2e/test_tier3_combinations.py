"""Tier 3: Cross-Feature Combinations.

Tests multi-feature interactions, serialization schemas, and pipeline contracts.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from tests_e2e.conftest import validate_empirical_results_schema


# ==============================================================================
# Cross-Feature: Evaluation Metric Serialization ↔ Paper Data Interface
# ==============================================================================
@pytest.mark.tier3
def test_combination_eval_metrics_to_paper_data_serialization(
    tmp_path: Path, authoritative_empirical_results: Dict[str, Any]
):
    """Verify evaluation outputs serialize to paper/data/empirical_benchmark_results.json format."""
    # Serialize to disk
    output_file = tmp_path / "empirical_benchmark_results.json"
    output_file.write_text(json.dumps(authoritative_empirical_results, indent=2), encoding="utf-8")

    # Read back and validate schema
    loaded = json.loads(output_file.read_text(encoding="utf-8"))
    validate_empirical_results_schema(loaded)

    # Cross-feature check: Verify noise degradation relationship holds in serialized data
    clean_acc = loaded["sweqa_clean_oracle"]["accuracy"]
    noisy_acc = loaded["sweqa_noisy_oracle"]["accuracy"]
    deg = loaded["sweqa_noisy_oracle"]["degradation"]
    assert abs(deg - (noisy_acc - clean_acc)) < 1e-4


# ==============================================================================
# Cross-Feature: MCQ Resilient Parser with SWE-QA and LongCodeQA Formats
# ==============================================================================
@pytest.mark.tier3
def test_combination_mcq_scoring_across_benchmarks():
    """Verify that simulated concise (E2B) and verbose (E4B) outputs both score correctly."""
    from tests_e2e.conftest import reference_resilient_mcq_parser

    # Simulated outputs from different models across SWE-QA and LongCodeQA
    simulated_cases = [
        # SWE-QA concise (E2B)
        {"gold": "A", "output": "A", "benchmark": "SWE-QA"},
        # SWE-QA verbose preamble (E4B)
        {
            "gold": "B",
            "output": "Based on the git blame and docstring, the correct function is (B).",
            "benchmark": "SWE-QA",
        },
        # LongCodeQA concise (E2B)
        {"gold": "C", "output": " C. ", "benchmark": "LongCodeQA"},
        # LongCodeQA verbose preamble (E4B)
        {
            "gold": "D",
            "output": "After analyzing the repository class definitions, choice: D",
            "benchmark": "LongCodeQA",
        },
    ]

    for case in simulated_cases:
        parsed = reference_resilient_mcq_parser(case["output"])
        assert parsed == case["gold"], (
            f"Failed cross-benchmark MCQ scoring for {case['benchmark']}: "
            f"expected {case['gold']}, got {parsed}"
        )


# ==============================================================================
# Cross-Feature: CrossCodeLong Continuous Metrics Summary Integration
# ==============================================================================
@pytest.mark.tier3
def test_combination_crosscode_continuous_metrics_aggregation():
    """Verify aggregated summary retains continuous metrics when binary exact match is 0.0."""
    from bifocalgemma.evaluation.context import completion_metrics

    # Simulate 5 completion predictions with typical partial matches
    predictions = [
        ("def calc(a, b): return a + b", "def calc(x, y): return x + y"),
        (
            "def filter_items(items): return [i for i in items if i > 0]",
            "def filter_items(lst): return [x for x in lst if x > 0]",
        ),
        ("class Node: pass", "class Node:\n    def __init__(self): pass"),
        ("result = query.fetch()", "result = query.execute()"),
        ("import os\nimport sys", "import sys\nimport os"),
    ]

    scored_records = []
    for pred, ref in predictions:
        metrics = completion_metrics(pred, ref)
        scored_records.append(metrics)

    # Compute summary aggregates
    total = len(scored_records)
    exact_match_rate = sum(r["exact_match"] for r in scored_records) / total
    mean_similarity = sum(r["sequence_similarity"] for r in scored_records) / total
    mean_f1 = sum(r["regex_identifier_set_f1"] for r in scored_records) / total

    # Under strict exact match, rate may be low or zero
    assert exact_match_rate < 0.5

    # But continuous metrics MUST provide meaningful positive signal
    assert mean_similarity > 0.3
    assert mean_f1 > 0.2

    # Verify structured summary dictionary format
    summary_dict = {
        "crosscodelong": {
            "exact_match": exact_match_rate,
            "sequence_similarity": mean_similarity,
            "regex_f1": mean_f1,
        }
    }
    assert summary_dict["crosscodelong"]["sequence_similarity"] > 0.0
    assert summary_dict["crosscodelong"]["regex_f1"] > 0.0
