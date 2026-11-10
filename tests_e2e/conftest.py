"""Pytest fixtures and helper utilities for BifocalGemma E2E test suite."""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


def pytest_configure(config):
    """Register custom markers for the 4-tier E2E testing framework."""
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary & Corner Cases tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Combination tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Application Scenario tests")
    config.addinivalue_line("markers", "tier5: Tier 5 Adversarial Stress & Hardening tests")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_repair_row() -> Dict[str, Any]:
    """Sample repository repair task row."""
    return {
        "instance_id": "test-repair-001",
        "task_type": "repair",
        "record_type": "repository_patch_v1",
        "instruction": "Fix the off-by-one boundary error in slice calculation.",
        "context": "def slice_bounds(length, window):\n    return length - window + 1\n",
        "role": "eval",
    }


@pytest.fixture
def sample_mcq_row() -> Dict[str, Any]:
    """Sample multiple-choice code comprehension question row."""
    return {
        "instance_id": "test-mcq-001",
        "task_type": "mcq",
        "benchmark": "SWE-QA",
        "instruction": "Answer the code comprehension question. Return only A, B, C, or D.\n"
        "Which function handles connection retries?\nA) connect()\nB) retry_loop()\nC) close()\nD) ping()",
        "context": "class Connection:\n    def retry_loop(self):\n        pass\n",
        "role": "eval",
        "max_encoder_tokens": 8192,
        "max_new_tokens": 128,
    }


@pytest.fixture
def sample_crosscode_row() -> Dict[str, Any]:
    """Sample CrossCodeLong cross-file completion row."""
    return {
        "instance_id": "ccl-sample-001",
        "task_type": "completion",
        "benchmark": "CrossCodeLongEval",
        "instruction": "Complete the missing Python code between PREFIX and SUFFIX. Return only that code.",
        "context": "### PREFIX\ndef compute_hash(val):\n### MISSING CODE\n### SUFFIX\n    return result\n",
        "role": "eval",
        "max_encoder_tokens": 8192,
        "max_new_tokens": 1024,
    }


@pytest.fixture
def authoritative_empirical_results() -> Dict[str, Any]:
    """Authoritative empirical benchmark results verified across existing runs."""
    return {
        "sweqa_clean_oracle": {
            "accuracy": 0.7388,
            "n": 9072,
        },
        "sweqa_noisy_oracle": {
            "accuracy": 0.7369,
            "n": 9072,
            "degradation": -0.0019,
        },
        "longcodeqa": {
            "accuracy": 0.5514,
            "n": 107,
        },
        "humanevalfix_clean": {
            "syntax_pass_rate": 0.3171,
            "functional_pass_rate": 0.0192,
            "headline": 0.0061,
        },
        "crosscodelong": {
            "sequence_similarity": 0.3187,
            "regex_f1": 0.3670,
            "exact_match": 0.0,
        },
        "gsm8k_8shot": {
            "accuracy": 0.7309,
            "n": 1319,
        },
    }


def validate_empirical_results_schema(data: Dict[str, Any]) -> None:
    """Validate data dictionary against the PROJECT.md Interface Contract."""
    required_structure = {
        "sweqa_clean_oracle": {"accuracy": float, "n": int},
        "sweqa_noisy_oracle": {"accuracy": float, "n": int, "degradation": float},
        "longcodeqa": {"accuracy": float, "n": int},
        "humanevalfix_clean": {
            "syntax_pass_rate": float,
            "functional_pass_rate": float,
            "headline": float,
        },
        "crosscodelong": {
            "sequence_similarity": float,
            "regex_f1": float,
            "exact_match": (float, int, bool),
        },
        "gsm8k_8shot": {"accuracy": float, "n": int},
    }

    for benchmark_key, fields in required_structure.items():
        assert benchmark_key in data, f"Missing benchmark key '{benchmark_key}' in results data"
        record = data[benchmark_key]
        assert isinstance(record, dict), f"Benchmark record for '{benchmark_key}' must be a dict"
        for field, expected_type in fields.items():
            assert field in record, f"Missing field '{field}' in '{benchmark_key}'"
            val = record[field]
            if isinstance(expected_type, tuple):
                assert any(isinstance(val, t) for t in expected_type), (
                    f"Field '{benchmark_key}.{field}' has invalid type {type(val)}"
                )
            else:
                assert isinstance(val, expected_type), (
                    f"Field '{benchmark_key}.{field}' expected {expected_type}, got {type(val)}"
                )

            # Value range checks
            if field in (
                "accuracy",
                "syntax_pass_rate",
                "functional_pass_rate",
                "headline",
                "sequence_similarity",
                "regex_f1",
            ):
                assert 0.0 <= float(val) <= 1.0, (
                    f"Value of '{benchmark_key}.{field}' must be in [0.0, 1.0], got {val}"
                )
            elif field == "n":
                assert val > 0, f"Count 'n' in '{benchmark_key}' must be positive, got {val}"

    # Check degradation calculation consistency
    clean_acc = data["sweqa_clean_oracle"]["accuracy"]
    noisy_acc = data["sweqa_noisy_oracle"]["accuracy"]
    reported_deg = data["sweqa_noisy_oracle"]["degradation"]
    expected_deg = noisy_acc - clean_acc
    assert abs(reported_deg - expected_deg) < 1e-3, (
        f"Inconsistent noise degradation: reported {reported_deg}, calculated {expected_deg}"
    )


def validate_paper_manuscript(text: str) -> Dict[str, bool]:
    """Validate required sections in a scientific paper manuscript (Markdown or LaTeX)."""
    required_sections = [
        "abstract",
        "introduction",
        "architecture",
        "setup",
        "results",
        "limitations",
        "conclusion",
    ]
    lowered = text.lower()
    results = {}
    for sec in required_sections:
        # Check standard markdown/latex heading patterns
        patterns = [
            rf"#+\s*.*{sec}.*",
            rf"\\section\*?\{{.*{sec}.*\}}",
            rf"\b{sec}\b",
        ]
        found = any(re.search(p, lowered) for p in patterns)
        results[sec] = found
    return results


def reference_resilient_mcq_parser(text: str) -> Optional[str]:
    """Delegate directly to production extractor in bifocalgemma.evaluation.context."""
    from bifocalgemma.evaluation.context import extract_mcq_letter

    return extract_mcq_letter(text)
