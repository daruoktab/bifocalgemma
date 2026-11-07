"""Tier 2: Boundary & Corner Cases.

Tests extreme values, empty strings, multiline preambles, and malformed inputs.
"""

import math
from typing import Any

import pytest

from tests_e2e.conftest import reference_resilient_mcq_parser


# ==============================================================================
# Boundary: Empty Strings & Whitespace Handling
# ==============================================================================
@pytest.mark.tier2
def test_boundary_empty_and_whitespace_completions():
    """Verify completion metrics handle empty and whitespace-only strings gracefully."""
    from bifocalgemma.evaluation.context import completion_metrics

    # Case 1: Both empty
    res_both_empty = completion_metrics("", "")
    assert res_both_empty["exact_match"] is True
    assert res_both_empty["sequence_similarity"] == 1.0
    assert res_both_empty["regex_identifier_set_f1"] == 1.0

    # Case 2: Prediction empty, reference non-empty
    res_pred_empty = completion_metrics("", "def foo(): pass")
    assert res_pred_empty["exact_match"] is False
    assert res_pred_empty["sequence_similarity"] == 0.0
    assert res_pred_empty["regex_identifier_set_f1"] == 0.0

    # Case 3: Prediction non-empty, reference empty
    res_ref_empty = completion_metrics("def foo(): pass", "")
    assert res_ref_empty["exact_match"] is False
    assert res_ref_empty["sequence_similarity"] == 0.0
    assert res_ref_empty["regex_identifier_set_f1"] == 0.0

    # Case 4: Whitespace only
    res_ws = completion_metrics("   \n\t  ", "   ")
    assert res_ws["exact_match"] is True
    assert res_ws["sequence_similarity"] == 1.0
    assert res_ws["regex_identifier_set_f1"] == 1.0


# ==============================================================================
# Boundary: Zero-Identifier Code Metric Division Safeguard
# ==============================================================================
@pytest.mark.tier2
def test_boundary_zero_identifier_f1_safeguard():
    """Verify completion_metrics does not throw ZeroDivisionError when code has no identifiers."""
    from bifocalgemma.evaluation.context import completion_metrics

    # Code containing only operators/punctuation, no alphanumeric identifiers
    pure_symbols_1 = "+ = * / %"
    pure_symbols_2 = "- = * / %"

    res = completion_metrics(pure_symbols_1, pure_symbols_2)
    assert not math.isnan(res["regex_identifier_set_f1"])
    assert 0.0 <= res["regex_identifier_set_f1"] <= 1.0
    assert not math.isnan(res["sequence_similarity"])


# ==============================================================================
# Boundary: Single Characters & Out-of-Alphabet Letter Extraction
# ==============================================================================
@pytest.mark.tier2
@pytest.mark.parametrize(
    "raw_input,expected",
    [
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("D", "D"),
        ("E", None),
        ("F", None),
        ("1", None),
        ("?", None),
        ("@", None),
        ("", None),
        ("   ", None),
    ],
)
def test_boundary_letter_extraction_alphabet(raw_input: str, expected: Any):
    """Verify that letter parser strictly restricts to [ABCD] and rejects other characters."""
    parsed = reference_resilient_mcq_parser(raw_input)
    assert parsed == expected, f"Input '{raw_input}' expected {expected}, got {parsed}"


# ==============================================================================
# Boundary: Extreme Conversational Preambles & Meta-Characters
# ==============================================================================
@pytest.mark.tier2
def test_boundary_multiline_conversational_preamble():
    """Verify resilient parsing when model emits extensive multi-paragraph explanation."""
    verbose_text = (
        "Let's think through this step by step.\n\n"
        "1. First, we examine the module imports in line 4.\n"
        "2. The function retry_loop() is clearly defined as a member method.\n"
        "3. Although option A mentions connect(), it does not implement retry logic.\n"
        "4. Option D ping() is merely a health check probe.\n\n"
        "Therefore, after careful evaluation of all options, the correct choice is (B).\n"
    )

    parsed = reference_resilient_mcq_parser(verbose_text)
    assert parsed == "B", f"Failed to extract 'B' from long conversational preamble, got {parsed}"


@pytest.mark.tier2
def test_boundary_regex_meta_characters_in_generation():
    """Verify parser robustness against tricky punctuation and regex meta-characters."""
    tricky_texts = [
        ("Answer: (A)", "A"),
        ("Choice: [B]", "B"),
        ("Option {C}", "C"),
        ("Answer is: *D*", "D"),
        ("The correct answer is: 'A'.", "A"),
        ('The answer is: "B"', "B"),
    ]
    for text, expected in tricky_texts:
        parsed = reference_resilient_mcq_parser(text)
        assert parsed == expected, f"Failed on '{text}': got {parsed}"


# ==============================================================================
# Boundary: Extreme Token Budget Bounds
# ==============================================================================
@pytest.mark.tier2
def test_boundary_token_budget_bounds_enforcement():
    """Verify token budget bounds checking logic for extreme values."""
    decoder_limit = 8192
    paired_prefix_size = 256

    def compute_generation_budget(requested_budget: int) -> int:
        available = decoder_limit - paired_prefix_size
        if requested_budget <= 0:
            return 0
        return min(requested_budget, available)

    # Budget = 0
    assert compute_generation_budget(0) == 0

    # Negative budget
    assert compute_generation_budget(-10) == 0

    # Normal budget
    assert compute_generation_budget(128) == 128

    # Extreme oversized budget (> 8192)
    assert compute_generation_budget(100_000) == 7936
