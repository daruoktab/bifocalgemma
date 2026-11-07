"""Tier 5: Adversarial Stress-Testing & Robustness Hardening.

Empirical verification suite challenging:
1. Regex MCQ parser against complex preambles, formatting variations, distractor invalidation, and ambiguity guards.
2. Prompt neutrality across all task kinds and baseline vs paired modes.
3. Continuous metrics under boundary conditions (disjoint identifiers, Unicode, pure symbols, diffs).
4. RepoNIAH distractor harness across varying context lengths, depths, reproducibility, and scorer edge cases.
"""

import hashlib

import pytest

from bifocalgemma.data.context_tasks import task_prompt
from bifocalgemma.evaluation.context import (
    build_repo_niah_case,
    completion_metrics,
    decoder_prompt,
    extract_mcq_letter,
    score_repo_niah_case,
    strict_letter,
)

# ==============================================================================
# 1. Regex MCQ Parser Adversarial Challenges
# ==============================================================================

@pytest.mark.tier5
@pytest.mark.parametrize(
    "raw_input,expected",
    [
        ("Answer: (A)", "A"),
        ("Choice: [B]", "B"),
        ("Option {C}", "C"),
        ("Answer is: *D*", "D"),
        ("The correct answer is: 'A'.", "A"),
        ('The answer is: "B"', "B"),
        ("Therefore, the answer is: C", "C"),
        ("Hence, option (D) is correct.", "D"),
        ("So conclusion: A.", "A"),
        ("Based on the code analysis, choice is B.", "B"),
    ],
)
def test_adversarial_mcq_isolated_formatting(raw_input: str, expected: str):
    """Verify parser extracts correct letter under various single-option formatting styles."""
    parsed = extract_mcq_letter(raw_input)
    assert parsed == expected, f"Failed on '{raw_input}': expected '{expected}', got '{parsed}'"


@pytest.mark.tier5
def test_adversarial_mcq_ambiguity_guards():
    """Verify that genuine ambiguity does NOT hallucinate false certainty."""
    # Direct "or / vs / nor" ambiguity without explicit final answer
    ambiguous_samples = [
        "It could be either A or D depending on the version.",
        "Both A and B seem valid here.",
        "Neither A nor C matches the requirements.",
        "Comparing A vs B, the difference is negligible.",
    ]
    for text in ambiguous_samples:
        parsed = extract_mcq_letter(text)
        assert parsed is None, (
            f"Expected None for ambiguous text '{text}', but got '{parsed}'"
        )


@pytest.mark.tier5
def test_adversarial_mcq_split_ambiguity_behavior():
    """Evaluate behavior when candidate options are mentioned in separate sentences without direct connector."""
    text = "Option A might work under condition 1. But Option D could also work under condition 2."
    parsed = extract_mcq_letter(text)
    # Empirically records whether the parser returns None (safe) or hallucinations 'D' (picks last mention).
    # Current behavior: returns 'D' because Stage 2 matches both and picks matches[-1].
    assert parsed in ("D", None), f"Unexpected parsed letter: {parsed}"


@pytest.mark.tier5
def test_adversarial_mcq_ruling_out_distractors_vulnerability():
    """Verify that ruling out an option with markdown/quotes extracts the correct answer.

    When the target answer is wrapped in quotes or markdown bold, resilient parsing
    prioritizes the conclusive answer over mentioned distractor options.
    """
    adversarial_cases = [
        ("Option B is wrong. The correct answer is 'A'.", "A"),
        ('Option B is wrong. The correct answer is "A".', "A"),
        ("Option B is wrong. The correct answer is **A**.", "A"),
        ("Option B is wrong. The correct answer is `A`.", "A"),
        ("Option B is wrong. The correct answer is [A].", "A"),
        ("Option B is wrong. The correct answer is {A}.", "A"),
    ]
    for text, expected in adversarial_cases:
        actual = extract_mcq_letter(text)
        assert actual == expected, f"Failed on '{text}': expected '{expected}', got '{actual}'"


@pytest.mark.tier5
def test_adversarial_mcq_verbose_reasoning_preambles():
    """Stress-test preambles where model compares options before concluding.

    Example from dispatch: 'I considered A and C, but ultimately B is correct: (B)'
    """
    case1 = "I considered A and C, but ultimately B is correct: (B)"
    case2 = "I considered A and C, but the correct answer is B."
    case3 = "I considered A and C, but the correct answer: B."
    case4 = "I considered A and C, but the answer is B."

    assert extract_mcq_letter(case3) == "B"
    assert extract_mcq_letter(case4) == "B"
    res1 = extract_mcq_letter(case1)
    res2 = extract_mcq_letter(case2)
    assert res1 == "B", f"Expected 'B' for case1, got {res1}"
    assert res2 == "B", f"Expected 'B' for case2, got {res2}"


@pytest.mark.tier5
def test_adversarial_mcq_adverb_preambles():
    """Verify that affirmative adverbs do not corrupt option letter extraction."""
    adverb_cases = [
        ("The answer is clearly (A)", "A"),
        ("The answer is definitely (A)", "A"),
        ("The answer is actually (C)", "C"),
        ("The correct answer is clearly A.", "A"),
    ]
    for text, expected in adverb_cases:
        actual = extract_mcq_letter(text)
        assert actual == expected, f"Failed on '{text}': expected '{expected}', got '{actual}'"


@pytest.mark.tier5
def test_adversarial_mcq_lowercase_letter_normalization():
    """Verify that bare lowercase letter responses are supported and normalized to uppercase."""
    lowercase_bare = [("b", "B"), ("(b)", "B"), ("b.", "B"), ("  c  ", "C"), ("a)", "A")]
    for s, expected in lowercase_bare:
        assert extract_mcq_letter(s) == expected, f"Expected '{expected}' for lowercase input '{s}'"

    # Also formatted lowercase with keyword like 'Answer: b' or 'Option: (b)' works
    assert extract_mcq_letter("Answer: b") == "B"
    assert extract_mcq_letter("Option: (b)") == "B"


# ==============================================================================
# 2. Prompt Neutrality Across Tasks & Modes
# ==============================================================================

@pytest.mark.tier5
def test_adversarial_prompt_neutrality_matrix():
    """Verify prompt neutrality across exhaustive combination of task kinds and modes."""
    forbidden_architectural_phrases = [
        "using the encoder context",
        "use the supplied encoder context",
        "encoder: object",
        "manifold warping",
        "cross-attention",
        "bifocal",
    ]

    all_task_types = [
        "repair",
        "localization",
        "completion",
        "mcq",
        "retrieval",
        "repository_patch_v1",
    ]

    for task_type in all_task_types:
        row = {
            "task_type": task_type,
            "instruction": f"Execute {task_type} instructions cleanly.",
            "context": f"# Fake context for {task_type} with internal details.\ndef dummy(): pass\n",
        }
        if task_type == "repository_patch_v1":
            row["record_type"] = "repository_patch_v1"

        for paired in (True, False):
            # Test decoder_prompt
            dp = decoder_prompt(row, paired=paired)
            for phrase in forbidden_architectural_phrases:
                assert phrase not in dp.lower(), (
                    f"Forbidden phrase '{phrase}' found in decoder_prompt for task '{task_type}' (paired={paired})"
                )

            # In paired mode, decoder_prompt MUST NOT contain raw context
            if paired:
                assert row["context"] not in dp, (
                    f"Context leaked into paired decoder_prompt for task '{task_type}'"
                )
            else:
                assert row["context"] in dp, (
                    f"Context missing from baseline decoder_prompt for task '{task_type}'"
                )

            # Test task_prompt for supported SFT tasks
            if task_type in ("repair", "localization", "completion"):
                tp = task_prompt(row, paired=paired)
                for phrase in forbidden_architectural_phrases:
                    assert phrase not in tp.lower(), (
                        f"Forbidden phrase '{phrase}' found in task_prompt for task '{task_type}' (paired={paired})"
                    )


@pytest.mark.tier5
def test_adversarial_context_containing_forbidden_phrases():
    """Verify that even if raw context contains architectural phrasing, paired prompt stays clean."""
    poisoned_row = {
        "task_type": "repair",
        "instruction": "Fix bug in calculation module.",
        "context": "# Historical note: using the encoder context and encoder: object is legacy.\n",
    }
    # In paired mode, decoder_prompt must only emit instruction
    paired_prompt = decoder_prompt(poisoned_row, paired=True)
    assert "using the encoder context" not in paired_prompt.lower()
    assert "encoder: object" not in paired_prompt.lower()
    assert poisoned_row["instruction"] in paired_prompt


# ==============================================================================
# 3. Continuous Metrics Adversarial Boundaries
# ==============================================================================

@pytest.mark.tier5
def test_adversarial_metrics_disjoint_identifiers():
    """Verify completion_metrics on code with completely disjoint identifiers (pure variables)."""
    pred = "alpha_val = 10\nbeta_sum = 20\n"
    ref = "gamma_out = 30\ndelta_res = 40\n"

    res = completion_metrics(pred, ref)
    assert res["exact_match"] is False
    assert res["regex_identifier_set_f1"] == 0.0
    assert 0.0 <= res["sequence_similarity"] < 1.0


@pytest.mark.tier5
def test_adversarial_metrics_keyword_identifier_inflation():
    """Verify that language keywords (def, return) are filtered out, yielding 0.0 F1 between unrelated functions."""
    pred = "def calculate_sum(val_a, val_b):\n    return val_a + val_b\n"
    ref = "def render_view(context_x, template_y):\n    return context_x.format(template_y)\n"

    res = completion_metrics(pred, ref)
    assert res["exact_match"] is False
    # Verified: 'def' and 'return' are excluded, preventing keyword inflation on disjoint functions
    assert res["regex_identifier_set_f1"] == 0.0, (
        f"Expected keywords to be filtered (F1=0.0), got {res['regex_identifier_set_f1']}"
    )


@pytest.mark.tier5
def test_adversarial_metrics_identical_sequences():
    """Verify completion_metrics on identical code strings."""
    code = "class Controller:\n    def execute(self):\n        return True\n"
    res = completion_metrics(code, code)
    assert res["exact_match"] is True
    assert res["sequence_similarity"] == 1.0
    assert res["regex_identifier_set_f1"] == 1.0


@pytest.mark.tier5
def test_adversarial_metrics_unicode_identifiers_blind_spot():
    """Verify Unicode identifiers (PEP 3131) are captured correctly and yield 0.0 F1 on disjoint variables."""
    pred_unicode = "α_coef = 10\n"
    ref_unicode = "β_val = 20\n"

    res = completion_metrics(pred_unicode, ref_unicode)
    assert res["exact_match"] is False
    # Verified: Unicode identifiers are extracted and compared cleanly
    assert res["regex_identifier_set_f1"] == 0.0, (
        f"Expected F1=0.0 for disjoint Unicode identifiers, got {res['regex_identifier_set_f1']}"
    )


@pytest.mark.tier5
def test_adversarial_metrics_pure_operator_zero_division_guard():
    """Verify that disjoint pure symbol strings receive F1 = 0.0 rather than 1.0."""
    pred_symbols = "+ = * / %"
    ref_symbols = "- = * / %"

    res = completion_metrics(pred_symbols, ref_symbols)
    assert res["exact_match"] is False
    assert res["regex_identifier_set_f1"] == 0.0, (
        f"Expected F1=0.0 from zero-identifier guard fallback, got {res['regex_identifier_set_f1']}"
    )


@pytest.mark.tier5
def test_adversarial_metrics_multiline_diff_inflation():
    """Demonstrate how diff header keywords inflate continuous metrics between unrelated patches."""
    diff_pred = """--- a/service/auth.py
+++ b/service/auth.py
@@ -1,3 +1,3 @@
-def authenticate(user):
+def verify_token(user):
"""
    diff_ref = """--- a/storage/db.py
+++ b/storage/db.py
@@ -1,3 +1,3 @@
-def query_records(table):
+def fetch_rows(table):
"""
    res = completion_metrics(diff_pred, diff_ref)
    assert res["exact_match"] is False
    # Sequence similarity and F1 are significantly non-zero (> 0.40) solely due to diff headers (--- a/, +++ b/, def)
    assert res["sequence_similarity"] > 0.40
    assert res["regex_identifier_set_f1"] > 0.30


# ==============================================================================
# 4. RepoNIAH Distractor Harness Adversarial Challenge
# ==============================================================================

@pytest.mark.tier5
@pytest.mark.parametrize("depth", [1000, 4000, 8000, 16000, 32000])
@pytest.mark.parametrize("depth_ratio", [0.0, 0.5, 1.0])
def test_adversarial_reponiah_depth_and_ratio_extremes(depth: int, depth_ratio: float):
    """Verify RepoNIAH haystack generation and needle embedding across extreme depths and ratios."""
    pub, gold = build_repo_niah_case(depth, depth_ratio, seed=42)
    haystack = pub["context"]
    needle = gold["needle"]

    assert needle in haystack, f"Needle '{needle}' missing from haystack at depth={depth}, ratio={depth_ratio}"

    # Verify monotonic position
    needle_pos = haystack.find(needle)
    rel_pos = needle_pos / len(haystack)

    if depth_ratio == 0.0:
        assert rel_pos < 0.10, f"Needle at ratio 0.0 should be near start, got {rel_pos:.3f}"
    elif depth_ratio == 0.5:
        assert 0.35 <= rel_pos <= 0.65, f"Needle at ratio 0.5 should be near center, got {rel_pos:.3f}"
    elif depth_ratio == 1.0:
        assert rel_pos > 0.80, f"Needle at ratio 1.0 should be near end, got {rel_pos:.3f}"


@pytest.mark.tier5
def test_adversarial_reponiah_deterministic_reproducibility():
    """Verify that RepoNIAH generation is 100% deterministically reproducible across calls."""
    pub1, gold1 = build_repo_niah_case(8000, 0.5, seed=999)
    pub2, gold2 = build_repo_niah_case(8000, 0.5, seed=999)

    assert pub1 == pub2
    assert gold1 == gold2
    assert hashlib.sha256(pub1["context"].encode()).hexdigest() == hashlib.sha256(pub2["context"].encode()).hexdigest()

    # Different seeds must yield different needles
    pub3, gold3 = build_repo_niah_case(8000, 0.5, seed=1000)
    assert gold1["needle"] != gold3["needle"]


@pytest.mark.tier5
def test_adversarial_reponiah_scorer_negation_vulnerability():
    """Verify that RepoNIAH scorer rejects negated needle answers.

    If a model outputs: 'The secret key is definitely NOT secret_omega_42',
    score_repo_niah_case evaluates correct=False due to negation guard.
    """
    pub = {"instance_id": "test_case", "depth_ratio": 0.5, "target_context_tokens": 4000}
    gold = {"needle": "secret_omega_42"}

    # Exact match
    res_exact = score_repo_niah_case(pub, gold, "secret_omega_42")
    assert res_exact["correct"] is True

    # Adversarial negation output
    res_negated = score_repo_niah_case(pub, gold, "The secret key is definitely NOT secret_omega_42, it is missing.")
    # Confirms negation guard: negated completion evaluates to False
    assert res_negated["correct"] is False, "Expected negation guard to reject negated needle"

    # Intervening words in negation output
    negated_cases = [
        "The repository does not contain secret_omega_42",
        "I did not find secret_omega_42",
        "The codebase does not have secret_omega_42",
        "We could not locate secret_omega_42",
        "The repository doesn't contain secret_omega_42",
        "The codebase doesn't have secret_omega_42",
        "It doesn't match secret_omega_42",
        "I can't find secret_omega_42",
        "It won't find secret_omega_42",
        "We don’t have secret_omega_42",
        "The repository doesn’t contain secret_omega_42",
    ]
    for case in negated_cases:
        res = score_repo_niah_case(pub, gold, case)
        assert res["correct"] is False, f"Expected negation guard to reject '{case}'"


@pytest.mark.tier5
def test_adversarial_mcq_delimited_negation_and_conjunction_guards():
    """Verify delimited negative statements and conjoined options do not falsely match."""
    cases = [
        ("Option (A) is wrong", None),
        ("Option [A] is incorrect", None),
        ("Option (C) is suboptimal", None),
        ("The answer is not (A)", None),
        ("The answer is definitely not (B).", None),
        ("The answer is (A) or (B)", None),
        ("The answer is (A) and (B)", None),
        ("Option (A) is wrong. Option (C) is suboptimal. Therefore, the answer is B.", "B"),
        ("Option {A} is wrong", None),
        ("Option {A} is invalid", None),
        ("Option A is false.", None),
        ("The answer is {A} or {B}", None),
        ("It cannot be A.", None),
        ("The answer could not be A.", None),
        ("The answer is not necessarily A.", None),
        ("{A}", "A"),
    ]
    assert strict_letter("{A}", allow_fallback=False) == "A"
    for text, expected in cases:
        actual = extract_mcq_letter(text)
        assert actual == expected, f"Failed on '{text}': expected '{expected}', got '{actual}'"


@pytest.mark.tier5
def test_adversarial_mcq_redos_scaling():
    """Verify linear/sub-millisecond execution on adversarial repeated adverbs (no ReDoS)."""
    import time
    for n in [2, 10, 25, 50]:
        text = "therefore " + "clearly " * n + "something"
        t0 = time.perf_counter()
        res = extract_mcq_letter(text)
        dt = time.perf_counter() - t0
        assert res is None
        assert dt < 0.05, f"ReDoS detected at n={n}: took {dt:.4f}s"

