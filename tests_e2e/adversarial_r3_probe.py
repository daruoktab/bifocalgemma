"""Empirical Adversarial Challenge Suite for BifocalGemma Gate Round 3.

Exhaustively verifies:
1. Resolution status of all 9 Gate Round 2 failing cases.
2. MCQ Delimiters, Punctuation, Lowercase, and Preambles.
3. Ambiguity Guards and Comparative Ruling-Out.
4. RepoNIAH Distractor Retrieval: Affirmative Accuracy vs Negation Guard Coverage.
5. Catastrophic Backtracking (ReDoS) scaling in Conclusive MCQ Regex.
6. Python 3 Unicode Identifiers and Metric Zero-Div Guards.
7. Zero Prompt Leak Scans.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src").resolve()))

from bifocalgemma.evaluation.context import (
    completion_metrics,
    extract_mcq_letter,
    score_repo_niah_case,
)


def verify_gate_round_2_resolved_cases():
    """Verify that all 9 previously failing cases from Round 2 are resolved."""
    print("\n" + "=" * 60)
    print("PROBE 1: Gate Round 2 Previous 9 Failures Verification")
    print("=" * 60)

    r2_cases = [
        # MCQ Adverbs
        ("The answer is clearly (A)", "A"),
        ("The answer is definitely (A)", "A"),
        ("The answer is actually (C)", "C"),
        ("The correct answer is clearly A.", "A"),
        ("The answer is doubtful", None),
        # Delimiter angle bracket (was expected None)
        ("<B>", None),
    ]

    r2_niah_cases = [
        ("The repository does not contain secret_omega_42", False),
        ("I did not find secret_omega_42", False),
        ("The codebase does not have secret_omega_42", False),
        ("We could not locate secret_omega_42", False),
    ]

    failures = []
    for text, expected in r2_cases:
        actual = extract_mcq_letter(text)
        if actual != expected:
            failures.append((text, expected, actual))
        print(f"  MCQ: {text!r} -> {actual!r} (expected {expected!r})")

    row = {"instance_id": "test_r2", "depth_ratio": 0.5, "target_context_tokens": 4000}
    gold = {"needle": "secret_omega_42"}
    for text, expected in r2_niah_cases:
        res = score_repo_niah_case(row, gold, text)
        actual = res["correct"]
        if actual != expected:
            failures.append((text, expected, actual))
        print(f"  NIAH: {text!r} -> {actual} (expected {expected})")

    print(f"Result: {len(r2_cases) + len(r2_niah_cases) - len(failures)}/{len(r2_cases) + len(r2_niah_cases)} passed.")
    return failures


def probe_mcq_delimiters_and_wrappers():
    """Verify delimiters, quotes, markdown, brackets, and XML tags."""
    print("\n" + "=" * 60)
    print("PROBE 2: Delimiters, Markdown Wrappers, and XML Structure")
    print("=" * 60)

    cases = [
        ("'A'", "A"),
        ('"B"', "B"),
        ("`C`", "C"),
        ("*D*", "D"),
        ("**A**", "A"),
        ("***B***", "B"),
        ("[C]", "C"),
        ("{D}", "D"),
        ("(A)", "A"),
        ("['A']", "A"),
        ('["B"]', "B"),
        ("('C')", "C"),
        ("(`D`)", "D"),
        ("**'A'**", "A"),
        ("`**B**`", "B"),
        ("The answer is: 'A'.", "A"),
        ('The answer is: "B"', "B"),
        ("The answer is `C`", "C"),
        ("The answer is **D**", "D"),
        ("The answer is [A]", "A"),
        ("The answer is {B}", "B"),
        ("The answer is (C)", "C"),
        ("<answer>A</answer>", "A"),
        ("<answer>B</answer>", "B"),
        ("<answer>(C)</answer>", "C"),
        ("<option>D</option>", "D"),
    ]

    failures = []
    for text, expected in cases:
        actual = extract_mcq_letter(text)
        if actual != expected:
            failures.append((text, expected, actual))
    print(f"Result: {len(cases) - len(failures)}/{len(cases)} passed.")
    return failures


def probe_mcq_preambles_and_ambiguity():
    """Verify complex conversational reasoning preambles and ambiguity guards."""
    print("\n" + "=" * 60)
    print("PROBE 3: Preambles, Distractor Elimination, and Ambiguity Guards")
    print("=" * 60)

    cases = [
        ("I considered A and C, but ultimately B is correct: (B)", "B"),
        ("I considered A and C, but the correct answer is B.", "B"),
        ("I considered A and C, but the answer is B.", "B"),
        ("Option B is wrong. The correct answer is 'A'.", "A"),
        ("Option A has a syntax error. Option C is suboptimal. Therefore, the answer is B.", "B"),
        ("We can eliminate A and D immediately. Comparing B and C, option B is correct.", "B"),
        ("It cannot be A because of type mismatch. Thus, option D is correct.", "D"),
        ("<think>\nA is buggy, C is deprecated\n</think>\nThe correct answer is B.", "B"),
        ("The answer is clearly and definitely A.", "A"),
        ("The answer is absolutely B.", "B"),
        ("The answer is undoubtedly C.", "C"),
        # Ambiguity guards (must return None)
        ("It could be either A or B depending on python version.", None),
        ("Both A and B are valid solutions.", None),
        ("Neither A nor C matches the specification.", None),
        ("Comparing A vs B, neither is clearly superior.", None),
        ("I am stuck between option A and option D.", None),
        ("The answer is not B.", None),
        ("The answer is definitely not B.", None),
        ("Option C is incorrect.", None),
        ("Option D is wrong.", None),
    ]

    failures = []
    for text, expected in cases:
        actual = extract_mcq_letter(text)
        if actual != expected:
            failures.append((text, expected, actual))
    print(f"Result: {len(cases) - len(failures)}/{len(cases)} passed.")
    return failures


def probe_reponiah_negation_coverage():
    """Audit RepoNIAH distractor negative rejection vs affirmative retrievals."""
    print("\n" + "=" * 60)
    print("PROBE 4: RepoNIAH Distractor Negation Coverage & Contraction Vulnerabilities")
    print("=" * 60)

    row = {"instance_id": "test_niah", "depth_ratio": 0.5, "target_context_tokens": 4000}
    gold = {"needle": "secret_omega_42"}

    cases = [
        # Affirmative (MUST be True)
        ("secret_omega_42", True, "bare needle"),
        ("The key is secret_omega_42", True, "prefix sentence"),
        ("Found needle: `secret_omega_42` in repo", True, "backticked found"),
        ("The secret key is 'secret_omega_42'.", True, "single-quoted affirmative"),
        ("Initial guess was not secret_omega_42, but the actual key is secret_omega_42.", True, "negated then affirmed"),
        # Supported Negations (MUST be False)
        ("The secret key is definitely NOT secret_omega_42, it is missing.", False, "definitely NOT"),
        ("The key is not secret_omega_42", False, "is not"),
        ("It never matches secret_omega_42", False, "never matches"),
        ("There is no secret_omega_42 found in the context.", False, "no ... found"),
        ("The repo was built without secret_omega_42.", False, "without"),
        ("I didn't find secret_omega_42", False, "didn't find"),
        ("We don't have secret_omega_42", False, "don't have"),
        ("The repository does not contain secret_omega_42", False, "does not contain"),
        ("I did not find secret_omega_42", False, "did not find"),
        ("We could not locate secret_omega_42", False, "could not locate"),
        ("We couldn't find secret_omega_42", False, "couldn't find"),
        # Unhandled Contractions & Curly Apostrophes (Adversarial Gaps)
        ("The repository doesn't contain secret_omega_42", False, "doesn't contain (3rd person)"),
        ("The codebase doesn't have secret_omega_42", False, "doesn't have (3rd person)"),
        ("It doesn't match secret_omega_42", False, "doesn't match"),
        ("I can't find secret_omega_42", False, "can't find"),
        ("It won't find secret_omega_42", False, "won't find"),
        ("We don’t have secret_omega_42", False, "don’t have (curly apostrophe U+2019)"),
        ("The repository doesn’t contain secret_omega_42", False, "doesn’t contain (curly apostrophe U+2019)"),
    ]

    failures = []
    for text, expected, label in cases:
        res = score_repo_niah_case(row, gold, text)
        actual = res["correct"]
        if actual != expected:
            failures.append((text, expected, actual, label))
            print(f"  [FAIL] {label:42s}: got {actual!s:5s} (expected {expected!s:5s}) -> {text!r}")
        else:
            print(f"  [PASS] {label:42s}: got {actual!s:5s} (expected {expected!s:5s})")

    print(f"Result: {len(cases) - len(failures)}/{len(cases)} passed ({len(failures)} failures).")
    return failures


def probe_redos_catastrophic_backtracking():
    """Stress-test regex scaling in fallback_letter_regex."""
    print("\n" + "=" * 60)
    print("PROBE 5: ReDoS / Catastrophic Backtracking Scaling Audit")
    print("=" * 60)

    times = []
    for n in [2, 5, 8, 10, 12, 14]:
        text = "therefore " + "clearly " * n + "something"
        t0 = time.perf_counter()
        extract_mcq_letter(text)
        dt = time.perf_counter() - t0
        times.append((n, dt))
        print(f"  Adverb count n={n:2d} -> {dt:.6f}s")

    # Ratio of growth between n=10 and n=14
    t10 = times[3][1]
    t14 = times[5][1]
    ratio = t14 / max(1e-6, t10)
    print(f"Growth from n=10 to n=14: {ratio:.1f}x (exponential ~2^(14-10) = 16x)")
    return times


def probe_unicode_and_metrics():
    """Verify Unicode identifier handling and continuous metric edge cases."""
    print("\n" + "=" * 60)
    print("PROBE 6: Unicode Identifiers and Metric Zero-Div Guards")
    print("=" * 60)

    cases = [
        ("α_coef = 10\n", "β_val = 20\n", False, 0.0),
        ("α_coef = 10\n", "α_coef = 20\n", False, 1.0),
        ("变量_1 = 100\n", "变量_2 = 200\n", False, 0.0),
        ("переменная = 1\n", "значение = 2\n", False, 0.0),
        ("x = y + 1\n", "x = y + 1\n", True, 1.0),
        ("+ - * /", "+ - * /", True, 1.0),
        ("+ - * /", "= & | ^", False, 0.0),
        ("", "", True, 1.0),
        ("", "abc = 1", False, 0.0),
    ]

    failures = []
    for pred, ref, exp_em, exp_f1 in cases:
        metrics = completion_metrics(pred, ref)
        if metrics["exact_match"] != exp_em or abs(metrics["regex_identifier_set_f1"] - exp_f1) > 1e-4:
            failures.append((pred, ref, exp_em, metrics["exact_match"], exp_f1, metrics["regex_identifier_set_f1"]))
    print(f"Result: {len(cases) - len(failures)}/{len(cases)} passed.")
    return failures


def probe_prompt_neutrality_scan():
    """Scan all evaluation templates for forbidden prompt leak strings."""
    print("\n" + "=" * 60)
    print("PROBE 7: Prompt Neutrality & Architectural Leak Scan")
    print("=" * 60)

    eval_dir = Path("src/bifocalgemma/evaluation")
    data_dir = Path("src/bifocalgemma/data")
    eval_files = list(eval_dir.glob("*.py")) + [data_dir / "context_tasks.py", data_dir / "context_benchmarks.py"]

    forbidden = [
        "using the encoder context",
        "use the supplied encoder context",
        "encoder: object",
    ]

    leaks = []
    for f in eval_files:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), 1):
            line_lower = line.lower()
            for phrase in forbidden:
                if phrase in line_lower:
                    leaks.append((str(f), line_no, phrase, line.strip()))

    print(f"Scanned {len(eval_files)} files. Leaks detected: {len(leaks)}")
    return leaks


if __name__ == "__main__":
    f1 = verify_gate_round_2_resolved_cases()
    f2 = probe_mcq_delimiters_and_wrappers()
    f3 = probe_mcq_preambles_and_ambiguity()
    f4 = probe_reponiah_negation_coverage()
    times = probe_redos_catastrophic_backtracking()
    f6 = probe_unicode_and_metrics()
    f7 = probe_prompt_neutrality_scan()

    print("\n" + "=" * 60)
    print("EMPIRICAL ADVERSARIAL CHALLENGE SUMMARY")
    print("=" * 60)
    print(f"Probe 1 (Round 2 Resolved Cases):     {'PASS (0 failures)' if not f1 else f'FAIL ({len(f1)} failures)'}")
    print(f"Probe 2 (Delimiters & Markdown):      {'PASS (0 failures)' if not f2 else f'FAIL ({len(f2)} failures)'}")
    print(f"Probe 3 (Preambles & Ambiguity):      {'PASS (0 failures)' if not f3 else f'FAIL ({len(f3)} failures)'}")
    print(f"Probe 4 (RepoNIAH Negation Coverage): {'PASS (0 failures)' if not f4 else f'FAIL ({len(f4)} failures)'}")
    print(f"Probe 5 (ReDoS Scaling Audit):        {'DETECTED (O(2^N) exponential backtracking)' if times[-1][1] > 0.05 else 'PASS'}")
    print(f"Probe 6 (Unicode & Metrics):          {'PASS (0 failures)' if not f6 else f'FAIL ({len(f6)} failures)'}")
    print(f"Probe 7 (Prompt Neutrality Leaks):    {'PASS (0 leaks)' if not f7 else f'FAIL ({len(f7)} leaks)'}")

    total_defects = len(f1) + len(f2) + len(f3) + len(f4) + (1 if times[-1][1] > 0.05 else 0) + len(f6) + len(f7)
    print(f"\nTOTAL DEFECTS SURFACED: {total_defects}")
    if total_defects > 0:
        sys.exit(1)
    sys.exit(0)
