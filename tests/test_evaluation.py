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


def test_resilient_mcq_uses_terminal_final_answer_after_questioned_candidate():
    from bifocalgemma.evaluation.context import extract_mcq_letter, strict_letter

    response = "A is correct? No, the final answer is B."
    assert extract_mcq_letter(response) == "B"
    assert strict_letter(response) is None


def test_crosscodelong_normalize_completion_prediction_strips_fences_and_suffix_overlap():
    from bifocalgemma.evaluation.context import (
        completion_metrics,
        normalize_completion_prediction,
        score_context_case,
    )

    fenced_with_suffix = (
        "Here is the completion:\n```python\n    return a + b\n```\n### SUFFIX\n    return c"
    )
    assert normalize_completion_prediction(fenced_with_suffix, suffix="\n    return c") == (
        "    return a + b"
    )
    metrics = completion_metrics(fenced_with_suffix, "    return a + b", suffix="\n    return c")
    assert metrics["exact_match"] is False
    assert metrics["normalized_exact_match"] is True
    assert metrics["normalized_sequence_similarity"] == 1.0
    assert metrics["normalized_regex_f1"] == 1.0

    inside_fence = "```python\n### MISSING CODE\n    return a + b\n### SUFFIX\n    return c\n```"
    assert normalize_completion_prediction(inside_fence, suffix="\n    return c") == (
        "    return a + b"
    )
    assert normalize_completion_prediction("Sure:\n```python\n    x = 1\n```\nDone.") == "    x = 1"
    assert normalize_completion_prediction("Sure:\n```python\n    x = 1") == "    x = 1"

    echoed_suffix = "    values = [1, 2]\n    total = sum(values)"
    suffix = "\n    total = sum(values)\n    return total"
    assert normalize_completion_prediction(echoed_suffix, suffix=suffix) == "    values = [1, 2]"
    overlap_metrics = completion_metrics(echoed_suffix, "    values = [1, 2]", suffix=suffix)
    assert overlap_metrics["exact_match"] is False
    assert overlap_metrics["normalized_exact_match"] is True

    docstring_code = 'def helper():\n    """Example ```inline``` snippet."""\n    return 1'
    assert normalize_completion_prediction(docstring_code) == docstring_code

    case_score = score_context_case(
        {"task_type": "completion", "context": "    x = 1\n### SUFFIX\n    return c"},
        {"reference": "    return a + b"},
        fenced_with_suffix,
        directory=None,
        python_executable=None,
        allow_execution=False,
    )
    assert case_score["exact_match"] is False
    assert case_score["normalized_exact_match"] is True


def test_decoder_prompt_baseline_zero_duplication_and_paired_gsm8k_parity():
    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.evaluation.context import decoder_prompt
    from bifocalgemma.evaluation.gsm8k import gsm8k_prompt_ids

    sweqa_instruction = "Answer the code comprehension question.\nWhich function?"
    sweqa_context = sweqa_instruction + "\n\ndef f(): return 1"
    assert (
        decoder_prompt(
            {"task_type": "mcq", "instruction": sweqa_instruction, "context": sweqa_context},
            paired=False,
        )
        == sweqa_context
    )

    longcode_instruction = "Answer about repo.\nQuestion:\nWhat does run() return?"
    longcode_context = (
        "Answer about repo.\n\ndef run(): return 7\n\nQuestion:\nWhat does run() return?"
    )
    assert (
        decoder_prompt(
            {"task_type": "mcq", "instruction": longcode_instruction, "context": longcode_context},
            paired=False,
        )
        == longcode_context
    )

    repair_row = {
        "task_type": "repair",
        "instruction": "Fix bug.",
        "context": "### Issue\nFix bug.\n\n### File: a.py\nx=0",
        "context_selection": {"gold_used": False, "windows": [{"path": "a.py"}]},
    }
    unpaired_repair = decoder_prompt(repair_row, paired=False)
    assert unpaired_repair.count("Fix bug.") == 1
    paired_repair = decoder_prompt(repair_row, paired=True)
    assert "Target file: a.py\n\nFix bug." in paired_repair

    gsm_row = {
        "role": "eval",
        "benchmark": "gsm8k",
        "instruction": "Solve step by step. End with ####.",
        "question": "What is 2 + 3?",
        "context": "Q: demo\nA: #### 1\n\nQ: What is 2 + 3?",
    }
    paired_gsm = decoder_prompt(gsm_row, paired=True)
    assert paired_gsm == "Solve step by step. End with ####.\n\nQuestion: What is 2 + 3?"

    seen_paired_prompts: list[str] = []

    class SpyTokenizer:
        bos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            seen_paired_prompts.append(messages[0]["content"])
            return messages[0]["content"]

        def encode(self, text, add_special_tokens=False):
            return [2, 5]

    gsm8k_prompt_ids(EXPERIMENT_SPECS[0], gsm_row, SpyTokenizer(), SpyTokenizer())
    assert seen_paired_prompts == [paired_gsm]


def test_prompt_ids_instance_id_isolation_and_gold_rejection():
    import pytest

    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.evaluation.context import prompt_ids

    seen_texts: list[str] = []

    class SpyTokenizer:
        bos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            seen_texts.append(content)
            return content

        def encode(self, text, add_special_tokens=False):
            seen_texts.append(text)
            return [2, 10, 11]

    row = {
        "role": "eval",
        "instance_id": "SECRET_INSTANCE_ID_999",
        "task_type": "completion",
        "instruction": "Complete the body",
        "context": "def f():\n    pass",
    }
    spy = SpyTokenizer()
    for spec in (EXPERIMENT_SPECS[0], EXPERIMENT_SPECS[4]):
        prompt_ids(row, spec, spy if spec.encoder else None, spy, encoder_limit=256)
    assert seen_texts and all("SECRET_INSTANCE_ID_999" not in t for t in seen_texts)

    for bad in (
        {**row, "canonical_solution": "return 1"},
        {**row, "gold_answer": "B"},
        {**row, "context_selection": {"gold_used": True, "windows": []}},
    ):
        with pytest.raises(ValueError, match="benchmark_gold_in_model_input"):
            prompt_ids(bad, EXPERIMENT_SPECS[0], spy, spy, encoder_limit=256)


if __name__ == "__main__":
    unittest.main()
