"""Tier 4: Real-World Application Scenarios.

Tests complete end-to-end workflows: mock evaluation pipelines and paper document validation.
"""

import json
from pathlib import Path

import pytest

from tests_e2e.conftest import (
    reference_resilient_mcq_parser,
    validate_paper_manuscript,
)


# ==============================================================================
# Scenario 1: End-to-End Mock Evaluation Pipeline Run
# ==============================================================================
@pytest.mark.tier4
def test_scenario_end_to_end_mock_evaluation_pipeline(tmp_path: Path):
    """Simulate a complete evaluation session: prompt rendering, generation, scoring, and summary."""
    from bifocalgemma.evaluation.context import completion_metrics, decoder_prompt

    # Step 1: Ingest mock evaluation cohort
    mock_dataset = [
        {
            "instance_id": "eval-mcq-001",
            "task_type": "mcq",
            "benchmark": "SWE-QA",
            "instruction": "Answer the code comprehension question. Return only A, B, C, or D.\n"
            "Which class implements DecoupledCrossAttentionBridge?\nA) GatedCrossAttentionBridge\nB) SimpleBridge\nC) SwiGLU\nD) LayerNorm",
            "context": "class GatedCrossAttentionBridge:\n    pass\n",
            "role": "eval",
            "gold": {"answer": "A"},
        },
        {
            "instance_id": "eval-comp-001",
            "task_type": "completion",
            "benchmark": "CrossCodeLongEval",
            "instruction": "Complete the missing Python code.",
            "context": "def is_even(n):\n",
            "role": "eval",
            "gold": {"reference": "    return n % 2 == 0\n"},
        },
    ]

    # Step 2: Render prompts for paired model
    rendered_prompts = []
    for row in mock_dataset:
        prompt = decoder_prompt(row, paired=True)
        # Verify neutral prompt rule
        assert "Use the supplied encoder context." not in prompt
        rendered_prompts.append((row, prompt))

    # Step 3: Simulate model generation outputs
    simulated_outputs = {
        "eval-mcq-001": "Based on the class declaration, the correct answer is (A).",
        "eval-comp-001": "    return n % 2 == 0\n",
    }

    # Step 4: Scoring and journal recording
    journal = []
    for row, prompt in rendered_prompts:
        inst_id = row["instance_id"]
        pred_text = simulated_outputs[inst_id]

        if row["task_type"] == "mcq":
            parsed_letter = reference_resilient_mcq_parser(pred_text)
            is_correct = parsed_letter == row["gold"]["answer"]
            score_entry = {
                "instance_id": inst_id,
                "parsed": parsed_letter,
                "correct": is_correct,
                "status": "SCORED" if parsed_letter else "INVALID_OUTPUT",
            }
        elif row["task_type"] == "completion":
            metrics = completion_metrics(pred_text, row["gold"]["reference"])
            score_entry = {
                "instance_id": inst_id,
                **metrics,
                "correct": metrics["exact_match"],
                "status": "SCORED",
            }
        journal.append(score_entry)

    # Step 5: Verify journal and aggregate metrics
    assert len(journal) == 2
    assert journal[0]["correct"] is True
    assert journal[0]["status"] == "SCORED"
    assert journal[1]["correct"] is True
    assert journal[1]["sequence_similarity"] == 1.0

    # Step 6: Persist journal and verify structure
    journal_path = tmp_path / "evaluation_journal.json"
    journal_path.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    assert journal_path.exists()


# ==============================================================================
# Scenario 2: Scientific Paper Document Validation
# ==============================================================================
@pytest.mark.tier4
def test_scenario_scientific_paper_document_validation(project_root: Path):
    """End-to-end validation of the complete scientific paper manuscript."""
    draft_file = project_root / "paper/PAPER_DRAFT.md"
    tex_file = project_root / "paper/main.tex"

    # If paper has not yet been authored by M3 track, test a reference paper fixture
    if not draft_file.exists() and not tex_file.exists():
        reference_fixture = """# BifocalGemma: Resilient Repository-Level Code Intelligence
## Abstract
We present BifocalGemma coupling a 32k bidirectional encoder with an 8k causal decoder...
## 1. Introduction
Repository-scale code intelligence poses fundamental challenges...
## 2. BifocalGemma Architecture
The model couples an encoder and causal decoder via SwiGLUDualGateBridge...
## 3. Experimental Setup
We evaluate on SWE-QA, LongCodeQA, and CrossCodeLong...
## 4. Empirical Results
On SWEQA Noisy Oracle, BifocalGemma achieves 73.69% vs 71.87% (+1.82 pp win)...
## 5. Methodological Autopsy & Limitations
We dissect the attention dilution penalty (-10.28 pp) and HumanEvalFix pathologies...
## 6. Conclusion
BifocalGemma proves that decoupled bidirectional encoding provides noise resilience...
"""
        results = validate_paper_manuscript(reference_fixture)
        assert all(results.values()), f"Reference fixture failed validation: {results}"
        pytest.skip("paper/ directory not yet populated; reference paper fixture validated.")
    else:
        target = draft_file if draft_file.exists() else tex_file
        content = target.read_text(encoding="utf-8")
        results = validate_paper_manuscript(content)
        missing = [sec for sec, ok in results.items() if not ok]
        assert not missing, f"Paper manuscript at {target} missing sections: {missing}"


# ==============================================================================
# Scenario 3: Training Compatibility Preflight Verification Flow
# ==============================================================================
@pytest.mark.tier4
def test_scenario_training_compatibility_preflight(project_root: Path):
    """Verify that training resume compatibility file is valid JSON and contains required keys."""
    config_file = project_root / "configs/training_resume_compatibility.json"
    assert config_file.exists(), f"Missing config file: {config_file}"

    data = json.loads(config_file.read_text(encoding="utf-8"))
    required_keys = [
        "version",
        "reason",
        "baseline_runtime_sha256",
        "baseline_training_semantics_sha256",
    ]
    for key in required_keys:
        assert key in data, f"Missing required configuration key: {key}"

    # Verify sha256 formatting
    for hash_key in ["baseline_runtime_sha256", "baseline_training_semantics_sha256"]:
        val = data[hash_key]
        assert len(val) == 64, f"Hash key '{hash_key}' must be 64-char hex string, got '{val}'"
