"""Task boundaries and objective construction must not leak targets or benchmark gold."""

import importlib.util
from pathlib import Path

import pytest

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.context_tasks import normalize_context_task, repair_preimage_coverage
from bifocalgemma.evaluation.context import completion_metrics, prompt_ids, strict_letter


def task():
    return {
        "record_type": "context_sft_v1",
        "role": "train",
        "task_type": "localization",
        "instance_id": "fixture",
        "group": "fixture/repo",
        "instruction": "Find bug",
        "context": "### File: a.py\nx=0",
        "target_text": '["a.py"]',
        "source": "nebius/SWE-rebench-V2",
        "source_key": "swe_rebench_v2",
        "source_revision": "a" * 40,
        "snapshot_sha256": "b" * 64,
        "commit": "c" * 40,
        "repos": "fixture/repo",
        "context_selection": {"gold_used": False, "windows": [{"path": "a.py"}]},
    }


def test_eval_role_and_invisible_localization_target_rejected():
    assert normalize_context_task(task())["target_text"] == '["a.py"]'
    with pytest.raises(ValueError, match="not_training"):
        normalize_context_task({**task(), "role": "eval"})
    with pytest.raises(ValueError, match="not_visible"):
        normalize_context_task({**task(), "target_text": '["hidden.py"]'})
    with pytest.raises(ValueError, match="benchmark_source"):
        normalize_context_task({**task(), "benchmark": "SWE-QA"})


def test_short_answer_scoring_rejects_unrequested_prose():
    assert strict_letter(" B\n") == "B"
    assert strict_letter("B)") == "B"
    assert strict_letter("The answer is B") is None
    assert strict_letter("A or B") is None
    assert completion_metrics("return a + b", "return a+b")["exact_match"]
    assert not completion_metrics("return a-b", "return a+b")["exact_match"]


def test_repair_requires_visible_preimage_without_refilling_context():
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -10,3 +10,3 @@\n a\n-b\n+c\n d\n"
    selection = {
        "gold_used": False,
        "windows": [{"path": "a.py", "start_line": 10, "end_line": 12}],
    }
    assert repair_preimage_coverage(patch, selection)
    selection["windows"][0]["end_line"] = 11
    assert not repair_preimage_coverage(patch, selection)
    selection["windows"][0]["path"] = "unrelated.py"
    assert not repair_preimage_coverage(patch, selection)


def test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline():
    from types import SimpleNamespace

    from bifocalgemma.config import EXPERIMENT_SPECS

    prompts = []
    decoder = SimpleNamespace(
        bos_token_id=2,
        encode=lambda text, **kwargs: [2, 7],
        apply_chat_template=lambda messages, **kwargs: (
            prompts.append(messages[0]["content"]) or "formatted"
        ),
    )
    row = {
        "role": "eval",
        "record_type": "repository_patch_v1",
        "instruction": "Fix the function.",
        "context": "x=0",
    }
    for spec in (EXPERIMENT_SPECS[0], EXPERIMENT_SPECS[4]):
        prompt_ids(row, spec, decoder if spec.encoder else None, decoder, encoder_limit=100)
    assert all(text.startswith("Return only a unified diff") for text in prompts)
    assert row["instruction"] == "Fix the function."


def test_dependency_completion_removes_target_and_retains_real_helper():
    path = Path(__file__).resolve().parents[1] / "tools/prepare_context_data.py"
    spec = importlib.util.spec_from_file_location("context_preparer_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = "from .helper import combine\n\ndef run(x):\n    result = combine(x, 1)\n    adjusted = result * 2 + result * 3\n    return adjusted + result\n"
    snapshot = RepositorySnapshot(
        "fixture/repo",
        "a" * 40,
        (
            RepositoryFile("pkg/owner.py", owner),
            RepositoryFile("pkg/helper.py", "def combine(a, b):\n    return a + b\n"),
            RepositoryFile("broken.py", "x=0\n"),
        ),
    )
    result = module.completion_from_snapshot(
        {
            "target_patch": "diff --git a/broken.py b/broken.py\n--- a/broken.py\n+++ b/broken.py\n@@ -1 +1 @@\n-x=0\n+x=1\n"
        },
        snapshot,
    )
    assert result is not None
    instruction, context, target, selection = result
    assert "pkg/helper.py" in context and "# <MISSING_CODE>" in context
    assert target not in context and target in owner
    assert "combine" in target and selection["gold_used"] is False
    assert "run" in instruction
