"""Actual repair prefixes must fit equally for all six model architectures."""

import json

import pytest
import torch

from bifocalgemma.config import get_v2_experiment
from bifocalgemma.evaluation.context import decoder_prompt, run_context_benchmark
from bifocalgemma.training.v2 import MODELS


class TextTokenizer:
    bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [2] + [7] * len(text)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "USER:" + messages[0]["content"] + ":MODEL"

    def decode(self, ids, **kwargs):
        return "fixture"


@pytest.mark.parametrize("key", list(MODELS))
@pytest.mark.parametrize("task", ["repair", "completion", "mcq"])
def test_shared_budget_uses_full_actual_prompt_without_architecture_specific_truncation(
    tmp_path, monkeypatch, key, task
):
    import bifocalgemma.data.tokenization as tokenization
    import bifocalgemma.evaluation.context as context

    tokenizer = TextTokenizer()
    monkeypatch.setattr(tokenization, "load_verified_tokenizer", lambda *a: (tokenizer, {}))
    row = {
        "instance_id": "test",
        "role": "eval",
        "task_type": task,
        "context": "source code",
        "instruction": "Fix this issue",
        "max_new_tokens": 128,
    }
    root = tmp_path / "data"
    (root / "eval_private").mkdir(parents=True)
    (root / "eval_inputs.jsonl").write_text(json.dumps(row) + "\n")
    (root / "eval_private/specs.jsonl").write_text(json.dumps({"instance_id": "test"}) + "\n")
    limits = []

    def generate(model, enc, prefix, *, max_new_tokens, max_total_decoder_tokens, **kwargs):
        from types import SimpleNamespace

        assert len(prefix) + max_new_tokens <= max_total_decoder_tokens
        expected = 128 - len(
            tokenizer.encode("USER:" + decoder_prompt(row, paired=True) + ":MODEL")
        )
        assert max_new_tokens == expected and expected > 0
        limits.append(max_new_tokens)
        return SimpleNamespace(token_ids=(7, 106), stop_reason="eos")

    monkeypatch.setattr(context, "greedy_decode", generate)
    monkeypatch.setattr(
        context, "score_context_case", lambda *a, **k: {"correct": True, "status": "SCORED"}
    )
    spec = get_v2_experiment(MODELS[key][0])
    report = run_context_benchmark(
        torch.nn.Linear(1, 1),
        spec,
        root,
        tmp_path / "result",
        tmp_path,
        checkpoint_identity={"sha": "unchanged"},
        encoder_limit=1024,
        decoder_limit=128,
        max_new_tokens=128,
        use_cache=True,
    )
    assert report["total"] == 1 and len(limits) == 1


def test_repair_record_type_has_same_prefix_as_explicit_repair_task():
    row = {"instruction": "Fix", "context": "code", "record_type": "repository_patch_v1"}
    assert decoder_prompt(row, paired=True) == decoder_prompt(
        {**row, "task_type": "repair"}, paired=True
    )
