"""Committed predictions resume exactly once and never use private answers as prompts."""

import json
from types import SimpleNamespace

import pytest
import torch

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.evaluation.context import prompt_ids, run_context_benchmark


class FixtureTokenizer:
    bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert "PRIVATE_EXPECTED" not in text
        return [2, 7, 8] if add_special_tokens else [2, 9]

    def apply_chat_template(self, *args, **kwargs):
        return "fixture-prefix"

    def decode(self, ids, **kwargs):
        return "B" if tuple(ids) == (4,) else "invalid fixture output"


class FixtureGenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.encoder = object()
        self.decoder = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
        self.encoded = 0

    def encode_context(self, ids, mask):
        self.encoded += 1
        return torch.zeros(1, 1, 1)

    def decode_logits(self, ids, mask, states, encoder_mask):
        logits = torch.zeros(1, 1, 128)
        logits[0, 0, 4 if ids.shape[1] == 2 else 106] = 1
        return logits


def test_prediction_journal_resume_and_checkpoint_mismatch(tmp_path, monkeypatch):
    import bifocalgemma.data.tokenization as tokenization

    monkeypatch.setattr(
        tokenization, "load_verified_tokenizer", lambda *a: (FixtureTokenizer(), {})
    )
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "eval_private").mkdir()
    public = [
        {
            "instance_id": f"case-{i}",
            "role": "eval",
            "task_type": "mcq",
            "context": "public context",
            "instruction": "Choose A or B",
            "max_new_tokens": 2,
        }
        for i in range(2)
    ]
    gold = [
        {"instance_id": f"case-{i}", "answer": "B" if i == 0 else "A", "extra": "PRIVATE_EXPECTED"}
        for i in range(2)
    ]
    for path, rows in (
        (root / "eval_inputs.jsonl", public),
        (root / "eval_private/specs.jsonl", gold),
    ):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    model = FixtureGenerationModel()

    def fail_upload():
        raise RuntimeError("fixture interrupted after prediction commit")

    settings = dict(
        checkpoint_identity={"sha": "first"}, encoder_limit=32, decoder_limit=32, upload_every=1
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run_context_benchmark(
            model,
            EXPERIMENT_SPECS[0],
            root,
            tmp_path / "eval",
            tmp_path,
            prediction_callback=fail_upload,
            **settings,
        )
    assert model.encoded == 1
    report = run_context_benchmark(
        model, EXPERIMENT_SPECS[0], root, tmp_path / "eval", tmp_path, **settings
    )
    assert model.encoded == 2 and report["total"] == 2 and report["accuracy"] == 0.5
    again = run_context_benchmark(
        model, EXPERIMENT_SPECS[0], root, tmp_path / "eval", tmp_path, **settings
    )
    assert model.encoded == 2 and again["total"] == 2
    with pytest.raises(ValueError, match="changed"):
        run_context_benchmark(
            model,
            EXPERIMENT_SPECS[0],
            root,
            tmp_path / "eval",
            tmp_path,
            **{**settings, "checkpoint_identity": {"sha": "second"}},
        )


def test_gold_in_public_row_is_rejected_before_tokenization():
    row = {"role": "eval", "context": "code", "instruction": "question", "answer": "B"}
    with pytest.raises(ValueError, match="gold"):
        prompt_ids(
            row, EXPERIMENT_SPECS[0], FixtureTokenizer(), FixtureTokenizer(), encoder_limit=32
        )
