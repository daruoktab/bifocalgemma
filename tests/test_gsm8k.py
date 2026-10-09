"""GSM8K gold isolation, frozen identity and official-style scoring semantics."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.evaluation import gsm8k


def fixture_rows():
    return [{"question": "What is 1 plus 1?", "answer": "1 + 1 = <<1+1=2>>2.\n#### 2"}]


def test_target_is_separate_and_shots_use_only_train():
    rows = fixture_rows()
    public, gold, _ = gsm8k.make_records(rows, [], shots=0)
    modified = copy.deepcopy(rows)
    modified[0]["answer"] = "SECRET WRONG SOLUTION\n#### 3"
    other, other_gold, _ = gsm8k.make_records(modified, [], shots=0)
    assert public == other and gold != other_gold
    assert not {"answer", "target_patch", "solution"}.intersection(public[0])
    train = [{"question": f"DEMO {i}", "answer": f"#### {i}"} for i in range(8)]
    with_shots, _, _ = gsm8k.make_records(rows, train, shots=8)
    assert all(f"DEMO {i}" in with_shots[0]["context"] for i in range(8))
    assert "<<" not in public[0]["context"]
    with pytest.raises(ValueError, match="shots"):
        gsm8k.make_records(rows, [], shots=8)


def test_release_extraction_and_fixed_denominator():
    assert gsm8k.extract_release_answer("steps\n#### -1,234.0") == "-1234.0"
    assert gsm8k.extract_release_answer("#### 2 then #### 3") == "2"
    assert gsm8k.extract_release_answer("the answer is 2") is None
    public, gold, _ = gsm8k.make_records(fixture_rows() * 4, [], shots=0)
    predictions = [
        {
            "instance_id": public[0]["instance_id"],
            "completion": "#### 2",
            "stop_reason": "length_limit",
        },
        {"instance_id": public[1]["instance_id"], "completion": "#### 2.0"},
        {"instance_id": public[2]["instance_id"], "completion": "#### 2", "status": "ERROR"},
    ]
    report = gsm8k.score_gsm8k(public, gold, predictions)
    assert report["total"] == 4 and report["accuracy"] == 0.25
    assert report["counts"] == {
        "correct": 1,
        "incorrect": 1,
        "generation_error": 1,
        "missing_prediction": 1,
        "generation_length_limit": 1,
    }
    assert gsm8k.score_gsm8k(public, gold, [])["accuracy"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        gsm8k.score_gsm8k(public, gold, predictions + predictions[:1])
    with pytest.raises(ValueError, match="unknown"):
        gsm8k.score_gsm8k(public, gold, [{"instance_id": "unknown", "completion": "#### 2"}])
    with pytest.raises(ValueError, match="mixed_experiment"):
        gsm8k.score_gsm8k(
            public,
            gold,
            [
                {**predictions[0], "experiment": "a"},
                {**predictions[1], "experiment": "b"},
            ],
        )


def test_verified_preparation_and_input_only_loading(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    content = json.dumps(fixture_rows()[0]).encode() + b"\n"
    for split in ("test", "train"):
        (source / (split + ".jsonl")).write_bytes(content)
    monkeypatch.setattr(
        gsm8k,
        "SOURCE_FILES",
        {split: (hashlib.sha256(content).hexdigest(), 1) for split in ("train", "test")},
    )
    output = tmp_path / "prepared"
    manifest = gsm8k.prepare_gsm8k(source, output)
    public, loaded = gsm8k.load_gsm8k(output)
    assert loaded == manifest and len(public) == 1
    (output / "hidden_gold.jsonl").unlink()
    assert gsm8k.load_gsm8k(output)[0] == public  # Generation never reads hidden gold.
    with pytest.raises(FileExistsError):
        gsm8k.prepare_gsm8k(source, output)
    (output / "inputs.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file_integrity"):
        gsm8k.load_gsm8k(output)
    (source / "test.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="source_hash"):
        gsm8k.verified_source_rows(source, "test")


class FakeTokenizer:
    bos_token_id = 2

    def __init__(self):
        self.prompts = []

    def encode(self, text, add_special_tokens):
        return [2] + list(range(3, 3 + len(text)))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.prompts.append(messages[0]["content"])
        return messages[0]["content"]

    def decode(self, ids, **kwargs):
        assert ids == (7,)
        return "#### 2"


@pytest.mark.parametrize("spec", EXPERIMENT_SPECS)
def test_six_generation_paths_cache_assets_and_never_read_gold(spec, monkeypatch):
    import bifocalgemma.data.tokenization as assets
    import bifocalgemma.inference.generation as inference

    loaded = []
    tokenizer = FakeTokenizer()

    def loader(reference, cache):
        loaded.append(reference)
        return tokenizer, {}

    def decode(model, encoder_ids, prefix_ids, **kwargs):
        assert bool(encoder_ids) == bool(spec.encoder)
        assert prefix_ids[0] == 2
        return SimpleNamespace(token_ids=(7, 106), stop_reason="eos")

    monkeypatch.setattr(assets, "load_verified_tokenizer", loader)
    monkeypatch.setattr(inference, "greedy_decode", decode)
    public, _, _ = gsm8k.make_records(fixture_rows() * 2, [], shots=0)
    predictions = list(gsm8k.generate_gsm8k(SimpleNamespace(spec=spec), spec, public, "cache"))
    assert len(predictions) == 2 and all(row["completion"] == "#### 2" for row in predictions)
    assert len(loaded) == (2 if spec.encoder else 1)
    assert all("#### 2" not in prompt for prompt in tokenizer.prompts)
    with pytest.raises(ValueError, match="target_in_input"):
        gsm8k.gsm8k_prompt_ids(spec, {**public[0], "answer": "secret"}, tokenizer, tokenizer)
    if spec.encoder:
        with pytest.raises(ValueError, match="encoder_overflow"):
            gsm8k.gsm8k_prompt_ids(spec, public[0], tokenizer, tokenizer, max_encoder_tokens=1)


def test_bad_generation_budget_rejected_before_loading_weights(tmp_path, monkeypatch):
    import bifocalgemma.data.tokenization as assets
    import bifocalgemma.models.pretrained as pretrained

    spec = EXPERIMENT_SPECS[4]
    public, _, _ = gsm8k.make_records(fixture_rows(), [], shots=0)
    monkeypatch.setattr(gsm8k, "load_gsm8k", lambda path: (public, {}))
    monkeypatch.setattr(assets, "load_verified_tokenizer", lambda ref, cache: (FakeTokenizer(), {}))

    def fail(*args, **kwargs):
        raise AssertionError("Weights must not be loaded for a failed data budget")

    monkeypatch.setattr(pretrained, "load_local_native_model", fail)
    args = SimpleNamespace(
        output=tmp_path / "predictions.jsonl",
        prepared=tmp_path,
        experiment=spec.name,
        tokenizer_cache=tmp_path,
        max_new_tokens=2048,
        decoder_limit=2048,
        encoder_limit=2048,
    )
    with pytest.raises(ValueError, match="before_model_loading"):
        gsm8k.generate_run(args)


def test_cli_rejects_prediction_from_another_prompt_protocol(tmp_path, monkeypatch):
    import sys

    public, gold, _ = gsm8k.make_records(fixture_rows(), [], shots=0)
    monkeypatch.setattr(
        gsm8k, "load_gsm8k", lambda path, include_gold: (public, gold, {"fingerprint": "zero-shot"})
    )
    predictions = tmp_path / "wrong.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "instance_id": public[0]["instance_id"],
                "completion": "#### 2",
                "manifest_fingerprint": "eight-shot",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "score.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsm8k",
            "score",
            "--prepared",
            str(tmp_path),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="protocol_mismatch"):
        gsm8k.main()
    assert not output.exists()
