"""Regression tests for evaluation protocol, overflow fallback, and external runner fixes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bifocalgemma.config import make_experiment_spec
from bifocalgemma.evaluation.benchmark import run_batched_context_benchmark
from bifocalgemma.evaluation.context import (
    decoder_prompt,
    run_context_benchmark,
    split_context_for_mode,
)
from bifocalgemma.evaluation.external_runner import run_external_benchmark
from bifocalgemma.evaluation.protocol import build_effective_benchmark_spec, repair_report_metrics


class _CharTokenizer:
    """Minimal deterministic tokenizer whose token count scales with character count."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "plain"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kwargs):
        text = messages[0]["content"]
        return text if not tokenize else self.encode(text, add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [10 + (ord(ch) % 200) for ch in text]
        if add_special_tokens:
            return [self.bos_token_id, *ids]
        return [self.bos_token_id, *ids]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return "A"


class _DummyEvalModel(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.anchor = torch.nn.Parameter(torch.zeros(1))


def test_number_lines_false_preserves_genuine_digit_colon_source_lines() -> None:
    row = {
        "instance_id": "row-1",
        "task_type": "qa",
        "instruction": "What does the config say?",
        "context": (
            "1: preamble item\n\n"
            "### File: app.py\n"
            "def f():\n"
            "1: genuine source comment\n"
            "42: another literal line\n"
        ),
        "role": "eval",
    }
    dec_numbered, _ = split_context_for_mode(row, context_mode="full", number_lines=True)
    dec_unnumbered, _ = split_context_for_mode(row, context_mode="full", number_lines=False)

    assert "1: def f():" in dec_numbered
    assert "2: 1: genuine source comment" in dec_numbered
    assert "1: preamble item" in dec_unnumbered
    assert "def f():" in dec_unnumbered
    assert "1: genuine source comment" in dec_unnumbered
    assert "42: another literal line" in dec_unnumbered
    assert "2: 1: genuine source comment" not in dec_unnumbered


def test_recount_functional_pass_rate_returns_none_on_partial_execution() -> None:
    mixed_records = [
        {"patch_applied": True, "recount_patch_applied": True, "recount_functional_pass": None},
        {"patch_applied": False, "recount_patch_applied": False, "recount_functional_pass": False},
    ]
    metrics = repair_report_metrics(mixed_records)
    assert metrics["recount_patch_apply_rate"] == 0.5
    assert metrics["recount_functional_pass_rate"] is None

    full_records = [
        {"patch_applied": True, "recount_patch_applied": True, "recount_functional_pass": True},
        {"patch_applied": False, "recount_patch_applied": False, "recount_functional_pass": False},
    ]
    full_metrics = repair_report_metrics(full_records)
    assert full_metrics["recount_functional_pass_rate"] == 0.5


def test_effective_spec_includes_overflow_fallback_policy() -> None:
    spec = make_experiment_spec("e2b-it")
    model = _DummyEvalModel(spec)
    public = [
        {"instance_id": "c1", "role": "eval", "task_type": "qa", "instruction": "Q", "context": "C"}
    ]
    gold = [{"instance_id": "c1", "answer": "A"}]
    eff = build_effective_benchmark_spec(
        spec=spec,
        model=model,
        public_rows=public,
        gold_rows=gold,
        collection="longcodeqa",
        encoder_limit=64,
        decoder_limit=64,
        suite_cap=16,
        use_cache=False,
        allow_execution=False,
    )
    assert eff["generation"]["line_number_overflow_fallback"] == "unnumbered_on_overflow"


def _write_mcq_benchmark(directory: Path, row: dict, gold: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (directory / "hidden_gold.jsonl").write_text(json.dumps(gold) + "\n", encoding="utf-8")


def test_overflow_fallback_recomputes_budget_and_records_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    import bifocalgemma.data.tokenization as tok_mod
    import bifocalgemma.evaluation.benchmark as bench_mod
    import bifocalgemma.evaluation.context as ctx_mod

    spec = make_experiment_spec("e2b-it")
    model = _DummyEvalModel(spec)
    tok = _CharTokenizer()

    lines = "\n".join("# <MISSING_CODE>" if i == 300 else f"x{i}" for i in range(600))
    row = {
        "instance_id": "comp-1",
        "role": "eval",
        "task_type": "completion",
        "benchmark": "CrossCodeLongEval",
        "instruction": "Complete.",
        "context": f"### File: a.py\n{lines}",
    }
    gold = {"instance_id": "comp-1", "reference": "x300"}

    numbered_len = len(
        tok.encode(split_context_for_mode(row, context_mode="full", number_lines=True)[0])
    )
    unnumbered_len = len(
        tok.encode(split_context_for_mode(row, context_mode="full", number_lines=False)[0])
    )
    paired_numbered_len = len(
        tok.encode(decoder_prompt(row, paired=True, number_lines=True), add_special_tokens=False)
    )
    paired_unnumbered_len = len(
        tok.encode(decoder_prompt(row, paired=True, number_lines=False), add_special_tokens=False)
    )
    assert numbered_len - unnumbered_len > 2500
    assert paired_numbered_len > paired_unnumbered_len

    decoder_limit = paired_numbered_len + 50
    cap = 2500
    suite_cap = 2000
    expected_new_tokens = min(suite_cap, decoder_limit - paired_unnumbered_len)

    monkeypatch.setattr(tok_mod, "load_verified_tokenizer", lambda *_a, **_k: (tok, {}))
    monkeypatch.setattr(
        ctx_mod,
        "greedy_decode",
        lambda *_a, **_k: SimpleNamespace(token_ids=(65,), stop_reason="eos"),
    )
    monkeypatch.setattr(
        bench_mod,
        "batched_greedy_decode",
        lambda *_a, **_k: [SimpleNamespace(token_ids=(65,), stop_reason="eos")],
    )

    bench_dir = tmp_path / "longcodeqa"
    _write_mcq_benchmark(bench_dir, row, gold)

    seq_out = tmp_path / "seq_out"
    seq_report = run_context_benchmark(
        model,
        spec,
        bench_dir,
        seq_out,
        tokenizer_cache=tmp_path,
        checkpoint_identity={"id": "ckpt-test"},
        encoder_limit=cap,
        decoder_limit=decoder_limit,
        max_new_tokens=suite_cap,
        use_cache=False,
    )
    seq_rec = json.loads(
        (seq_out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert seq_rec["number_lines"] is False
    assert seq_rec["overflow_unnumbered_fallback"] is True
    assert seq_rec["paired_decoder_prefix_tokens"] == paired_unnumbered_len
    assert seq_rec["generation_max_new_tokens"] == expected_new_tokens
    assert seq_report["unnumbered_fallback_count"] == 1
    assert "unnumbered line-prefix fallback on overflow" in seq_report["protocol"]

    batched_out = tmp_path / "batched_out"
    batched_report = run_batched_context_benchmark(
        model,
        spec,
        bench_dir,
        batched_out,
        tokenizer_cache=tmp_path,
        checkpoint_identity={"id": "ckpt-test"},
        encoder_limit=cap,
        decoder_limit=decoder_limit,
        max_new_tokens=suite_cap,
        use_cache=False,
    )
    batched_rec = json.loads(
        (batched_out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert batched_rec["number_lines"] is False
    assert batched_rec["overflow_unnumbered_fallback"] is True
    assert batched_rec["paired_decoder_prefix_tokens"] == paired_unnumbered_len
    assert batched_rec["generation_max_new_tokens"] == expected_new_tokens
    assert batched_report["unnumbered_fallback_count"] == 1


def test_external_runner_resume_and_signature_guard(tmp_path: Path, monkeypatch) -> None:
    import bifocalgemma.evaluation.external_runner as ext_mod

    bench_dir = tmp_path / "benchmarks" / "longcodeqa"
    row1 = {
        "instance_id": "ext-1",
        "role": "eval",
        "task_type": "mcq",
        "benchmark": "LongCodeQA",
        "instruction": "Pick A.",
        "context": "### File: a.py\nx = 1\n",
    }
    row2 = {
        "instance_id": "ext-2",
        "role": "eval",
        "task_type": "mcq",
        "benchmark": "LongCodeQA",
        "instruction": "Pick A.",
        "context": "### File: b.py\ny = 2\n",
    }
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "inputs.jsonl").write_text(
        json.dumps(row1) + "\n" + json.dumps(row2) + "\n", encoding="utf-8"
    )
    (bench_dir / "hidden_gold.jsonl").write_text(
        json.dumps({"instance_id": "ext-1", "answer": "A", "options": ["A", "B"]})
        + "\n"
        + json.dumps({"instance_id": "ext-2", "answer": "A", "options": ["A", "B"]})
        + "\n",
        encoding="utf-8",
    )

    tok = _CharTokenizer()
    dummy_model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(
        ext_mod,
        "resolve_external_model_spec",
        lambda _k: {
            "key": "ext-test",
            "model_id": "org/ext-test",
            "revision": "rev1",
            "max_position_embeddings": 4096,
            "enable_thinking": False,
        },
    )
    monkeypatch.setattr(
        ext_mod,
        "generate_external_batch",
        lambda _m, _t, prompts, **_k: [((65,), "eos", "A") for _ in prompts],
    )

    out_dir = tmp_path / "ext_out"
    item = {"collection": "longcodeqa", "max_new_tokens": 16, "decoder_limit": 512}
    report1 = run_external_benchmark(
        "ext-test",
        item,
        tmp_path / "benchmarks",
        out_dir,
        model=dummy_model,
        tokenizer=tok,
    )
    assert report1["total"] == 2
    assert (out_dir / "evaluation_state.json").exists()

    # Simulate partial crash: keep only first line of predictions.jsonl and resume.
    lines = (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    (out_dir / "predictions.jsonl").write_text(lines[0] + "\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        ext_mod,
        "generate_external_batch",
        lambda _m, _t, prompts, **_k: (
            calls.append(len(prompts)),
            [((65,), "eos", "A") for _ in prompts],
        )[1],
    )
    report2 = run_external_benchmark(
        "ext-test",
        item,
        tmp_path / "benchmarks",
        out_dir,
        model=dummy_model,
        tokenizer=tok,
    )
    assert calls == [1]
    assert report2["total"] == 2

    # Changing protocol (e.g. context_mode) must fail instead of silently overwriting.
    with pytest.raises(ValueError, match="External benchmark/model/protocol changed"):
        run_external_benchmark(
            "ext-test",
            item,
            tmp_path / "benchmarks",
            out_dir,
            context_mode="local_only",
            model=dummy_model,
            tokenizer=tok,
        )
