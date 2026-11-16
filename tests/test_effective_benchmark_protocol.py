"""Executed budgets and semantic signatures reject incompatible result reuse."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from bifocalgemma.config import BENCHMARKS, MATRIX_MODELS, make_experiment_spec
from bifocalgemma.evaluation.context import score_context_case
from bifocalgemma.evaluation.protocol import (
    build_effective_benchmark_spec,
    effective_generation_budget,
    mcq_report_metrics,
    scorer_fingerprint,
)


@pytest.mark.parametrize(
    "suite,row,prefix,expected",
    [
        (32, 8192, 100, 32),
        (2048, 8192, 100, 2048),
        (8192, None, 100, 8092),
        (2048, 32, 100, 32),
        (8192, 8192, 8191, 1),
    ],
)
def test_effective_allowance(suite, row, prefix, expected):
    assert effective_generation_budget(suite, row_cap=row, paired_prefix_tokens=prefix) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"suite_cap": 0},
        {"suite_cap": True},
        {"suite_cap": 32, "row_cap": -1},
        {"suite_cap": 32, "row_cap": 32.0},
        {"suite_cap": 32, "paired_prefix_tokens": -1},
        {"suite_cap": 32, "paired_prefix_tokens": 8192},
    ],
)
def test_invalid_allowance_fails_explicitly(kwargs):
    with pytest.raises(ValueError, match="benchmark_"):
        effective_generation_budget(**{"paired_prefix_tokens": 100, **kwargs})


def test_suite_caps_and_ten_model_allowances():
    # Shared paired chat prefix reserves the same output allowance for all ten.
    for item in BENCHMARKS:
        if "qa_" in item["collection"] or "codeqa_" in item["collection"]:
            assert item["max_new_tokens"] == 256
        budgets = [
            effective_generation_budget(
                item["max_new_tokens"], row_cap=8192, paired_prefix_tokens=119
            )
            for _ in MATRIX_MODELS
        ]
        assert len(set(budgets)) == 1


@pytest.mark.parametrize(
    "prediction,strict,resilient",
    [
        ("B", True, True),
        ("(b).", True, True),
        ("The answer is B because the code returns zero.", False, True),
        ("A or B", False, False),
        ("", False, False),
    ],
)
def test_strict_primary_and_resilient_diagnostic_share_prediction(prediction, strict, resilient):
    score = score_context_case(
        {"task_type": "mcq"},
        {"answer": "B"},
        prediction,
        directory=None,
        python_executable=None,
        allow_execution=False,
    )
    assert score["correct"] is strict
    assert score["strict_correct"] is strict
    assert score["resilient_correct"] is resilient
    report = mcq_report_metrics([score])
    assert report["primary_metric"] == "mcq_strict_accuracy"
    assert report["strict_accuracy"] == float(strict)
    assert report["resilient_accuracy"] == float(resilient)


def test_effective_spec_binds_gold_caps_and_runtime_spec():
    spec = make_experiment_spec("e2b-it")
    kwargs: dict[str, Any] = dict(
        spec=spec,
        model=SimpleNamespace(spec=spec),
        public_rows=[{"instance_id": "x", "max_new_tokens": 32}],
        gold_rows=[{"instance_id": "x", "answer": "B"}],
        collection="fixture",
        encoder_limit=32768,
        decoder_limit=8192,
        suite_cap=32,
        use_cache=True,
        allow_execution=False,
    )
    first = build_effective_benchmark_spec(**kwargs)
    assert first["scoring"]["fingerprint"] == scorer_fingerprint()
    changed: dict[str, Any] = {**kwargs, "suite_cap": 2048}
    assert build_effective_benchmark_spec(**changed) != first
    changed = {**kwargs, "gold_rows": [{"instance_id": "x", "answer": "C"}]}
    assert build_effective_benchmark_spec(**changed) != first
    changed = {**kwargs, "model": SimpleNamespace(spec=make_experiment_spec("e4b-it"))}
    with pytest.raises(ValueError, match="runtime_model_spec_mismatch"):
        build_effective_benchmark_spec(**changed)


def test_all_ten_batched_runners_execute_shared_allowance_and_reject_changed_gold(
    tmp_path, monkeypatch
):
    import bifocalgemma.data.tokenization as tokenization
    import bifocalgemma.evaluation.benchmark as benchmark
    from bifocalgemma.inference.generation import GreedyGeneration

    class Tokenizer:
        bos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            return [2, 5, 6] if add_special_tokens else [2, 7]

        def apply_chat_template(self, *args, **kwargs):
            return "chat-prefix"

        def decode(self, *args, **kwargs):
            return "The answer is B because of the code."

    observed = []

    def decode(model, enc, pref, *, max_new_tokens_list, **kwargs):
        observed.extend(max_new_tokens_list)
        return [
            GreedyGeneration((4,), "eos", len(e), len(p)) for e, p in zip(enc, pref, strict=True)
        ]

    monkeypatch.setattr(tokenization, "load_verified_tokenizer", lambda *a: (Tokenizer(), {}))
    monkeypatch.setattr(benchmark, "batched_greedy_decode", decode)
    root = tmp_path / "collection"
    (root / "eval_private").mkdir(parents=True)
    (root / "eval_inputs.jsonl").write_text(
        json.dumps(
            {
                "role": "eval",
                "instance_id": "case",
                "task_type": "mcq",
                "instruction": "Which option?",
                "context": "Source context",
                "max_new_tokens": 8192,
            }
        )
        + "\n"
    )
    gold_path = root / "eval_private/specs.jsonl"
    gold_path.write_text(json.dumps({"instance_id": "case", "answer": "B"}) + "\n")
    kwargs: dict[str, Any] = dict(
        checkpoint_identity={"sha": "fixture"},
        batch_size=2,
        encoder_limit=32768,
        decoder_limit=8192,
        max_new_tokens=32,
        allow_execution=False,
    )
    for key in MATRIX_MODELS:
        kwargs["model_key"] = key
        report = benchmark.run_batched_context_benchmark(
            torch.nn.Linear(1, 1),
            make_experiment_spec(key),
            root,
            tmp_path / key,
            tmp_path,
            **kwargs,
        )
        assert report["strict_accuracy"] == report["accuracy"] == 0
        assert report["resilient_accuracy"] == 1
        record = json.loads((tmp_path / key / "predictions.jsonl").read_text())
        assert record["generation_max_new_tokens"] == 32
        assert record["paired_decoder_prefix_tokens"] == 2
    assert observed == [32] * 10
    state = tmp_path / key / "evaluation_state.json"
    saved_state = state.read_bytes()
    journal = tmp_path / key / "predictions.jsonl"
    saved_journal = journal.read_bytes()
    state.unlink()
    with pytest.raises(ValueError, match="no protocol identity"):
        benchmark.run_batched_context_benchmark(
            torch.nn.Linear(1, 1), make_experiment_spec(key), root, tmp_path / key, tmp_path, **kwargs
        )
    assert not state.exists() and journal.read_bytes() == saved_journal
    state.write_bytes(saved_state)
    record = json.loads(saved_journal)
    del record["protocol_signature"]
    journal.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="journal protocol signature mismatch"):
        benchmark.run_batched_context_benchmark(
            torch.nn.Linear(1, 1), make_experiment_spec(key), root, tmp_path / key, tmp_path, **kwargs
        )
    journal.write_bytes(saved_journal)
    gold_path.write_text(json.dumps({"instance_id": "case", "answer": "C"}) + "\n")
    with pytest.raises(ValueError, match="protocol changed"):
        benchmark.run_batched_context_benchmark(
            torch.nn.Linear(1, 1),
            make_experiment_spec(key),
            root,
            tmp_path / key,
            tmp_path,
            **kwargs,
        )


def test_prepare_sampled_benchmark_collection_invalidates_stale_cache(tmp_path, monkeypatch):
    import bifocalgemma.evaluation.benchmark as benchmark

    src_dir = tmp_path / "source_collection"
    (src_dir / "eval_private").mkdir(parents=True)
    (src_dir / "snapshots").mkdir()
    (src_dir / "snapshots/repo.txt").write_text("snapshot-v1\n", encoding="utf-8")

    public_rows = [
        {
            "instance_id": f"case-{i}",
            "role": "eval",
            "task_type": "mcq",
            "instruction": "Choose A or B.",
            "context": f"public-v1-{i}",
        }
        for i in range(5)
    ]
    gold_rows = [
        {"instance_id": f"case-{i}", "answer": "A", "snapshot_file": "snapshots/repo.txt"}
        for i in range(5)
    ]

    def write_source():
        (src_dir / "eval_inputs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in public_rows), encoding="utf-8"
        )
        (src_dir / "eval_private/specs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in gold_rows), encoding="utf-8"
        )

    write_source()

    writes = 0
    orig_atomic_json = benchmark.atomic_json

    def counted_atomic_json(path, payload):
        nonlocal writes
        writes += 1
        return orig_atomic_json(path, payload)

    monkeypatch.setattr(benchmark, "atomic_json", counted_atomic_json)
    dest_dir = tmp_path / "sampled"

    resolved = benchmark.prepare_sampled_benchmark_collection(
        src_dir, sample_limit=3, sample_seed=17, output_dir=dest_dir
    )
    assert resolved == dest_dir.resolve()
    assert writes == 1
    manifest_1 = json.loads((dest_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    sampled_ids = manifest_1["sampled_instance_ids"]
    assert len(sampled_ids) == 3

    # Unchanged source hits cache cleanly.
    benchmark.prepare_sampled_benchmark_collection(
        src_dir, sample_limit=3, sample_seed=17, output_dir=dest_dir
    )
    assert writes == 1

    # Public content change with identical sampled_instance_ids invalidates cache.
    sampled_idx = int(sampled_ids[0].split("-")[1])
    public_rows[sampled_idx]["context"] = "public-v2-mutated"
    write_source()
    benchmark.prepare_sampled_benchmark_collection(
        src_dir, sample_limit=3, sample_seed=17, output_dir=dest_dir
    )
    assert writes == 2
    manifest_2 = json.loads((dest_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest_2["sampled_instance_ids"] == sampled_ids
    assert manifest_2["inputs_sha256"] != manifest_1["inputs_sha256"]
    assert manifest_2["gold_sha256"] == manifest_1["gold_sha256"]
    cached_public = [
        json.loads(line)
        for line in (dest_dir / "eval_inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert cached_public[0]["context"] == "public-v2-mutated"

    # Gold content change with identical sampled_instance_ids invalidates cache.
    gold_rows[sampled_idx]["answer"] = "B"
    write_source()
    benchmark.prepare_sampled_benchmark_collection(
        src_dir, sample_limit=3, sample_seed=17, output_dir=dest_dir
    )
    assert writes == 3
    manifest_3 = json.loads((dest_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest_3["sampled_instance_ids"] == sampled_ids
    assert manifest_3["gold_sha256"] != manifest_2["gold_sha256"]
    cached_gold = [
        json.loads(line)
        for line in (dest_dir / "eval_private/specs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert cached_gold[0]["answer"] == "B"

    # On-disk tamper of cached JSONL without touching manifest also forces refresh.
    (dest_dir / "eval_inputs.jsonl").write_text('{"instance_id":"tampered"}\n', encoding="utf-8")
    benchmark.prepare_sampled_benchmark_collection(
        src_dir, sample_limit=3, sample_seed=17, output_dir=dest_dir
    )
    assert writes == 4
    refreshed_public = [
        json.loads(line)
        for line in (dest_dir / "eval_inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert refreshed_public[0]["context"] == "public-v2-mutated"
