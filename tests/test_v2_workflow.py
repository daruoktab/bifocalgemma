"""One Run trains one model, resumes evaluation, and never repeats completed work."""

import ast
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.models.native import build_native_tiny_model
from bifocalgemma.training import v2
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.stages import atomic_json
from tests.test_offline_sft_training import examples


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    state: dict[str, Any] = {"loads": [], "training_calls": [], "benchmarks": [], "uploads": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(v2, "recover_remote_run", lambda *args, **kwargs: None)

    class Dashboard:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, event, **kwargs):
            pass

    monkeypatch.setattr(v2, "TrainingDashboard", Dashboard)
    monkeypatch.setattr(v2, "RemoteHeartbeat", Dashboard)

    def download(spec, *args):
        state["loads"].append(spec.name)
        return build_native_tiny_model(spec), {"source_identity_verified": True}

    monkeypatch.setattr(v2, "download_model", download)
    monkeypatch.setattr(v2, "fetch_collection", lambda *args, **kwargs: None)

    def prepare(spec, *args, **kwargs):
        rows = [
            replace(row, encoder_ids=row.encoder_ids if spec.encoder else ()) for row in examples()
        ]
        return rows, rows[:2], {"fingerprint": "fixture"}

    monkeypatch.setattr(v2, "prepare_curated_training", prepare)

    def execution(model, *args, **kwargs):
        policy = configure_memory_efficient_training(
            model, {"query_chunk_size": 3, "loss_chunk_size": 2}
        )
        return {"memory_policy": policy, "batch_size": 1, "accumulation": 4}

    monkeypatch.setattr(v2, "select_execution", execution)

    def fit(*args, **kwargs):
        state["training_calls"].append(args[0].spec.name)
        return fit_offline_sft(*args, **kwargs)

    monkeypatch.setattr(v2, "fit_offline_sft", fit)

    def benchmark(model, spec, collection, destination, *args, **kwargs):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        name = Path(collection).name
        state["benchmarks"].append(name)
        journal = destination / "predictions.jsonl"
        completed = journal.read_text().splitlines() if journal.exists() else []
        for index in range(len(completed), 2):
            with journal.open("a") as stream:
                stream.write(json.dumps({"case": index}) + "\n")
            if state.get("interrupt_collection") == name and not state.get("interrupted"):
                state["interrupted"] = True
                raise KeyboardInterrupt("Molab session ended during benchmark")
        report = {"collection": name, "cases": 2}
        atomic_json(destination / "benchmark_report.json", report)
        atomic_json(destination / "evaluation_state.json", {"status": "COMPLETED"})
        return report

    monkeypatch.setattr(v2, "run_context_benchmark", benchmark)

    def synchronize(repo, root, prefix, **kwargs):
        state["uploads"].append(
            (repo, sorted(path.relative_to(root).as_posix() for path in root.rglob("*.json")))
        )
        return {"revision": "b" * 40, "retained_checkpoints": ["checkpoint-000001"]}

    monkeypatch.setattr(v2, "synchronize_v2", synchronize)

    def run(key="e2b-it"):
        return v2.run_training_and_benchmark(
            key, token="test", output_root=tmp_path, cache=tmp_path / "cache"
        )

    yield run, state
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("key", list(v2.MODELS))
def test_one_run_produces_checkpoint_and_all_benchmarks_and_rerun_skips_models(workflow, key):
    run, state = workflow
    result = run(key)
    expected = [item["collection"] for item in v2.BENCHMARKS]
    assert result["status"] == "COMPLETED" and len(result["benchmarks"]) == len(expected)
    assert state["benchmarks"] == expected
    assert len(state["training_calls"]) == 1 and len(state["loads"]) == 2
    repo, inventory = state["uploads"][-1]
    assert repo == result["repo"] == "daruokta/bifocalgemma-v2-" + key
    assert any(path.endswith("/READY.json") for path in inventory)
    assert sum(path.endswith("/benchmark_report.json") for path in inventory) == len(expected)
    again = run(key)
    assert again["status"] == "COMPLETED" and again["benchmarks"] == result["benchmarks"]
    assert len(state["training_calls"]) == 1 and len(state["loads"]) == 2
    assert state["benchmarks"] == expected


def test_run_again_after_benchmark_interruption_keeps_training_and_prediction_journal(workflow):
    run, state = workflow
    interrupted = v2.BENCHMARKS[1]["collection"]
    state["interrupt_collection"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        run()
    assert len(state["training_calls"]) == 1
    result = run()
    assert result["status"] == "COMPLETED" and len(state["training_calls"]) == 1
    assert len(state["loads"]) == 3  # train + interrupted eval + resumed eval
    assert state["benchmarks"].count(v2.BENCHMARKS[0]["collection"]) == 1
    assert state["benchmarks"].count(interrupted) == 2
    assert all(row["cases"] == 2 for row in result["benchmarks"])


def test_training_failure_never_launches_benchmarks(monkeypatch):
    calls = []

    def fail(model_key, *, mode, **kwargs):
        calls.append(mode)
        raise RuntimeError("GPU preflight failed")

    monkeypatch.setattr(v2, "run_v2", fail)
    with pytest.raises(RuntimeError, match="preflight failed"):
        v2.run_training_and_benchmark("e2b-it", token="test")
    assert calls == ["train"]


@pytest.mark.parametrize("from_environment", [False, True])
def test_notebook_starts_with_token_without_any_mode_selection(monkeypatch, from_environment):
    notebook_root = Path(
        os.environ.get(
            "BIFOCALGEMMA_NOTEBOOK_DIR", Path(__file__).resolve().parents[1] / "notebooks"
        )
    )
    path = notebook_root / "train_e2b_it.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cell = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "run_training_and_benchmark"
            for n in ast.walk(node)
        )
    )
    cell.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(path), "exec"), namespace)
    calls = []

    def run(key, *, token, mo, monitor):
        calls.append((key, token))
        return {"model": "fixture", "status": "COMPLETED", "repo": "fixture/repo", "benchmarks": []}

    monkeypatch.setattr(v2, "run_training_and_benchmark", run)
    if from_environment:
        monkeypatch.setenv("HF_TOKEN", "fixture-token")
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)

    def stop(condition, message):
        if condition:
            raise AssertionError(message)

    mo = SimpleNamespace(
        stop=stop,
        md=lambda message: message,
        vstack=lambda items: items,
        ui=SimpleNamespace(table=lambda rows, **kwargs: rows),
    )
    form = SimpleNamespace(value=None if from_environment else {"token": "fixture-token"})
    namespace["_"]("e2b-it", object(), mo, form)
    assert calls == [("e2b-it", "fixture-token")]
