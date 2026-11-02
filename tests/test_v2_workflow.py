"""One Run trains one model, resumes evaluation, and never repeats completed work."""

import ast
import copy
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


def test_benchmark_protocol_update_keeps_the_original_hub_directories():
    original = {
        "humanevalfix_python_v2": "trained-humanevalfix_python_v2-e3c83d573a81",
        "sweqa_oracle_gemma32k_dec8k_v1": "trained-sweqa_oracle_gemma32k_dec8k_v1-70311dbddb7c",
        "sweqa_noisy_oracle_gemma32k_dec8k_v1": "trained-sweqa_noisy_oracle_gemma32k_dec8k_v1-b7791f09e27b",
    }
    for item, settings, _ in v2.benchmark_jobs("trained"):
        path = v2.benchmark_directory_name(settings)
        if item["collection"] in original:
            assert path == original[item["collection"]]
        changed = copy.deepcopy(settings)
        changed["protocol"] = "another-evaluation-protocol"
        assert v2.benchmark_directory_name(changed) == path


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    state: dict[str, Any] = {"loads": [], "training_calls": [], "benchmarks": [], "uploads": []}
    state["profile"] = {
        "batch_size": 1,
        "accumulation": 4,
        "evidence_sha256": "f" * 64,
        "memory_policy": {"query_chunk_size": 3, "loss_chunk_size": 2},
    }
    state["training_configs"] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(v2, "recover_remote_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        v2,
        "resolve_shared_profile",
        lambda *args, **kwargs: state["profile"],
    )

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
        profile = kwargs["execution"]
        policy = configure_memory_efficient_training(model, profile["memory_policy"])
        return {
            "memory_policy": policy,
            "batch_size": profile["batch_size"],
            "accumulation": profile["accumulation"],
        }

    monkeypatch.setattr(v2, "select_execution", execution)

    def fit(*args, **kwargs):
        state["training_calls"].append(args[0].spec.name)
        state["training_configs"].append(
            (kwargs["batch_size"], kwargs["accumulation"], args[0].memory_policy)
        )
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


def test_all_six_models_follow_one_profile_instead_of_tuning_independently(workflow):
    run, state = workflow
    state["profile"].update(
        batch_size=2,
        accumulation=2,
        memory_policy={
            "attention_backend": "tiled_eager_fp32",
            "query_chunk_size": 1024,
            "loss_chunk_size": 128,
            "gradient_checkpointing": False,
        },
    )
    for key in v2.MODELS:
        assert run(key)["status"] == "COMPLETED"
    assert len(state["training_configs"]) == 6
    first = state["training_configs"][0]
    assert all(config == first for config in state["training_configs"])
    assert first[0:2] == (2, 2) and first[2]["gradient_checkpointing"] is False


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


def test_old_benchmarks_are_reset_once_while_completed_training_is_reused(workflow, tmp_path):
    run, state = workflow
    first = run()
    ledger_path = tmp_path / "e2b-it/train/pipeline_state.json"
    ledger = json.loads(ledger_path.read_text())
    training = ledger["stages"]["training"]
    ledger["benchmark_reset_id"] = "previous-protocol"
    atomic_json(ledger_path, ledger)
    before = len(state["benchmarks"])
    second = run()
    assert first["status"] == second["status"] == "COMPLETED"
    assert len(state["training_calls"]) == 1
    assert len(state["benchmarks"]) - before == len(v2.BENCHMARKS)
    assert json.loads(ledger_path.read_text())["stages"]["training"] == training
    run()
    assert len(state["benchmarks"]) == before + len(v2.BENCHMARKS)


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
            "BIFOCALGEMMA_NOTEBOOK_DIR", Path(__file__).resolve().parents[1] / "notebooks/legacy/v2"
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
