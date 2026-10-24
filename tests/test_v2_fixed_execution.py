"""Fixed execution, resumable guard evidence, and rollback of probe updates."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch

from bifocalgemma.config import EXPERIMENT_NAMES, get_v2_experiment
from bifocalgemma.models.flex_attention import flex_kernel_options
from bifocalgemma.models.native import build_native_tiny_model
from bifocalgemma.training import v2_preflight as guard
from bifocalgemma.training.policy import configure_policy_model
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.stages import atomic_json
from bifocalgemma.training.telemetry import TrainingDashboard
from bifocalgemma.training.v2_hub import RemoteHeartbeat, synchronize_preflight
from tests.test_offline_sft_training import examples as paired_examples
from tools.notebook_bundle import source_bundle_bytes


def examples():
    return [replace(item, encoder_ids=()) for item in paired_examples()]


@pytest.fixture
def fixture_guard(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    policy = unified_recipe("decoder_only")
    policy.update(rank=2, alpha=4)
    model = build_native_tiny_model(get_v2_experiment("Gemma-4-E2B-IT"))
    configure_policy_model(model, policy)
    model.eval()
    props = SimpleNamespace(name="Fixture GPU", major=12, minor=0, total_memory=96 * 2**30)
    monkeypatch.setattr(guard, "_gpu_context", lambda _: (torch.device("cpu"), props))
    budget = [80 * 2**30]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (budget[0], 96 * 2**30))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 20 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(
        guard,
        "resource_snapshot",
        lambda _: {
            "gpu_peak_allocated_gib": 20.0,
            "gpu_allocated_gib": 2.0,
            "gpu_reserved_gib": 3.0,
            "gpu_free_gib": budget[0] / 2**30,
        },
    )
    events, saves = [], []
    options = dict(
        state_path=tmp_path / "execution_preflight.json",
        identity={"data": "fixed-fixture"},
        training_policy=policy,
        progress=events.append,
        on_saved=lambda: saves.append(
            json.loads((tmp_path / "execution_preflight.json").read_text())
        ),
    )
    yield model, options, events, saves, budget
    torch.set_num_threads(previous_threads)


def test_guard_runs_two_phases_restores_weights_rng_and_reuses_evidence(fixture_guard, monkeypatch):
    model, options, events, saves, budget = fixture_guard
    weights = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    original = guard._measure_phase
    calls = []

    def measure(model, probe, optimizer, phase, *args):
        assert optimizer.state and all(
            optimizer.state[p] for p in model.parameters() if p.requires_grad
        )
        calls.append(phase)
        return original(model, probe, optimizer, phase, *args)

    monkeypatch.setattr(guard, "_measure_phase", measure)
    result = guard.select_execution(model, examples(), examples()[:1], **options)
    assert calls == ["training", "evaluation"]
    assert result["batch_size"] == 1 and result["accumulation"] == 4
    assert result["memory_policy"]["query_chunk_size"] == 512
    assert result["status"] == "VALIDATED_ON_THIS_GPU"
    assert result["optimizer_updates_retained"] == 0 and not model.training
    assert any(s["phases"].get("training", {}).get("status") == "COMPLETED" for s in saves)
    for name, expected in weights.items():
        torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    again = guard.select_execution(model, examples(), examples()[:1], **options)
    assert again == result and calls == ["training", "evaluation"]
    assert events[-1]["stage"] == "preflight_reused"
    budget[0] = 1 * 2**30
    with pytest.raises(RuntimeError, match="Current GPU budget"):
        guard.select_execution(model, examples(), examples()[:1], **options)
    assert len(calls) == 2


def test_guard_resumes_only_unfinished_phase_and_rejects_changed_or_corrupt_receipt(
    fixture_guard,
    monkeypatch,
):
    model, options, _, _, _ = fixture_guard
    original = guard._measure_phase
    calls = []

    def measure(*args):
        phase = args[3]
        calls.append(phase)
        if phase == "evaluation" and calls.count(phase) == 1:
            raise KeyboardInterrupt("Molab session ended")
        return original(*args)

    monkeypatch.setattr(guard, "_measure_phase", measure)
    weights = copy.deepcopy(model.state_dict())
    with pytest.raises(KeyboardInterrupt):
        guard.select_execution(model, examples(), examples()[:1], **options)
    saved = json.loads(options["state_path"].read_text())
    assert saved["status"] == "INTERRUPTED"
    assert saved["phases"]["training"]["status"] == "COMPLETED"
    for name, expected in weights.items():
        torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)
    result = guard.select_execution(model, examples(), examples()[:1], **options)
    assert calls == ["training", "evaluation", "evaluation"]
    assert result["status"] == "VALIDATED_ON_THIS_GPU" and "error" not in result
    with pytest.raises(ValueError, match="identity changed"):
        guard.select_execution(
            model, examples(), examples()[:1], **{**options, "identity": {"data": "changed"}}
        )
    result["batch_size"] = 2
    atomic_json(options["state_path"], result)
    with pytest.raises(ValueError, match="checksum mismatch"):
        guard.select_execution(model, examples(), examples()[:1], **options)


def test_guard_failure_is_durable_and_rolls_back_probe_parameters(fixture_guard, monkeypatch):
    model, options, _, saves, _ = fixture_guard
    weights = copy.deepcopy(model.state_dict())

    def oom(*args):
        with torch.no_grad():
            next(p for p in model.parameters() if p.requires_grad).add_(42)
        raise torch.cuda.OutOfMemoryError("Fixture allocation failure")

    monkeypatch.setattr(guard, "_measure_phase", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        guard.select_execution(model, examples(), examples()[:1], **options)
    assert saves[-1]["status"] == "FAILED"
    assert saves[-1]["phases"]["training"]["status"] == "FAILED"
    for name, expected in weights.items():
        torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)


def test_preflight_upload_verifies_exact_evidence_without_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "local"
    atomic_json(root / "pipeline_state.json", {"identity": "fixture"})
    atomic_json(root / "diagnostics/execution_preflight.json", {"status": "RUNNING"})
    remote = {}

    class Api:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, *args, **kwargs):
            pass

        def model_info(self, *args):
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, *args, **kwargs):
            for op in kwargs["operations"]:
                remote[op.path_in_repo] = op.path_or_fileobj
            return SimpleNamespace(oid="b" * 40)

    def download(repo, name, **kwargs):
        assert kwargs["revision"] == "b" * 40
        path = tmp_path / "remote" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[name])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = synchronize_preflight("fixture/model", root, "family/model/train", token="test")
    assert result["revision"] == "b" * 40
    assert len(remote) == 3
    assert not any("training/" in n or n.endswith("/upload_complete.json") for n in remote)
    assert not (root / "training").exists()


def test_stage_counters_do_not_leak_and_flex_backward_tiles_cover_head256(tmp_path, monkeypatch):
    for dtype in (torch.float32, torch.bfloat16):
        for width in (256, 512):
            options = flex_kernel_options(width, dtype)
            assert options["bwd_BLOCK_M1"] == options["bwd_BLOCK_N2"] == 16
            assert options["bwd_num_stages"] == 1
    dashboard = TrainingDashboard(None, refresh_seconds=0)
    dashboard({"stage": "tokenization", "items_done": 4934, "items_total": 4934})
    dashboard({"stage": "preflight_optimizer_init"})
    assert "items_done" not in dashboard.state
    heartbeat = RemoteHeartbeat("fixture/model", tmp_path, "family/model/train", token="test")
    heartbeat.last_success = float("inf")  # Keep this test offline.
    heartbeat({"stage": "tokenization", "items_done": 4934, "items_total": 4934})
    heartbeat({"stage": "run_started", "batch_size": 1, "accumulation": 4, "gpu_reserved_gib": 80})
    status = json.loads((tmp_path / "last_seen.json").read_text())
    assert "items_total" not in status and status["batch_size"] == 1
    assert status["gpu_reserved_gib"] == 80


def test_runtime_archive_is_identical_with_windows_or_linux_newlines(tmp_path):
    for name in (
        "src/model.py",
        "configs/policy.json",
        "LICENSE",
        "tools/notebook_dependencies.py",
        "tools/notebook_launcher.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"line one\nline two\n")
    linux = source_bundle_bytes(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert source_bundle_bytes(tmp_path) == linux


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("index", range(6))
def test_guard_cuda_bf16_restores_all_six_models_and_cached_receipt(tmp_path, index):
    family = "decoder_only" if index >= 4 else "symmetric" if index < 2 else "asymmetric"
    model = build_native_tiny_model(get_v2_experiment(EXPERIMENT_NAMES[index])).to(
        "cuda", torch.bfloat16
    )
    policy = unified_recipe(family)
    policy.update(rank=2, alpha=4)
    configure_policy_model(model, policy)
    data = examples() if index >= 4 else paired_examples()
    before = {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    batch_size = (1, 2, 4)[index % 3]
    opts = dict(
        state_path=tmp_path / "preflight.json",
        identity={"model": index},
        training_policy=policy,
        min_headroom_gib=0.25,
        execution={
            "memory_policy": guard.FIXED_MEMORY_POLICY,
            "batch_size": batch_size,
            "accumulation": 4 // batch_size,
        },
    )
    result = guard.select_execution(model, data, data[:1], **opts)
    assert result["status"] == "VALIDATED_ON_THIS_GPU"
    assert result["batch_size"] == batch_size and result["accumulation"] == 4 // batch_size
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), expected, atol=0, rtol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert guard.select_execution(model, data, data[:1], **opts) == result
    if model.spec.name == "BifocalGemma-E4B-E4B":
        from bifocalgemma.training.shared_execution import benchmark_candidate

        cohort = [data[i % len(data)] for i in range(8)]
        timing = benchmark_candidate(model, cohort, opts["execution"], policy)
        assert timing["seconds_per_update"] > 0 and len(timing["seconds_by_window"]) == 2
        assert timing["compile_warmup_seconds"] > 0
        for name, expected in before.items():
            torch.testing.assert_close(model.state_dict()[name].cpu(), expected, atol=0, rtol=0)
