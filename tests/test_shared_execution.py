"""One leader, one profile, durable calibration, and identical follower settings."""

import copy
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx2
import huggingface_hub
import pytest
import torch
from huggingface_hub.errors import HfHubHTTPError

from bifocalgemma.models.memory_efficient import DEFAULT_MEMORY_POLICY
from bifocalgemma.training import shared_execution as shared
from bifocalgemma.training.v2_preflight import seal_report
from tests.test_offline_sft_training import examples


def valid_profile(contract=None, settings=None) -> dict[str, Any]:
    settings: dict[str, Any] = settings or {
        "batch_size": 2,
        "accumulation": 2,
        "memory_policy": {
            "attention_backend": "tiled_eager_fp32",
            "query_chunk_size": 1024,
            "loss_chunk_size": 128,
            "gradient_checkpointing": True,
        },
    }
    evidence = seal_report(
        {
            "status": "VALIDATED_ON_THIS_GPU",
            **settings,
            "memory_policy": {**DEFAULT_MEMORY_POLICY, **settings["memory_policy"]},
            "phases": {
                p: {"status": "COMPLETED", "estimated_peak_headroom_gib": 5.0}
                for p in ("training", "evaluation")
            },
        }
    )
    return seal_report(
        {
            "format": "shared_execution_profile_v1",
            "leader": "e4b-e4b",
            "contract": contract or {"source": "fixture"},
            **settings,
            "gpu": {"name": "Fixture GPU"},
            "min_headroom_gib": 4.0,
            "validation": evidence,
        }
    )


def test_profile_rejects_corruption_mismatched_guard_and_effective_batch():
    profile = valid_profile()
    assert shared.validate_profile(profile, profile["contract"]) == profile
    damaged = copy.deepcopy(profile)
    damaged["batch_size"] = 4
    with pytest.raises(ValueError, match="checksum"):
        shared.validate_profile(damaged, profile["contract"])
    damaged = seal_report(damaged)
    with pytest.raises(ValueError, match="effective batch"):
        shared.validate_profile(damaged, profile["contract"])
    damaged["accumulation"] = 1
    with pytest.raises(ValueError, match="differ from"):
        shared.validate_profile(seal_report(damaged), profile["contract"])
    with pytest.raises(ValueError, match="another source/data"):
        shared.validate_profile(profile, {"source": "changed"})


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(1))
    model = SimpleNamespace(
        spec=SimpleNamespace(name="BifocalGemma-E4B-E4B"),
        parameters=lambda: iter([parameter]),
        zero_grad=lambda **kwargs: None,
    )
    props = SimpleNamespace(name="Fixture GPU", major=12, minor=0, total_memory=96 * 2**30)
    monkeypatch.setattr(shared, "_gpu_context", lambda _: (torch.device("cpu"), props))
    monkeypatch.setattr(shared, "configure_memory_efficient_training", lambda *args: None)
    monkeypatch.setattr(shared, "numerical_probe", lambda *args: (1.0, {"p": torch.ones(1)}))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    @contextmanager
    def probe(*args):
        yield object()

    monkeypatch.setattr(shared, "probe_optimizer", probe)
    trials, guards = [], []
    interrupt = [False]

    def bench(model, cohort, settings, policy):
        key = (
            settings["memory_policy"]["attention_backend"],
            settings["batch_size"],
            settings["memory_policy"]["query_chunk_size"],
            settings["memory_policy"]["gradient_checkpointing"],
        )
        trials.append(key)
        if interrupt[0] and len(trials) == 3:
            raise KeyboardInterrupt("Molab session ended")
        score = {1: 10, 2: 5, 4: 1}[settings["batch_size"]]
        score -= settings["memory_policy"]["query_chunk_size"] / 100000
        if not settings["memory_policy"]["gradient_checkpointing"]:
            score -= 0.5
        return {
            "status": "SCREENED",
            "settings": settings,
            "seconds_per_update": score,
            "gpu_peak_allocated_gib": 50,
            "gpu_reserved_gib": 93,
        }

    def guard(*args, **kwargs):
        settings = kwargs["execution"]
        guards.append(settings)
        if settings["batch_size"] == 4:
            raise torch.cuda.OutOfMemoryError("Fixture maximum-shape allocation failed")
        return valid_profile(settings=settings)["validation"]

    monkeypatch.setattr(shared, "benchmark_candidate", bench)
    monkeypatch.setattr(shared, "select_execution", guard)
    data = [
        replace(examples()[0], instance_id=f"case-{i}", encoder_ids=tuple(range(i + 1)))
        for i in range(16)
    ]
    yield model, data, tmp_path, trials, guards, interrupt


def test_only_leader_selects_fastest_maximum_shape_survivor_and_resumes_trials(calibration):
    model, data, root, trials, guards, interrupt = calibration
    interrupt[0] = True
    with pytest.raises(KeyboardInterrupt):
        shared.calibrate_shared_profile(model, data, data[:2], {}, {"source": "fixture"}, root)
    saved = json.loads((root / "diagnostics/calibration.json").read_text())
    assert sum(r["status"] == "SCREENED" for r in saved["trials"].values()) == 2
    interrupt[0] = False
    result = shared.calibrate_shared_profile(model, data, data[:2], {}, {"source": "fixture"}, root)
    shared.validate_profile(result, {"source": "fixture"})
    assert result["batch_size"] == 2 and result["accumulation"] == 2
    assert result["memory_policy"]["query_chunk_size"] == 2048
    assert trials.count(trials[0]) == 1 and trials.count(trials[1]) == 1
    assert trials.count(trials[2]) == 2  # Interrupted trial, not completed trials, repeats.
    assert all(s["batch_size"] == 4 for s in guards[:-1])
    completed_trials = len(trials)
    again = shared.calibrate_shared_profile(model, data, data[:2], {}, {"source": "fixture"}, root)
    assert again == result and len(trials) == completed_trials


def test_kernel_shared_memory_failure_is_not_device_vram_and_skips_backend(
    calibration, monkeypatch
):
    model, data, root, trials, _, _ = calibration
    backend = [None]
    monkeypatch.setattr(
        shared,
        "configure_memory_efficient_training",
        lambda model, p: backend.__setitem__(0, p["attention_backend"]),
    )

    def numerical(*args):
        if backend[0] == "flex_attention":
            raise RuntimeError(
                "OutOfMemoryError: out of resource: Required: 114688 Hardware limit:101376"
            )
        return 1.0, {"p": torch.ones(1)}

    monkeypatch.setattr(shared, "numerical_probe", numerical)
    shared.calibrate_shared_profile(model, data, data[:2], {}, {"source": "fixture"}, root)
    saved = json.loads((root / "diagnostics/calibration.json").read_text())
    assert saved["backend_checks"]["flex_attention"]["error_kind"] == "kernel_shared_memory_limit"
    assert not any(row[0] == "flex_attention" for row in trials)
    assert (
        shared.error_kind(torch.cuda.OutOfMemoryError("CUDA allocation failed"))
        == "device_vram_oom"
    )


def test_followers_wait_for_one_profile_and_pin_it_locally(tmp_path, monkeypatch):
    profile = valid_profile()
    now, calls = [0.0], []
    monkeypatch.setattr(shared.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(shared.time, "sleep", lambda n: now.__setitem__(0, now[0] + n))

    def fetch(*args, **kwargs):
        calls.append(1)
        return profile if len(calls) >= 3 else None

    monkeypatch.setattr(shared, "read_remote_profile", fetch)
    events = []
    first = shared.resolve_shared_profile(
        "e2b-it",
        tmp_path / "first",
        profile["contract"],
        token="test",
        progress=events.append,
        wait_seconds=90,
    )
    second = shared.resolve_shared_profile(
        "e4b-it", tmp_path / "second", profile["contract"], token="test"
    )
    assert first == second == profile and len(events) == 2
    count = len(calls)
    assert (
        shared.resolve_shared_profile(
            "e2b-it", tmp_path / "first", profile["contract"], token="test"
        )
        == profile
    )
    assert len(calls) == count


def test_leader_does_not_recalibrate_just_because_profile_lookup_is_offline(tmp_path, monkeypatch):
    now, calls = [0.0], []
    monkeypatch.setattr(shared.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(shared.time, "sleep", lambda n: now.__setitem__(0, now[0] + n))

    def fetch(*args, **kwargs):
        calls.append(now[0])
        if len(calls) == 1:
            raise httpx2.ConnectError("Network disconnected")
        return None

    monkeypatch.setattr(shared, "read_remote_profile", fetch)
    assert (
        shared.resolve_shared_profile(
            "e4b-e4b", tmp_path, {"source": "fixture"}, token="test", wait_seconds=90
        )
        is None
    )
    assert len(calls) == 2 and calls[1] >= 60


def test_shared_profile_upload_retries_without_losing_local_selection(monkeypatch):
    profile = valid_profile()
    now, attempts, events = [0.0], [], []
    monkeypatch.setattr(shared.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(shared.time, "sleep", lambda n: now.__setitem__(0, now[0] + n))

    def publish(*args, **kwargs):
        attempts.append(now[0])
        if len(attempts) == 1:
            raise httpx2.ConnectError("Network disconnected")
        return profile

    monkeypatch.setattr(shared, "publish_profile", publish)
    assert shared.publish_with_retry(profile, token="test", progress=events.append) == profile
    assert len(attempts) == 2 and attempts[1] >= 60
    assert events and all(e["upload_pending"] for e in events)


def test_profile_publish_parent_guard_keeps_concurrent_winner(monkeypatch):
    own = valid_profile()
    other = valid_profile(
        settings={"batch_size": 1, "accumulation": 4, "memory_policy": own["memory_policy"]}
    )
    commits = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="b" * 40 if commits else "a" * 40)

        def create_commit(self, *args, **kwargs):
            commits.append(kwargs)
            response = httpx2.Response(
                412, request=httpx2.Request("POST", "https://huggingface.co/fixture")
            )
            raise HfHubHTTPError("Parent changed", response=response)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(
        shared,
        "read_remote_profile",
        lambda *a, **kw: other if kw["revision"] == "b" * 40 else None,
    )
    assert shared.publish_profile(own, token="test") == other
    assert len(commits) == 1 and commits[0]["parent_commit"] == "a" * 40


def test_length_sample_avoids_giving_absolute_maximum_disproportionate_weight():
    data = [
        replace(examples()[0], instance_id=str(i), encoder_ids=(1,) * (i + 1)) for i in range(120)
    ]
    cohort = shared.representative_cohort(data)
    assert len(cohort) == 12 and len({x.instance_id for x in cohort}) == 12
    assert max(len(x.encoder_ids) for x in cohort) < 120
    assert shared.representative_cohort(data) == cohort
