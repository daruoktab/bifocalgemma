"""Calibrate only E4B→E4B; all six notebooks consume one immutable profile."""

import gc
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.models.memory_efficient import (
    DEFAULT_MEMORY_POLICY,
    configure_memory_efficient_training,
)
from bifocalgemma.training.benchmark_upload import retry_delay
from bifocalgemma.training.optimizers import build_policy_optimizer
from bifocalgemma.training.policy import build_training_groups
from bifocalgemma.training.stages import atomic_json
from bifocalgemma.training.telemetry import resource_snapshot
from bifocalgemma.training.v2_preflight import (
    EXECUTION_VERSION,
    _gpu_context,
    execution_settings,
    seal_report,
    select_execution,
    verify_report,
)

ASSETS_REPO = "daruokta/bifocalgemma-v2-assets"
PROFILE_FILE = "profiles/training.json"
LEADER = "e4b-e4b"
HEADROOM_GIB = 4.0


def shared_contract(identity):
    return {
        k: identity[k]
        for k in (
            "execution_policy",
            "source_fingerprint",
            "dataset_repo",
            "dataset_revision",
            "collection",
            "seed",
            "effective_batch",
            "runtime_versions",
        )
    }


def validate_profile(profile, contract):
    verify_report(profile)
    if profile.get("format") != "shared_execution_profile_v1" or profile.get("leader") != LEADER:
        raise ValueError("Invalid shared execution profile format or calibration leader.")
    if profile.get("contract") != contract:
        raise ValueError("Shared execution profile belongs to another source/data contract.")
    settings = execution_settings(profile)
    policy = settings["memory_policy"]
    if policy.get("attention_backend") not in {"tiled_eager_fp32", "flex_attention", "hybrid_flex"}:
        raise ValueError("Unsupported shared attention backend.")
    if (
        policy.get("query_chunk_size") not in {512, 1024, 2048}
        or policy.get("loss_chunk_size") != 128
    ):
        raise ValueError("Shared attention/loss chunks are outside the calibrated search space.")
    if type(policy.get("gradient_checkpointing")) is not bool:
        raise ValueError("Shared checkpointing setting must be explicit.")
    if profile.get("min_headroom_gib") != HEADROOM_GIB:
        raise ValueError("Shared profile has an unexpected memory margin.")
    evidence = profile.get("validation", {})
    verify_report(evidence)
    if evidence.get("status") != "VALIDATED_ON_THIS_GPU":
        raise ValueError("Shared profile has no completed maximum-shape validation.")
    if (
        evidence.get("batch_size"),
        evidence.get("accumulation"),
        evidence.get("memory_policy"),
    ) != (
        settings["batch_size"],
        settings["accumulation"],
        {**DEFAULT_MEMORY_POLICY, **policy},
    ):
        raise ValueError("Shared profile settings differ from the validated maximum-shape guard.")
    for phase in ("training", "evaluation"):
        row = evidence.get("phases", {}).get(phase, {})
        if (
            row.get("status") != "COMPLETED"
            or row.get("estimated_peak_headroom_gib", -1) < HEADROOM_GIB
        ):
            raise ValueError(
                "Shared profile is missing a completed phase or sufficient memory margin."
            )
    return profile


def read_remote_profile(contract, *, token, revision=None):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    api = HfApi(token=token)
    if revision is None:
        revision = api.repo_info(ASSETS_REPO, repo_type="dataset").sha
    try:
        filename = hf_hub_download(
            ASSETS_REPO, PROFILE_FILE, repo_type="dataset", revision=revision, token=token
        )
    except EntryNotFoundError:
        return None
    profile = json.loads(Path(filename).read_text())
    verify_report(profile)
    if profile.get("contract") != contract:
        return None
    validate_profile(profile, contract)
    return profile


def resolve_shared_profile(model_key, root, contract, *, token, progress=None, wait_seconds=21600):
    """Followers wait before downloading backbones; a cached run pins its profile."""
    path = Path(root) / "diagnostics/shared_profile.json"
    if path.exists():
        return validate_profile(json.loads(path.read_text()), contract)
    deadline = time.monotonic() + wait_seconds
    failures = 0
    retry_at = 0.0
    while True:
        lookup_succeeded = False
        try:
            profile = None
            if time.monotonic() >= retry_at:
                profile = read_remote_profile(contract, token=token)
                failures = 0
                lookup_succeeded = True
        except Exception as error:
            delay = retry_delay(error, failures + 1, time.time())
            if delay is None:
                raise
            failures += 1
            retry_at = time.monotonic() + delay
            profile = None
        if profile is not None:
            atomic_json(path, profile)
            return profile
        if model_key == LEADER and lookup_succeeded:
            return None
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Shared profile is not ready. Run the E4B→E4B notebook first, then rerun."
            )
        if progress:
            progress(
                {
                    "stage": "preflight_waiting_shared_profile",
                    "message": "Menunggu HF untuk memeriksa profil bersama; kalibrasi belum diulang."
                    if model_key == LEADER
                    else "Menunggu profil bersama dari E4B→E4B; model belum dimuat. Tidak mencari setting sendiri.",
                }
            )
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def publish_profile(profile, *, token):
    """Never replace another valid selection for the same experiment contract."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import HfHubHTTPError

    validate_profile(profile, profile["contract"])
    api = HfApi(token=token)
    for attempt in range(3):
        parent = api.repo_info(ASSETS_REPO, repo_type="dataset").sha
        existing = read_remote_profile(profile["contract"], token=token, revision=parent)
        if existing is not None:
            return existing
        try:
            commit = api.create_commit(
                ASSETS_REPO,
                repo_type="dataset",
                parent_commit=parent,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=PROFILE_FILE,
                        path_or_fileobj=json.dumps(profile, indent=2).encode(),
                    )
                ],
                commit_message="Publish E4B-E4B calibrated shared training profile",
            )
            from huggingface_hub import hf_hub_download

            remote = hf_hub_download(
                ASSETS_REPO, PROFILE_FILE, repo_type="dataset", revision=commit.oid, token=token
            )
            checked = json.loads(Path(remote).read_text())
            validate_profile(checked, profile["contract"])
            if checked != profile:
                raise ValueError("Published shared execution profile differs from local evidence.")
            return checked
        except HfHubHTTPError as error:
            if error.response.status_code != 412 or attempt == 2:
                raise
    raise RuntimeError("Shared profile commit retry exhausted.")


def publish_with_retry(profile, *, token, progress=None):
    failures = 0
    while True:
        try:
            return publish_profile(profile, token=token)
        except Exception as error:
            delay = retry_delay(error, failures + 1, time.time())
            if delay is None:
                raise
            failures += 1
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                if progress:
                    progress(
                        {
                            "stage": "preflight_shared_profile_upload_pending",
                            "upload_pending": True,
                            "message": f"Profil bersama sudah tersimpan lokal; upload HF tertunda, retry dalam {max(1, int(deadline - time.monotonic()))} detik.",
                        }
                    )
                time.sleep(min(30, max(0, deadline - time.monotonic())))


@contextmanager
def probe_optimizer(model, training_policy):
    """Rollback all trainable updates and RNG; optimizer state stays on device."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    backups = [p.detach().cpu().clone() for p in parameters]
    cpu_rng, cuda_rng, old_mode = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
        model.training,
    )
    optimizer = None
    try:
        optimizer = build_policy_optimizer(
            build_training_groups(model, training_policy, total_steps=1), training_policy
        )
        for p in parameters:
            p.grad = torch.zeros_like(p)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        yield optimizer
    finally:
        with torch.no_grad():
            for parameter, backup in zip(parameters, backups, strict=True):
                parameter.copy_(backup)
        model.zero_grad(set_to_none=True)
        del optimizer, backups
        model.train(old_mode)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        gc.collect()
        torch.cuda.empty_cache()


def candidate_settings():
    candidates = []
    for backend in ("tiled_eager_fp32", "flex_attention", "hybrid_flex"):
        for chunk in (512, 1024, 2048) if backend == "tiled_eager_fp32" else (512,):
            for batch in (1, 2, 4):
                name = {
                    "tiled_eager_fp32": "tiled",
                    "flex_attention": "flex",
                    "hybrid_flex": "hybrid",
                }[backend]
                candidates.append(
                    (
                        f"{name}-b{batch}-c{chunk}",
                        {
                            "batch_size": batch,
                            "accumulation": 4 // batch,
                            "memory_policy": {
                                "attention_backend": backend,
                                "query_chunk_size": chunk,
                                "loss_chunk_size": 128,
                                "gradient_checkpointing": True,
                            },
                        },
                    )
                )
    return candidates


def error_kind(error):
    message = str(error).lower()
    if "out of resource" in message and ("hardware limit" in message or "shared" in message):
        return "kernel_shared_memory_limit"
    if isinstance(error, torch.cuda.OutOfMemoryError) or "cuda out of memory" in message:
        return "device_vram_oom"
    return "kernel_or_numerical_error"


def numerical_probe(model, item):
    model.train()
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    batch = RepairCollator()([item]).to(device)
    result = model.forward_loss(batch)
    if result.loss is None or not bool(torch.isfinite(result.loss)):
        raise RuntimeError("Nonfinite backend-check loss.")
    result.loss.backward()
    gradients = {
        n: p.grad.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    loss = float(result.loss.detach())
    del result, batch
    model.zero_grad(set_to_none=True)
    return loss, gradients


def representative_cohort(train):
    """Sample equal-frequency length bins; maxima are only for capacity tests."""
    count = min(12, len(train) // 4 * 4)
    if count < 8:
        raise ValueError("Calibration requires at least eight training examples.")
    if hasattr(train, "rows"):
        order = sorted(range(len(train)), key=lambda i: sum(x[1] for x in train.rows[i]["offsets"]))
    else:
        order = sorted(
            range(len(train)),
            key=lambda i: (
                len(train[i].encoder_ids)
                + len(train[i].decoder_prefix_ids)
                + len(train[i].target_ids)
            ),
        )
    chosen = [order[min(len(order) - 1, int((i + 0.5) * len(order) / count))] for i in range(count)]
    random.Random(17).shuffle(chosen)
    return [train[i] for i in chosen]


def benchmark_candidate(model, cohort, settings, training_policy):
    """Real effective-batch-four windows, not duplicated maximum examples."""
    device = next(model.parameters()).device
    configure_memory_efficient_training(model, settings["memory_policy"])
    batch_size = settings["batch_size"]
    seconds = []
    with probe_optimizer(model, training_policy) as optimizer:
        model.train()
        # Warm every timed batch shape (including B2/B4), so first compilation is
        # recorded separately instead of making a fused candidate appear slower.
        warmup_started = time.perf_counter()
        for offset in range(0, len(cohort), 4):
            items = cohort[offset : offset + 4]
            for i in range(0, len(items), batch_size):
                warmup = RepairCollator()(items[i : i + batch_size]).to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model.forward_loss(warmup)
                if output.loss is None or not bool(torch.isfinite(output.loss)):
                    raise RuntimeError("Nonfinite calibration warmup loss.")
                output.loss.backward()
                del output, warmup
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_started
        torch.cuda.reset_peak_memory_stats(device)
        for offset in range(0, len(cohort), 4):
            items = cohort[offset : offset + 4]
            batches = [
                RepairCollator()(items[i : i + batch_size])
                for i in range(0, len(items), batch_size)
            ]
            token_count = sum(int((b.labels != -100).sum()) for b in batches)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for batch_cpu in batches:
                batch = batch_cpu.to(device)
                output = model.forward_loss(batch)
                if output.loss is None or not bool(torch.isfinite(output.loss)):
                    raise RuntimeError("Nonfinite calibration loss.")
                count = int((batch_cpu.labels != -100).sum())
                (output.loss * (count / token_count)).backward()
                del output, batch
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0, foreach=False
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite calibration gradients.")
            optimizer.step()
            torch.cuda.synchronize(device)
            seconds.append(time.perf_counter() - started)
        stats = resource_snapshot(device)
    return {
        "status": "SCREENED",
        "settings": settings,
        "compile_warmup_seconds": warmup_seconds,
        "seconds_by_window": seconds,
        "seconds_per_update": sum(seconds) / len(seconds),
        **stats,
    }


def calibrate_shared_profile(
    model, train, dev, training_policy, contract, root, *, progress=None, on_saved=None
) -> dict[str, Any]:
    """Only the leader searches. All candidate outcomes are durable and resumable."""
    device, props = _gpu_context(model)
    if model.spec.name != "BifocalGemma-E4B-E4B":
        raise ValueError("Only E4B→E4B may choose the shared execution profile.")
    if len(train) < 8:
        raise ValueError("Calibration requires at least eight training examples.")
    root = Path(root)
    path = root / "diagnostics/calibration.json"
    gpu = {
        "name": props.name,
        "capability": [props.major, props.minor],
        "total_memory": props.total_memory,
    }
    signature = stable_fingerprint({"contract": contract, "gpu": gpu, "version": EXECUTION_VERSION})
    state: dict[str, Any] = {
        "format": "shared_calibration_v1",
        "signature": signature,
        "status": "RUNNING",
        "backend_checks": {},
        "trials": {},
        "guards": {},
    }
    if path.exists():
        state = json.loads(path.read_text())
        verify_report(state)
        if state.get("signature") != signature:
            raise ValueError("Calibration source/data/GPU changed; use a fresh run directory.")
    cohort = representative_cohort(train)
    state["cohort_sha256"] = stable_fingerprint([x.instance_id for x in cohort])

    def save():
        atomic_json(path, seal_report(state))
        if on_saved:
            on_saved()

    def emit(message, **event):
        if progress:
            progress({"stage": "preflight_calibration", "message": message, **event})

    def trial(name, settings):
        if state["trials"].get(name, {}).get("status") in {"SCREENED", "FAILED"}:
            return
        state["trials"][name] = {"status": "RUNNING", "settings": settings}
        save()
        done = sum(r.get("status") in {"SCREENED", "FAILED"} for r in state["trials"].values())
        emit(
            f"Kalibrasi E4B→E4B: kandidat {done + 1}/maksimal17 {name}; {len(cohort)} contoh nyata, effective batch4.",
            candidate=name,
            items_done=done,
            items_total=17,
            batch_size=settings["batch_size"],
            accumulation=settings["accumulation"],
            **settings["memory_policy"],
        )
        try:
            state["trials"][name] = benchmark_candidate(model, cohort, settings, training_policy)
        except (RuntimeError, ValueError, AssertionError, ImportError) as error:
            state["trials"][name] = {
                "status": "FAILED",
                "settings": settings,
                "reason": str(error)[:1200],
                "error_kind": error_kind(error),
            }
        save()
        outcome = state["trials"][name]
        if outcome["status"] == "SCREENED":
            emit(
                f"Kalibrasi {name}: {outcome['seconds_per_update']:.3f} detik/update; "
                f"peak allocated {outcome.get('gpu_peak_allocated_gib', 0):.2f} GiB.",
                candidate=name,
                candidate_status="SCREENED",
            )
        else:
            emit(
                f"Kalibrasi {name}: dilewati ({outcome['error_kind']}); {outcome['reason']}",
                candidate=name,
                candidate_status="FAILED",
            )

    # One numerical/kernel capability check per backend, not once per batch/chunk.
    checks = state["backend_checks"]
    pending = [b for b in ("tiled_eager_fp32", "flex_attention", "hybrid_flex") if b not in checks]
    if pending:
        with probe_optimizer(model, training_policy):
            configure_memory_efficient_training(model, {"attention_backend": "tiled_eager_fp32"})
            check_item = min(
                cohort,
                key=lambda x: len(x.encoder_ids) + len(x.decoder_prefix_ids) + len(x.target_ids),
            )
            ref_loss, reference = numerical_probe(model, check_item)
            for backend in pending:
                emit(f"Kalibrasi E4B→E4B: pemeriksaan numerik/kernel {backend}.")
                try:
                    configure_memory_efficient_training(model, {"attention_backend": backend})
                    loss, gradients = numerical_probe(model, check_item)
                    if abs(loss - ref_loss) > max(1e-3, abs(ref_loss) * 0.01):
                        raise ValueError("Backend loss differs from tiled reference.")
                    errors = []
                    for name, truth in reference.items():
                        actual = gradients.get(name)
                        if actual is None or not bool(torch.isfinite(actual).all()):
                            raise ValueError("Backend lost a finite trainable gradient.")
                        errors.append(float((actual - truth).norm() / truth.norm().clamp_min(1e-8)))
                    if max(errors, default=0) > 0.05:
                        raise ValueError("Backend gradient differs from tiled reference.")
                    checks[backend] = {
                        "status": "PASS",
                        "loss_difference": loss - ref_loss,
                        "max_gradient_relative_l2_error": max(errors, default=0),
                    }
                except (RuntimeError, ValueError, AssertionError, ImportError) as error:
                    checks[backend] = {
                        "status": "FAILED",
                        "reason": str(error)[:1200],
                        "error_kind": error_kind(error),
                    }
                finally:
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                save()
    for name, settings in candidate_settings():
        if checks[settings["memory_policy"]["attention_backend"]]["status"] == "PASS":
            trial(name, settings)
    ranked = sorted(
        (n for n, r in state["trials"].items() if r["status"] == "SCREENED"),
        key=lambda n: state["trials"][n]["seconds_per_update"],
    )
    # Test removing activation recomputation only for the two fastest screened settings.
    bases = [n for n in ranked if not n.endswith("-full")][:2]
    for name in bases:
        settings = state["trials"][name]["settings"]
        trial(
            name + "-full",
            {
                **settings,
                "memory_policy": {**settings["memory_policy"], "gradient_checkpointing": False},
            },
        )
    ranked = sorted(
        (n for n, r in state["trials"].items() if r["status"] == "SCREENED"),
        key=lambda n: state["trials"][n]["seconds_per_update"],
    )
    for name in ranked:
        if state["guards"].get(name, {}).get("status") == "FAILED":
            continue
        settings = state["trials"][name]["settings"]
        emit(f"Kalibrasi E4B→E4B: validasi maksimum kandidat {name}; margin {HEADROOM_GIB:g} GiB.")
        try:
            result = select_execution(
                model,
                train,
                dev,
                training_policy=training_policy,
                execution=settings,
                identity=contract,
                min_headroom_gib=HEADROOM_GIB,
                progress=progress,
                state_path=root / "diagnostics/calibration" / (name + ".json"),
                on_saved=on_saved,
            )
        except (RuntimeError, ValueError, AssertionError, ImportError) as error:
            state["guards"][name] = {
                "status": "FAILED",
                "reason": str(error)[:1200],
                "error_kind": error_kind(error),
            }
            save()
            continue
        state["guards"][name] = result
        state["status"] = "COMPLETED"
        save()
        return seal_report(
            {
                "format": "shared_execution_profile_v1",
                "leader": LEADER,
                "contract": contract,
                "gpu": gpu,
                **settings,
                "min_headroom_gib": HEADROOM_GIB,
                "calibration": {
                    "candidate": name,
                    "seconds_per_update": state["trials"][name]["seconds_per_update"],
                    "cohort_sha256": state["cohort_sha256"],
                    "examples": len(cohort),
                    "scope": "bounded representative timing; not a proof of global optimality",
                },
                "validation": result,
            }
        )
    state["status"] = "FAILED"
    save()
    raise RuntimeError(
        "No shared E4B→E4B candidate passed kernel, numerics and maximum-shape memory checks."
    )
