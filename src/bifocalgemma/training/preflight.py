"""A resumable two-phase guard for the one shared execution profile."""

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.training.optimizers import build_policy_optimizer
from bifocalgemma.training.policy import build_training_groups
from bifocalgemma.training.stages import atomic_json
from bifocalgemma.training.telemetry import resource_snapshot

EXECUTION_VERSION = "shared_e4b_e4b_v1"
FIXED_MEMORY_POLICY = {
    "attention_backend": "tiled_eager_fp32",
    "query_chunk_size": 512,
    "loss_chunk_size": 128,
    "gradient_checkpointing": True,
}


def run_name(mode):
    return "pilot" if mode == "pilot" else "train"


def execution_settings(execution=None) -> dict[str, Any]:
    settings: dict[str, Any] = execution or {
        "memory_policy": FIXED_MEMORY_POLICY,
        "batch_size": 1,
        "accumulation": 4,
    }
    if any(type(settings.get(k)) is not int for k in ("batch_size", "accumulation")) or (
        settings.get("batch_size"),
        settings.get("accumulation"),
    ) not in {(1, 4), (2, 2), (4, 1)}:
        raise ValueError("Shared execution must preserve effective batch four.")
    return {
        "memory_policy": dict(settings["memory_policy"]),
        "batch_size": settings["batch_size"],
        "accumulation": settings["accumulation"],
    }


def _gpu_context(model):
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Production memory guard requires the target CUDA GPU.")
    return device, torch.cuda.get_device_properties(device)


def maximum_probe(train, dev):
    """Combine actual train/dev maxima once; never shorten the context to fit."""
    extremes = []
    for examples in (train, dev):
        if not examples:
            raise ValueError("The fixed memory guard requires nonempty train and dev sets.")
        if hasattr(examples, "extreme_examples"):
            extremes.extend(examples.extreme_examples())
        else:
            extremes.extend(
                (
                    max(examples, key=lambda x: len(x.encoder_ids)),
                    max(examples, key=lambda x: len(x.decoder_prefix_ids) + len(x.target_ids)),
                )
            )
    encoder = max(extremes, key=lambda x: len(x.encoder_ids))
    decoder = max(extremes, key=lambda x: len(x.decoder_prefix_ids) + len(x.target_ids))
    return TokenizedRepairExample(
        "shape-only-train-dev-maxima",
        encoder.encoder_ids,
        decoder.decoder_prefix_ids,
        decoder.target_ids,
        {"source": "shape_stress_only"},
    )


def seal_report(report: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in report.items() if k != "evidence_sha256"}
    return {**payload, "evidence_sha256": stable_fingerprint(payload)}


def verify_report(report):
    if report.get("evidence_sha256") != seal_report(report)["evidence_sha256"]:
        raise ValueError("Memory-guard receipt checksum mismatch; inspect the saved report.")


def current_capacity(device, extra_peak_gib, min_headroom_gib):
    free, _ = torch.cuda.mem_get_info(device)
    reclaimable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    available = (free + reclaimable) / 2**30
    if available < extra_peak_gib + min_headroom_gib:
        raise RuntimeError(
            f"Current GPU budget {available:.2f} GiB cannot cover cached additional peak "
            f"{extra_peak_gib:.2f} GiB plus {min_headroom_gib:g} GiB headroom. "
            "Release other GPU workloads before resuming; context and batch were not changed."
        )
    return available


def _measure_phase(
    model, probe, optimizer, phase, device, min_headroom_gib, batch_size=1
) -> dict[str, Any]:
    """Optimizer states stay resident during both passes, like real training."""
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batch = RepairCollator()([probe] * batch_size).to(device)
    model.train(phase == "training")
    if phase == "training":
        output = model.forward_loss(batch)
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise RuntimeError("Nonfinite memory-guard training loss.")
        output.loss.backward()
        gradients = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(p.grad).all()) for p in gradients):
            raise RuntimeError("Missing or nonfinite memory-guard gradients.")
        loss = float(output.loss.detach())
        del output, batch
        norm = torch.nn.utils.clip_grad_norm_(gradients, 1.0, foreach=False)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Nonfinite memory-guard gradient norm.")
        optimizer.step()
        del gradients
    else:
        with torch.no_grad():
            output = model.forward_loss(batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError("Nonfinite memory-guard evaluation loss.")
            loss = float(output.loss)
        del output, batch
    torch.cuda.synchronize(device)
    stats = resource_snapshot(device)
    free, total = torch.cuda.mem_get_info(device)
    external = max(0, total - free - torch.cuda.memory_reserved(device))
    headroom = (total - external - torch.cuda.max_memory_allocated(device)) / 2**30
    if headroom < min_headroom_gib:
        raise RuntimeError(
            f"Shared microbatch {batch_size} memory guard leaves {headroom:.2f} GiB peak headroom; "
            f"requires {min_headroom_gib:g} GiB. Full training was not started."
        )
    return {
        "status": "COMPLETED",
        "phase": phase,
        "seconds": time.perf_counter() - started,
        "loss": loss,
        "estimated_peak_headroom_gib": headroom,
        **stats,
    }


def select_execution(
    model,
    train,
    dev,
    *,
    progress=None,
    min_headroom_gib=8,
    state_path=None,
    identity=None,
    on_saved=None,
    training_policy=None,
    execution=None,
):
    """Validate the chosen shared profile, with no per-model setting search.

    Reuse requires matching source/data/GPU, intact evidence and current capacity.
    Probe optimizer changes are restored exactly; they are never training steps.
    """
    device, props = _gpu_context(model)
    settings = execution_settings(execution)
    batch_size, accumulation = settings["batch_size"], settings["accumulation"]
    policy = configure_memory_efficient_training(model, settings["memory_policy"])
    probe = maximum_probe(train, dev)
    signature = stable_fingerprint(
        {
            "version": EXECUTION_VERSION,
            "identity": identity,
            "gpu": [props.name, props.major, props.minor, props.total_memory],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "memory_policy": policy,
            "batch_size": batch_size,
            "accumulation": accumulation,
            "min_headroom_gib": min_headroom_gib,
            "parameters": [
                (n, list(p.shape), str(p.dtype), p.requires_grad)
                for n, p in model.named_parameters()
            ],
            "probe": [len(probe.encoder_ids), len(probe.decoder_prefix_ids), len(probe.target_ids)],
        }
    )
    report: dict[str, Any] = {
        "version": EXECUTION_VERSION,
        "signature": signature,
        "status": "RUNNING",
        "gpu": props.name,
        "memory_policy": policy,
        "batch_size": batch_size,
        "accumulation": accumulation,
        "min_headroom_gib": min_headroom_gib,
        "phases": {},
        "encoder_tokens": len(probe.encoder_ids),
        "decoder_tokens": len(probe.decoder_prefix_ids) + len(probe.target_ids) - 1,
        "optimizer_updates_retained": 0,
    }
    path = Path(state_path) if state_path is not None else None
    if path is not None and path.exists():
        previous = json.loads(path.read_text())
        verify_report(previous)
        if previous.get("signature") != signature:
            raise ValueError("Saved memory guard identity changed; use a fresh run directory.")
        report = previous

    def emit(event):
        if progress:
            progress(
                {
                    "batch_size": batch_size,
                    "accumulation": accumulation,
                    **policy,
                    "encoder_tokens": report["encoder_tokens"],
                    "decoder_tokens": report["decoder_tokens"],
                    **event,
                }
            )

    def save():
        nonlocal report
        report = seal_report(report)
        if path is not None:
            atomic_json(path, report)
        if on_saved:
            on_saved()

    phases = ("training", "evaluation")
    completed = [report["phases"].get(p, {}) for p in phases]
    baseline = torch.cuda.memory_allocated(device) / 2**30
    extra = max((p.get("additional_peak_gib", 0) for p in completed), default=0)
    current_capacity(device, extra, min_headroom_gib)
    if all(p.get("status") == "COMPLETED" for p in completed):
        report["status"] = "VALIDATED_ON_THIS_GPU"
        report.pop("error", None)
        save()
        emit(
            {
                "stage": "preflight_reused",
                "items_done": 2,
                "items_total": 2,
                "message": "Memory guard 2/2 reused; current GPU capacity checked. No sweep.",
            }
        )
        return report

    if training_policy is None:
        raise ValueError("Memory guard requires the actual training optimizer policy.")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    old_mode = model.training
    parameters = [p for p in model.parameters() if p.requires_grad]
    backups = [p.detach().cpu().clone() for p in parameters]
    optimizer = None
    try:
        report.pop("error", None)
        save()
        emit(
            {
                "stage": "preflight_optimizer_init",
                "items_done": 0,
                "items_total": 2,
                "message": "Memory guard: allocate actual optimizer states; no backend search.",
            }
        )
        groups = build_training_groups(model, training_policy, total_steps=1)
        optimizer = build_policy_optimizer(groups, training_policy)
        for p in parameters:
            p.grad = torch.zeros_like(p)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for index, phase in enumerate(phases, 1):
            if report["phases"].get(phase, {}).get("status") == "COMPLETED":
                emit(
                    {
                        "stage": "preflight_phase_reused",
                        "items_done": index,
                        "items_total": 2,
                        "message": f"Memory guard {index}/2 {phase}: reused.",
                    }
                )
                continue
            report["status"] = "RUNNING"
            report["phases"][phase] = {"status": "RUNNING"}
            save()
            emit(
                {
                    "stage": "preflight_phase_started",
                    "items_done": index - 1,
                    "items_total": 2,
                    "preflight_phase": phase,
                    "message": f"Memory guard {index}/2 {phase}; {policy['attention_backend']}, "
                    f"chunk{policy['query_chunk_size']}, microbatch{batch_size}, accumulation{accumulation}.",
                }
            )
            row = _measure_phase(
                model, probe, optimizer, phase, device, min_headroom_gib, batch_size
            )
            row["additional_peak_gib"] = max(0, row["gpu_peak_allocated_gib"] - baseline)
            report["phases"][phase] = row
            save()
            emit(
                {
                    "stage": "preflight_phase_complete",
                    "items_done": index,
                    "items_total": 2,
                    "preflight_phase": phase,
                    **row,
                    "message": f"Memory guard {index}/2 {phase}: PASS; peak allocated "
                    f"{row['gpu_peak_allocated_gib']:.2f} GiB; headroom "
                    f"{row['estimated_peak_headroom_gib']:.2f} GiB.",
                }
            )
        report["status"] = "VALIDATED_ON_THIS_GPU"
        save()
        return report
    except BaseException as error:
        report["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        report["error"] = type(error).__name__ + ": " + str(error)[:1000]
        for phase_state in report["phases"].values():
            if phase_state.get("status") == "RUNNING":
                phase_state.update(status=report["status"], error=report["error"])
        report = seal_report(report)
        if path is not None:
            atomic_json(path, report)
        if on_saved:
            try:
                on_saved()
            except Exception as upload_error:
                error.add_note("Memory-guard status upload failed: " + type(upload_error).__name__)
        raise
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
