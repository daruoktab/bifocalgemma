"""Ten-model matrix training, evaluation, and notebook entry point."""

from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint

import bifocalgemma.config as _config
from bifocalgemma.config import (
    ACCUMULATION,
    BATCH_SIZE,
    BENCHMARKS,
    CLEAN_DEFAULT_HF_REPO,
    CLEAN_RECIPE_VERSION,
    COLLECTION,
    DATASET_REPO,
    DATASET_REVISION,
    MATRIX_MODELS,
    MATRIX_RUN_NAMESPACE,
    TOTAL_STEPS,
    V4_BENCHMARKS,
    V4_COLLECTION,
    make_experiment_spec,
)
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.releases import fetch_collection
from bifocalgemma.evaluation.benchmark import (
    resolve_optimal_eval_batch_size,
    run_batched_context_benchmark,
)
from bifocalgemma.evaluation.context import V2_EVALUATION_PROTOCOL
from bifocalgemma.evaluation.protocol import (
    BENCHMARK_RESET_ID,
    EFFECTIVE_BENCHMARK_PROTOCOL,
    scorer_fingerprint,
)
from bifocalgemma.models.cross_attention import configure_full_rank_cross_attention
from bifocalgemma.models.memory_efficient import (
    configure_memory_efficient_training,
)
from bifocalgemma.training.benchmark_reset import reset_local_benchmarks
from bifocalgemma.training.benchmark_upload import BenchmarkUploadQueue
from bifocalgemma.training.checkpoint import (
    load_matrix_adapters_for_inference,
    prune_matrix_checkpoints,
    restore_matrix_checkpoint,
    save_matrix_checkpoint,
    synchronize_matrix,
)
from bifocalgemma.training.curated_sft import prepare_curated_training
from bifocalgemma.training.hub import (
    RemoteHeartbeat,
    benchmark_directory_name,
    download_model,
    recover_remote_run,
    target_fingerprint,
)
from bifocalgemma.training.objectives import (
    compute_example_loss_weights,
    compute_step_loss,
    is_cross_bridge_parameter,
    resolve_diff_token_ids,
)
from bifocalgemma.training.offline_sft import evaluate_sft_loss
from bifocalgemma.training.policy import configure_policy_model, update_policy_learning_rates
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.runtime import (
    ensure_hardware_memory_guards,
    patch_matrix_tokenization,
)
from bifocalgemma.training.schedule import (
    ScheduleRecord,
    SharedSchedule,
    _digest,
    build_shared_schedule,
    persist_shared_schedule,
)
from bifocalgemma.training.source_identity import source_fingerprints
from bifocalgemma.training.stage import (
    build_bifocalgemma_adamw,
    clip_optimizer_group_grads,
    configure_matrix_stage,
)
from bifocalgemma.training.stages import (
    StageLedger,
    atomic_json,
    latest_checkpoint,
)
from bifocalgemma.training.telemetry import TrainingDashboard

logger = logging.getLogger(__name__)


def _validate_num_epochs(num_epochs: int | float) -> int | float:
    if (
        isinstance(num_epochs, bool)
        or not isinstance(num_epochs, (int, float))
        or not math.isfinite(num_epochs)
        or num_epochs <= 0
    ):
        raise ValueError("num_epochs must be a finite positive number.")
    return int(num_epochs) if float(num_epochs).is_integer() else float(num_epochs)


def _resolve_example_diff_flag(item: Any) -> bool | None:
    """Resolve optional explicit unified-diff flag from a TokenizedRepairExample's metadata."""
    meta = getattr(item, "metadata", None)
    if not isinstance(meta, dict):
        return None
    if meta.get("is_unified_diff") is not None:
        return bool(meta["is_unified_diff"])
    target_format = meta.get("target_format")
    if target_format is not None:
        return str(target_format).strip().lower() == "unified_diff"
    return None


def _build_schedule_for_epochs(
    schedule_records: list[dict[str, Any]],
    *,
    dataset_identity: dict[str, Any],
    seed: int,
    effective_batch: int,
    num_epochs: int | float = 1,
) -> SharedSchedule:
    """Build deterministic shared schedule supporting multi-epoch (`1`, `1.5`, `2`, ...) runs.

    Delegates `num_epochs` directly to `build_shared_schedule` when supported, and falls back
    to deterministic per-epoch permutation concatenation when called against a 1-epoch builder.
    """
    resolved_epochs = _validate_num_epochs(num_epochs)
    sig_params = inspect.signature(build_shared_schedule).parameters
    if "num_epochs" in sig_params:
        return build_shared_schedule(
            schedule_records,
            dataset_identity=dataset_identity,
            seed=seed,
            effective_batch=effective_batch,
            num_epochs=resolved_epochs,  # type: ignore[call-arg]
        )
    if resolved_epochs == 1:
        return build_shared_schedule(
            schedule_records,
            dataset_identity=dataset_identity,
            seed=seed,
            effective_batch=effective_batch,
        )
    base_count = len(schedule_records)
    if base_count < 1:
        raise ValueError("Cannot build an empty training schedule.")
    target_count = max(1, round(base_count * float(resolved_epochs)))
    full_epochs = math.ceil(target_count / base_count)
    all_records: list[ScheduleRecord] = []
    all_indices: list[int] = []
    epoch_identity = {**dataset_identity, "num_epochs": resolved_epochs}
    for ep in range(full_epochs):
        ep_sched = build_shared_schedule(
            schedule_records,
            dataset_identity=epoch_identity,
            seed=seed + ep,
            effective_batch=effective_batch,
        )
        all_records.extend(ep_sched.records)
        all_indices.extend(ep_sched.ordered_indices)
    trimmed_records = tuple(all_records[:target_count])
    trimmed_indices = tuple(all_indices[:target_count])
    prelim = SharedSchedule(
        records=trimmed_records,
        ordered_indices=trimmed_indices,
        dataset_identity_json=json.dumps(
            epoch_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ),
        seed=seed,
        effective_batch=effective_batch,
        signature="",
    )
    payload = prelim.to_manifest()
    del payload["signature"]
    return SharedSchedule(
        records=prelim.records,
        ordered_indices=prelim.ordered_indices,
        dataset_identity_json=prelim.dataset_identity_json,
        seed=seed,
        effective_batch=effective_batch,
        signature=_digest(payload),
    )


def _resolve_profile_settings(profile: str | None) -> dict[str, Any]:
    """Resolve dataset, batching, epoch, and benchmark defaults for v3 (legacy) or v4."""
    resolved = "v3" if profile is None else profile.strip().lower()
    if resolved not in {"v3", "v4"}:
        raise ValueError(f"Unknown training profile {profile!r}; expected 'v3' or 'v4'.")
    if resolved == "v4":
        v4_batch = int(getattr(_config, "V4_BATCH_SIZE", BATCH_SIZE))
        v4_accum = int(getattr(_config, "V4_ACCUMULATION", ACCUMULATION))
        return {
            "profile": "v4",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": str(getattr(_config, "V4_DATASET_REVISION", DATASET_REVISION)),
            "collection": str(getattr(_config, "V4_COLLECTION", V4_COLLECTION)),
            "batch_size": v4_batch,
            "accumulation": v4_accum,
            "default_epochs": float(getattr(_config, "V4_NUM_EPOCHS", 1.5)),
            "base_examples": int(
                getattr(
                    _config,
                    "V4_TOTAL_EXAMPLES",
                    getattr(_config, "TOTAL_EXAMPLES", TOTAL_STEPS * v4_batch * v4_accum),
                )
            ),
            "default_stage1_fraction": float(getattr(_config, "V4_STAGE1_FRACTION", 0.06)),
            "benchmarks": list(getattr(_config, "V4_BENCHMARKS", V4_BENCHMARKS)),
            "run_namespace": str(getattr(_config, "V4_MATRIX_RUN_NAMESPACE", "versionless-repair-v4")),
        }
    return {
        "profile": "v3",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "collection": COLLECTION,
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
        "default_epochs": 1,
        "base_examples": int(getattr(_config, "TOTAL_EXAMPLES", TOTAL_STEPS * BATCH_SIZE * ACCUMULATION)),
        "default_stage1_fraction": float(
            getattr(_config, "STAGE1_FRACTION", getattr(_config, "V4_STAGE1_FRACTION", 0.06))
        ),
        "benchmarks": list(BENCHMARKS),
        "run_namespace": MATRIX_RUN_NAMESPACE,
    }


def resolve_training_contract(
    spec: Any,
    *,
    total_steps: int | None = None,
    stage1_steps: int | None = None,
    stage1_fraction: float | None = None,
    num_epochs: int | float | None = None,
    pool_directory: str | Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Declare local supplements, multi-epoch horizons, and short warm-start vs 0%-freeze stage split."""
    prof = _resolve_profile_settings(profile)
    resolved_epochs = _validate_num_epochs(
        prof["default_epochs"] if num_epochs is None else num_epochs
    )
    dataset = {
        "dataset_repo": prof["dataset_repo"],
        "dataset_revision": prof["dataset_revision"],
        "collection": prof["collection"],
    }
    batch_size = int(prof["batch_size"])
    accumulation = int(prof["accumulation"])
    eff_batch = batch_size * accumulation
    if pool_directory is not None:
        manifest = json.loads((Path(pool_directory) / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("format") != "bifocalgemma_curated_pool_v1"
            or stable_fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
            != manifest.get("fingerprint")
        ):
            raise ValueError("Local pool manifest integrity failed.")
        size = len(manifest["record_ids"]["train"])
        if size < 1:
            raise ValueError("Local training pool is empty.")
        scheduled_examples = (
            size if resolved_epochs == 1 else max(1, round(size * float(resolved_epochs)))
        )
        updates = (scheduled_examples + eff_batch - 1) // eff_batch
        dataset = {
            "dataset_repo": "local_curated_pool",
            "dataset_revision": manifest["fingerprint"],
            "collection": Path(pool_directory).name,
        }
    else:
        if prof["profile"] == "v3" and resolved_epochs == 1:
            updates = TOTAL_STEPS
        else:
            base_examples = int(prof["base_examples"])
            scheduled_examples = (
                base_examples
                if resolved_epochs == 1
                else max(1, round(base_examples * float(resolved_epochs)))
            )
            updates = (scheduled_examples + eff_batch - 1) // eff_batch

    total = updates if total_steps is None else total_steps
    if type(total) is not int or not 1 <= total <= updates:
        raise ValueError("total_steps must fit the positive training horizon.")

    if spec.encoder is None:
        first = total
    elif stage1_steps is not None:
        first = stage1_steps
    else:
        default_frac = float(prof["default_stage1_fraction"])
        frac = default_frac if stage1_fraction is None else stage1_fraction
        if (
            isinstance(frac, bool)
            or not isinstance(frac, (int, float))
            or not math.isfinite(frac)
            or not 0.0 <= frac <= 1.0
        ):
            raise ValueError("stage1_fraction must lie in [0.0, 1.0].")
        if frac == 0.0:
            first = 0
        elif frac == 1.0:
            first = total
        else:
            first = max(1, round(total * float(frac)))
            if total > 1:
                first = min(first, total - 1)

    if type(first) is not int or not 0 <= first <= total:
        raise ValueError("stage1_steps must fall within the total training horizon.")

    schedule_mode = (
        "decoder_only"
        if spec.encoder is None
        else ("asymmetric_warmup_joint" if first == 0 else "short_warmstart_joint")
    )
    result = {
        **dataset,
        "total_steps": total,
        "stage1_steps": first,
        "stage2_steps": total - first,
        "stage1_fraction": (first / total) if total > 0 else 0.0,
        "num_epochs": resolved_epochs,
        "schedule_mode": schedule_mode,
    }
    if prof["profile"] == "v4":
        result["profile"] = "v4"
        result["batch_size"] = batch_size
        result["accumulation"] = accumulation
    return result


def run_matrix_stage(
    model: Any,
    train_examples: Any,
    dev_examples: Any,
    stage_name: str,
    *,
    policy: dict[str, Any],
    max_steps: int,
    freeze_decoder: bool,
    run_dir: Path,
    compatibility: dict[str, Any],
    start_global_step: int,
    batch_size: int = BATCH_SIZE,
    accumulation: int = ACCUMULATION,
    clip_norm: float = 1.0,
    seed: int = 17,
    progress: Any = None,
    checkpoint_callback: Any = None,
    save_interval_seconds: float = 900.0,
    save_every_steps: int = 500,
    resume_from_checkpoint: Path | str | None = None,
    shared_schedule: SharedSchedule,
    decoder: Any = None,
    diff_token_ids: dict[str, tuple[int, ...]] | None = None,
    total_steps: int | None = None,
    asymmetric_warmup: bool | None = None,
    loss_reduction: str | None = None,
    length_cap: int | None = None,
) -> dict[str, Any]:
    """Execute one training stage (Stage 1 warm-start or Stage 2 / Step-0 joint) with v4 AdamW."""
    ensure_hardware_memory_guards()
    run_dir = Path(run_dir).resolve()
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    if batch_size * accumulation != shared_schedule.effective_batch:
        raise ValueError("Execution batch disagrees with shared schedule.")
    if seed != shared_schedule.seed:
        raise ValueError("Execution seed disagrees with shared schedule.")
    schedule_base_examples = getattr(shared_schedule, "base_examples", None)
    schedule_num_epochs = getattr(shared_schedule, "num_epochs", None)
    is_multi_epoch_schedule = (
        schedule_base_examples is not None
        or (schedule_num_epochs is not None and float(schedule_num_epochs) != 1.0)
        or (len(set(shared_schedule.ordered_indices)) < len(shared_schedule.ordered_indices))
    )
    if schedule_base_examples is not None:
        if len(train_examples) != int(schedule_base_examples):
            raise ValueError("Training cohort disagrees with shared schedule.")
    elif is_multi_epoch_schedule:
        if not shared_schedule.ordered_indices or len(train_examples) <= max(
            shared_schedule.ordered_indices
        ):
            raise ValueError("Training cohort disagrees with multi-epoch shared schedule.")
    elif len(train_examples) != shared_schedule.total_examples:
        raise ValueError("Training cohort disagrees with shared schedule.")
    if max_steps < 1 or start_global_step < 0:
        raise ValueError("Invalid stage horizon.")
    shared_schedule.validate_cursor(start_global_step + max_steps)
    persist_shared_schedule(train_dir / "shared_schedule.json", shared_schedule)

    resolved_horizon = total_steps if total_steps is not None else (start_global_step + max_steps)
    resolved_loss_reduction = (
        loss_reduction
        if loss_reduction is not None
        else str(getattr(_config, "LOSS_REDUCTION", "per_example_mean"))
    )
    resolved_length_cap = (
        length_cap if length_cap is not None else int(getattr(_config, "LOSS_LENGTH_CAP", 256))
    )
    groups, stage_summary = configure_matrix_stage(
        model,
        policy,
        freeze_decoder=freeze_decoder,
        total_steps=resolved_horizon,
        warmup_offset=start_global_step,
        asymmetric_warmup=asymmetric_warmup,
    )
    effective_freeze_decoder = stage_summary["freeze_decoder"]
    optimizer = build_bifocalgemma_adamw(groups, policy=policy)
    trainable_params = [p for g in groups for p in g["params"]]

    decoder_snapshot = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if n.startswith("decoder.")
        and not is_cross_bridge_parameter(n)
        and n.endswith((".lora_A", ".lora_B"))
    }

    start_step = 0
    history: list[dict[str, Any]] = []

    if resume_from_checkpoint is not None:
        restored = restore_matrix_checkpoint(
            model,
            optimizer,
            Path(resume_from_checkpoint),
            expected_compatibility=compatibility,
            expected_schedule=shared_schedule,
            expected_stage=(stage_name, start_global_step, max_steps),
        )
        state = restored.get("training_state", {})
        if state.get("stage") == stage_name:
            start_step = state["global_step"] - start_global_step
            if state.get("step") != start_step or not 0 <= start_step <= max_steps:
                raise ValueError("Checkpoint stage cursor disagrees with stage horizon.")
            history = state.get("history", [])
        elif state["global_step"] != start_global_step:
            raise ValueError("Checkpoint is not at the stage transition boundary.")
    elif start_global_step:
        raise ValueError("A later stage requires the preceding trained checkpoint.")

    model.train()
    last_save_time = time.monotonic()
    collator = RepairCollator()

    for step in range(start_step, max_steps):
        global_step = start_global_step + step + 1
        step_started = time.perf_counter()

        raw_item_groups: list[list[Any]] = []
        raw_record_groups: list[list[Any]] = []
        micro_batches = []
        window_indices = shared_schedule.window(global_step - 1)
        scheduled_records = shared_schedule.records[
            (global_step - 1) * shared_schedule.effective_batch : global_step * shared_schedule.effective_batch
        ]
        for offset in range(0, len(window_indices), batch_size):
            indices = window_indices[offset : offset + batch_size]
            items = [train_examples[i] for i in indices]
            records_slice = list(scheduled_records[offset : offset + batch_size])
            raw_item_groups.append(items)
            raw_record_groups.append(records_slice)
            micro_batches.append(collator(items))
        drawn_items = [item for items in raw_item_groups for item in items]
        if [item.instance_id for item in drawn_items] != [record.instance_id for record in scheduled_records]:
            raise ValueError("Tokenized loader order disagrees with the canonical shared schedule.")

        token_count = sum(int((b.labels != -100).sum()) for b in micro_batches)
        if not token_count:
            raise ValueError("Shared schedule window has no supervised target tokens.")

        active_diff_token_ids = getattr(model, "diff_token_ids", None) or diff_token_ids

        def backward_window(
            batches: list[Any],
            item_groups: list[list[Any]],
            record_groups: list[list[Any]],
        ) -> float:
            batch_weights = [
                float(
                    compute_example_loss_weights(
                        batch_cpu.labels,
                        reduction=resolved_loss_reduction,
                        length_cap=resolved_length_cap,
                    )
                    .sum()
                    .item()
                )
                for batch_cpu in batches
            ]
            total_weight = sum(batch_weights)
            if total_weight <= 0.0:
                raise ValueError("Shared schedule window has zero effective loss weight.")
            loss_value = 0.0
            for batch_cpu, items_slice, records_slice, batch_weight in zip(
                batches, item_groups, record_groups, batch_weights, strict=True
            ):
                if batch_weight <= 0.0:
                    continue
                batch = batch_cpu.to(device)
                scale = batch_weight / total_weight
                batch_task_types = [
                    (
                        item.metadata.get("task_type")
                        if isinstance(getattr(item, "metadata", None), dict)
                        and item.metadata.get("task_type")
                        else record.task
                    )
                    for item, record in zip(items_slice, records_slice, strict=True)
                ]
                batch_diff_flags = [_resolve_example_diff_flag(item) for item in items_slice]
                step_loss = compute_step_loss(
                    model,
                    batch,
                    stage_name=stage_name,
                    diff_token_ids=active_diff_token_ids,
                    token_weighting="diff_marker_v1",
                    task_types=batch_task_types,
                    is_unified_diff=batch_diff_flags,
                    loss_reduction=resolved_loss_reduction,
                    length_cap=resolved_length_cap,
                )
                if step_loss is None or not bool(torch.isfinite(step_loss)):
                    raise RuntimeError(f"Nonfinite loss at step {step + 1} (global {global_step})")
                (step_loss * scale).backward()
                loss_value += step_loss.detach().item() * scale
                del step_loss, batch
            return loss_value

        optimizer.zero_grad(set_to_none=True)
        torch_rng = torch.get_rng_state()
        python_rng = random.getstate()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        try:
            mean_loss = backward_window(micro_batches, raw_item_groups, raw_record_groups)
        except torch.cuda.OutOfMemoryError:
            # Replay the complete update: a failed backward may have written partial
            # gradients. No parameter/optimizer/cursor advance happens on failure.
            optimizer.zero_grad(set_to_none=True)
            if not any(len(items) > 1 for items in raw_item_groups):
                raise
            torch.set_rng_state(torch_rng)
            random.setstate(python_rng)
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(cuda_rng)
                torch.cuda.empty_cache()
            mean_loss = backward_window(
                [collator([item]) for item in drawn_items],
                [[item] for item in drawn_items],
                [[record] for record in scheduled_records],
            )

        if effective_freeze_decoder:
            leaked = [
                n
                for n, p in model.named_parameters()
                if n.startswith("decoder.")
                and not is_cross_bridge_parameter(n)
                and p.grad is not None
                and bool(p.grad.abs().sum() > 0)
            ]
            if leaked:
                raise RuntimeError(
                    f"Frozen decoder received nonzero parameter gradients: {leaked[:3]}"
                )

        grads = [p.grad for p in trainable_params if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError(f"Missing or nonfinite gradients at step {global_step}.")

        grad_norm, group_grad_norms = clip_optimizer_group_grads(
            optimizer, default_clip_norm=clip_norm
        )
        rates = update_policy_learning_rates(optimizer, global_step - 1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and torch.cuda.memory_reserved(device) > 28 * 2**30:
            torch.cuda.empty_cache()

        step_seconds = time.perf_counter() - step_started
        row = {
            "step": global_step,
            "stage_step": step + 1,
            "stage": stage_name,
            "freeze_decoder": effective_freeze_decoder,
            "max_steps": total_steps if total_steps is not None else TOTAL_STEPS,
            "train_loss": mean_loss,
            "target_tokens": token_count,
            "gradient_norm": grad_norm,
            "group_gradient_norms": group_grad_norms,
            "learning_rates": rates,
            "step_seconds": step_seconds,
            "batch_size": batch_size,
            "accumulation": accumulation,
            "instance_ids": [item.instance_id for items in raw_item_groups for item in items],
            "schedule_signature": shared_schedule.signature,
            "example_cursor": shared_schedule.validate_cursor(global_step),
            "target_tokens_by_task": {
                task: sum(
                    len(item.target_ids)
                    for item, record in zip(drawn_items, scheduled_records, strict=True)
                    if record.task == task
                )
                for task in sorted({record.task for record in scheduled_records})
            },
            "examples_by_source": {
                source: sum(record.source == source for record in scheduled_records)
                for source in sorted({record.source for record in scheduled_records})
            },
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            ),
        }
        history.append(row)

        if progress:
            progress({"stage": "matrix_training_step", **row})

        is_stage_end = step + 1 == max_steps
        is_interval_step = global_step % save_every_steps == 0
        is_time_up = time.monotonic() - last_save_time >= save_interval_seconds

        if is_stage_end or is_interval_step or is_time_up:
            ckpt_path = save_matrix_checkpoint(
                model,
                optimizer,
                train_dir / f"checkpoint-{global_step:06d}",
                global_step=global_step,
                stage_name=stage_name,
                compatibility=compatibility,
                training_state={
                    "step": step + 1,
                    "global_step": global_step,
                    "schedule_signature": shared_schedule.signature,
                    "example_cursor": shared_schedule.validate_cursor(global_step),
                    "history": history[-100:],
                },
            )
            last_save_time = time.monotonic()
            prune_matrix_checkpoints(train_dir, keep_limit=2)
            if checkpoint_callback:
                checkpoint_callback(ckpt_path)

    if effective_freeze_decoder:
        max_decoder_delta = max(
            (
                float((model.get_parameter(n).detach().cpu() - before_t).abs().max())
                for n, before_t in decoder_snapshot.items()
            ),
            default=0.0,
        )
        if max_decoder_delta != 0.0:
            raise RuntimeError(f"Frozen decoder weights mutated during stage: {max_decoder_delta}")

    held_out = None
    if dev_examples:
        model_key = str(compatibility.get("model_key", "4base-4it"))
        eval_bs = (
            resolve_optimal_eval_batch_size(model_key, None, device=device)
            if batch_size >= BATCH_SIZE
            else batch_size
        )
        while True:
            try:
                held_out = evaluate_sft_loss(
                    model,
                    dev_examples,
                    batch_size=eval_bs,
                    device=device,
                    event_callback=progress,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if eval_bs <= 1:
                    raise
                eval_bs = max(1, eval_bs // 2)

    return {
        "stage": stage_name,
        "completed_steps": max_steps,
        "end_global_step": start_global_step + max_steps,
        "held_out_loss": held_out,
        "schedule": {
            "signature": shared_schedule.signature,
            "start_update": start_global_step,
            "end_update": start_global_step + max_steps,
            "synthetic_replacements": 0,
        },
        "last_checkpoint": str(train_dir / f"checkpoint-{(start_global_step + max_steps):06d}"),
    }


def _schedule_signature_matches(train_dir: Path, expected_signature: Any) -> bool:
    """Return True when the persisted schedule file matches and validates records against its signature."""
    sched_file = train_dir / "shared_schedule.json"
    if not isinstance(expected_signature, str) or not expected_signature or not sched_file.is_file():
        return False
    try:
        sched_data = json.loads(sched_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(sched_data, dict):
        return False
    if sched_data.get("signature") != expected_signature:
        return False
    ordered_examples = sched_data.get("ordered_examples")
    if not isinstance(ordered_examples, list) or not ordered_examples:
        return False
    if sched_data.get("total_examples") != len(ordered_examples):
        return False
    try:
        for record in ordered_examples:
            if not isinstance(record, dict):
                return False
            ScheduleRecord.from_mapping(record)
        payload = {k: v for k, v in sched_data.items() if k != "signature"}
        return _digest(payload) == expected_signature
    except Exception:
        return False


def _latest_checkpoint_matches(
    train_dir: Path,
    *,
    expected_step: int,
    expected_signature: Any,
    spec: Any,
    model_key: str,
    training_semantics_sha256: str,
    resolved_total: int,
    resolved_stage1: int,
    resolved_stage2: int,
) -> bool:
    """Return True when the latest retained checkpoint matches the finished training identity."""
    try:
        ckpt = latest_checkpoint(train_dir)
    except Exception:
        return False
    if not ckpt:
        return False
    manifest_path = Path(ckpt) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    compat = meta.get("compatibility", {})
    return (
        meta.get("global_step") == expected_step
        and compat.get("recipe_version") == CLEAN_RECIPE_VERSION
        and compat.get("experiment") == spec.name
        and compat.get("model_key") == model_key
        and compat.get("model_spec") == spec.to_dict()
        and compat.get("training_semantics_sha256") == training_semantics_sha256
        and compat.get("schedule_signature") == expected_signature
        and compat.get("total_steps") == resolved_total
        and compat.get("stage1_steps") == resolved_stage1
        and compat.get("stage2_steps") == resolved_stage2
    )


def _valid_safetensors_file(path: Path) -> bool:
    """Return True when path is a non-empty readable safetensors file containing at least one tensor."""
    if not path.is_file() or path.is_symlink():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return len(list(handle.keys())) > 0
    except Exception:
        return False


def _local_standalone_export_valid(export_dir: Path) -> bool:
    """Return True when export_dir contains valid config.json and safetensors weights or shards."""
    config_path = export_dir / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        return False
    try:
        cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(cfg_data, dict) or not cfg_data:
        return False
    single_weights = export_dir / "model.safetensors"
    if single_weights.exists():
        return _valid_safetensors_file(single_weights)
    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        return False
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    raw_shards = set(weight_map.values())
    if not raw_shards or not all(isinstance(s, str) and s.strip() for s in raw_shards):
        return False
    resolved_root = export_dir.resolve()
    for shard in sorted(raw_shards):
        shard_path = (export_dir / shard).resolve()
        if not shard_path.is_relative_to(resolved_root) or not _valid_safetensors_file(shard_path):
            return False
    return True


def _standalone_export_exists(
    export_dir: Path,
    *,
    upload: bool,
    token: str | None,
    repo: str,
    subfolder: str,
) -> bool:
    """Require validated local weights and matching remote bytes when uploading.

    A filename on the Hub cannot establish that an export is usable. Cold
    recovery without local export evidence falls back to checkpoint restoration
    and export instead of declaring the run complete from remote names alone.
    """
    local_valid = _local_standalone_export_valid(export_dir)
    if not local_valid:
        return False
    if upload and token and repo:
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=token).model_info(repo, files_metadata=True)
            remote_files = {entry.rfilename: entry for entry in info.siblings or []}
            names = {"config.json"}
            if (export_dir / "model.safetensors").is_file():
                names.add("model.safetensors")
            else:
                names.add("model.safetensors.index.json")
                index = json.loads((export_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
                names.update(index["weight_map"].values())
            for name in names:
                path = export_dir / name
                remote = remote_files.get(f"{subfolder}/pretrained_model/{name}")
                if remote is None or remote.size != path.stat().st_size:
                    return False
                with path.open("rb") as stream:
                    if remote.lfs is not None:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                        if digest != remote.lfs.sha256:
                            return False
                    else:
                        # Non-LFS Hub metadata uses the Git blob object ID.
                        blob = hashlib.sha1(usedforsecurity=False)
                        blob.update(f"blob {remote.size}\0".encode("ascii"))
                        while chunk := stream.read(1024 * 1024):
                            blob.update(chunk)
                        if blob.hexdigest() != remote.blob_id:
                            return False
            return True
        except Exception:
            return False
    return True


def _final_stage_info(spec: Any, resolved_stage2: int) -> tuple[str, str, str]:
    """Return `(ledger_stage, checkpoint_stage, report_key)` for the run's final training stage."""
    if spec.encoder is None:
        return "full_decoder_sft", "full_decoder_sft", "stage1"
    if resolved_stage2 > 0:
        return "stage2_joint", "joint", "stage2"
    return "stage1_frozen", "frozen_decoder", "stage1"


def _backfill_final_stage(
    ledger: StageLedger,
    train_dir: Path,
    *,
    final_stage: str,
    expected_ckpt_stage: str,
    resolved_total: int,
    stage_result: Any,
) -> bool:
    """Record a missing final stage receipt from an already-verified final checkpoint."""
    try:
        ckpt_path_str = latest_checkpoint(train_dir)
    except Exception:
        return False
    if not ckpt_path_str:
        return False
    ckpt_dir = Path(ckpt_path_str)
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        ckpt_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if ckpt_meta.get("global_step") != resolved_total or ckpt_meta.get("stage") != expected_ckpt_stage:
        return False
    ckpt_files = [p for p in ckpt_dir.glob("*") if p.is_file()]
    if not ckpt_files:
        return False
    ledger.finish(final_stage, ckpt_files, stage_result)
    return True


def _reusable_training_report(
    *,
    root: Path,
    train_dir: Path,
    rep_file: Path,
    ledger: StageLedger,
    identity: dict[str, Any],
    spec: Any,
    model_key: str,
    repo: str,
    subfolder: str,
    resolved_total: int,
    resolved_stage1: int,
    resolved_stage2: int,
    force_fresh: bool,
    standalone_repo_id: str | None,
    upload: bool,
    token: str | None,
) -> dict[str, Any] | None:
    """Return verified training_report.json payload when training + export are already complete."""
    if force_fresh or (standalone_repo_id and standalone_repo_id.strip()):
        return None
    if not rep_file.is_file():
        return None
    final_stage, expected_ckpt_stage, report_stage_key = _final_stage_info(spec, resolved_stage2)
    stage_completed = ledger.completed(final_stage)
    stage_status = ledger.state.get("stages", {}).get(final_stage, {}).get("status")
    if not stage_completed and stage_status == "FAILED":
        return None
    try:
        report_meta = json.loads(rep_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(report_meta, dict):
        return None
    if (
        report_meta.get("status") != "COMPLETED"
        or report_meta.get("export_error") is not None
        or report_meta.get("recipe_version") != CLEAN_RECIPE_VERSION
        or report_meta.get("model_key") != model_key
        or report_meta.get("spec") != spec.to_dict()
        or report_meta.get("repo") != repo
        or report_meta.get("subfolder") != subfolder
        or report_meta.get("global_step") != resolved_total
        or report_meta.get("training_semantics_sha256") != identity["training_semantics_sha256"]
    ):
        return None
    schedule_sig = report_meta.get("schedule_signature")
    if not _schedule_signature_matches(train_dir, schedule_sig):
        return None
    if not _latest_checkpoint_matches(
        train_dir,
        expected_step=resolved_total,
        expected_signature=schedule_sig,
        spec=spec,
        model_key=model_key,
        training_semantics_sha256=identity["training_semantics_sha256"],
        resolved_total=resolved_total,
        resolved_stage1=resolved_stage1,
        resolved_stage2=resolved_stage2,
    ):
        return None
    if not _standalone_export_exists(
        root / "pretrained_model",
        upload=upload,
        token=token,
        repo=repo,
        subfolder=subfolder,
    ):
        return None
    if not stage_completed:
        if not _backfill_final_stage(
            ledger,
            train_dir,
            final_stage=final_stage,
            expected_ckpt_stage=expected_ckpt_stage,
            resolved_total=resolved_total,
            stage_result=report_meta.get(report_stage_key),
        ):
            return None
    return report_meta


def run_matrix_training(
    model_key: str,
    *,
    token: str | None = None,
    mo: Any = None,
    output_root: str | Path | None = None,
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
    repo_id: str | None = None,
    standalone_repo_id: str | None = None,
    total_steps: int | None = None,
    stage1_steps: int | None = None,
    stage1_fraction: float | None = None,
    num_epochs: int | float | None = None,
    loss_reduction: str | None = None,
    length_cap: int | None = None,
    force_fresh: bool = False,
    pool_directory: str | Path | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Execute complete matrix training run with short warm-start or 0%-freeze asymmetric warmup."""
    ensure_hardware_memory_guards()
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    prof = _resolve_profile_settings(profile)
    resolved_output_root = (
        output_root if output_root is not None else f"outputs/matrix/{prof['run_namespace']}"
    )
    resolved_batch_size = int(prof["batch_size"])
    resolved_accumulation = int(prof["accumulation"])
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    policy = unified_recipe(cfg["family"])
    repo = (
        (repo_id.strip() if repo_id and repo_id.strip() else None)
        or cfg.get("repo")
        or CLEAN_DEFAULT_HF_REPO
    )
    subfolder = cfg.get("subfolder", model_key)

    contract = resolve_training_contract(
        spec,
        total_steps=total_steps,
        stage1_steps=stage1_steps,
        stage1_fraction=stage1_fraction,
        num_epochs=num_epochs,
        pool_directory=pool_directory,
        profile=profile,
    )
    resolved_total, resolved_stage1, resolved_stage2 = (
        contract[key] for key in ("total_steps", "stage1_steps", "stage2_steps")
    )
    resolved_epochs = contract["num_epochs"]
    resolved_loss_reduction = (
        loss_reduction
        if loss_reduction is not None
        else str(getattr(_config, "LOSS_REDUCTION", "per_example_mean"))
    )
    resolved_length_cap = (
        length_cap if length_cap is not None else int(getattr(_config, "LOSS_LENGTH_CAP", 256))
    )

    root = (Path(resolved_output_root) / model_key / "train").resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subfolder}/{spec.name}/train"

    dashboard = (
        monitor
        if monitor is not None
        else TrainingDashboard(mo, experiment=cfg["title"], device="cuda")
    )
    dashboard.run_dir = root

    train_dir = root / "training"
    bench_dir = root / "benchmarks"
    ledger_file = root / "pipeline_state.json"
    legacy_ledger_file = root / "ledger.json"
    rep_file = train_dir / "training_report.json"

    def wipe_stale_artifacts() -> None:
        for target in (train_dir, bench_dir):
            if target.is_symlink() or target.resolve().parent != root:
                raise ValueError("Refusing to clear artifacts outside the selected run directory.")
        if train_dir.exists():
            shutil.rmtree(train_dir, ignore_errors=True)
        if bench_dir.exists():
            shutil.rmtree(bench_dir, ignore_errors=True)
        if ledger_file.exists():
            ledger_file.unlink(missing_ok=True)
        if legacy_ledger_file.exists():
            legacy_ledger_file.unlink(missing_ok=True)

    def check_legacy_artifacts() -> bool:
        if legacy_ledger_file.is_file():
            try:
                l_data = json.loads(legacy_ledger_file.read_text(encoding="utf-8"))
                rec_ver = l_data.get("recipe_version") or l_data.get("identity", {}).get(
                    "recipe_version"
                )
                if rec_ver != CLEAN_RECIPE_VERSION:
                    return True
            except Exception:
                return True
        if ledger_file.is_file():
            try:
                l_data = json.loads(ledger_file.read_text(encoding="utf-8"))
                rec_ver = l_data.get("recipe_version") or l_data.get("identity", {}).get(
                    "recipe_version"
                )
                if rec_ver != CLEAN_RECIPE_VERSION:
                    return True
            except Exception:
                return True
        if rep_file.is_file():
            try:
                rep_data = json.loads(rep_file.read_text(encoding="utf-8"))
                if rep_data.get("recipe_version") != CLEAN_RECIPE_VERSION:
                    return True
            except Exception:
                return True
        try:
            existing_ckpt = latest_checkpoint(train_dir)
        except Exception:
            return True
        if existing_ckpt:
            manifest_file = Path(existing_ckpt) / "manifest.json"
            if manifest_file.is_file():
                try:
                    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    rec_ver = m_data.get("recipe_version") or m_data.get("compatibility", {}).get(
                        "recipe_version"
                    )
                    if rec_ver != CLEAN_RECIPE_VERSION:
                        return True
                    compat = m_data.get("compatibility", {})
                    if compat.get("experiment") and compat.get("experiment") != spec.name:
                        return True
                except Exception:
                    return True
            else:
                return True
        return False

    if force_fresh:
        wipe_stale_artifacts()
    else:
        if upload and token:
            try:
                recover_remote_run(repo, root, prefix, token=token)
            except ValueError:
                raise
            except Exception as _rec_exc:
                logger.warning("Remote recovery skipped: %s", _rec_exc)
        if check_legacy_artifacts():
            raise ValueError(
                "Incompatible or corrupt training artifacts; choose a new output_root or explicitly force_fresh."
            )

    identity = {
        "format": "bifocalgemma_pipeline_v1",
        "version": "bifocalgemma_v3_swiglu_cross",
        "recipe_version": CLEAN_RECIPE_VERSION,
        "training_semantics_sha256": source_fingerprints()["training_semantics_sha256"],
        "spec": spec.to_dict(),
        "policy": policy,
        **contract,
        "run_name": "train",
        "seed": 17,
        "total_steps": resolved_total,
        "stage1_steps": resolved_stage1,
        "stage2_steps": resolved_stage2,
        "batch_size": resolved_batch_size,
        "accumulation": resolved_accumulation,
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
    }
    ledger_file = root / "pipeline_state.json"
    ledger = StageLedger(root, identity)
    if not ledger.path.exists():
        atomic_json(ledger.path, ledger.state)

    heartbeat = RemoteHeartbeat(repo, root, prefix, token=token) if upload and token else None
    queue = BenchmarkUploadQueue(
        root,
        backoff_path=root / ".upload-backoff.json",
        progress=dashboard,
        interval=300,
        label="Checkpoint / results",
    )

    def progress(event: dict[str, Any]) -> None:
        event = {
            **event,
            "upload_pending": queue.state.get("pending", False),
            "final_synced": queue.state.get("final_synced", False),
        }
        dashboard(event)
        if heartbeat:
            heartbeat(
                event,
                force=event.get("stage")
                in {"training_completed", "benchmark_completed", "stage_completed"},
            )

    def snapshot(*, final: bool = False) -> Any:
        if not upload or not token:
            return None
        queue.mark_pending()
        result = queue.attempt(
            lambda: synchronize_matrix(repo, root, prefix, token=token, require_checkpoint=True),
            final=final,
        )
        if result and heartbeat:
            dashboard(
                {
                    "stage": "remote_snapshot_verified",
                    "message": "HF verified @ " + result["revision"],
                    "last_remote_checkpoint": result["retained_checkpoints"][-1]
                    if result["retained_checkpoints"]
                    else None,
                    "remote_revision": result["revision"],
                    "upload_pending": False,
                }
            )
            heartbeat(
                {
                    "remote_revision": result["revision"],
                    "last_remote_checkpoint": result["retained_checkpoints"][-1]
                    if result["retained_checkpoints"]
                    else None,
                    "upload_pending": False,
                    "final_synced": final,
                },
                force=final,
            )
        return result

    progress({"stage": "starting_training", "message": f"Starting {cfg['title']} ({model_key})"})

    reusable_report = _reusable_training_report(
        root=root,
        train_dir=train_dir,
        rep_file=rep_file,
        ledger=ledger,
        identity=identity,
        spec=spec,
        model_key=model_key,
        repo=repo,
        subfolder=subfolder,
        resolved_total=resolved_total,
        resolved_stage1=resolved_stage1,
        resolved_stage2=resolved_stage2,
        force_fresh=force_fresh,
        standalone_repo_id=standalone_repo_id,
        upload=upload,
        token=token,
    )
    if reusable_report is not None:
        progress({"stage": "training_completed", "step": resolved_total, "reused": True})
        return reusable_report

    # Download model backbone
    model, provenance = download_model(spec, cache, token, progress)
    configure_policy_model(model, copy.deepcopy(policy))
    configure_memory_efficient_training(model)
    cross_info = configure_full_rank_cross_attention(model)
    if spec.encoder is None:
        progress({"stage": "configured_decoder_baseline", **cross_info})
    else:
        progress({"stage": "configured_cross_attention", **cross_info})

    # Prepare curated dataset
    pool = (
        Path(pool_directory)
        if pool_directory is not None
        else Path(cache) / "dataset" / contract["collection"]
    )
    if pool_directory is None:
        fetch_collection(
            contract["dataset_repo"],
            contract["dataset_revision"],
            contract["collection"],
            pool,
            expected_role="train",
            token=token,
        )
    train_data, dev_data, manifest = prepare_curated_training(
        spec,
        pool,
        Path(cache) / "tokenizers",
        packed_cache=Path(cache) / "packed",
        progress=progress,
    )
    if pool_directory is not None and manifest["pool_fingerprint"] != contract["dataset_revision"]:
        raise ValueError("Local training pool changed during preparation.")
    manifest["matched_target_fingerprints"] = {
        "train": target_fingerprint(train_data),
        "dev": target_fingerprint(dev_data),
    }
    shared_schedule = _build_schedule_for_epochs(
        manifest["schedule_records"],
        dataset_identity={
            "repo": contract["dataset_repo"],
            "revision": contract["dataset_revision"],
            "collection": contract["collection"],
            "pool_fingerprint": manifest["pool_fingerprint"],
        },
        seed=17,
        effective_batch=resolved_batch_size * resolved_accumulation,
        num_epochs=resolved_epochs,
    )
    if resolved_total > shared_schedule.total_updates:
        raise ValueError("Training horizon exceeds the immutable training schedule.")
    persist_shared_schedule(train_dir / "shared_schedule.json", shared_schedule)

    decoder_tokenizer = None
    try:
        from bifocalgemma.data.tokenization import load_verified_tokenizer

        decoder_tokenizer, _ = load_verified_tokenizer(spec.decoder, Path(cache) / "tokenizers")
    except Exception as exc:
        logger.warning(
            "Could not load decoder tokenizer for diff token resolution; falling back to Gemma defaults: %s",
            exc,
        )
    if not hasattr(model, "diff_token_ids") or getattr(model, "diff_token_ids", None) is None:
        object.__setattr__(model, "diff_token_ids", resolve_diff_token_ids(decoder_tokenizer))

    compatibility = {
        "architecture": (
            "swiglu_dual_gate_decoupled_cross_v3"
            if spec.encoder is not None
            else "decoder_only_causal_v3"
        ),
        "recipe_version": CLEAN_RECIPE_VERSION,
        "experiment": spec.name,
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
        "model_spec": spec.to_dict(),
        "sources": provenance,
        "data_fingerprint": manifest["fingerprint"],
        "matched_target_fingerprints": manifest["matched_target_fingerprints"],
        "seed": 17,
        "batch_size": resolved_batch_size,
        "accumulation": resolved_accumulation,
        "num_epochs": resolved_epochs,
        "schedule_signature": shared_schedule.signature,
        "schedule_total_updates": shared_schedule.total_updates,
        "training_semantics_sha256": identity["training_semantics_sha256"],
        "sampling_policy": (
            "shared_one_pass_without_replacement_v1"
            if resolved_epochs == 1
            else f"shared_multi_epoch_v1:{resolved_epochs}"
        ),
        "loss_policy": "task_gated_diff_marker_v2",
        "loss_reduction": resolved_loss_reduction,
        "loss_length_cap": resolved_length_cap,
        "schedule_mode": contract["schedule_mode"],
        "training_policy": policy,
        "memory_policy": model.memory_policy,
        "total_steps": resolved_total,
        "stage1_steps": resolved_stage1,
        "stage2_steps": resolved_stage2,
    }

    # Determine resume point
    resume_ckpt = latest_checkpoint(train_dir)
    current_global_step = 0
    if resume_ckpt:
        manifest_meta = json.loads(
            (Path(resume_ckpt) / "manifest.json").read_text(encoding="utf-8")
        )
        current_global_step = manifest_meta.get("global_step", 0)
        if manifest_meta.get("compatibility") != compatibility:
            raise ValueError("Checkpoint training/data/model compatibility changed; choose a new run.")
        if not 0 <= current_global_step <= resolved_total:
            raise ValueError("Checkpoint cursor exceeds this run's horizon.")

    previous_report = json.loads(rep_file.read_text(encoding="utf-8")) if rep_file.is_file() else {}
    if previous_report and previous_report.get("schedule_signature") != shared_schedule.signature:
        raise ValueError("Existing training report belongs to a different shared schedule.")
    stage1_report = previous_report.get("stage1")
    if stage1_report is None:
        stage1_key = "full_decoder_sft" if spec.encoder is None else "stage1_frozen"
        stage1_report = ledger.state.get("stages", {}).get(stage1_key, {}).get("result")
    stage2_report = previous_report.get("stage2")
    if stage2_report is None and spec.encoder is not None and resolved_stage2 > 0:
        stage2_report = ledger.state.get("stages", {}).get("stage2_joint", {}).get("result")
    if spec.encoder is None:
        if current_global_step < resolved_total:
            progress(
                {
                    "stage": "starting_decoder_sft",
                    "step": current_global_step,
                    "max_steps": resolved_total,
                }
            )
            stage1_report = run_matrix_stage(
                model,
                train_data,
                dev_data,
                stage_name="full_decoder_sft",
                policy=copy.deepcopy(policy),
                max_steps=resolved_total,
                freeze_decoder=False,
                run_dir=root,
                compatibility=compatibility,
                start_global_step=0,
                batch_size=resolved_batch_size,
                accumulation=resolved_accumulation,
                clip_norm=1.0,
                seed=17,
                progress=progress,
                checkpoint_callback=lambda p: snapshot(),
                resume_from_checkpoint=resume_ckpt if current_global_step > 0 else None,
                decoder=decoder_tokenizer,
                total_steps=resolved_total,
                shared_schedule=shared_schedule,
                loss_reduction=resolved_loss_reduction,
                length_cap=resolved_length_cap,
            )
            progress(
                {
                    "stage": "decoder_sft_completed",
                    "last_checkpoint": stage1_report["last_checkpoint"],
                }
            )
            stage1_files = [
                p for p in Path(stage1_report["last_checkpoint"]).glob("*") if p.is_file()
            ]
            ledger.finish("full_decoder_sft", stage1_files, stage1_report)
            snapshot(final=True)
            stage2_report = None
    else:
        if current_global_step < resolved_stage1:
            progress({"stage": "starting_stage1_frozen", "step": current_global_step})
            stage1_report = run_matrix_stage(
                model,
                train_data,
                dev_data,
                stage_name="frozen_decoder",
                policy=copy.deepcopy(policy),
                max_steps=resolved_stage1,
                freeze_decoder=True,
                run_dir=root,
                compatibility=compatibility,
                start_global_step=0,
                batch_size=resolved_batch_size,
                accumulation=resolved_accumulation,
                clip_norm=1.0,
                seed=17,
                progress=progress,
                checkpoint_callback=lambda p: snapshot(),
                resume_from_checkpoint=resume_ckpt if current_global_step > 0 else None,
                decoder=decoder_tokenizer,
                total_steps=resolved_total,
                shared_schedule=shared_schedule,
                loss_reduction=resolved_loss_reduction,
                length_cap=resolved_length_cap,
            )
            progress(
                {
                    "stage": "stage1_completed",
                    "last_checkpoint": stage1_report["last_checkpoint"],
                }
            )
            stage1_files = [
                p for p in Path(stage1_report["last_checkpoint"]).glob("*") if p.is_file()
            ]
            ledger.finish("stage1_frozen", stage1_files, stage1_report)
            snapshot(final=True)
            current_global_step = resolved_stage1
            resume_ckpt = stage1_report["last_checkpoint"]

        if current_global_step < resolved_total:
            progress({"stage": "starting_stage2_joint", "step": current_global_step})
            stage2_steps_remaining = resolved_total - resolved_stage1
            stage2_report = run_matrix_stage(
                model,
                train_data,
                dev_data,
                stage_name="joint",
                policy=copy.deepcopy(policy),
                max_steps=stage2_steps_remaining,
                freeze_decoder=False,
                run_dir=root,
                compatibility=compatibility,
                start_global_step=resolved_stage1,
                batch_size=resolved_batch_size,
                accumulation=resolved_accumulation,
                clip_norm=1.0,
                seed=17,
                progress=progress,
                checkpoint_callback=lambda p: snapshot(),
                resume_from_checkpoint=resume_ckpt,
                decoder=decoder_tokenizer,
                total_steps=resolved_total,
                shared_schedule=shared_schedule,
                asymmetric_warmup=(resolved_stage1 == 0),
                loss_reduction=resolved_loss_reduction,
                length_cap=resolved_length_cap,
            )
            progress(
                {
                    "stage": "stage2_completed",
                    "last_checkpoint": stage2_report["last_checkpoint"],
                }
            )
            stage2_files = [
                p for p in Path(stage2_report["last_checkpoint"]).glob("*") if p.is_file()
            ]
            ledger.finish("stage2_joint", stage2_files, stage2_report)
            snapshot(final=True)

    # Ensure model has the latest trained weights and adapters loaded prior to export
    final_ckpt = latest_checkpoint(train_dir)
    if final_ckpt is not None:
        logger.info("Restoring latest trained checkpoint %s into model before export", final_ckpt)
        restore_matrix_checkpoint(
            model=model,
            optimizer=None,
            checkpoint_dir=Path(final_ckpt),
            expected_compatibility=compatibility,
            expected_schedule=shared_schedule,
        )
        final_stage, expected_ckpt_stage, report_stage_key = _final_stage_info(
            spec, resolved_stage2
        )
        if (
            not ledger.completed(final_stage)
            and ledger.state.get("stages", {}).get(final_stage, {}).get("status") != "FAILED"
        ):
            stage_payload = stage2_report if report_stage_key == "stage2" else stage1_report
            _backfill_final_stage(
                ledger,
                train_dir,
                final_stage=final_stage,
                expected_ckpt_stage=expected_ckpt_stage,
                resolved_total=resolved_total,
                stage_result=stage_payload,
            )

    # Export standalone Hugging Face PreTrainedModel artifact
    export_dir = root / "pretrained_model"
    export_succeeded = False
    export_error_msg = None
    try:
        from bifocalgemma.modeling_bifocalgemma import export_matrix_pretrained_model

        export_matrix_pretrained_model(
            model=model,
            spec=spec,
            export_dir=export_dir,
            tokenizer=decoder_tokenizer,
            push_to_hub=upload and bool(token),
            repo_id=repo if upload and token else None,
            token=token,
            path_in_repo=f"{subfolder}/pretrained_model",
            tokenizer_cache=Path(cache) / "tokenizers",
        )
        if standalone_repo_id and standalone_repo_id.strip() and upload and token:
            export_matrix_pretrained_model(
                model=model,
                spec=spec,
                export_dir=export_dir,
                tokenizer=decoder_tokenizer,
                push_to_hub=True,
                repo_id=standalone_repo_id.strip(),
                token=token,
                path_in_repo=None,
                tokenizer_cache=Path(cache) / "tokenizers",
            )
        export_succeeded = True
        progress({"stage": "pretrained_model_exported", "export_dir": str(export_dir)})
    except Exception as exc:
        export_error_msg = str(exc)
        logger.warning("Failed to export standalone pretrained model: %s", exc)

    final_report = {
        "status": "COMPLETED" if export_succeeded else "COMPLETED_WITHOUT_STANDALONE_EXPORT",
        "recipe_version": CLEAN_RECIPE_VERSION,
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
        "spec": spec.to_dict(),
        "global_step": resolved_total,
        "batch_size": resolved_batch_size,
        "accumulation": resolved_accumulation,
        "provenance": provenance,
        "schedule_signature": shared_schedule.signature,
        "schedule_total_examples": shared_schedule.total_examples,
        "schedule_total_updates": shared_schedule.total_updates,
        "realized_example_cursor": shared_schedule.validate_cursor(resolved_total),
        "realized_id_order_sha256": stable_fingerprint(
            shared_schedule.ordered_ids[: shared_schedule.validate_cursor(resolved_total)]
        ),
        "loss_policy": "task_gated_diff_marker_v2",
        "loss_reduction": resolved_loss_reduction,
        "loss_length_cap": resolved_length_cap,
        "num_epochs": resolved_epochs,
        "schedule_mode": contract["schedule_mode"],
        "explicit_prefix_pilot": resolved_total < shared_schedule.total_updates,
        "training_semantics_sha256": identity["training_semantics_sha256"],
        "stage1": stage1_report,
        "stage2": stage2_report,
        "last_checkpoint": str(train_dir / f"checkpoint-{resolved_total:06d}"),
        "pretrained_model": str(export_dir) if export_succeeded else None,
        "export_error": export_error_msg,
    }
    atomic_json(root / "training/training_report.json", final_report)
    snapshot(final=True)
    progress({"stage": "training_completed", "step": resolved_total})

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_report


def run_matrix_benchmark(
    model_key: str,
    *,
    token: str | None = None,
    mo: Any = None,
    output_root: str | Path | None = None,
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
    repo_id: str | None = None,
    standalone_repo_id: str | None = None,
    total_steps: int | None = None,
    stage1_steps: int | None = None,
    stage1_fraction: float | None = None,
    num_epochs: int | float | None = None,
    force_fresh: bool = False,
    pool_directory: str | Path | None = None,
    profile: str | None = None,
    benchmark_suite: list[dict[str, Any]] | None = None,
    context_mode: str | None = None,
) -> dict[str, Any]:
    """Execute matrix benchmark suite (8-benchmark v3 or 10-benchmark v4) with dynamic batching."""
    ensure_hardware_memory_guards()
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    prof = _resolve_profile_settings(profile)
    resolved_output_root = (
        output_root if output_root is not None else f"outputs/matrix/{prof['run_namespace']}"
    )
    resolved_batch_size = int(prof["batch_size"])
    resolved_accumulation = int(prof["accumulation"])
    active_benchmarks = (
        list(benchmark_suite) if benchmark_suite is not None else list(prof["benchmarks"])
    )
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    resolved_context_mode = (
        context_mode
        if context_mode is not None
        else (
            ("disjoint" if spec.encoder is not None else "full")
            if prof["profile"] == "v4"
            else None
        )
    )
    policy = unified_recipe(cfg["family"])
    repo = (
        (repo_id.strip() if repo_id and repo_id.strip() else None)
        or cfg.get("repo")
        or CLEAN_DEFAULT_HF_REPO
    )
    subfolder = cfg.get("subfolder", model_key)

    contract = resolve_training_contract(
        spec,
        total_steps=total_steps,
        stage1_steps=stage1_steps,
        stage1_fraction=stage1_fraction,
        num_epochs=num_epochs,
        pool_directory=pool_directory,
        profile=profile,
    )
    resolved_total, resolved_stage1, resolved_stage2 = (
        contract[key] for key in ("total_steps", "stage1_steps", "stage2_steps")
    )

    root = (Path(resolved_output_root) / model_key / "train").resolve()
    prefix = f"{subfolder}/{spec.name}/train"

    dashboard = (
        monitor
        if monitor is not None
        else TrainingDashboard(mo, experiment=cfg["title"], device="cuda")
    )
    dashboard.run_dir = root

    bench_dir = root / "benchmarks"
    if force_fresh:
        if bench_dir.exists():
            if bench_dir.is_symlink() or bench_dir.resolve().parent != root:
                raise ValueError("Refusing to clear benchmarks outside the selected run directory.")
            shutil.rmtree(bench_dir, ignore_errors=True)
    elif upload and token:
        try:
            recover_remote_run(repo, root, prefix, token=token)
        except ValueError:
            raise
        except Exception as _rec_exc:
            logger.warning("Remote benchmark recovery skipped: %s", _rec_exc)

    reset_local_benchmarks(root, BENCHMARK_RESET_ID)

    identity = {
        "format": "bifocalgemma_pipeline_v1",
        "version": "bifocalgemma_v3_swiglu_cross",
        "recipe_version": CLEAN_RECIPE_VERSION,
        "training_semantics_sha256": source_fingerprints()["training_semantics_sha256"],
        "spec": spec.to_dict(),
        "policy": policy,
        **contract,
        "run_name": "train",
        "seed": 17,
        "total_steps": resolved_total,
        "stage1_steps": resolved_stage1,
        "stage2_steps": resolved_stage2,
        "batch_size": resolved_batch_size,
        "accumulation": resolved_accumulation,
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
    }
    ledger = StageLedger(root, identity)
    if not ledger.path.exists():
        atomic_json(ledger.path, ledger.state)

    heartbeat = RemoteHeartbeat(repo, root, prefix, token=token) if upload and token else None
    queue = BenchmarkUploadQueue(
        root,
        backoff_path=root / ".upload-backoff.json",
        progress=dashboard,
        interval=1800,
        label="Benchmark upload",
    )

    def progress(event: dict[str, Any]) -> None:
        dashboard(event)
        if heartbeat:
            heartbeat(event, force=event.get("stage") in {"benchmark_completed"})

    def snapshot(*, final: bool = False) -> Any:
        if not upload or not token:
            return None
        queue.mark_pending()
        result = queue.attempt(
            lambda: synchronize_matrix(repo, root, prefix, token=token, require_checkpoint=True),
            final=final,
        )
        if result and heartbeat:
            heartbeat(
                {
                    "remote_revision": result["revision"],
                    "upload_pending": False,
                    "final_synced": final,
                },
                force=final,
            )
        return result

    checkpoint_path = latest_checkpoint(root / "training")
    if not checkpoint_path:
        raise ValueError(f"No completed training checkpoint found in {root / 'training'}.")

    manifest_file = Path(checkpoint_path) / "manifest.json"
    if manifest_file.is_file():
        try:
            m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            rec_ver = m_data.get("recipe_version") or m_data.get("compatibility", {}).get(
                "recipe_version"
            )
            if rec_ver != CLEAN_RECIPE_VERSION:
                raise ValueError(
                    f"Checkpoint in {checkpoint_path} has legacy recipe_version {rec_ver!r} "
                    f"(expected {CLEAN_RECIPE_VERSION!r}). Training must be run with clean recipe."
                )
            compat = m_data.get("compatibility", {})
            if (
                compat.get("model_spec") != spec.to_dict()
                or compat.get("training_semantics_sha256") != identity["training_semantics_sha256"]
                or m_data.get("global_step") != resolved_total
            ):
                raise ValueError("Benchmark checkpoint does not match the completed training protocol.")
        except json.JSONDecodeError:
            raise ValueError(f"Corrupt manifest in {manifest_file}.")

    checkpoint_identity = json.loads(
        (Path(checkpoint_path) / "READY.json").read_text(encoding="utf-8")
    )

    reports = []
    missing_jobs = []
    active_scorer_fp = scorer_fingerprint()

    for item in active_benchmarks:
        settings = {
            "benchmark": item,
            "execution": True,
            "protocol": V2_EVALUATION_PROTOCOL,
            "effective_protocol": EFFECTIVE_BENCHMARK_PROTOCOL,
            "reset_id": BENCHMARK_RESET_ID,
            "scorer_fingerprint": active_scorer_fp,
            "snapshot": "trained",
            "checkpoint_identity": checkpoint_identity,
            **({"context_mode": resolved_context_mode} if resolved_context_mode is not None else {}),
        }
        dest = root / "benchmarks" / benchmark_directory_name(settings)
        stage_key = f"benchmark_{item['collection']}"
        if not force_fresh and ledger.completed(stage_key, settings=settings):
            report_file = dest / "benchmark_report.json"
            if report_file.is_file():
                try:
                    reports.append(json.loads(report_file.read_text(encoding="utf-8")))
                    logger.info("Skipping completed benchmark %s", item["collection"])
                    continue
                except Exception:
                    pass
            stage_result = ledger.state.get("stages", {}).get(stage_key, {}).get("result")
            if isinstance(stage_result, dict):
                reports.append(stage_result)
                logger.info("Skipping completed benchmark %s via stage ledger", item["collection"])
                continue
        missing_jobs.append((item, settings, dest))

    if not missing_jobs:
        snapshot(final=True)
        progress(
            {
                "stage": "benchmark_completed",
                "items_done": len(reports),
                "items_total": len(active_benchmarks),
            }
        )
        return {
            "status": "COMPLETED" if not queue.state.get("pending") else "PENDING_UPLOAD",
            "model": spec.name,
            "repo": repo,
            "subfolder": subfolder,
            "benchmarks": reports,
        }

    # Load model for inference
    model, _ = download_model(spec, cache, token, progress)
    configure_policy_model(model, copy.deepcopy(policy))
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    load_matrix_adapters_for_inference(model, checkpoint_path)

    device = next(model.parameters()).device
    bench_repo = prof["dataset_repo"]
    bench_revision = prof["dataset_revision"]
    for item, settings, destination in missing_jobs:
        source_collection = str(item.get("source_collection") or item["collection"])
        collection_dir = Path(cache) / "benchmarks" / source_collection
        fetch_collection(
            bench_repo,
            bench_revision,
            source_collection,
            collection_dir,
            expected_role="eval",
            include_gold=True,
            token=token,
        )
        eval_batch_size = resolve_optimal_eval_batch_size(model_key, item, device=device)
        report = run_batched_context_benchmark(
            model,
            spec,
            collection_dir,
            destination,
            Path(cache) / "tokenizers",
            checkpoint_identity=checkpoint_identity,
            model_key=model_key,
            benchmark_item=item,
            batch_size=eval_batch_size,
            encoder_limit=item["encoder_limit"],
            decoder_limit=item["decoder_limit"],
            max_new_tokens=item["max_new_tokens"],
            allow_execution=True,
            progress=progress,
            prediction_callback=lambda: snapshot(),
            upload_every=100,
            use_cache=True,
            context_mode=resolved_context_mode,
            repair_mode=item.get("repair_mode"),
        )
        reports.append(report)
        ledger.finish(
            f"benchmark_{item['collection']}",
            [
                destination / "benchmark_report.json",
                destination / "predictions.jsonl",
                destination / "evaluation_state.json",
            ],
            report,
            settings=settings,
        )
        snapshot(final=True)

    snapshot(final=True)
    progress(
        {
            "stage": "benchmark_completed",
            "items_done": len(reports),
            "items_total": len(active_benchmarks),
        }
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "COMPLETED" if not queue.state.get("pending") else "PENDING_UPLOAD",
        "model": spec.name,
        "repo": repo,
        "subfolder": subfolder,
        "benchmarks": reports,
    }


def run_matrix_training_and_benchmark(
    model_key: str,
    *,
    token: str | None = None,
    mo: Any = None,
    output_root: str | Path | None = None,
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
    repo_id: str | None = None,
    standalone_repo_id: str | None = None,
    total_steps: int | None = None,
    stage1_steps: int | None = None,
    stage1_fraction: float | None = None,
    num_epochs: int | float | None = None,
    loss_reduction: str | None = None,
    length_cap: int | None = None,
    force_fresh: bool = False,
    pool_directory: str | Path | None = None,
    profile: str | None = None,
    benchmark_suite: list[dict[str, Any]] | None = None,
    context_mode: str | None = None,
) -> dict[str, Any]:
    """Notebook entry point: execute matrix training curriculum, then run benchmark suite."""
    training_res = run_matrix_training(
        model_key,
        token=token,
        mo=mo,
        output_root=output_root,
        cache=cache,
        upload=upload,
        monitor=monitor,
        repo_id=repo_id,
        standalone_repo_id=standalone_repo_id,
        total_steps=total_steps,
        stage1_steps=stage1_steps,
        stage1_fraction=stage1_fraction,
        num_epochs=num_epochs,
        loss_reduction=loss_reduction,
        length_cap=length_cap,
        force_fresh=force_fresh,
        pool_directory=pool_directory,
        profile=profile,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluation_res = run_matrix_benchmark(
        model_key,
        token=token,
        mo=mo,
        output_root=output_root,
        cache=cache,
        upload=upload,
        monitor=monitor,
        repo_id=repo_id,
        standalone_repo_id=standalone_repo_id,
        total_steps=total_steps,
        stage1_steps=stage1_steps,
        stage1_fraction=stage1_fraction,
        num_epochs=num_epochs,
        force_fresh=force_fresh,
        pool_directory=pool_directory,
        profile=profile,
        benchmark_suite=benchmark_suite,
        context_mode=context_mode,
    )
    return {
        "status": evaluation_res["status"],
        "model": evaluation_res["model"],
        "model_key": model_key,
        "repo": evaluation_res["repo"],
        "subfolder": evaluation_res.get("subfolder", model_key),
        "training": training_res,
        "benchmarks": evaluation_res["benchmarks"],
    }
