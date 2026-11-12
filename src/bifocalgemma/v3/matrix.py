"""V3 ten-model matrix training, evaluation, and notebook entry point."""

from __future__ import annotations

import copy
import gc
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint

from bifocalgemma.config import CheckpointRef, ExperimentSpec
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.releases import fetch_collection
from bifocalgemma.evaluation.context import (
    V2_EVALUATION_PROTOCOL,
)
from bifocalgemma.models.memory_efficient import (
    configure_memory_efficient_training,
)
from bifocalgemma.training.benchmark_upload import BenchmarkUploadQueue
from bifocalgemma.training.curated_sft import prepare_curated_training
from bifocalgemma.training.hub import (
    RemoteHeartbeat,
    benchmark_directory_name,
    download_model,
    recover_remote_run,
    target_fingerprint,
)
from bifocalgemma.training.offline_sft import evaluate_sft_loss
from bifocalgemma.training.policy import configure_policy_model, update_policy_learning_rates
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.schedule import (
    SharedSchedule,
    build_shared_schedule,
    persist_shared_schedule,
)
from bifocalgemma.training.source_identity import source_fingerprints
from bifocalgemma.training.stages import (
    StageLedger,
    atomic_json,
    latest_checkpoint,
)
from bifocalgemma.training.telemetry import TrainingDashboard
from bifocalgemma.v3.attention import configure_full_rank_cross_attention
from bifocalgemma.v3.benchmark import (
    resolve_optimal_eval_batch_size,
    run_batched_context_benchmark,
)
from bifocalgemma.v3.checkpoint import (
    load_matrix_adapters_for_inference,
    prune_matrix_checkpoints,
    restore_matrix_checkpoint,
    save_matrix_checkpoint,
    synchronize_matrix_v3,
)
from bifocalgemma.v3.constants import (
    ACCUMULATION,
    BATCH_SIZE,
    BENCHMARKS,
    CLEAN_DEFAULT_HF_REPO,
    CLEAN_RECIPE_VERSION,
    COLLECTION,
    DATASET_REPO,
    DATASET_REVISION,
    MATRIX_MODELS,
    TOTAL_STEPS,
    WARMUP_RATIO,
)
from bifocalgemma.v3.objectives import (
    compute_v3_step_loss,
    is_cross_bridge_parameter,
    resolve_diff_token_ids,
)
from bifocalgemma.v3.runtime import (
    ensure_hardware_memory_guards,
    patch_matrix_tokenization,
)
from bifocalgemma.v3.stage import configure_matrix_stage

logger = logging.getLogger(__name__)


def resolve_training_contract(spec, *, total_steps=None, stage1_steps=None, pool_directory=None) -> dict[str, Any]:
    """Declare local supplements and pilots explicitly; never replace the pinned pool."""
    dataset = {"dataset_repo": DATASET_REPO, "dataset_revision": DATASET_REVISION, "collection": COLLECTION}
    updates = TOTAL_STEPS
    if pool_directory is not None:
        manifest = json.loads((Path(pool_directory) / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("format") != "bifocalgemma_curated_pool_v1"
            or stable_fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest.get("fingerprint")):
            raise ValueError("Local pool manifest integrity failed.")
        size = len(manifest["record_ids"]["train"])
        if size < 1:
            raise ValueError("Local training pool is empty.")
        updates = (size + BATCH_SIZE * ACCUMULATION - 1) // (BATCH_SIZE * ACCUMULATION)
        dataset = {"dataset_repo": "local_curated_pool", "dataset_revision": manifest["fingerprint"],
                   "collection": Path(pool_directory).name}
    total = updates if total_steps is None else total_steps
    if type(total) is not int or not 1 <= total <= updates:
        raise ValueError("total_steps must fit the positive one-pass training horizon.")
    first = total if spec.encoder is None else (max(1, total // 2) if stage1_steps is None else stage1_steps)
    if type(first) is not int or not 0 <= first <= total:
        raise ValueError("stage1_steps must fall within the total training horizon.")
    return {**dataset, "total_steps": total, "stage1_steps": first, "stage2_steps": total - first}


def make_experiment_spec(model_key: str) -> ExperimentSpec:
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    decoder_ref = CheckpointRef(
        model_id=cfg["decoder_id"],
        revision=cfg["decoder_revision"],
        size=cfg["decoder_size"],
        role=cfg["decoder_role"],
    )
    if cfg["encoder_id"] is None:
        return ExperimentSpec(
            name=f"Baseline-{cfg['decoder_size']}-IT-{model_key}",
            encoder=None,
            decoder=decoder_ref,
            attention_mode="causal",
            tie_encoder_decoder_words=False,
            training_recipe="direct_sft",
            distillation=False,
            share_ple=False,
            architecture_version="normalized_merged_v2",
        )
    encoder_ref = CheckpointRef(
        model_id=cfg["encoder_id"],
        revision=cfg["encoder_revision"],
        size=cfg["encoder_size"],
        role=cfg["encoder_role"],
    )
    name = f"BifocalGemma-{cfg['encoder_size']}-{cfg['decoder_size']}-{model_key}"
    return ExperimentSpec(
        name=name,
        encoder=encoder_ref,
        decoder=decoder_ref,
        attention_mode="merged",
        tie_encoder_decoder_words=False,
        training_recipe="direct_sft",
        distillation=False,
        share_ple=False,
        architecture_version="normalized_merged_v2",
    )


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
) -> dict[str, Any]:
    """Execute one training stage (Stage 1 frozen or Stage 2 joint) with AdamW and batch=2."""
    ensure_hardware_memory_guards()
    run_dir = Path(run_dir).resolve()
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    if batch_size * accumulation != shared_schedule.effective_batch:
        raise ValueError("Execution batch disagrees with shared schedule.")
    if seed != shared_schedule.seed:
        raise ValueError("Execution seed disagrees with shared schedule.")
    if len(train_examples) != shared_schedule.total_examples:
        raise ValueError("Training cohort disagrees with shared schedule.")
    if max_steps < 1 or start_global_step < 0:
        raise ValueError("Invalid stage horizon.")
    shared_schedule.validate_cursor(start_global_step + max_steps)
    persist_shared_schedule(train_dir / "shared_schedule.json", shared_schedule)

    groups, stage_summary = configure_matrix_stage(
        model, policy, freeze_decoder=freeze_decoder, total_steps=max_steps
    )
    effective_freeze_decoder = stage_summary["freeze_decoder"]
    opt_cfg = policy.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        eps=float(opt_cfg.get("eps", 1e-8)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        foreach=False,
    )
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
            model, optimizer, Path(resume_from_checkpoint),
            expected_compatibility=compatibility, expected_schedule=shared_schedule,
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
        micro_batches = []
        window_indices = shared_schedule.window(global_step - 1)
        for offset in range(0, len(window_indices), batch_size):
            indices = window_indices[offset : offset + batch_size]
            items = [train_examples[i] for i in indices]
            raw_item_groups.append(items)
            micro_batches.append(collator(items))
        scheduled_records = shared_schedule.records[
            (global_step - 1) * shared_schedule.effective_batch : global_step * shared_schedule.effective_batch
        ]
        drawn_items = [item for items in raw_item_groups for item in items]
        if [item.instance_id for item in drawn_items] != [record.instance_id for record in scheduled_records]:
            raise ValueError("Tokenized loader order disagrees with the canonical shared schedule.")

        token_count = sum(int((b.labels != -100).sum()) for b in micro_batches)
        if not token_count:
            raise ValueError("Shared schedule window has no supervised target tokens.")

        active_diff_token_ids = getattr(model, "diff_token_ids", None) or diff_token_ids
        def backward_window(batches):
            loss_value = 0.0
            for batch_cpu in batches:
                batch = batch_cpu.to(device)
                count = int((batch_cpu.labels != -100).sum())
                scale = count / token_count
                step_loss = compute_v3_step_loss(
                    model, batch, stage_name=stage_name, diff_token_ids=active_diff_token_ids,
                    token_weighting="uniform_target_v1",
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
            mean_loss = backward_window(micro_batches)
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
            mean_loss = backward_window([collator([item]) for item in drawn_items])

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

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm, foreach=False)
        )
        rates = update_policy_learning_rates(optimizer, step)
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
            "learning_rates": rates,
            "step_seconds": step_seconds,
            "batch_size": batch_size,
            "accumulation": accumulation,
            "instance_ids": [item.instance_id for items in raw_item_groups for item in items],
            "schedule_signature": shared_schedule.signature,
            "example_cursor": shared_schedule.validate_cursor(global_step),
            "target_tokens_by_task": {
                task: sum(len(item.target_ids) for item, record in zip(drawn_items, scheduled_records, strict=True) if record.task == task)
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


def run_matrix_training(
    model_key: str,
    *,
    token: str | None = None,
    mo: Any = None,
    output_root: str | Path = "outputs/matrix",
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
    repo_id: str | None = None,
    standalone_repo_id: str | None = None,
    total_steps: int | None = None,
    stage1_steps: int | None = None,
    force_fresh: bool = False,
    pool_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Execute complete two-stage matrix training run with configurable steps and clean recipe version."""
    ensure_hardware_memory_guards()
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    policy = unified_recipe(cfg["family"])
    for grp_settings in policy.get("groups", {}).values():
        grp_settings["warmup_ratio"] = WARMUP_RATIO
    repo = (
        (repo_id.strip() if repo_id and repo_id.strip() else None)
        or cfg.get("repo")
        or CLEAN_DEFAULT_HF_REPO
    )
    subfolder = cfg.get("subfolder", model_key)

    contract = resolve_training_contract(spec, total_steps=total_steps, stage1_steps=stage1_steps,
                                         pool_directory=pool_directory)
    resolved_total, resolved_stage1, resolved_stage2 = (contract[key] for key in ("total_steps", "stage1_steps", "stage2_steps"))

    root = (Path(output_root) / model_key / "train").resolve()
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

    def wipe_stale_artifacts():
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
            raise ValueError("Incompatible or corrupt training artifacts; choose a new output_root or explicitly force_fresh.")

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
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
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

    def progress(event: dict[str, Any]):
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

    def snapshot(*, final: bool = False):
        if not upload or not token:
            return None
        queue.mark_pending()
        result = queue.attempt(
            lambda: synchronize_matrix_v3(repo, root, prefix, token=token, require_checkpoint=True),
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
    pool = Path(pool_directory) if pool_directory is not None else Path(cache) / "dataset" / COLLECTION
    if pool_directory is None:
        fetch_collection(
            DATASET_REPO, DATASET_REVISION, COLLECTION, pool, expected_role="train", token=token
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
    shared_schedule = build_shared_schedule(
        manifest["schedule_records"],
        dataset_identity={
            "repo": contract["dataset_repo"], "revision": contract["dataset_revision"],
            "collection": contract["collection"], "pool_fingerprint": manifest["pool_fingerprint"],
        },
        seed=17, effective_batch=BATCH_SIZE * ACCUMULATION,
    )
    if resolved_total > shared_schedule.total_updates:
        raise ValueError("Training horizon exceeds the immutable one-pass schedule.")
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
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
        "schedule_signature": shared_schedule.signature,
        "schedule_total_updates": shared_schedule.total_updates,
        "training_semantics_sha256": identity["training_semantics_sha256"],
        "sampling_policy": "shared_one_pass_without_replacement_v1",
        "loss_policy": "uniform_target_v1",
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
    stage2_report = previous_report.get("stage2")
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
                batch_size=BATCH_SIZE,
                accumulation=ACCUMULATION,
                clip_norm=1.0,
                seed=17,
                progress=progress,
                checkpoint_callback=lambda p: snapshot(),
                resume_from_checkpoint=resume_ckpt if current_global_step > 0 else None,
                decoder=decoder_tokenizer,
                total_steps=resolved_total,
                shared_schedule=shared_schedule,
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
                batch_size=BATCH_SIZE,
                accumulation=ACCUMULATION,
                clip_norm=1.0,
                seed=17,
                progress=progress,
                checkpoint_callback=lambda p: snapshot(),
                resume_from_checkpoint=resume_ckpt if current_global_step > 0 else None,
                decoder=decoder_tokenizer,
                total_steps=resolved_total,
                shared_schedule=shared_schedule,
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
                batch_size=BATCH_SIZE,
                accumulation=ACCUMULATION,
                clip_norm=1.0,
                seed=17,
                progress=progress,
                checkpoint_callback=lambda p: snapshot(),
                resume_from_checkpoint=resume_ckpt,
                decoder=decoder_tokenizer,
                total_steps=resolved_total,
                shared_schedule=shared_schedule,
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
            model=model, optimizer=None, checkpoint_dir=Path(final_ckpt),
            expected_compatibility=compatibility, expected_schedule=shared_schedule,
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
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
        "provenance": provenance,
        "schedule_signature": shared_schedule.signature,
        "schedule_total_examples": shared_schedule.total_examples,
        "schedule_total_updates": shared_schedule.total_updates,
        "realized_example_cursor": shared_schedule.validate_cursor(resolved_total),
        "realized_id_order_sha256": stable_fingerprint(shared_schedule.ordered_ids[:shared_schedule.validate_cursor(resolved_total)]),
        "loss_policy": "uniform_target_v1",
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
    output_root: str | Path = "outputs/matrix",
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
    repo_id: str | None = None,
    standalone_repo_id: str | None = None,
    total_steps: int | None = None,
    stage1_steps: int | None = None,
    force_fresh: bool = False,
    pool_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Execute 8-benchmark suite (6 full first + 2 deterministic 100-sample last) with dynamic batching."""
    ensure_hardware_memory_guards()
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    policy = unified_recipe(cfg["family"])
    for grp_settings in policy.get("groups", {}).values():
        grp_settings["warmup_ratio"] = WARMUP_RATIO
    repo = (repo_id.strip() if repo_id and repo_id.strip() else None) or cfg["repo"]
    subfolder = cfg.get("subfolder", model_key)

    contract = resolve_training_contract(spec, total_steps=total_steps, stage1_steps=stage1_steps,
                                         pool_directory=pool_directory)
    resolved_total, resolved_stage1, resolved_stage2 = (contract[key] for key in ("total_steps", "stage1_steps", "stage2_steps"))

    root = (Path(output_root) / model_key / "train").resolve()
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
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
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

    def progress(event: dict[str, Any]):
        dashboard(event)
        if heartbeat:
            heartbeat(event, force=event.get("stage") in {"benchmark_completed"})

    def snapshot(*, final: bool = False):
        if not upload or not token:
            return None
        queue.mark_pending()
        result = queue.attempt(
            lambda: synchronize_matrix_v3(repo, root, prefix, token=token, require_checkpoint=True),
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
            if (compat.get("model_spec") != spec.to_dict()
                or compat.get("training_semantics_sha256") != identity["training_semantics_sha256"]
                or m_data.get("global_step") != resolved_total):
                raise ValueError("Benchmark checkpoint does not match the completed training protocol.")
        except json.JSONDecodeError:
            raise ValueError(f"Corrupt manifest in {manifest_file}.")

    checkpoint_identity = json.loads(
        (Path(checkpoint_path) / "READY.json").read_text(encoding="utf-8")
    )

    reports = []
    missing_jobs = []

    for item in BENCHMARKS:
        settings = {
            "benchmark": item,
            "execution": True,
            "protocol": V2_EVALUATION_PROTOCOL,
            "snapshot": "trained",
        }
        dest = root / "benchmarks" / benchmark_directory_name(settings)
        missing_jobs.append((item, settings, dest))

    if not missing_jobs:
        snapshot(final=True)
        progress(
            {
                "stage": "benchmark_completed",
                "items_done": len(reports),
                "items_total": len(BENCHMARKS),
            }
        )
        return {
            "status": "COMPLETED",
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
    for item, settings, destination in missing_jobs:
        collection_dir = Path(cache) / "benchmarks" / item["collection"]
        fetch_collection(
            DATASET_REPO,
            DATASET_REVISION,
            item["collection"],
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
        snapshot(final=False)

    snapshot(final=True)
    progress(
        {
            "stage": "benchmark_completed",
            "items_done": len(reports),
            "items_total": len(BENCHMARKS),
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
    output_root: str | Path = "outputs/matrix",
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
    repo_id: str | None = None,
    standalone_repo_id: str | None = None,
    total_steps: int | None = None,
    stage1_steps: int | None = None,
    force_fresh: bool = False,
    pool_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Notebook entry point: execute matrix training curriculum, then run 8-benchmark suite."""
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
        force_fresh=force_fresh,
        pool_directory=pool_directory,
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
        force_fresh=force_fresh,
        pool_directory=pool_directory,
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
