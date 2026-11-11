"""Bounded LoRA SFT/checkpoint helpers for the matched Molab notebooks.

Adapter-only checkpoints assume the pinned pretrained sources are available.
No RL, model publication, code execution or temporal attention cache.
"""

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.training.policy import (
    build_training_groups,
    optimizer_group_manifest,
    update_policy_learning_rates,
)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_sft_checkpoint(
    model,
    optimizer,
    state,
    run_dir,
    compatibility,
    *,
    checkpoint_format="bifocalgemma_notebook_sft_v1",
):
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"checkpoint-{state['global_step']:06d}"
    if final.exists():
        raise FileExistsError(f"Checkpoint exists; refusing overwrite: {final}")
    stage = root / (".partial-" + uuid4().hex)
    stage.mkdir()
    parameters = {
        name: p.detach().cpu().contiguous()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    if not all(bool(torch.isfinite(value).all()) for value in parameters.values()):
        raise ValueError("Refusing to save nonfinite trainable parameters.")
    save_file(parameters, str(stage / "adapters.safetensors"))
    payload = {
        "optimizer": cpu_tree(optimizer.state_dict()),
        "training_state": state,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(payload, stage / "trainer_state.pt")
    metadata = {
        "format": checkpoint_format,
        "global_step": state["global_step"],
        "compatibility": compatibility,
        "trainable_names": sorted(parameters),
    }
    if checkpoint_format in {"bifocalgemma_notebook_sft_v2", "bifocalgemma_normalized_sft_v3"}:
        metadata["optimizer_groups"] = optimizer_group_manifest(optimizer.param_groups)
    (stage / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    marker = {
        name: file_digest(stage / name)
        for name in ("adapters.safetensors", "trainer_state.pt", "manifest.json")
    }
    (stage / "READY.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if stage.resolve().parent != root or final.resolve().parent != root:
        raise ValueError("Checkpoint targets escaped run directory.")
    from bifocalgemma.training.stages import retry_windows_file_operation

    retry_windows_file_operation(lambda: stage.rename(final))
    return str(final)


def restore_sft_checkpoint(
    model,
    optimizer,
    checkpoint_dir,
    compatibility,
    dataset_size,
    *,
    checkpoint_format="bifocalgemma_notebook_sft_v1",
):
    root = Path(checkpoint_dir)
    marker = json.loads((root / "READY.json").read_text(encoding="utf-8"))
    required = {"adapters.safetensors", "trainer_state.pt", "manifest.json"}
    if set(marker) != required or any(
        file_digest(root / name) != marker[name] for name in required
    ):
        raise ValueError("Checkpoint is incomplete or checksum validation failed.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != checkpoint_format:
        raise ValueError(
            "Checkpoint format/architecture mismatch: grouped policy requires v2 or its versioned successor; legacy checkpoints cannot resume unified_v2. Use the original notebook for legacy checkpoints."
        )
    if manifest["compatibility"] != compatibility:
        raise ValueError("Checkpoint experiment/data/tokenizer/optimizer config mismatch.")
    if checkpoint_format in {"bifocalgemma_notebook_sft_v2", "bifocalgemma_normalized_sft_v3"}:
        if manifest.get("optimizer_groups") != optimizer_group_manifest(optimizer.param_groups):
            raise ValueError("Checkpoint optimizer group order/names/policy mismatch.")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if sorted(parameters) != manifest["trainable_names"]:
        raise ValueError("Checkpoint trainable parameter names differ.")
    payload = torch.load(root / "trainer_state.pt", map_location="cpu", weights_only=True)
    if checkpoint_format in {"bifocalgemma_notebook_sft_v2", "bifocalgemma_normalized_sft_v3"}:
        saved_groups = payload["optimizer"]["param_groups"]
        if optimizer_group_manifest(saved_groups) != optimizer_group_manifest(
            optimizer.param_groups
        ):
            raise ValueError("Saved optimizer groups differ from current policy.")
        if any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, optimizer.param_groups, strict=True)
        ):
            raise ValueError("Saved optimizer parameter counts differ.")
    state = payload["training_state"]
    order, cursor = state["order"], state["cursor"]
    if sorted(order) != list(range(dataset_size)) or not 0 <= cursor <= len(order):
        raise ValueError("Invalid checkpoint sampler order/cursor.")
    if state["global_step"] != manifest["global_step"]:
        raise ValueError("Checkpoint step mismatch.")
    with safe_open(str(root / "adapters.safetensors"), framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(parameters):
            raise ValueError("Checkpoint adapter keys differ.")
        for name, p in parameters.items():
            if tuple(reader.get_slice(name).get_shape()) != tuple(p.shape):
                raise ValueError(f"Checkpoint shape mismatch: {name}")
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(reader.get_tensor(name))
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    if payload["cuda_rng"]:
        if len(payload["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA device count differs.")
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return state


def validation_batches(examples, batch_size):
    """Keep source/task attribution exact without changing the evaluation cohort."""
    pending = []
    previous_key = None
    for item in examples:
        key = (item.metadata.get("source", "unknown"), item.metadata.get("task_type", "repair"))
        if pending and (len(pending) == batch_size or key != previous_key):
            yield pending
            pending = []
        pending.append(item)
        previous_key = key
    if pending:
        yield pending


def evaluate_sft_loss(model, examples, *, batch_size, device, event_callback=None, step=0):
    previous = model.training
    model.eval()
    total, tokens, by_source, by_task = 0.0, 0, {}, {}
    if event_callback:
        event_callback(
            {
                "stage": "validation_started",
                "step": step,
                "eval_examples_done": 0,
                "eval_examples_total": len(examples),
            }
        )
    try:
        with torch.no_grad():
            done = 0
            for items in validation_batches(examples, batch_size):
                batch = RepairCollator()(items).to(device)
                # Save ID/length before execution so a failed validation batch is reproducible.
                if event_callback:
                    event_callback(
                        {
                            "stage": "validation_batch",
                            "step": step,
                            "instance_ids": [x.instance_id for x in items],
                            "encoder_tokens": batch.encoder_input_ids.shape[1],
                            "decoder_tokens": batch.decoder_input_ids.shape[1],
                            "batch_target_tokens": int((batch.labels != -100).sum()),
                        }
                    )
                output = (
                    model.forward_loss(batch) if hasattr(model, "forward_loss") else model(batch)
                )
                assert output.loss is not None
                if not bool(torch.isfinite(output.loss)):
                    raise RuntimeError("Nonfinite held-out loss.")
                count = int((batch.labels != -100).sum())
                loss_sum = output.loss.detach().item() * count
                total += loss_sum
                tokens += count
                sources = {x.metadata.get("source", "unknown") for x in items}
                source = next(iter(sources)) if len(sources) == 1 else "mixed_batch"
                entry = by_source.setdefault(
                    source, {"loss_sum": 0.0, "supervised_tokens": 0, "examples": 0}
                )
                entry["loss_sum"] += loss_sum
                entry["supervised_tokens"] += count
                entry["examples"] += len(items)
                tasks = {x.metadata.get("task_type", "repair") for x in items}
                task = next(iter(tasks)) if len(tasks) == 1 else "mixed_batch"
                task_entry = by_task.setdefault(
                    task, {"loss_sum": 0.0, "supervised_tokens": 0, "examples": 0}
                )
                task_entry["loss_sum"] += loss_sum
                task_entry["supervised_tokens"] += count
                task_entry["examples"] += len(items)
                del output, batch
                done += len(items)
                if event_callback and (
                    done == len(items) or done == len(examples) or done % (25 * batch_size) == 0
                ):
                    event_callback(
                        {
                            "stage": "validation_progress",
                            "step": step,
                            "eval_examples_done": done,
                            "eval_examples_total": len(examples),
                            "eval_running_loss": total / tokens if tokens else None,
                            "eval_target_tokens": tokens,
                        }
                    )
    finally:
        model.train(previous)
    if not tokens:
        raise ValueError("Held-out split has no supervised tokens.")
    result = {
        "loss": total / tokens,
        "supervised_tokens": tokens,
        "examples": len(examples),
        "by_source": {
            name: {
                "loss": v["loss_sum"] / v["supervised_tokens"],
                "supervised_tokens": v["supervised_tokens"],
                "examples": v["examples"],
            }
            for name, v in by_source.items()
            if v["supervised_tokens"]
        },
        "by_task_type": {
            name: {
                "loss": v["loss_sum"] / v["supervised_tokens"],
                "supervised_tokens": v["supervised_tokens"],
                "examples": v["examples"],
            }
            for name, v in by_task.items()
            if v["supervised_tokens"]
        },
    }
    if event_callback:
        event_callback(
            {
                "stage": "validation_complete",
                "step": step,
                "dev_loss": result["loss"],
                "eval_examples_done": len(examples),
                "eval_examples_total": len(examples),
                "eval": result,
            }
        )
    return result


def fit_offline_sft(
    model,
    train,
    dev,
    run_dir,
    compatibility,
    *,
    max_steps=100,
    epochs=1,
    batch_size=1,
    eval_batch_size=None,
    accumulation=4,
    learning_rate=2e-5,
    warmup_steps=5,
    clip_norm=1.0,
    eval_every=25,
    save_every=25,
    seed=17,
    resume_from=None,
    progress=None,
    data_manifest=None,
    training_policy=None,
    event_callback=None,
    save_total_limit=2,
    checkpoint_callback=None,
    save_interval_seconds=None,
) -> "dict[str, Any]":
    import time
    import traceback

    from bifocalgemma.training.stages import retained_checkpoints
    from bifocalgemma.training.telemetry import TrainingDashboard, append_event, resource_snapshot
    from bifocalgemma.training.timing import DurableClock

    segment_started = time.perf_counter()
    last_saved_at = segment_started
    if save_interval_seconds is not None and save_interval_seconds <= 0:
        raise ValueError("Checkpoint interval must be positive.")
    evaluation_seconds = checkpoint_seconds = 0.0
    if (
        min(max_steps, epochs, batch_size, accumulation, eval_every, save_every) < 1
        or learning_rate <= 0
    ):
        raise ValueError("Invalid training settings.")
    if not train or not dev or clip_norm <= 0 or warmup_steps < 0 or save_total_limit < 1:
        raise ValueError("Need nonempty train/dev and valid clipping/warmup.")
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    if type(eval_batch_size) is not int or eval_batch_size < 1:
        raise ValueError("Validation batch size must be a positive integer.")
    root = Path(run_dir)
    if not resume_from and root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Run directory already contains artifacts; choose new RUN_NAME or RESUME_FROM."
        )
    root.mkdir(parents=True, exist_ok=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters.")
    if getattr(model, "memory_policy", None):
        compatibility = {**compatibility, "memory_policy": copy.deepcopy(model.memory_policy)}
    checkpoint_format = "bifocalgemma_notebook_sft_v1"
    horizon = min(
        max_steps,
        epochs * ((len(train) + batch_size * accumulation - 1) // (batch_size * accumulation)),
    )
    if training_policy is None:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0, foreach=False)
    else:
        from bifocalgemma.training.optimizers import build_policy_optimizer

        groups = build_training_groups(model, training_policy, total_steps=horizon)
        optimizer = build_policy_optimizer(groups, training_policy)
        compatibility = {
            **compatibility,
            "training_policy": copy.deepcopy(training_policy),
            "optimizer_groups": optimizer_group_manifest(groups),
        }
        checkpoint_format = "bifocalgemma_notebook_sft_v2"
        if training_policy.get("version") == "unified_v2":
            checkpoint_format = "bifocalgemma_normalized_sft_v3"
    device = next(model.parameters()).device
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    state: dict = {"global_step": 0, "epoch": 0, "order": order, "cursor": 0, "history": []}
    # All resume validation precedes changes to run metadata or diagnostics.
    if resume_from:
        state = restore_sft_checkpoint(
            model,
            optimizer,
            resume_from,
            compatibility,
            len(train),
            checkpoint_format=checkpoint_format,
        )
    clock = DurableClock(
        root / "timing.json",
        checkpoint_seconds=state.get("accumulated_seconds", 0),
        started=segment_started,
    )
    if data_manifest is not None:
        (root / "data_manifest.json").write_text(
            json.dumps(data_manifest, indent=2), encoding="utf-8"
        )
    (root / "run_config.json").write_text(json.dumps(compatibility, indent=2), encoding="utf-8")
    log_file = root / "history.jsonl"
    if resume_from:
        log_file.write_text(
            "".join(json.dumps(r) + "\n" for r in state["history"]), encoding="utf-8"
        )
    model.train()
    checkpoints = []
    last_checkpoint = str(resume_from) if resume_from else None
    last_checkpoint_step = state["global_step"] if resume_from else -1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latest_batch = {}
    phase = "run_started"
    if event_callback is None:
        event_callback = TrainingDashboard(None, device=device)

    def emit(event):
        nonlocal phase, latest_batch
        phase = event["stage"]
        if "instance_ids" in event:
            latest_batch = {
                k: event[k]
                for k in ("instance_ids", "encoder_tokens", "decoder_tokens", "batch_target_tokens")
                if k in event
            }
        row = {
            "step": state["global_step"],
            "epoch_fraction": state["epoch"] + state["cursor"] / len(train),
            "max_steps": max_steps,
            "accumulation": accumulation,
            "elapsed_seconds": clock.elapsed(),
            "segment_seconds": time.perf_counter() - segment_started,
            "unobserved_gap_seconds": clock.gap,
            **event,
            **resource_snapshot(device),
        }
        payload = append_event(root / "events.jsonl", row)
        clock.persist()
        if event_callback:
            event_callback(payload)
        return payload

    old_callback = getattr(model, "_stage_callback", None)

    def model_stage(event):
        # Suppress per-stack validation events; validation counters remain visible.
        if model.training:
            emit(event)

    model._stage_callback = model_stage
    try:
        emit(
            {
                "stage": "run_started",
                "run_dir": str(root.resolve()),
                "train_examples": len(train),
                "dev_examples": len(dev),
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "eval_every": eval_every,
                "save_every": save_every,
                "optimizer_name": (training_policy or {}).get("optimizer", {}).get("name", "adamw"),
                "optimizer_groups": optimizer_group_manifest(optimizer.param_groups)
                if training_policy is not None
                else None,
                "last_checkpoint": last_checkpoint,
                "memory_policy": getattr(model, "memory_policy", None),
                **getattr(model, "memory_policy", {}),
                "history": state["history"],
            }
        )
        while state["global_step"] < max_steps and state["epoch"] < epochs:
            if state["cursor"] >= len(state["order"]):
                state["epoch"] += 1
                if state["epoch"] >= epochs:
                    break
                state["order"] = list(range(len(train)))
                random.Random(seed + state["epoch"]).shuffle(state["order"])
                state["cursor"] = 0
            step_started = time.perf_counter()
            micro_batches, item_batches = [], []
            for micro in range(accumulation):
                if state["cursor"] >= len(state["order"]):
                    break
                indices = state["order"][state["cursor"] : state["cursor"] + batch_size]
                state["cursor"] += len(indices)
                items = [train[index] for index in indices]
                item_batches.append(items)
                micro_batches.append(RepairCollator()(items))
            token_count = sum(int((batch.labels != -100).sum()) for batch in micro_batches)
            if not token_count:
                raise ValueError("No supervised tokens in optimizer window.")
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            window_sources: dict[str, dict[str, float]] = {}
            phase_times: dict[str, float] = {"forward_seconds": 0.0, "backward_seconds": 0.0}
            window_tasks: dict[str, dict[str, float]] = {}
            for micro, (batch_cpu, items) in enumerate(
                zip(micro_batches, item_batches, strict=True), 1
            ):
                batch = batch_cpu.to(device)
                count = int((batch_cpu.labels != -100).sum())
                emit(
                    {
                        "stage": "microbatch_started",
                        "microbatch": micro,
                        "instance_ids": [x.instance_id for x in items],
                        "encoder_tokens": batch.encoder_input_ids.shape[1],
                        "decoder_tokens": batch.decoder_input_ids.shape[1],
                        "batch_target_tokens": count,
                        "optimizer_window_target_tokens": token_count,
                    }
                )
                phase_started = time.perf_counter()
                emit({"stage": "forward_started"})
                output = (
                    model.forward_loss(batch) if hasattr(model, "forward_loss") else model(batch)
                )
                assert output.loss is not None
                if not bool(torch.isfinite(output.loss)):
                    raise RuntimeError("Nonfinite training loss.")
                phase_times["forward_seconds"] += time.perf_counter() - phase_started
                emit({"stage": "backward_started"})
                phase_started = time.perf_counter()
                (output.loss * (count / token_count)).backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                phase_times["backward_seconds"] += time.perf_counter() - phase_started
                batch_loss = output.loss.detach().item()
                mean_loss += batch_loss * (count / token_count)
                sources = {x.metadata.get("source", "unknown") for x in items}
                source = next(iter(sources)) if len(sources) == 1 else "mixed_batch"
                entry = window_sources.setdefault(source, {"loss_sum": 0.0, "supervised_tokens": 0})
                entry["loss_sum"] += batch_loss * count
                entry["supervised_tokens"] += count
                tasks = {x.metadata.get("task_type", "repair") for x in items}
                task = next(iter(tasks)) if len(tasks) == 1 else "mixed_batch"
                task_entry = window_tasks.setdefault(
                    task, {"loss_sum": 0.0, "supervised_tokens": 0}
                )
                task_entry["loss_sum"] += batch_loss * count
                task_entry["supervised_tokens"] += count
                del output, batch
                emit({"stage": "microbatch_complete", "microbatch": micro})
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError("Missing/nonfinite training gradients.")
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, foreach=False)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite gradient norm.")
            fraction = min(1.0, (state["global_step"] + 1) / warmup_steps) if warmup_steps else 1.0
            if training_policy is None:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * fraction
                rates = {"default": learning_rate * fraction}
            else:
                rates = update_policy_learning_rates(optimizer, state["global_step"])
            emit({"stage": "optimizer_started", "learning_rates": rates})
            diagnostics = None
            before = None
            if (training_policy or {}).get("version") == "unified_v2" and state[
                "global_step"
            ] % 10 == 0:
                diagnostics = {
                    g["name"]: {
                        "gradient_norm_after_clip": sum(
                            float(p.grad.float().square().sum())
                            for p in g["params"]
                            if p.grad is not None
                        )
                        ** 0.5
                    }
                    for g in optimizer.param_groups
                }
                before = {id(p): p.detach().clone() for p in parameters}
            optimizer_started = time.perf_counter()
            optimizer.step()
            if before is not None:
                assert diagnostics is not None
                for g in optimizer.param_groups:
                    update = (
                        sum(float((p.detach() - before[id(p)]).square().sum()) for p in g["params"])
                        ** 0.5
                    )
                    magnitude = sum(float(before[id(p)].square().sum()) for p in g["params"]) ** 0.5
                    diagnostics[g["name"]].update(
                        update_norm=update,
                        parameter_norm_before=magnitude,
                        relative_update=update / max(magnitude, 1e-30),
                    )
                del before
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phase_times["optimizer_seconds"] = time.perf_counter() - optimizer_started
            del gradients, micro_batches
            state["global_step"] += 1
            seconds = time.perf_counter() - step_started
            row = {
                "step": state["global_step"],
                "epoch": state["epoch"],
                "epoch_fraction": state["epoch"] + state["cursor"] / len(train),
                "train_loss": mean_loss,
                "target_tokens": token_count,
                "gradient_norm": norm.detach().item(),
                "step_seconds": seconds,
                **phase_times,
                "target_tokens_per_second": token_count / seconds,
                "by_source": {
                    name: {
                        "loss": v["loss_sum"] / v["supervised_tokens"],
                        "supervised_tokens": v["supervised_tokens"],
                    }
                    for name, v in window_sources.items()
                },
                "last_batch": latest_batch,
                "by_task_type": {
                    name: {
                        "loss": v["loss_sum"] / v["supervised_tokens"],
                        "supervised_tokens": v["supervised_tokens"],
                    }
                    for name, v in window_tasks.items()
                },
                **resource_snapshot(device),
            }
            if training_policy is None:
                row["learning_rate"] = rates["default"]
            else:
                row["learning_rates"] = rates
            if hasattr(optimizer, "last_diagnostics"):
                row["optimizer_diagnostics"] = copy.deepcopy(optimizer.last_diagnostics)
            if diagnostics is not None:
                row["optimizer_diagnostics"] = diagnostics
            if device.type == "cuda":
                row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            recent_times = [r.get("step_seconds", seconds) for r in state["history"][-19:]] + [
                seconds
            ]
            row["eta_train_seconds"] = (
                sum(recent_times) / len(recent_times) * max(0, max_steps - state["global_step"])
            )
            emit({"stage": "optimizer_step_complete", **row, "learning_rates": rates})
            # Final CE is computed once below, including when the last step
            # happens to be a periodic validation boundary.
            if state["global_step"] % eval_every == 0 and state["global_step"] < horizon:
                started = time.perf_counter()
                row["eval"] = evaluate_sft_loss(
                    model,
                    dev,
                    batch_size=eval_batch_size,
                    device=device,
                    event_callback=emit,
                    step=state["global_step"],
                )
                row["eval_seconds"] = time.perf_counter() - started
                evaluation_seconds += row["eval_seconds"]
            state["history"].append(row)
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if progress:
                progress(row)
            if state["global_step"] % save_every == 0 or (
                save_interval_seconds is not None
                and time.perf_counter() - last_saved_at >= save_interval_seconds
            ):
                started = time.perf_counter()
                emit({"stage": "checkpoint_started"})
                state["accumulated_seconds"] = clock.elapsed()
                last_checkpoint = save_sft_checkpoint(
                    model,
                    optimizer,
                    state,
                    root,
                    compatibility,
                    checkpoint_format=checkpoint_format,
                )
                checkpoints = retained_checkpoints(root, limit=save_total_limit)
                if checkpoint_callback:
                    checkpoint_callback(last_checkpoint)
                checkpoint_seconds += time.perf_counter() - started
                last_checkpoint_step = state["global_step"]
                last_saved_at = time.perf_counter()
                clock.persist(force=True)
                emit(
                    {
                        "stage": "checkpoint_complete",
                        "last_checkpoint": last_checkpoint,
                        "checkpoint_seconds": time.perf_counter() - started,
                    }
                )
        started = time.perf_counter()
        evaluation = evaluate_sft_loss(
            model,
            dev,
            batch_size=eval_batch_size,
            device=device,
            event_callback=emit,
            step=state["global_step"],
        )
        evaluation_seconds += time.perf_counter() - started
        if state["global_step"] != last_checkpoint_step:
            started = time.perf_counter()
            emit({"stage": "checkpoint_started"})
            state["accumulated_seconds"] = clock.elapsed()
            last_checkpoint = save_sft_checkpoint(
                model, optimizer, state, root, compatibility, checkpoint_format=checkpoint_format
            )
            checkpoints = retained_checkpoints(root, limit=save_total_limit)
            if checkpoint_callback:
                checkpoint_callback(last_checkpoint)
            checkpoint_seconds += time.perf_counter() - started
            emit(
                {
                    "stage": "checkpoint_complete",
                    "last_checkpoint": last_checkpoint,
                    "checkpoint_seconds": time.perf_counter() - started,
                }
            )
        checkpoints = retained_checkpoints(root, limit=save_total_limit)
        clock.close()
        result = {
            "status": "COMPLETED",
            "global_step": state["global_step"],
            "epoch": state["epoch"],
            "history": state["history"],
            "held_out_eval": evaluation,
            "checkpoints": checkpoints,
            "resume_from": str(resume_from) if resume_from else None,
            "checkpoint_format": checkpoint_format,
            "memory_policy": getattr(model, "memory_policy", None),
            "training_policy": copy.deepcopy(training_policy),
            "cadence": {"eval_every": eval_every, "save_every": save_every, "total_steps": horizon},
            "last_checkpoint": last_checkpoint,
            "segment_seconds": time.perf_counter() - segment_started,
            "segment_evaluation_seconds": evaluation_seconds,
            "segment_checkpoint_seconds": checkpoint_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
            "resources": resource_snapshot(device),
            "events_file": str(root / "events.jsonl"),
            "monitor_file": str(root / "monitor.html"),
        }
        result["timing"] = {
            **clock.summary(),
            "segment_wall_seconds": result["segment_seconds"],
            "segment_evaluation_seconds": evaluation_seconds,
            "segment_checkpoint_seconds": checkpoint_seconds,
        }
        result["functional_benchmark_run"] = False
        result["optimizer_groups"] = (
            optimizer_group_manifest(optimizer.param_groups)
            if training_policy is not None
            else None
        )
        (root / "training_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(
            {
                "stage": "completed",
                "dev_loss": evaluation["loss"],
                "last_checkpoint": last_checkpoint,
            }
        )
        return result
    except BaseException as exc:
        clock.close("INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
        failed_stage = phase
        failure = {
            "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
            "stage": failed_stage,
            "global_step": state["global_step"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "last_checkpoint": last_checkpoint,
            "last_batch": latest_batch,
            "traceback": traceback.format_exc(),
            "resources": resource_snapshot(device),
        }
        (root / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        emit(
            {
                "stage": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "failed_stage": failed_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "last_checkpoint": last_checkpoint,
                **latest_batch,
            }
        )
        model.zero_grad(set_to_none=True)
        optimizer.state.clear()
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        model._stage_callback = old_callback
