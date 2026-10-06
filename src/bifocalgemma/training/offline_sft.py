"""Bounded LoRA SFT/checkpoint helpers for the matched Molab notebooks.

Adapter-only checkpoints assume the pinned pretrained sources are available.
No RL, model publication, code execution or temporal attention cache.
"""

import copy
import hashlib
import json
import random
from pathlib import Path
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
    if checkpoint_format == "bifocalgemma_notebook_sft_v2":
        metadata["optimizer_groups"] = optimizer_group_manifest(optimizer.param_groups)
    (stage / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    marker = {
        name: file_digest(stage / name)
        for name in ("adapters.safetensors", "trainer_state.pt", "manifest.json")
    }
    (stage / "READY.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if stage.resolve().parent != root or final.resolve().parent != root:
        raise ValueError("Checkpoint targets escaped run directory.")
    stage.rename(final)
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
            "Checkpoint format mismatch: grouped policy requires v2; resume v1 with the previous notebook."
        )
    if manifest["compatibility"] != compatibility:
        raise ValueError("Checkpoint experiment/data/tokenizer/optimizer config mismatch.")
    if checkpoint_format == "bifocalgemma_notebook_sft_v2":
        if manifest.get("optimizer_groups") != optimizer_group_manifest(optimizer.param_groups):
            raise ValueError("Checkpoint optimizer group order/names/policy mismatch.")
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if sorted(parameters) != manifest["trainable_names"]:
        raise ValueError("Checkpoint trainable parameter names differ.")
    payload = torch.load(root / "trainer_state.pt", map_location="cpu", weights_only=True)
    if checkpoint_format == "bifocalgemma_notebook_sft_v2":
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


def evaluate_sft_loss(model, examples, *, batch_size, device):
    previous = model.training
    model.eval()
    total, tokens = 0.0, 0
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = RepairCollator()(examples[start : start + batch_size]).to(device)
                output = model(batch)
                assert output.loss is not None
                if not bool(torch.isfinite(output.loss)):
                    raise RuntimeError("Nonfinite held-out loss.")
                count = int((batch.labels != -100).sum())
                total += output.loss.detach().item() * count
                tokens += count
    finally:
        model.train(previous)
    if not tokens:
        raise ValueError("Held-out split has no supervised tokens.")
    return {"loss": total / tokens, "supervised_tokens": tokens, "examples": len(examples)}


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
):
    if (
        min(max_steps, epochs, batch_size, accumulation, eval_every, save_every) < 1
        or learning_rate <= 0
    ):
        raise ValueError("Invalid training settings.")
    if not train or not dev or clip_norm <= 0 or warmup_steps < 0:
        raise ValueError("Need nonempty train/dev and valid clipping/warmup.")
    root = Path(run_dir)
    if not resume_from and root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Run directory already contains artifacts; choose new RUN_NAME or RESUME_FROM."
        )
    root.mkdir(parents=True, exist_ok=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters.")
    checkpoint_format = "bifocalgemma_notebook_sft_v1"
    if training_policy is None:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0, foreach=False)
    else:
        groups = build_training_groups(model, training_policy)
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0, foreach=False)
        compatibility = {
            **compatibility,
            "training_policy": copy.deepcopy(training_policy),
            "optimizer_groups": optimizer_group_manifest(groups),
        }
        checkpoint_format = "bifocalgemma_notebook_sft_v2"
    device = next(model.parameters()).device
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    state: dict = {"global_step": 0, "epoch": 0, "order": order, "cursor": 0, "history": []}
    if resume_from:
        state = restore_sft_checkpoint(
            model,
            optimizer,
            resume_from,
            compatibility,
            len(train),
            checkpoint_format=checkpoint_format,
        )
    if data_manifest is not None:
        (root / "data_manifest.json").write_text(
            json.dumps(data_manifest, indent=2), encoding="utf-8"
        )
    (root / "run_config.json").write_text(json.dumps(compatibility, indent=2), encoding="utf-8")
    model.train()
    checkpoints = []
    last_checkpoint_step = state["global_step"] if resume_from else -1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_file = root / "history.jsonl"
    while state["global_step"] < max_steps and state["epoch"] < epochs:
        if state["cursor"] >= len(state["order"]):
            state["epoch"] += 1
            if state["epoch"] >= epochs:
                break
            state["order"] = list(range(len(train)))
            random.Random(seed + state["epoch"]).shuffle(state["order"])
            state["cursor"] = 0
        micro_batches = []
        for micro in range(accumulation):
            if state["cursor"] >= len(state["order"]):
                break
            indices = state["order"][state["cursor"] : state["cursor"] + batch_size]
            state["cursor"] += len(indices)
            micro_batches.append(RepairCollator()([train[index] for index in indices]))
        token_count = sum(int((batch.labels != -100).sum()) for batch in micro_batches)
        if not token_count:
            raise ValueError("No supervised tokens in optimizer window.")
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for batch_cpu in micro_batches:
            batch = batch_cpu.to(device)
            output = model(batch)
            assert output.loss is not None
            if not bool(torch.isfinite(output.loss)):
                raise RuntimeError("Nonfinite training loss.")
            count = int((batch_cpu.labels != -100).sum())
            (output.loss * (count / token_count)).backward()
            mean_loss += output.loss.detach().item() * (count / token_count)
            del output, batch
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError("Missing/nonfinite training gradients.")
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, foreach=False)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Nonfinite gradient norm.")
        rates = None
        fraction = min(1.0, (state["global_step"] + 1) / warmup_steps) if warmup_steps else 1.0
        if training_policy is None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * fraction
        else:
            rates = update_policy_learning_rates(optimizer, state["global_step"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del gradients, micro_batches
        state["global_step"] += 1
        row = {
            "step": state["global_step"],
            "epoch": state["epoch"],
            "train_loss": mean_loss,
            "target_tokens": token_count,
            "learning_rate": learning_rate * fraction,
            "gradient_norm": norm.detach().item(),
        }
        if rates is not None:
            row.pop("learning_rate")
            row["learning_rates"] = rates
        if state["global_step"] % eval_every == 0:
            row["eval"] = evaluate_sft_loss(model, dev, batch_size=batch_size, device=device)
        if device.type == "cuda":
            row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        state["history"].append(row)
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if progress:
            progress(row)
        if state["global_step"] % save_every == 0:
            checkpoints.append(
                save_sft_checkpoint(
                    model,
                    optimizer,
                    state,
                    root,
                    compatibility,
                    checkpoint_format=checkpoint_format,
                )
            )
            last_checkpoint_step = state["global_step"]
    evaluation = evaluate_sft_loss(model, dev, batch_size=batch_size, device=device)
    if state["global_step"] != last_checkpoint_step:
        checkpoints.append(
            save_sft_checkpoint(
                model, optimizer, state, root, compatibility, checkpoint_format=checkpoint_format
            )
        )
    return {
        "status": "COMPLETED",
        "global_step": state["global_step"],
        "epoch": state["epoch"],
        "history": state["history"],
        "held_out_eval": evaluation,
        "checkpoints": checkpoints,
        "resume_from": str(resume_from) if resume_from else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "functional_benchmark_run": False,
        "checkpoint_format": checkpoint_format,
        "training_policy": copy.deepcopy(training_policy),
        "optimizer_groups": optimizer_group_manifest(optimizer.param_groups)
        if training_policy is not None
        else None,
    }
