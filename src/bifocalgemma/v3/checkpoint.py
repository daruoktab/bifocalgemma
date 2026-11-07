"""V3 checkpointing, resume, pruning, and Hugging Face synchronization."""

from __future__ import annotations

import copy
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint
from safetensors.torch import load_file, save_file

from bifocalgemma.data.curation import file_digest
from bifocalgemma.training.memory_probe import ProbeLoRALinear
from bifocalgemma.training.stages import (
    atomic_json,
    validate_checkpoint_files,
)
from bifocalgemma.v3.constants import (
    CLEAN_RECIPE_VERSION,
    STAGE1_STEPS,
    TOTAL_STEPS,
)
from bifocalgemma.v3.objectives import (
    is_cross_bridge_parameter,
)


def save_matrix_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_dir: Path,
    global_step: int,
    stage_name: str,
    compatibility: dict[str, Any],
    training_state: dict[str, Any],
) -> Path:
    """Save checkpoint with READY.json, manifest.json, adapters.safetensors, and trainer_state.pt."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    stage_dir = checkpoint_dir.with_name("." + checkpoint_dir.name + ".tmp")
    stage_dir.mkdir(parents=True, exist_ok=True)

    parameters = {
        name: param.detach().cpu().contiguous()
        for name, param in model.named_parameters()
        if is_cross_bridge_parameter(name) or name.endswith((".lora_A", ".lora_B"))
    }
    save_file(parameters, str(stage_dir / "adapters.safetensors"))

    trainer_payload = {
        "optimizer": optimizer.state_dict(),
        "training_state": {**training_state, "global_step": global_step, "stage": stage_name},
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(trainer_payload, stage_dir / "trainer_state.pt")

    manifest = {
        "format": "bifocalgemma_full_matrix_v1",
        "recipe_version": compatibility.get("recipe_version", CLEAN_RECIPE_VERSION),
        "global_step": global_step,
        "stage": stage_name,
        "compatibility": compatibility,
        "trainable_names": sorted(parameters),
    }
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ready_marker = {
        name: file_digest(stage_dir / name)
        for name in ("adapters.safetensors", "trainer_state.pt", "manifest.json")
    }
    (stage_dir / "READY.json").write_text(json.dumps(ready_marker, indent=2), encoding="utf-8")

    from bifocalgemma.training.stages import retry_windows_file_operation

    retry_windows_file_operation(lambda: stage_dir.rename(checkpoint_dir))
    return checkpoint_dir


def restore_optimizer_state_by_name(
    model: Any,
    optimizer: Any,
    saved_opt_state: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Map and restore optimizer state tensors (exp_avg, exp_avg_sq, step) by parameter names.

    Iterates through model.named_parameters(), locates parameter state in saved_opt_state["state"]
    using checkpoint param_names mapping, and copies exp_avg, exp_avg_sq, and step directly into
    the active optimizer state tensor for matching parameters.
    """
    saved_groups = saved_opt_state.get("param_groups", [])
    saved_state = saved_opt_state.get("state", {})
    if not saved_state:
        return {"restored": 0, "skipped_shape": 0, "missing": 0, "active_total": 0}

    # 1. Build checkpoint param_id -> param_name mapping
    checkpoint_id_to_name: dict[int, str] = {}
    for g in saved_groups:
        p_names = g.get("param_names")
        p_ids = g.get("params", [])
        if p_names and len(p_names) == len(p_ids):
            for pid, name in zip(p_ids, p_names):
                checkpoint_id_to_name[
                    int(pid) if isinstance(pid, str) and pid.isdigit() else pid
                ] = name

    # Fallback to manifest trainable_names if param_names was not in groups
    if not checkpoint_id_to_name and manifest and "trainable_names" in manifest:
        trainable_names = manifest["trainable_names"]
        all_pids = [pid for g in saved_groups for pid in g.get("params", [])]
        if len(all_pids) == len(trainable_names):
            for pid, name in zip(all_pids, trainable_names):
                checkpoint_id_to_name[
                    int(pid) if isinstance(pid, str) and pid.isdigit() else pid
                ] = name

    checkpoint_name_to_state: dict[str, dict[str, Any]] = {}
    for pid, st in saved_state.items():
        lookup_pid = int(pid) if isinstance(pid, str) and pid.isdigit() else pid
        if lookup_pid in checkpoint_id_to_name:
            checkpoint_name_to_state[checkpoint_id_to_name[lookup_pid]] = st

    # 2. Iterate through model.named_parameters() and map to active optimizer parameters
    active_params = {p for g in optimizer.param_groups for p in g.get("params", [])}

    restored_count = 0
    skipped_shape = 0
    missing_count = 0

    for name, param in model.named_parameters():
        if param not in active_params:
            continue
        if name not in checkpoint_name_to_state:
            missing_count += 1
            continue

        src_state = checkpoint_name_to_state[name]
        # Shape verification guard
        if "exp_avg" in src_state and src_state["exp_avg"].shape != param.shape:
            skipped_shape += 1
            continue
        if "exp_avg_sq" in src_state and src_state["exp_avg_sq"].shape != param.shape:
            skipped_shape += 1
            continue

        # Build restored state dictionary with device and dtype matching
        new_state = {}
        for k, v in src_state.items():
            if isinstance(v, torch.Tensor):
                if k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    new_state[k] = v.to(device=param.device, dtype=param.dtype).clone()
                elif k == "step":
                    new_state[k] = v.to(device=param.device).clone()
                else:
                    new_state[k] = v.to(device=param.device).clone()
            else:
                new_state[k] = copy.deepcopy(v)

        optimizer.state[param] = new_state
        restored_count += 1

    return {
        "restored": restored_count,
        "skipped_shape": skipped_shape,
        "missing": missing_count,
        "active_total": len(active_params),
    }


def restore_matrix_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Restore checkpoint and verified parameters; load trainer state and RNG."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    validate_checkpoint_files(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))

    weights = load_file(str(checkpoint_dir / "adapters.safetensors"), device="cpu")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in weights.items():
            if name in parameters:
                parameters[name].copy_(tensor.to(parameters[name].device))
            else:
                raise ValueError(f"Adapter parameter {name} not found in model.")

    trainer_payload = torch.load(
        checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    optimizer_restored_stats = None
    if "optimizer" in trainer_payload and optimizer is not None:
        try:
            optimizer.load_state_dict(trainer_payload["optimizer"])
        except Exception:
            # Group structure varies between stages (e.g. Stage 1 -> Stage 2 adds decoder).
            # Perform parameter-name-matched optimizer state transfer.
            optimizer_restored_stats = restore_optimizer_state_by_name(
                model=model,
                optimizer=optimizer,
                saved_opt_state=trainer_payload["optimizer"],
                manifest=manifest,
            )
    if "python_rng" in trainer_payload:
        random.setstate(trainer_payload["python_rng"])
    if "torch_rng" in trainer_payload:
        torch.set_rng_state(trainer_payload["torch_rng"])
    if trainer_payload.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(trainer_payload["cuda_rng"])

    return {
        "global_step": manifest["global_step"],
        "stage": manifest.get("stage", "unknown"),
        "training_state": trainer_payload.get("training_state", {}),
        "manifest": manifest,
        "optimizer_restored": optimizer_restored_stats,
    }


def prune_matrix_checkpoints(train_dir: Path, keep_limit: int = 2) -> list[str]:
    """Keep the milestone checkpoint-005551 and the last keep_limit checkpoints."""
    train_dir = Path(train_dir).resolve()
    candidates = sorted(
        [p for p in train_dir.glob("checkpoint-*") if (p / "READY.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if len(candidates) <= keep_limit:
        return [p.name for p in candidates]
    milestone = {
        p
        for p in candidates
        if p.name in {f"checkpoint-{STAGE1_STEPS:06d}", f"checkpoint-{TOTAL_STEPS:06d}"}
    }
    latest = set(candidates[-keep_limit:])
    keep = milestone | latest
    for p in candidates:
        if p not in keep:
            try:
                shutil.rmtree(p)
            except Exception:
                pass
    return sorted(p.name for p in keep)


def synchronize_matrix_v3(
    repo_id: str,
    root: Path | str,
    prefix: str,
    *,
    token: str,
    require_checkpoint: bool = True,
    max_retries: int = 10,
    base_delay: float = 0.05,
) -> dict[str, Any]:
    """Synchronize verified checkpoints and benchmarks to a shared multi-model Hub repo with contention retry."""
    import re

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    from bifocalgemma.training.hub import checked_prefix

    checked_prefix(prefix)
    root = Path(root).resolve()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    checkpoints = sorted(
        p for p in (root / "training").glob("checkpoint-*") if (p / "READY.json").exists()
    )
    for p in checkpoints:
        validate_checkpoint_files(p)

    milestone_names = {f"checkpoint-{STAGE1_STEPS:06d}", f"checkpoint-{TOTAL_STEPS:06d}"}
    retained_set = set(checkpoints[-2:]) | {p for p in checkpoints if p.name in milestone_names}
    retained = sorted(retained_set)
    if not retained and require_checkpoint:
        raise ValueError("No complete checkpoint to snapshot.")
    keep = {p.name for p in retained}

    payloads: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        relative = p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("Snapshot escaped run directory.")
        if not (
            relative == "pipeline_state.json"
            or relative.startswith(("training/", "benchmarks/", "diagnostics/"))
        ):
            continue
        m = re.match(r"training/(checkpoint-\d+)/", relative)
        if m and m[1] not in keep:
            continue
        if p.suffix not in {".json", ".jsonl", ".html", ".pt", ".safetensors"}:
            continue
        payloads[relative] = p

    receipt = {
        "status": "COMPLETED",
        "repo_id": repo_id,
        "prefix": prefix,
        "checkpoint_only": False,
        "retained_checkpoints": sorted(keep),
        "files": {n: file_digest(p) for n, p in payloads.items()},
        "format": "verified_snapshot_v2",
    }
    operations = [
        CommitOperationAdd(path_in_repo=prefix + "/" + n, path_or_fileobj=str(p))
        for n, p in payloads.items()
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=prefix + "/upload_complete.json",
            path_or_fileobj=json.dumps(receipt, indent=2).encode(),
        )
    )

    retryable_codes = {412, 429, 500, 502, 503, 504}
    for attempt in range(max_retries):
        parent = api.model_info(repo_id).sha
        try:
            commit = api.create_commit(
                repo_id,
                operations=operations,
                parent_commit=parent,
                commit_message="Save verified v2 checkpoint and results snapshot",
            )
            break
        except HfHubHTTPError as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code not in retryable_codes or attempt == max_retries - 1:
                raise
            if base_delay > 0:
                delay = base_delay * (1.5 ** min(attempt, 6)) * (0.5 + random.random())
                time.sleep(delay)
    else:
        raise RuntimeError("Snapshot commit retry exhausted.")

    visible = set(api.list_repo_files(repo_id, revision=commit.oid))
    if not all(prefix + "/" + n in visible for n in payloads):
        raise ValueError("Snapshot inventory not visible.")
    for p in retained:
        remote = hf_hub_download(
            repo_id,
            prefix + "/training/" + p.name + "/READY.json",
            revision=commit.oid,
            token=token,
        )
        if Path(remote).read_bytes() != (p / "READY.json").read_bytes():
            raise ValueError("Remote checkpoint marker mismatch.")

    pattern = re.compile(re.escape(prefix) + r"/training/(checkpoint-\d+)/")
    obsolete = [n for n in visible if (m := pattern.match(n)) and m[1] not in keep]
    revision = commit.oid
    if obsolete:
        for attempt in range(max_retries):
            parent = commit.oid if attempt == 0 else api.model_info(repo_id).sha
            try:
                deleted = api.create_commit(
                    repo_id,
                    operations=[CommitOperationDelete(path_in_repo=n) for n in obsolete],
                    parent_commit=parent,
                    commit_message="Prune older checkpoints after verifying replacement",
                )
                revision = deleted.oid
                break
            except HfHubHTTPError as error:
                status_code = getattr(getattr(error, "response", None), "status_code", None)
                if status_code not in retryable_codes or attempt == max_retries - 1:
                    raise
                if base_delay > 0:
                    delay = base_delay * (1.5 ** min(attempt, 6)) * (0.5 + random.random())
                    time.sleep(delay)

    atomic_json(root / "upload_complete.json", receipt)
    return {"revision": revision, "retained_checkpoints": sorted(keep), "verified_at": time.time()}


synchronize_v2 = synchronize_matrix_v3


def load_matrix_adapters_for_inference(model: Any, checkpoint_dir: Path | str) -> dict[str, Any]:
    """Load adapters for post-training benchmark evaluation; unfreeze scaling."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    manifest = validate_checkpoint_files(checkpoint_dir)
    safetensors_path = checkpoint_dir / "adapters.safetensors"
    weights = load_file(str(safetensors_path), device="cpu")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in weights.items():
            if name in parameters:
                parameters[name].copy_(tensor.to(parameters[name].device))
            else:
                raise ValueError(f"Unknown adapter parameter in checkpoint: {name}")

    policy = manifest.get("compatibility", {}).get("training_policy", {})
    alpha = policy.get("alpha", 16)
    rank = policy.get("rank", 16)
    scaling = alpha / rank if rank else 1.0
    for module in model.modules():
        if isinstance(module, ProbeLoRALinear):
            module.scaling = scaling

    if hasattr(model, "_tie_words"):
        model._tie_words()
    model.eval()
    return manifest
