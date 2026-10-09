"""Small optimization/VRAM probe, not the production SFT trainer.

Modes: round_robin or overlap_graphs. The latter retains all models' activation
 graphs before backward, but does not claim concurrent CUDA kernel execution.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbeLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = 2.0  # alpha = 2 * rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.normal_(self.lora_A, std=0.02)

    def forward(self, values):
        delta = F.linear(F.linear(values.float(), self.lora_A), self.lora_B)
        return self.base(values) + delta.to(values.dtype) * self.scaling


def configure_probe_training(model, *, policy: str = "lora", rank: int = 16) -> dict:
    if policy not in {"lora", "full"}:
        raise ValueError("Training policy must be lora or full.")
    model.requires_grad_(policy == "full")
    adapted = []
    if policy == "lora":
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            }:
                parent_name, attribute = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_name), attribute, ProbeLoRALinear(module, rank))
                adapted.append(name)
        # Asymmetric alignment is new, so it must remain trainable too.
        projection = getattr(model, "encoder_projection", None)
        if isinstance(projection, nn.Linear):
            projection.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters were selected.")
    return {
        "policy": policy,
        "rank": rank if policy == "lora" else None,
        "adapter_modules": adapted,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_bytes": tensor_bytes(trainable),
    }


def tensor_bytes(tensors) -> int:
    seen = set()
    total = 0
    for tensor in tensors:
        # Parameters are unique across the iterator; object identity also dedups optimizer aliases.
        if id(tensor) not in seen:
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def optimizer_bytes(optimizers) -> int:
    return tensor_bytes(
        value
        for optimizer in optimizers
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def gpu_memory(device) -> dict:
    if torch.device(device).type != "cuda":
        return dict(
            allocated_bytes=None,
            reserved_bytes=None,
            free_bytes=None,
            total_bytes=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
        )
    free, total = torch.cuda.mem_get_info(device)
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        free_bytes=free,
        total_bytes=total,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def profile_training_group(models, batches, *, steps=2, mode="overlap_graphs", learning_rate=1e-4):
    if not models or len(models) != len(batches) or steps < 1:
        raise ValueError("Supply one batch per model and at least one optimization step.")
    if mode not in {"round_robin", "overlap_graphs"}:
        raise ValueError("Mode must be round_robin or overlap_graphs.")
    parameter_groups = [[p for p in model.parameters() if p.requires_grad] for model in models]
    parameter_ids = [id(p) for group in parameter_groups for p in group]
    if not all(parameter_groups) or len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("Models need independent nonempty trainable parameter sets.")
    device = next(models[0].parameters()).device
    if any(p.device != device for model in models for p in model.parameters()):
        raise ValueError("All models must be resident on the same device.")
    optimizers = [
        torch.optim.AdamW(group, lr=learning_rate, foreach=False) for group in parameter_groups
    ]
    for model in models:
        model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    before = gpu_memory(device)
    started = time.perf_counter()
    losses = []
    snapshots = []
    success = False
    try:
        for step in range(steps):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            if mode == "overlap_graphs":
                outputs = [model(batch) for model, batch in zip(models, batches, strict=True)]
                graphs = [output.loss for output in outputs]
                if any(loss is None for loss in graphs):
                    raise ValueError("Model did not supply a supervised loss.")
                snapshots.append(
                    {"step": step, "stage": "all_forward_graphs", **gpu_memory(device)}
                )
                for loss in graphs:
                    assert loss is not None
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite loss.")
                    loss.backward()
                    losses.append(loss.detach().item())
                del loss, graphs, outputs
            else:
                for model, batch in zip(models, batches, strict=True):
                    output = model(batch)
                    if output.loss is None or not bool(torch.isfinite(output.loss)):
                        raise RuntimeError("Missing/nonfinite loss.")
                    output.loss.backward()
                    losses.append(output.loss.detach().item())
                    del output
            for group, optimizer in zip(parameter_groups, optimizers, strict=True):
                gradients = [p.grad for p in group if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise RuntimeError("Missing/nonfinite gradients.")
                if not any(bool(g.abs().sum() > 0) for g in gradients):
                    raise RuntimeError("All trainable gradients are zero.")
                optimizer.step()
                del gradients
            snapshots.append({"step": step, "stage": "optimizer_step", **gpu_memory(device)})
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        states = [
            value
            for optimizer in optimizers
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ]
        result = {
            "status": "PASS",
            "model_count": len(models),
            "mode": mode,
            "steps_per_model": steps,
            "optimizer": "torch.optim.AdamW(foreach=False)",
            "optimizer_tensor_dtypes": sorted({str(t.dtype) for t in states}),
            "optimizer_state_bytes": optimizer_bytes(optimizers),
            "optimizer_gpu_state_bytes": tensor_bytes(t for t in states if t.device.type == "cuda"),
            "parameter_dtypes": sorted(
                {str(p.dtype) for model in models for p in model.parameters()}
            ),
            "parameter_bytes": tensor_bytes(p for model in models for p in model.parameters()),
            "trainable_bytes": tensor_bytes(p for group in parameter_groups for p in group),
            "seconds": time.perf_counter() - started,
            "losses": losses,
            "before": before,
            "after": gpu_memory(device),
            "snapshots": snapshots,
            "independent_weights": True,
            "concurrent_cuda_kernels_claimed": False,
        }
        success = True
        return result
    finally:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
        if not success:
            # Caller owns models and may release them after OOM; no success is fabricated.
            for model in models:
                model.zero_grad(set_to_none=True)


def synthetic_stress_target(prefix_length, target_ids, decoder_tokens, padding_ids):
    """Fill a synthetic tensor-stress target exactly; never truncate real examples."""
    if (
        type(prefix_length) is not int
        or prefix_length < 1
        or type(decoder_tokens) is not int
        or not 1 <= decoder_tokens <= 131072
        or not target_ids
        or target_ids[-1] != 106
        or not padding_ids
    ):
        raise ValueError("invalid_synthetic_stress_boundary")
    extra = decoder_tokens + 1 - prefix_length - len(target_ids)
    if extra < 0:
        raise ValueError("synthetic_stress_budget_too_small_no_truncation")
    padding = (tuple(padding_ids) * ((extra + len(padding_ids) - 1) // len(padding_ids)))[:extra]
    return (*target_ids[:-1], *padding, 106)


def run_length_grid(names, length_grid, probe, *, progress=None) -> dict:
    """One model at a time; stop larger boundaries for that model after a failure."""
    allowed = {
        "BifocalGemma-E2B-E2B",
        "BifocalGemma-E4B-E4B",
        "BifocalGemma-E2B-E4B",
        "BifocalGemma-E4B-E2B",
        "Gemma-4-E2B-IT",
        "Gemma-4-E4B-IT",
    }
    if not names or len(set(names)) != len(names) or not set(names) <= allowed:
        raise ValueError("invalid_length_grid_experiments")
    grid = list(length_grid)
    if not grid or any(
        len(pair) != 2 or any(type(v) is not int or not 64 <= v <= 32768 for v in pair)
        for pair in grid
    ):
        raise ValueError("invalid_length_grid_boundaries")
    if any(a[0] > b[0] or a[1] > b[1] or a == b for a, b in zip(grid, grid[1:])):
        raise ValueError("length_grid_must_increase")
    rows: list[dict] = []
    for name in names:
        for index, (encoder, paired_decoder) in enumerate(grid):
            decoder = (
                encoder + paired_decoder + 512 if name.startswith("Gemma-4-") else paired_decoder
            )
            result = probe(name, encoder, decoder)
            rows.append(
                {
                    "experiment": name,
                    "encoder_context_budget": encoder,
                    "paired_decoder_budget": paired_decoder,
                    "actual_decoder_budget": decoder,
                    "result": result,
                    "larger_boundaries_not_attempted": grid[index + 1 :]
                    if result["status"] != "PASS"
                    else [],
                }
            )
            if progress is not None:
                progress(rows[-1])
            if result["status"] != "PASS":
                break
    return {
        "scope": "synthetic_exact_length_training_memory_not_model_quality",
        "rows": rows,
        "one_resident_model_at_a_time": True,
        "all_configurations_passed_maximum": len(rows) == len(names) * len(grid)
        and all(row["result"]["status"] == "PASS" for row in rows),
    }
