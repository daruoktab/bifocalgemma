"""Measure candidates on actual shapes; never choose fused attention by assumption."""

import gc
import time
from typing import Any

import torch

from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.memory_efficient import (
    configure_memory_efficient_training,
    run_training_memory_preflight,
)


def representative_examples(examples):
    if hasattr(examples, "rows"):
        order = sorted(
            range(len(examples)), key=lambda i: sum(x[1] for x in examples.rows[i]["offsets"])
        )
    else:
        order = sorted(
            range(len(examples)),
            key=lambda i: (
                len(examples[i].encoder_ids)
                + len(examples[i].decoder_prefix_ids)
                + len(examples[i].target_ids)
            ),
        )
    return [examples[order[i]] for i in (0, len(order) // 2, len(order) - 1)]


def select_execution(model, train, dev, *, progress=None, min_headroom_gib=8):
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("Production preflight requires the target CUDA GPU.")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    old_mode = model.training
    candidates = []
    probes = representative_examples(train)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def step(item, batch_size):
        model.train()
        model.zero_grad(set_to_none=True)
        batch = RepairCollator()([item] * batch_size).to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.forward_loss(batch)
        out.loss.backward()
        torch.cuda.synchronize()
        return float(out.loss.detach()), time.perf_counter() - start

    try:
        # Numerical comparison on an actual short example, including all trainable gradients.
        configure_memory_efficient_training(model, {"attention_backend": "tiled_eager_fp32"})
        ref_loss, _ = step(probes[0], 1)
        reference = {n: p.grad.detach().cpu().clone() for n, p in trainable if p.grad is not None}
        for backend, chunk in [
            ("tiled_eager_fp32", 128),
            ("tiled_eager_fp32", 256),
            ("tiled_eager_fp32", 512),
            ("flex_attention", 128),
            ("hybrid_flex", 128),
        ]:
            for batch_size in (1, 2):
                policy = {"attention_backend": backend, "query_chunk_size": chunk}
                row: dict[str, Any] = {
                    "backend": backend,
                    "query_chunk_size": chunk,
                    "batch_size": batch_size,
                    "eligible": False,
                }
                try:
                    configure_memory_efficient_training(model, policy)
                    loss, compile_time = step(probes[0], 1)
                    if abs(loss - ref_loss) > max(1e-3, abs(ref_loss) * 0.01):
                        raise ValueError("Candidate loss differs from tiled reference.")
                    gradient_errors = {}
                    for n, p in trainable:
                        if n in reference:
                            if p.grad is None:
                                raise ValueError("Candidate lost a trainable gradient: " + n)
                            grad = p.grad.detach().float().cpu()
                            truth = reference[n].float()
                            error = float((grad - truth).norm() / truth.norm().clamp_min(1e-8))
                            gradient_errors[n] = error
                            if not torch.isfinite(grad).all() or error > 0.05:
                                raise ValueError(f"Candidate gradient differs for {n}: {error:.6g}")
                    row.update(
                        loss_difference=loss - ref_loss,
                        max_gradient_relative_l2_error=max(gradient_errors.values(), default=0.0),
                    )
                    model.zero_grad(set_to_none=True)
                    stress = run_training_memory_preflight(
                        model,
                        train,
                        batch_size=batch_size,
                        min_headroom_gib=min_headroom_gib,
                        progress=progress,
                    )
                    validation = run_training_memory_preflight(
                        model,
                        dev,
                        batch_size=batch_size,
                        min_headroom_gib=min_headroom_gib,
                        progress=progress,
                    )
                    times = []
                    warmup_times = []
                    for item in probes:
                        warmup_times.append(step(item, batch_size)[1])
                        times.append(sum(step(item, batch_size)[1] for _ in range(2)) / 2)
                    row.update(
                        eligible=True,
                        first_call_seconds=compile_time,
                        compile_warmup_seconds_by_shape=warmup_times,
                        seconds_by_shape=times,
                        effective_update_seconds=sum(times) * 4 / batch_size,
                        train_memory=stress,
                        validation_memory=validation,
                    )
                except (
                    RuntimeError,
                    ValueError,
                    AssertionError,
                    ImportError,
                ) as error:
                    row.update(reason=type(error).__name__ + ": " + str(error)[:1000])
                finally:
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                candidates.append(row)
                if progress:
                    progress({"stage": "backend_candidate", **row})
        eligible = [r for r in candidates if r["eligible"]]
        if not eligible:
            raise RuntimeError(
                "No backend/microbatch passed numerical and 8 GiB memory checks. " + str(candidates)
            )
        chosen = min(eligible, key=lambda r: r["effective_update_seconds"])
        policy = configure_memory_efficient_training(
            model,
            {
                "attention_backend": chosen["backend"],
                "query_chunk_size": chosen["query_chunk_size"],
            },
        )
        return {
            "status": "VALIDATED_ON_THIS_GPU",
            "gpu": torch.cuda.get_device_name(device),
            "memory_policy": policy,
            "batch_size": chosen["batch_size"],
            "accumulation": 4 // chosen["batch_size"],
            "candidates": candidates,
            "layer_backend_plan": [
                {
                    "role": role,
                    "layer": i,
                    "head_dim": layer.self_attn.head_dim,
                    "implementation": "flex_attention"
                    if chosen["backend"] == "flex_attention"
                    or (chosen["backend"] == "hybrid_flex" and layer.self_attn.head_dim <= 256)
                    else "tiled_eager_fp32",
                }
                for role in ("encoder", "decoder")
                if getattr(model, role, None) is not None
                for i, layer in enumerate(getattr(model, role).layers)
            ],
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(old_mode)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
