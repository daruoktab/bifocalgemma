"""Deterministic pilot cohorts and encoder-use tests, without generation."""

import hashlib
from collections import defaultdict
from dataclasses import replace
from typing import Any

import torch

from bifocalgemma.training.offline_sft import evaluate_sft_loss


def stratified_subset(examples, count, seed=17):
    strata = defaultdict(list)
    if hasattr(examples, "rows"):
        metadata = [(r["instance_id"], r.get("metadata", {})) for r in examples.rows]
    else:
        metadata = [(x.instance_id, x.metadata) for x in examples]
    for i, (identifier, meta) in enumerate(metadata):
        key = (meta.get("source", "unknown"), meta.get("task_type", "repair"))
        rank = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
        strata[key].append((rank, i))
    buckets = [sorted(v) for _, v in sorted(strata.items())]
    chosen = []
    for level in range(max(map(len, buckets))):
        for bucket in buckets:
            if level < len(bucket):
                chosen.append(bucket[level][1])
            if len(chosen) >= min(count, len(examples)):
                return [examples[i] for i in chosen]
    return [examples[i] for i in chosen]


def encoder_diagnostics(model, examples, *, progress=None) -> dict[str, Any]:
    device = next(model.parameters()).device
    cohort = stratified_subset(examples, 256)
    if model.encoder is None:
        return {
            "status": "NOT_APPLICABLE_DECODER_ONLY",
            "instances": [x.instance_id for x in cohort],
        }
    if len(cohort) < 2:
        raise ValueError("Encoder permutation requires at least two held-out examples.")
    # A fixed derangement leaves decoder tokens, targets and masks exactly unchanged.
    swapped = [
        replace(item, encoder_ids=cohort[(i + 1) % len(cohort)].encoder_ids)
        for i, item in enumerate(cohort)
    ]
    original = evaluate_sft_loss(
        model, cohort, batch_size=1, device=device, event_callback=progress
    )
    perturbed = evaluate_sft_loss(
        model, swapped, batch_size=1, device=device, event_callback=progress
    )
    masses = {}
    attentions = [layer.self_attn for layer in model.decoder.layers]
    try:
        for attention in attentions:
            attention.collect_encoder_mass = True
            attention.encoder_mass_samples = []
        evaluate_sft_loss(model, cohort[:8], batch_size=1, device=device)
        for i, attention in enumerate(attentions):
            values = attention.encoder_mass_samples
            masses[str(i)] = sum(values) / len(values) if values else None
    finally:
        for attention in attentions:
            attention.collect_encoder_mass = False
            attention.encoder_mass_samples = []
    return {
        "instances": [x.instance_id for x in cohort],
        "permuted_context_from": [
            cohort[(i + 1) % len(cohort)].instance_id for i in range(len(cohort))
        ],
        "original": original,
        "permuted": perturbed,
        "loss_increase": perturbed["loss"] - original["loss"],
        "encoder_attention_mass_last_valid_queries": masses,
        "interpretation": "Positive delta supports useful encoder information on this cohort; not proof of benchmark superiority.",
    }


def sample_encoder_mass(attention, query, key, encoder_valid, decoder_valid):
    from bifocalgemma.models.memory_efficient import joint_visibility_tile

    with torch.no_grad():
        indices = torch.where(decoder_valid[0].bool())[0][-4:]
        if indices.numel() == 0:
            return
        # Called only for the small, single-example diagnostic cohort.
        q = (
            query[:, :, indices]
            .float()
            .reshape(
                query.shape[0], key.shape[1], attention.num_key_value_groups, -1, query.shape[-1]
            )
        )
        k = key.float()
        allowed = joint_visibility_tile(
            encoder_valid,
            decoder_valid,
            int(indices[0]),
            int(indices[-1]) + 1,
            attention.sliding_window,
        )
        scores = (q @ k[:, :, None].transpose(-1, -2)).reshape(
            query.shape[0], query.shape[1], len(indices), key.shape[2]
        ) * attention.scaling
        probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        mass = probabilities[..., decoder_valid.shape[1] :].sum(-1).mean()
        attention.encoder_mass_samples.append(float(mass))
