"""Matched full-pool tokenization and complete-epoch SFT for four pairs/two baselines."""

import math
from pathlib import Path
from typing import Any

from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.training.repository_sft import tokenize_repository_record


def prepare_curated_training(
    spec, pool_directory, tokenizer_cache, *, packed_cache=None, progress=None
):
    """Consume all frozen IDs in order; no sampling, re-split or overflow dropping."""
    rows, pool = load_curated_pool(pool_directory)
    limits = pool["report"]["tokens"]["limits"]
    decoder, dec_manifest = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder, enc_manifest = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))
        if spec.encoder
        else (None, None)
    )
    cache_identity = {
        "pool_fingerprint": pool["fingerprint"],
        "limits": limits,
        "encoder_files": enc_manifest["files"] if enc_manifest else None,
        "decoder_files": dec_manifest["files"],
        "prompt_recipe": "context_mixture_v1",
    }

    def tokenize(row):
        context = row.get("context") or (
            f"### Instruction\n{row['instruction']}\n\n### File: {row['path']}\n{row['before']}"
        )
        return tokenize_repository_record(
            {**row, "context": context},
            spec,
            encoder,
            decoder,
            encoder_limit=limits["encoder"],
            decoder_limit=limits["pair_decoder"] if spec.encoder else limits["baseline_decoder"],
        )

    outputs: dict[str, Any]
    if packed_cache is not None:
        from bifocalgemma.data.packed_tokens import build_packed_cache

        outputs, receipt = build_packed_cache(
            Path(packed_cache) / stable_fingerprint(cache_identity)[:24],
            cache_identity,
            rows,
            tokenize,
            progress=progress,
        )
        tokenized_fingerprint = stable_fingerprint(receipt)
    else:
        outputs = {split: [tokenize(row) for row in items] for split, items in rows.items()}
        tokenized_fingerprint = stable_fingerprint(
            {
                split: [
                    (x.instance_id, x.encoder_ids, x.decoder_prefix_ids, x.target_ids)
                    for x in items
                ]
                for split, items in outputs.items()
            }
        )
    for split, items in outputs.items():
        identifiers = (
            [row["instance_id"] for row in items.rows]
            if hasattr(items, "rows")
            else [item.instance_id for item in items]
        )
        if identifiers != pool["record_ids"][split]:
            raise ValueError("Frozen cohort identity changed during tokenization.")
    manifest = {
        "format": "curated_full_pool_sft_v1",
        "pool_fingerprint": pool["fingerprint"],
        "experiment": spec.name,
        "limits": limits,
        "instances": pool["record_ids"],
        "tokenizer_assets": {"encoder": enc_manifest, "decoder": dec_manifest},
        "pool_report": {
            key: pool["report"][key]
            for key in ("research_role", "task_counts", "training_plan", "split")
            if key in pool["report"]
        },
        "tokenized_fingerprint": tokenized_fingerprint,
        "token_storage": "lossless_int32_disk_cache"
        if packed_cache is not None
        else "python_tuples",
    }
    manifest["fingerprint"] = stable_fingerprint(manifest)
    return outputs["train"], outputs["dev"], manifest


def fit_curated_sft(
    model,
    train,
    dev,
    output,
    manifest,
    *,
    epochs=1,
    batch_size=1,
    accumulation=4,
    max_steps=None,
    eval_every=500,
    save_every=500,
    resume_from=None,
    progress=None,
    seed=17,
    event_callback=None,
    memory_policy=None,
) -> dict[str, Any]:
    """One run/model, all epochs by default; checkpoint v2 supports explicit resume."""
    from bifocalgemma.training.offline_sft import fit_offline_sft
    from bifocalgemma.training.policy import configure_policy_model, default_training_profiles

    if epochs < 1 or batch_size < 1 or accumulation < 1 or not train or not dev:
        raise ValueError("Invalid complete-epoch budget.")
    planned_updates = epochs * math.ceil(len(train) / (batch_size * accumulation))
    budget = planned_updates if max_steps is None else max_steps
    if budget < 1:
        raise ValueError("Invalid optimizer update limit.")
    family = (
        "decoder_only"
        if model.encoder is None
        else ("symmetric" if model.spec.tie_encoder_decoder_words else "asymmetric")
    )
    policy = default_training_profiles()[family]
    configure_policy_model(model, policy)
    from bifocalgemma.models.memory_efficient import (
        configure_memory_efficient_training,
        run_training_memory_preflight,
    )

    configure_memory_efficient_training(model, memory_policy)
    memory_preflight = run_training_memory_preflight(model, train, progress=event_callback)
    from importlib.metadata import version

    import torch

    compatibility = {
        "architecture": "native_curated_pool_sft_v1",
        "experiment": model.spec.name,
        "model_spec": model.spec.to_dict(),
        "torch": str(torch.__version__),
        "transformers": version("transformers"),
        "backbone_dimensions": {
            "decoder_hidden_size": model.decoder.config.hidden_size,
            "decoder_vocab_size": model.decoder.config.vocab_size,
            "encoder_hidden_size": model.encoder.config.hidden_size
            if model.encoder is not None
            else None,
        },
        "family": family,
        "data_fingerprint": manifest["fingerprint"],
        "pool_fingerprint": manifest["pool_fingerprint"],
        "training_policy": policy,
        "batch_size": batch_size,
        "accumulation": accumulation,
        "seed": seed,
    }
    result = fit_offline_sft(
        model,
        train,
        dev,
        Path(output),
        compatibility,
        max_steps=budget,
        epochs=epochs,
        batch_size=batch_size,
        accumulation=accumulation,
        training_policy=policy,
        eval_every=eval_every,
        save_every=save_every,
        resume_from=resume_from,
        progress=progress,
        event_callback=event_callback,
        seed=seed,
        data_manifest=manifest,
    )
    result["memory_preflight"] = memory_preflight
    result["planned_full_epoch_updates"] = planned_updates
    result["explicit_max_steps"] = max_steps
    return result
