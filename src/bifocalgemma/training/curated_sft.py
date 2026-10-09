"""Matched full-pool tokenization and complete-epoch SFT for four pairs/two baselines."""

import math
from pathlib import Path

from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.policy import configure_policy_model, default_training_profiles
from bifocalgemma.training.repository_sft import tokenize_repository_record


def prepare_curated_training(spec, pool_directory, tokenizer_cache):
    """Consume all frozen IDs in order; no sampling, re-split or overflow dropping."""
    rows, pool = load_curated_pool(pool_directory)
    limits = pool["report"]["tokens"]["limits"]
    decoder, dec_manifest = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder, enc_manifest = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))
        if spec.encoder
        else (None, None)
    )
    outputs = {}
    for split, items in rows.items():
        outputs[split] = []
        for row in items:
            context = (
                f"### Instruction\n{row['instruction']}\n\n### File: {row['path']}\n{row['before']}"
            )
            tokenized = tokenize_repository_record(
                {**row, "context": context},
                spec,
                encoder,
                decoder,
                encoder_limit=limits["encoder"],
                decoder_limit=limits["pair_decoder"]
                if spec.encoder
                else limits["baseline_decoder"],
            )
            outputs[split].append(tokenized)
        if [item.instance_id for item in outputs[split]] != pool["record_ids"][split]:
            raise ValueError("Frozen cohort identity changed during tokenization.")
    manifest = {
        "format": "curated_full_pool_sft_v1",
        "pool_fingerprint": pool["fingerprint"],
        "experiment": spec.name,
        "limits": limits,
        "instances": pool["record_ids"],
        "tokenizer_assets": {"encoder": enc_manifest, "decoder": dec_manifest},
        "pool_report": pool["report"],
        "tokenized_fingerprint": stable_fingerprint(
            {
                split: [
                    (x.instance_id, x.encoder_ids, x.decoder_prefix_ids, x.target_ids)
                    for x in items
                ]
                for split, items in outputs.items()
            }
        ),
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
):
    """One run/model, all epochs by default; checkpoint v2 supports explicit resume."""
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
    compatibility = {
        "architecture": "native_curated_pool_sft_v1",
        "experiment": model.spec.name,
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
        seed=seed,
        data_manifest=manifest,
    )
    result["planned_full_epoch_updates"] = planned_updates
    result["explicit_max_steps"] = max_steps
    return result
