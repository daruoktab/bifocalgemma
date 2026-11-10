"""Repository export tokenization and matched-policy SFT for all six configurations."""

from pathlib import Path

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.offline_sft import (
    prepare_editing_records,
    split_editing_records,
    stable_fingerprint,
)
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.data.tokenization import load_verified_tokenizer


def tokenize_repository_record(
    row, spec, encoder, decoder, *, encoder_limit=4096, decoder_limit=8192
):
    if row.get("record_type") == "context_sft_v1":
        from bifocalgemma.data.context_tasks import tokenize_context_task

        return tokenize_context_task(
            row, spec, encoder, decoder, encoder_limit=encoder_limit, decoder_limit=decoder_limit
        )
    if row.get("role") != "train":
        raise ValueError("evaluation_records_are_not_training_data")
    context = row["context"]
    encoder_ids = tuple(encoder.encode(context, add_special_tokens=True)) if spec.encoder else ()
    if spec.encoder:
        prompt = "Return only a unified diff fulfilling the instruction.\n\n" + row["instruction"]
    else:
        prompt = "Return only a unified diff fulfilling the instruction.\n\n" + context
    messages = [{"role": "user", "content": prompt}]
    prefix = decoder.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = decoder.apply_chat_template(
        messages + [{"role": "assistant", "content": row["target_patch"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
    full_ids = tuple(decoder.encode(full.rstrip(), add_special_tokens=False))
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("prefix_target_boundary")
    target = full_ids[len(prefix_ids) :]
    if not prefix_ids or prefix_ids[0] != 2 or not target or target[-1] != 106:
        raise ValueError("repository_special_token_policy")
    if len(encoder_ids) > encoder_limit or len(full_ids) - 1 > decoder_limit:
        raise ValueError("repository_token_overflow")
    return TokenizedRepairExample(
        row["instance_id"],
        encoder_ids,
        prefix_ids,
        target,
        metadata={
            "source": row.get("source", "unknown"),
            "task_type": row.get("task_type", "repair"),
        },
    )


def prepare_repository_training(
    path,
    spec,
    tokenizer_cache,
    *,
    encoder_limit=4096,
    decoder_limit=8192,
    train_limit=2000,
    dev_limit=200,
    seed=17,
):
    records, filters = prepare_editing_records(
        read_source_rows(path), source="editpackft", raw_limit=100000
    )
    if not all(r.get("record_type") == "repository_patch_v1" for r in records):
        raise ValueError("This runner requires repository exports, not mixed legacy records.")
    # Freeze one eligible cohort across all six models before grouped splitting.
    # Baselines have room for the same visible context plus decoder targets.
    cache = {}

    def assets(reference):
        if reference.model_id not in cache:
            cache[reference.model_id] = load_verified_tokenizer(reference, Path(tokenizer_cache))
        return cache[reference.model_id][0]

    baseline_limit = encoder_limit + decoder_limit + 512
    matched, matched_filters = [], {}
    for row in records:
        try:
            for candidate in EXPERIMENT_SPECS:
                tokenize_repository_record(
                    row,
                    candidate,
                    assets(candidate.encoder) if candidate.encoder else None,
                    assets(candidate.decoder),
                    encoder_limit=encoder_limit,
                    decoder_limit=decoder_limit if candidate.encoder else baseline_limit,
                )
            matched.append(row)
        except ValueError as error:
            reason = str(error)
            matched_filters[reason] = matched_filters.get(reason, 0) + 1
    records = matched
    train, dev = split_editing_records(
        records, seed=seed, train_limit=train_limit, dev_limit=dev_limit
    )
    decoder, dec_manifest = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder, enc_manifest = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))
        if spec.encoder
        else (None, None)
    )
    counts = {}
    tokenized = []
    for split in (train, dev):
        examples = []
        for row in split:
            try:
                examples.append(
                    tokenize_repository_record(
                        row,
                        spec,
                        encoder,
                        decoder,
                        encoder_limit=encoder_limit,
                        decoder_limit=decoder_limit if spec.encoder else baseline_limit,
                    )
                )
            except ValueError as error:
                reason = str(error)
                counts[reason] = counts.get(reason, 0) + 1
        tokenized.append(examples)
    if not all(tokenized):
        raise ValueError("Empty train/dev after token filtering.")
    manifest = {
        "format": "repository_context_diff_v1",
        "split_policy": "canonical_repository_group_holdout",
        "record_filter_counts": filters,
        "matched_cohort_filter_counts": matched_filters,
        "matched_cohort_ids": [r["instance_id"] for r in records],
        "token_filter_counts": counts,
        "seed": seed,
        "encoder_limit": encoder_limit,
        "decoder_limit": decoder_limit if spec.encoder else baseline_limit,
        "tokenizer_assets": {"encoder": enc_manifest, "decoder": dec_manifest},
        "instances": {
            "train": [x.instance_id for x in tokenized[0]],
            "dev": [x.instance_id for x in tokenized[1]],
        },
        "source_provenance": [
            {
                k: row[k]
                for k in ("instance_id", "source", "source_revision", "snapshot_sha256", "group")
            }
            for row in records
        ],
        "tokenized_fingerprint": stable_fingerprint(
            [
                [(x.instance_id, x.encoder_ids, x.decoder_prefix_ids, x.target_ids) for x in items]
                for items in tokenized
            ]
        ),
    }
    manifest["fingerprint"] = stable_fingerprint(manifest)
    return tokenized[0], tokenized[1], manifest


def fit_repository_sft(
    model,
    train,
    dev,
    output,
    manifest,
    *,
    max_steps=100,
    epochs=1,
    accumulation=4,
    resume_from=None,
):
    from bifocalgemma.training.offline_sft import fit_offline_sft
    from bifocalgemma.training.policy import configure_policy_model, default_training_profiles

    family = (
        "decoder_only"
        if model.encoder is None
        else ("symmetric" if model.spec.tie_encoder_decoder_words else "asymmetric")
    )
    policy = default_training_profiles()[family]
    configure_policy_model(model, policy)
    compatibility = {
        "architecture": "native_repository_sft_v1",
        "experiment": model.spec.name,
        "family": family,
        "data_fingerprint": manifest["fingerprint"],
        "training_policy": policy,
    }
    return fit_offline_sft(
        model,
        train,
        dev,
        Path(output),
        compatibility,
        max_steps=max_steps,
        epochs=epochs,
        accumulation=accumulation,
        training_policy=policy,
        resume_from=resume_from,
    )
