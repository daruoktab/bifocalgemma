"""Repository export tokenization and matched-policy SFT for all six configurations."""

from pathlib import Path

from bifocalgemma.config import EXPERIMENT_SPECS
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
    from bifocalgemma.data.context_tasks import tokenize_context_task

    if row.get("role") != "train":
        raise ValueError("evaluation_records_are_not_training_data")
    if row.get("record_type") not in {"context_sft_v1", "grounded_context_sft_v1"}:
        # Raw repository rows always supervise a unified diff. Use the same
        # whitespace-preserving target encoder as the context SFT path.
        row = {**row, "target_text": row["target_patch"], "task_type": "repair"}
        row.setdefault("source", "unknown")
    return tokenize_context_task(
        row, spec, encoder, decoder, encoder_limit=encoder_limit, decoder_limit=decoder_limit
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
