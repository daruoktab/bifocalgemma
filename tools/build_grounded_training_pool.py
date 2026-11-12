"""Build a separate, locally validated source-grounded proxy pool; never publish it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bifocalgemma.data.context_tasks import normalize_context_task
from bifocalgemma.data.grounded_tasks import VERSION, generate_grounded_tasks
from bifocalgemma.data.offline_sft import curated_pool_digest, load_curated_pool, stable_fingerprint
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.training.repository_sft import tokenize_repository_record
from bifocalgemma.v3.constants import MATRIX_MODELS
from bifocalgemma.v3.matrix import make_experiment_spec

LIMITS = {"encoder": 32768, "pair_decoder": 8192, "baseline_decoder": 41472}


def build_pool(
    source: Path, output: Path, tokenizer_cache: Path, *, max_parents: int = 100, seed: int = 17
):
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Choose a new output directory separate from the frozen source pool.")
    if max_parents < 1:
        raise ValueError("max_parents must be positive.")
    rows, original = load_curated_pool(source)
    parents = {
        split: [r for r in items if r.get("record_type") == "context_sft_v1"]
        for split, items in rows.items()
    }
    train_groups = {row["group"] for row in parents["train"]}
    if train_groups & {row["group"] for row in parents["dev"]}:
        raise ValueError("Parent training/dev groups overlap; a new independent split is required.")
    tokenizers: dict[str, Any] = {}
    specs = [make_experiment_spec(key) for key in MATRIX_MODELS]
    for spec in specs:
        for checkpoint in (spec.encoder, spec.decoder):
            if checkpoint is not None and checkpoint.model_id not in tokenizers:
                tokenizers[checkpoint.model_id] = load_verified_tokenizer(
                    checkpoint, tokenizer_cache
                )[0]
    result: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    rejected: Counter[str] = Counter()
    selected_parents: dict[str, list[str]] = {}
    for split, candidates in parents.items():
        candidates.sort(
            key=lambda r: hashlib.sha256(
                f"{VERSION}:{seed}:{r['instance_id']}".encode()
            ).hexdigest()
        )
        selected = candidates[:max_parents]
        selected_parents[split] = [row["instance_id"] for row in selected]
        for parent in selected:
            for row in generate_grounded_tasks(parent, seed=seed):
                encoder_lengths, paired_lengths, baseline_lengths, target_lengths = [], [], [], []
                try:
                    for spec in specs:
                        item = tokenize_repository_record(
                            row,
                            spec,
                            tokenizers[spec.encoder.model_id] if spec.encoder else None,
                            tokenizers[spec.decoder.model_id],
                            encoder_limit=LIMITS["encoder"],
                            decoder_limit=LIMITS["pair_decoder"]
                            if spec.encoder
                            else LIMITS["baseline_decoder"],
                        )
                        length = len(item.decoder_prefix_ids) + len(item.target_ids) - 1
                        if spec.encoder:
                            encoder_lengths.append(len(item.encoder_ids))
                            paired_lengths.append(length)
                        else:
                            baseline_lengths.append(length)
                        target_lengths.append(len(item.target_ids))
                except ValueError as error:
                    rejected[str(error)] += 1
                    continue
                row["token_lengths"] = {
                    "encoder": max(encoder_lengths),
                    "pair_decoder": max(paired_lengths),
                    "baseline_decoder": max(baseline_lengths),
                    "target": max(target_lengths),
                }
                normalize_context_task(row)
                result[split].append(row)
        if not result[split]:
            raise ValueError(f"No admitted {split} rows; source remains unchanged.")
    # Repeated evidence must never cross the held-out boundary, even if groups drift.
    evidence = {
        split: {row["derivation"]["parent_context_sha256"] for row in items}
        for split, items in result.items()
    }
    if evidence["train"] & evidence["dev"]:
        raise ValueError("Source context crosses training/dev splits.")
    stage = output.with_name("." + output.name + ".building")
    stage.mkdir(parents=True, exist_ok=False)
    for split, items in result.items():
        with (stage / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in items:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "format": "bifocalgemma_curated_pool_v1",
        "files": {
            f"{split}.jsonl": curated_pool_digest(stage / f"{split}.jsonl") for split in result
        },
        "record_ids": {
            split: [row["instance_id"] for row in items] for split, items in result.items()
        },
        "report": {
            "research_role": "source_grounded_proxy_supplement_not_human_software_qa",
            "generator": VERSION,
            "seed": seed,
            "max_parents_per_split": max_parents,
            "parent_pool_fingerprint": original["fingerprint"],
            "parent_files": original["files"],
            "parent_ids": selected_parents,
            "split": "inherited_parent_group_disjoint_v1",
            "snapshot_policy": "inherited_immutable_pre_fix_training_contexts",
            "context_source_license_policy": "nearest_license_for_each_visible_file_v1",
            "tokens": {"limits": LIMITS, "validated_model_keys": list(MATRIX_MODELS)},
            "task_counts": {
                split: dict(Counter(row["task_type"] for row in items))
                for split, items in result.items()
            },
            "rejected": dict(rejected),
            "benchmark_questions_used": False,
            "main_matrix_pool_modified": False,
            "recommended_mixture_ratio": None,
        },
    }
    manifest["fingerprint"] = stable_fingerprint(manifest)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_curated_pool(stage)
    stage.rename(output)
    return {
        "output": str(output),
        "fingerprint": manifest["fingerprint"],
        "counts": {split: len(items) for split, items in result.items()},
        "rejected": dict(rejected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    parser.add_argument("--max-parents", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(
        json.dumps(
            build_pool(
                args.source,
                args.output,
                args.tokenizer_cache,
                max_parents=args.max_parents,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
