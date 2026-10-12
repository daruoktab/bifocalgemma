"""Recheck every frozen benchmark with the official config-declared Gemma tokenizer."""
# ruff: noqa: E402 -- standalone static-audit bootstrap

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["USE_TORCH"] = "0"
sys.path.insert(0, str(ROOT / "src"))

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.data.curation import file_digest
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.evaluation.context import prompt_ids
from bifocalgemma.training.stages import atomic_json


def audit(directory, assets):
    root = Path(directory)
    report = json.loads((root / "preparation_report.json").read_text(encoding="utf-8"))
    for filename, field in [
        ("eval_inputs.jsonl", "public_sha256"),
        ("eval_private/specs.jsonl", "gold_sha256"),
    ]:
        if file_digest(root / filename) != report[field]:
            raise ValueError("benchmark_bytes_changed")
    public = list(read_source_rows(root / "eval_inputs.jsonl"))
    hidden = list(read_source_rows(root / "eval_private/specs.jsonl"))
    ids = [r["instance_id"] for r in public]
    if (
        len(set(ids)) != len(ids)
        or ids != [r["instance_id"] for r in hidden]
        or len(ids) != report["count"]
    ):
        raise ValueError("benchmark_ids_or_counts_changed")
    maxima = {s.name: {"encoder": 0, "prefix": 0} for s in EXPERIMENT_SPECS}
    cached = {}
    scope = Counter()
    for index, (row, gold) in enumerate(zip(public, hidden, strict=True), 1):
        scope[row["task_scope"]] += 1
        for spec in EXPERIMENT_SPECS:
            key = (
                spec.encoder.model_id if spec.encoder else None,
                spec.decoder.model_id,
                row["context"],
                row["instruction"],
            )
            if key not in cached:
                enc, prefix = prompt_ids(
                    row,
                    spec,
                    assets.get(spec.encoder.model_id) if spec.encoder else None,
                    assets[spec.decoder.model_id],
                    encoder_limit=32768,
                )
                cached[key] = {"encoder": len(enc), "prefix": len(prefix)}
            lengths = cached[key]
            if lengths != row["token_lengths"][spec.name]:
                raise ValueError("official_tokenizer_dispatch_changed_lengths:" + spec.name)
            cap = 8192 if spec.encoder else 41472
            if lengths["prefix"] + row["max_new_tokens"] > cap:
                raise ValueError("benchmark_generation_budget")
            for field, count in lengths.items():
                maxima[spec.name][field] = max(maxima[spec.name][field], count)
        cached.clear()  # Do not retain repository text/IDs across the complete cohort.
        if row["task_type"] == "mcq" and gold["answer"] not in "ABCD":
            raise ValueError("invalid_mcq_gold")
        if row["task_type"] == "completion":
            if len(gold["reference"].strip()) >= 64 and gold["reference"] in row["context"]:
                raise ValueError("visible_reference")
            for name in {s.decoder.model_id for s in EXPERIMENT_SPECS}:
                if (
                    len(assets[name].encode(gold["reference"], add_special_tokens=False))
                    > row["max_new_tokens"]
                ):
                    raise ValueError("reference_output_budget")
        if index % 1000 == 0:
            print(
                json.dumps(
                    {"stage": "benchmark_token_audit", "collection": root.name, "done": index}
                ),
                flush=True,
            )
    return {
        "count": len(ids),
        "max_lengths": maxima,
        "task_scopes": dict(scope),
        "public_gold_sha256_verified": True,
        "official_class_matches_frozen_lengths": True,
        "silent_truncation": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    assets = {}
    for spec in EXPERIMENT_SPECS:
        for ref in (spec.encoder, spec.decoder):
            if ref and ref.model_id not in assets:
                assets[ref.model_id] = load_verified_tokenizer(
                    ref, ROOT / "outputs/context_tokenizers"
                )[0]
    results = {}
    roots = (
        []
        if args.legacy_only
        else sorted((ROOT / "outputs/context_benchmarks_v1").glob("*_gemma32k_dec8k_v1"))
    )
    for root in roots:
        results[root.name] = audit(root, assets)
        print(
            json.dumps(
                {"collection": root.name, "status": "PASS", "count": results[root.name]["count"]}
            ),
            flush=True,
        )
    if args.legacy_only:
        for name, directory, filename, expected in [
            ("humanevalfix_python_v2", "humanevalfix_prepared", "eval_inputs.jsonl", 164),
            ("gsm8k_zero_shot_v2", "gsm8k_zero_shot_v1", "inputs.jsonl", 1319),
            ("gsm8k_eight_shot_v2", "gsm8k_eight_shot_v1", "inputs.jsonl", 1319),
        ]:
            rows = list(read_source_rows(ROOT / "outputs" / directory / filename))
            maxima = {s.name: {"encoder": 0, "prefix": 0} for s in EXPERIMENT_SPECS}
            if len(rows) != expected:
                raise ValueError("legacy_benchmark_count_changed")
            for row in rows:
                for spec in EXPERIMENT_SPECS:
                    enc, prefix = prompt_ids(
                        row,
                        spec,
                        assets.get(spec.encoder.model_id) if spec.encoder else None,
                        assets[spec.decoder.model_id],
                        encoder_limit=32768,
                    )
                    cap = 8192 if spec.encoder else 41472
                    if len(prefix) >= cap:
                        raise ValueError("legacy_benchmark_prompt_overflow")
                    maxima[spec.name]["encoder"] = max(maxima[spec.name]["encoder"], len(enc))
                    maxima[spec.name]["prefix"] = max(maxima[spec.name]["prefix"], len(prefix))
            results[name] = {"count": len(rows), "max_lengths": maxima, "silent_truncation": False}
            print(
                json.dumps({"collection": name, "status": "PASS", "count": len(rows)}), flush=True
            )
    payload = {"format": "bifocalgemma_context_token_audit_v1", "collections": results}
    payload["fingerprint"] = stable_fingerprint(payload)
    filename = (
        "context_legacy_benchmark_token_audit_32k_v1.json"
        if args.legacy_only
        else "context_token_audit_32k_v1.json"
    )
    atomic_json(ROOT / "outputs" / filename, payload)


if __name__ == "__main__":
    main()
