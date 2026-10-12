"""Pinned GSM8K preparation and release-style scoring, separate from repair SFT."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SOURCE_FILES = {
    "train": ("17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465", 7473),
    "test": ("3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14", 1319),
}
INSTRUCTION = (
    "Solve the math problem step by step. End with #### followed by the final numeric answer."
)


def extract_release_answer(completion):
    """Match the release's first #### answer and comma removal; no numeric eval."""
    match = re.search(r"#### (\-?[0-9\.\,]+)", completion)
    return match.group(1).strip().replace(",", "") if match else None


def verified_source_rows(directory, split):
    expected_sha, expected_count = SOURCE_FILES[split]
    data = (Path(directory) / (split + ".jsonl")).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("gsm8k_source_hash_mismatch:" + split)
    rows = [json.loads(line) for line in data.splitlines() if line]
    if len(rows) != expected_count:
        raise ValueError("gsm8k_source_count_mismatch:" + split)
    return rows


def fetch_gsm8k_sources(directory):
    """Opt-in fetch of immutable public JSONL; verify hashes before caching."""
    import urllib.request

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for split, (expected_sha, _) in SOURCE_FILES.items():
        destination = directory / (split + ".jsonl")
        if destination.exists():
            verified_source_rows(directory, split)
            continue
        url = f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/grade_school_math/data/{split}.jsonl"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(8_000_001)
        if len(data) > 8_000_000 or hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError("gsm8k_download_hash_or_size")
        destination.write_bytes(data)


def make_records(test_rows, train_rows, *, shots=0):
    """Pure adapter for tests; production preparation verifies both source files."""
    if shots not in (0, 8) or (shots and len(train_rows) < shots):
        raise ValueError("gsm8k_shots_must_be_zero_or_eight")
    examples = "\n\n".join(
        "Question: " + row["question"] + "\nAnswer: " + re.sub(r"<<.*?>>", "", row["answer"])
        for row in train_rows[:shots]
    )
    inputs, gold = [], []
    for offset, row in enumerate(test_rows):
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
            raise ValueError("gsm8k_missing_question_or_answer")
        reference = extract_release_answer(answer)
        if reference is None:
            raise ValueError("gsm8k_invalid_reference_answer")
        identifier = f"gsm8k:test:{offset:04d}"
        inputs.append(
            {
                "instance_id": identifier,
                "role": "eval",
                "benchmark": "gsm8k",
                "question": question,
                "instruction": INSTRUCTION,
                "context": (examples + "\n\n" if examples else "")
                + "Question: "
                + question
                + "\nAnswer:",
            }
        )
        gold.append({"instance_id": identifier, "answer": reference})
    return inputs, gold, hashlib.sha256(examples.encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare_gsm8k(source_directory, output, *, shots=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new GSM8K output directory.")
    inputs, gold, prefix_sha = make_records(
        verified_source_rows(source_directory, "test"),
        verified_source_rows(source_directory, "train"),
        shots=shots,
    )
    output.mkdir(parents=True)
    for filename, rows in (("inputs.jsonl", inputs), ("hidden_gold.jsonl", gold)):
        (output / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    manifest = {
        "format": "bifocalgemma_gsm8k_eval_v1",
        "source_revision": REVISION,
        "source_files": {key: list(value) for key, value in SOURCE_FILES.items()},
        "role": "eval",
        "shots": shots,
        "demonstration_prefix_sha256": prefix_sha,
        "instances": [row["instance_id"] for row in inputs],
        "files": {name: digest(output / name) for name in ("inputs.jsonl", "hidden_gold.jsonl")},
        "protocol": {
            "instruction": INSTRUCTION,
            "enable_thinking": False,
            "calculator": False,
            "tools": False,
            "decoding": "greedy",
            "metric": "release_regex_string_accuracy",
            "test_answers_in_input": False,
        },
    }
    manifest["fingerprint"] = fingerprint(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_gsm8k(directory, *, include_gold=False):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if manifest.get("format") != "bifocalgemma_gsm8k_eval_v1" or fingerprint(
        identity
    ) != manifest.get("fingerprint"):
        raise ValueError("gsm8k_manifest_integrity")
    if manifest.get("source_revision") != REVISION or manifest.get("role") != "eval":
        raise ValueError("gsm8k_source_or_role")
    result = []
    for filename in ("inputs.jsonl", "hidden_gold.jsonl") if include_gold else ("inputs.jsonl",):
        if digest(root / filename) != manifest["files"][filename]:
            raise ValueError("gsm8k_file_integrity:" + filename)
        rows = [
            json.loads(line)
            for line in (root / filename).read_text(encoding="utf-8").splitlines()
            if line
        ]
        identifiers = [row["instance_id"] for row in rows]
        if identifiers != manifest["instances"] or len(set(identifiers)) != len(identifiers):
            raise ValueError("gsm8k_frozen_identity")
        if filename == "inputs.jsonl" and any(
            row.get("role") != "eval"
            or any(key in row for key in ("answer", "target_patch", "solution"))
            for row in rows
        ):
            raise ValueError("gsm8k_target_in_input_or_wrong_role")
        result.append(rows)
    return (*result, manifest)


def score_gsm8k(inputs, gold, predictions):
    identifiers = [row["instance_id"] for row in inputs]
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or identifiers != [row["instance_id"] for row in gold]
    ):
        raise ValueError("gsm8k_score_identity")
    predictions = list(predictions)
    experiments = {row["experiment"] for row in predictions if "experiment" in row}
    if len(experiments) > 1:
        raise ValueError("gsm8k_mixed_experiment_predictions")
    by_id = {}
    for row in predictions:
        identifier = row["instance_id"]
        if identifier not in identifiers or identifier in by_id:
            raise ValueError("gsm8k_unknown_or_duplicate_prediction")
        by_id[identifier] = row
    counts = Counter()
    results = []
    for item, reference in zip(inputs, gold, strict=True):
        identifier = item["instance_id"]
        predicted = by_id.get(identifier)
        answer = None
        if predicted is None:
            status = "missing_prediction"
        elif predicted.get("status", "OK") != "OK":
            status = "generation_error"
        else:
            completion = predicted.get("completion")
            if not isinstance(completion, str):
                raise ValueError("gsm8k_completion_must_be_text")
            answer = extract_release_answer(completion)
            status = (
                "invalid_answer"
                if answer is None
                else ("correct" if answer == reference["answer"] else "incorrect")
            )
            if predicted.get("stop_reason") == "length_limit":
                counts["generation_length_limit"] += 1
        counts[status] += 1
        results.append({"instance_id": identifier, "status": status, "parsed_answer": answer})
    return {
        "metric": "release_regex_string_accuracy",
        "total": len(inputs),
        "correct": counts["correct"],
        "accuracy": counts["correct"] / len(inputs),
        "counts": dict(counts),
        "results": results,
    }


def gsm8k_prompt_ids(spec, row, encoder, decoder, *, max_encoder_tokens=2048):
    """Public math prompt only; verified tokenizers are loaded once by the runner."""
    if not 1 <= max_encoder_tokens <= 131072:
        raise ValueError("gsm8k_invalid_encoder_budget")
    if row.get("role") != "eval" or any(
        key in row for key in ("answer", "target_patch", "solution")
    ):
        raise ValueError("gsm8k_target_in_input_or_wrong_role")
    enc = ()
    if spec.encoder:
        if encoder is None:
            raise ValueError("gsm8k_missing_encoder_tokenizer")
        enc = tuple(encoder.encode(row["context"], add_special_tokens=True))
        if len(enc) > max_encoder_tokens:
            raise ValueError("gsm8k_encoder_overflow")
        prompt = row["instruction"] + "\nUse the supplied encoder context."
    else:
        prompt = row["instruction"] + "\n\n" + row["context"]
    rendered = decoder.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix = tuple(decoder.encode(rendered, add_special_tokens=False))
    if not prefix or prefix[0] != decoder.bos_token_id:
        raise ValueError("gsm8k_bos_policy")
    return enc, prefix


def generate_gsm8k(
    model,
    spec,
    inputs,
    tokenizer_cache,
    *,
    max_new_tokens=2048,
    max_encoder_tokens=2048,
    max_total_decoder_tokens=4096,
):
    """Generate from public inputs only; no gold argument or code execution."""
    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
    from bifocalgemma.inference.generation import greedy_decode

    if model.spec != spec:
        raise ValueError("gsm8k_model_spec_mismatch")
    decoder, _ = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))[0] if spec.encoder else None
    )
    for row in inputs:
        enc, prefix = gsm8k_prompt_ids(
            spec, row, encoder, decoder, max_encoder_tokens=max_encoder_tokens
        )
        generated = greedy_decode(
            model,
            enc,
            prefix,
            max_new_tokens=max_new_tokens,
            max_total_decoder_tokens=max_total_decoder_tokens,
        )
        ids = generated.token_ids
        if ids and ids[-1] in (1, 106, 50):
            ids = ids[:-1]
        yield {
            "instance_id": row["instance_id"],
            "experiment": spec.name,
            "completion": decode_text(decoder, ids, skip_special_tokens=False),
            "status": "OK",
            "stop_reason": generated.stop_reason,
        }


def generate_run(args):
    """Original local checkpoints only; no automatic weights/downloads or gold access."""
    from importlib.metadata import version

    import torch

    from bifocalgemma.config import get_experiment
    from bifocalgemma.models.pretrained import (
        load_local_native_model,
        load_sft_adapters_for_inference,
    )

    metadata_path = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("Choose a new GSM8K prediction path.")
    public, manifest = load_gsm8k(args.prepared)
    spec = get_experiment(args.experiment)
    from bifocalgemma.data.tokenization import load_verified_tokenizer

    decoder, _ = load_verified_tokenizer(spec.decoder, args.tokenizer_cache)
    encoder = (
        load_verified_tokenizer(spec.encoder, args.tokenizer_cache)[0] if spec.encoder else None
    )
    if args.max_new_tokens < 1 or not 1 <= args.decoder_limit <= 131072:
        raise ValueError("gsm8k_invalid_generation_budget")
    for row in public:
        _, prefix = gsm8k_prompt_ids(
            spec, row, encoder, decoder, max_encoder_tokens=args.encoder_limit
        )
        if len(prefix) + args.max_new_tokens > args.decoder_limit:
            raise ValueError("gsm8k_decoder_budget_exceeded_before_model_loading")
    model, provenance = load_local_native_model(
        spec, args.decoder_checkpoint, args.encoder_checkpoint
    )
    if args.adapter_checkpoint:
        provenance["sft_checkpoint"] = load_sft_adapters_for_inference(
            model, args.adapter_checkpoint
        )
    torch.cuda.reset_peak_memory_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.output.open("x", encoding="utf-8") as handle:
        for row in generate_gsm8k(
            model,
            spec,
            public,
            args.tokenizer_cache,
            max_new_tokens=args.max_new_tokens,
            max_encoder_tokens=args.encoder_limit,
            max_total_decoder_tokens=args.decoder_limit,
        ):
            row["manifest_fingerprint"] = manifest["fingerprint"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            print(json.dumps({"completed": completed, "total": len(public)}), flush=True)
    report = {
        "status": "COMPLETED",
        "count": completed,
        "dataset_fingerprint": manifest["fingerprint"],
        "experiment": spec.name,
        "provenance": provenance,
        "max_new_tokens": args.max_new_tokens,
        "encoder_limit": args.encoder_limit,
        "decoder_limit": args.decoder_limit,
        "use_temporal_cache": False,
        "predictions_sha256": digest(args.output),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_scope": "inference after model load, not training memory",
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
            "gpu": torch.cuda.get_device_name(),
        },
    }
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-directory", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--shots", type=int, choices=(0, 8), default=0)
    prepare.add_argument(
        "--fetch", action="store_true", help="Fetch pinned public JSONL, not model weights."
    )
    score = sub.add_parser("score")
    score.add_argument("--prepared", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    from bifocalgemma.config import EXPERIMENT_NAMES

    generate = sub.add_parser("generate")
    generate.add_argument("--prepared", type=Path, required=True)
    generate.add_argument("--experiment", choices=EXPERIMENT_NAMES, required=True)
    generate.add_argument("--decoder-checkpoint", type=Path, required=True)
    generate.add_argument("--encoder-checkpoint", type=Path)
    generate.add_argument("--adapter-checkpoint", type=Path)
    generate.add_argument("--tokenizer-cache", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--encoder-limit", type=int, default=2048)
    generate.add_argument("--decoder-limit", type=int, default=4096)
    generate.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.fetch:
            fetch_gsm8k_sources(args.source_directory)
        manifest = prepare_gsm8k(args.source_directory, args.output, shots=args.shots)
        print(
            json.dumps(
                {
                    "prepared": len(manifest["instances"]),
                    "shots": args.shots,
                    "fingerprint": manifest["fingerprint"],
                }
            )
        )
    elif args.command == "generate":
        generate_run(args)
    else:
        if args.output.exists():
            raise FileExistsError("Choose a new GSM8K score path.")
        inputs, gold, manifest = load_gsm8k(args.prepared, include_gold=True)
        predictions = [
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if any(row.get("manifest_fingerprint") != manifest["fingerprint"] for row in predictions):
            raise ValueError("gsm8k_prediction_protocol_mismatch")
        report = score_gsm8k(inputs, gold, predictions)
        report["manifest_fingerprint"] = manifest["fingerprint"]
        report["predictions_sha256"] = digest(args.predictions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
