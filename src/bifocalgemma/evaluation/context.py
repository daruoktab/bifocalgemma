"""Restartable static-context evaluation with fixed denominators and explicit protocols."""

import json
import os
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.training.stages import atomic_json

EVALUATION_PROTOCOL = "greedy_no_cache_context_v2_explicit_repair_format"


def greedy_decode(*args, **kwargs):
    from bifocalgemma.inference.generation import greedy_decode as generate

    return generate(*args, **kwargs)


def prompt_ids(row, spec, encoder, decoder, *, encoder_limit):
    if row.get("role") != "eval" or any(
        k in row
        for k in (
            "answer",
            "correct_answer",
            "correct_letter",
            "gold_patch",
            "reference",
            "target_patch",
            "test_script",
            "test_code",
            "target_text",
            "groundtruth",
            "is_hard",
        )
    ):
        raise ValueError("benchmark_gold_in_model_input")
    context = row["context"]
    instruction = row["instruction"]
    if row.get("record_type") == "repository_patch_v1" or row.get("task_type") == "repair":
        instruction = "Return only a unified diff fulfilling the instruction.\n" + instruction
    enc = ()
    if spec.encoder:
        enc = tuple(encoder.encode(context, add_special_tokens=True))
        if len(enc) > encoder_limit:
            raise ValueError("encoder_context_overflow")
        prompt = instruction + "\nUse the supplied encoder context."
    else:
        prompt = instruction + "\n\n" + context
    text = decoder.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix = tuple(decoder.encode(text, add_special_tokens=False))
    if not prefix or prefix[0] != decoder.bos_token_id:
        raise ValueError("benchmark_bos_policy")
    return enc, prefix


def strict_letter(text):
    match = re.fullmatch(r"\s*([ABCD])\s*[.)]?\s*", text)
    return match[1] if match else None


def completion_metrics(prediction, reference):
    clean_prediction = "".join(prediction.split())
    clean_reference = "".join(reference.split())
    predicted_ids = set(re.findall(r"\b[A-Za-z_]\w*\b", prediction))
    reference_ids = set(re.findall(r"\b[A-Za-z_]\w*\b", reference))
    overlap = len(predicted_ids & reference_ids)
    f1 = (
        2 * overlap / (len(predicted_ids) + len(reference_ids))
        if predicted_ids or reference_ids
        else 1.0
    )
    return {
        "exact_match": clean_prediction == clean_reference,
        "sequence_similarity": SequenceMatcher(
            None, clean_prediction, clean_reference, autojunk=False
        ).ratio(),
        "regex_identifier_set_f1": f1,
    }


def score_context_case(row, gold, completion, *, directory, python_executable, allow_execution):
    kind = row.get("task_type")
    if kind == "mcq":
        parsed = strict_letter(completion)
        return {
            "correct": parsed == gold["answer"],
            "parsed_answer": parsed,
            "status": "INVALID_OUTPUT" if parsed is None else "SCORED",
        }
    if kind == "completion":
        metrics = completion_metrics(completion, gold["reference"])
        return {**metrics, "correct": metrics["exact_match"], "status": "SCORED"}
    if str(row.get("benchmark", "")).lower() == "gsm8k" or "question" in row and "answer" in gold:
        from bifocalgemma.evaluation.gsm8k import extract_release_answer

        parsed = extract_release_answer(completion)
        return {
            "correct": parsed == gold["answer"],
            "parsed_answer": parsed,
            "status": "INVALID_OUTPUT" if parsed is None else "SCORED",
        }
    if "snapshot_file" in gold:
        from dataclasses import asdict

        from bifocalgemma.evaluation.native_python import score_python_patch
        from bifocalgemma.repository_study import snapshot_from_json

        path = (Path(directory) / gold["snapshot_file"]).resolve()
        if not path.is_relative_to(Path(directory).resolve()):
            raise ValueError("benchmark_snapshot_path")
        result = score_python_patch(
            row["instance_id"],
            snapshot_from_json(path),
            completion,
            gold.get("test_script") or "",
            python_executable=python_executable,
            expected_snapshot_sha256=gold["snapshot_sha256"],
            allow_code_execution=allow_execution,
            timeout_seconds=30,
            expected_python_version=gold.get("python_environment"),
        )
        if result.status == "INFRA_ERROR":
            raise RuntimeError("Functional benchmark infrastructure failed: " + result.reason)
        return {**asdict(result), "correct": result.functional_pass}
    raise ValueError("unsupported_benchmark_scorer")


def run_context_benchmark(
    model,
    spec,
    directory,
    output,
    tokenizer_cache,
    *,
    checkpoint_identity,
    encoder_limit=8192,
    decoder_limit=2048,
    max_new_tokens=2048,
    allow_execution=True,
    python_executable=None,
    progress=None,
    prediction_callback=None,
    upload_every=100,
):
    """Commit each prediction once. A failed process resumes the next uncommitted case."""
    import sys

    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
    from bifocalgemma.training.telemetry import resource_snapshot

    root, output = Path(directory), Path(output)
    inputs_path = (
        root / "eval_inputs.jsonl"
        if (root / "eval_inputs.jsonl").exists()
        else root / "inputs.jsonl"
    )
    gold_path = (
        root / "eval_private/specs.jsonl"
        if (root / "eval_private/specs.jsonl").exists()
        else root / "hidden_gold.jsonl"
    )
    public = list(read_source_rows(inputs_path))
    hidden = list(read_source_rows(gold_path))
    ids = [r["instance_id"] for r in public]
    if not ids or len(set(ids)) != len(ids) or ids != [r["instance_id"] for r in hidden]:
        raise ValueError("benchmark_public_gold_cohort_mismatch")
    gold = {r["instance_id"]: r for r in hidden}
    signature = stable_fingerprint(
        {
            "model": spec.to_dict(),
            "checkpoint": checkpoint_identity,
            "inputs": public,
            "gold": hidden,
            "encoder_limit": encoder_limit,
            "decoder_limit": decoder_limit,
            "max_new_tokens": max_new_tokens,
            "execution": allow_execution,
            "protocol": EVALUATION_PROTOCOL,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "evaluation_state.json"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("Benchmark/checkpoint/protocol changed; choose a new output.")
    else:
        atomic_json(state_path, {"signature": signature, "total": len(ids)})
    predictions = output / "predictions.jsonl"
    completed = []
    if predictions.exists():
        # Only the final incomplete append can be discarded after a process crash.
        with predictions.open("rb+") as stream:
            offset = 0
            for line in stream:
                if not line.endswith(b"\n"):
                    if stream.read():
                        raise ValueError("Corrupt prediction journal before final record.")
                    stream.truncate(offset)
                    break
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if stream.read():
                        raise ValueError("Corrupt prediction journal before final record.")
                    stream.truncate(offset)
                    break
                completed.append(value)
                offset += len(line)
    completed_ids = [r["instance_id"] for r in completed]
    if completed_ids != ids[: len(completed_ids)]:
        raise ValueError("Prediction journal identity/order mismatch.")
    decoder, _ = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))[0] if spec.encoder else None
    )
    started = time.perf_counter()
    segment_initial_count = len(completed)
    for index, row in enumerate(public[len(completed) :], len(completed) + 1):
        cap = min(encoder_limit, row.get("max_encoder_tokens", encoder_limit))
        enc, prefix = prompt_ids(row, spec, encoder, decoder, encoder_limit=cap)
        # Same output allowance for pairs and baselines: reserve the paired IT prefix.
        paired_prompt = row["instruction"] + "\nUse the supplied encoder context."
        paired_rendered = decoder.apply_chat_template(
            [{"role": "user", "content": paired_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        paired_prefix_size = len(decoder.encode(paired_rendered, add_special_tokens=False))
        new_tokens = min(
            row.get("max_new_tokens", max_new_tokens), decoder_limit - paired_prefix_size
        )
        if new_tokens < 1:
            raise ValueError("benchmark_decoder_prefix_exceeds_budget")
        total = decoder_limit if spec.encoder else cap + decoder_limit + 512
        generated = greedy_decode(
            model, enc, prefix, max_new_tokens=new_tokens, max_total_decoder_tokens=total
        )
        tokens = generated.token_ids
        if tokens and tokens[-1] in (1, 106, 50):
            tokens = tokens[:-1]
        completion = decode_text(decoder, tokens, skip_special_tokens=False)
        scoring = score_context_case(
            row,
            gold[row["instance_id"]],
            completion,
            directory=root,
            python_executable=python_executable or sys.executable,
            allow_execution=allow_execution,
        )
        record = {
            "instance_id": row["instance_id"],
            "completion": completion,
            "stop_reason": generated.stop_reason,
            "encoder_tokens": len(enc),
            "decoder_prefix_tokens": len(prefix),
            "generated_tokens": len(tokens),
            **scoring,
        }
        with predictions.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        completed.append(record)
        if prediction_callback and (index % upload_every == 0 or index == len(ids)):
            prediction_callback()
        if progress:
            progress(
                {
                    "stage": "benchmark_progress",
                    "items_done": index,
                    "items_total": len(ids),
                    "message": f"{root.name}: {index:,}/{len(ids):,}",
                    "benchmark": root.name,
                    "benchmark_elapsed_seconds": time.perf_counter() - started,
                    "benchmark_eta_seconds": (time.perf_counter() - started)
                    / max(1, index - segment_initial_count)
                    * max(0, len(ids) - index),
                    "benchmark_correct": sum(r["correct"] is True for r in completed),
                    "last_prediction": record,
                    **resource_snapshot(next(model.parameters()).device),
                }
            )
    scored = all(r["correct"] is not None for r in completed)
    report = {
        "status": "COMPLETED",
        "signature": signature,
        "experiment": spec.name,
        "total": len(ids),
        "correct": sum(bool(r["correct"]) for r in completed) if scored else None,
        "accuracy": sum(bool(r["correct"]) for r in completed) / len(ids) if scored else None,
        "score_available": scored,
        "statuses": dict(Counter(r["status"] for r in completed)),
        "generation_length_limit": sum(r["stop_reason"] == "length_limit" for r in completed),
        "segment_seconds": time.perf_counter() - started,
        "official_protocol": False,
        "protocol_version": EVALUATION_PROTOCOL,
        "protocol": "greedy_no_temporal_cache; fixed eligible cohort; no context truncation",
        "resources": resource_snapshot(next(model.parameters()).device),
    }
    for key in ("exact_match", "sequence_similarity", "regex_identifier_set_f1"):
        if completed and key in completed[0]:
            report[key] = sum(float(r[key]) for r in completed) / len(ids)
    atomic_json(output / "benchmark_report.json", report)
    return report
