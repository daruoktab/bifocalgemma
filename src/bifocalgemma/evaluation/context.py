"""Restartable static-context evaluation with fixed denominators and explicit protocols."""

import hashlib
import json
import keyword
import os
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.training.stages import atomic_json

EVALUATION_PROTOCOL = "greedy_no_cache_context_v2_explicit_repair_format"
V2_EVALUATION_PROTOCOL = "greedy_cached_normalized_merged_v3_exact_prompt_budget"
BENCHMARK_RESET_ID = "2026-10-26-exact-prompt-budget-v3"


def greedy_decode(*args, **kwargs):
    from bifocalgemma.inference.generation import greedy_decode as generate

    return generate(*args, **kwargs)


def decoder_prompt(row, *, paired):
    """One prompt renderer for actual decoding and the shared output allowance."""
    instruction = row["instruction"]
    if row.get("record_type") == "repository_patch_v1" or row.get("task_type") == "repair":
        instruction = "Return only a unified diff fulfilling the instruction.\n" + instruction
    if paired:
        return instruction
    return instruction + "\n\n" + row["context"]


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
    enc = ()
    if spec.encoder:
        enc = tuple(encoder.encode(context, add_special_tokens=True))
        if len(enc) > encoder_limit:
            raise ValueError("encoder_context_overflow")
    prompt = decoder_prompt(row, paired=spec.encoder is not None)
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


DELIM_OPEN = r"[*'\"`\(\{\[]*"
DELIM_CLOSE = r"[*'\"`\)\}\]]*"
LETTER = r"([A-Da-d])"
LETTER_TOKEN = rf"{DELIM_OPEN}{LETTER}{DELIM_CLOSE}(?!\w)"
LETTER_RAW = r"[A-Da-d]"
LETTER_TOKEN_NC = rf"{DELIM_OPEN}{LETTER_RAW}{DELIM_CLOSE}(?!\w)"
OPT_ADVERBS = r"(?:(?:clearly|definitely|actually|obviously|certainly|surely|indeed|simply|basically|likely|probably|\w+ly)\s+){0,2}"
CONJ_GUARD = (
    rf"(?!\s*{DELIM_CLOSE}\s+(?:and|or|nor|vs)\s+(?:option\s+|choice\s+)?{LETTER_TOKEN_NC})"
)


def fallback_letter_regex(text: str) -> str | None:
    """Resilient multi-stage regex extractor for reasoning preambles from models like E4B."""
    if not text:
        return None

    # Ambiguity check: e.g. "A or B", "A and B", "A vs B", "between option A and option D"
    has_ambiguity = bool(
        re.search(
            rf"{LETTER_TOKEN_NC}\s+(?:or|nor|vs|and)\s+(?:option\s+|choice\s+)?{LETTER_TOKEN_NC}",
            text,
            re.I,
        )
        or re.search(
            rf"\b(?:between|either)\b.*?{LETTER_TOKEN_NC}.*?\b(?:and|or|nor)\b.*?{LETTER_TOKEN_NC}",
            text,
            re.I | re.S,
        )
    )

    # Priority 1: Conclusive explicit statements with positive resolution
    conclusive_patterns = [
        rf"(?:ultimately\s+)?(?:option\s+|choice\s+)?{LETTER_TOKEN}\s+(?:is\s+(?:{OPT_ADVERBS})?(?:the\s+)?correct\s*(?:answer|choice|option)?){CONJ_GUARD}",
        rf"(?:final\s+answer|correct\s+(?:answer|option|choice|letter)|the\s+correct\s+answer)(?:\s+is)?(?:\s*:)?\s*{OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}",
        rf"(?:therefore|thus|hence|so|in\s+conclusion|finally)[,\s]+(?:(?:[^\w\n]*\w+){{0,6}}\s+)?(?:(?:the\s+answer|option|choice)\s+is\s+|it\s+is\s+)?{LETTER_TOKEN}\s*[.]?\s*$",
        rf"(?<!not\s)(?:the\s+answer|answer)\s+is(?:\s*:)?\s*(?!(?:(?:\w+\s+)?not\b)){OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}",
        rf"(?:it\s+is|it's)\s+{OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}(?:\.|\b|\s|$)",
        rf"(?:select|choose|pick)(?:\s+option|\s+choice|\s+letter)?(?:\s*:)?\s*{OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}",
    ]
    for pat in conclusive_patterns:
        matches = list(re.finditer(pat, text, re.I))
        if matches:
            return matches[-1].group(1).upper()

    if has_ambiguity:
        return None

    # Priority 2: Structured prefix pattern: "Answer: B", "Option: B", "Choice: B", "Option (C)"
    prefix_pat = rf"(?:answer|option|choice)(?:\s+is)?(?:\s*:)?\s*(?!(?:(?:\w+\s+)?not\b)){OPT_ADVERBS}{LETTER_TOKEN}(?!\s*{DELIM_CLOSE}\s+(?:is\s+wrong|is\s+incorrect|is\s+not|is\s+suboptimal|is\s+invalid|is\s+false|is\s+not\s+right|and|or|nor|vs)){CONJ_GUARD}"
    prefix_matches = list(re.finditer(prefix_pat, text, re.I))
    if prefix_matches:
        return prefix_matches[-1].group(1).upper()

    # Priority 3: Single unique isolated letter candidate across text (guarded against negative statements)
    letters = re.findall(rf"(?<![<\w]){LETTER}(?![>\w])", text, re.I)
    upper_letters = [item.upper() for item in letters]
    if len(set(upper_letters)) == 1:
        cand = upper_letters[0]
        cand_token = rf"{DELIM_OPEN}{cand}{DELIM_CLOSE}(?!\w)"
        neg_guard = (
            re.search(
                rf"\b(?:not|never|no|isn['’]t|wasn['’]t|cannot|can['’]t|couldn['’]t|won['’]t)\b(?:\s+\w+){{0,2}}\s+(?:option\s+|choice\s+)?{cand_token}",
                text,
                re.I,
            )
            or re.search(
                rf"(?:option\s+|choice\s+)?{cand_token}\s+(?:is\s+)?(?:not\s+correct|wrong|incorrect|invalid|suboptimal|false|not\s+right)",
                text,
                re.I,
            )
            or re.search(
                rf"(?:answer|option|choice)\s+is\s+(?:(?:\w+\s+){{0,2}}not\s+){cand_token}",
                text,
                re.I,
            )
        )
        if not neg_guard:
            return cand

    return None


def strict_letter(text: str, *, allow_fallback: bool = False) -> str | None:
    """Extract option letter. Strict exact match by default; resilient fallback regex when enabled."""
    if not text:
        return None
    # Stage 1: Exact isolated letter with optional delimiters and punctuation
    match = re.fullmatch(rf"\s*{DELIM_OPEN}{LETTER}{DELIM_CLOSE}\s*[.)]?\s*", text, re.I)
    if match:
        return match.group(1).upper()
    if allow_fallback:
        return fallback_letter_regex(text)
    return None


def extract_mcq_letter(text: str) -> str | None:
    """Resilient MCQ option extractor combining strict matching and fallback regex."""
    return strict_letter(text, allow_fallback=True)


def completion_metrics(prediction, reference):
    clean_prediction = "".join(prediction.split())
    clean_reference = "".join(reference.split())
    raw_pred_ids = re.findall(r"\b[^\W\d]\w*\b", prediction)
    raw_ref_ids = re.findall(r"\b[^\W\d]\w*\b", reference)
    predicted_ids = {s for s in raw_pred_ids if s.isidentifier() and not keyword.iskeyword(s)}
    reference_ids = {s for s in raw_ref_ids if s.isidentifier() and not keyword.iskeyword(s)}
    overlap = len(predicted_ids & reference_ids)
    if predicted_ids or reference_ids:
        f1 = 2 * overlap / (len(predicted_ids) + len(reference_ids))
    else:
        f1 = 1.0 if clean_prediction == clean_reference else 0.0
    seq_sim = SequenceMatcher(None, clean_prediction, clean_reference, autojunk=False).ratio()
    return {
        "exact_match": clean_prediction == clean_reference,
        "sequence_similarity": seq_sim,
        "regex_identifier_set_f1": f1,
        "regex_f1": f1,
    }


def score_repo_niah_case(
    row: dict[str, Any], gold: dict[str, Any], completion: str
) -> dict[str, Any]:
    """Score a RepoNIAH needle retrieval completion."""
    expected = str(
        gold.get("needle") or gold.get("answer") or gold.get("gold_needle") or ""
    ).strip()
    cleaned = completion.strip().strip("'\"`")
    correct = False
    if expected:
        if expected == cleaned:
            correct = True
        else:
            pattern = re.compile(re.escape(expected), re.IGNORECASE)
            matches = list(pattern.finditer(completion))
            negation_pattern = re.compile(
                r"\b(?:not|never|no|without|isn['’]t|wasn['’]t|ain['’]t|aren['’]t|don['’]t|didn['’]t|doesn['’]t|can['’]t|cannot|couldn['’]t|won['’]t|hasn['’]t|haven['’]t|definitely\s+not)\b(?:\s+\w+){0,3}\s*$",
                re.IGNORECASE,
            )
            for m in matches:
                start = m.start()
                preceding = completion[max(0, start - 40) : start]
                if not negation_pattern.search(preceding):
                    correct = True
                    break
    return {
        "correct": correct,
        "parsed_answer": cleaned,
        "expected_needle": expected,
        "depth_ratio": row.get("depth_ratio"),
        "target_context_tokens": row.get("target_context_tokens"),
        "status": "SCORED",
    }


def score_context_case(
    row, gold, completion, *, directory, python_executable, allow_execution, include_syntax=False
):
    kind = row.get("task_type")
    if kind == "mcq":
        parsed = extract_mcq_letter(completion)
        return {
            "correct": parsed == gold["answer"],
            "parsed_answer": parsed,
            "status": "INVALID_OUTPUT" if parsed is None else "SCORED",
        }
    if kind == "completion":
        metrics = completion_metrics(completion, gold["reference"])
        return {
            **metrics,
            "correct": metrics["exact_match"],
            "primary_continuous_score": metrics["sequence_similarity"],
            "status": "SCORED",
        }
    if str(row.get("benchmark", "")).lower() == "gsm8k" or "question" in row and "answer" in gold:
        from bifocalgemma.evaluation.gsm8k import extract_release_answer

        parsed = extract_release_answer(completion)
        return {
            "correct": parsed == gold["answer"],
            "parsed_answer": parsed,
            "status": "INVALID_OUTPUT" if parsed is None else "SCORED",
        }
    if kind == "retrieval" or str(row.get("benchmark", "")).lower() == "reponiah":
        return score_repo_niah_case(row, gold, completion)
    if "snapshot_file" in gold:
        from dataclasses import asdict

        from bifocalgemma.evaluation.native_python import score_python_patch
        from bifocalgemma.repository_study import snapshot_from_json

        path = (Path(directory) / gold["snapshot_file"]).resolve()
        if not path.is_relative_to(Path(directory).resolve()):
            raise ValueError("benchmark_snapshot_path")
        snapshot = snapshot_from_json(path)
        result = score_python_patch(
            row["instance_id"],
            snapshot,
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
        record: dict[str, Any] = {**asdict(result), "correct": result.functional_pass}
        if include_syntax:
            from bifocalgemma.evaluation.syntax import patch_syntax

            record.update(patch_syntax(snapshot, completion, applied=result.patch_applied))
        return record
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
    use_cache=False,
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
    protocol_version = V2_EVALUATION_PROTOCOL if use_cache else EVALUATION_PROTOCOL
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
            "protocol": protocol_version,
            "generation_budget_policy": "actual_paired_prompt_v1",
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
                except ValueError:
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
    from bifocalgemma.training.timing import DurableClock

    clock = DurableClock(output / "timing.json", started=started)
    segment_initial_count = len(completed)
    heartbeat_at = 0.0
    heartbeat_stage = None

    def activity(event, *, force=False):
        nonlocal heartbeat_at, heartbeat_stage
        now = time.monotonic()
        if not force and event.get("stage") == heartbeat_stage and now - heartbeat_at < 5:
            return
        heartbeat_at = now
        heartbeat_stage = event.get("stage")
        payload = {"updated_at": time.time(), "benchmark": root.name, **event}
        atomic_json(output / "activity.json", payload)
        clock.persist(force=force)
        if progress:
            progress(payload)

    for index, row in enumerate(public[len(completed) :], len(completed) + 1):
        case_started = time.monotonic()
        case_info = {
            "instance_id": row["instance_id"],
            "items_done": index - 1,
            "items_total": len(ids),
            "active_case": index,
        }
        activity(
            {**case_info, "stage": "benchmark_case_started", "generated_tokens": 0}, force=True
        )
        cap = min(encoder_limit, row.get("max_encoder_tokens", encoder_limit))
        enc, prefix = prompt_ids(row, spec, encoder, decoder, encoder_limit=cap)
        # Same output allowance for pairs and baselines: reserve the paired IT prefix.
        paired_prompt = decoder_prompt(row, paired=True)
        paired_rendered = decoder.apply_chat_template(
            [{"role": "user", "content": paired_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        paired_prefix_size = len(decoder.encode(paired_rendered, add_special_tokens=False))
        row_budget = int(row.get("max_new_tokens", max_new_tokens))
        if (
            str(row.get("task_type", "")).lower() in {"mcq", "qa", "retrieval"}
            and max_new_tokens >= 256
        ):
            row_budget = max(row_budget, 256)
        new_tokens = min(row_budget, decoder_limit - paired_prefix_size)
        if new_tokens < 1:
            raise ValueError("benchmark_decoder_prefix_exceeds_budget")
        total = decoder_limit if spec.encoder else cap + decoder_limit + 512
        if len(prefix) + new_tokens > total:
            raise ValueError("benchmark_decoder_prefix_exceeds_budget")
        generated = greedy_decode(
            model,
            enc,
            prefix,
            max_new_tokens=new_tokens,
            max_total_decoder_tokens=total,
            progress=lambda event: activity(
                {**case_info, **event, "case_elapsed_seconds": time.monotonic() - case_started}
            ),
            **({"use_cache": True} if use_cache else {}),
        )
        tokens = generated.token_ids
        if tokens and tokens[-1] in (1, 106, 50):
            tokens = tokens[:-1]
        completion = decode_text(decoder, tokens, skip_special_tokens=False)
        activity(
            {**case_info, "stage": "benchmark_scoring", "generated_tokens": len(tokens)}, force=True
        )
        scoring = score_context_case(
            row,
            gold[row["instance_id"]],
            completion,
            directory=root,
            python_executable=python_executable or sys.executable,
            allow_execution=allow_execution,
            **({"include_syntax": True} if use_cache else {}),
        )
        record = {
            "instance_id": row["instance_id"],
            "completion": completion,
            "stop_reason": generated.stop_reason,
            "encoder_tokens": len(enc),
            "decoder_prefix_tokens": len(prefix),
            "generated_tokens": len(tokens),
            "case_seconds": time.monotonic() - case_started,
            **scoring,
        }
        with predictions.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        completed.append(record)
        activity(
            {
                **case_info,
                "stage": "benchmark_prediction_saved",
                "items_done": index,
                "generated_tokens": len(tokens),
            },
            force=True,
        )
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
                    "benchmark_elapsed_seconds": clock.elapsed(),
                    "unobserved_gap_seconds": clock.gap,
                    "benchmark_eta_seconds": (time.perf_counter() - started)
                    / max(1, index - segment_initial_count)
                    * max(0, len(ids) - index),
                    "benchmark_correct": sum(r["correct"] is True for r in completed),
                    "last_prediction": record,
                    **resource_snapshot(next(model.parameters()).device),
                }
            )
    scored = all(r["correct"] is not None for r in completed)
    clock.close()
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
        "timing": clock.summary(),
        "official_protocol": False,
        "protocol_version": protocol_version,
        "protocol": ("greedy_cached" if use_cache else "greedy_no_temporal_cache")
        + "; fixed eligible cohort; no context truncation",
        "resources": resource_snapshot(next(model.parameters()).device),
    }
    for key in (
        "exact_match",
        "sequence_similarity",
        "regex_identifier_set_f1",
        "regex_f1",
        "primary_continuous_score",
    ):
        if completed and key in completed[0]:
            report[key] = sum(float(r[key]) for r in completed) / len(ids)
    if completed and "patch_applied" in completed[0]:
        report["patch_apply_rate"] = sum(bool(r["patch_applied"]) for r in completed) / len(ids)
        checked = [r for r in completed if r.get("syntax_valid") is not None]
        report["syntax_checked"] = len(checked)
        report["syntax_success_rate"] = (
            sum(bool(r["syntax_valid"]) for r in checked) / len(checked) if checked else None
        )
    atomic_json(output / "benchmark_report.json", report)
    activity(
        {"stage": "benchmark_completed", "items_done": len(ids), "items_total": len(ids)},
        force=True,
    )
    return report


def format_crosscodelong_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate and return CrossCodeLong continuous metrics matching PROJECT.md interface contract."""
    n = max(1, len(records))
    seq_sim = sum(float(r.get("sequence_similarity", 0.0)) for r in records) / n
    f1 = sum(float(r.get("regex_f1", r.get("regex_identifier_set_f1", 0.0))) for r in records) / n
    em = sum(float(bool(r.get("exact_match", False))) for r in records) / n
    return {
        "sequence_similarity": round(seq_sim, 4),
        "regex_f1": round(f1, 4),
        "exact_match": round(em, 4),
    }


# =====================================================================
# RepoNIAH (Repository Needle-In-A-Haystack) Distractor Harness
# =====================================================================

REPONIAH_DEFAULT_DEPTHS = (4096, 8192, 16384, 32768)
REPONIAH_DEFAULT_DEPTH_RATIOS = (0.1, 0.25, 0.5, 0.75, 0.9)


def make_distractor_python_chunk(index: int) -> str:
    """Generate realistic, syntax-valid Python distractor code."""
    return f"""class SubsystemController_{index}:
    \"\"\"Internal subsystem handler for repository pipeline stage {index}.\"\"\"
    def __init__(self, buffer_size: int = 1024, debug_flag: bool = False):
        self.buffer_size = buffer_size
        self.debug_flag = debug_flag
        self.registry_{index} = {{}}

    def process_telemetry_{index}(self, payload: dict) -> bool:
        if not self.debug_flag:
            return True
        key = f"metric_{index}"
        self.registry_{index}[key] = payload.get("val", 0) * {index + 1}
        return len(self.registry_{index}) > 0

    def compute_offsets_{index}(self, base: int, step: int) -> list[int]:
        return [base + i * step for i in range({(index % 5) + 3})]
"""


def generate_repo_niah_haystack(
    target_tokens: int,
    needle_line: str,
    depth_ratio: float,
    tokenizer=None,
) -> str:
    """Construct a synthetic repository haystack of target_tokens with needle embedded at depth_ratio."""
    blocks: list[tuple[str, int]] = []
    current_tokens = 0
    idx = 0
    while current_tokens < target_tokens:
        block = make_distractor_python_chunk(idx)
        tok_count = (
            len(tokenizer.encode(block, add_special_tokens=False))
            if tokenizer
            else max(1, len(block) // 4)
        )
        blocks.append((block, tok_count))
        current_tokens += tok_count
        idx += 1

    total_tok = sum(tc for _, tc in blocks)
    target_needle_pos = int(depth_ratio * total_tok)

    accum = 0
    insert_idx = len(blocks) // 2
    for i, (_, tc) in enumerate(blocks):
        if accum + tc >= target_needle_pos:
            insert_idx = i
            break
        accum += tc

    needle_block = f"\n# --- System Security Configuration ---\n{needle_line}\n# ---------------------------------------\n"
    text_blocks = (
        [b[0] for b in blocks[:insert_idx]] + [needle_block] + [b[0] for b in blocks[insert_idx:]]
    )
    return "\n".join(text_blocks)


def build_repo_niah_case(
    context_tokens: int,
    depth_ratio: float,
    seed: int = 42,
    *,
    key_name: str = "SECRET_KEY",
    needle_value: str | None = None,
    tokenizer=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a single RepoNIAH benchmark case (public input row, hidden gold spec)."""
    if needle_value is None:
        needle_value = f"alpha_bravo_{seed}_{hashlib.sha256(str(seed).encode()).hexdigest()[:8]}"
    needle_line = f'{key_name} = "{needle_value}"'
    instruction = f"What is the value of {key_name} in the codebase? Return only the value."
    haystack = generate_repo_niah_haystack(
        context_tokens, needle_line, depth_ratio, tokenizer=tokenizer
    )
    instance_id = f"reponiah_{context_tokens}_{int(depth_ratio * 100)}_{seed}"
    public = {
        "instance_id": instance_id,
        "role": "eval",
        "task_type": "retrieval",
        "benchmark": "RepoNIAH",
        "instruction": instruction,
        "context": haystack,
        "target_context_tokens": context_tokens,
        "depth_ratio": depth_ratio,
        "max_encoder_tokens": max(32768, context_tokens),
        "max_new_tokens": 128,
    }
    gold = {
        "instance_id": instance_id,
        "needle": needle_value,
        "key_name": key_name,
        "answer": needle_value,
        "task_type": "retrieval",
    }
    return public, gold


def run_repo_niah_eval(
    model,
    spec,
    output_directory: Path | str,
    tokenizer_cache: Path | str,
    *,
    depths: tuple[int, ...] = REPONIAH_DEFAULT_DEPTHS,
    depth_ratios: tuple[float, ...] = REPONIAH_DEFAULT_DEPTH_RATIOS,
    cases_per_depth: int = 5,
    seed: int = 42,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Execute the full RepoNIAH distractor harness evaluating needle retrieval across context depths."""
    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer

    out_path = Path(output_directory)
    out_path.mkdir(parents=True, exist_ok=True)
    decoder, _ = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))[0] if spec.encoder else None
    )

    cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for d in depths:
        for r in depth_ratios:
            for s in range(cases_per_depth):
                case_seed = seed + s * 100 + int(r * 1000)
                pub, g = build_repo_niah_case(d, r, seed=case_seed, tokenizer=encoder or decoder)
                cases.append((pub, g))

    predictions: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, list[bool]]] = {f"{d}": {} for d in depths}
    for pub, g in cases:
        enc = tuple(encoder.encode(pub["context"], add_special_tokens=True)) if encoder else ()
        prompt = decoder_prompt(pub, paired=encoder is not None)
        text = decoder.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = tuple(decoder.encode(text, add_special_tokens=False))
        max_total = pub["target_context_tokens"] + 1024 if not encoder else 2048
        gen = greedy_decode(
            model,
            enc,
            prefix,
            max_new_tokens=pub["max_new_tokens"],
            max_total_decoder_tokens=max_total,
            **({"use_cache": True} if use_cache else {}),
        )
        tokens = gen.token_ids
        if tokens and tokens[-1] in (1, 106, 50):
            tokens = tokens[:-1]
        completion = decode_text(decoder, tokens, skip_special_tokens=False)
        scoring = score_repo_niah_case(pub, g, completion)
        record = {
            "instance_id": pub["instance_id"],
            "depth_ratio": pub["depth_ratio"],
            "target_context_tokens": pub["target_context_tokens"],
            "completion": completion,
            **scoring,
        }
        predictions.append(record)
        d_key = f"{pub['target_context_tokens']}"
        r_key = f"{pub['depth_ratio']}"
        matrix[d_key].setdefault(r_key, []).append(scoring["correct"])

    acc_by_depth: dict[str, float] = {}
    for d_str, r_map in matrix.items():
        all_d_correct = [v for vals in r_map.values() for v in vals]
        acc_by_depth[d_str] = sum(all_d_correct) / len(all_d_correct) if all_d_correct else 0.0

    overall = sum(r["correct"] for r in predictions) / len(predictions) if predictions else 0.0
    summary = {
        "status": "COMPLETED",
        "benchmark": "RepoNIAH",
        "total_cases": len(predictions),
        "overall_accuracy": overall,
        "accuracy_by_depth": acc_by_depth,
        "depth_matrix": {
            d: {r: sum(vals) / len(vals) if vals else 0.0 for r, vals in r_map.items()}
            for d, r_map in matrix.items()
        },
    }
    atomic_json(out_path / "reponiah_summary.json", summary)
    with (out_path / "reponiah_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summary


evaluate_repo_niah = run_repo_niah_eval
