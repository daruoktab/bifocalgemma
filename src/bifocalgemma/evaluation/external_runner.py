"""Thin AutoModelForCausalLM + AutoTokenizer evaluation runner for external baselines.

Runs the exact BifocalGemma benchmark cohorts, prompt renderers, and scorers without
Gemma-specific tokenizer/EOS assumptions (handles ``bos_token_id=None``, resolves
native EOS + ``<|im_end|>``/``<turn|>`` tokens, and passes ``enable_thinking=False``
when supported).
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from bifocalgemma.config import (
    BENCHMARK_SAMPLE_SEED,
    EXTERNAL_COMPARISON_MODELS,
    V4_BENCHMARKS,
)
from bifocalgemma.data.curation import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.evaluation.benchmark import prepare_sampled_benchmark_collection
from bifocalgemma.evaluation.context import (
    REPONIAH_DEFAULT_DEPTH_RATIOS,
    REPONIAH_DEFAULT_DEPTHS,
    _resolve_repair_mode,
    build_repo_niah_case,
    decoder_prompt,
    format_crosscodelong_summary,
    score_context_case,
    score_repo_niah_case,
)
from bifocalgemma.evaluation.protocol import (
    BENCHMARK_RESET_ID,
    MAX_DECODER_TOKENS,
    gsm8k_report_metrics,
    mcq_report_metrics,
    repair_report_metrics,
    scorer_fingerprint,
)
from bifocalgemma.training.stages import atomic_json

EXTERNAL_CONTEXT_MODES = ("full", "local_only")
_CANDIDATE_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<turn|>", "<eos>")


def resolve_external_model_spec(model_key_or_id: str) -> dict[str, Any]:
    """Resolve a pinned external comparison model config by short key or HF model_id."""
    if model_key_or_id in EXTERNAL_COMPARISON_MODELS:
        return {"key": model_key_or_id, **EXTERNAL_COMPARISON_MODELS[model_key_or_id]}
    needle = model_key_or_id.strip().lower()
    for key, cfg in EXTERNAL_COMPARISON_MODELS.items():
        if key.lower() == needle or str(cfg["model_id"]).lower() == needle:
            return {"key": key, **cfg}
    raise ValueError(
        f"Unknown external model {model_key_or_id!r}. "
        f"Choose from {list(EXTERNAL_COMPARISON_MODELS)}."
    )


def resolve_external_benchmarks(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Resolve benchmark configs from ``V4_BENCHMARKS`` by collection name or alias."""
    if not names or names == ["all"]:
        return [dict(item) for item in V4_BENCHMARKS]
    by_name = {item["collection"]: item for item in V4_BENCHMARKS}
    resolved: list[dict[str, Any]] = []
    for raw in names:
        key = raw.strip()
        if key in by_name:
            resolved.append(dict(by_name[key]))
            continue
        if key.lower() == "reponiah" and "reponiah_gemma32k_dec8k_v1" in by_name:
            resolved.append(dict(by_name["reponiah_gemma32k_dec8k_v1"]))
            continue
        matches = [item for col, item in by_name.items() if col.startswith(key)]
        if len(matches) == 1:
            resolved.append(dict(matches[0]))
            continue
        raise ValueError(f"Unknown benchmark {raw!r}. Available: {list(by_name)}.")
    return resolved


def _collect_int_ids(value: Any, target: list[int]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int) and value >= 0:
        if value not in target:
            target.append(value)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_int_ids(item, target)


def resolve_external_eos_ids(tokenizer: Any, model: Any = None) -> tuple[int, ...]:
    """Resolve native EOS token IDs without Gemma-hardcoded IDs like ``(1, 106, 50)``."""
    eos_ids: list[int] = []
    _collect_int_ids(getattr(tokenizer, "eos_token_id", None), eos_ids)
    if model is not None:
        _collect_int_ids(
            getattr(getattr(model, "generation_config", None), "eos_token_id", None), eos_ids
        )
        _collect_int_ids(getattr(getattr(model, "config", None), "eos_token_id", None), eos_ids)

    unk_id = getattr(tokenizer, "unk_token_id", None)
    special_tokens = set(getattr(tokenizer, "all_special_tokens", ()) or ())
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else {}
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)

    if callable(convert):
        for token in _CANDIDATE_STOP_TOKENS:
            if special_tokens and token not in special_tokens and token not in vocab:
                continue
            if not special_tokens and vocab and token not in vocab:
                continue
            tok_id = convert(token)
            if isinstance(tok_id, int) and tok_id >= 0 and tok_id != unk_id:
                _collect_int_ids(tok_id, eos_ids)

    if not eos_ids:
        raise ValueError("external_runner_missing_eos_token_id")
    return tuple(eos_ids)


def format_external_prompt(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    context_mode: str = "full",
    repair_mode: str | None = None,
    enable_thinking: bool | None = False,
    number_lines: bool = True,
) -> tuple[str, tuple[int, ...]]:
    """Render and tokenize an external decoder-only prompt without BOS assertions."""
    if context_mode not in EXTERNAL_CONTEXT_MODES:
        raise ValueError(
            f"External decoder-only models only support {EXTERNAL_CONTEXT_MODES}, got {context_mode!r}."
        )
    prompt = decoder_prompt(
        row,
        paired=False,
        context_mode=context_mode,
        repair_mode=repair_mode,
        number_lines=number_lines,
    )
    messages = [{"role": "user", "content": prompt}]
    has_template = getattr(tokenizer, "chat_template", None) is not None
    rendered = prompt
    if has_template or callable(getattr(tokenizer, "apply_chat_template", None)):
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        except ValueError:
            rendered = prompt

    encoded = tuple(int(tok) for tok in tokenizer.encode(rendered, add_special_tokens=False))
    if not encoded:
        raise ValueError("external_runner_empty_prompt_ids")
    return rendered, encoded


def extract_generated_tokens(
    token_ids: list[int] | tuple[int, ...],
    eos_ids: tuple[int, ...],
) -> tuple[tuple[int, ...], str]:
    """Slice generated IDs up to the first resolved EOS token and report stop reason."""
    eos_set = set(eos_ids)
    kept: list[int] = []
    for tok in token_ids:
        if tok in eos_set:
            return tuple(kept), "eos"
        kept.append(tok)
    return tuple(kept), "length_limit"


def _resolve_pad_token_id(tokenizer: Any, eos_ids: tuple[int, ...]) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_id, int) and pad_id >= 0:
        return pad_id
    return eos_ids[0]


def generate_external_batch(
    model: Any,
    tokenizer: Any,
    batch_prefix_ids: list[tuple[int, ...]],
    *,
    max_new_tokens_list: list[int],
    eos_ids: tuple[int, ...],
) -> list[tuple[tuple[int, ...], str, str]]:
    """Run greedy left-padded ``model.generate`` for a batch of tokenized prompts."""
    if not batch_prefix_ids:
        return []
    pad_id = _resolve_pad_token_id(tokenizer, eos_ids)
    bsz = len(batch_prefix_ids)
    max_pref_len = max(len(p) for p in batch_prefix_ids)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    input_ids = torch.full((bsz, max_pref_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_pref_len), dtype=torch.long, device=device)
    for b, pref in enumerate(batch_prefix_ids):
        offset = max_pref_len - len(pref)
        input_ids[b, offset:] = torch.tensor(pref, dtype=torch.long, device=device)
        attention_mask[b, offset:] = 1

    max_new = max(max_new_tokens_list)
    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new,
        "do_sample": False,
        "eos_token_id": list(eos_ids) if len(eos_ids) > 1 else eos_ids[0],
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    previous_mode = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.inference_mode():
            sequences = model.generate(**gen_kwargs)
    finally:
        if hasattr(model, "train"):
            model.train(previous_mode)

    results: list[tuple[tuple[int, ...], str, str]] = []
    for b in range(bsz):
        raw_gen = [int(t) for t in sequences[b, max_pref_len:].tolist()]
        budget = max_new_tokens_list[b]
        if len(raw_gen) > budget:
            raw_gen = raw_gen[:budget]
        kept_ids, stop_reason = extract_generated_tokens(raw_gen, eos_ids)
        completion = tokenizer.decode(list(kept_ids), skip_special_tokens=True)
        if not isinstance(completion, str):
            raise ValueError("external_runner_decode_must_return_text")
        results.append((kept_ids, stop_reason, completion))
    return results


def load_hf_external_model_and_tokenizer(
    model_cfg: dict[str, Any],
    *,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Load HF ``AutoTokenizer`` and ``AutoModelForCausalLM`` for an external comparison model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(model_cfg["model_id"])
    revision = str(model_cfg["revision"])
    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if getattr(tokenizer, "padding_side", None) != "left":
        setattr(tokenizer, "padding_side", "left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    return model, tokenizer


def _build_reponiah_cohort(
    benchmark_item: dict[str, Any],
    tokenizer: Any = None,
    *,
    max_position_embeddings: int | None = None,
    enable_thinking: bool | None = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    depths = tuple(benchmark_item.get("depths", REPONIAH_DEFAULT_DEPTHS))
    depth_ratios = tuple(benchmark_item.get("depth_ratios", REPONIAH_DEFAULT_DEPTH_RATIOS))
    cases_per_depth = int(benchmark_item.get("cases_per_depth", 5))
    seed = int(benchmark_item.get("seed", 42))
    suite_cap = int(benchmark_item.get("max_new_tokens", 128))
    max_prompt_tokens = (
        max(1, max_position_embeddings - suite_cap) if max_position_embeddings is not None else None
    )
    public: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for d in depths:
        for r in depth_ratios:
            for s in range(cases_per_depth):
                case_seed = seed + s * 100 + int(r * 1000)
                pub, g = build_repo_niah_case(
                    d,
                    r,
                    seed=case_seed,
                    tokenizer=tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    enable_thinking=enable_thinking,
                )
                public.append(pub)
                gold.append(g)
    return public, gold


def _load_benchmark_cohort(
    benchmark_item: dict[str, Any],
    benchmarks_root: Path,
    *,
    tokenizer: Any = None,
    max_samples: int | None = None,
    max_position_embeddings: int | None = None,
    enable_thinking: bool | None = False,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    collection = str(benchmark_item["collection"])
    if "reponiah" in collection.lower():
        public, gold = _build_reponiah_cohort(
            benchmark_item,
            tokenizer=tokenizer,
            max_position_embeddings=max_position_embeddings,
            enable_thinking=enable_thinking,
        )
        if max_samples is not None and max_samples > 0:
            public = public[:max_samples]
            gold = gold[:max_samples]
        return benchmarks_root / collection, public, gold

    source_name = str(benchmark_item.get("source_collection") or collection)
    raw_dir = benchmarks_root / source_name
    sample_limit = benchmark_item.get("sample_limit")
    if max_samples is not None and max_samples > 0:
        sample_limit = min(max_samples, sample_limit) if sample_limit else max_samples
    sample_seed = int(benchmark_item.get("sample_seed", BENCHMARK_SAMPLE_SEED))
    eff_dir = prepare_sampled_benchmark_collection(
        raw_dir, sample_limit=sample_limit, sample_seed=sample_seed
    )
    inputs_path = (
        eff_dir / "eval_inputs.jsonl"
        if (eff_dir / "eval_inputs.jsonl").exists()
        else eff_dir / "inputs.jsonl"
    )
    gold_path = (
        eff_dir / "eval_private/specs.jsonl"
        if (eff_dir / "eval_private/specs.jsonl").exists()
        else eff_dir / "hidden_gold.jsonl"
    )
    public = list(read_source_rows(inputs_path))
    hidden = list(read_source_rows(gold_path))
    if max_samples is not None and max_samples > 0 and len(public) > max_samples:
        public = public[:max_samples]
        hidden = hidden[:max_samples]
    return eff_dir, public, hidden


def run_external_benchmark(
    model_key_or_id: str,
    benchmark_item: dict[str, Any],
    benchmarks_root: Path | str,
    output_dir: Path | str,
    *,
    context_mode: str = "full",
    repair_mode: str | None = None,
    batch_size: int = 1,
    max_samples: int | None = None,
    allow_execution: bool = True,
    python_executable: str | None = None,
    dry_run: bool = False,
    local_files_only: bool = False,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Evaluate one external model on one benchmark cohort using the exact internal scorers."""
    import sys

    if context_mode not in EXTERNAL_CONTEXT_MODES:
        raise ValueError(
            f"External decoder-only models only support {EXTERNAL_CONTEXT_MODES}, got {context_mode!r}."
        )
    model_cfg = resolve_external_model_spec(model_key_or_id)
    collection = str(benchmark_item["collection"])
    eff_repair_mode = repair_mode or benchmark_item.get("repair_mode")
    root_path = Path(benchmarks_root)
    out_path = Path(output_dir)

    if dry_run:
        cohort_exists = (
            "reponiah" in collection.lower()
            or (root_path / str(benchmark_item.get("source_collection") or collection)).exists()
        )
        sample_count = None
        if cohort_exists:
            _, pub, _ = _load_benchmark_cohort(
                benchmark_item, root_path, tokenizer=tokenizer, max_samples=max_samples
            )
            sample_count = len(pub)
        return {
            "status": "DRY_RUN",
            "dry_run": True,
            "model_key": model_cfg["key"],
            "model_id": model_cfg["model_id"],
            "revision": model_cfg["revision"],
            "collection": collection,
            "context_mode": context_mode,
            "repair_mode": eff_repair_mode,
            "max_position_embeddings": model_cfg["max_position_embeddings"],
            "cohort_available": cohort_exists,
            "planned_cases": sample_count,
            "batch_size": max(1, batch_size),
        }

    if model is None or tokenizer is None:
        loaded_model, loaded_tok = load_hf_external_model_and_tokenizer(
            model_cfg, local_files_only=local_files_only
        )
        model = model or loaded_model
        tokenizer = tokenizer or loaded_tok

    eos_ids = resolve_external_eos_ids(tokenizer, model)
    suite_cap = int(benchmark_item.get("max_new_tokens", 2048))
    max_pos = int(model_cfg.get("max_position_embeddings", 32768))
    decoder_limit = min(int(benchmark_item.get("decoder_limit", MAX_DECODER_TOKENS)), max_pos)
    enable_thinking = model_cfg.get("enable_thinking", False)

    eff_dir, public, hidden = _load_benchmark_cohort(
        benchmark_item,
        root_path,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_position_embeddings=max_pos,
        enable_thinking=enable_thinking,
    )
    ids = [r["instance_id"] for r in public]
    if not ids or len(set(ids)) != len(ids) or ids != [r["instance_id"] for r in hidden]:
        raise ValueError("benchmark_public_gold_cohort_mismatch")
    gold_by_id = {r["instance_id"]: r for r in hidden}

    out_path.mkdir(parents=True, exist_ok=True)
    signature = stable_fingerprint(
        {
            "model_key": model_cfg["key"],
            "model_id": model_cfg["model_id"],
            "revision": model_cfg["revision"],
            "collection": collection,
            "context_mode": context_mode,
            "repair_mode": eff_repair_mode,
            "allow_execution": allow_execution,
            "max_position_embeddings": max_pos,
            "suite_cap": suite_cap,
            "decoder_limit": decoder_limit,
            "enable_thinking": enable_thinking,
            "line_number_overflow_fallback": "unnumbered_on_overflow",
            "scorer_fingerprint": scorer_fingerprint(),
            "ids": ids,
        }
    )
    state_path = out_path / "evaluation_state.json"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("External benchmark/model/protocol changed; choose a new output.")
    else:
        if (out_path / "predictions.jsonl").exists():
            raise ValueError("Prediction journal has no protocol identity; choose a new output.")
        atomic_json(state_path, {"signature": signature, "total": len(ids)})

    predictions_path = out_path / "predictions.jsonl"
    completed: list[dict[str, Any]] = []
    if predictions_path.exists():
        with predictions_path.open("rb+") as stream:
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
    if any(record.get("protocol_signature") != signature for record in completed):
        raise ValueError("Prediction journal protocol signature mismatch.")

    remaining_public = public[len(completed) :]
    started = time.perf_counter()

    step_batch = max(1, batch_size)
    with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(remaining_public), step_batch):
            chunk = remaining_public[start : start + step_batch]
            prompts_ids: list[tuple[int, ...]] = []
            budgets: list[int] = []
            used_number_lines_list: list[bool] = []
            for row in chunk:
                used_number_lines = True
                _, pref = format_external_prompt(
                    row,
                    tokenizer,
                    context_mode=context_mode,
                    repair_mode=eff_repair_mode,
                    enable_thinking=enable_thinking,
                    number_lines=True,
                )
                row_cap = row.get("max_new_tokens")
                budget = min(suite_cap, row_cap) if row_cap else suite_cap
                if len(pref) + budget > max_pos:
                    _, pref_unnumbered = format_external_prompt(
                        row,
                        tokenizer,
                        context_mode=context_mode,
                        repair_mode=eff_repair_mode,
                        enable_thinking=enable_thinking,
                        number_lines=False,
                    )
                    pref = pref_unnumbered
                    used_number_lines = False
                if len(pref) + budget > max_pos:
                    remaining = max_pos - len(pref)
                    if remaining < 1:
                        raise ValueError("external_prompt_exceeds_max_position_embeddings")
                    budget = remaining
                prompts_ids.append(pref)
                budgets.append(budget)
                used_number_lines_list.append(used_number_lines)

            chunk_started = time.monotonic()
            outputs = generate_external_batch(
                model,
                tokenizer,
                prompts_ids,
                max_new_tokens_list=budgets,
                eos_ids=eos_ids,
            )
            per_case_seconds = (time.monotonic() - chunk_started) / max(1, len(chunk))
            for row, pref, budget, used_number_lines, (gen_ids, stop_reason, completion) in zip(
                chunk, prompts_ids, budgets, used_number_lines_list, outputs, strict=True
            ):
                row_repair_mode = _resolve_repair_mode(
                    {**row, "collection": collection}, eff_repair_mode
                )
                if "reponiah" in collection.lower() or row.get("task_type") == "retrieval":
                    scoring = score_repo_niah_case(row, gold_by_id[row["instance_id"]], completion)
                else:
                    scoring = score_context_case(
                        row,
                        gold_by_id[row["instance_id"]],
                        completion,
                        directory=eff_dir,
                        python_executable=python_executable or sys.executable,
                        allow_execution=allow_execution,
                        repair_mode=row_repair_mode,
                    )
                record = {
                    "protocol_signature": signature,
                    "instance_id": row["instance_id"],
                    "model_key": model_cfg["key"],
                    "model_id": model_cfg["model_id"],
                    "revision": model_cfg["revision"],
                    "context_mode": context_mode,
                    "completion": completion,
                    "stop_reason": stop_reason,
                    "decoder_prefix_tokens": len(pref),
                    "generation_max_new_tokens": budget,
                    "generated_tokens": len(gen_ids),
                    "number_lines": used_number_lines,
                    "overflow_unnumbered_fallback": not used_number_lines,
                    "eos_ids_used": list(eos_ids),
                    "case_seconds": per_case_seconds,
                    **scoring,
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                completed.append(record)
            stream.flush()
            os.fsync(stream.fileno())

    scored = all(r["correct"] is not None for r in completed)
    report: dict[str, Any] = {
        "status": "COMPLETED",
        "signature": signature,
        "model_key": model_cfg["key"],
        "model_id": model_cfg["model_id"],
        "revision": model_cfg["revision"],
        "collection": collection,
        "context_mode": context_mode,
        "repair_mode": eff_repair_mode,
        "reset_id": BENCHMARK_RESET_ID,
        "scorer_fingerprint": scorer_fingerprint(),
        "eos_ids": list(eos_ids),
        "total": len(ids),
        "correct": sum(bool(r["correct"]) for r in completed) if scored else None,
        "accuracy": sum(bool(r["correct"]) for r in completed) / len(ids) if scored else None,
        "score_available": scored,
        "statuses": dict(Counter(r["status"] for r in completed)),
        "generation_length_limit": sum(r["stop_reason"] == "length_limit" for r in completed),
        "unnumbered_fallback_count": sum(
            bool(r.get("overflow_unnumbered_fallback")) for r in completed
        ),
        "segment_seconds": time.perf_counter() - started,
        "decoder_limit": decoder_limit,
    }
    report.update(mcq_report_metrics(completed))
    report.update(gsm8k_report_metrics(completed))
    report.update(repair_report_metrics(completed))
    if completed and "exact_match" in completed[0]:
        report.update(format_crosscodelong_summary(completed))
    atomic_json(out_path / "benchmark_report.json", report)
    return report
