"""Canonical, versioned generation and scoring contract for context benchmarks."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import keyword
import textwrap
from pathlib import Path
from typing import Any

EFFECTIVE_BENCHMARK_PROTOCOL = "context_effective_v1_strict_mcq_bounded_generation"
GENERATION_BUDGET_POLICY = "min_suite_row_actual_paired_prefix_v1"
MCQ_SCORER_MODE = "strict_primary_resilient_diagnostic_v1"
MAX_DECODER_TOKENS = 8192


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"benchmark_invalid_{name}")
    return value


def effective_generation_budget(
    suite_cap: int,
    *,
    row_cap: int | None = None,
    paired_prefix_tokens: int,
    decoder_limit: int = MAX_DECODER_TOKENS,
) -> int:
    """Reserve the actual paired prompt equally for baseline and paired models.

    Row metadata may restrict the suite allowance, but can never enlarge it.
    A smaller caller-supplied decoder window remains a hard limit as well.
    """
    suite_cap = _positive_integer(suite_cap, "suite_cap")
    decoder_limit = min(_positive_integer(decoder_limit, "decoder_limit"), MAX_DECODER_TOKENS)
    if (
        isinstance(paired_prefix_tokens, bool)
        or not isinstance(paired_prefix_tokens, int)
        or paired_prefix_tokens < 0
    ):
        raise ValueError("benchmark_invalid_paired_prefix_tokens")
    if row_cap is not None:
        row_cap = _positive_integer(row_cap, "row_cap")
    remaining = decoder_limit - paired_prefix_tokens
    if remaining < 1:
        raise ValueError("benchmark_decoder_prefix_exceeds_budget")
    return min(suite_cap, row_cap if row_cap is not None else suite_cap, remaining)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scorer_fingerprint() -> str:
    """Hash parser/scorer semantics, including regex globals, rather than a label alone."""
    from bifocalgemma.evaluation import context, gsm8k

    functions = (
        context.strict_letter,
        context.fallback_letter_regex,
        context._extract_match_letter,
        context.extract_mcq_letter,
        context.completion_metrics,
        context.score_context_case,
        context.score_repo_niah_case,
        gsm8k.extract_release_answer,
    )
    return _fingerprint(
        {
            "mode": MCQ_SCORER_MODE,
            "functions": [
                ast.dump(
                    ast.parse(textwrap.dedent(inspect.getsource(function))),
                    include_attributes=False,
                )
                for function in functions
            ],
            "regex": {
                name: getattr(context, name)
                for name in (
                    "DELIM_OPEN",
                    "DELIM_CLOSE",
                    "LETTER",
                    "STRICT_DELIM_OPEN",
                    "STRICT_DELIM_CLOSE",
                    "LETTER_TOKEN",
                    "LETTER_TOKEN_NC",
                    "OPT_ADVERBS",
                    "CONJ_GUARD",
                )
            },
            "python_keywords": keyword.kwlist,
            "evaluation_source": {
                path.name: ast.dump(
                    ast.parse(path.read_text(encoding="utf-8")), include_attributes=False
                )
                for path in sorted(Path(context.__file__).parent.glob("*.py"))
            },
        }
    )


def _runtime_model_identity(model: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    runtime_spec = getattr(model, "spec", None)
    spec_to_dict = getattr(runtime_spec, "to_dict", None)
    if callable(spec_to_dict):
        identity["spec"] = spec_to_dict()
    for name in ("encoder", "decoder"):
        config = getattr(getattr(model, name, None), "config", None)
        config_to_dict = getattr(config, "to_dict", None)
        if callable(config_to_dict):
            identity[name + "_config"] = config_to_dict()
    return identity


def build_effective_benchmark_spec(
    *,
    spec: Any,
    model: Any,
    public_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    collection: str,
    encoder_limit: int,
    decoder_limit: int,
    suite_cap: int,
    use_cache: bool,
    allow_execution: bool,
    sample_limit: int | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    """Resolve the executed contract used by signatures, journals and reports."""
    _positive_integer(encoder_limit, "encoder_limit")
    effective_generation_budget(suite_cap, paired_prefix_tokens=0, decoder_limit=decoder_limit)
    for row in public_rows:
        if "max_new_tokens" in row:
            _positive_integer(row["max_new_tokens"], "row_cap")
    runtime_identity = _runtime_model_identity(model)
    if "spec" in runtime_identity and runtime_identity["spec"] != spec.to_dict():
        raise ValueError("benchmark_runtime_model_spec_mismatch")
    return {
        "version": EFFECTIVE_BENCHMARK_PROTOCOL,
        "model_spec": spec.to_dict(),
        "runtime_model": runtime_identity,
        "collection": collection,
        "sample_limit": sample_limit,
        "sample_seed": sample_seed,
        "sample_ids": [row["instance_id"] for row in public_rows],
        "inputs_fingerprint": _fingerprint(public_rows),
        "gold_fingerprint": _fingerprint(gold_rows),
        "generation": {
            "policy": GENERATION_BUDGET_POLICY,
            "suite_max_new_tokens": suite_cap,
            "encoder_limit": encoder_limit,
            "decoder_limit": min(decoder_limit, MAX_DECODER_TOKENS),
            "prefix": "actual_paired_chat_template_enable_thinking_false",
            "decoding": "greedy_cached" if use_cache else "greedy_no_temporal_cache",
        },
        "scoring": {
            "mcq_mode": MCQ_SCORER_MODE,
            "fingerprint": scorer_fingerprint(),
            "execution": allow_execution,
        },
    }


def mcq_report_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose both parsers on the same predictions with an unambiguous headline."""
    mcq = [record for record in records if "strict_correct" in record]
    if not mcq:
        return {}
    strict = sum(bool(record["strict_correct"]) for record in mcq) / len(mcq)
    resilient = sum(bool(record["resilient_correct"]) for record in mcq) / len(mcq)
    return {
        "primary_metric": "mcq_strict_accuracy",
        "accuracy_alias": "strict_accuracy",
        "strict_accuracy": strict,
        "resilient_accuracy": resilient,
        "resilient_metric_role": "diagnostic_only_same_predictions",
        "strict_invalid_outputs": sum(record["strict_parsed_answer"] is None for record in mcq),
        "resilient_invalid_outputs": sum(
            record["resilient_parsed_answer"] is None for record in mcq
        ),
        "mcq_scored_count": len(mcq),
    }
