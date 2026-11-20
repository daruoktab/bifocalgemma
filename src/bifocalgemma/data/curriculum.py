"""Anti-catastrophic-forgetting comprehension replay curriculum."""

from __future__ import annotations

import random
from typing import Any

from bifocalgemma.config import COMPREHENSION_MIX_RATIO
from bifocalgemma.contracts import TokenizedRepairExample

# ---------------------------------------------------------------------------
# Anti-Catastrophic-Forgetting Dataset Curriculum (10-15% Comprehension / QA Mix)
# ---------------------------------------------------------------------------

COMPREHENSION_TASK_TYPES = frozenset(
    {
        "localization",
        "completion",
        "qa",
        "long_context_qa",
        "mcq",
        "span_copy",
        "code_qa_mcq",
        "comprehension",
        "source_extraction",
        "source_mcq",
        "source_reconstruction",
    }
)

V4_DEFAULT_MIX: dict[str, float] = {
    "repair_loc": 0.35,
    "fim_completion": 0.25,
    "long_context_qa": 0.25,
    "retention": 0.15,
}

_V4_BUCKET_BY_TASK: dict[str, str] = {
    "repair": "repair_loc",
    "localization": "repair_loc",
    "completion": "fim_completion",
    "fim": "fim_completion",
    "qa": "long_context_qa",
    "long_context_qa": "long_context_qa",
    "mcq": "long_context_qa",
    "comprehension": "long_context_qa",
    "span_copy": "long_context_qa",
    "code_qa_mcq": "long_context_qa",
    "source_extraction": "long_context_qa",
    "source_mcq": "long_context_qa",
    "source_reconstruction": "long_context_qa",
    "retention": "retention",
    "instruction": "retention",
}

COMPREHENSION_INSTRUCTION_PROMPTS: dict[str, str] = {
    "span_copy": "Extract and copy the exact target identifier or code span from the context.",
    "code_qa_mcq": "Identify the correct symbol or answer based on the context.",
}

MAX_BASELINE_COMPREHENSION_TOKENS = 4096

# Pre-computed canonical Gemma tokenizer prefix IDs for offline/unit-test fallback when decoder is not provided:
# - span_copy: '<bos><|turn>user\nExtract and copy the exact target identifier or code span from the context.<turn|>\n<|turn>model\n'
# - code_qa_mcq: '<bos><|turn>user\nIdentify the correct symbol or answer based on the context.<turn|>\n<|turn>model\n'
CANONICAL_COMPREHENSION_PREFIX_IDS: dict[str, tuple[int, ...]] = {
    "span_copy": (
        2,
        105,
        2364,
        107,
        82138,
        532,
        4865,
        506,
        4453,
        3328,
        31744,
        653,
        3393,
        15184,
        699,
        506,
        4403,
        236761,
        106,
        107,
        105,
        4368,
        107,
    ),
    "code_qa_mcq": (
        2,
        105,
        2364,
        107,
        137938,
        506,
        4338,
        5404,
        653,
        3890,
        2721,
        580,
        506,
        4403,
        236761,
        106,
        107,
        105,
        4368,
        107,
    ),
}


def encode_comprehension_prefix(
    kind: str,
    decoder: Any = None,
) -> tuple[int, ...]:
    """Tokenize chat template prompt for comprehension task kind into decoder prefix token IDs."""
    if decoder is not None and hasattr(decoder, "apply_chat_template"):
        prompt = COMPREHENSION_INSTRUCTION_PROMPTS.get(
            kind,
            "Extract and copy the exact target identifier or code span from the context.",
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = decoder.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = decoder.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = tuple(decoder.encode(rendered, add_special_tokens=False))
        if encoded:
            return encoded
    return CANONICAL_COMPREHENSION_PREFIX_IDS.get(
        kind,
        CANONICAL_COMPREHENSION_PREFIX_IDS["span_copy"],
    )


def is_comprehension_example(example: Any) -> bool:
    """Return True if a TokenizedRepairExample is a non-diff comprehension/QA/span-copy/completion task."""
    meta = getattr(example, "metadata", None) or {}
    task_type = str(meta.get("task_type", "repair")).lower()
    return task_type in COMPREHENSION_TASK_TYPES


def synthesize_comprehension_examples(
    examples: list[TokenizedRepairExample],
    count: int,
    *,
    seed: int = 17,
    decoder: Any = None,
) -> list[TokenizedRepairExample]:
    """Synthesize deterministic exact span-copy and concise Code QA / MCQ examples from context tokens.

    Prevents catastrophic forgetting on `longcodeqa`, `sweqa`, and `gsm8k` while forcing the
    cross-attention bridge to copy exact token spans from the encoder context.
    """
    if count <= 0 or not examples:
        return []
    rng = random.Random(f"bifocalgemma_v3_comprehension_synth_{seed}")
    synthesized: list[TokenizedRepairExample] = []
    n_pool = len(examples)

    # Pre-tokenize comprehension prefix IDs once for both task kinds
    prefix_by_kind = {
        kind: encode_comprehension_prefix(kind, decoder=decoder)
        for kind in ("span_copy", "code_qa_mcq")
    }

    for idx in range(count):
        base = examples[(idx * 7 + rng.randint(0, max(0, n_pool - 1))) % n_pool]
        eos_token = base.target_ids[-1] if base.target_ids else 107
        kind = "span_copy" if idx % 2 == 0 else "code_qa_mcq"
        max_base_token = max(
            (max(base.decoder_prefix_ids) if base.decoder_prefix_ids else 0),
            (max(base.encoder_ids) if base.encoder_ids else 0),
            (max(base.target_ids) if base.target_ids else 0),
        )
        if decoder is None and max_base_token < 256:
            prefix = tuple(tok if tok < 128 else (3 + (tok % 100)) for tok in prefix_by_kind[kind])
        else:
            prefix = prefix_by_kind[kind]

        if base.encoder_ids:
            source_tokens = base.encoder_ids
        else:
            # A decoder-only baseline sees only the prefix we keep. Sample the
            # copy target from that same window: truncating after sampling can
            # ask the model to copy text that is no longer in its input.
            visible_budget = max(0, MAX_BASELINE_COMPREHENSION_TOKENS - len(prefix))
            source_tokens = base.decoder_prefix_ids[:visible_budget]

        interior = source_tokens[1:] if len(source_tokens) > 1 else source_tokens
        if not interior:
            continue

        if kind == "span_copy":
            # Exact Span-Copy task: copy a contiguous 4-24 token span from context
            span_len = min(len(interior), max(1, min(16, max(2, len(interior) // 2))))
            max_start = max(0, len(interior) - span_len)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            span = tuple(interior[start : start + span_len])
            target = (*span, eos_token) if span[-1] != eos_token else span
        else:
            # Concise Code QA / MCQ identifier answer task: short 1-4 token concise answer + EOS
            ans_len = min(len(interior), max(1, min(4, len(interior))))
            max_start = max(0, len(interior) - ans_len)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            ans = tuple(interior[start : start + ans_len])
            target = (*ans, eos_token) if ans[-1] != eos_token else ans

        meta = dict(base.metadata or {})
        meta["task_type"] = kind
        meta["anti_forgetting_replay"] = True
        meta["instruction"] = COMPREHENSION_INSTRUCTION_PROMPTS[kind]

        effective_prefix = prefix if base.encoder_ids else (*prefix, *source_tokens)

        synthesized.append(
            TokenizedRepairExample(
                instance_id=f"{base.instance_id}__{kind}_{idx}",
                encoder_ids=base.encoder_ids,
                decoder_prefix_ids=effective_prefix,
                target_ids=target,
                metadata=meta,
            )
        )
    return synthesized


def build_anti_forgetting_curriculum(
    train_examples: list[TokenizedRepairExample],
    *,
    stage_name: str = "frozen_decoder",
    comprehension_ratio: float = COMPREHENSION_MIX_RATIO,
    seed: int = 17,
    decoder: Any = None,
) -> tuple[list[TokenizedRepairExample], dict[str, Any]]:
    """Construct a balanced Stage 1 / Stage 2 curriculum with 10-15% Code QA / MCQ & span-copy replay.

    Prioritizes authentic completion and localization records from the dataset pool first,
    supplementing with synthesized span_copy and code_qa_mcq examples up to the target ratio.
    """
    total = len(train_examples)
    if total == 0:
        return [], {
            "total_examples": 0,
            "comprehension_examples": 0,
            "comprehension_ratio": 0.0,
            "synthesized_examples": 0,
            "stage_name": stage_name,
        }

    ratio = max(0.10, min(0.25, float(comprehension_ratio)))
    target_comp_count = max(1, round(total * ratio)) if total >= 4 else 0

    existing_comp = [ex for ex in train_examples if is_comprehension_example(ex)]
    repair_pool = [ex for ex in train_examples if not is_comprehension_example(ex)]

    rng = random.Random(f"curriculum_{stage_name}_{seed}")

    # Prioritize authentic completion/localization records; synthesize deficit to reach target_comp_count
    if len(existing_comp) >= target_comp_count:
        comp_pool = list(existing_comp)
        rng.shuffle(comp_pool)
        selected_comp = comp_pool[:target_comp_count]
        synthesized: list[TokenizedRepairExample] = []
    else:
        synth_needed = target_comp_count - len(existing_comp)
        synthesized = synthesize_comprehension_examples(
            train_examples,
            synth_needed,
            seed=seed + (101 if stage_name == "joint" else 0),
            decoder=decoder,
        )
        selected_comp = list(existing_comp) + synthesized
        rng.shuffle(selected_comp)

    rng.shuffle(repair_pool)
    remaining_slots = max(0, total - len(selected_comp))
    base_pool = repair_pool if repair_pool else list(train_examples)
    selected_repair = [base_pool[i % len(base_pool)] for i in range(remaining_slots)]

    # Interleave comprehension examples uniformly across the epoch; in Stage 1 also front-load warmup
    curriculum: list[TokenizedRepairExample] = []
    if stage_name in ("frozen_decoder", "stage1", "stage1_bridge_warmup") and len(selected_comp) >= 4:
        front_count = max(1, len(selected_comp) // 3)
        curriculum.extend(selected_comp[:front_count])
        rest_comp = selected_comp[front_count:]
    else:
        rest_comp = list(selected_comp)

    comp_idx = 0
    rep_idx = 0
    remaining_total = total - len(curriculum)
    stride = (
        max(2, round(remaining_total / max(1, len(rest_comp))))
        if rest_comp
        else remaining_total + 1
    )

    for pos in range(remaining_total):
        if comp_idx < len(rest_comp) and (
            (pos + 1) % stride == 0 or rep_idx >= len(selected_repair)
        ):
            curriculum.append(rest_comp[comp_idx])
            comp_idx += 1
        elif rep_idx < len(selected_repair):
            curriculum.append(selected_repair[rep_idx])
            rep_idx += 1
        elif comp_idx < len(rest_comp):
            curriculum.append(rest_comp[comp_idx])
            comp_idx += 1

    curriculum = curriculum[:total]
    actual_comp = sum(1 for ex in curriculum if is_comprehension_example(ex))
    stats = {
        "stage_name": stage_name,
        "total_examples": len(curriculum),
        "comprehension_examples": actual_comp,
        "comprehension_ratio": round(actual_comp / max(1, len(curriculum)), 4),
        "synthesized_examples": len(synthesized),
    }
    return curriculum, stats


def classify_v4_bucket(item: Any) -> str:
    """Classify a raw record dict or TokenizedRepairExample into one of the four v4 buckets."""
    if isinstance(item, dict):
        task_type = str(item.get("task_type") or "repair").lower()
        if item.get("v4_bucket") in V4_DEFAULT_MIX:
            return str(item["v4_bucket"])
    else:
        meta = getattr(item, "metadata", None) or {}
        task_type = str(meta.get("task_type") or "repair").lower()
    return _V4_BUCKET_BY_TASK.get(task_type, "repair_loc")


def _encoder_token_count(item: Any) -> int:
    if isinstance(item, dict):
        lengths = item.get("token_lengths") or {}
        if isinstance(lengths, dict) and isinstance(lengths.get("encoder"), int):
            return lengths["encoder"]
        ctx = str(item.get("context") or "")
        return max(1, len(ctx) // 4)
    enc = getattr(item, "encoder_ids", ())
    return len(enc)


def build_v4_curriculum(
    records: list[dict[str, Any]],
    *,
    target_count: int | None = None,
    target_mix: dict[str, float] | None = None,
    min_long_context_ratio: float = 0.40,
    long_context_threshold: int = 8192,
    seed: int = 17,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select and balance a v4 pool targeting 35/25/25/15 mix and >=40% 8k-32k rows."""
    if not records:
        return [], {
            "total_examples": 0,
            "bucket_counts": {},
            "bucket_ratios": {},
            "long_context_count": 0,
            "long_context_ratio": 0.0,
        }
    mix = dict(target_mix or V4_DEFAULT_MIX)
    total = len(records) if target_count is None else max(1, min(len(records), target_count))
    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in V4_DEFAULT_MIX}
    for row in records:
        by_bucket.setdefault(classify_v4_bucket(row), []).append(row)

    rng = random.Random(f"bifocalgemma_v4_curriculum_{seed}")
    for bucket, items in by_bucket.items():
        items.sort(key=lambda r: str(r.get("instance_id", "")))
        rng.shuffle(items)
        # Partition long-context (>=8k) first within each bucket so we can hit >=40% long rows
        long_items = [r for r in items if _encoder_token_count(r) >= long_context_threshold]
        short_items = [r for r in items if _encoder_token_count(r) < long_context_threshold]
        by_bucket[bucket] = long_items + short_items

    present_buckets = [b for b, items in by_bucket.items() if items]
    active_weights = {b: max(0.0, float(mix.get(b, 0.0))) for b in present_buckets}
    weight_sum = sum(active_weights.values()) or float(len(present_buckets))
    quotas: dict[str, int] = {
        b: int(total * (active_weights[b] / weight_sum)) for b in present_buckets
    }
    remainder = total - sum(quotas.values())
    for b in sorted(present_buckets, key=lambda k: -(active_weights[k] / weight_sum)):
        if remainder <= 0:
            break
        quotas[b] += 1
        remainder -= 1

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    long_target = round(total * min_long_context_ratio)
    long_selected = 0

    for bucket in present_buckets:
        items = by_bucket[bucket]
        quota = min(len(items), quotas[bucket])
        long_pool = [r for r in items if _encoder_token_count(r) >= long_context_threshold]
        short_pool = [r for r in items if _encoder_token_count(r) < long_context_threshold]
        want_long = min(len(long_pool), max(1 if long_pool and quota > 0 else 0, round(quota * min_long_context_ratio)))
        if long_selected < long_target:
            want_long = min(len(long_pool), max(want_long, quota))
        chosen = long_pool[:want_long]
        if len(chosen) < quota:
            chosen.extend(short_pool[: quota - len(chosen)])
        if len(chosen) < quota:
            chosen.extend(long_pool[want_long : want_long + (quota - len(chosen))])
        for row in chosen:
            rid = str(row.get("instance_id", id(row)))
            if rid not in used_ids:
                used_ids.add(rid)
                selected.append(row)
                if _encoder_token_count(row) >= long_context_threshold:
                    long_selected += 1

    # Fill any remaining shortfall up to `total` prioritizing long-context items
    if len(selected) < total:
        leftovers = [
            row
            for row in records
            if str(row.get("instance_id", id(row))) not in used_ids
        ]
        leftovers.sort(
            key=lambda r: (
                -(_encoder_token_count(r) >= long_context_threshold),
                str(r.get("instance_id", "")),
            )
        )
        selected.extend(leftovers[: total - len(selected)])

    bucket_counts: dict[str, int] = {}
    for row in selected:
        b = classify_v4_bucket(row)
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    actual_long = sum(1 for r in selected if _encoder_token_count(r) >= long_context_threshold)
    stats = {
        "total_examples": len(selected),
        "bucket_counts": bucket_counts,
        "bucket_ratios": {
            b: round(c / max(1, len(selected)), 4) for b, c in sorted(bucket_counts.items())
        },
        "long_context_count": actual_long,
        "long_context_ratio": round(actual_long / max(1, len(selected)), 4),
    }
    return selected, stats
