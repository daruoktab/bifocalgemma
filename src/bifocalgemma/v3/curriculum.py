"""Anti-catastrophic-forgetting comprehension replay curriculum."""

from __future__ import annotations

import random
from typing import Any

from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.v3.constants import (
    COMPREHENSION_MIX_RATIO,
)

# ---------------------------------------------------------------------------
# Anti-Catastrophic-Forgetting Dataset Curriculum (10-15% Comprehension / QA Mix)
# ---------------------------------------------------------------------------

COMPREHENSION_TASK_TYPES = frozenset(
    {"localization", "completion", "qa", "mcq", "span_copy", "code_qa_mcq", "comprehension"}
)

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
