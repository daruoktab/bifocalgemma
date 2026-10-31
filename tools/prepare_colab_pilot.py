"""Build a standalone Colab G4 pilot script for E2B->E2B encoder adaptation experiments."""

import argparse
import ast
import copy
import difflib
import hashlib
import inspect
import json
import keyword
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from notebook_bundle import source_bundle_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = (
    "direct_joint",
    "frozen_then_joint",
    "span_ul2_then_joint",
    "it_encoder_frozen_then_joint",
    "full_cross_bf16_frozen_then_joint",
)
PYTHON_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "Exception",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "open",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "ValueError",
    "zip",
    "self",
    "cls",
}


def stratified_raw_subset(
    rows: list[dict[str, Any]], count: int, seed: int = 17
) -> list[dict[str, Any]]:
    """Select the exact same stratified instance order as v2_diagnostics.stratified_subset."""
    if not rows or count <= 0:
        return []
    strata: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
        assert isinstance(meta, dict)
        key = (str(meta.get("source", "unknown")), str(meta.get("task_type", "repair")))
        rank = hashlib.sha256(f"{seed}:{row['instance_id']}".encode("utf-8")).hexdigest()
        strata[key].append((rank, index))
    buckets = [sorted(values) for _, values in sorted(strata.items())]
    chosen: list[int] = []
    for level in range(max(len(bucket) for bucket in buckets)):
        for bucket in buckets:
            if level < len(bucket):
                chosen.append(bucket[level][1])
            if len(chosen) >= min(count, len(rows)):
                return [rows[i] for i in chosen]
    return [rows[i] for i in chosen]


def record_context_text(row: dict[str, Any]) -> str:
    context = row.get("context")
    if isinstance(context, str) and context.strip():
        return context
    return f"### Instruction\n{row['instruction']}\n\n### File: {row['path']}\n{row['before']}"


def extract_resolvable_span(
    context: str, instance_id: str, *, seed: int = 17, min_lines: int = 3, max_lines: int = 8
) -> tuple[str, str, int, int] | None:
    """Pick a deterministic contiguous code span and replace it once with # <MISSING_CODE>."""
    lines = context.splitlines()
    if len(lines) < min_lines + 2:
        return None

    def eligible_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith(("### ", "diff --git", "--- ", "+++ ", "@@ ")):
            return False
        return True

    candidates: list[tuple[int, int]] = []
    rng = random.Random(f"{seed}:span:{instance_id}")
    span_len = rng.randint(min_lines, max_lines)
    for length in (span_len, min_lines, 2):
        for start in range(1, max(1, len(lines) - length)):
            stop = start + length
            window = lines[start:stop]
            if not all(eligible_line(line) for line in window):
                continue
            target_span = "\n".join(window)
            if len(target_span.strip()) < 16 or "# <MISSING_CODE>" in target_span:
                continue
            prefix = "\n".join(lines[:start])
            suffix = "\n".join(lines[stop:])
            if target_span in prefix or target_span in suffix:
                continue
            candidates.append((start, stop))
        if candidates:
            break
    if not candidates:
        return None
    start, stop = candidates[rng.randrange(len(candidates))]
    target_span = "\n".join(lines[start:stop])
    indent_match = re.match(r"^(\s*)", lines[start])
    indent = indent_match.group(1) if indent_match else ""
    sentinel = f"{indent}# <MISSING_CODE>"
    corrupted_lines = [*lines[:start], sentinel, *lines[stop:]]
    corrupted_context = "\n".join(corrupted_lines)
    if context.endswith("\n"):
        corrupted_context += "\n"
    if corrupted_context.count("# <MISSING_CODE>") != 1 or target_span in corrupted_context:
        return None
    return corrupted_context, target_span, start, stop


def encode_chat_pair(
    spec: Any,
    encoder_tokenizer: Any,
    decoder_tokenizer: Any,
    *,
    instance_id: str,
    context_text: str,
    user_prompt: str,
    target_text: str,
    encoder_limit: int,
    decoder_limit: int,
    metadata: dict[str, Any],
) -> Any:
    from bifocalgemma.contracts import TokenizedRepairExample

    enc_ids = (
        tuple(encoder_tokenizer.encode(context_text, add_special_tokens=True))
        if spec.encoder is not None and context_text
        else ()
    )
    messages = [{"role": "user", "content": user_prompt}]
    prefix_text = decoder_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = decoder_tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": target_text}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_ids = tuple(decoder_tokenizer.encode(prefix_text, add_special_tokens=False))
    full_ids = tuple(decoder_tokenizer.encode(full_text.rstrip(), add_special_tokens=False))
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("prefix_target_boundary")
    target_ids = full_ids[len(prefix_ids) :]
    if not prefix_ids or not target_ids:
        raise ValueError("empty_tokenized_sequence")
    if len(enc_ids) > encoder_limit:
        enc_ids = enc_ids[:encoder_limit]
    if len(full_ids) - 1 > decoder_limit:
        allowed_target = max(2, decoder_limit + 1 - len(prefix_ids))
        target_ids = (*target_ids[: allowed_target - 1], target_ids[-1])
    return TokenizedRepairExample(
        instance_id=instance_id,
        encoder_ids=enc_ids,
        decoder_prefix_ids=prefix_ids,
        target_ids=target_ids,
        metadata=metadata,
    )


def build_span_denoising_examples(
    rows: list[dict[str, Any]],
    spec: Any,
    encoder_tokenizer: Any,
    decoder_tokenizer: Any,
    *,
    fallback_examples: list[Any] | None = None,
    encoder_limit: int = 32768,
    decoder_limit: int = 8192,
    seed: int = 17,
) -> tuple[list[Any], dict[str, int]]:
    """Create same-dataset UL2-style span recovery examples without external corpora."""
    from bifocalgemma.training.repository_sft import tokenize_repository_record

    examples: list[Any] = []
    counts = {"native_completion": 0, "constructed_span_mask": 0, "fallback_original": 0}
    for index, row in enumerate(rows):
        context = record_context_text(row)
        if row.get("task_type") == "completion" and context.count("# <MISSING_CODE>") == 1:
            target_text = str(row.get("target_text") or "")
            prompt = (
                "Return only the missing Python code, preserving its indentation.\n"
                "Use the supplied encoder context.\n"
                + str(row.get("instruction") or "Restore the code marked by # <MISSING_CODE>.")
            )
            examples.append(
                encode_chat_pair(
                    spec,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    instance_id=row["instance_id"],
                    context_text=context,
                    user_prompt=prompt,
                    target_text=target_text,
                    encoder_limit=encoder_limit,
                    decoder_limit=decoder_limit,
                    metadata={
                        "source": row.get("source", "unknown"),
                        "task_type": "span_denoising",
                        "origin_task_type": "completion",
                    },
                )
            )
            counts["native_completion"] += 1
            continue
        extracted = extract_resolvable_span(context, str(row["instance_id"]), seed=seed)
        if extracted is not None:
            corrupted_context, target_span, start, stop = extracted
            prompt = (
                "Return only the missing Python code, preserving its indentation.\n"
                "Use the supplied encoder context.\n"
                f"Restore the exact Python code span at # <MISSING_CODE> ({stop - start} lines)."
            )
            examples.append(
                encode_chat_pair(
                    spec,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    instance_id=row["instance_id"],
                    context_text=corrupted_context,
                    user_prompt=prompt,
                    target_text=target_span,
                    encoder_limit=encoder_limit,
                    decoder_limit=decoder_limit,
                    metadata={
                        "source": row.get("source", "unknown"),
                        "task_type": "span_denoising",
                        "origin_task_type": row.get("task_type", "repair"),
                    },
                )
            )
            counts["constructed_span_mask"] += 1
        elif fallback_examples is not None and index < len(fallback_examples):
            examples.append(fallback_examples[index])
            counts["fallback_original"] += 1
        else:
            examples.append(
                tokenize_repository_record(
                    {**row, "context": context},
                    spec,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    encoder_limit=encoder_limit,
                    decoder_limit=decoder_limit,
                )
            )
            counts["fallback_original"] += 1
    return examples, counts


def rename_identifier_counterfactual(
    context: str,
    target_span: str,
    first_line: str,
    last_line: str,
    instance_id: str,
    *,
    seed: int = 17,
) -> dict[str, str] | None:
    """Rename an interior identifier so only the encoder context reveals the new name."""
    span_lines = target_span.splitlines()
    if len(span_lines) < 3:
        return None
    interior = "\n".join(span_lines[1:-1])
    boundary = first_line + "\n" + last_line
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", interior)
    candidates: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if (
            token in seen
            or keyword.iskeyword(token)
            or token in PYTHON_BUILTINS
            or token.startswith("__")
            or re.search(rf"\b{re.escape(token)}\b", boundary)
        ):
            continue
        seen.add(token)
        candidates.append(token)
    if not candidates:
        return None
    digest = hashlib.sha256(f"{seed}:rename:{instance_id}".encode("utf-8")).hexdigest()
    chosen = candidates[int(digest[:8], 16) % len(candidates)]
    replacement = f"bf_id_{digest[:6]}"
    if replacement in context or replacement in target_span:
        return None
    pattern = re.compile(rf"\b{re.escape(chosen)}\b")
    renamed_context = pattern.sub(replacement, context)
    renamed_target = pattern.sub(replacement, target_span)
    if renamed_target == target_span or replacement in boundary:
        return None
    return {
        "original_identifier": chosen,
        "renamed_identifier": replacement,
        "renamed_context": renamed_context,
        "renamed_target": renamed_target,
    }


def build_source_copy_probes(
    rows: list[dict[str, Any]],
    spec: Any,
    encoder_tokenizer: Any,
    decoder_tokenizer: Any,
    *,
    count: int = 16,
    encoder_limit: int = 32768,
    decoder_limit: int = 8192,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Build held-out probes for span recovery, verbatim copying, and counterfactual renaming."""
    probes: list[dict[str, Any]] = []
    for row in rows:
        if len(probes) >= count:
            break
        context = record_context_text(row)
        extracted = extract_resolvable_span(
            context, str(row["instance_id"]), seed=seed, min_lines=4, max_lines=8
        )
        if extracted is None:
            continue
        corrupted_context, target_span, start, stop = extracted
        span_lines = target_span.splitlines()
        first_line, last_line = span_lines[0], span_lines[-1]
        renamed = rename_identifier_counterfactual(
            context, target_span, first_line, last_line, str(row["instance_id"]), seed=seed
        )
        if renamed is None:
            continue
        copy_prompt = (
            "Copy the exact Python code block from the supplied encoder context starting with the line:\n"
            f"{first_line}\n"
            "and ending with the line:\n"
            f"{last_line}\n"
            "Return only the verbatim Python code block, preserving its indentation."
        )
        span_prompt = (
            "Return only the missing Python code, preserving its indentation.\n"
            "Use the supplied encoder context.\n"
            f"Restore the exact Python code span at # <MISSING_CODE> ({stop - start} lines)."
        )
        copy_example = encode_chat_pair(
            spec,
            encoder_tokenizer,
            decoder_tokenizer,
            instance_id=f"{row['instance_id']}::copy",
            context_text=context,
            user_prompt=copy_prompt,
            target_text=target_span,
            encoder_limit=encoder_limit,
            decoder_limit=decoder_limit,
            metadata={"probe_type": "verifiable_copy", "source": row.get("source", "unknown")},
        )
        renamed_example = encode_chat_pair(
            spec,
            encoder_tokenizer,
            decoder_tokenizer,
            instance_id=f"{row['instance_id']}::renamed",
            context_text=renamed["renamed_context"],
            user_prompt=copy_prompt,
            target_text=renamed["renamed_target"],
            encoder_limit=encoder_limit,
            decoder_limit=decoder_limit,
            metadata={
                "probe_type": "renamed_identifier_copy",
                "source": row.get("source", "unknown"),
            },
        )
        span_example = encode_chat_pair(
            spec,
            encoder_tokenizer,
            decoder_tokenizer,
            instance_id=f"{row['instance_id']}::span",
            context_text=corrupted_context,
            user_prompt=span_prompt,
            target_text=target_span,
            encoder_limit=encoder_limit,
            decoder_limit=decoder_limit,
            metadata={"probe_type": "span_recovery", "source": row.get("source", "unknown")},
        )
        probes.append(
            {
                "instance_id": row["instance_id"],
                "source": row.get("source", "unknown"),
                "first_line": first_line,
                "last_line": last_line,
                "target_span": target_span,
                "renamed_target": renamed["renamed_target"],
                "original_identifier": renamed["original_identifier"],
                "renamed_identifier": renamed["renamed_identifier"],
                "copy_example": copy_example,
                "renamed_example": renamed_example,
                "span_example": span_example,
            }
        )
    return probes


def extract_single_window_solution_repair(
    row: dict[str, Any], *, include_headers: bool = True
) -> tuple[str, str] | None:
    """Extract a single-file Python window and line-aligned solution.py unified diff from a training repair row."""
    if row.get("task_type", "repair") != "repair":
        return None
    context = record_context_text(row)
    patch = str(row.get("target_text") or row.get("target_patch") or "").strip()
    if not patch.startswith("diff --git ") or patch.count("diff --git ") != 1:
        return None
    header_match = re.match(r"^diff --git a/(\S+) b/(\S+)\n", patch)
    if not header_match or header_match.group(1) != header_match.group(2):
        return None
    orig_path = header_match.group(1)
    hunks = list(
        re.finditer(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$",
            patch,
            flags=re.M,
        )
    )
    if not hunks:
        return None
    first_old_line = int(hunks[0].group(1))
    last_hunk = hunks[-1]
    last_old_len = int(last_hunk.group(2)) if last_hunk.group(2) is not None else 1
    last_old_line = int(last_hunk.group(1)) + max(1, last_old_len) - 1

    win_pattern = re.compile(r"^### File: (\S+?)(?: \(lines (\d+)-(\d+)\))?\n", flags=re.M)
    matches = list(win_pattern.finditer(context))
    if not matches:
        return None
    for idx, m in enumerate(matches):
        w_path = m.group(1)
        if w_path != orig_path:
            continue
        w_start = int(m.group(2)) if m.group(2) else 1
        body_start = m.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
        w_body = context[body_start:body_end].strip("\n")
        if not w_body.strip():
            continue
        w_lines = w_body.splitlines()
        w_end = int(m.group(3)) if m.group(3) else (w_start + len(w_lines) - 1)
        if w_start <= first_old_line and last_old_line <= w_end:
            offset = w_start - 1

            def _shift(hm: re.Match[str], _off: int = offset) -> str:
                os_line = max(1, int(hm.group(1)) - _off)
                oc_part = f",{hm.group(2)}" if hm.group(2) is not None else ""
                ns_line = max(1, int(hm.group(3)) - _off)
                nc_part = f",{hm.group(4)}" if hm.group(4) is not None else ""
                tail = hm.group(5) or ""
                return f"@@ -{os_line}{oc_part} +{ns_line}{nc_part} @@{tail}"

            rewritten = re.sub(
                r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$",
                _shift,
                patch,
                flags=re.M,
            )
            rewritten = rewritten.replace(f"a/{orig_path}", "a/solution.py").replace(
                f"b/{orig_path}", "b/solution.py"
            )
            if not rewritten.endswith("\n"):
                rewritten += "\n"
            if include_headers:
                rendered_ctx = (
                    "### Issue\n"
                    "Fix the buggy implementation to satisfy its function specification.\n\n"
                    f"### File: solution.py (lines 1-{len(w_lines)})\n"
                    f"{w_body}\n"
                )
            else:
                rendered_ctx = f"{w_body}\n"
            return rendered_ctx, rewritten
    return None


def build_full_cross_cohorts(
    rows: list[dict[str, Any]],
    spec: Any,
    encoder_tokenizer: Any,
    decoder_tokenizer: Any,
    *,
    sft_examples: list[Any],
    ul2_examples: list[Any],
    encoder_limit: int = 32768,
    decoder_limit: int = 8192,
    seed: int = 17,
) -> tuple[list[Any], list[Any], dict[str, int]]:
    """Build Stage 1 (pure encoder grounding) and Stage 2 (50/50 grounded + SFT) cohorts from train_raw."""
    stage1: list[Any] = []
    counts = {
        "verifiable_copy": 0,
        "renamed_identifier_copy": 0,
        "span_denoising": 0,
        "grounded_solution_repair": 0,
        "fallback_sft": 0,
    }
    for index, row in enumerate(rows):
        context = record_context_text(row)
        slot = index % 4
        built = None
        if slot in (0, 1):
            extracted = extract_resolvable_span(
                context, str(row["instance_id"]), seed=seed + index, min_lines=4, max_lines=8
            )
            if extracted is not None:
                _, target_span, _, _ = extracted
                span_lines = target_span.splitlines()
                first_line, last_line = span_lines[0], span_lines[-1]
                copy_prompt = (
                    "Copy the exact Python code block from the supplied encoder context starting with the line:\n"
                    f"{first_line}\n"
                    "and ending with the line:\n"
                    f"{last_line}\n"
                    "Return only the verbatim Python code block, preserving its indentation."
                )
                if slot == 1:
                    renamed = rename_identifier_counterfactual(
                        context,
                        target_span,
                        first_line,
                        last_line,
                        str(row["instance_id"]),
                        seed=seed + index,
                    )
                    if renamed is not None:
                        try:
                            built = encode_chat_pair(
                                spec,
                                encoder_tokenizer,
                                decoder_tokenizer,
                                instance_id=f"{row['instance_id']}::train_renamed",
                                context_text=renamed["renamed_context"],
                                user_prompt=copy_prompt,
                                target_text=renamed["renamed_target"],
                                encoder_limit=encoder_limit,
                                decoder_limit=decoder_limit,
                                metadata={
                                    "source": row.get("source", "unknown"),
                                    "task_type": "renamed_identifier_copy",
                                },
                            )
                            counts["renamed_identifier_copy"] += 1
                        except ValueError:
                            built = None
                if built is None:
                    try:
                        built = encode_chat_pair(
                            spec,
                            encoder_tokenizer,
                            decoder_tokenizer,
                            instance_id=f"{row['instance_id']}::train_copy",
                            context_text=context,
                            user_prompt=copy_prompt,
                            target_text=target_span,
                            encoder_limit=encoder_limit,
                            decoder_limit=decoder_limit,
                            metadata={
                                "source": row.get("source", "unknown"),
                                "task_type": "verifiable_copy",
                            },
                        )
                        counts["verifiable_copy"] += 1
                    except ValueError:
                        built = None
        elif slot == 3:
            sol = extract_single_window_solution_repair(row, include_headers=(index % 8 == 3))
            if sol is not None:
                sol_ctx, sol_patch = sol
                sol_prompt = (
                    "Return only a unified diff fulfilling the instruction.\n"
                    "Fix the buggy implementation to satisfy its function specification.\n"
                    "Use the supplied encoder context."
                )
                try:
                    built = encode_chat_pair(
                        spec,
                        encoder_tokenizer,
                        decoder_tokenizer,
                        instance_id=f"{row['instance_id']}::solution_repair",
                        context_text=sol_ctx,
                        user_prompt=sol_prompt,
                        target_text=sol_patch,
                        encoder_limit=encoder_limit,
                        decoder_limit=decoder_limit,
                        metadata={
                            "source": row.get("source", "unknown"),
                            "task_type": "grounded_solution_repair",
                        },
                    )
                    counts["grounded_solution_repair"] += 1
                except ValueError:
                    built = None
        if built is None:
            if index < len(ul2_examples):
                built = ul2_examples[index]
                counts["span_denoising"] += 1
            else:
                built = sft_examples[index]
                counts["fallback_sft"] += 1
        stage1.append(built)

    stage2: list[Any] = []
    for index in range(len(sft_examples)):
        stage2.append(sft_examples[index] if index % 2 == 0 else stage1[index])
    return stage1, stage2, counts


def is_cross_bridge_parameter(name: str) -> bool:
    """Return True if parameter belongs to encoder_projection or the full-rank cross-attention bridge."""
    return name.startswith("encoder_projection.") or any(
        tok in name
        for tok in (
            ".cross_k_proj.",
            ".cross_v_proj.",
            ".cross_k_norm.",
            ".cross_v_norm.",
            ".cross_q_norm.",
            ".cross_gate",
        )
    )


def configure_full_rank_cross_attention(model: Any) -> dict[str, Any]:
    """Attach full-rank 16-bit (bfloat16 compute) cross-attention K/V projections, unrotated cross-query norm, and learnable gates."""
    import torch
    from torch.nn import functional as F
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

    import bifocalgemma.models.inference_cache as inference_cache_mod
    from bifocalgemma.models.memory_efficient import (
        TiledMergedAttention,
        attention_tile,
        tiled_attention,
    )

    orig_cached_cls = getattr(
        inference_cache_mod, "_orig_cached_attention", inference_cache_mod.CachedAttention
    )
    setattr(inference_cache_mod, "_orig_cached_attention", orig_cached_cls)

    class Bf16FullLinear(torch.nn.Module):
        """Full-rank linear projection with 16-bit bfloat16 Tensor Core compute and FP32 master weight."""

        def __init__(self, base_layer: torch.nn.Module) -> None:
            super().__init__()
            base: Any = getattr(base_layer, "base", base_layer)
            w = base.weight.detach().float().clone()
            self.in_features = int(w.shape[1])
            self.out_features = int(w.shape[0])
            self.weight = torch.nn.Parameter(w)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]  # torch Module callable attribute.
            compute_dtype = (
                torch.bfloat16
                if (x.device.type == "cuda" or x.dtype == torch.bfloat16)
                else x.dtype
            )
            return F.linear(x.to(compute_dtype), self.weight.to(compute_dtype)).to(x.dtype)

    class FullCrossTiledAttention(TiledMergedAttention):
        """Decoupled causal self-attention and unrotated full-rank 16-bit gated cross-attention."""

        def forward(  # ty: ignore[invalid-method-override]  # Runtime replacement accepts merged context.
            self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Any,
            shared_kv_states: dict[str, Any],
            past_key_values: Any = None,
            *,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_valid: torch.Tensor | None = None,
            decoder_valid: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            if past_key_values is not None:
                raise ValueError("Tiled training attention does not support temporal cache.")
            if encoder_hidden_states is None or encoder_valid is None or decoder_valid is None:
                raise ValueError("Merged attention requires encoder states and validity masks.")
            batch, length, _ = hidden_states.shape
            shape = (batch, length, -1, self.head_dim)
            cos, sin = position_embeddings
            raw_q = self.q_proj(hidden_states).view(shape)
            query_self = apply_rotary_pos_emb(
                self.q_norm(raw_q), cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            query_cross = self.cross_q_norm(raw_q).to(raw_q.dtype).transpose(1, 2)

            has_encoder_tokens = encoder_hidden_states.shape[1] > 0 and bool(encoder_valid.any())
            if self.is_kv_shared_layer:
                key_self, value_self, cross_key, cross_value = shared_kv_states[self.layer_type]
            else:
                if self.v_proj is None:
                    raise ValueError("Merged branch requires separate value projection.")
                key_self = self.k_norm(self.k_proj(hidden_states).view(shape))
                key_self = apply_rotary_pos_emb(key_self, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                value_self = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
                if has_encoder_tokens:
                    cross_shape = (
                        batch,
                        encoder_hidden_states.shape[1],
                        self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                        self.head_dim,
                    )
                    cross_key = (
                        self.cross_k_norm(
                            self.cross_k_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2)
                    )
                    cross_value = (
                        self.cross_v_norm(
                            self.cross_v_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2)
                    )
                else:
                    cross_key, cross_value = None, None
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (
                    key_self,
                    value_self,
                    cross_key,
                    cross_value,
                )

            self_mask = {"valid": decoder_valid, "bidirectional": False}
            self_out = tiled_attention(
                query_self,
                key_self,
                value_self,
                chunk_size=self.query_chunk_size,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                attention_mask=self_mask,
                window=self.sliding_window,
                backend=getattr(self, "attention_backend", "tiled_eager_fp32"),
            )

            if has_encoder_tokens and cross_key is not None and cross_value is not None:
                pieces = []
                for start in range(0, length, self.query_chunk_size):
                    stop = min(length, start + self.query_chunk_size)
                    cross_mask = (
                        encoder_valid[:, None, None, :].bool()
                        & decoder_valid[:, None, start:stop, None].bool()
                    )
                    pieces.append(
                        attention_tile(
                            query_cross[:, :, start:stop],
                            cross_key,
                            cross_value,
                            cross_mask,
                            self.scaling,
                            0.0,
                            True,
                        )
                    )
                cross_out = torch.cat(pieces, dim=2)
                gate_logits = torch.einsum(
                    "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
                ) * (4.0 * (self.head_dim**-0.5))
                query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
                head_gate = torch.tanh(self.cross_gate.float() * 10.0).view(1, -1, 1, 1)
                effective_gate = (head_gate * query_gate).to(self_out.dtype)
                if getattr(self, "collect_encoder_mass", False):
                    with torch.no_grad():
                        indices = torch.where(decoder_valid[0].bool())[0][-4:]
                        if indices.numel() > 0:
                            mass = float(effective_gate[0, :, indices].float().abs().mean())
                            self.encoder_mass_samples.append(mass)
                result = self_out + effective_gate * cross_out
            else:
                if getattr(self, "collect_encoder_mass", False):
                    self.encoder_mass_samples.append(0.0)
                result = self_out

            result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
            return self.o_proj(result), None

    class FullCrossCachedAttention(orig_cached_cls):  # type: ignore[valid-type,misc]
        """Cached inference attention matching FullCrossTiledAttention."""

        def forward(
            self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Any,
            shared_kv_states: dict[str, Any],
            past_key_values: Any = None,
            *,
            inference_cache: Any = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_valid: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            if not getattr(self, "full_rank_cross_attention", False):
                return super().forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    shared_kv_states,
                    past_key_values=past_key_values,
                    inference_cache=inference_cache,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_valid=encoder_valid,
                    **kwargs,
                )
            if self.training or torch.is_grad_enabled():
                raise ValueError("Inference cache requires eval and inference_mode/no_grad.")
            batch, length, _ = hidden_states.shape
            shape = (batch, length, -1, self.head_dim)
            cos, sin = position_embeddings
            raw_q = self.q_proj(hidden_states).view(shape)
            query_self = apply_rotary_pos_emb(
                self.q_norm(raw_q), cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            query_cross = self.cross_q_norm(raw_q).to(raw_q.dtype).transpose(1, 2)
            cache = inference_cache
            if cache is None:
                raise ValueError("Explicit inference cache required.")
            positions = torch.arange(cache.offset, cache.offset + length, device=query_self.device)
            if self.is_kv_shared_layer:
                key_self, value_self, cross = shared_kv_states[self.layer_type]
                key_positions, self_length = cache.shared_positions[self.layer_type]
            else:
                raw_key = self.k_proj(hidden_states).view(shape)
                raw_value = (
                    self.v_proj(hidden_states).view(shape) if self.v_proj is not None else raw_key
                )
                key_self = apply_rotary_pos_emb(
                    self.k_norm(raw_key), cos, sin, unsqueeze_dim=2
                ).transpose(1, 2)
                value_self = self.v_norm(raw_value).transpose(1, 2)
                previous = cache.layers.get(self.layer_idx)
                key_positions = positions
                if previous is not None:
                    key_self = torch.cat((previous[0], key_self), dim=2)
                    value_self = torch.cat((previous[1], value_self), dim=2)
                    key_positions = torch.cat((previous[2], positions))
                keep = self.sliding_window if self.sliding_window else key_self.shape[2]
                stored_key, stored_value, stored_positions = (
                    key_self[:, :, -keep:],
                    value_self[:, :, -keep:],
                    key_positions[-keep:],
                )
                cross = previous[3] if previous is not None else None
                if (
                    encoder_hidden_states is not None
                    and encoder_hidden_states.shape[1] > 0
                    and cross is None
                ):
                    cross_shape = (
                        batch,
                        encoder_hidden_states.shape[1],
                        self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                        self.head_dim,
                    )
                    cross = (
                        self.cross_k_norm(
                            self.cross_k_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2),
                        self.cross_v_norm(
                            self.cross_v_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2),
                    )
                cache.layers[self.layer_idx] = (
                    stored_key,
                    stored_value,
                    stored_positions,
                    cross,
                )
                self_length = key_self.shape[2]
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (key_self, value_self, cross)
                cache.shared_positions[self.layer_type] = (key_positions, self_length)

            self_pieces = []
            for start in range(0, length, 128):
                stop = min(length, start + 128)
                visible = key_positions[None, :] <= positions[start:stop, None]
                if self.sliding_window:
                    visible &= (
                        key_positions[None, :] > positions[start:stop, None] - self.sliding_window
                    )
                allowed = visible[None, None].expand(batch, 1, -1, -1)
                self_pieces.append(
                    F.scaled_dot_product_attention(
                        query_self[:, :, start:stop],
                        key_self,
                        value_self,
                        attn_mask=allowed,
                        dropout_p=0.0,
                        scale=self.scaling,
                        enable_gqa=True,
                    )
                )
            self_out = torch.cat(self_pieces, dim=2)

            if cross is not None and encoder_valid is not None and bool(encoder_valid.any()):
                cross_pieces = []
                for start in range(0, length, 128):
                    stop = min(length, start + 128)
                    cross_mask = (
                        encoder_valid[:, None, None, :].bool().expand(batch, 1, stop - start, -1)
                    )
                    cross_pieces.append(
                        attention_tile(
                            query_cross[:, :, start:stop],
                            cross[0],
                            cross[1],
                            cross_mask,
                            self.scaling,
                            0.0,
                            True,
                        )
                    )
                cross_out = torch.cat(cross_pieces, dim=2)
                gate_logits = torch.einsum(
                    "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
                ) * (4.0 * (self.head_dim**-0.5))
                query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
                head_gate = torch.tanh(self.cross_gate.float() * 10.0).view(1, -1, 1, 1)
                effective_gate = (head_gate * query_gate).to(self_out.dtype)
                result = self_out + effective_gate * cross_out
            else:
                result = self_out

            result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
            return self.o_proj(result), None

    setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)
    kv_owner_layers = 0
    total_cross_params = 0
    for layer in model.decoder.layers:
        attn = layer.self_attn
        q_base = getattr(attn.q_proj, "base", attn.q_proj)
        dev = q_base.weight.device
        num_heads = int(q_base.weight.shape[0] // attn.head_dim)
        attn.cross_q_norm = copy.deepcopy(attn.q_norm).float().to(dev)
        attn.cross_gate = torch.nn.Parameter(
            torch.full((num_heads,), 0.015, dtype=torch.float32, device=dev)
        )
        attn.cross_gate_proj = torch.nn.Parameter(
            torch.zeros((num_heads, attn.head_dim), dtype=torch.float32, device=dev)
        )
        if not attn.is_kv_shared_layer:
            attn.cross_k_proj = Bf16FullLinear(attn.k_proj).to(dev)
            attn.cross_v_proj = Bf16FullLinear(attn.v_proj).to(dev)
            attn.cross_k_norm = copy.deepcopy(attn.k_norm).float().to(dev)
            attn.cross_v_norm = copy.deepcopy(attn.v_norm).float().to(dev)
            kv_owner_layers += 1
        attn.full_rank_cross_attention = True
        attn.__class__ = FullCrossTiledAttention

    for n, p in model.named_parameters():
        if is_cross_bridge_parameter(n):
            total_cross_params += int(p.numel())
    return {
        "mode": "full_rank_cross_bf16",
        "kv_owner_layers": kv_owner_layers,
        "total_decoder_layers": len(model.decoder.layers),
        "total_cross_bridge_parameters": total_cross_params,
    }


def configure_pilot_stage(
    model: Any, policy: dict[str, Any], *, freeze_decoder: bool, total_steps: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Configure trainable groups for either frozen-decoder alignment or full joint SFT."""
    from bifocalgemma.training.memory_probe import ProbeLoRALinear

    if total_steps < 1:
        raise ValueError("Stage horizon must be a positive integer.")
    default_scaling = policy["alpha"] / policy["rank"]
    for name, module in model.named_modules():
        if isinstance(module, ProbeLoRALinear):
            if name.startswith("decoder."):
                module.scaling = 0.0 if freeze_decoder else default_scaling
            else:
                module.scaling = default_scaling
    for name, param in model.named_parameters():
        if is_cross_bridge_parameter(name):
            param.requires_grad_(True)
        elif name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(True)
        elif name.startswith("decoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(not freeze_decoder)
        else:
            param.requires_grad_(False)

    active_names = (
        ("encoder", "alignment") if freeze_decoder else ("encoder", "decoder", "alignment")
    )
    entries: dict[str, list[tuple[str, Any]]] = {name: [] for name in active_names}
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        if is_cross_bridge_parameter(name):
            entries["alignment"].append((name, param))
        elif name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            entries["encoder"].append((name, param))
        elif (
            not freeze_decoder
            and name.startswith("decoder.")
            and name.endswith((".lora_A", ".lora_B"))
        ):
            entries["decoder"].append((name, param))
        else:
            raise ValueError(f"Unexpected trainable parameter in pilot stage: {name}")

    groups: list[dict[str, Any]] = []
    for group_name in active_names:
        if not entries[group_name]:
            raise ValueError(f"Empty trainable group in pilot stage: {group_name}")
        settings = policy["groups"][group_name]
        warmup_ratio = settings.get("warmup_ratio", 0.03)
        warmup_steps = min(
            total_steps, max(settings.get("warmup_steps", 0), math.ceil(total_steps * warmup_ratio))
        )
        groups.append(
            {
                "name": group_name,
                "params": [p for _, p in entries[group_name]],
                "param_names": [n for n, _ in entries[group_name]],
                "lr": settings["lr"],
                "base_lr": settings["lr"],
                "warmup_steps": warmup_steps,
                "scheduler": settings.get("scheduler", "cosine"),
                "total_steps": total_steps,
                "warmup_ratio": warmup_ratio,
                "min_lr_ratio": settings.get("min_lr_ratio", 0.1),
            }
        )
    summary = {
        "freeze_decoder": freeze_decoder,
        "active_groups": list(active_names),
        "trainable_parameters": {
            g["name"]: sum(int(p.numel()) for p in g["params"]) for g in groups
        },
        "decoder_trainable_tensors": sum(
            1
            for n, p in model.named_parameters()
            if n.startswith("decoder.") and not is_cross_bridge_parameter(n) and p.requires_grad
        ),
    }
    return groups, summary


def run_pilot_stage(
    model: Any,
    train_examples: list[Any],
    dev_examples: list[Any],
    run_dir: Path,
    *,
    policy: dict[str, Any],
    max_steps: int,
    freeze_decoder: bool,
    batch_size: int = 1,
    accumulation: int = 4,
    clip_norm: float = 1.0,
    seed: int = 17,
    progress: Any = None,
    checkpoint_store: Any = None,
) -> dict[str, Any]:
    """Execute one controlled pilot training stage and verify parameter freeze invariants."""
    import torch
    from safetensors.torch import save_file

    from bifocalgemma.data.collator import RepairCollator
    from bifocalgemma.training.offline_sft import evaluate_sft_loss
    from bifocalgemma.training.policy import update_policy_learning_rates
    from bifocalgemma.training.stages import atomic_json

    if max_steps < 1 or batch_size < 1 or accumulation < 1 or not train_examples:
        raise ValueError("Invalid pilot stage training configuration.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    groups, stage_summary = configure_pilot_stage(
        model, policy, freeze_decoder=freeze_decoder, total_steps=max_steps
    )
    opt_cfg = policy.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        eps=float(opt_cfg.get("eps", 1e-8)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        foreach=False,
    )
    trainable_params = [p for g in groups for p in g["params"]]
    decoder_Snapshot = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if n.startswith("decoder.")
        and not is_cross_bridge_parameter(n)
        and n.endswith((".lora_A", ".lora_B"))
    }
    encoder_initial = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if (n.startswith("encoder.") and n.endswith((".lora_A", ".lora_B")))
        or is_cross_bridge_parameter(n)
    }

    order = list(range(len(train_examples)))
    random.Random(seed).shuffle(order)
    cursor = 0
    epoch = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()

    start_step = 0
    if checkpoint_store is not None:
        saved = checkpoint_store.load(model, optimizer)
        if saved is not None:
            start_step = saved["step"]
            order, cursor, epoch = saved["order"], saved["cursor"], saved["epoch"]
            history = saved["history"]
            decoder_Snapshot = saved["decoder_initial"]
            encoder_initial = saved["encoder_initial"]

    for step in range(start_step, max_steps):
        if cursor >= len(order):
            epoch += 1
            order = list(range(len(train_examples)))
            random.Random(seed + epoch).shuffle(order)
            cursor = 0
        step_started = time.perf_counter()
        micro_batches = []
        item_batches = []
        for _ in range(accumulation):
            if cursor >= len(order):
                break
            indices = order[cursor : cursor + batch_size]
            cursor += len(indices)
            items = [train_examples[i] for i in indices]
            item_batches.append(items)
            micro_batches.append(RepairCollator()(items))
        token_count = sum(int((b.labels != -100).sum()) for b in micro_batches)
        if not token_count:
            raise ValueError("No supervised tokens in pilot optimizer window.")
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for batch_cpu in micro_batches:
            batch = batch_cpu.to(device)
            count = int((batch_cpu.labels != -100).sum())
            output = model.forward_loss(batch) if hasattr(model, "forward_loss") else model(batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError("Nonfinite pilot training loss.")
            (output.loss * (count / token_count)).backward()
            mean_loss += output.loss.detach().item() * (count / token_count)
            del output, batch
        if freeze_decoder:
            leaked = [
                n
                for n, p in model.named_parameters()
                if n.startswith("decoder.")
                and not is_cross_bridge_parameter(n)
                and p.grad is not None
                and bool(p.grad.abs().sum() > 0)
            ]
            if leaked:
                raise RuntimeError(
                    f"Frozen decoder received nonzero parameter gradients: {leaked[:3]}"
                )
        grads = [p.grad for p in trainable_params if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("Missing or nonfinite pilot gradients.")
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm, foreach=False)
        )
        rates = update_policy_learning_rates(optimizer, step)
        group_diag = None
        before = None
        if step == 0 or (step + 1) % max(1, min(25, max_steps // 4)) == 0 or step + 1 == max_steps:
            group_diag = {
                g["name"]: {
                    "gradient_norm_after_clip": sum(
                        float(p.grad.float().square().sum())
                        for p in g["params"]
                        if p.grad is not None
                    )
                    ** 0.5
                }
                for g in optimizer.param_groups
            }
            before = {id(p): p.detach().clone() for p in trainable_params}
        optimizer.step()
        if before is not None and group_diag is not None:
            for g in optimizer.param_groups:
                update = (
                    sum(float((p.detach() - before[id(p)]).square().sum()) for p in g["params"])
                    ** 0.5
                )
                magnitude = sum(float(before[id(p)].square().sum()) for p in g["params"]) ** 0.5
                group_diag[g["name"]].update(
                    update_norm=update,
                    parameter_norm_before=magnitude,
                    relative_update=update / max(magnitude, 1e-30),
                )
            del before
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda" and torch.cuda.memory_reserved(device) > 28 * 2**30:
            torch.cuda.empty_cache()
        step_seconds = time.perf_counter() - step_started
        row: dict[str, Any] = {
            "step": step + 1,
            "max_steps": max_steps,
            "train_loss": mean_loss,
            "target_tokens": token_count,
            "gradient_norm": grad_norm,
            "learning_rates": rates,
            "step_seconds": step_seconds,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            ),
        }
        if group_diag is not None:
            row["optimizer_diagnostics"] = group_diag
        row["active_training_seconds"] = sum(r["step_seconds"] for r in history) + step_seconds
        history.append(row)
        if checkpoint_store is not None and ((step + 1) % 25 == 0 or step + 1 == max_steps):
            checkpoint_store.save(
                model,
                optimizer,
                {
                    "step": step + 1,
                    "order": order,
                    "cursor": cursor,
                    "epoch": epoch,
                    "history": history,
                    "decoder_initial": decoder_Snapshot,
                    "encoder_initial": encoder_initial,
                },
            )
        if checkpoint_store is None and (step + 1) % 50 == 0:
            checkpoint_tensors = {
                n: (
                    p.detach().to(torch.bfloat16).cpu().contiguous()
                    if checkpoint_store is None and (".cross_k_proj." in n or ".cross_v_proj." in n)
                    else p.detach().cpu().contiguous()
                )
                for n, p in model.named_parameters()
                if is_cross_bridge_parameter(n) or n.endswith((".lora_A", ".lora_B"))
            }
            save_file(checkpoint_tensors, str(run_dir / "stage_adapters.safetensors"))
            del checkpoint_tensors

        if progress:
            progress(
                {
                    "stage": "pilot_step",
                    "freeze_decoder": freeze_decoder,
                    **row,
                }
            )

    max_decoder_delta = max(
        (
            float((model.get_parameter(n).detach().cpu() - before_t).abs().max())
            for n, before_t in decoder_Snapshot.items()
        ),
        default=0.0,
    )
    if freeze_decoder and max_decoder_delta != 0.0:
        raise RuntimeError(f"Frozen decoder weights mutated during stage: {max_decoder_delta}")
    max_encoder_delta = max(
        (
            float((model.get_parameter(n).detach().cpu() - before_t).abs().max())
            for n, before_t in encoder_initial.items()
        ),
        default=0.0,
    )
    if checkpoint_store is not None:
        model.pilot_journal = checkpoint_store.journal()
    held_out = (
        evaluate_sft_loss(
            model, dev_examples, batch_size=batch_size, device=device, event_callback=progress
        )
        if dev_examples
        else None
    )
    saved_tensors = {
        n: (
            p.detach().to(torch.bfloat16).cpu().contiguous()
            if checkpoint_store is None and (".cross_k_proj." in n or ".cross_v_proj." in n)
            else p.detach().cpu().contiguous()
        )
        for n, p in model.named_parameters()
        if is_cross_bridge_parameter(n) or n.endswith((".lora_A", ".lora_B"))
    }
    save_file(saved_tensors, str(run_dir / "stage_adapters.safetensors"))
    head_count = min(20, len(history))
    tail_count = min(20, len(history))
    report = {
        "status": "COMPLETED",
        "stage_summary": stage_summary,
        "steps_completed": len(history),
        "wall_seconds": time.perf_counter() - started,
        "active_training_seconds": sum(r["step_seconds"] for r in history),
        "mean_step_seconds": sum(r["step_seconds"] for r in history) / len(history),
        "initial_window_loss": sum(r["train_loss"] for r in history[:head_count]) / head_count,
        "final_window_loss": sum(r["train_loss"] for r in history[-tail_count:]) / tail_count,
        "max_decoder_lora_abs_delta": max_decoder_delta,
        "max_encoder_or_alignment_abs_delta": max_encoder_delta,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        ),
        "held_out_eval": held_out,
        "optimizer_diagnostics_samples": [
            {"step": r["step"], "groups": r["optimizer_diagnostics"]}
            for r in history
            if "optimizer_diagnostics" in r
        ],
    }
    atomic_json(run_dir / "stage_report.json", report)
    (run_dir / "history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in history), encoding="utf-8"
    )
    return report


def evaluate_source_copy_probes(
    model: Any,
    probes: list[dict[str, Any]],
    decoder_tokenizer: Any,
    *,
    max_new_tokens: int = 256,
    generate_limit: int = 8,
) -> dict[str, Any]:
    """Measure exact source copying, context sensitivity, and counterfactual identifier recall."""

    from bifocalgemma.data.tokenization import decode_text
    from bifocalgemma.evaluation.context import completion_metrics
    from bifocalgemma.inference.generation import greedy_decode
    from bifocalgemma.training.offline_sft import evaluate_sft_loss

    if not probes:
        return {"probe_count": 0}
    device = next(model.parameters()).device
    previous = model.training
    model.eval()
    try:
        copy_examples = [p["copy_example"] for p in probes]
        renamed_examples = [p["renamed_example"] for p in probes]
        span_examples = [p["span_example"] for p in probes]
        empty_copy = [replace(ex, encoder_ids=()) for ex in copy_examples]
        swapped_copy = [
            replace(ex, encoder_ids=copy_examples[(i + 1) % len(copy_examples)].encoder_ids)
            for i, ex in enumerate(copy_examples)
        ]
        # Counterfactual: target has renamed_identifier, but encoder has the UNRENAMED context.
        unrenamed_for_renamed = [
            replace(ex, encoder_ids=copy_examples[i].encoder_ids)
            for i, ex in enumerate(renamed_examples)
        ]
        empty_for_renamed = [replace(ex, encoder_ids=()) for ex in renamed_examples]

        nll = {
            "verifiable_copy_correct_nll": evaluate_sft_loss(
                model, copy_examples, batch_size=1, device=device
            )["loss"],
            "verifiable_copy_empty_nll": evaluate_sft_loss(
                model, empty_copy, batch_size=1, device=device
            )["loss"],
            "verifiable_copy_swapped_nll": (
                evaluate_sft_loss(model, swapped_copy, batch_size=1, device=device)["loss"]
                if len(copy_examples) > 1
                else None
            ),
            "span_recovery_correct_nll": evaluate_sft_loss(
                model, span_examples, batch_size=1, device=device
            )["loss"],
            "renamed_identifier_correct_nll": evaluate_sft_loss(
                model, renamed_examples, batch_size=1, device=device
            )["loss"],
            "renamed_identifier_unrenamed_context_nll": evaluate_sft_loss(
                model, unrenamed_for_renamed, batch_size=1, device=device
            )["loss"],
            "renamed_identifier_empty_context_nll": evaluate_sft_loss(
                model, empty_for_renamed, batch_size=1, device=device
            )["loss"],
        }
        nll["copy_context_gain_nll"] = (
            nll["verifiable_copy_empty_nll"] - nll["verifiable_copy_correct_nll"]
        )
        nll["renamed_counterfactual_gain_nll"] = (
            nll["renamed_identifier_unrenamed_context_nll"] - nll["renamed_identifier_correct_nll"]
        )

        generations: list[dict[str, Any]] = []
        selected = probes[: min(generate_limit, len(probes))]
        for item in selected:
            journal = getattr(model, "pilot_journal", None)
            cache_key = "copy-" + item["instance_id"]
            cached = journal.get(cache_key) if journal else None
            if cached is not None:
                generations.append(cached)
                continue
            row_out: dict[str, Any] = {
                "instance_id": item["instance_id"],
                "original_identifier": item["original_identifier"],
                "renamed_identifier": item["renamed_identifier"],
            }
            for kind, ex_key, ref_key in (
                ("verifiable_copy", "copy_example", "target_span"),
                ("renamed_copy", "renamed_example", "renamed_target"),
                ("span_recovery", "span_example", "target_span"),
            ):
                ex = item[ex_key]
                gen = greedy_decode(
                    model,
                    ex.encoder_ids,
                    ex.decoder_prefix_ids,
                    max_new_tokens=max_new_tokens,
                    max_total_decoder_tokens=8192,
                    use_cache=True,
                )
                tokens = gen.token_ids
                while tokens and tokens[-1] in (1, 106, 50):
                    tokens = tokens[:-1]
                text = decode_text(decoder_tokenizer, tokens, skip_special_tokens=False).strip()
                clean = re.sub(r"^```[A-Za-z0-9_-]*\n|\n```$", "", text).strip()
                metrics = completion_metrics(clean, item[ref_key])
                row_out[kind] = {
                    "exact_match": metrics["exact_match"],
                    "sequence_similarity": metrics["sequence_similarity"],
                    "identifier_f1": metrics["regex_identifier_set_f1"],
                    "renamed_identifier_recalled": (
                        item["renamed_identifier"] in clean if kind == "renamed_copy" else None
                    ),
                    "original_identifier_emitted": item["original_identifier"] in clean,
                    "preview": clean[:240],
                    "prediction": clean,
                }
            generations.append(row_out)
            if journal:
                journal.put(cache_key, row_out)

        def mean_metric(kind: str, field: str) -> float:
            vals = [float(g[kind][field]) for g in generations]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "probe_count": len(probes),
            "generation_count": len(generations),
            "teacher_forced_nll": nll,
            "generation_summary": {
                "verifiable_copy_exact_match_rate": mean_metric("verifiable_copy", "exact_match"),
                "verifiable_copy_similarity": mean_metric("verifiable_copy", "sequence_similarity"),
                "verifiable_copy_identifier_f1": mean_metric("verifiable_copy", "identifier_f1"),
                "span_recovery_exact_match_rate": mean_metric("span_recovery", "exact_match"),
                "span_recovery_similarity": mean_metric("span_recovery", "sequence_similarity"),
                "span_recovery_identifier_f1": mean_metric("span_recovery", "identifier_f1"),
                "renamed_copy_exact_match_rate": mean_metric("renamed_copy", "exact_match"),
                "renamed_copy_similarity": mean_metric("renamed_copy", "sequence_similarity"),
                "renamed_identifier_recall_rate": (
                    sum(1 for g in generations if g["renamed_copy"]["renamed_identifier_recalled"])
                    / len(generations)
                    if generations
                    else 0.0
                ),
            },
            "samples": generations,
        }
    finally:
        model.train(previous)


def normalize_and_recount_patch(
    completion_text: str, snapshot: Any, *, default_path: str = "solution.py"
) -> str | None:
    """Construct a diagnostic canonical git diff if hunk counts or git headers were the only issue."""
    from bifocalgemma.data.patches import extract_unified_diff

    raw = completion_text.strip()
    try:
        patch = extract_unified_diff(raw)
    except ValueError:
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\n|\n```$", "", raw.strip()).strip()
        if "@@ " not in stripped:
            return None
        if not stripped.startswith("diff --git "):
            if not stripped.startswith("--- "):
                stripped = f"--- a/{default_path}\n+++ b/{default_path}\n" + stripped
            patch = f"diff --git a/{default_path} b/{default_path}\n" + stripped
        else:
            patch = stripped
    if not patch.endswith("\n"):
        patch += "\n"
    source_files = {f.path: f.content for f in snapshot.files}
    if default_path not in source_files:
        return None
    with tempfile.TemporaryDirectory(prefix="pilot-recount-") as temp_dir:
        work = Path(temp_dir)
        for rel_path, content in source_files.items():
            target = work / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        proc = subprocess.run(
            ["git", "apply", "--recount", "--whitespace=nowarn", "-"],
            input=patch.encode("utf-8"),
            cwd=work,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        changed = (work / default_path).read_text(encoding="utf-8")
        source = source_files[default_path]
        if changed == source:
            return None
        return f"diff --git a/{default_path} b/{default_path}\n" + "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                changed.splitlines(keepends=True),
                fromfile=f"a/{default_path}",
                tofile=f"b/{default_path}",
            )
        )


def evaluate_humaneval_subset(
    model: Any,
    spec: Any,
    directory: Path,
    tokenizer_cache: Path,
    *,
    limit: int = 16,
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    """Evaluate functional code repair, context NLL, source grounding, and recount counterfactuals."""
    import torch
    import torch.nn.functional as F

    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
    from bifocalgemma.evaluation.context import prompt_ids, score_context_case
    from bifocalgemma.inference.generation import greedy_decode
    from bifocalgemma.repository_study import snapshot_from_json

    directory = Path(directory)
    rows = [
        json.loads(line)
        for line in (directory / "eval_inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = {
        r["instance_id"]: r
        for r in (
            json.loads(line)
            for line in (directory / "eval_private/specs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    priority_ids = ["Python/0", "Python/1", "Python/39"]
    priority = [r for r in rows if r["instance_id"] in priority_ids]
    others = [r for r in rows if r["instance_id"] not in priority_ids]
    selected = (priority + others)[: max(1, min(limit, len(rows)))]

    decoder_tok, _ = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder_tok = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))[0]
        if spec.encoder is not None
        else None
    )
    previous = model.training
    model.eval()
    cases: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(selected):
            identifier = row["instance_id"]
            journal = getattr(model, "pilot_journal", None)
            cached = journal.get("repair-" + identifier) if journal else None
            if cached is not None:
                cases.append(cached)
                continue
            gold_row = gold[identifier]
            snapshot = snapshot_from_json(directory / gold_row["snapshot_file"])
            target_file = snapshot.files[0].path if snapshot.files else "solution.py"
            enc_ids, prefix_ids = prompt_ids(
                row, spec, encoder_tok, decoder_tok, encoder_limit=32768
            )
            target_ids = tuple(decoder_tok.encode(gold_row["gold_patch"], add_special_tokens=False))
            other_row = selected[(index + 1) % len(selected)]
            swapped_ids = (
                tuple(encoder_tok.encode(other_row["context"], add_special_tokens=True))
                if encoder_tok is not None
                else ()
            )
            losses: dict[str, float] = {}
            with torch.inference_mode():
                for variant, ctx_ids in (
                    ("correct", enc_ids),
                    ("empty", ()),
                    ("swapped", swapped_ids),
                ):
                    if encoder_tok is None and variant != "correct":
                        continue
                    enc_t = torch.tensor([ctx_ids], device="cuda", dtype=torch.long)
                    enc_mask = torch.ones_like(enc_t, dtype=torch.bool)
                    states = (
                        model.encode_context(enc_t, enc_mask) if encoder_tok is not None else None
                    )
                    full = prefix_ids + target_ids
                    dec_in = torch.tensor([full[:-1]], device="cuda", dtype=torch.long)
                    logits = model.decode_logits(
                        dec_in,
                        torch.ones_like(dec_in, dtype=torch.bool),
                        states,
                        enc_mask if encoder_tok is not None else None,
                    )
                    offset = len(prefix_ids) - 1
                    labels = torch.tensor([target_ids], device="cuda", dtype=torch.long)
                    losses[variant] = float(
                        F.cross_entropy(logits[:, offset:].float().flatten(0, 1), labels.flatten())
                    )
                    del logits

                generated = greedy_decode(
                    model,
                    enc_ids,
                    prefix_ids,
                    max_new_tokens=max_new_tokens,
                    max_total_decoder_tokens=41472,
                    use_cache=True,
                )
            tokens = generated.token_ids
            while tokens and tokens[-1] in (1, 106, 50):
                tokens = tokens[:-1]
            text = decode_text(decoder_tok, tokens, skip_special_tokens=False)
            strict_score = score_context_case(
                row,
                gold_row,
                text,
                directory=directory,
                python_executable=sys.executable,
                allow_execution=True,
            )
            snapshot = snapshot_from_json(directory / gold_row["snapshot_file"])
            recounted_score = None
            if not strict_score.get("patch_applied"):
                canonical = normalize_and_recount_patch(text, snapshot, default_path=target_file)
                if canonical is not None:
                    recounted_score = score_context_case(
                        row,
                        gold_row,
                        canonical,
                        directory=directory,
                        python_executable=sys.executable,
                        allow_execution=True,
                    )
            source_ids = {
                tok
                for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", row["context"])
                if not keyword.iskeyword(tok) and tok not in PYTHON_BUILTINS
            }
            patch_preimage_lines = [
                line[1:]
                for line in text.splitlines()
                if (line.startswith("-") and not line.startswith("---"))
                or (line.startswith(" ") and "@@" not in line)
            ]
            preimage_ids = {
                tok
                for tok in re.findall(
                    r"\b[A-Za-z_][A-Za-z0-9_]*\b", "\n".join(patch_preimage_lines)
                )
                if not keyword.iskeyword(tok) and tok not in PYTHON_BUILTINS
            }
            grounding_precision = (
                len(preimage_ids & source_ids) / len(preimage_ids) if preimage_ids else 0.0
            )
            cases.append(
                {
                    "instance_id": identifier,
                    "loss": losses,
                    "strict_patch_applied": bool(strict_score.get("patch_applied")),
                    "strict_functional_pass": bool(strict_score.get("functional_pass")),
                    "strict_reason": strict_score.get("reason"),
                    "recount_patch_applied": bool(
                        strict_score.get("patch_applied")
                        or (recounted_score and recounted_score.get("patch_applied"))
                    ),
                    "recount_functional_pass": bool(
                        strict_score.get("functional_pass")
                        or (recounted_score and recounted_score.get("functional_pass"))
                    ),
                    "preimage_identifier_grounding_precision": grounding_precision,
                    "prediction_preview": text[:360],
                    "prediction": text,
                }
            )
            if journal:
                journal.put("repair-" + identifier, cases[-1])
        count = len(cases)
        return {
            "evaluated_cases": count,
            "mean_correct_nll": sum(c["loss"]["correct"] for c in cases) / count,
            "mean_empty_nll": (
                sum(c["loss"]["empty"] for c in cases if "empty" in c["loss"]) / count
                if any("empty" in c["loss"] for c in cases)
                else None
            ),
            "mean_swapped_nll": (
                sum(c["loss"]["swapped"] for c in cases if "swapped" in c["loss"]) / count
                if any("swapped" in c["loss"] for c in cases)
                else None
            ),
            "strict_apply_count": sum(1 for c in cases if c["strict_patch_applied"]),
            "strict_apply_rate": sum(1 for c in cases if c["strict_patch_applied"]) / count,
            "strict_pass_count": sum(1 for c in cases if c["strict_functional_pass"]),
            "strict_pass_rate": sum(1 for c in cases if c["strict_functional_pass"]) / count,
            "recount_apply_count": sum(1 for c in cases if c["recount_patch_applied"]),
            "recount_apply_rate": sum(1 for c in cases if c["recount_patch_applied"]) / count,
            "recount_pass_count": sum(1 for c in cases if c["recount_functional_pass"]),
            "recount_pass_rate": sum(1 for c in cases if c["recount_functional_pass"]) / count,
            "mean_preimage_identifier_grounding": (
                sum(c["preimage_identifier_grounding_precision"] for c in cases) / count
            ),
            "cases": cases,
        }
    finally:
        model.train(previous)


def evaluate_pilot_checkpoint(
    model: Any,
    spec: Any,
    dev_examples: list[Any],
    copy_probes: list[dict[str, Any]],
    humaneval_dir: Path,
    tokenizer_cache: Path,
    *,
    humaneval_limit: int = 16,
    progress: Any = None,
) -> dict[str, Any]:
    """Run all diagnostic gates (dev loss, context swap, attention mass, source copy, HumanEvalFix)."""
    from bifocalgemma.data.tokenization import load_verified_tokenizer
    from bifocalgemma.training.v2_diagnostics import encoder_diagnostics

    decoder_tok, _ = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    if progress:
        progress({"stage": "evaluating_encoder_diagnostics"})
    enc_diag = encoder_diagnostics(model, dev_examples, progress=progress)
    if progress:
        progress({"stage": "evaluating_source_copy_probes"})
    copy_eval = evaluate_source_copy_probes(model, copy_probes, decoder_tok)
    if progress:
        progress({"stage": "evaluating_humanevalfix_subset"})
    humaneval_eval = evaluate_humaneval_subset(
        model, spec, humaneval_dir, tokenizer_cache, limit=humaneval_limit
    )
    return {
        "dev_loss": enc_diag["original"]["loss"],
        "dev_permuted_loss": enc_diag["permuted"]["loss"],
        "dev_context_loss_delta": enc_diag["loss_increase"],
        "dev_by_task_type": enc_diag["original"].get("by_task_type", {}),
        "dev_by_source": enc_diag["original"].get("by_source", {}),
        (
            "effective_gate_magnitude"
            if getattr(model.decoder.layers[0].self_attn, "full_rank_cross_attention", False)
            else "encoder_attention_mass"
        ): enc_diag["encoder_attention_mass_last_valid_queries"],
        "source_copy": copy_eval,
        "humanevalfix": humaneval_eval,
    }


def pilot_e2b_e2b(
    runtime_root: Path,
    *,
    steps_per_arm: int = 694,
    arms: tuple[str, ...] = DEFAULT_ARMS,
    humaneval_limit: int = 16,
    output_root: Path | None = None,
    encoder_source: str = "it",
    event_callback: Any = None,
    durable: Any = None,
) -> dict[str, Any]:
    """Run the 1/4-budget E2B->E2B adaptation pilot on a dedicated Colab G4 GPU."""
    import gc
    import os

    import torch
    from huggingface_hub import get_token

    from bifocalgemma.config import get_v2_experiment
    from bifocalgemma.data.offline_sft import load_curated_pool
    from bifocalgemma.data.releases import fetch_collection
    from bifocalgemma.data.tokenization import load_verified_tokenizer
    from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
    from bifocalgemma.training.policy import configure_policy_model
    from bifocalgemma.training.recipes import unified_recipe
    from bifocalgemma.training.repository_sft import tokenize_repository_record
    from bifocalgemma.training.stages import atomic_json
    from bifocalgemma.training.telemetry import append_event
    from bifocalgemma.training.v2 import (
        COLLECTION,
        DATASET_REPO,
        DATASET_REVISION,
        download_model,
        target_fingerprint,
    )

    root = Path(output_root) if output_root is not None else Path("/content/bifocalgemma-test")
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    secret = Path("/content/.bifocalgemma-hf-token")
    if output_root is None and secret.is_file():
        try:
            os.environ["HF_TOKEN"] = secret.read_text(encoding="utf-8").strip()
        finally:
            secret.unlink(missing_ok=True)
    token = get_token()
    if not token:
        raise RuntimeError("Missing authorized HF credentials for gated backbones/dataset.")
    if not torch.cuda.is_available():
        raise RuntimeError("Colab CUDA GPU required.")
    props = torch.cuda.get_device_properties(0)
    if (
        not any(k in props.name for k in ("6000", "A100", "H100", "L4"))
        or props.total_memory / 2**30 < 38
    ):
        raise RuntimeError(
            f"Expected modern CUDA GPU with at least 38 GiB VRAM, got {props.name} ({props.total_memory / 2**30:.1f} GiB)."
        )
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")

    def event(*args: Any, **details: Any) -> None:
        stage = args[0] if args else details.pop("stage", "event")
        sub_stage = details.pop("stage", None)
        if sub_stage is not None and sub_stage != stage:
            details["sub_stage"] = sub_stage
        payload = {"stage": stage, "time": time.time(), **details}
        append_event(root / "events.jsonl", payload)
        atomic_json(root / "status.json", payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        step_val = details.get("step", 0)
        max_step_val = details.get("max_steps", 0)
        eval_done = details.get("eval_examples_done")
        eval_total = details.get("eval_examples_total")
        eval_notify = (
            stage in ("initial_eval", "stage1_eval", "stage2_eval", "eval_progress")
        ) and (
            (eval_done and eval_total and eval_done in (1, eval_total // 2, eval_total))
            or (
                "sub_stage" in details
                and details["sub_stage"]
                in (
                    "validation_started",
                    "evaluating_encoder_diagnostics",
                    "evaluating_source_copy_probes",
                    "evaluating_humanevalfix_subset",
                )
            )
            or ("dev_loss" in details)
        )
        should_notify = (
            stage in ("arm_completed", "pilot_complete", "pilot_error")
            or eval_notify
            or (
                stage in ("stage1_progress", "stage2_progress", "pilot_step")
                and (
                    step_val in (0, 1)
                    or step_val % 25 == 0
                    or (max_step_val and step_val == max_step_val)
                )
            )
        )
        if event_callback is not None:
            event_callback(payload)
        if should_notify and output_root is None:
            try:
                import urllib.request

                summary_str = f"[{stage}] " + ", ".join(
                    f"{k}={v}" for k, v in list(details.items())[:4] if k != "eval"
                )
                req = urllib.request.Request(
                    "https://ntfy.sh/bifocalgemma-daru",
                    data=summary_str[:500].encode("utf-8"),
                    headers={"Title": f"BifocalGemma: {stage}"},
                )
                urllib.request.urlopen(req, timeout=3)
            except Exception:
                pass

    report_path = root / "pilot_e2b_e2b.json"
    report: dict[str, Any] = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {
            "experiment": "BifocalGemma-E2B-E2B-Base-Control"
            if durable
            else "BifocalGemma-E2B-E2B-Quarter-Pilot",
            "hardware": {
                "gpu": props.name,
                "vram_gib": props.total_memory / 2**30,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "runtime_root": str(runtime_root),
            },
            "dataset": {
                "repo": DATASET_REPO,
                "revision": DATASET_REVISION,
                "collection": COLLECTION,
            },
            "config": {
                "steps_per_arm": steps_per_arm,
                "effective_batch": 4,
                "batch_size": 1,
                "accumulation": 4,
                "query_chunk_size": 256,
                "loss_chunk_size": 128,
                "gradient_checkpointing": True,
                "arms": list(arms),
                "humaneval_limit": humaneval_limit,
                "seed": 17,
            },
            "arms": {},
            "hf_writes": durable is not None,
        }
    )

    def save_report() -> None:
        atomic_json(report_path, report)

    event("fetching_datasets")
    pool_dir = cache / "dataset" / COLLECTION
    fetch_collection(
        DATASET_REPO, DATASET_REVISION, COLLECTION, pool_dir, expected_role="train", token=token
    )
    humaneval_dir = cache / "benchmarks/humanevalfix_python_v2"
    fetch_collection(
        DATASET_REPO,
        DATASET_REVISION,
        humaneval_dir.name,
        humaneval_dir,
        expected_role="eval",
        include_gold=True,
        token=token,
    )
    raw_rows, pool_manifest = load_curated_pool(pool_dir)
    limits = pool_manifest["report"]["tokens"]["limits"]
    quarter_train_count = math.ceil(len(raw_rows["train"]) / 4)
    needed_train_count = min(len(raw_rows["train"]), max(quarter_train_count, steps_per_arm * 4))
    train_raw = stratified_raw_subset(raw_rows["train"], needed_train_count, seed=17)
    dev_raw = stratified_raw_subset(raw_rows["dev"], 256, seed=17)

    base_spec = get_v2_experiment("BifocalGemma-E2B-E2B")
    it_encoder_spec = replace(base_spec, encoder=base_spec.decoder)
    # Ensure both base and IT tokenizers/weights are downloaded and verified once.
    event("warming_backbones")
    warm_model, warm_prov = download_model(
        base_spec, cache, token, lambda e: event("download", **e)
    )
    del warm_model
    gc.collect()
    torch.cuda.empty_cache()

    tok_cache = cache / "tokenizers"
    dec_tok, _ = load_verified_tokenizer(base_spec.decoder, tok_cache)
    assert base_spec.encoder is not None
    enc_tok, _ = load_verified_tokenizer(base_spec.encoder, tok_cache)

    train_enc_limit = limits["encoder"]
    train_dec_limit = limits["pair_decoder"]

    def tokenize_row(
        row: dict[str, Any],
        spec_obj: Any,
        *,
        enc_limit: int | None = None,
        dec_limit: int | None = None,
    ) -> Any:
        return tokenize_repository_record(
            {**row, "context": record_context_text(row)},
            spec_obj,
            enc_tok,
            dec_tok,
            encoder_limit=enc_limit or limits["encoder"],
            decoder_limit=dec_limit or limits["pair_decoder"],
        )

    event("tokenizing_pilot_cohorts", train_examples=len(train_raw), dev_examples=len(dev_raw))
    sft_train = [
        tokenize_row(r, base_spec, enc_limit=train_enc_limit, dec_limit=train_dec_limit)
        for r in train_raw
    ]
    sft_dev = [tokenize_row(r, base_spec) for r in dev_raw]
    ul2_train, ul2_counts = build_span_denoising_examples(
        train_raw,
        base_spec,
        enc_tok,
        dec_tok,
        fallback_examples=sft_train,
        encoder_limit=train_enc_limit,
        decoder_limit=train_dec_limit,
        seed=17,
    )
    full_cross_stage1, full_cross_stage2, full_cross_counts = build_full_cross_cohorts(
        train_raw,
        base_spec,
        enc_tok,
        dec_tok,
        sft_examples=sft_train,
        ul2_examples=ul2_train,
        encoder_limit=train_enc_limit,
        decoder_limit=train_dec_limit,
        seed=17,
    )
    copy_probes = build_source_copy_probes(
        dev_raw,
        base_spec,
        enc_tok,
        dec_tok,
        count=16,
        encoder_limit=limits["encoder"],
        decoder_limit=limits["pair_decoder"],
        seed=17,
    )
    if durable is not None:
        durable.bind(
            {
                "spec": base_spec.to_dict(),
                "encoder_source": encoder_source,
                "policy": unified_recipe("symmetric"),
                "steps": steps_per_arm,
                "cohorts": [
                    durable.cohort(items)
                    for items in (full_cross_stage1, full_cross_stage2, sft_dev)
                ],
            }
        )
    report["dataset"].update(
        full_train_pool=len(raw_rows["train"]),
        quarter_train_subset=len(sft_train),
        dev_subset=len(sft_dev),
        copy_probe_count=len(copy_probes),
        ul2_span_composition=ul2_counts,
        full_cross_stage1_composition=full_cross_counts,
        train_target_fingerprint=target_fingerprint(sft_train),
        dev_target_fingerprint=target_fingerprint(sft_dev),
        provenance=warm_prov,
    )
    save_report()

    memory_options = {
        "attention_backend": "tiled_eager_fp32",
        "query_chunk_size": 256,
        "loss_chunk_size": 128,
        "gradient_checkpointing": True,
    }
    policy = unified_recipe("symmetric")

    for arm in arms:
        if report["arms"].get(arm, {}).get("status") == "COMPLETED":
            event("arm_skipped_already_completed", arm=arm)
            continue
        event("arm_started", arm=arm, steps=steps_per_arm)
        spec_for_arm = (
            it_encoder_spec
            if arm == "it_encoder_frozen_then_joint"
            or (arm == "full_cross_bf16_frozen_then_joint" and encoder_source == "it")
            else base_spec
        )
        arm_dir = root / "arms" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        model, _ = download_model(spec_for_arm, cache, token, lambda e: None)
        configure_policy_model(model, copy.deepcopy(policy))
        configure_memory_efficient_training(model, memory_options)
        cross_arch_info = None
        if arm == "full_cross_bf16_frozen_then_joint":
            cross_arch_info = configure_full_rank_cross_attention(model)
            event("configured_full_rank_cross_attention", **cross_arch_info)

        arm_record: dict[str, Any] = {
            "arm": arm,
            "encoder_source": spec_for_arm.encoder.model_id if spec_for_arm.encoder else None,
            "decoder_source": spec_for_arm.decoder.model_id,
            "cross_attention_architecture": cross_arch_info,
            "stages": {},
        }
        if durable is not None:
            durable.model = model
        report["arms"][arm] = arm_record
        save_report()

        try:
            if "initial_eval" not in report:
                event("evaluating_untrained_initial_model")
                if durable is not None:
                    durable.evaluation("initial")
                report["initial_eval"] = evaluate_pilot_checkpoint(
                    model,
                    spec_for_arm,
                    sft_dev,
                    copy_probes,
                    humaneval_dir,
                    tok_cache,
                    humaneval_limit=humaneval_limit,
                    progress=lambda e: event("initial_eval", **e),
                )
                save_report()

            if arm == "direct_joint":
                train_stage = run_pilot_stage(
                    model,
                    sft_train,
                    sft_dev,
                    arm_dir / "joint",
                    policy=copy.deepcopy(policy),
                    max_steps=steps_per_arm,
                    freeze_decoder=False,
                    batch_size=1,
                    accumulation=4,
                    seed=17,
                    progress=lambda e, a=arm: event("train_progress", arm=a, **e),
                )
                final_eval = evaluate_pilot_checkpoint(
                    model,
                    spec_for_arm,
                    sft_dev,
                    copy_probes,
                    humaneval_dir,
                    tok_cache,
                    humaneval_limit=humaneval_limit,
                    progress=lambda e, a=arm: event("eval_progress", arm=a, **e),
                )
                arm_record["stages"]["joint"] = {
                    "training": train_stage,
                    "evaluation": final_eval,
                }
                arm_record["final_evaluation"] = final_eval
            else:
                freeze_steps = max(1, steps_per_arm // 2)
                joint_steps = max(1, steps_per_arm - freeze_steps)
                if arm == "full_cross_bf16_frozen_then_joint":
                    stage1_data = full_cross_stage1
                    stage2_data = full_cross_stage2
                    stage1_name = "frozen_decoder_full_cross_bf16"
                elif arm == "span_ul2_then_joint":
                    stage1_data = ul2_train
                    stage2_data = sft_train
                    stage1_name = "frozen_decoder_span_ul2"
                else:
                    stage1_data = sft_train
                    stage2_data = sft_train
                    stage1_name = "frozen_decoder_sft"
                stage1_train = run_pilot_stage(
                    model,
                    stage1_data,
                    sft_dev,
                    arm_dir / "stage1_frozen",
                    policy=copy.deepcopy(policy),
                    max_steps=freeze_steps,
                    freeze_decoder=True,
                    checkpoint_store=durable.stage("frozen") if durable else None,
                    batch_size=1,
                    accumulation=4,
                    seed=17,
                    progress=lambda e, a=arm: event("stage1_progress", arm=a, **e),
                )
                if durable is not None:
                    durable.evaluation("frozen")
                stage1_eval = evaluate_pilot_checkpoint(
                    model,
                    spec_for_arm,
                    sft_dev,
                    copy_probes,
                    humaneval_dir,
                    tok_cache,
                    humaneval_limit=humaneval_limit,
                    progress=lambda e, a=arm: event("stage1_eval", arm=a, **e),
                )
                arm_record["stages"][stage1_name] = {
                    "training": stage1_train,
                    "evaluation": stage1_eval,
                }
                save_report()

                stage2_train = run_pilot_stage(
                    model,
                    stage2_data,
                    sft_dev,
                    arm_dir / "stage2_joint",
                    policy=copy.deepcopy(policy),
                    max_steps=joint_steps,
                    freeze_decoder=False,
                    checkpoint_store=durable.stage("joint") if durable else None,
                    batch_size=1,
                    accumulation=4,
                    seed=1017,
                    progress=lambda e, a=arm: event("stage2_progress", arm=a, **e),
                )
                if durable is not None:
                    durable.evaluation("joint")
                stage2_eval = evaluate_pilot_checkpoint(
                    model,
                    spec_for_arm,
                    sft_dev,
                    copy_probes,
                    humaneval_dir,
                    tok_cache,
                    humaneval_limit=humaneval_limit,
                    progress=lambda e, a=arm: event("stage2_eval", arm=a, **e),
                )
                arm_record["stages"]["joint_sft"] = {
                    "training": stage2_train,
                    "evaluation": stage2_eval,
                }
                arm_record["final_evaluation"] = stage2_eval

            arm_record["status"] = "COMPLETED"
            save_report()
            event(
                "arm_completed",
                arm=arm,
                dev_loss=arm_record["final_evaluation"]["dev_loss"],
                context_delta=arm_record["final_evaluation"]["dev_context_loss_delta"],
                copy_similarity=arm_record["final_evaluation"]["source_copy"]["generation_summary"][
                    "verifiable_copy_similarity"
                ],
                renamed_recall=arm_record["final_evaluation"]["source_copy"]["generation_summary"][
                    "renamed_identifier_recall_rate"
                ],
                humaneval_strict_pass=arm_record["final_evaluation"]["humanevalfix"][
                    "strict_pass_rate"
                ],
                humaneval_recount_pass=arm_record["final_evaluation"]["humanevalfix"][
                    "recount_pass_rate"
                ],
            )
        finally:
            if durable is not None:
                durable.model = None
            del model
            gc.collect()
            torch.cuda.empty_cache()

    comparison = []
    for arm_name, data in report["arms"].items():
        if data.get("status") != "COMPLETED":
            continue
        ev = data["final_evaluation"]
        copy_sum = ev["source_copy"]["generation_summary"]
        he = ev["humanevalfix"]
        comparison.append(
            {
                "arm": arm_name,
                "encoder_source": data["encoder_source"],
                "dev_loss": ev["dev_loss"],
                "dev_context_loss_delta": ev["dev_context_loss_delta"],
                "verifiable_copy_exact_match": copy_sum["verifiable_copy_exact_match_rate"],
                "verifiable_copy_similarity": copy_sum["verifiable_copy_similarity"],
                "span_recovery_similarity": copy_sum["span_recovery_similarity"],
                "renamed_identifier_recall": copy_sum["renamed_identifier_recall_rate"],
                "humaneval_strict_apply_rate": he["strict_apply_rate"],
                "humaneval_strict_pass_rate": he["strict_pass_rate"],
                "humaneval_recount_apply_rate": he["recount_apply_rate"],
                "humaneval_recount_pass_rate": he["recount_pass_rate"],
                "humaneval_identifier_grounding": he["mean_preimage_identifier_grounding"],
            }
        )
    report["comparison_table"] = comparison
    report["status"] = "COMPLETED"
    save_report()
    event("pilot_complete", arms_completed=len(comparison))
    return report


def _worker_source() -> str:
    functions = (
        stratified_raw_subset,
        record_context_text,
        extract_resolvable_span,
        encode_chat_pair,
        build_span_denoising_examples,
        rename_identifier_counterfactual,
        build_source_copy_probes,
        extract_single_window_solution_repair,
        build_full_cross_cohorts,
        is_cross_bridge_parameter,
        configure_full_rank_cross_attention,
        configure_pilot_stage,
        run_pilot_stage,
        evaluate_source_copy_probes,
        normalize_and_recount_patch,
        evaluate_humaneval_subset,
        evaluate_pilot_checkpoint,
        pilot_e2b_e2b,
    )
    preamble = (
        "import copy, difflib, hashlib, json, keyword, math, random, re, subprocess, sys, tempfile, time\n"
        "from collections import defaultdict\n"
        "from dataclasses import replace\n"
        "from pathlib import Path\n"
        "from typing import Any\n"
        f"DEFAULT_ARMS = {DEFAULT_ARMS!r}\n"
        f"PYTHON_BUILTINS = {sorted(PYTHON_BUILTINS)!r}\n\n"
    )
    body = "\n\n".join(inspect.getsource(fn) for fn in functions)
    source = preamble + body
    ast.parse(source)
    return source


def render(
    root: Path = ROOT,
    *,
    steps_per_arm: int = 694,
    arms: tuple[str, ...] = DEFAULT_ARMS,
    humaneval_limit: int = 16,
) -> str:
    manifest = json.loads((root / "notebooks/runtime-manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source_bundle_bytes(root)).hexdigest() != manifest["sha256"]:
        raise ValueError("Publish the matching runtime before exporting the Colab pilot.")
    loader = (root / "tools/runtime_bootstrap.py").read_text(encoding="utf-8")
    worker_defs = _worker_source()
    child = (
        "import os, runpy, sys\nfrom pathlib import Path\n"
        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
        "root = Path(sys.argv[1])\n"
        "runpy.run_path(str(root / 'tools/notebook_launcher.py'))['prepare_runtime'](root)\n"
        + worker_defs
        + "\n_env_arms = os.environ.get('BIFOCAL_PILOT_ARMS')\n"
        + f"_arms = tuple(a.strip() for a in _env_arms.split(',') if a.strip()) if _env_arms else {tuple(arms)!r}\n"
        + "_env_steps = os.environ.get('BIFOCAL_PILOT_STEPS')\n"
        + f"_steps = int(_env_steps) if _env_steps else {steps_per_arm}\n"
        + f"pilot_e2b_e2b(root, steps_per_arm=_steps, arms=_arms, humaneval_limit={humaneval_limit})\n"
    )
    ast.parse(child)
    return (
        '"""Generated Colab G4 pilot for BifocalGemma-E2B-E2B encoder adaptation."""\n'
        + loader
        + "\nimport subprocess\nimport sys\nfrom pathlib import Path\n"
        + f"manifest = {manifest!r}\n"
        + "root = load_runtime(manifest['repo_id'], manifest['filename'], "
        "manifest['revision'], manifest['sha256'])\n"
        + "child = Path('/content/bifocalgemma_pilot_worker.py')\n"
        + f"child.write_text({child!r}, encoding='utf-8')\n"
        + "results = Path('/content/bifocalgemma-test')\n"
        + "results.mkdir(parents=True, exist_ok=True)\n"
        + "tok_path = Path('/content/.bifocalgemma-hf-token')\n"
        + "import os\n"
        + "hf_token = os.environ.get('HF_TOKEN')\n"
        + "if not hf_token and tok_path.is_file():\n"
        + "    try:\n"
        + "        hf_token = tok_path.read_text(encoding='utf-8').strip()\n"
        + "    except Exception:\n"
        + "        pass\n"
        + "def _push_checkpoint_to_hf(results_dir, token, repo_id='daruokta/bifocalgemma-pilot-results'):\n"
        + "    if not token:\n"
        + "        return\n"
        + "    try:\n"
        + "        from huggingface_hub import HfApi\n"
        + "        api = HfApi(token=token)\n"
        + "        api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)\n"
        + "        for path in sorted(results_dir.rglob('*.safetensors')):\n"
        + "            rel = str(path.relative_to(results_dir)).replace('\\\\', '/')\n"
        + "            api.upload_file(path_or_fileobj=str(path), path_in_repo=rel, repo_id=repo_id, repo_type='dataset')\n"
        + "        rpt = results_dir / 'pilot_e2b_e2b.json'\n"
        + "        if rpt.is_file():\n"
        + "            api.upload_file(path_or_fileobj=str(rpt), path_in_repo='pilot_e2b_e2b.json', repo_id=repo_id, repo_type='dataset')\n"
        + "        wlog = results_dir / 'worker.log'\n"
        + "        if wlog.is_file():\n"
        + "            api.upload_file(path_or_fileobj=str(wlog), path_in_repo='worker.log', repo_id=repo_id, repo_type='dataset')\n"
        + "        evs = results_dir / 'events.jsonl'\n"
        + "        if evs.is_file():\n"
        + "            api.upload_file(path_or_fileobj=str(evs), path_in_repo='events.jsonl', repo_id=repo_id, repo_type='dataset')\n"
        + "        print(f'[HF Periodic Sync] Pushed latest checkpoints & logs to {repo_id}', flush=True)\n"
        + "    except Exception as exc:\n"
        + "        print(f'[HF Periodic Sync Warning] Could not sync checkpoint: {exc}', flush=True)\n"
        + "try:\n"
        + "    with (results / 'worker.log').open('a', encoding='utf-8') as log:\n"
        + "        with subprocess.Popen([sys.executable, '-u', str(child), str(root)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, encoding='utf-8', errors='replace') as process:\n"
        + "            import json\n"
        + "            last_sync_step = -1\n"
        + "            for line in process.stdout:\n"
        + "                log.write(line)\n"
        + "                log.flush()\n"
        + "                print(line, end='', flush=True)\n"
        + "                if hf_token and line.startswith('{'):\n"
        + "                    try:\n"
        + "                        ev = json.loads(line)\n"
        + "                        stg = ev.get('stage')\n"
        + "                        step = ev.get('step', 0)\n"
        + "                        if ((stg in ('stage1_progress', 'stage2_progress', 'pilot_step') and step and step % 50 == 0 and step != last_sync_step) or stg in ('stage1_eval', 'stage2_eval', 'arm_completed')):\n"
        + "                            last_sync_step = step\n"
        + "                            _push_checkpoint_to_hf(results, hf_token)\n"
        + "                    except Exception:\n"
        + "                        pass\n"
        + "            if process.wait() != 0:\n"
        + "                raise subprocess.CalledProcessError(process.returncode, process.args)\n"
        + "finally:\n"
        + "    import os, zipfile\n"
        + "    _push_checkpoint_to_hf(results, hf_token)\n"
        + "    try:\n"
        + "        with zipfile.ZipFile('/content/bifocalgemma-test.zip', 'w', zipfile.ZIP_DEFLATED) as archive:\n"
        + "            for path in sorted(results.rglob('*')):\n"
        + "                relative = path.relative_to(results)\n"
        + "                if path.is_file() and relative.parts[0] != 'cache' and path.suffix in {'.json', '.jsonl', '.log', '.safetensors'}:\n"
        + "                    archive.write(path, str(relative))\n"
        + "    except Exception as ze:\n"
        + "        print(f'[Zip Warning] Could not archive all files: {ze}', flush=True)\n"
        + "    try:\n"
        + "        from huggingface_hub import HfApi\n"
        + "        if hf_token:\n"
        + "            hf_api = HfApi(token=hf_token)\n"
        + "            repo_id = 'daruokta/bifocalgemma-pilot-results'\n"
        + "            hf_api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)\n"
        + "            zip_file = Path('/content/bifocalgemma-test.zip')\n"
        + "            if zip_file.is_file():\n"
        + "                hf_api.upload_file(path_or_fileobj=str(zip_file), path_in_repo='bifocalgemma-test.zip', repo_id=repo_id, repo_type='dataset')\n"
        + "            json_file = results / 'pilot_e2b_e2b.json'\n"
        + "            if json_file.is_file():\n"
        + "                hf_api.upload_file(path_or_fileobj=str(json_file), path_in_repo='pilot_e2b_e2b.json', repo_id=repo_id, repo_type='dataset')\n"
        + "            worker_log = results / 'worker.log'\n"
        + "            if worker_log.is_file():\n"
        + "                hf_api.upload_file(path_or_fileobj=str(worker_log), path_in_repo='worker.log', repo_id=repo_id, repo_type='dataset')\n"
        + "            print('[Colab Auto-Sync] Successfully uploaded results & logs to https://huggingface.co/datasets/daruokta/bifocalgemma-pilot-results', flush=True)\n"
        + "    except Exception as exc:\n"
        + "        print(f'[Colab Auto-Sync Error] Failed to sync to Hugging Face: {exc}', flush=True)\n"
        + "        try:\n"
        + "            import urllib.request\n"
        + "            urllib.request.urlopen(urllib.request.Request('https://ntfy.sh/bifocalgemma-daru', data=f'HF Upload Error: {exc}'.encode('utf-8'), headers={'Title': 'BifocalGemma Sync Error'}), timeout=5)\n"
        + "        except Exception:\n"
        + "            pass\n"
        + "    finally:\n"
        + "        tok_path.unlink(missing_ok=True)\n"
        + "    try:\n"
        + "        import urllib.request\n"
        + "        urllib.request.urlopen(urllib.request.Request('https://ntfy.sh/bifocalgemma-daru', data=b'Pilot run finished! Results uploaded to HF.', headers={'Title': 'BifocalGemma Complete'}), timeout=5)\n"
        + "    except Exception:\n"
        + "        pass\n"
        + "    try:\n"
        + "        from google.colab import runtime\n"
        + "        print('[Colab Auto-Shutdown] Releasing VM runtime to prevent idle compute unit usage...', flush=True)\n"
        + "        runtime.unassign()\n"
        + "    except Exception:\n"
        + "        pass\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps-per-arm",
        type=int,
        default=694,
        help="Optimizer steps per arm (default 694 = 2,776 total steps across 4 arms = 1/4 full epoch).",
    )
    parser.add_argument(
        "--per-arm-quarter",
        action="store_true",
        help="Use 2,776 optimizer steps (full 1/4 epoch) for each individual arm.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=DEFAULT_ARMS,
        default=list(DEFAULT_ARMS),
        help="Subset of pilot arms to execute.",
    )
    parser.add_argument(
        "--humaneval-limit",
        type=int,
        default=16,
        help="Number of HumanEvalFix Python cases to run per stage evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/colab/pilot_e2b_e2b.py",
    )
    args = parser.parse_args()
    steps = 2776 if args.per_arm_quarter else args.steps_per_arm
    args.output.parent.mkdir(parents=True, exist_ok=True)
    script = render(
        steps_per_arm=steps,
        arms=tuple(args.arms),
        humaneval_limit=args.humaneval_limit,
    )
    args.output.write_text(script, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "script": str(args.output),
                "steps_per_arm": steps,
                "arms": list(args.arms),
                "total_planned_steps": steps * len(args.arms),
                "humaneval_limit": args.humaneval_limit,
                "hf_writes": False,
            }
        )
    )


if __name__ == "__main__":
    main()
