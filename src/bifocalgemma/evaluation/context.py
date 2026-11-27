"""Restartable static-context evaluation with fixed denominators and explicit protocols."""

import ast
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

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.context_tasks import (
    format_numbered_code,
)
from bifocalgemma.data.context_tasks import (
    split_disjoint_context as split_disjoint_context,
)
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.evaluation.protocol import (
    BENCHMARK_RESET_ID as BENCHMARK_RESET_ID,
)
from bifocalgemma.evaluation.protocol import (
    build_effective_benchmark_spec,
    effective_generation_budget,
    gsm8k_report_metrics,
    mcq_report_metrics,
    repair_report_metrics,
)
from bifocalgemma.training.stages import atomic_json

EVALUATION_PROTOCOL = "greedy_no_cache_context_v5_prompt_parity_resilient_scorers"
V2_EVALUATION_PROTOCOL = "greedy_cached_context_v5_prompt_parity_resilient_scorers"


def greedy_decode(*args, **kwargs):
    from bifocalgemma.inference.generation import greedy_decode as generate

    return generate(*args, **kwargs)


_GOLD_INPUT_KEYS = (
    "answer",
    "canonical_solution",
    "correct_answer",
    "correct_letter",
    "gold",
    "gold_answer",
    "gold_needle",
    "gold_patch",
    "groundtruth",
    "is_hard",
    "needle",
    "reference",
    "solution",
    "target_patch",
    "target_text",
    "test_code",
    "test_script",
)
_REPAIR_INSTRUCTION_PREFIX = "Return only a unified diff fulfilling the instruction."
_FULL_FUNCTION_REPAIR_PREFIX = (
    "Return only the complete fixed Python code fulfilling the instruction."
)
VALID_CONTEXT_MODES = ("full", "disjoint", "local_only")
VALID_REPAIR_MODES = ("unified_diff", "full_function")
DEFAULT_LOCAL_CURSOR_WINDOW_LINES = 48
_FILE_HEADER_RE = re.compile(r"^### File:\s*([^\s(]+)", re.MULTILINE)
_FILE_SECTION_SPLIT_RE = re.compile(r"(?=^### File:\s*)", re.MULTILINE)
_FIM_MISSING_LINE_RE = re.compile(r"^[ \t]*###\s*MISSING CODE[ \t]*$", re.MULTILINE)
_FORBIDDEN_PROMPT_PHRASES = (
    "using the encoder context",
    "use the supplied encoder context",
    "encoder: object",
    "manifold warping",
    "cross-attention",
    "bifocal",
)


def _deduplicate_longcodeqa_headers(text: str) -> str:
    """Collapse duplicated Repository:/Question: header prefixes in LongCodeQA prompts."""
    if not text:
        return text
    text = re.sub(r"(?im)^([ \t]*Repository:[ \t]*)(?:Repository:[ \t]*)+", r"\1", text)
    text = re.sub(r"(?im)^([ \t]*Question:[ \t]*)(?:Question:[ \t]*)+", r"\1", text)
    text = re.sub(r"(?im)^([ \t]*Repository:)[ \t]*\r?\n[ \t]*Repository:[ \t]*", r"\1 ", text)
    text = re.sub(r"(?im)^([ \t]*Question:)[ \t]*\r?\n[ \t]*Question:[ \t]*", r"\1 ", text)
    return text


def _normalize_fim_cursor_marker(text: str) -> str:
    """Unify CrossCodeLong ### MISSING CODE marker to # <MISSING_CODE> matching SFT training."""
    if not text or "MISSING CODE" not in text:
        return text
    return _FIM_MISSING_LINE_RE.sub("# <MISSING_CODE>", text)


def _number_lines(lines: list[str], start_line: int = 1) -> str:
    return format_numbered_code("\n".join(lines), start_line=start_line, skip_if_numbered=False)


def _resolve_repair_mode(row: dict[str, Any], repair_mode: str | None = None) -> str:
    if repair_mode is not None:
        if repair_mode not in VALID_REPAIR_MODES:
            raise ValueError(f"unsupported_repair_mode:{repair_mode}")
        return repair_mode
    row_mode = row.get("repair_mode")
    if row_mode in VALID_REPAIR_MODES:
        return str(row_mode)
    col = str(row.get("collection") or row.get("benchmark") or "").lower()
    if "humanevalfix_python_full" in col or "full_function" in col:
        return "full_function"
    return "unified_diff"


def _parse_fim_sections(context: str) -> tuple[str, str, str] | None:
    """Extract (cross_file, prefix, suffix) from CrossCodeLong FIM contexts."""
    if "### PREFIX" not in context or "### SUFFIX" not in context:
        return None
    before_prefix, _, after_prefix = context.partition("### PREFIX")
    if after_prefix.startswith("\r\n"):
        after_prefix = after_prefix[2:]
    elif after_prefix.startswith("\n"):
        after_prefix = after_prefix[1:]
    prefix_raw, _, suffix_raw = after_prefix.partition("### SUFFIX")
    if suffix_raw.startswith("\r\n"):
        suffix_raw = suffix_raw[2:]
    elif suffix_raw.startswith("\n"):
        suffix_raw = suffix_raw[1:]
    prefix_clean = _FIM_MISSING_LINE_RE.sub("", prefix_raw)
    prefix_clean = prefix_clean.replace("# <MISSING_CODE>", "").rstrip("\r\n")
    cross_text = ""
    if "### CROSS-FILE CONTEXT" in before_prefix:
        _, _, cross_raw = before_prefix.partition("### CROSS-FILE CONTEXT")
        cross_text = cross_raw.strip("\r\n")
    return cross_text, prefix_clean, suffix_raw.rstrip("\r\n")


def _parse_file_sections_with_offsets(
    context: str,
) -> tuple[str, list[tuple[str, int, str]]] | None:
    """Split context into (preamble, [(path, start_line, body), ...]) when ### File: blocks exist."""
    if not _FILE_HEADER_RE.search(context):
        return None
    chunks = _FILE_SECTION_SPLIT_RE.split(context)
    preamble = ""
    files: list[tuple[str, int, str]] = []
    header_re = re.compile(
        r"^### File:\s*([^\n(]+?)(?:\s*\((?:lines\s+(\d+)-(\d+)[^)]*|[^)]*)\))?[ \t]*\r?\n?(.*)$",
        re.DOTALL,
    )
    for chunk in chunks:
        if not chunk.strip():
            continue
        match = header_re.match(chunk)
        if match:
            start_line = int(match.group(2)) if match.group(2) else 1
            files.append((match.group(1).strip(), start_line, match.group(4).rstrip("\r\n")))
        elif not files:
            preamble = (preamble + "\n\n" + chunk.strip()).strip() if preamble else chunk.strip()
    return (preamble, files) if files else None


def _parse_file_sections(context: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Split context into (preamble, [(path, body), ...]) when ### File: blocks exist."""
    parsed = _parse_file_sections_with_offsets(context)
    if parsed is None:
        return None
    preamble, files = parsed
    return preamble, [(path, body) for path, _, body in files]


def _target_window_bounds(
    row: dict[str, Any],
    target_path: str,
    total_lines: int,
    local_window_lines: int,
    *,
    base_start: int = 1,
) -> tuple[int, int]:
    """Return inclusive snippet-relative 1-based [start, end] line bounds for a local window."""
    if total_lines <= 0:
        return 1, 0
    sec_end = base_start + total_lines - 1
    selection = row.get("context_selection")
    if isinstance(selection, dict) and selection.get("gold_used") is not True:
        matched = [
            w
            for w in selection.get("windows", ())
            if isinstance(w, dict)
            and w.get("path") == target_path
            and type(w.get("start_line")) is int
            and type(w.get("end_line")) is int
            and w["end_line"] >= base_start
            and w["start_line"] <= sec_end
        ]
        if matched:
            center_start = min(w["start_line"] for w in matched)
            center_end = max(w["end_line"] for w in matched)
            rel_start = max(1, (center_start - base_start + 1) - local_window_lines)
            rel_end = min(total_lines, (center_end - base_start + 1) + local_window_lines)
            if rel_start <= rel_end:
                return rel_start, rel_end
    return 1, min(total_lines, 2 * local_window_lines)


def _single_repair_target_file(row: dict[str, Any]) -> str | None:
    """Return the unique visible file path from public metadata/context without touching gold."""
    paths: list[str] = []
    seen: set[str] = set()
    selection = row.get("context_selection")
    if isinstance(selection, dict) and selection.get("gold_used") is not True:
        for window in selection.get("windows", ()):
            path = window.get("path") if isinstance(window, dict) else None
            if isinstance(path, str) and path and path not in seen:
                seen.add(path)
                paths.append(path)
    context = row.get("context")
    if not paths and isinstance(context, str):
        for path in _FILE_HEADER_RE.findall(context):
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
    if len(paths) != 1:
        return None
    candidate = paths[0].strip()
    if not candidate or candidate == context:
        return None
    lowered = candidate.lower()
    if any(phrase in lowered for phrase in _FORBIDDEN_PROMPT_PHRASES):
        return None
    return candidate


def _context_already_includes_instruction(context: str, instruction: str) -> bool:
    if not instruction:
        return False
    if context.startswith(instruction):
        return True
    if "\nQuestion:" in instruction:
        goal, _, question_part = instruction.partition("\nQuestion:")
        q_stripped = question_part.strip()
        if (
            goal
            and context.startswith(goal.strip())
            and q_stripped
            and (
                f"Question:\n{q_stripped}" in context
                or f"Question: {q_stripped}" in context
                or f"Question:{q_stripped}" in context
            )
        ):
            return True
    return False


_SECTION_LINE_NUMBER_RE = re.compile(r"^(\d+): ?(.*)$")


def _strip_rendered_line_numbers(text: str) -> str:
    """Remove only the outermost line-number prefix added inside ``### `` code sections."""
    if not text:
        return text
    out_lines: list[str] = []
    in_section = False
    for line in text.split("\n"):
        if line.startswith("### "):
            in_section = True
            out_lines.append(line)
            continue
        if in_section:
            match = _SECTION_LINE_NUMBER_RE.match(line)
            if match:
                out_lines.append(match.group(2))
                continue
            if line:
                in_section = False
        out_lines.append(line)
    return "\n".join(out_lines)


def split_context_for_mode(
    row: dict[str, Any],
    *,
    context_mode: str,
    repair_mode: str | None = None,
    local_window_lines: int = DEFAULT_LOCAL_CURSOR_WINDOW_LINES,
    number_lines: bool = True,
) -> tuple[str, str]:
    """Return ``(decoder_text, encoder_text)`` for ``full``, ``disjoint``, or ``local_only``."""
    if context_mode not in VALID_CONTEXT_MODES:
        raise ValueError(f"unsupported_context_mode:{context_mode}")

    instruction = _deduplicate_longcodeqa_headers(str(row["instruction"]))
    context = _deduplicate_longcodeqa_headers(
        _normalize_fim_cursor_marker(str(row.get("context") or ""))
    )
    is_repair = row.get("record_type") == "repository_patch_v1" or row.get("task_type") == "repair"
    if is_repair:
        eff_repair_mode = _resolve_repair_mode(row, repair_mode)
        repair_prefix = (
            _FULL_FUNCTION_REPAIR_PREFIX
            if eff_repair_mode == "full_function"
            else _REPAIR_INSTRUCTION_PREFIX
        )
        clean_ctx = context
        for pfx in (_REPAIR_INSTRUCTION_PREFIX, _FULL_FUNCTION_REPAIR_PREFIX):
            if clean_ctx.startswith(pfx):
                clean_ctx = clean_ctx[len(pfx) :].lstrip("\r\n")
                break
        split_row = {
            **row,
            "task_type": "repair",
            "instruction": instruction,
            "context": clean_ctx,
        }
        split = split_disjoint_context(split_row, local_window_lines=local_window_lines)
        full_ctx = (
            split.full_context if number_lines else _strip_rendered_line_numbers(split.full_context)
        )
        dec_local_ctx = (
            split.decoder_local_context
            if number_lines
            else _strip_rendered_line_numbers(split.decoder_local_context)
        )
        enc_body = (
            split.encoder_context
            if number_lines
            else _strip_rendered_line_numbers(split.encoder_context)
        )
        target_file = _single_repair_target_file(row) or split.cursor_path
        if context_mode == "full":
            header = f"{repair_prefix}\n\n{instruction}"
            dec_text = f"{header}\n\n{full_ctx}" if full_ctx else header
            return dec_text, ""
        banner = f"Target file: {target_file}\n\n" if target_file else ""
        header = f"{repair_prefix}\n\n{banner}{instruction}"
        dec_text = f"{header}\n\n{dec_local_ctx}" if dec_local_ctx else header
        if context_mode == "local_only":
            return dec_text, ""
        enc_text = enc_body or f"### Issue\n{instruction}"
        return dec_text, enc_text

    has_structured_context = (
        _parse_fim_sections(context) is not None
        or _parse_file_sections(context) is not None
        or "### Numbered source\n" in context
        or "[start of " in context
    )
    if has_structured_context:
        task_type = str(row.get("task_type") or "completion")
        split_row = {
            **row,
            "task_type": task_type,
            "instruction": instruction,
            "context": context,
        }
        split = split_disjoint_context(split_row, local_window_lines=local_window_lines)
        full_ctx = (
            split.full_context if number_lines else _strip_rendered_line_numbers(split.full_context)
        )
        dec_local_ctx = (
            split.decoder_local_context
            if number_lines
            else _strip_rendered_line_numbers(split.decoder_local_context)
        )
        enc_body = (
            split.encoder_context
            if number_lines
            else _strip_rendered_line_numbers(split.encoder_context)
        )
        if context_mode == "full":
            dec_text = f"{instruction}\n\n{full_ctx}" if full_ctx else instruction
            return dec_text, ""
        dec_text = f"{instruction}\n\n{dec_local_ctx}" if dec_local_ctx else instruction
        if context_mode == "local_only":
            return dec_text, ""
        enc_text = enc_body or instruction
        return dec_text, enc_text

    if context_mode == "full":
        if _context_already_includes_instruction(context, instruction):
            return context, ""
        return (f"{instruction}\n\n{context}" if context else instruction), ""

    if str(row.get("benchmark", "")).lower() == "gsm8k":
        question = row.get("question")
        if isinstance(question, str) and question.strip():
            dec_text = f"{instruction}\n\nQuestion: {question}"
        else:
            dec_text = instruction
    else:
        dec_text = instruction
    if context_mode == "local_only":
        return dec_text, ""
    return dec_text, context


def _has_disjoint_structured_context(raw_context: str) -> bool:
    return (
        _parse_fim_sections(raw_context) is not None
        or _parse_file_sections(raw_context) is not None
        or "### Numbered source\n" in raw_context
        or "[start of " in raw_context
    )


def encoder_context(
    row: dict[str, Any],
    *,
    paired: bool,
    context_mode: str | None = None,
    repair_mode: str | None = None,
    number_lines: bool = True,
) -> str:
    """Return the encoder input string for the requested mode without duplicating local context."""
    if context_mode is not None:
        return split_context_for_mode(
            row,
            context_mode=context_mode,
            repair_mode=repair_mode,
            number_lines=number_lines,
        )[1]
    if not paired:
        return ""
    raw_context = _deduplicate_longcodeqa_headers(
        _normalize_fim_cursor_marker(str(row.get("context") or ""))
    )
    if _has_disjoint_structured_context(raw_context):
        return split_context_for_mode(
            row,
            context_mode="disjoint",
            repair_mode=repair_mode,
            number_lines=number_lines,
        )[1]
    return raw_context


def decoder_prompt(
    row,
    *,
    paired,
    context_mode: str | None = None,
    repair_mode: str | None = None,
    number_lines: bool = True,
):
    """One prompt renderer for actual decoding and the shared output allowance."""
    if context_mode is not None:
        return split_context_for_mode(
            row,
            context_mode=context_mode,
            repair_mode=repair_mode,
            number_lines=number_lines,
        )[0]
    if not paired:
        return split_context_for_mode(
            row,
            context_mode="full",
            repair_mode=repair_mode,
            number_lines=number_lines,
        )[0]

    raw_context = _deduplicate_longcodeqa_headers(
        _normalize_fim_cursor_marker(str(row.get("context") or ""))
    )
    if _has_disjoint_structured_context(raw_context):
        return split_context_for_mode(
            row,
            context_mode="disjoint",
            repair_mode=repair_mode,
            number_lines=number_lines,
        )[0]

    instruction = _deduplicate_longcodeqa_headers(str(row["instruction"]))
    is_repair = row.get("record_type") == "repository_patch_v1" or row.get("task_type") == "repair"
    if is_repair:
        eff_repair_mode = _resolve_repair_mode(row, repair_mode)
        repair_prefix = (
            _FULL_FUNCTION_REPAIR_PREFIX
            if eff_repair_mode == "full_function"
            else _REPAIR_INSTRUCTION_PREFIX
        )
        target_file = _single_repair_target_file(row)
        if target_file:
            return f"{repair_prefix}\n\nTarget file: {target_file}\n\n{instruction}"
        return f"{repair_prefix}\n\n{instruction}"

    if str(row.get("benchmark", "")).lower() == "gsm8k":
        question = row.get("question")
        if isinstance(question, str) and question.strip():
            return f"{instruction}\n\nQuestion: {question}"
    return instruction


def _prompt_ids_with_metadata(
    row,
    spec,
    encoder,
    decoder,
    *,
    encoder_limit,
    context_mode: str | None = None,
    repair_mode: str | None = None,
    number_lines: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    selection = row.get("context_selection")
    if (
        row.get("role") != "eval"
        or (isinstance(selection, dict) and selection.get("gold_used") is True)
        or any(k in row for k in _GOLD_INPUT_KEYS)
    ):
        raise ValueError("benchmark_gold_in_model_input")
    if context_mode is not None and context_mode not in VALID_CONTEXT_MODES:
        raise ValueError(f"unsupported_context_mode:{context_mode}")
    if context_mode == "disjoint" and not spec.encoder:
        raise ValueError("disjoint_context_mode_requires_encoder")

    use_encoder = bool(spec.encoder) if context_mode is None else context_mode == "disjoint"
    enc = ()
    if use_encoder:
        if encoder is None:
            raise ValueError("benchmark_missing_encoder_tokenizer")
        enc_text = encoder_context(
            row,
            paired=True,
            context_mode=context_mode,
            repair_mode=repair_mode,
            number_lines=number_lines,
        )
        enc = tuple(encoder.encode(enc_text, add_special_tokens=True)) if enc_text else ()
        enc_bos = getattr(encoder, "bos_token_id", None)
        if enc and enc_bos is not None and enc[0] != enc_bos:
            enc = (enc_bos, *enc)
        if len(enc) > encoder_limit:
            if number_lines:
                return _prompt_ids_with_metadata(
                    row,
                    spec,
                    encoder,
                    decoder,
                    encoder_limit=encoder_limit,
                    context_mode=context_mode,
                    repair_mode=repair_mode,
                    number_lines=False,
                )
            raise ValueError("encoder_context_overflow")
    prompt = decoder_prompt(
        row,
        paired=use_encoder,
        context_mode=context_mode,
        repair_mode=repair_mode,
        number_lines=number_lines,
    )
    try:
        text = decoder.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = decoder.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    prefix = tuple(decoder.encode(text, add_special_tokens=False))
    if not prefix or (decoder.bos_token_id is not None and prefix[0] != decoder.bos_token_id):
        raise ValueError("benchmark_bos_policy")
    return enc, prefix, number_lines


def _prompt_ids_impl(
    row,
    spec,
    encoder,
    decoder,
    *,
    encoder_limit,
    context_mode: str | None = None,
    repair_mode: str | None = None,
    number_lines: bool = True,
):
    enc, prefix, _ = _prompt_ids_with_metadata(
        row,
        spec,
        encoder,
        decoder,
        encoder_limit=encoder_limit,
        context_mode=context_mode,
        repair_mode=repair_mode,
        number_lines=number_lines,
    )
    return enc, prefix


def prompt_ids(
    row,
    spec,
    encoder,
    decoder,
    *,
    encoder_limit,
    context_mode: str | None = None,
    repair_mode: str | None = None,
):
    return _prompt_ids_impl(
        row,
        spec,
        encoder,
        decoder,
        encoder_limit=encoder_limit,
        context_mode=context_mode,
        repair_mode=repair_mode,
    )


DELIM_OPEN = r"[*'\"`\(\{\[]*"
DELIM_CLOSE = r"[*'\"`\)\}\]]*"
LETTER = r"([A-Da-d])"
STRICT_DELIM_OPEN = r"[*'\"`\(\{\[]+"
STRICT_DELIM_CLOSE = r"[*'\"`\)\}\]]+"

# Safe letter token:
# 1. Bare letter MUST be uppercase [A-D] (so common English prose like "is a function" or "returns a list" is never matched as option A).
# 2. Lowercase [a-d] is accepted ONLY when explicitly enclosed in delimiters (e.g. (a), [b], `c`, **d**).
LETTER_TOKEN = (
    rf"(?:(?<![\w<])([A-D])(?![\w>])"
    rf"|{STRICT_DELIM_OPEN}([A-Da-d]){STRICT_DELIM_CLOSE}(?!\w))"
)
LETTER_TOKEN_NC = (
    rf"(?:(?<![\w<])[A-D](?![\w>])"
    rf"|{STRICT_DELIM_OPEN}[A-Da-d]{STRICT_DELIM_CLOSE}(?!\w))"
)

OPT_ADVERBS = r"(?:(?:clearly|definitely|actually|obviously|certainly|surely|indeed|simply|basically|likely|probably|\w+ly)\s+){0,2}"
CONJ_GUARD = (
    rf"(?!\s*{DELIM_CLOSE}\s+(?:and|or|nor|vs)\s+(?:option\s+|choice\s+)?{LETTER_TOKEN_NC})"
)


def _extract_match_letter(match: re.Match[str]) -> str | None:
    """Return uppercase letter from match. Bare letter must be uppercase; delimited may be any case."""
    grps = match.groups()
    if len(grps) >= 2:
        bare_grp, delim_grp = grps[-2], grps[-1]
        if bare_grp and bare_grp.isupper():
            return bare_grp
        if delim_grp:
            return delim_grp.upper()
    for grp in grps:
        if grp and grp.isupper():
            return grp
    return None


def fallback_letter_regex(text: str) -> str | None:
    """Resilient multi-stage regex extractor for reasoning preambles from models like E4B."""
    if not text:
        return None

    final = re.search(
        rf"\bfinal\s+answer(?:\s+is)?\s*:?\s*{OPT_ADVERBS}{LETTER_TOKEN}\s*\.?\s*\Z",
        text,
        re.I,
    )
    if final:
        return _extract_match_letter(final)

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
        rf"(?:ultimately\s+)?(?:option\s+|choice\s+){LETTER_TOKEN}\s+is\s+(?:{OPT_ADVERBS})?(?:the\s+)?correct{CONJ_GUARD}",
        rf"(?:ultimately\s+)?(?:option\s+|choice\s+)?{LETTER_TOKEN}\s+(?:is\s+(?:{OPT_ADVERBS})?(?:the\s+)?correct\s*(?:answer|choice|option)?){CONJ_GUARD}",
        rf"(?:final\s+answer|correct\s+(?:answer|option|choice|letter)|the\s+correct\s+answer)(?:\s+is)?(?:\s*:)?\s*{OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}",
        rf"(?:therefore|thus|hence|so|in\s+conclusion|finally)[,\s]+(?:(?:[^\w\n]*\w+){{0,6}}\s+)?(?:(?:the\s+answer|option|choice)\s+is\s+|it\s+is\s+)?{LETTER_TOKEN}\s*[.]?\s*$",
        rf"(?<!not\s)(?:the\s+answer|answer)\s+is(?:\s*:)?\s*(?!(?:(?:\w+\s+)?not\b)){OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}",
        rf"(?:it\s+is|it's)\s+{OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}(?:\.|\b|\s|$)",
        rf"(?:select|choose|pick)(?:\s+option|\s+choice|\s+letter)?(?:\s*:)?\s*{OPT_ADVERBS}{LETTER_TOKEN}{CONJ_GUARD}\s*[.]?\s*$",
    ]
    for pat in conclusive_patterns:
        matches = list(re.finditer(pat, text, re.I | re.M))
        if matches:
            ltr = _extract_match_letter(matches[-1])
            if ltr:
                return ltr

    # Allow bare lowercase letter only if it directly follows an explicit answer verb at line end
    terminal_verb = re.search(
        r"(?:answer(?:\s+is)?|select|choose|pick)\s*:?\s*([a-d])\s*[.)]?\s*$", text, re.I
    )
    if terminal_verb:
        return terminal_verb.group(1).upper()

    if has_ambiguity:
        return None

    # Priority 2: Structured prefix pattern: "Answer: B", "Option: B", "Choice: B", "Option (C)"
    prefix_pat = rf"(?:answer|option|choice)(?:\s+is)?(?:\s*:)?\s*(?!(?:(?:\w+\s+)?not\b)){OPT_ADVERBS}{LETTER_TOKEN}(?!\s*{DELIM_CLOSE}\s+(?:is\s+wrong|is\s+incorrect|is\s+not|is\s+suboptimal|is\s+invalid|is\s+false|is\s+not\s+right|and|or|nor|vs)){CONJ_GUARD}"
    prefix_matches = list(re.finditer(prefix_pat, text, re.I))
    if prefix_matches:
        ltr = _extract_match_letter(prefix_matches[-1])
        if ltr:
            return ltr

    # Priority 3: Single unique isolated letter candidate across text (guarded against negative statements)
    cands: set[str] = set()
    for m in re.finditer(LETTER_TOKEN, text):
        ltr = _extract_match_letter(m)
        if ltr:
            cands.add(ltr)
    if len(cands) == 1:
        cand = cands.pop()
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


def _strip_fim_markers(text: str) -> str:
    text = re.sub(r"^\s*(?:###\s*MISSING CODE|#\s*<MISSING_CODE>)[ \t]*\r?\n?", "", text, count=1)
    return re.split(r"(?:^|\r?\n)\s*###\s*SUFFIX\b", text, maxsplit=1)[0]


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text
    if stripped.startswith("```"):
        closed_match = re.match(
            r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)\r?\n?```[ \t]*(?:\r?\n.*)?\Z",
            stripped,
            re.DOTALL,
        )
        if closed_match:
            return closed_match.group(1)
        return re.sub(r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n?", "", stripped, count=1)
    preamble_guard = r"(?!(?:def|class|import|from|return|if|for|while|with|try|except)\b|@)"
    preamble_match = re.match(
        rf"{preamble_guard}[^\n`]{{0,160}}\r?\n\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?```[ \t]*(?:\r?\n.*)?\Z",
        stripped,
        re.DOTALL,
    )
    if preamble_match:
        return preamble_match.group(1)
    unclosed_preamble_match = re.match(
        rf"{preamble_guard}[^\n`]{{0,160}}\r?\n\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*)\Z",
        stripped,
        re.DOTALL,
    )
    if unclosed_preamble_match:
        return unclosed_preamble_match.group(1)
    return text


def _strip_echoed_suffix(text: str, suffix: str | None) -> str:
    if not text or not suffix:
        return text
    norm_text = text.replace("\r\n", "\n")
    norm_suffix = suffix.replace("\r\n", "\n")
    if not norm_suffix.strip():
        return text
    trimmed_text = norm_text.rstrip()
    candidates = [norm_suffix.rstrip()]
    stripped_leading = norm_suffix.lstrip("\n").rstrip()
    if stripped_leading and stripped_leading not in candidates:
        candidates.append(stripped_leading)
    for cand in candidates:
        if not cand:
            continue
        if trimmed_text.endswith(cand):
            return trimmed_text[: len(trimmed_text) - len(cand)].rstrip("\n")
        lines = cand.splitlines()
        for count in range(len(lines), 0, -1):
            prefix = "\n".join(lines[:count])
            if not prefix.strip():
                continue
            if len("".join(prefix.split())) < 8:
                continue
            if trimmed_text.endswith(prefix):
                before_idx = len(trimmed_text) - len(prefix)
                if (
                    before_idx == 0
                    or trimmed_text[before_idx - 1] == "\n"
                    or not norm_suffix.startswith("\n")
                ):
                    return trimmed_text[:before_idx].rstrip("\n")
    return text


def normalize_completion_prediction(prediction: str, *, suffix: str | None = None) -> str:
    """Strip markdown code fences and echoed FIM suffix markers/overlap from completions."""
    if not prediction:
        return prediction
    text = _strip_fim_markers(prediction)
    text = _strip_markdown_fences(text)
    text = _strip_fim_markers(text)
    return _strip_echoed_suffix(text, suffix)


def _sequence_and_identifier_metrics(prediction: str, reference: str) -> tuple[bool, float, float]:
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
    return clean_prediction == clean_reference, seq_sim, f1


def completion_metrics(prediction, reference, *, suffix: str | None = None):
    raw_em, raw_sim, raw_f1 = _sequence_and_identifier_metrics(prediction, reference)
    normalized_prediction = normalize_completion_prediction(prediction, suffix=suffix)
    norm_em, norm_sim, norm_f1 = _sequence_and_identifier_metrics(normalized_prediction, reference)
    return {
        "exact_match": raw_em,
        "sequence_similarity": raw_sim,
        "regex_identifier_set_f1": raw_f1,
        "regex_f1": raw_f1,
        "normalized_exact_match": norm_em,
        "normalized_sequence_similarity": norm_sim,
        "normalized_regex_identifier_set_f1": norm_f1,
        "normalized_regex_f1": norm_f1,
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


def _extract_full_python_code(completion: str) -> str:
    """Extract clean Python source from raw or markdown-fenced full-function completions."""
    if not isinstance(completion, str):
        return ""
    stripped = _strip_markdown_fences(completion).strip("\r\n")
    return stripped


def _merge_full_function_into_file(before: str, candidate: str) -> str:
    """Splice single-function completions into ``before`` when imports/helpers are omitted."""
    try:
        before_tree = ast.parse(before)
        cand_tree = ast.parse(candidate)
    except SyntaxError:
        return candidate
    if not cand_tree.body:
        return candidate
    before_funcs = {
        node.name: node
        for node in before_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    cand_funcs = [
        node for node in cand_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    # Only splice when the completion consists solely of existing top-level functions and
    # the original file has additional top-level statements (imports, constants, helpers).
    if (
        len(cand_funcs) == len(cand_tree.body)
        and len(before_tree.body) > len(cand_tree.body)
        and all(node.name in before_funcs for node in cand_funcs)
    ):
        before_lines = before.splitlines(keepends=True)
        cand_lines = candidate.splitlines(keepends=True)
        # Replace from bottom to top so earlier line offsets remain valid.
        replacements = sorted(
            ((before_funcs[node.name], node) for node in cand_funcs),
            key=lambda pair: pair[0].lineno,
            reverse=True,
        )
        for old_node, new_node in replacements:
            start = (
                old_node.decorator_list[0].lineno if old_node.decorator_list else old_node.lineno
            ) - 1
            end = old_node.end_lineno or old_node.lineno
            new_start = (
                new_node.decorator_list[0].lineno if new_node.decorator_list else new_node.lineno
            ) - 1
            new_end = new_node.end_lineno or new_node.lineno
            snippet = "".join(cand_lines[new_start:new_end])
            if not snippet.endswith("\n"):
                snippet += "\n"
            before_lines[start:end] = [snippet]
        return "".join(before_lines)
    return candidate if candidate.endswith("\n") else candidate + "\n"


def _score_full_function_repair(
    row: dict[str, Any],
    gold: dict[str, Any],
    snapshot: RepositorySnapshot,
    completion: str,
    *,
    python_executable: str,
    allow_execution: bool,
) -> dict[str, Any]:
    """Score full-function pass@1 completion by diffing against the target file and running tests."""
    from dataclasses import asdict

    from bifocalgemma.data.patches import _build_canonical_patch
    from bifocalgemma.evaluation.native_python import score_python_patch

    code = _extract_full_python_code(completion)
    if not code.strip():
        return {
            "instance_id": row["instance_id"],
            "status": "SYNTAX_ERROR",
            "patch_applied": False,
            "functional_pass": False,
            "pass_at_1": False,
            "correct": False,
            "repair_mode": "full_function",
            "syntax_valid": False,
            "reason": "empty_full_function_completion",
            "stdout": "",
            "stderr": "",
        }
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {
            "instance_id": row["instance_id"],
            "status": "SYNTAX_ERROR",
            "patch_applied": False,
            "functional_pass": False,
            "pass_at_1": False,
            "correct": False,
            "repair_mode": "full_function",
            "syntax_valid": False,
            "reason": f"syntax_error: {exc}",
            "stdout": "",
            "stderr": "",
        }

    target_path = _single_repair_target_file(row)
    file_map = {f.path: f.content for f in snapshot.files}
    if target_path not in file_map:
        if len(snapshot.files) == 1:
            target_path = snapshot.files[0].path
        elif "solution.py" in file_map:
            target_path = "solution.py"
        else:
            target_path = snapshot.files[0].path
    before = file_map[target_path]
    after = _merge_full_function_into_file(before, code)
    if after == before:
        if not allow_execution:
            return {
                "instance_id": row["instance_id"],
                "status": "STATIC_ONLY",
                "patch_applied": True,
                "functional_pass": None,
                "pass_at_1": None,
                "correct": None,
                "repair_mode": "full_function",
                "syntax_valid": True,
                "reason": "Native execution disabled.",
                "stdout": "",
                "stderr": "",
            }
        after = before + ("\n" if before and not before.endswith("\n") else "") + "# unchanged\n"
    after_snapshot = RepositorySnapshot(
        snapshot.repo,
        snapshot.base_commit,
        tuple(
            RepositoryFile(f.path, after if f.path == target_path else f.content)
            for f in snapshot.files
        ),
    )
    patch = _build_canonical_patch(snapshot, after_snapshot, {target_path})
    result = score_python_patch(
        row["instance_id"],
        snapshot,
        patch,
        gold.get("test_script") or "",
        python_executable=python_executable,
        expected_snapshot_sha256=gold["snapshot_sha256"],
        allow_code_execution=allow_execution,
        timeout_seconds=30,
        expected_python_version=gold.get("python_environment"),
    )
    if result.status == "INFRA_ERROR":
        raise RuntimeError("Functional benchmark infrastructure failed: " + result.reason)
    return {
        **asdict(result),
        "correct": result.functional_pass,
        "pass_at_1": result.functional_pass,
        "repair_mode": "full_function",
        "syntax_valid": True,
    }


def score_context_case(
    row,
    gold,
    completion,
    *,
    directory,
    python_executable,
    allow_execution,
    include_syntax=False,
    repair_mode=None,
):
    kind = row.get("task_type")
    if kind == "mcq":
        parsed = strict_letter(completion)
        resilient = extract_mcq_letter(completion)
        return {
            "correct": parsed == gold["answer"],
            "parsed_answer": parsed,
            "strict_correct": parsed == gold["answer"],
            "strict_parsed_answer": parsed,
            "resilient_correct": resilient == gold["answer"],
            "resilient_parsed_answer": resilient,
            "primary_metric": "mcq_strict_accuracy",
            "status": "INVALID_OUTPUT" if parsed is None else "SCORED",
        }
    if kind == "completion":
        suffix = None
        ctx = row.get("context")
        if isinstance(ctx, str):
            if "\n### SUFFIX\n" in ctx:
                _, _, suffix = ctx.rpartition("\n### SUFFIX\n")
            elif "\r\n### SUFFIX\r\n" in ctx:
                _, _, suffix = ctx.rpartition("\r\n### SUFFIX\r\n")
            elif "### SUFFIX" in ctx:
                _, _, suffix = ctx.rpartition("### SUFFIX")
                if suffix.startswith("\r\n"):
                    suffix = suffix[2:]
                elif suffix.startswith("\n"):
                    suffix = suffix[1:]
        metrics = completion_metrics(completion, gold["reference"], suffix=suffix)
        return {
            **metrics,
            "correct": metrics["exact_match"],
            "primary_continuous_score": metrics["sequence_similarity"],
            "status": "SCORED",
        }
    if str(row.get("benchmark", "")).lower() == "gsm8k" or "question" in row and "answer" in gold:
        from bifocalgemma.evaluation.gsm8k import (
            extract_release_answer,
            extract_resilient_answer,
        )

        parsed = extract_release_answer(completion)
        resilient = extract_resilient_answer(completion)
        return {
            "correct": parsed == gold["answer"],
            "parsed_answer": parsed,
            "strict_correct": parsed == gold["answer"],
            "strict_parsed_answer": parsed,
            "resilient_correct": resilient == gold["answer"],
            "resilient_parsed_answer": resilient,
            "primary_metric": "gsm8k_strict_accuracy",
            "status": "INVALID_OUTPUT" if parsed is None else "SCORED",
            "resilient_status": "INVALID_OUTPUT" if resilient is None else "SCORED",
        }
    if kind == "retrieval" or str(row.get("benchmark", "")).lower() == "reponiah":
        return score_repo_niah_case(row, gold, completion)
    if "snapshot_file" in gold:
        from dataclasses import asdict

        from bifocalgemma.data.patches import diagnose_patch_application
        from bifocalgemma.evaluation.native_python import score_python_patch
        from bifocalgemma.repository_study import snapshot_from_json

        path = (Path(directory) / gold["snapshot_file"]).resolve()
        if not path.is_relative_to(Path(directory).resolve()):
            raise ValueError("benchmark_snapshot_path")
        snapshot = snapshot_from_json(path)
        eff_repair_mode = _resolve_repair_mode(row, repair_mode)
        if eff_repair_mode == "full_function":
            return _score_full_function_repair(
                row,
                gold,
                snapshot,
                completion,
                python_executable=python_executable,
                allow_execution=allow_execution,
            )

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
        diag = diagnose_patch_application(
            snapshot,
            completion,
            default_path=_single_repair_target_file(row),
        )
        recount_pass: bool | None
        if result.patch_applied:
            recount_pass = result.functional_pass
        elif diag.recount_applied and diag.normalized_patch:
            if allow_execution:
                recount_res = score_python_patch(
                    row["instance_id"],
                    snapshot,
                    diag.normalized_patch,
                    gold.get("test_script") or "",
                    python_executable=python_executable,
                    expected_snapshot_sha256=gold["snapshot_sha256"],
                    allow_code_execution=True,
                    timeout_seconds=30,
                    expected_python_version=gold.get("python_environment"),
                )
                recount_pass = recount_res.functional_pass
            else:
                recount_pass = None
        else:
            recount_pass = False
        record: dict[str, Any] = {
            **asdict(result),
            "correct": result.functional_pass,
            "repair_mode": "unified_diff",
            "recount_patch_applied": diag.recount_applied,
            "recount_diagnostic_mode": diag.diagnostic_mode,
            "recount_functional_pass": recount_pass,
        }
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
    context_mode=None,
    repair_mode=None,
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
    effective_protocol = build_effective_benchmark_spec(
        spec=spec,
        model=model,
        public_rows=public,
        gold_rows=hidden,
        collection=root.name,
        encoder_limit=encoder_limit,
        decoder_limit=decoder_limit,
        suite_cap=max_new_tokens,
        use_cache=use_cache,
        allow_execution=allow_execution,
        context_mode=context_mode,
        repair_mode=repair_mode,
    )
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
            "effective_protocol": effective_protocol,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "evaluation_state.json"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("Benchmark/checkpoint/protocol changed; choose a new output.")
    else:
        if (output / "predictions.jsonl").exists():
            raise ValueError("Prediction journal has no protocol identity; choose a new output.")
        atomic_json(
            state_path,
            {"signature": signature, "total": len(ids), "effective_protocol": effective_protocol},
        )
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
    if any(record.get("protocol_signature") != signature for record in completed):
        raise ValueError("Prediction journal protocol signature mismatch.")
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
        enc, prefix, used_number_lines = _prompt_ids_with_metadata(
            row,
            spec,
            encoder,
            decoder,
            encoder_limit=cap,
            context_mode=context_mode,
            repair_mode=repair_mode,
        )

        def _paired_budget_for(*, number_lines: bool) -> tuple[int, int]:
            # Same output allowance for pairs and baselines: reserve the paired IT prefix.
            paired_budget_mode = "disjoint" if context_mode in ("disjoint", "local_only") else None
            paired_prompt = decoder_prompt(
                row,
                paired=True,
                context_mode=paired_budget_mode,
                repair_mode=repair_mode,
                number_lines=number_lines,
            )
            try:
                paired_rendered = decoder.apply_chat_template(
                    [{"role": "user", "content": paired_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                paired_rendered = decoder.apply_chat_template(
                    [{"role": "user", "content": paired_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            paired_size = len(decoder.encode(paired_rendered, add_special_tokens=False))
            budget = effective_generation_budget(
                max_new_tokens,
                row_cap=row.get("max_new_tokens"),
                paired_prefix_tokens=paired_size,
                decoder_limit=decoder_limit,
            )
            return paired_size, budget

        paired_prefix_size, new_tokens = _paired_budget_for(number_lines=used_number_lines)
        uses_encoder = bool(spec.encoder) if context_mode is None else context_mode == "disjoint"
        total = decoder_limit if uses_encoder else cap + decoder_limit + 512
        if len(prefix) + new_tokens > total and not uses_encoder and used_number_lines:
            enc, prefix, used_number_lines = _prompt_ids_with_metadata(
                row,
                spec,
                encoder,
                decoder,
                encoder_limit=cap,
                context_mode=context_mode,
                repair_mode=repair_mode,
                number_lines=False,
            )
            paired_prefix_size, new_tokens = _paired_budget_for(number_lines=False)
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
            repair_mode=repair_mode,
        )
        record = {
            "protocol_signature": signature,
            "instance_id": row["instance_id"],
            "completion": completion,
            "stop_reason": generated.stop_reason,
            "encoder_tokens": len(enc),
            "decoder_prefix_tokens": len(prefix),
            "paired_decoder_prefix_tokens": paired_prefix_size,
            "generation_max_new_tokens": new_tokens,
            "generated_tokens": len(tokens),
            "number_lines": used_number_lines,
            "overflow_unnumbered_fallback": not used_number_lines,
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
        "unnumbered_fallback_count": sum(
            bool(r.get("overflow_unnumbered_fallback")) for r in completed
        ),
        "segment_seconds": time.perf_counter() - started,
        "timing": clock.summary(),
        "official_protocol": False,
        "protocol_version": protocol_version,
        "reset_id": BENCHMARK_RESET_ID,
        "effective_protocol": effective_protocol,
        "primary_metric": "task_specific_accuracy",
        "accuracy_alias": "task_specific_accuracy",
        "protocol": ("greedy_cached" if use_cache else "greedy_no_temporal_cache")
        + "; fixed eligible cohort; unnumbered line-prefix fallback on overflow (no token truncation)",
        "resources": resource_snapshot(next(model.parameters()).device),
    }
    report.update(mcq_report_metrics(completed))
    report.update(gsm8k_report_metrics(completed))
    report.update(repair_report_metrics(completed))
    for key in (
        "exact_match",
        "sequence_similarity",
        "regex_identifier_set_f1",
        "regex_f1",
        "normalized_exact_match",
        "normalized_sequence_similarity",
        "normalized_regex_identifier_set_f1",
        "normalized_regex_f1",
        "primary_continuous_score",
    ):
        if completed and key in completed[0]:
            report[key] = sum(float(r[key]) for r in completed) / len(ids)
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
    norm_sim = (
        sum(
            float(r.get("normalized_sequence_similarity", r.get("sequence_similarity", 0.0)))
            for r in records
        )
        / n
    )
    norm_f1 = (
        sum(
            float(
                r.get(
                    "normalized_regex_f1",
                    r.get("normalized_regex_identifier_set_f1", r.get("regex_f1", 0.0)),
                )
            )
            for r in records
        )
        / n
    )
    norm_em = (
        sum(
            float(bool(r.get("normalized_exact_match", r.get("exact_match", False))))
            for r in records
        )
        / n
    )
    return {
        "sequence_similarity": round(seq_sim, 4),
        "regex_f1": round(f1, 4),
        "exact_match": round(em, 4),
        "normalized_sequence_similarity": round(norm_sim, 4),
        "normalized_regex_f1": round(norm_f1, 4),
        "normalized_exact_match": round(norm_em, 4),
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


def _count_niah_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _encoded_context_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return max(1, len(text) // 4)
    enc = tuple(tokenizer.encode(text, add_special_tokens=True))
    bos = getattr(tokenizer, "bos_token_id", None)
    if enc and bos is not None and enc[0] != bos:
        return len(enc) + 1
    return len(enc)


def _render_niah_prompt_tokens(
    instruction: str,
    haystack: str,
    tokenizer,
    *,
    enable_thinking: bool | None = False,
) -> int:
    user_content = f"{instruction}\n\n{haystack}" if haystack else f"{instruction}\n\n"
    if tokenizer is None:
        return max(1, len(user_content) // 4)
    rendered = user_content
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if has_template or callable(getattr(tokenizer, "apply_chat_template", None)):
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}], **kwargs
            )
        except TypeError:
            kwargs.pop("enable_thinking", None)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}], **kwargs
            )
        except ValueError:
            rendered = user_content
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def generate_repo_niah_haystack(
    target_tokens: int,
    needle_line: str,
    depth_ratio: float,
    tokenizer=None,
    *,
    reserved_tokens: int = 0,
) -> str:
    """Construct a synthetic repository haystack <= target_tokens with needle embedded at depth_ratio."""
    needle_block = f"\n# --- System Security Configuration ---\n{needle_line}\n# ---------------------------------------\n"
    needle_tokens = _count_niah_tokens(needle_block, tokenizer)
    special_overhead = (
        len(tokenizer.encode("", add_special_tokens=True)) if tokenizer is not None else 0
    )
    if tokenizer is not None and getattr(tokenizer, "bos_token_id", None) is not None:
        special_overhead = max(special_overhead, 1)
    eff_reserved = max(0, reserved_tokens)
    distractor_budget = max(0, target_tokens - needle_tokens - special_overhead - eff_reserved)
    sep_tokens = max(1, _count_niah_tokens("\n", tokenizer)) if tokenizer is not None else 1

    blocks: list[tuple[str, int]] = []
    current_tokens = 0
    idx = 0
    while True:
        block = make_distractor_python_chunk(idx)
        tok_count = _count_niah_tokens(block, tokenizer)
        sep = sep_tokens if blocks else 0
        if current_tokens + sep + tok_count > distractor_budget:
            break
        blocks.append((block, tok_count))
        current_tokens += sep + tok_count
        idx += 1

    total_tok = sum(tc for _, tc in blocks)
    target_needle_pos = int(depth_ratio * total_tok)

    accum = 0
    insert_idx = len(blocks)
    for i, (_, tc) in enumerate(blocks):
        if accum + tc > target_needle_pos:
            insert_idx = i
            break
        accum += tc

    left_blocks = [b[0] for b in blocks[:insert_idx]]
    right_blocks = [b[0] for b in blocks[insert_idx:]]
    haystack = "\n".join([*left_blocks, needle_block, *right_blocks])

    if tokenizer is not None:
        while (left_blocks or right_blocks) and (
            _encoded_context_tokens(haystack, tokenizer) + eff_reserved > target_tokens
        ):
            if len(right_blocks) > len(left_blocks) or not left_blocks:
                right_blocks.pop()
            else:
                left_blocks.pop(0)
            haystack = "\n".join([*left_blocks, needle_block, *right_blocks])
    return haystack


def build_repo_niah_case(
    context_tokens: int,
    depth_ratio: float,
    seed: int = 42,
    *,
    key_name: str = "SECRET_KEY",
    needle_value: str | None = None,
    tokenizer=None,
    reserved_tokens: int | None = None,
    max_prompt_tokens: int | None = None,
    enable_thinking: bool | None = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a single RepoNIAH benchmark case (public input row, hidden gold spec)."""
    if needle_value is None:
        needle_value = f"alpha_bravo_{seed}_{hashlib.sha256(str(seed).encode()).hexdigest()[:8]}"
    needle_line = f'{key_name} = "{needle_value}"'
    instruction = f"What is the value of {key_name} in the codebase? Return only the value."
    eff_target = (
        min(context_tokens, max_prompt_tokens) if max_prompt_tokens is not None else context_tokens
    )
    if reserved_tokens is None:
        if tokenizer is not None:
            prompt_overhead = _render_niah_prompt_tokens(
                instruction, "", tokenizer, enable_thinking=enable_thinking
            )
            eff_reserved = prompt_overhead + 4
        else:
            eff_reserved = max(16, len(instruction) // 3)
    else:
        eff_reserved = max(0, reserved_tokens)

    haystack = generate_repo_niah_haystack(
        eff_target,
        needle_line,
        depth_ratio,
        tokenizer=tokenizer,
        reserved_tokens=eff_reserved,
    )
    if tokenizer is not None:
        for _ in range(4):
            prompt_toks = _render_niah_prompt_tokens(
                instruction, haystack, tokenizer, enable_thinking=enable_thinking
            )
            enc_toks = _encoded_context_tokens(haystack, tokenizer)
            overflow = max(prompt_toks - eff_target, enc_toks - context_tokens)
            if overflow <= 0:
                break
            eff_reserved += overflow + 4
            haystack = generate_repo_niah_haystack(
                eff_target,
                needle_line,
                depth_ratio,
                tokenizer=tokenizer,
                reserved_tokens=eff_reserved,
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
        for pred_rec in predictions:
            f.write(json.dumps(pred_rec, ensure_ascii=False) + "\n")
    return summary


evaluate_repo_niah = run_repo_niah_eval
