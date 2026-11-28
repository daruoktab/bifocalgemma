"""Target-only SFT for explicit repository repair, localization, completion, and v4 disjoint context tasks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bifocalgemma.contracts import TokenizedRepairExample

TASK_INSTRUCTIONS = {
    "repair": "Return only a unified diff fulfilling the instruction.",
    "localization": "Return only a JSON list of repository file paths relevant to fixing the issue.",
    "completion": "Return only the missing Python code, preserving its indentation.",
    "source_extraction": "Return only the exact requested source text.",
    "source_mcq": "Return only one option letter: A, B, C, or D.",
    "source_reconstruction": "Return only the missing source line, preserving indentation.",
    "qa": "Answer the question concisely and accurately based on the provided context.",
    "long_context_qa": "Answer the question concisely and accurately based on the provided context.",
    "mcq": "Answer the code comprehension question. Return only A, B, C, or D.",
    "niah": "Answer the question concisely and accurately based on the provided context.",
    "retrieval": "Answer the question concisely and accurately based on the provided context.",
    "math": "Solve the math problem step by step and provide the final answer clearly.",
    "retention": "Follow the instruction accurately and adhere strictly to all formatting constraints.",
    "instruction": "Follow the instruction accurately and adhere strictly to all formatting constraints.",
}

FORBIDDEN_SELECTION_SOURCES = frozenset(
    {
        "gold",
        "gold_patch",
        "gold_target",
        "oracle",
        "target",
        "target_patch",
        "target_text",
        "groundtruth",
        "ground_truth",
        "solution",
        "canonical_solution",
    }
)

PERMITTED_V4_CODE_LICENSES = frozenset(
    {
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "cc0-1.0",
        "unlicense",
    }
)
PERMITTED_V4_DATASET_LICENSES = PERMITTED_V4_CODE_LICENSES | frozenset({"cc-by-4.0", "odc-by"})

BLOCKED_V4_DATASETS = frozenset(
    {
        "microsoft/NextCoderDataset",
        "KodCode/KodCode-V1",
        "KodCode/KodCode-V1-SFT-R1",
        "tianyang/repobench_python_v1.1",
        "Steefano/LCB",
        "lailaelkoussy/swe-qa",
        "swe-qa/SWE-QA-Benchmark",
        "zai-org/LongBench-v2",
        "Qurrent/RES-Q",
        "nuprl/canitedit",
        "bigcode/humanevalpack",
        "evalplus/repoqa",
    }
)

V4_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # Repair & Localization (~35%)
    "swe_fixer": {
        "dataset_id": "internlm/SWE-Fixer-Train-110K",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "swe_rebench_v2": {
        "dataset_id": "nebius/SWE-rebench-V2",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "cc-by-4.0",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "swe_smith_py": {
        "dataset_id": "SWE-bench/SWE-smith-py",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "r2e_gym": {
        "dataset_id": "R2E-Gym/R2E-Gym-Subset",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "swe_gym": {
        "dataset_id": "SWE-Gym/SWE-Gym",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "opc_sft_stage1": {
        "dataset_id": "OpenCoder-LLM/opc-sft-stage1",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "opc_sft_stage2": {
        "dataset_id": "OpenCoder-LLM/opc-sft-stage2",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "opencodeinstruct": {
        "dataset_id": "nvidia/OpenCodeInstruct",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "cc-by-4.0",
        "requires_row_license_check": False,
        "min_test_score": 0.9,
        "quarantine_repos": True,
    },
    "self_oss_instruct": {
        "dataset_id": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "swe_smith_trajectories": {
        "dataset_id": "SWE-bench/SWE-smith-trajectories",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "swe_agent_trajectories": {
        "dataset_id": "nebius/SWE-agent-trajectories",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "cc-by-4.0",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    "editpackft": {
        "dataset_id": "nuprl/EditPackFT",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": True,
        "quarantine_repos": True,
        "capped": True,
    },
    "commitpackft": {
        "dataset_id": "bigcode/commitpackft",
        "bucket": "repair_loc",
        "default_task_type": "repair",
        "dataset_license": "mit",
        "requires_row_license_check": True,
        "quarantine_repos": True,
        "capped": True,
    },
    # FIM / Completion (~25%)
    "fim_midtraining": {
        "dataset_id": "TIGER-Lab/FIM-Midtraining-400K",
        "bucket": "fim_completion",
        "default_task_type": "completion",
        "dataset_license": "multiple-oss-licenses",
        "requires_row_license_check": True,
        "quarantine_repos": True,
    },
    "high_coder": {
        "dataset_id": "Crownelius/High-Coder-SFT-Medium",
        "bucket": "fim_completion",
        "default_task_type": "completion",
        "dataset_license": "mit",
        "requires_row_license_check": False,
        "quarantine_repos": True,
    },
    # Long-Context QA / Comprehension (~25%)
    "longalign": {
        "dataset_id": "zai-org/LongAlign-10k",
        "bucket": "long_context_qa",
        "default_task_type": "long_context_qa",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "max_tokens": 32768,
        "quarantine_repos": False,
    },
    "smoltalk_longalign": {
        "dataset_id": "HuggingFaceTB/smoltalk",
        "config": "longalign",
        "bucket": "long_context_qa",
        "default_task_type": "long_context_qa",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "max_tokens": 32768,
        "quarantine_repos": False,
    },
    "ultradata_agent": {
        "dataset_id": "openbmb/UltraData-SFT-Agent-2609",
        "bucket": "long_context_qa",
        "default_task_type": "long_context_qa",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "max_tokens": 32768,
        "quarantine_repos": False,
    },
    # Retention (~15%)
    "smoltalk_constraints": {
        "dataset_id": "HuggingFaceTB/smoltalk",
        "config": "smol-constraints",
        "bucket": "retention",
        "default_task_type": "retention",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "quarantine_repos": False,
    },
    "smoltalk_apigen": {
        "dataset_id": "HuggingFaceTB/smoltalk",
        "config": "apigen-80k",
        "bucket": "retention",
        "default_task_type": "retention",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "quarantine_repos": False,
    },
    "smoltalk_metamathqa": {
        "dataset_id": "HuggingFaceTB/smoltalk",
        "config": "metamathqa-50k",
        "bucket": "retention",
        "default_task_type": "retention",
        "dataset_license": "apache-2.0",
        "requires_row_license_check": False,
        "dedup_gsm8k": True,
        "quarantine_repos": False,
    },
    "openmathinstruct2": {
        "dataset_id": "nvidia/OpenMathInstruct-2",
        "bucket": "retention",
        "default_task_type": "retention",
        "dataset_license": "cc-by-4.0",
        "requires_row_license_check": False,
        "dedup_gsm8k": True,
        "quarantine_repos": False,
    },
}


def is_permissive_license(license_name: str | None, *, code_only: bool = True) -> bool:
    """Validate an SPDX or normalized license identifier against permissive allowlists."""
    if not isinstance(license_name, str) or not license_name.strip():
        return False
    normalized = license_name.strip().lower().replace("_", "-")
    if any(
        bad in normalized
        for bad in ("-nc", "noncommercial", "non-commercial", "-nd", "gpl", "agpl", "proprietary", "unknown", "sspl")
    ):
        return False
    parts = [
        p.strip("() \t")
        for p in re.split(r"\s+and\s+|\s+or\s+|,", normalized)
        if p.strip("() \t")
    ]
    allowed = PERMITTED_V4_CODE_LICENSES if code_only else PERMITTED_V4_DATASET_LICENSES
    return bool(parts) and all(part in allowed for part in parts)


_NUMBERED_LINE_RE = re.compile(r"^(\d+):\s?(.*)$")
_FILE_SECTION_HEADER_RE = re.compile(
    r"^### File:\s*([^\n(]+?)(?:\s*\((?:lines\s+(\d+)-(\d+)[^)]*|[^)]*)\))?[ \t]*$",
    re.MULTILINE,
)
_FIM_CURSOR_MARKER_RE = re.compile(
    r"^[ \t]*(?:\d+:\s*)?(?:###\s*MISSING CODE|#\s*<MISSING_CODE>)[ \t]*$",
    re.MULTILINE,
)


def _is_numbered_block(lines: list[str]) -> bool:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return False
    return all(_NUMBERED_LINE_RE.match(line) for line in non_empty[: min(5, len(non_empty))])


def format_numbered_code(
    code: str,
    *,
    start_line: int = 1,
    skip_if_numbered: bool = False,
) -> str:
    """Format code lines with explicit 1-based (or start_line-based) `N: ...` line numbers."""
    if type(start_line) is not int or start_line < 1:
        raise ValueError("start_line must be an integer >= 1")
    if not code:
        return ""
    lines = code.splitlines()
    if skip_if_numbered and _is_numbered_block(lines):
        return "\n".join(lines)
    return "\n".join(f"{start_line + idx}: {line}" for idx, line in enumerate(lines))


@dataclass(frozen=True)
class DisjointContextSplit:
    """Strictly disjoint encoder/decoder context partitions with full-context parity."""

    encoder_context: str
    decoder_local_context: str
    full_context: str
    task_type: str
    cursor_path: str | None = None
    local_line_range: tuple[int, int] | None = None
    dropped_block_count: int = 0


def _strip_leading_instruction(context: str, instruction: str) -> str:
    stripped = context
    for prefix in (
        f"### Instruction\n{instruction}",
        f"### Issue\n{instruction}",
        instruction,
    ):
        if prefix and stripped.startswith(prefix):
            remainder = stripped[len(prefix) :]
            if not remainder or remainder.startswith("\n"):
                return remainder.lstrip("\n")
    return stripped


def _extract_numbered_entries(lines: list[str], default_start: int) -> list[tuple[int, str]]:
    if _is_numbered_block(lines):
        entries: list[tuple[int, str]] = []
        next_num = default_start
        for line in lines:
            match = _NUMBERED_LINE_RE.match(line)
            if match:
                num = int(match.group(1))
                entries.append((num, match.group(2)))
                next_num = num + 1
            else:
                entries.append((next_num, line))
                next_num += 1
        return entries
    return [(default_start + idx, line) for idx, line in enumerate(lines)]


def _render_entries(entries: list[tuple[int, str]]) -> str:
    return "\n".join(f"{num}: {text}" for num, text in entries)


def _split_crosscodelong_context(
    body: str,
    task_type: str,
    local_window_lines: int,
) -> DisjointContextSplit:
    cross_text = ""
    rest = body
    if "### CROSS-FILE CONTEXT\n" in rest:
        _, _, after_cross = rest.partition("### CROSS-FILE CONTEXT\n")
        if "\n### PREFIX\n" in after_cross:
            cross_text, _, rest = after_cross.partition("\n### PREFIX\n")
            rest = "### PREFIX\n" + rest
        elif "### PREFIX\n" in after_cross:
            cross_text, _, rest = after_cross.partition("### PREFIX\n")
            rest = "### PREFIX\n" + rest

    prefix_text = ""
    suffix_text = ""
    if "### PREFIX\n" in rest:
        _, _, after_pre = rest.partition("### PREFIX\n")
        if "\n### SUFFIX\n" in after_pre:
            prefix_raw, _, suffix_text = after_pre.partition("\n### SUFFIX\n")
        elif after_pre.startswith("### SUFFIX\n"):
            prefix_raw, suffix_text = "", after_pre[len("### SUFFIX\n") :]
        elif "### SUFFIX\n" in after_pre:
            prefix_raw, _, suffix_text = after_pre.partition("### SUFFIX\n")
        else:
            prefix_raw, suffix_text = after_pre, ""
        prefix_text = _FIM_CURSOR_MARKER_RE.sub("", prefix_raw).rstrip("\n")
        suffix_text = _FIM_CURSOR_MARKER_RE.sub("", suffix_text).rstrip("\n")

    pre_lines = prefix_text.splitlines()
    suf_lines = suffix_text.splitlines()
    pre_entries = _extract_numbered_entries(pre_lines, 1)
    cursor_num = (pre_entries[-1][0] + 1) if pre_entries else 1
    suf_entries = _extract_numbered_entries(suf_lines, cursor_num + 1)

    split_pre_idx = max(0, len(pre_entries) - local_window_lines)
    distant_pre = pre_entries[:split_pre_idx]
    local_pre = pre_entries[split_pre_idx:]
    local_suf = suf_entries[:local_window_lines]
    distant_suf = suf_entries[local_window_lines:]

    encoder_parts: list[str] = []
    if cross_text.strip():
        encoder_parts.append(
            "### Cross-File Context\n" + format_numbered_code(cross_text.strip("\n"), start_line=1, skip_if_numbered=True)
        )
    if distant_pre:
        encoder_parts.append(
            f"### Target File Prefix (lines {distant_pre[0][0]}-{distant_pre[-1][0]})\n"
            + _render_entries(distant_pre)
        )
    if distant_suf:
        encoder_parts.append(
            f"### Target File Suffix (lines {distant_suf[0][0]}-{distant_suf[-1][0]})\n"
            + _render_entries(distant_suf)
        )

    local_entries = [*local_pre, (cursor_num, "# <MISSING_CODE>"), *local_suf]
    local_range = (local_entries[0][0], local_entries[-1][0])
    decoder_local = (
        f"### Local Cursor (lines {local_range[0]}-{local_range[1]})\n"
        + _render_entries(local_entries)
    )
    encoder_context = "\n\n".join(encoder_parts)
    full_context = "\n\n".join(part for part in (encoder_context, decoder_local) if part)
    return DisjointContextSplit(
        encoder_context=encoder_context,
        decoder_local_context=decoder_local,
        full_context=full_context,
        task_type=task_type,
        cursor_path=None,
        local_line_range=local_range,
    )


def _anchor_index_for_repair(
    entries: list[tuple[int, str]],
    instruction: str,
) -> int:
    terms = {
        tok.lower()
        for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]{2,}", instruction)
        if tok.lower() not in {"the", "and", "for", "with", "that", "this", "from", "fix", "bug", "error", "code", "file", "function"}
    }
    if not terms or not entries:
        return 0
    best_idx = 0
    best_score = 0
    for idx, (_, line) in enumerate(entries):
        lowered = line.lower()
        score = sum(1 for t in terms if t in lowered)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _parse_file_sections(body: str) -> tuple[str, list[tuple[str, int, str]]]:
    matches = list(_FILE_SECTION_HEADER_RE.finditer(body))
    if not matches:
        return "", []
    preamble = body[: matches[0].start()].strip("\n")
    sections: list[tuple[str, int, str]] = []
    for idx, match in enumerate(matches):
        path = match.group(1).strip()
        start_line = int(match.group(2)) if match.group(2) else 1
        content_start = match.end()
        if content_start < len(body) and body[content_start] == "\n":
            content_start += 1
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        code = body[content_start:content_end].rstrip("\n")
        sections.append((path, start_line, code))
    return preamble, sections


def split_disjoint_context(
    row: Mapping[str, Any],
    *,
    local_window_lines: int = 48,
) -> DisjointContextSplit:
    """Split a task row into strictly disjoint encoder context and decoder local cursor window.

    Security & parity contract:
    - Never reads target/gold fields (`target_text`, `target_patch`, `gold_patch`, etc.).
    - Guarantees zero line duplication between `encoder_context` and `decoder_local_context`.
    - Guarantees `full_context` contains the exact union of `encoder_context` and
      `decoder_local_context` with identical `1: ...` line numbering.
    """
    if type(local_window_lines) is not int or local_window_lines < 1:
        raise ValueError("local_window_lines must be a positive integer")

    task_type = str(row.get("task_type") or "repair")
    instruction = str(row.get("instruction") or "").replace("\r\n", "\n")
    raw_context = str(row.get("context") or "").replace("\r\n", "\n")
    body = _strip_leading_instruction(raw_context, instruction)

    # 1. Grounded tasks (`### Numbered source`)
    if "### Numbered source\n" in body or body.startswith("### Numbered source\n"):
        _, _, numbered_body = body.partition("### Numbered source\n")
        entries = _extract_numbered_entries(numbered_body.splitlines(), 1)
        if task_type == "source_reconstruction":
            cursor_idx = next(
                (i for i, (_, text) in enumerate(entries) if "# <MISSING_CODE>" in text),
                None,
            )
            if cursor_idx is not None:
                local_start = max(0, cursor_idx - local_window_lines)
                local_end = min(len(entries), cursor_idx + local_window_lines + 1)
                distant_before = entries[:local_start]
                local_entries = entries[local_start:local_end]
                distant_after = entries[local_end:]
                enc_parts: list[str] = []
                if distant_before:
                    enc_parts.append(
                        f"### Numbered source (lines {distant_before[0][0]}-{distant_before[-1][0]})\n"
                        + _render_entries(distant_before)
                    )
                if distant_after:
                    enc_parts.append(
                        f"### Numbered source (lines {distant_after[0][0]}-{distant_after[-1][0]})\n"
                        + _render_entries(distant_after)
                    )
                local_range = (local_entries[0][0], local_entries[-1][0])
                dec_local = (
                    f"### Local Cursor (lines {local_range[0]}-{local_range[1]})\n"
                    + _render_entries(local_entries)
                )
                enc_ctx = "\n\n".join(enc_parts)
                full_ctx = "\n\n".join(p for p in (enc_ctx, dec_local) if p)
                return DisjointContextSplit(
                    encoder_context=enc_ctx,
                    decoder_local_context=dec_local,
                    full_context=full_ctx,
                    task_type=task_type,
                    local_line_range=local_range,
                )
        enc_ctx = "### Numbered source\n" + _render_entries(entries) if entries else ""
        return DisjointContextSplit(
            encoder_context=enc_ctx,
            decoder_local_context="",
            full_context=enc_ctx,
            task_type=task_type,
        )

    # 2. CrossCodeLongEval FIM layout (`### PREFIX` / `### SUFFIX`)
    if "### PREFIX\n" in body and (
        "### SUFFIX\n" in body or "### MISSING CODE" in body or "# <MISSING_CODE>" in body
    ):
        return _split_crosscodelong_context(body, task_type, local_window_lines)

    # 3. Retention / pure instruction tasks without code context
    if task_type in {"retention", "instruction"} and "### File:" not in body:
        cleaned = body.strip("\n")
        if not cleaned or cleaned == instruction.strip("\n"):
            return DisjointContextSplit("", "", "", task_type=task_type)
        enc_ctx = format_numbered_code(cleaned, start_line=1, skip_if_numbered=True)
        return DisjointContextSplit(enc_ctx, "", enc_ctx, task_type=task_type)

    # 4. File-section layout (`### File: ...`) or LongCodeQA / raw single-file context
    preamble, sections = _parse_file_sections(body)
    if not sections:
        # Check for LongCodeQA `[start of path]` blocks
        lqa_matches = list(
            re.finditer(r"\[start of ([^\]]+)\]\n(.*?)\n\[end of [^\]]+\]", body, re.S)
        )
        if lqa_matches:
            sections = [
                (m.group(1).strip(), 1, m.group(2).rstrip("\n")) for m in lqa_matches
            ]
            preamble = ""
        elif body.strip():
            fallback_path = row.get("path")
            if not isinstance(fallback_path, str) or not fallback_path or fallback_path == "repository_context":
                fallback_path = "target.py" if task_type in {"repair", "completion", "localization"} else "context.txt"
            sections = [(fallback_path, 1, body.rstrip("\n"))]
            preamble = ""

    if not sections:
        enc_ctx = preamble.strip("\n")
        return DisjointContextSplit(enc_ctx, "", enc_ctx, task_type=task_type)

    # QA / MCQ / source_extraction / source_mcq: all numbered context in encoder, 0 local window in decoder
    if task_type in {"qa", "long_context_qa", "mcq", "source_extraction", "source_mcq"}:
        enc_parts = []
        if preamble.strip():
            enc_parts.append(preamble.strip("\n"))
        for path, base_start, code in sections:
            entries = _extract_numbered_entries(code.splitlines(), base_start)
            if not entries:
                continue
            enc_parts.append(
                f"### File: {path} (lines {entries[0][0]}-{entries[-1][0]})\n"
                + _render_entries(entries)
            )
        enc_ctx = "\n\n".join(enc_parts)
        return DisjointContextSplit(
            encoder_context=enc_ctx,
            decoder_local_context="",
            full_context=enc_ctx,
            task_type=task_type,
        )

    # Completion / Repair / Localization: select ONE cursor section and split local window vs disjoint remainder
    section_entries = [
        (path, _extract_numbered_entries(code.splitlines(), base_start))
        for path, base_start, code in sections
    ]

    cursor_sec_idx = 0
    cursor_entry_idx: int | None = None
    for sec_idx, (_, entries) in enumerate(section_entries):
        for ent_idx, (_, text) in enumerate(entries):
            if "# <MISSING_CODE>" in text:
                cursor_sec_idx = sec_idx
                cursor_entry_idx = ent_idx
                break
        if cursor_entry_idx is not None:
            break

    local_entries: list[tuple[int, str]] = []
    distant_before: list[tuple[int, str]] = []
    distant_after: list[tuple[int, str]] = []

    selection = row.get("context_selection")
    sel_windows = (
        [w for w in selection.get("windows", ()) if isinstance(w, dict) and isinstance(w.get("path"), str)]
        if isinstance(selection, dict) and selection.get("gold_used") is not True
        else []
    )

    if cursor_entry_idx is None:
        # Check non-gold selection windows with explicit line bounds first (preserving windowed file offsets).
        best_window_match: tuple[int, int, int] | None = None
        for sec_idx, (path, entries) in enumerate(section_entries):
            if not entries:
                continue
            sec_min, sec_max = entries[0][0], entries[-1][0]
            overlapping = [
                w
                for w in sel_windows
                if w.get("path") == path
                and type(w.get("start_line")) is int
                and type(w.get("end_line")) is int
                and w["end_line"] >= sec_min
                and w["start_line"] <= sec_max
            ]
            if overlapping:
                best_window_match = (
                    sec_idx,
                    min(w["start_line"] for w in overlapping),
                    max(w["end_line"] for w in overlapping),
                )
                break

        if best_window_match is not None:
            cursor_sec_idx, center_start, center_end = best_window_match
            cursor_path, cursor_entries = section_entries[cursor_sec_idx]
            g_start = center_start - local_window_lines
            g_end = center_end + local_window_lines
            distant_before = [(num, txt) for num, txt in cursor_entries if num < g_start]
            local_entries = [(num, txt) for num, txt in cursor_entries if g_start <= num <= g_end]
            distant_after = [(num, txt) for num, txt in cursor_entries if num > g_end]
        else:
            # Never touch gold/target fields! Match path, non-gold window path, or instruction keywords only.
            hint_path = row.get("path")
            if (not isinstance(hint_path, str) or not hint_path or hint_path == "repository_context") and len(
                {w["path"] for w in sel_windows}
            ) == 1:
                hint_path = sel_windows[0]["path"]
            if isinstance(hint_path, str) and hint_path and hint_path != "repository_context":
                for sec_idx, (path, _) in enumerate(section_entries):
                    if path == hint_path:
                        cursor_sec_idx = sec_idx
                        break
            else:
                lowered_instr = instruction.lower()
                for sec_idx, (path, _) in enumerate(section_entries):
                    if path.lower() in lowered_instr or path.rsplit("/", 1)[-1].lower() in lowered_instr:
                        cursor_sec_idx = sec_idx
                        break
            _, target_entries = section_entries[cursor_sec_idx]
            cursor_entry_idx = _anchor_index_for_repair(target_entries, instruction)

    cursor_path, cursor_entries = section_entries[cursor_sec_idx]
    if cursor_entries and not local_entries:
        assert cursor_entry_idx is not None
        if "# <MISSING_CODE>" in cursor_entries[cursor_entry_idx][1]:
            local_start = max(0, cursor_entry_idx - local_window_lines)
            local_end = min(len(cursor_entries), cursor_entry_idx + local_window_lines + 1)
        else:
            # For repair/localization anchored at file header or matched line
            if cursor_entry_idx == 0:
                local_start = 0
                local_end = min(len(cursor_entries), 2 * local_window_lines)
            else:
                local_start = max(0, cursor_entry_idx - local_window_lines)
                local_end = min(len(cursor_entries), cursor_entry_idx + local_window_lines + 1)
        local_entries = cursor_entries[local_start:local_end]
        distant_before = cursor_entries[:local_start]
        distant_after = cursor_entries[local_end:]

    enc_parts = []
    if preamble.strip():
        enc_parts.append(preamble.strip("\n"))

    for sec_idx, (path, entries) in enumerate(section_entries):
        if sec_idx != cursor_sec_idx:
            if entries:
                enc_parts.append(
                    f"### File: {path} (lines {entries[0][0]}-{entries[-1][0]})\n"
                    + _render_entries(entries)
                )
        else:
            if distant_before:
                enc_parts.append(
                    f"### File: {path} (lines {distant_before[0][0]}-{distant_before[-1][0]})\n"
                    + _render_entries(distant_before)
                )
            if distant_after:
                enc_parts.append(
                    f"### File: {path} (lines {distant_after[0][0]}-{distant_after[-1][0]})\n"
                    + _render_entries(distant_after)
                )

    if local_entries:
        local_range = (local_entries[0][0], local_entries[-1][0])
        dec_local = (
            f"### Local Cursor: {cursor_path} (lines {local_range[0]}-{local_range[1]})\n"
            + _render_entries(local_entries)
        )
    else:
        local_range = None
        dec_local = ""

    enc_ctx = "\n\n".join(enc_parts)
    full_ctx = "\n\n".join(p for p in (enc_ctx, dec_local) if p)
    return DisjointContextSplit(
        encoder_context=enc_ctx,
        decoder_local_context=dec_local,
        full_context=full_ctx,
        task_type=task_type,
        cursor_path=cursor_path,
        local_line_range=local_range,
    )


_LEXICAL_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "have",
        "has",
        "are",
        "was",
        "were",
        "been",
        "will",
        "would",
        "could",
        "should",
        "what",
        "which",
        "where",
        "when",
        "does",
        "return",
        "only",
        "code",
        "file",
        "files",
        "function",
        "class",
        "method",
        "value",
        "based",
        "context",
        "question",
        "answer",
        "provided",
        "codebase",
        "repository",
        "following",
        "instruction",
        "issue",
        "bug",
        "error",
        "fix",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,}")
_DYNAMIC_WINDOW_LINES = 40
_MAX_DYNAMIC_BLOCK_CHARS = 1800
_RECONSTRUCT_LINE_RE = re.compile(r"\bsource line\s+(\d+)\b", re.IGNORECASE)


def _split_long_prose_line(line: str, max_chars: int = _MAX_DYNAMIC_BLOCK_CHARS) -> list[str]:
    """Split an oversized single-line prose paragraph at sentence or whitespace boundaries."""
    if len(line) <= max_chars:
        return [line]
    segments: list[str] = []
    remaining = line
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_at = -1
        for sep in (". ", "? ", "! ", "; ", ", ", " "):
            pos = window.rfind(sep)
            if pos >= max_chars // 3:
                split_at = pos + (1 if sep != " " else 0)
                break
        if split_at <= 0:
            split_at = max_chars
        piece = remaining[:split_at].rstrip()
        if piece:
            segments.append(piece)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        segments.append(remaining)
    return segments or [line[:max_chars]]


def reserved_output_token_budget(task_type: str, *, decoder_limit: int = 8192) -> int:
    """Return output token reserve strictly from public task_type metadata (never target length)."""
    if type(decoder_limit) is not int or decoder_limit <= 0:
        return 0
    kind = (task_type or "repair").strip().lower()
    if kind in {
        "qa",
        "long_context_qa",
        "mcq",
        "source_mcq",
        "source_extraction",
        "localization",
        "niah",
        "retrieval",
    }:
        base = 512
    elif kind in {"math", "gsm8k", "retention", "instruction"}:
        base = 2048
    elif kind in {"completion", "repair", "source_reconstruction"}:
        base = 2304
    else:
        base = 2048
    return min(base, decoder_limit // 2)


def _count_context_tokens(
    text: str,
    tokenizer=None,
    *,
    chars_per_token: float = 3.2,
) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    denom = max(0.5, float(chars_per_token))
    return max(1, math.ceil(len(text) / denom))


def _valid_non_gold_selection_windows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    selection = row.get("context_selection")
    if not isinstance(selection, Mapping):
        return []
    if selection.get("gold_used") is True:
        return []
    for meta_key in ("source", "strategy", "method", "selection_source"):
        val = selection.get(meta_key)
        if isinstance(val, str) and val.strip().lower() in FORBIDDEN_SELECTION_SOURCES:
            return []
    windows: list[dict[str, Any]] = []
    for window in selection.get("windows", ()):
        if not isinstance(window, Mapping):
            continue
        if window.get("gold_used") is True:
            continue
        w_src = window.get("source")
        if isinstance(w_src, str) and w_src.strip().lower() in FORBIDDEN_SELECTION_SOURCES:
            continue
        if isinstance(window.get("path"), str) and window["path"].strip():
            windows.append(dict(window))
    return windows


@dataclass(frozen=True)
class _DynamicSection:
    sec_idx: int
    kind: str  # "file" | "cross_file" | "target_file" | "numbered_source" | "preamble" | "raw"
    path: str | None
    entries: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _DynamicBlock:
    block_idx: int
    sec_idx: int
    kind: str
    path: str | None
    entries: tuple[tuple[int, str], ...]
    has_cursor: bool

    @property
    def start_line(self) -> int:
        return self.entries[0][0]

    @property
    def end_line(self) -> int:
        return self.entries[-1][0]


def _extract_dynamic_sections(
    row: Mapping[str, Any],
    body: str,
    task_type: str,
) -> list[_DynamicSection]:
    if not body.strip():
        return []

    # 1. Grounded tasks (`### Numbered source`)
    if "### Numbered source\n" in body or body.startswith("### Numbered source\n"):
        _, _, numbered_body = body.partition("### Numbered source\n")
        entries = tuple(_extract_numbered_entries(numbered_body.splitlines(), 1))
        return [_DynamicSection(0, "numbered_source", None, entries)] if entries else []

    # 2. CrossCodeLong FIM layout (`### PREFIX` / `### SUFFIX`)
    if "### PREFIX\n" in body and (
        "### SUFFIX\n" in body or "### MISSING CODE" in body or "# <MISSING_CODE>" in body
    ):
        cross_text = ""
        rest = body
        if "### CROSS-FILE CONTEXT\n" in rest:
            _, _, after_cross = rest.partition("### CROSS-FILE CONTEXT\n")
            if "\n### PREFIX\n" in after_cross:
                cross_text, _, rest = after_cross.partition("\n### PREFIX\n")
                rest = "### PREFIX\n" + rest
            elif "### PREFIX\n" in after_cross:
                cross_text, _, rest = after_cross.partition("### PREFIX\n")
                rest = "### PREFIX\n" + rest
        prefix_text = ""
        suffix_text = ""
        if "### PREFIX\n" in rest:
            _, _, after_pre = rest.partition("### PREFIX\n")
            if "\n### SUFFIX\n" in after_pre:
                prefix_raw, _, suffix_text = after_pre.partition("\n### SUFFIX\n")
            elif after_pre.startswith("### SUFFIX\n"):
                prefix_raw, suffix_text = "", after_pre[len("### SUFFIX\n") :]
            elif "### SUFFIX\n" in after_pre:
                prefix_raw, _, suffix_text = after_pre.partition("### SUFFIX\n")
            else:
                prefix_raw, suffix_text = after_pre, ""
            prefix_text = _FIM_CURSOR_MARKER_RE.sub("", prefix_raw).rstrip("\n")
            suffix_text = _FIM_CURSOR_MARKER_RE.sub("", suffix_text).rstrip("\n")
        pre_entries = _extract_numbered_entries(prefix_text.splitlines(), 1)
        cursor_num = (pre_entries[-1][0] + 1) if pre_entries else 1
        suf_entries = _extract_numbered_entries(suffix_text.splitlines(), cursor_num + 1)
        sections: list[_DynamicSection] = []
        if cross_text.strip():
            cross_entries = tuple(_extract_numbered_entries(cross_text.strip("\n").splitlines(), 1))
            if cross_entries:
                sections.append(_DynamicSection(len(sections), "cross_file", None, cross_entries))
        target_entries = tuple([*pre_entries, (cursor_num, "# <MISSING_CODE>"), *suf_entries])
        sections.append(_DynamicSection(len(sections), "target_file", None, target_entries))
        return sections

    # 3. File sections (`### File: ...`) or LongCodeQA `[start of path]`
    preamble, parsed_files = _parse_file_sections(body)
    if not parsed_files:
        lqa_matches = list(
            re.finditer(r"\[start of ([^\]]+)\]\n(.*?)\n\[end of [^\]]+\]", body, re.S)
        )
        if lqa_matches:
            parsed_files = [
                (m.group(1).strip(), 1, m.group(2).rstrip("\n")) for m in lqa_matches
            ]
            preamble = ""

    if parsed_files:
        sections = []
        if preamble.strip():
            pre_lines = tuple(_extract_numbered_entries(preamble.strip("\n").splitlines(), 1))
            if pre_lines:
                sections.append(_DynamicSection(len(sections), "preamble", None, pre_lines))
        for path, base_start, code in parsed_files:
            entries = tuple(_extract_numbered_entries(code.splitlines(), base_start))
            if entries:
                sections.append(_DynamicSection(len(sections), "file", path, entries))
        return sections

    # 4. Raw unstructured context (LongAlign, RepoNIAH, SWE-QA, single-file raw code, etc.)
    raw_lines = body.strip("\n").splitlines()
    if not raw_lines:
        return []
    fallback_path = row.get("path")
    if task_type in {"repair", "completion", "localization"}:
        entries = tuple(_extract_numbered_entries(raw_lines, 1))
        resolved_path = (
            fallback_path
            if isinstance(fallback_path, str)
            and fallback_path
            and fallback_path != "repository_context"
            else "target.py"
        )
        return [_DynamicSection(0, "file", resolved_path, entries)]
    if _is_numbered_block(raw_lines):
        entries = tuple(_extract_numbered_entries(raw_lines, 1))
        return [_DynamicSection(0, "numbered_source", None, entries)]
    expanded_lines: list[str] = []
    for raw_line in raw_lines:
        if len(raw_line) > _MAX_DYNAMIC_BLOCK_CHARS:
            expanded_lines.extend(_split_long_prose_line(raw_line, _MAX_DYNAMIC_BLOCK_CHARS))
        else:
            expanded_lines.append(raw_line)
    entries = tuple((idx + 1, seg) for idx, seg in enumerate(expanded_lines))
    return [_DynamicSection(0, "raw", None, entries)]


def _render_block_group(
    kind: str,
    path: str | None,
    entries: list[tuple[int, str]],
    *,
    number_lines: bool = True,
) -> str:
    if not entries:
        return ""
    if kind in {"preamble", "raw"}:
        return "\n".join(text for _, text in entries)
    start_line, end_line = entries[0][0], entries[-1][0]
    if kind == "numbered_source":
        body = _render_entries(entries)
        return f"### Numbered source (lines {start_line}-{end_line})\n{body}"
    body = _render_entries(entries) if number_lines else "\n".join(text for _, text in entries)
    if kind == "file" and path:
        return f"### File: {path} (lines {start_line}-{end_line})\n{body}"
    if kind == "cross_file":
        return f"### Cross-File Context (lines {start_line}-{end_line})\n{body}"
    if kind == "target_file":
        return f"### Target File (lines {start_line}-{end_line})\n{body}"
    return body


def _reassemble_blocks_in_order(
    blocks: list[_DynamicBlock],
    selected: set[int],
    *,
    number_lines: bool = True,
) -> str:
    """Reassemble selected blocks in original document/line order, merging adjacent runs."""
    ordered = [blocks[i] for i in sorted(selected)]
    if not ordered:
        return ""
    parts: list[str] = []
    cur_sec = ordered[0].sec_idx
    cur_kind = ordered[0].kind
    cur_path = ordered[0].path
    cur_entries: list[tuple[int, str]] = list(ordered[0].entries)
    prev_block_idx = ordered[0].block_idx

    for blk in ordered[1:]:
        if blk.sec_idx == cur_sec and blk.block_idx == prev_block_idx + 1:
            cur_entries.extend(blk.entries)
        else:
            rendered = _render_block_group(
                cur_kind, cur_path, cur_entries, number_lines=number_lines
            )
            if rendered:
                parts.append(rendered)
            cur_sec = blk.sec_idx
            cur_kind = blk.kind
            cur_path = blk.path
            cur_entries = list(blk.entries)
        prev_block_idx = blk.block_idx

    rendered = _render_block_group(cur_kind, cur_path, cur_entries, number_lines=number_lines)
    if rendered:
        parts.append(rendered)
    return "\n\n".join(parts)


def _extract_query_terms(instruction: str) -> tuple[set[str], set[str]]:
    """Extract lowercase lexical terms and exact-case identifiers from public instruction."""
    raw_tokens = _IDENTIFIER_RE.findall(instruction)
    lower_terms: set[str] = set()
    exact_terms: set[str] = set()
    for tok in raw_tokens:
        low = tok.lower()
        if len(low) >= 2 and low not in _LEXICAL_STOPWORDS:
            lower_terms.add(low)
            if "_" in tok or any(ch.isupper() for ch in tok):
                exact_terms.add(tok)
        # Split snake_case and camelCase sub-identifiers for richer lexical matching
        for part in re.split(r"_+|(?<=[a-z])(?=[A-Z])", tok):
            p_low = part.lower()
            if len(p_low) >= 3 and p_low not in _LEXICAL_STOPWORDS:
                lower_terms.add(p_low)
    return lower_terms, exact_terms


def dynamic_sliding_bifocal_split(
    row: Mapping[str, Any],
    *,
    decoder_context_token_budget: int = 5632,
    encoder_token_budget: int = 32768,
    decoder_tokenizer=None,
    encoder_tokenizer=None,
    chars_per_token: float = 3.2,
    allow_lexical_routing: bool = True,
    number_lines: bool = True,
) -> DisjointContextSplit:
    """Dynamically route context between decoder fovea and disjoint encoder periphery.

    Security & parity contract:
    - Strictly uses ONLY public non-gold signals (cursor anchor, non-gold context_selection
      excluding FORBIDDEN_SELECTION_SOURCES, and public instruction lexical/IDF overlap).
    - NEVER reads target/gold fields (`target_text`, `target_patch`, `gold_patch`, etc.).
    - Grounded line-lookup tasks (`source_extraction`, `source_mcq`) route 100% of numbered
      source lines to `encoder_context` so MCQ option strings never leak into the decoder fovea.
    - Short-context fast path: if context is empty or fits inside `decoder_context_token_budget`,
      100% goes to `decoder_local_context` and `encoder_context == ""`.
    - Long-context path: packs top-scored bounded windows in original order into
      `decoder_local_context` and strictly disjoint remaining blocks in original order into
      `encoder_context`, tracking `dropped_block_count` so callers can trigger unnumbered fallback.
    """
    task_type = str(row.get("task_type") or "repair")
    instruction = str(row.get("instruction") or "").replace("\r\n", "\n")
    raw_context = str(row.get("context") or "").replace("\r\n", "\n")
    body = _strip_leading_instruction(raw_context, instruction)

    if not body.strip() or (
        task_type in {"retention", "instruction"}
        and "### File:" not in body
        and body.strip("\n") == instruction.strip("\n")
    ):
        return DisjointContextSplit(
            encoder_context="",
            decoder_local_context="",
            full_context="",
            task_type=task_type,
        )

    sections = _extract_dynamic_sections(row, body, task_type)
    if not sections:
        return DisjointContextSplit(
            encoder_context="",
            decoder_local_context="",
            full_context="",
            task_type=task_type,
        )

    is_grounded_line_lookup = task_type in {"source_extraction", "source_mcq"}
    eff_dec_budget = 0 if is_grounded_line_lookup else decoder_context_token_budget
    eff_allow_lexical = False if is_grounded_line_lookup else allow_lexical_routing

    # Chunk sections into ordered contiguous windows bounded by both line count and character budget
    blocks: list[_DynamicBlock] = []
    for sec in sections:
        chunk_list: list[tuple[int, str]] = []
        chunk_chars = 0
        for entry in sec.entries:
            entry_chars = len(entry[1]) + 1
            if chunk_list and (
                len(chunk_list) >= _DYNAMIC_WINDOW_LINES
                or chunk_chars + entry_chars > _MAX_DYNAMIC_BLOCK_CHARS
            ):
                chunk = tuple(chunk_list)
                has_cursor = any(
                    ("# <MISSING_CODE>" in text or "<CURSOR>" in text) for _, text in chunk
                )
                blocks.append(
                    _DynamicBlock(
                        block_idx=len(blocks),
                        sec_idx=sec.sec_idx,
                        kind=sec.kind,
                        path=sec.path,
                        entries=chunk,
                        has_cursor=has_cursor,
                    )
                )
                chunk_list = []
                chunk_chars = 0
            chunk_list.append(entry)
            chunk_chars += entry_chars
        if chunk_list:
            chunk = tuple(chunk_list)
            has_cursor = any(
                ("# <MISSING_CODE>" in text or "<CURSOR>" in text) for _, text in chunk
            )
            blocks.append(
                _DynamicBlock(
                    block_idx=len(blocks),
                    sec_idx=sec.sec_idx,
                    kind=sec.kind,
                    path=sec.path,
                    entries=chunk,
                    has_cursor=has_cursor,
                )
            )

    all_indices = set(range(len(blocks)))
    full_rendered = _reassemble_blocks_in_order(blocks, all_indices, number_lines=number_lines)

    cursor_block: _DynamicBlock | None = None
    if task_type == "source_reconstruction":
        recon_match = _RECONSTRUCT_LINE_RE.search(instruction)
        if recon_match is not None:
            target_line_no = int(recon_match.group(1))
            cursor_block = next(
                (
                    b
                    for b in blocks
                    if any(
                        num == target_line_no and "# <MISSING_CODE>" in text
                        for num, text in b.entries
                    )
                ),
                None,
            )
    if cursor_block is None:
        cursor_block = next((b for b in blocks if b.has_cursor), None)

    default_cursor_path = cursor_block.path if cursor_block is not None else blocks[0].path
    default_line_range = (
        (cursor_block.start_line, cursor_block.end_line)
        if cursor_block is not None
        else (blocks[0].start_line, blocks[-1].end_line)
    )

    # Short-context fast path: 100% in decoder_local_context, encoder_context = ""
    if eff_dec_budget > 0:
        full_dec_tokens = _count_context_tokens(
            full_rendered, decoder_tokenizer, chars_per_token=chars_per_token
        )
        if full_dec_tokens <= eff_dec_budget:
            return DisjointContextSplit(
                encoder_context="",
                decoder_local_context=full_rendered,
                full_context=full_rendered,
                task_type=task_type,
                cursor_path=default_cursor_path,
                local_line_range=default_line_range,
                dropped_block_count=0,
            )

    # Score every block strictly from public non-gold signals
    scores: list[float] = [0.0] * len(blocks)

    # Signal 1: Cursor anchor (+1000) expanding outward ~65% preceding / ~35% following
    if cursor_block is not None:
        anchor_idx = cursor_block.block_idx
        anchor_sec = cursor_block.sec_idx
        scores[anchor_idx] += 1000.0
        for blk in blocks:
            if blk.sec_idx != anchor_sec or blk.block_idx == anchor_idx:
                continue
            dist = anchor_idx - blk.block_idx
            eff_dist = (dist / 0.65) if dist > 0 else ((-dist) / 0.35)
            scores[blk.block_idx] += 900.0 / (1.0 + eff_dist)

    # Signal 2: Non-gold context_selection (excluding FORBIDDEN_SELECTION_SOURCES) (+500)
    sel_windows = _valid_non_gold_selection_windows(row)
    if sel_windows:
        for blk in blocks:
            if not blk.path:
                continue
            best_sel = 0.0
            for win in sel_windows:
                if win["path"] != blk.path:
                    continue
                w_start = win.get("start_line")
                w_end = win.get("end_line")
                if type(w_start) is int and type(w_end) is int:
                    if blk.end_line >= w_start and blk.start_line <= w_end:
                        best_sel = max(best_sel, 500.0)
                    else:
                        line_gap = (
                            (w_start - blk.end_line)
                            if blk.end_line < w_start
                            else (blk.start_line - w_end)
                        )
                        best_sel = max(best_sel, 250.0 / (1.0 + line_gap / 40.0))
                else:
                    best_sel = max(best_sel, 500.0)
            scores[blk.block_idx] += best_sel

    # Signal 3: Public instruction identifier & lexical IDF overlap + path mention
    lower_terms, exact_terms = (
        _extract_query_terms(instruction) if eff_allow_lexical else (set(), set())
    )
    lowered_instruction = instruction.lower()
    if lower_terms or exact_terms:
        block_term_sets: list[set[str]] = []
        block_raw_texts: list[str] = []
        df: dict[str, int] = {}
        for blk in blocks:
            raw_blk = "\n".join(text for _, text in blk.entries)
            if blk.path:
                raw_blk = f"{blk.path}\n{raw_blk}"
            block_raw_texts.append(raw_blk)
            toks = {t.lower() for t in _IDENTIFIER_RE.findall(raw_blk)}
            block_term_sets.append(toks)
            for t in lower_terms:
                if t in toks:
                    df[t] = df.get(t, 0) + 1
        num_blocks = len(blocks)
        idf = {
            t: math.log((1.0 + num_blocks) / (1.0 + df.get(t, 0))) + 1.0
            for t in lower_terms
        }
        for blk in blocks:
            toks = block_term_sets[blk.block_idx]
            raw_blk = block_raw_texts[blk.block_idx]
            lex_score = sum(idf[t] for t in lower_terms if t in toks)
            exact_hits = sum(1 for t in exact_terms if t in raw_blk)
            lex_score += 2.5 * exact_hits
            if blk.path and (
                blk.path.lower() in lowered_instruction
                or blk.path.rsplit("/", 1)[-1].lower() in lowered_instruction
            ):
                lex_score += 150.0
            scores[blk.block_idx] += lex_score

    # Pre-compute per-block decoder token costs respecting number_lines
    dec_block_costs = [
        _count_context_tokens(
            _render_block_group(
                blk.kind, blk.path, list(blk.entries), number_lines=number_lines
            ),
            decoder_tokenizer,
            chars_per_token=chars_per_token,
        )
        for blk in blocks
    ]

    # Pack top blocks up to eff_dec_budget
    ranked_indices = sorted(range(len(blocks)), key=lambda i: (-scores[i], i))
    selected_dec: set[int] = set()
    used_dec_tokens = 0
    if eff_dec_budget > 0:
        for idx in ranked_indices:
            cost = dec_block_costs[idx] + (2 if selected_dec else 0)
            if used_dec_tokens + cost <= eff_dec_budget:
                selected_dec.add(idx)
                used_dec_tokens += cost

    # Strictly disjoint remaining blocks packed up to encoder_token_budget
    remaining_indices = [i for i in range(len(blocks)) if i not in selected_dec]
    selected_enc: set[int] = set()
    if remaining_indices and encoder_token_budget > 0:
        if encoder_tokenizer is decoder_tokenizer:
            enc_block_costs = dec_block_costs
        else:
            enc_block_costs = [
                _count_context_tokens(
                    _render_block_group(
                        blk.kind, blk.path, list(blk.entries), number_lines=number_lines
                    ),
                    encoder_tokenizer,
                    chars_per_token=chars_per_token,
                )
                for blk in blocks
            ]
        total_rem_cost = sum(enc_block_costs[i] for i in remaining_indices) + 2 * max(
            0, len(remaining_indices) - 1
        )
        if total_rem_cost <= encoder_token_budget:
            selected_enc = set(remaining_indices)
        else:
            used_enc_tokens = 0
            for idx in sorted(remaining_indices, key=lambda i: (-scores[i], i)):
                cost = enc_block_costs[idx] + (2 if selected_enc else 0)
                if used_enc_tokens + cost <= encoder_token_budget:
                    selected_enc.add(idx)
                    used_enc_tokens += cost

    dropped_blocks = len(blocks) - len(selected_dec) - len(selected_enc)
    decoder_local_context = _reassemble_blocks_in_order(
        blocks, selected_dec, number_lines=number_lines
    )
    encoder_context = _reassemble_blocks_in_order(
        blocks, selected_enc, number_lines=number_lines
    )

    if selected_dec:
        if cursor_block is not None and cursor_block.block_idx in selected_dec:
            chosen_cursor_path = cursor_block.path
            chosen_range = (cursor_block.start_line, cursor_block.end_line)
        else:
            top_dec_blk = blocks[max(selected_dec, key=lambda i: (scores[i], -i))]
            chosen_cursor_path = top_dec_blk.path
            chosen_range = (top_dec_blk.start_line, top_dec_blk.end_line)
    else:
        chosen_cursor_path = default_cursor_path
        chosen_range = None

    return DisjointContextSplit(
        encoder_context=encoder_context,
        decoder_local_context=decoder_local_context,
        full_context=full_rendered,
        task_type=task_type,
        cursor_path=chosen_cursor_path,
        local_line_range=chosen_range,
        dropped_block_count=dropped_blocks,
    )


def repair_preimage_coverage(patch, selection):
    """Audit cohort eligibility after issue-only selection; never add gold-selected windows."""
    from bifocalgemma.data.patches import patch_paths

    paths = patch_paths(patch)
    windows = {}
    for window in selection.get("windows", []):
        windows.setdefault(window["path"], []).append((window["start_line"], window["end_line"]))
    blocks = re.split(r"(?=^diff --git )", patch, flags=re.M)
    checked = 0
    for block in blocks:
        header = re.match(r"diff --git a/(\S+) b/(\S+)\n", block)
        if not header:
            continue
        path = header[1]
        if path not in paths or path not in windows or re.search(r"^--- /dev/null$", block, re.M):
            return False
        for hunk in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", block, re.M):
            start = int(hunk[1])
            length = int(hunk[2]) if hunk[2] is not None else 1
            end = start + max(1, length) - 1
            cursor = start
            for left, right in sorted(windows[path]):
                if left <= cursor <= right:
                    cursor = right + 1
                if cursor > end:
                    break
            if start < 1 or cursor <= end:
                return False
            checked += 1
    return checked > 0


def normalize_context_task(row):
    if row.get("role") != "train" or row.get("record_type") not in {
        "context_sft_v1",
        "grounded_context_sft_v1",
    }:
        raise ValueError("context_task_is_not_training_data")
    kind = row.get("task_type")
    if kind not in TASK_INSTRUCTIONS:
        raise ValueError("unsupported_context_task")
    is_v4 = row.get("context_format") == "v4" or row.get("disjoint_context") is True
    if row.get("record_type") == "grounded_context_sft_v1":
        from bifocalgemma.data.grounded_tasks import validate_grounded_task

        validate_grounded_task(row)
    elif not is_v4 and kind not in {"repair", "completion", "localization"}:
        raise ValueError("grounded_task_requires_evidence")
    for key in (
        "instance_id",
        "group",
        "instruction",
        "context",
        "target_text",
        "source",
        "source_revision",
        "snapshot_sha256",
        "commit",
        "repos",
    ):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError("missing_context_task_field:" + key)
    if row.get("context_selection", {}).get("gold_used") is not False:
        raise ValueError("context_task_contains_gold_selected_input")
    if not re.fullmatch(r"[0-9a-f]{64}", row["snapshot_sha256"]):
        raise ValueError("invalid_context_snapshot_hash")

    if row.get("source") in BLOCKED_V4_DATASETS or row.get("benchmark"):
        raise ValueError("benchmark_source_cannot_enter_context_training")

    legacy_sources = {
        "swe_rebench_v2": "nebius/SWE-rebench-V2",
        "swe_fixer": "internlm/SWE-Fixer-Train-110K",
    }
    expected_sources = (
        {k: v["dataset_id"] for k, v in V4_DATASET_REGISTRY.items()}
        if is_v4
        else legacy_sources
    )
    if row.get("source_key") not in expected_sources:
        raise ValueError("benchmark_source_cannot_enter_context_training")
    if row["source"] != expected_sources[row["source_key"]]:
        raise ValueError("context_source_provenance_mismatch")
    if is_v4:
        entry = V4_DATASET_REGISTRY[row["source_key"]]
        if entry.get("requires_row_license_check") and not is_permissive_license(
            row.get("license"), code_only=True
        ):
            raise ValueError("v4_row_license_not_permitted")
    if kind == "completion" and (
        row["context"].count("# <MISSING_CODE>") != 1 or row["target_text"] in row["context"]
    ):
        raise ValueError("completion_target_masking_failed")
    if kind == "localization":
        paths = json.loads(row["target_text"])
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise ValueError("invalid_localization_target")
        visible = {w["path"] for w in row["context_selection"].get("windows", [])}
        if not set(paths) <= visible:
            raise ValueError("localization_target_not_visible")
    return {
        **row,
        "path": "repository_context",
        "before": row["context"],
        "after": "",
        "pair_hash": hashlib.sha256(
            (row["context"] + "\0" + row["target_text"]).encode()
        ).hexdigest(),
        "instruction_hash": hashlib.sha256(row["instruction"].encode()).hexdigest(),
        "license": str(row.get("license") or "unknown"),
    }


def _use_disjoint_mode(row: Mapping[str, Any], disjoint_context: bool | None) -> bool:
    if disjoint_context is not None:
        return disjoint_context
    return bool(
        row.get("disjoint_context") is True
        or row.get("context_format") == "v4"
        or row.get("record_type") == "context_sft_v4"
    )


def _apply_decoder_chat_template(decoder, prompt_content: str) -> str:
    messages = [{"role": "user", "content": prompt_content}]
    try:
        return decoder.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return decoder.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def task_prompt(
    row,
    *,
    paired,
    disjoint_context: bool | None = None,
    decoder_local_only: bool = False,
    local_window_lines: int = 48,
    context_split: DisjointContextSplit | None = None,
):
    kind = row.get("task_type", "repair")
    if kind not in TASK_INSTRUCTIONS:
        raise ValueError("unsupported_sft_task")
    use_disjoint = _use_disjoint_mode(row, disjoint_context)
    if not paired:
        if not use_disjoint and not decoder_local_only:
            prompt = TASK_INSTRUCTIONS[kind] + "\n\n"
            return prompt + row["context"]
        split = (
            context_split
            if context_split is not None
            else split_disjoint_context(row, local_window_lines=local_window_lines)
        )
        header = TASK_INSTRUCTIONS[kind] + "\n\n" + row["instruction"]
        if decoder_local_only:
            return (
                f"{header}\n\n{split.decoder_local_context}"
                if split.decoder_local_context
                else header
            )
        return f"{header}\n\n{split.full_context}" if split.full_context else header

    if not use_disjoint and not decoder_local_only:
        prompt = TASK_INSTRUCTIONS[kind] + "\n\n"
        return prompt + row["instruction"]
    split = (
        context_split
        if context_split is not None
        else split_disjoint_context(row, local_window_lines=local_window_lines)
    )
    header = TASK_INSTRUCTIONS[kind] + "\n\n" + row["instruction"]
    if split.decoder_local_context:
        return f"{header}\n\n{split.decoder_local_context}"
    return header


def _resolve_eot_token_id(decoder) -> int | None:
    """Resolve the closing turn delimiter (<turn|>), falling back to eos_token_id if absent."""
    eot_token_id = decoder.convert_tokens_to_ids("<turn|>")
    unk_token_id = getattr(decoder, "unk_token_id", None)
    if (
        not isinstance(eot_token_id, int)
        or eot_token_id < 0
        or (unk_token_id is not None and eot_token_id == unk_token_id)
    ):
        eot_token_id = getattr(decoder, "eos_token_id", None)
    return eot_token_id if isinstance(eot_token_id, int) and eot_token_id >= 0 else None


def _anti_shortcut_budget(
    row: Mapping[str, Any],
    base_budget: int,
    *,
    task_type: str,
    raw_context_tokens: int,
) -> int:
    if raw_context_tokens <= 512 or base_budget <= 0:
        return base_budget
    digest = hashlib.sha256(str(row.get("instance_id") or "").encode("utf-8")).digest()
    u = int.from_bytes(digest[:4], "big") / float(1 << 32)
    kind = (task_type or "repair").strip().lower()
    if kind in {"qa", "long_context_qa", "mcq", "source_mcq", "source_extraction", "localization"} and u < 0.25:
        return 0
    if u < 0.35:
        return min(base_budget, 512)
    if u < 0.55:
        return min(base_budget, 1024)
    return base_budget


def tokenize_context_task(
    row,
    spec,
    encoder,
    decoder,
    *,
    encoder_limit,
    decoder_limit,
    disjoint_context: bool | None = None,
    decoder_local_only: bool = False,
    local_window_lines: int | None = None,
    training_anti_shortcut: bool = False,
):
    use_disjoint = _use_disjoint_mode(row, disjoint_context)
    if decoder_local_only is False:
        decoder_local_only = bool(row.get("decoder_local_only", False))
    if local_window_lines is None and isinstance(row.get("local_window_lines"), int):
        local_window_lines = row["local_window_lines"]
    if training_anti_shortcut is False:
        training_anti_shortcut = bool(row.get("training_anti_shortcut", False))

    target_text = row.get("target_text", row.get("target_patch"))
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("empty_sft_target")
    eot_token_id = _resolve_eot_token_id(decoder)
    # Encode target_text directly without apply_chat_template (which runs `| trim`) or
    # `.rstrip()`, preserving leading indentation and trailing newlines verbatim.
    target_body_ids = tuple(decoder.encode(target_text, add_special_tokens=False))
    if eot_token_id is None or not target_body_ids:
        raise ValueError("context_task_special_tokens")
    target = target_body_ids + (eot_token_id,)

    context_split: DisjointContextSplit | None = None
    if spec.encoder is not None and use_disjoint:
        if local_window_lines is not None:
            context_split = split_disjoint_context(row, local_window_lines=local_window_lines)
            prompt_content = task_prompt(
                row,
                paired=True,
                disjoint_context=True,
                decoder_local_only=decoder_local_only,
                local_window_lines=local_window_lines,
                context_split=context_split,
            )
            prefix = _apply_decoder_chat_template(decoder, prompt_content)
            prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
        else:
            task_type = str(row.get("task_type") or "repair")
            reserved_out = reserved_output_token_budget(task_type, decoder_limit=decoder_limit)
            empty_split = DisjointContextSplit("", "", "", task_type=task_type)
            base_prompt = task_prompt(
                row,
                paired=True,
                disjoint_context=True,
                decoder_local_only=decoder_local_only,
                context_split=empty_split,
            )
            base_prefix_tokens = len(
                decoder.encode(
                    _apply_decoder_chat_template(decoder, base_prompt), add_special_tokens=False
                )
            )
            dec_ctx_budget = max(0, decoder_limit - reserved_out - base_prefix_tokens - 16)
            if training_anti_shortcut:
                raw_ctx_tokens = _count_context_tokens(str(row.get("context") or ""), decoder)
                dec_ctx_budget = _anti_shortcut_budget(
                    row,
                    dec_ctx_budget,
                    task_type=task_type,
                    raw_context_tokens=raw_ctx_tokens,
                )
            enc_tok_budget = max(0, encoder_limit - 16)
            cur_budget = dec_ctx_budget
            prefix_ids = ()
            for _ in range(4):
                context_split = dynamic_sliding_bifocal_split(
                    row,
                    decoder_context_token_budget=cur_budget,
                    encoder_token_budget=enc_tok_budget,
                    decoder_tokenizer=decoder,
                    encoder_tokenizer=encoder if not decoder_local_only else decoder,
                )
                prompt_content = task_prompt(
                    row,
                    paired=True,
                    disjoint_context=True,
                    decoder_local_only=decoder_local_only,
                    context_split=context_split,
                )
                prefix = _apply_decoder_chat_template(decoder, prompt_content)
                prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
                overflow = (len(prefix_ids) + len(target) - 1) - decoder_limit
                if overflow <= 0 or cur_budget <= 0:
                    break
                cur_budget = max(0, cur_budget - overflow - 16)
        assert context_split is not None
        enc_context = "" if decoder_local_only else context_split.encoder_context
        if enc_context.strip() == "":
            enc: tuple[int, ...] = ()
        else:
            enc = tuple(encoder.encode(enc_context, add_special_tokens=True))
            enc_bos = getattr(encoder, "bos_token_id", None)
            if enc and enc_bos is not None and enc[0] != enc_bos:
                enc = (enc_bos, *enc)
    else:
        if decoder_local_only:
            enc = ()
        elif spec.encoder:
            enc_context = row["context"]
            if enc_context.strip() == "":
                enc = ()
            else:
                enc = tuple(encoder.encode(enc_context, add_special_tokens=True))
                if not enc and getattr(encoder, "bos_token_id", None) is not None:
                    enc = (encoder.bos_token_id,)
        else:
            enc = ()
        prompt_content = task_prompt(
            row,
            paired=spec.encoder is not None,
            disjoint_context=use_disjoint,
            decoder_local_only=decoder_local_only,
            local_window_lines=48 if local_window_lines is None else local_window_lines,
        )
        prefix = _apply_decoder_chat_template(decoder, prompt_content)
        prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))

    if (
        not prefix_ids
        or prefix_ids[0] != decoder.bos_token_id
        or (enc and enc[0] != encoder.bos_token_id)
    ):
        raise ValueError("context_task_special_tokens")
    if len(enc) > encoder_limit or len(prefix_ids) + len(target) - 1 > decoder_limit:
        raise ValueError("context_task_token_overflow")
    metadata: dict[str, Any] = {
        "source": row["source"],
        "task_type": row.get("task_type", "repair"),
    }
    if use_disjoint:
        metadata["disjoint_context"] = True
    if decoder_local_only:
        metadata["decoder_local_only"] = True
    return TokenizedRepairExample(
        row["instance_id"],
        enc,
        prefix_ids,
        target,
        metadata=metadata,
    )
