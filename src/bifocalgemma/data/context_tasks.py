"""Target-only SFT for explicit repository repair, localization, completion, and v4 disjoint context tasks."""

from __future__ import annotations

import hashlib
import json
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
    "retention": "Follow the instruction accurately and adhere strictly to all formatting constraints.",
    "instruction": "Follow the instruction accurately and adhere strictly to all formatting constraints.",
}

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
    parts = [p.strip() for p in re.split(r"\s+and\s+|\s+or\s+|,", normalized) if p.strip()]
    allowed = PERMITTED_V4_CODE_LICENSES if code_only else PERMITTED_V4_DATASET_LICENSES
    return bool(parts) and all(part in allowed for part in parts)


_NUMBERED_LINE_RE = re.compile(r"^(\d+):\s?(.*)$")
_FILE_SECTION_HEADER_RE = re.compile(
    r"^### File:\s*([^\n(]+?)(?:\s*\(lines\s+(\d+)-(\d+)\))?\s*$", re.MULTILINE
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
        if "\n### MISSING CODE" in after_pre:
            prefix_text, _, after_missing = after_pre.partition("\n### MISSING CODE")
            if "### SUFFIX\n" in after_missing:
                _, _, suffix_text = after_missing.partition("### SUFFIX\n")
        elif "### SUFFIX\n" in after_pre:
            prefix_text, _, suffix_text = after_pre.partition("### SUFFIX\n")
        else:
            prefix_text = after_pre

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
    instruction = str(row.get("instruction") or "")
    raw_context = str(row.get("context") or "")
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
    if "### PREFIX\n" in body and ("### SUFFIX\n" in body or "### MISSING CODE" in body):
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

    if cursor_entry_idx is None:
        # Never touch gold/target fields! Match path or instruction keywords only.
        hint_path = row.get("path")
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
    if cursor_entries:
        if "# <MISSING_CODE>" in cursor_entries[cursor_entry_idx][1]:
            local_start = max(0, cursor_entry_idx - local_window_lines)
            local_end = min(len(cursor_entries), cursor_entry_idx + local_window_lines + 1)
        else:
            # For repair/localization anchored at file header or matched line
            if cursor_entry_idx == 0:
                local_start = 0
                local_end = min(len(cursor_entries), local_window_lines)
            else:
                local_start = max(0, cursor_entry_idx - local_window_lines)
                local_end = min(len(cursor_entries), cursor_entry_idx + local_window_lines + 1)
        local_entries = cursor_entries[local_start:local_end]
        distant_before = cursor_entries[:local_start]
        distant_after = cursor_entries[local_end:]
    else:
        local_entries, distant_before, distant_after = [], [], []

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
        return bool(disjoint_context)
    return bool(
        row.get("disjoint_context") is True
        or row.get("context_format") == "v4"
        or row.get("record_type") == "context_sft_v4"
    )


def task_prompt(
    row,
    *,
    paired,
    disjoint_context: bool | None = None,
    decoder_local_only: bool = False,
    local_window_lines: int = 48,
):
    kind = row.get("task_type", "repair")
    if kind not in TASK_INSTRUCTIONS:
        raise ValueError("unsupported_sft_task")
    use_disjoint = _use_disjoint_mode(row, disjoint_context)
    if not use_disjoint and not decoder_local_only:
        prompt = TASK_INSTRUCTIONS[kind] + "\n\n"
        return prompt + row["instruction"] if paired else prompt + row["context"]
    split = split_disjoint_context(row, local_window_lines=local_window_lines)
    header = TASK_INSTRUCTIONS[kind] + "\n\n" + row["instruction"]
    if decoder_local_only or paired:
        return (
            f"{header}\n\n{split.decoder_local_context}"
            if split.decoder_local_context
            else header
        )
    return f"{header}\n\n{split.full_context}" if split.full_context else header


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
    local_window_lines: int = 48,
):
    use_disjoint = _use_disjoint_mode(row, disjoint_context)
    if decoder_local_only is False:
        decoder_local_only = bool(row.get("decoder_local_only", False))
    if local_window_lines == 48 and isinstance(row.get("local_window_lines"), int):
        local_window_lines = row["local_window_lines"]

    if decoder_local_only:
        enc: tuple[int, ...] = ()
    elif spec.encoder:
        if use_disjoint:
            enc_context = split_disjoint_context(
                row, local_window_lines=local_window_lines
            ).encoder_context
        else:
            enc_context = row["context"]
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
        local_window_lines=local_window_lines,
    )
    messages = [{"role": "user", "content": prompt_content}]
    prefix = decoder.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    target_text = row.get("target_text", row.get("target_patch"))
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("empty_sft_target")
    prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
    eot_token_id = _resolve_eot_token_id(decoder)
    # Encode target_text directly without apply_chat_template (which runs `| trim`) or
    # `.rstrip()`, preserving leading indentation and trailing newlines verbatim.
    target_body_ids = tuple(decoder.encode(target_text, add_special_tokens=False))
    if (
        not prefix_ids
        or prefix_ids[0] != decoder.bos_token_id
        or eot_token_id is None
        or not target_body_ids
        or (enc and enc[0] != encoder.bos_token_id)
    ):
        raise ValueError("context_task_special_tokens")
    target = target_body_ids + (eot_token_id,)
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
