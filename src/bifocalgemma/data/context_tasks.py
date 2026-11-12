"""Target-only SFT for explicit repository repair, localization and completion tasks."""

import hashlib
import json
import re

from bifocalgemma.contracts import TokenizedRepairExample

TASK_INSTRUCTIONS = {
    "repair": "Return only a unified diff fulfilling the instruction.",
    "localization": "Return only a JSON list of repository file paths relevant to fixing the issue.",
    "completion": "Return only the missing Python code, preserving its indentation.",
    "source_extraction": "Return only the exact requested source text.",
    "source_mcq": "Return only one option letter: A, B, C, or D.",
    "source_reconstruction": "Return only the missing source line, preserving indentation.",
}


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
    if row.get("role") != "train" or row.get("record_type") not in {"context_sft_v1", "grounded_context_sft_v1"}:
        raise ValueError("context_task_is_not_training_data")
    kind = row.get("task_type")
    if kind not in TASK_INSTRUCTIONS:
        raise ValueError("unsupported_context_task")
    if row.get("record_type") == "grounded_context_sft_v1":
        from bifocalgemma.data.grounded_tasks import validate_grounded_task

        validate_grounded_task(row)
    elif kind not in {"repair", "completion", "localization"}:
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
    expected_sources = {
        "swe_rebench_v2": "nebius/SWE-rebench-V2",
        "swe_fixer": "internlm/SWE-Fixer-Train-110K",
    }
    if row.get("benchmark") or row.get("source_key") not in {"swe_rebench_v2", "swe_fixer"}:
        raise ValueError("benchmark_source_cannot_enter_context_training")
    if row["source"] != expected_sources[row["source_key"]]:
        raise ValueError("context_source_provenance_mismatch")
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


def task_prompt(row, *, paired):
    kind = row.get("task_type", "repair")
    if kind not in TASK_INSTRUCTIONS:
        raise ValueError("unsupported_sft_task")
    prompt = TASK_INSTRUCTIONS[kind] + "\n\n"
    return prompt + row["instruction"] if paired else prompt + row["context"]


def tokenize_context_task(row, spec, encoder, decoder, *, encoder_limit, decoder_limit):
    context = row["context"]
    enc = tuple(encoder.encode(context, add_special_tokens=True)) if spec.encoder else ()
    messages = [{"role": "user", "content": task_prompt(row, paired=spec.encoder is not None)}]
    prefix = decoder.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    target_text = row.get("target_text", row.get("target_patch"))
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("empty_sft_target")
    full = decoder.apply_chat_template(
        messages + [{"role": "assistant", "content": target_text}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
    full_ids = tuple(decoder.encode(full.rstrip(), add_special_tokens=False))
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("prefix_target_boundary")
    target = full_ids[len(prefix_ids) :]
    if (
        not prefix_ids
        or prefix_ids[0] != decoder.bos_token_id
        or not target
        or target[-1] != decoder.convert_tokens_to_ids("<turn|>")
        or (enc and enc[0] != encoder.bos_token_id)
    ):
        raise ValueError("context_task_special_tokens")
    if len(enc) > encoder_limit or len(full_ids) - 1 > decoder_limit:
        raise ValueError("context_task_token_overflow")
    return TokenizedRepairExample(
        row["instance_id"],
        enc,
        prefix_ids,
        target,
        metadata={"source": row["source"], "task_type": row.get("task_type", "repair")},
    )
