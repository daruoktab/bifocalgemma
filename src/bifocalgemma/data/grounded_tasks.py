"""Deterministic, evidence-checked source retrieval supplements, never benchmark QA."""

from __future__ import annotations

import hashlib
import random
from typing import Any

VERSION = "grounded_source_proxy_v1"
TASKS = ("source_extraction", "source_mcq", "source_reconstruction")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render(source: str, line: int, task: str, options: list[str]):
    lines = source.splitlines()
    answer = lines[line - 1]
    numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(lines, 1))
    if task == "source_extraction":
        instruction = f"Return the exact source text at numbered line {line}, without its line number or Markdown."
        target = answer
    elif task == "source_mcq":
        instruction = (
            f"Which option is the exact text at numbered line {line}? Return only its letter.\n"
            + "\n".join(f"{label}. {text}" for label, text in zip("ABCD", options, strict=True))
        )
        target = "ABCD"[options.index(answer)]
    elif task == "source_reconstruction":
        lines[line - 1] = "# <MISSING_CODE>"
        numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(lines, 1))
        instruction = f"Reconstruct source line {line} replacing # <MISSING_CODE>. Return only that line, preserving indentation."
        target = answer
    else:
        raise ValueError("unsupported_grounded_task")
    return instruction, f"### Instruction\n{instruction}\n\n### Numbered source\n{numbered}", target


def generate_grounded_tasks(parent: dict[str, Any], *, seed: int = 17) -> list[dict[str, Any]]:
    """Use only an admitted, pre-fix training context and retain its frozen split.

    These mechanically answerable proxies teach source lookup and denoising.
    They do not supply human-authored software-understanding QA or new repairs.
    Original targets and evaluation questions never enter the generator.
    """
    from bifocalgemma.data.context_tasks import normalize_context_task
    from bifocalgemma.data.offline_sft import validate_context_license_record

    normalize_context_task(parent)
    validate_context_license_record(parent)
    if parent.get("record_type") != "context_sft_v1" or parent.get("pool_split") not in {
        "train",
        "dev",
    }:
        raise ValueError("grounded_parent_requires_frozen_training_split")
    source = parent["context"]
    lines = source.splitlines()
    eligible = [
        i
        for i, text in enumerate(lines, 1)
        if 8 <= len(text.strip()) <= 160
        and not text.lstrip().startswith(("#", "```"))
        and source.count(text) == 1
        and "<MISSING_CODE>" not in text
    ]
    if len(eligible) < 4:
        return []
    rng = random.Random(f"{VERSION}:{seed}:{parent['instance_id']}")
    line = rng.choice(eligible)
    answer = lines[line - 1]
    options = [answer, *rng.sample([lines[i - 1] for i in eligible if i != line], 3)]
    rng.shuffle(options)
    result = []
    for task in TASKS:
        instruction, context, target = _render(source, line, task, options)
        if task == "source_reconstruction" and target in context:
            continue
        identifier = _sha(f"{VERSION}:{seed}:{parent['instance_id']}:{task}:{line}:{_sha(source)}")
        row = {
            key: parent[key]
            for key in (
                "role",
                "pool_split",
                "group",
                "source",
                "source_key",
                "source_revision",
                "snapshot_sha256",
                "commit",
                "repos",
                "context_selection",
                "context_source_code_license_evidence",
                "license",
            )
            if key in parent
        }
        row.update(
            instance_id=identifier,
            record_type="grounded_context_sft_v1",
            task_type=task,
            instruction=instruction,
            context=context,
            target_text=target,
            derivation={
                "version": VERSION,
                "seed": seed,
                "parent_id": parent["instance_id"],
                "parent_context_sha256": _sha(source),
                "line": line,
                "answer_sha256": _sha(answer),
                "options": options,
                "source_context": source,
            },
        )
        validate_grounded_task(row)
        result.append(row)
    return result


def validate_grounded_task(row: dict[str, Any]) -> None:
    """Reconstruct the exact prompt and answer from immutable evidence, fail closed."""
    evidence = row.get("derivation", {})
    if evidence.get("version") != VERSION or row.get("task_type") not in TASKS:
        raise ValueError("invalid_grounded_derivation")
    source = evidence.get("source_context")
    if not isinstance(source, str) or _sha(source) != evidence.get("parent_context_sha256"):
        raise ValueError("grounded_source_hash_mismatch")
    line = evidence.get("line")
    if type(line) is not int or not 1 <= line <= len(source.splitlines()):
        raise ValueError("invalid_grounded_evidence_line")
    answer = source.splitlines()[line - 1]
    options = evidence.get("options")
    if (
        not isinstance(options, list)
        or len(options) != 4
        or not all(isinstance(x, str) for x in options)
        or len(set(options)) != 4
        or options.count(answer) != 1
        or any(option not in source.splitlines() for option in options)
    ):
        raise ValueError("invalid_grounded_options")
    if _sha(answer) != evidence.get("answer_sha256"):
        raise ValueError("grounded_answer_hash_mismatch")
    expected = _render(source, line, row["task_type"], options)
    if expected != (row.get("instruction"), row.get("context"), row.get("target_text")):
        raise ValueError("grounded_prompt_or_target_mismatch")
    if row["task_type"] == "source_reconstruction" and answer in row["context"]:
        raise ValueError("grounded_reconstruction_target_visible")
    identifier = _sha(
        f"{VERSION}:{evidence.get('seed')}:{evidence.get('parent_id')}:{row['task_type']}:{line}:{_sha(source)}"
    )
    if row.get("instance_id") != identifier:
        raise ValueError("grounded_identity_mismatch")
