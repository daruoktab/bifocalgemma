"""Pure paired-file editing preparation shared by both standalone SFT notebooks."""

import difflib
import hashlib
import json
import random
from collections import Counter
from pathlib import PurePosixPath


def editing_record(raw: dict, *, source: str) -> dict:
    before, after = raw.get("old_contents"), raw.get("new_contents")
    instruction = raw.get("instruction") if source == "editpackft" else raw.get("subject")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("missing_instruction")
    if not isinstance(before, str) or not isinstance(after, str) or not before or before == after:
        raise ValueError("empty_or_unchanged_code")
    path = raw.get("old_file")
    if not isinstance(path, str) or raw.get("new_file") != path:
        raise ValueError("unsupported_rename_add_delete")
    canonical = PurePosixPath(path)
    if (
        canonical.is_absolute()
        or ".." in canonical.parts
        or "\\" in path
        or ":" in path
        or canonical.as_posix() != path
        or any(part.startswith(".") for part in canonical.parts)
        or any(char.isspace() for char in path)
    ):
        raise ValueError("unsupported_path")
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="\n",
        )
    )
    patch = f"diff --git a/{path} b/{path}\n"
    for line in lines:
        patch += line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
    pair_hash = hashlib.sha256((before + "\0" + after).encode("utf-8")).hexdigest()
    commit = str(raw.get("commit") or pair_hash)
    instruction_hash = hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest()
    identifier = hashlib.sha256(
        (source + commit + path + pair_hash + instruction_hash).encode("utf-8")
    ).hexdigest()
    return {
        "instance_id": identifier,
        "group": commit,
        "pair_hash": pair_hash,
        "instruction_hash": instruction_hash,
        "path": path,
        "before": before,
        "after": after,
        "instruction": instruction.strip(),
        "target_patch": patch,
        "source": source,
        "commit": commit,
        "repos": str(raw.get("repos") or ""),
        "license": str(raw.get("license") or "unknown"),
    }


def prepare_editing_records(raw_records, *, source, raw_limit=10000):
    if raw_limit < 2:
        raise ValueError("Need at least two raw records.")
    accepted, seen, filters = [], set(), Counter()
    for index, raw in enumerate(raw_records):
        if index >= raw_limit:
            break
        try:
            item = editing_record(raw, source=source)
        except ValueError as error:
            filters[str(error)] += 1
            continue
        if item["pair_hash"] in seen:
            filters["duplicate_before_after"] += 1
            continue
        seen.add(item["pair_hash"])
        accepted.append(item)
    return accepted, dict(filters)


def split_editing_records(records, *, seed=17, dev_fraction=0.1, train_limit=2000, dev_limit=200):
    if not 0 < dev_fraction < 1 or min(train_limit, dev_limit) < 1:
        raise ValueError("Invalid split/sample limits.")
    groups = sorted({record["group"] for record in records})
    if len(groups) < 2:
        raise ValueError("Need at least two distinct commit groups for a held-out split.")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * dev_fraction)))
    dev_groups = set(groups[:count])
    train = [r for r in records if r["group"] not in dev_groups][:train_limit]
    dev = [r for r in records if r["group"] in dev_groups][:dev_limit]
    if not train or not dev:
        raise ValueError("Empty train/dev split.")
    assert not ({r["group"] for r in train} & {r["group"] for r in dev})
    return train, dev


def editing_manifest(records):
    return [
        {
            key: row[key]
            for key in (
                "instance_id",
                "group",
                "pair_hash",
                "instruction_hash",
                "path",
                "source",
                "commit",
                "repos",
                "license",
            )
        }
        for row in records
    ]


def stable_fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
