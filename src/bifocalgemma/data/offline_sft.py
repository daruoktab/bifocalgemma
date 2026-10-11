"""Pure paired-file editing preparation shared by both standalone SFT notebooks."""

import difflib
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import PurePosixPath


def git_text_lines(text):
    """Git counts LF delimiters only; CR/formfeed/Unicode separators stay in content."""
    pieces = text.split("\n")
    return [line + "\n" for line in pieces[:-1]] + ([pieces[-1]] if pieces[-1] else [])


def editing_record(raw: dict, *, source: str) -> dict:
    if raw.get("record_type") == "repository_patch_v1":
        return repository_editing_record(raw)
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
            git_text_lines(before),
            git_text_lines(after),
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
    """Build run metadata from paired-file rows or native repository exports.

    Repository pools intentionally store snapshot/patch identity rather than the
    paired-file convenience hashes. Normalize a copy only; never rewrite frozen
    files, cohort IDs or split membership.
    """
    manifest = []
    for raw in records:
        row = raw
        if raw.get("record_type") == "repository_patch_v1":
            row = repository_editing_record(raw)
            for key in ("pair_hash", "instruction_hash"):
                if key in raw and raw[key] != row[key]:
                    raise ValueError("repository_manifest_hash_mismatch:" + key)
        manifest.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "record_type",
                        "source_revision",
                        "snapshot_sha256",
                        "context_selection",
                    )
                    if key in row
                },
                **{
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
                },
            }
        )
    return manifest


def stable_fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def repository_editing_record(raw):
    """Consume only target-free-context training exports from the repo preparer."""
    if raw.get("role") != "train":
        raise ValueError("evaluation_record_cannot_enter_training")
    if raw.get("source", "").lower() in {
        "qurrent/res-q",
        "nuprl/canitedit",
        "bigcode/humanevalpack",
        "evalplus/repoqa",
    }:
        raise ValueError("benchmark_source_cannot_enter_training")
    instruction, context, patch = (
        raw.get("instruction"),
        raw.get("context"),
        raw.get("target_patch"),
    )
    if not all(isinstance(x, str) and x.strip() for x in (instruction, context, patch)):
        raise ValueError("missing_repository_training_fields")
    if raw.get("context_selection", {}).get("gold_used") is not False:
        raise ValueError("repository_context_policy_not_target_free")
    snapshot = raw.get("snapshot_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", snapshot):
        raise ValueError("missing_repository_snapshot_hash")
    for key in ("instance_id", "group", "commit", "repos", "source", "source_revision"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError("missing_repository_provenance")
    if not patch.startswith("diff --git "):
        raise ValueError("unsupported_repository_target_patch")
    pair_hash = hashlib.sha256((snapshot + "\0" + patch).encode()).hexdigest()
    instruction_hash = hashlib.sha256(instruction.strip().encode()).hexdigest()
    return {
        **raw,
        "path": "repository_context",
        "before": context,
        "after": "",
        "pair_hash": pair_hash,
        "instruction_hash": instruction_hash,
        "license": str(raw.get("license") or "unknown"),
    }


def curated_pool_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curated_pool(directory):
    """Honor a frozen full-pool split; verify every file and record before training."""
    from pathlib import Path

    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "bifocalgemma_curated_pool_v1":
        raise ValueError("unsupported_curated_pool_format")
    fingerprint = manifest.get("fingerprint")
    if stable_fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}) != fingerprint:
        raise ValueError("curated_pool_manifest_integrity")
    repository_pool = "snapshot_policy" in manifest.get("report", {})
    if (
        repository_pool
        and manifest["report"].get("context_source_license_policy")
        != "nearest_license_for_each_visible_file_v1"
    ):
        raise ValueError("repository_pool_context_license_audit_required")
    rows = {}
    all_ids = set()
    for split in ("train", "dev"):
        name = split + ".jsonl"
        if curated_pool_digest(root / name) != manifest["files"].get(name):
            raise ValueError("curated_pool_file_integrity:" + name)
        with (root / name).open(encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
        if not items or [r["instance_id"] for r in items] != manifest["record_ids"][split]:
            raise ValueError("curated_pool_split_identity")
        for row in items:
            if row.get("pool_split") != split or row.get("role") != "train":
                raise ValueError("curated_pool_role_or_split")
            if row["instance_id"] in all_ids:
                raise ValueError("curated_pool_duplicate_identity")
            all_ids.add(row["instance_id"])
            if row.get("record_type") == "repository_patch_v1":
                repository_editing_record(row)
                if repository_pool:
                    validate_context_license_record(row)
            elif not all(
                isinstance(row.get(k), str) and row[k]
                for k in (
                    "instruction",
                    "before",
                    "after",
                    "target_patch",
                    "source",
                    "source_revision",
                )
            ):
                raise ValueError("curated_pool_missing_fields")
        rows[split] = items
    return rows, manifest


def validate_curated_pool_limits(manifest, *, encoder_limit, decoder_limit):
    """Reject control/manifest drift before training or loading model weights.

    Baselines must have room for the same context and paired target budget.
    Changing these controls requires a newly prepared, independently named pool.
    """
    if any(
        type(value) is not int or not 1 <= value <= 131072
        for value in (encoder_limit, decoder_limit)
    ):
        raise ValueError("Invalid requested pool token limits.")
    expected = {
        "encoder": encoder_limit,
        "pair_decoder": decoder_limit,
        "baseline_decoder": encoder_limit + decoder_limit + 512,
    }
    actual = manifest.get("report", {}).get("tokens", {}).get("limits")
    if actual != expected:
        raise ValueError(
            "Pool token limits differ: restore its limits or build a new named pool. "
            f"Requested {expected}; frozen {actual}."
        )
    return dict(actual)


def validate_context_license_record(row):
    """Check complete recorded coverage; artifact hashes bind this provenance."""
    import importlib

    PERMITTED_CODE_LICENSES = importlib.import_module(
        "bifocalgemma.data.curation"
    ).PERMITTED_CODE_LICENSES

    paths = {window["path"] for window in row.get("context_selection", {}).get("windows", [])}
    evidence = row.get("context_source_code_license_evidence", [])
    recorded = [item["path"] for item in evidence]
    if not paths or set(recorded) != paths or len(set(recorded)) != len(recorded):
        raise ValueError("repository_context_license_coverage")
    for item in evidence:
        if not item.get("licenses"):
            raise ValueError("repository_context_license_missing")
        for license in item["licenses"]:
            if (
                license.get("license") not in PERMITTED_CODE_LICENSES
                or not isinstance(license.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", license.get("sha256", ""))
            ):
                raise ValueError("repository_context_license_not_admitted")
