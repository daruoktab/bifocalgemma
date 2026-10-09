"""Pinned text Git archives and licensed, issue-only repository SFT admission."""

import ast
import hashlib
import http.client
import json
import re
import tarfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import replace
from pathlib import Path, PurePosixPath

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.curation import (
    PERMITTED_CODE_LICENSES,
    CodeOverlapIndex,
    code_signature,
    export_pool,
    file_digest,
    split_connected_records,
)
from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.patches import apply_patch_to_snapshot, patch_paths, safe_repo_path
from bifocalgemma.data.repository import (
    prepare_repository_record,
    read_source_rows,
    repo_identity,
    repository_task_from_row,
)


class ArchiveSnapshotStore:
    """Read GitHub codeload at a commit; never extract or execute repository code."""

    def __init__(
        self,
        root,
        *,
        allow_network=False,
        max_archive_bytes=100_000_000,
        max_snapshot_bytes=50_000_000,
        max_files=5000,
    ):
        self.root = Path(root).resolve()
        self.allow_network = allow_network
        self.max_archive_bytes = max_archive_bytes
        self.max_snapshot_bytes = max_snapshot_bytes
        self.max_files = max_files

    def snapshot(self, task):
        identity = repo_identity(task.repo)
        digest = hashlib.sha256((identity + "@" + task.base_commit).encode()).hexdigest()
        path = self.root / (digest + ".tar.gz")
        if not path.exists():
            if not self.allow_network:
                raise FileNotFoundError("Pinned snapshot archive is missing.")
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(
                        f"https://codeload.github.com/{identity}/tar.gz/{task.base_commit}",
                        timeout=45,
                    ) as response:
                        total = 0
                        with temporary.open("wb") as handle:
                            while chunk := response.read(1024 * 1024):
                                total += len(chunk)
                                if total > self.max_archive_bytes:
                                    raise ValueError("archive_download_budget_exceeded")
                                handle.write(chunk)
                    break
                except (
                    http.client.HTTPException,
                    urllib.error.URLError,
                    TimeoutError,
                    ConnectionError,
                ) as error:
                    if attempt == 2:
                        raise ValueError("snapshot_download_transport_failure") from error
            temporary.replace(path)
        if path.stat().st_size > self.max_archive_bytes:
            raise ValueError("archive_download_budget_exceeded")
        snapshot, coverage = read_text_archive(
            path,
            identity,
            task.base_commit,
            max_bytes=self.max_snapshot_bytes,
            max_files=self.max_files,
        )
        coverage.update(archive_sha256=file_digest(path), archive_bytes=path.stat().st_size)
        return snapshot, coverage


def read_text_archive(path, repo, requested_commit, *, max_bytes=50_000_000, max_files=5000):
    files = []
    total = 0
    seen = set()
    skipped = Counter()
    root = None
    commit = None
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not parts or member.name.startswith("/") or ".." in parts:
                raise ValueError("archive_path_escape")
            if root is None:
                root = parts[0]
                commit = root.rsplit("-", 1)[-1].lower()
                if not re.fullmatch("[0-9a-f]{40}", commit) or not commit.startswith(
                    requested_commit.lower()
                ):
                    raise ValueError("archive_commit_identity")
            if parts[0] != root:
                raise ValueError("archive_multiple_roots")
            if member.isdir():
                continue
            if not member.isfile():
                skipped["links_or_nonregular"] += 1
                continue
            if len(parts) < 2:
                raise ValueError("archive_missing_relative_path")
            total += member.size
            if total > max_bytes or len(files) >= max_files:
                raise ValueError("snapshot_resource_budget_exceeded")
            relative = "/".join(parts[1:])
            try:
                safe_repo_path(relative)
            except ValueError:
                skipped["unsupported_path"] += 1
                continue
            if relative.casefold() in seen:
                raise ValueError("snapshot_case_colliding_paths")
            seen.add(relative.casefold())
            stream = archive.extractfile(member)
            assert stream is not None
            content = stream.read(member.size + 1)
            if len(content) != member.size:
                raise ValueError("archive_member_size_mismatch")
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                skipped["non_utf8"] += 1
                continue
            if "\0" in text:
                skipped["binary"] += 1
                continue
            files.append(RepositoryFile(relative, text))
    if not commit or not files:
        raise ValueError("empty_text_snapshot")
    return RepositorySnapshot(repo, commit, tuple(files)), {
        "policy": "tracked_UTF8_regular_text_only_static_SFT",
        "text_files": len(files),
        "tracked_bytes": total,
        "skipped": dict(skipped),
        "functional_checkout_complete": False,
    }


def recognize_license(text):
    spdx = re.search(r"SPDX-License-Identifier:\s*([^\n]+)", text)
    if spdx:
        value = spdx.group(1).strip().lower()
        return value if value in PERMITTED_CODE_LICENSES else "unknown"
    lower = text.lower()
    if (
        "permission is hereby granted, free of charge" in lower
        and "the software is provided" in lower
    ):
        return "mit"
    if "apache license" in lower and "version 2.0" in lower:
        return "apache-2.0"
    if (
        "redistribution and use in source and binary forms" in lower
        and "this software is provided" in lower
    ):
        return (
            "bsd-3-clause"
            if ("neither the name" in lower or "endorse or promote" in lower)
            else "bsd-2-clause"
        )
    if (
        "permission to use, copy, modify, and/or distribute this software" in lower
        and "isc" in lower
    ):
        return "isc"
    if "this is free and unencumbered software released into the public domain" in lower:
        return "unlicense"
    return "unknown"


def resolved_license(name, content, files):
    """Follow an explicit in-repository RST license include, recording its identity."""
    import posixpath

    seen = set()
    for _ in range(8):
        if name in seen:
            raise ValueError("license_include_cycle")
        seen.add(name)
        recognized = recognize_license(content)
        if recognized != "unknown":
            return name, recognized, hashlib.sha256(content.encode()).hexdigest()
        if any(
            marker in content.lower()
            for marker in (
                "proprietary",
                "gnu general public license",
                "gnu lesser general public license",
            )
        ):
            break
        includes = re.findall(r"^\.\. (?:include|literalinclude)::\s+(\S+)\s*$", content, re.M)
        if len(includes) != 1:
            break
        candidate = posixpath.normpath(str(PurePosixPath(name).parent / includes[0]))
        safe_repo_path(candidate)
        if candidate not in files:
            break
        name, content = candidate, files[candidate]
    return name, "unknown", hashlib.sha256(content.encode()).hexdigest()


def path_license_evidence(snapshot, paths):
    """Check license boundaries for source paths, independently of gold patches."""
    files = {file.path: file.content for file in snapshot.files}
    evidence = []
    for target in paths:
        target = safe_repo_path(target)
        own = files.get(target, "")
        if "SPDX-License-Identifier:" in own:
            candidates = [(target, own)]
        else:
            candidates = []
            parent = PurePosixPath(target).parent
            while True:
                folder = "" if str(parent) == "." else str(parent) + "/"
                layer = [
                    (name, content)
                    for name, content in files.items()
                    if name.startswith(folder)
                    and "/" not in name[len(folder) :]
                    and name[len(folder) :].lower()
                    in {
                        "license",
                        "license.txt",
                        "license.md",
                        "license.rst",
                        "license-mit",
                        "license-apache",
                        "license-mit.txt",
                        "license-apache.txt",
                        "copying",
                        "copying.txt",
                    }
                ]
                if layer:
                    candidates = sorted(layer)
                    break
                if str(parent) == ".":
                    break
                parent = parent.parent
        if not candidates:
            raise ValueError("missing_source_code_license")
        recognized = [resolved_license(name, content, files) for name, content in candidates]
        if any(license not in PERMITTED_CODE_LICENSES for _, license, _ in recognized):
            raise ValueError("source_code_license_not_admitted")
        evidence.append(
            {
                "path": target,
                "licenses": [
                    {"path": name, "license": license, "sha256": digest}
                    for name, license, digest in recognized
                ],
            }
        )
    return evidence


def patch_license_evidence(snapshot, patch):
    return [
        {"target_path": item["path"], "licenses": item["licenses"]}
        for item in path_license_evidence(snapshot, patch_paths(patch))
    ]


def context_license_evidence(snapshot, selection):
    """Verify every file actually shown to the encoder/baseline; no oracle paths."""
    if selection.get("gold_used") is not False:
        raise ValueError("context_license_requires_target_free_selection")
    paths = sorted({window["path"] for window in selection.get("windows", [])})
    available = {file.path for file in snapshot.files}
    if not paths or not set(paths) <= available:
        raise ValueError("context_license_missing_selected_source")
    return path_license_evidence(snapshot, paths)


class RepositoryPreparationLedger:
    """Durable per-outcome corpus checkpoints; does not store model state."""

    def __init__(self, path, settings):
        import sqlite3

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS settings(signature TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outcomes(offset INTEGER PRIMARY KEY,payload TEXT)"
        )
        signature = stable_fingerprint(settings)
        stored = self.db.execute("SELECT signature FROM settings").fetchone()
        if stored and stored[0] != signature:
            raise ValueError("Repository preparation settings changed.")
        if not stored:
            self.db.execute("INSERT INTO settings VALUES (?)", (signature,))
        self.db.commit()

    def rows(self):
        for offset, payload in self.db.execute(
            "SELECT offset,payload FROM outcomes ORDER BY offset"
        ):
            yield offset, json.loads(payload)

    def save(self, offset, payload):
        self.db.execute("INSERT INTO outcomes VALUES (?,?)", (offset, json.dumps(payload)))
        self.db.commit()

    def close(self):
        self.db.close()


def build_repository_corpus(
    source,
    rows_file,
    foundation_dir,
    tokenizer_cache,
    study_config,
    output,
    cache,
    *,
    target_count=10000,
    per_repo_limit=50,
    context_limit=4096,
    decoder_limit=2048,
    disk_budget_gib=40,
    allow_network=False,
    progress=None,
):
    """Freeze a resource-bounded corpus; every exclusion is a recorded outcome."""
    import platform
    from importlib.metadata import version

    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.data.tokenization import load_verified_tokenizer
    from bifocalgemma.training.repository_sft import tokenize_repository_record

    if source not in {"swe_fixer", "swe_rebench_v2"}:
        raise ValueError("core_source_not_admitted")
    if min(target_count, per_repo_limit, context_limit, decoder_limit, disk_budget_gib) < 1:
        raise ValueError("invalid_corpus_budget")
    _, foundation = load_curated_pool(foundation_dir)
    registry = foundation["report"]["benchmark_quarantine"]
    blocked = set(registry["blocked_repositories"])
    slugs = set(registry["crosscodeeval_repository_slugs"])
    aliases = study_config["repo_aliases"]
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite repository corpus.")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    # Read-only opening cannot clear the already-completed foundation index.
    index = CodeOverlapIndex(
        storage_path=Path(foundation_dir).parent / "curation_indexes/benchmarks.sqlite",
        read_only=True,
    )
    store = ArchiveSnapshotStore(cache / "archives", allow_network=allow_network)
    first_ref, second_ref = EXPERIMENT_SPECS[0].encoder, EXPERIMENT_SPECS[1].encoder
    assert first_ref is not None and second_ref is not None
    encoder, encoder_assets = load_verified_tokenizer(first_ref, Path(tokenizer_cache))
    other_encoder, other_encoder_assets = load_verified_tokenizer(second_ref, Path(tokenizer_cache))
    decoder, decoder_assets = load_verified_tokenizer(
        EXPERIMENT_SPECS[0].decoder, Path(tokenizer_cache)
    )
    counts = Counter()
    accepted = []
    exclusions = []
    repo_counts = Counter()
    seen = set()
    settings = {
        "source": source,
        "source_file_sha256": file_digest(rows_file),
        "foundation_fingerprint": foundation["fingerprint"],
        "target_count": target_count,
        "per_repo_limit": per_repo_limit,
        "context_limit": context_limit,
        "decoder_limit": decoder_limit,
        "disk_budget_gib": disk_budget_gib,
        "admission_version": "repository_archive_v3_context_licenses",
        "study_config_fingerprint": stable_fingerprint(study_config),
        "tokenizer_assets_fingerprint": stable_fingerprint(
            [encoder_assets, other_encoder_assets, decoder_assets]
        ),
        "runtime": {
            "python": platform.python_version(),
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
        },
    }
    ledger = RepositoryPreparationLedger(
        cache / "ledgers" / (stable_fingerprint(settings) + ".sqlite"), settings
    )
    last_offset = -1
    for offset, outcome in ledger.rows():
        last_offset = offset
        counts["examined"] += 1
        if outcome.get("lineage"):
            seen.add(outcome["lineage"])
        if "record" in outcome:
            record = outcome["record"]
            accepted.append(record)
            repo_counts[record["canonical_repos"][0]] += 1
            counts["accepted"] += 1
        else:
            exclusions.append(outcome["exclusion"])
            counts["rejected:" + outcome["exclusion"]["reason"].split(":")[0]] += 1
    if progress and last_offset >= 0:
        progress({"repository_resume_offset": last_offset, "accepted": len(accepted)})
    # Stable task order is deliberately recorded; no claim of random population sampling.
    for offset, raw in enumerate(read_source_rows(rows_file)):
        if offset <= last_offset:
            continue
        if len(accepted) >= target_count:
            break
        counts["examined"] += 1
        outcome = {}
        identifier = str(raw.get("instance_id", offset))
        try:
            task = repository_task_from_row(source, raw)
            canonical = repo_identity(task.repo, aliases)
            if task.language not in {"python", "py"}:
                raise ValueError("non_python")
            if canonical in blocked or canonical.replace("/", "-") in slugs:
                raise ValueError("benchmark_repo_quarantine")
            if repo_counts[canonical] >= per_repo_limit:
                raise ValueError("per_repository_quota")
            identity = stable_fingerprint((canonical, task.base_commit, task.target_patch))
            if identity in seen:
                raise ValueError("duplicate_issue_patch_lineage")
            seen.add(identity)
            outcome["lineage"] = identity
            snapshot, coverage = store.snapshot(task)
            license_evidence = patch_license_evidence(snapshot, task.target_patch)
            actual_licenses = sorted(
                {item["license"] for entry in license_evidence for item in entry["licenses"]}
            )
            task = replace(task, license=" AND ".join(actual_licenses))
            row, _ = prepare_repository_record(
                task,
                snapshot,
                encoder,
                blocked_repos=blocked,
                aliases=aliases,
                max_context_tokens=context_limit,
            )
            try:
                visible_license_evidence = context_license_evidence(snapshot, row["context_selection"])
            except ValueError as error:
                raise ValueError("context_" + str(error)) from error
            fixed = apply_patch_to_snapshot(snapshot, task.target_patch)
            before_files = {f.path: f.content for f in snapshot.files}
            fixed_files = {f.path: f.content for f in fixed.files}
            for path in patch_paths(task.target_patch):
                for code in (before_files.get(path, ""), fixed_files.get(path, "")):
                    if code and index.match(code):
                        raise ValueError("benchmark_code_overlap")
                if path.endswith(".py") and path in fixed_files:
                    try:
                        ast.parse(fixed_files[path])
                    except (SyntaxError, ValueError, RecursionError) as error:
                        raise ValueError("invalid_target_python_syntax") from error
            pair = tokenize_repository_record(
                row,
                EXPERIMENT_SPECS[0],
                encoder,
                decoder,
                encoder_limit=context_limit,
                decoder_limit=decoder_limit,
            )
            other_len = len(other_encoder.encode(row["context"], add_special_tokens=True))
            if other_len > context_limit:
                raise ValueError("other_encoder_overflow")
            baseline_limit = context_limit + decoder_limit + 512
            baseline = tokenize_repository_record(
                row, EXPERIMENT_SPECS[4], None, decoder, decoder_limit=baseline_limit
            )
            row.update(
                before=row["context"],
                after="",
                path="repository_context",
                canonical_repos=[canonical],
                before_signature=code_signature(row["context"]),
                snapshot_coverage=coverage,
                source_code_license_evidence=license_evidence,
                context_source_code_license_evidence=visible_license_evidence,
                dataset_license="mit" if source == "swe_fixer" else "cc-by-4.0",
                source_key=source,
                token_lengths={
                    "encoder": max(len(pair.encoder_ids), other_len),
                    "pair_decoder": len(pair.decoder_prefix_ids) + len(pair.target_ids) - 1,
                    "target": len(pair.target_ids),
                    "baseline_decoder": len(baseline.decoder_prefix_ids)
                    + len(baseline.target_ids)
                    - 1,
                },
                changed_file_count=len(patch_paths(task.target_patch)),
            )
            accepted.append(row)
            outcome["record"] = row
            repo_counts[canonical] += 1
            counts["accepted"] += 1
            if progress:
                progress(
                    {
                        "repository_corpus": source,
                        "counts": dict(counts),
                        "last_accepted": identifier,
                    }
                )
        except (
            ValueError,
            KeyError,
            FileNotFoundError,
            urllib.error.URLError,
            TimeoutError,
            tarfile.TarError,
        ) as error:
            reason = str(error)
            counts["rejected:" + reason.split(":")[0]] += 1
            exclusion = {"upstream_row": offset, "instance_id": identifier, "reason": reason}
            exclusions.append(exclusion)
            outcome["exclusion"] = exclusion
        ledger.save(offset, outcome)
        if (
            sum(p.stat().st_size for p in (cache / "archives").glob("*") if p.is_file())
            > disk_budget_gib * 2**30
        ):
            counts["stopped_disk_budget"] += 1
            break
        if len(accepted) >= target_count:
            counts["stopped_target_count"] += 1
            break
    ledger.close()
    index.close()
    train, dev, split = split_connected_records(accepted)
    report = {
        "source": source,
        "preparation_settings": settings,
        "source_file_sha256": file_digest(rows_file),
        "foundation_fingerprint": foundation["fingerprint"],
        "benchmark_quarantine": registry,
        "admission": {
            "counts": dict(counts),
            "exclusions": exclusions,
            "sampling_policy": "upstream_order_with_per_repo_quota_and_recorded_resource_stop",
        },
        "tokens": {
            "limits": {
                "encoder": context_limit,
                "pair_decoder": decoder_limit,
                "baseline_decoder": context_limit + decoder_limit + 512,
            }
        },
        "split": split,
        "training_plan": {
            "epochs": 1,
            "batch_size": 1,
            "accumulation": 4,
            "updates_full_epoch": (len(train) + 3) // 4,
            "supervised_train_tokens": sum(r["token_lengths"]["target"] for r in train),
        },
        "context_source_license_policy": "nearest_license_for_each_visible_file_v1",
        "snapshot_policy": "immutable text archive; no functional-quality verification",
        "target_count": target_count,
        "resource_budget_gib": disk_budget_gib,
        "multi_file_tasks": sum(r["changed_file_count"] > 1 for r in accepted),
    }
    return export_pool(train + dev, output, report)


def prepare_repository_corpus(
    cache,
    output,
    foundation_dir,
    tokenizer_cache,
    study_config,
    *,
    source="swe_rebench_v2",
    token=False,
    target_count=10000,
    per_repo_limit=50,
    context_limit=2048,
    decoder_limit=2048,
    disk_budget_gib=20,
    progress=None,
):
    """Pinned source fetch plus full static snapshot/context/license admission."""
    from huggingface_hub import hf_hub_download

    from bifocalgemma.data.repository import SOURCE_REVISIONS

    if source not in {"swe_fixer", "swe_rebench_v2"}:
        raise ValueError("Unsupported repository source.")
    reference = SOURCE_REVISIONS[source]
    filename = (
        "swe-fixer-train-110k.jsonl"
        if source == "swe_fixer"
        else "data/train-00000-of-00001.parquet"
    )
    path = hf_hub_download(
        reference[0],
        filename,
        revision=reference[1],
        repo_type="dataset",
        cache_dir=Path(cache) / "hf",
        token=token,
    )
    return build_repository_corpus(
        source,
        path,
        foundation_dir,
        tokenizer_cache,
        study_config,
        output,
        Path(cache) / "repository_corpus",
        target_count=target_count,
        per_repo_limit=per_repo_limit,
        context_limit=context_limit,
        decoder_limit=decoder_limit,
        disk_budget_gib=disk_budget_gib,
        allow_network=True,
        progress=progress,
    )
