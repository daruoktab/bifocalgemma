"""Auditable full-corpus editing admission; never executes source code."""

import hashlib
import io
import json
import re
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict

from bifocalgemma.data.offline_sft import editing_record, stable_fingerprint
from bifocalgemma.data.python_quality import (
    SYNTAX_DIAGNOSTICS_POLICY,
    count_syntax_warnings,
    empty_syntax_warning_summary,
    inspect_python_target,
)
from bifocalgemma.data.repository import repo_identity

NORMALIZATION_POLICY = "python_structure_preserving_v2"

SOURCE_REFS = {
    "editpackft": {
        "id": "nuprl/EditPackFT",
        "revision": "067b4c882f44eca19329fedd275438fc3c060f3f",
        "dataset_license": "mit",
    },
    "commitpackft_python": {
        "id": "bigcode/commitpackft",
        "revision": "fc56fe33c030c6daa414c2b112c932b8eed085e6",
        "dataset_license": "mit",
    },
    "nextcoder": {
        "id": "microsoft/NextCoderDataset",
        "revision": "924e23fa2fae2d2db568c3f8f18015b616a53715",
        "dataset_license": "mit",
    },
}
PERMITTED_CODE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc0-1.0",
    "unlicense",
}


def code_tokens(code):
    """Ignore comments/indent width; retain statement and block boundaries."""
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.COMMENT,
        tokenize.NL,
    }
    try:
        return tuple(
            (
                tokenize.tok_name[t.type]
                if t.type in {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
                else t.string
            )
            for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type not in ignored
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return ("UNPARSEABLE_SOURCE", code)


def code_signature(code):
    return stable_fingerprint(code_tokens(code))


def code_shingles(code, width=8):
    tokens = code_tokens(code)
    # Short commonplace fragments are not enough to assert benchmark overlap.
    if len(tokens) < 32:
        return frozenset()
    return frozenset(
        int(hashlib.sha256("\0".join(tokens[i : i + width]).encode()).hexdigest()[:15], 16)
        for i in range(len(tokens) - width + 1)
    )


class CodeOverlapMatch(TypedDict):
    kind: str
    reference: str
    similarity: float


class CodeOverlapIndex:
    """Exact checks and token-shingle Jaccard, optionally backed by SQLite.

    Benchmark queries use every shingle. Within-pool near dedup can use a
    minimum-hash candidate heuristic; semantic rewrites remain outside scope.
    """

    def __init__(
        self, *, threshold=0.8, candidate_shingles=None, storage_path=None, read_only=False
    ):
        self.threshold = threshold
        self.candidate_shingles = candidate_shingles
        self.read_only = read_only
        # Legacy on-disk indexes do not attest complete postings. Read-only
        # readers conservatively avoid the intersection-count bound.
        self.full_postings = not candidate_shingles and not read_only
        self.exact = {}
        self.shingles = {}
        self.inverted = defaultdict(set)
        self.db = None
        if storage_path is not None:
            import sqlite3

            path = Path(storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if read_only:
                self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                try:
                    policy = self.db.execute(
                        "SELECT value FROM metadata WHERE key='normalization'"
                    ).fetchone()
                except sqlite3.OperationalError as error:
                    self.db.close()
                    raise ValueError(
                        "benchmark_index_normalization_mismatch: rebuild the index"
                    ) from error
                if policy != (NORMALIZATION_POLICY,):
                    self.db.close()
                    raise ValueError("benchmark_index_normalization_mismatch: rebuild the index")
                for table in ("exact", "codes", "postings"):
                    if self.db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] < 1:
                        raise ValueError("Incomplete benchmark index.")
                attestation = self.db.execute(
                    "SELECT value FROM metadata WHERE key='full_postings'"
                ).fetchone()
                if attestation not in (("true",), ("false",)):
                    raise ValueError("Benchmark index has no completed-build attestation.")
                self.full_postings = not candidate_shingles and self.db.execute(
                    "SELECT value FROM metadata WHERE key='full_postings'"
                ).fetchone() == ("true",)
                return
            self.db = sqlite3.connect(path)
            # This is a disposable index, never a raw dataset or curated export.
            self.db.executescript(
                "DROP TABLE IF EXISTS codes; DROP TABLE IF EXISTS postings; DROP TABLE IF EXISTS exact; CREATE TABLE codes(identity TEXT PRIMARY KEY, grams BLOB); CREATE TABLE exact(signature TEXT PRIMARY KEY, identity TEXT); CREATE TABLE postings(gram INTEGER,identity TEXT,PRIMARY KEY(gram,identity)) WITHOUT ROWID;"
            )
            self.db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
            self.db.execute("DELETE FROM metadata WHERE key='full_postings'")
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('normalization',?)",
                (NORMALIZATION_POLICY,),
            )

    def add(self, identity, code):
        import struct

        signature = code_signature(code)
        grams = code_shingles(code)
        indexed = sorted(grams)[: self.candidate_shingles] if self.candidate_shingles else grams
        if self.candidate_shingles:
            self.full_postings = False
        if self.db is not None:
            self.db.execute("INSERT OR IGNORE INTO exact VALUES (?,?)", (signature, identity))
            blob = struct.pack("<" + "Q" * len(grams), *sorted(grams))
            self.db.execute("INSERT OR REPLACE INTO codes VALUES (?,?)", (identity, blob))
            self.db.executemany(
                "INSERT OR IGNORE INTO postings VALUES (?,?)",
                ((gram, identity) for gram in indexed),
            )
        else:
            self.exact.setdefault(signature, identity)
            self.shingles[identity] = grams
            for gram in indexed:
                self.inverted[gram].add(identity)

    def match(self, code) -> CodeOverlapMatch | None:
        import struct

        signature = code_signature(code)
        if self.db is not None:
            row = self.db.execute(
                "SELECT identity FROM exact WHERE signature=?", (signature,)
            ).fetchone()
            exact = row[0] if row else None
        else:
            exact = self.exact.get(signature)
        if exact is not None:
            return {"kind": "normalized_exact", "reference": exact, "similarity": 1.0}
        grams = code_shingles(code)
        if not grams:
            return None
        candidates = (
            sorted(grams)[: self.candidate_shingles] if self.candidate_shingles else list(grams)
        )
        counts: Counter[str] = Counter()
        if self.db is not None:
            for offset in range(0, len(candidates), 800):
                chunk = candidates[offset : offset + 800]
                placeholders = ",".join("?" * len(chunk))
                rows = self.db.execute(
                    "SELECT identity,COUNT(*) FROM postings WHERE gram IN ("
                    + placeholders
                    + ") GROUP BY identity",
                    chunk,
                )
                for identity, count in rows:
                    counts[identity] += count
        else:
            counts.update(
                identity for gram in candidates for identity in self.inverted.get(gram, ())
            )
        for identity, shared_count in counts.most_common():
            # With full postings, shared_count is the exact intersection. Since
            # union >= len(grams), this upper bound cannot reject a valid match.
            # A minimum-hash index observes only partial counts: never bound it.
            if (
                self.full_postings
                and not self.candidate_shingles
                and shared_count / len(grams) < self.threshold
            ):
                continue
            if self.db is not None:
                blob = self.db.execute(
                    "SELECT grams FROM codes WHERE identity=?", (identity,)
                ).fetchone()[0]
                other = set(struct.unpack("<" + "Q" * (len(blob) // 8), blob))
            else:
                other = self.shingles[identity]
            intersection = len(grams & other)
            similarity = intersection / (len(grams) + len(other) - intersection)
            if similarity >= self.threshold:
                return {
                    "kind": "token_shingle_jaccard",
                    "reference": identity,
                    "similarity": similarity,
                }
        return None

    def close(self):
        if self.db is not None:
            if not self.read_only:
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('full_postings',?)",
                    ("true" if self.full_postings else "false",),
                )
            self.db.commit()
            self.db.close()
            self.db = None


def nextcoder_row(raw, *, offset):
    prompt, completion = raw.get("prompt"), raw.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError("nextcoder_missing_strings")
    if not re.match(r"Rewrite the given python program\b", prompt, flags=re.I):
        raise ValueError("non_python")
    fence = re.search(r"```(?:python|py)\s*\n(.*?)\n```\s*$", prompt, flags=re.S)
    if fence is None or prompt.count("```") != 2:
        raise ValueError("nextcoder_ambiguous_input_fence")
    before = fence.group(1) + "\n"
    output = re.fullmatch(r"\s*```(?:python|py)?\s*\n(.*?)\n```\s*", completion, flags=re.S)
    if output is None:
        # A plain full-code response is allowed, mixed explanation/fences are not.
        if "```" in completion:
            raise ValueError("nextcoder_ambiguous_output_fence")
        after = completion
    else:
        after = output.group(1) + "\n"
    instruction = prompt[: fence.start()]
    instruction = re.sub(
        r"^Rewrite the given python program as per the following instruction\.\s*",
        "",
        instruction,
        flags=re.I,
    )
    instruction = re.sub(
        r"Write the entire code and no other text in the response\.\s*$", "", instruction
    ).strip()
    # This source has no original repository provenance. Never invent a repo name.
    return {
        "old_file": "solution.py",
        "new_file": "solution.py",
        "old_contents": before,
        "new_contents": after,
        "instruction": instruction,
        "commit": code_signature(before),
        "license": "unknown",
        "dataset_license": "mit",
        "provenance_type": "synthetic_edit_from_unresolved_seed",
        "repos": "",
        "upstream_row": offset,
    }


def curate_paired_sources(
    sources,
    benchmark_index,
    *,
    blocked_repos,
    aliases=None,
    blocked_repo_slugs=(),
    progress=None,
    near_index_path=None,
):
    """Read all rows; exact/normalized edit dedup and static Python quality gates."""
    accepted = []
    rejected = []
    stats = {}
    warning_stats = {}
    seen_pairs = {}
    near_edits = CodeOverlapIndex(
        threshold=0.95, candidate_shingles=16, storage_path=near_index_path
    )
    blocked = {repo_identity(x, aliases) for x in blocked_repos}
    for source, rows in sources:
        if source not in SOURCE_REFS:
            raise ValueError("source_not_admitted: " + source)
        counts: Counter[str] = Counter()
        warning_summary = empty_syntax_warning_summary()
        for offset, raw in enumerate(rows):
            counts["raw"] += 1
            syntax_diagnostics = []
            try:
                normalized = nextcoder_row(raw, offset=offset) if source == "nextcoder" else raw
                row = editing_record(
                    normalized, source="editpackft" if source != "commitpackft_python" else source
                )
                if not row["path"].endswith(".py"):
                    raise ValueError("non_python_path")
                license_name = row["license"].strip().lower()
                if license_name not in PERMITTED_CODE_LICENSES and not (
                    source == "nextcoder" and license_name == "unknown"
                ):
                    raise ValueError("code_license_not_admitted:" + license_name)
                repos = [
                    repo_identity(x.strip(), aliases)
                    for x in re.split(r"[,;]", row["repos"])
                    if x.strip()
                ]
                if source != "nextcoder" and not repos:
                    raise ValueError("missing_repository_provenance")
                if set(repos) & blocked or any(
                    repo.replace("/", "-") in blocked_repo_slugs for repo in repos
                ):
                    raise ValueError("benchmark_repository_overlap")
                for code in (row["before"], row["after"]):
                    if "\0" in code:
                        raise ValueError("null_byte")
                    if code == row["after"]:
                        syntax_diagnostics = inspect_python_target(
                            code, filename=f"{source}[{offset}]:{row['path']}"
                        )
                        count_syntax_warnings(warning_summary, syntax_diagnostics)
                    overlap = benchmark_index.match(code)
                    if overlap:
                        raise ValueError(
                            "benchmark_code_overlap:" + overlap["kind"] + ":" + overlap["reference"]
                        )
                pair = stable_fingerprint(
                    (code_signature(row["before"]), code_signature(row["after"]))
                )
                if pair in seen_pairs:
                    raise ValueError("duplicate_normalized_edit:" + seen_pairs[pair])
                combined = row["before"] + "\n" + row["after"]
                near = near_edits.match(combined)
                if near:
                    raise ValueError("near_duplicate_edit:" + near["reference"])
                near_edits.add(source + ":" + str(offset), combined)
                seen_pairs[pair] = source + ":" + str(offset)
                row.update(
                    {
                        "source": SOURCE_REFS[source]["id"],
                        "source_key": source,
                        "source_revision": SOURCE_REFS[source]["revision"],
                        "dataset_license": SOURCE_REFS[source]["dataset_license"],
                        "code_license": license_name,
                        "code_license_verification": "unverified_original_seed"
                        if source == "nextcoder"
                        else "upstream_row_declared_not_checkout_verified",
                        "provenance_type": "synthetic_edit_from_unresolved_seed"
                        if source == "nextcoder"
                        else "upstream_git_export",
                        "git_commit": None if source == "nextcoder" else row["commit"],
                        "seed_program_hash": row["commit"] if source == "nextcoder" else None,
                        "role": "train",
                        "upstream_row": offset,
                        "canonical_repos": repos,
                        "before_signature": code_signature(row["before"]),
                        "normalized_pair_hash": pair,
                        "quality_gate": "static_python_parse_not_functional_validation",
                        "target_syntax_warnings": syntax_diagnostics,
                        "quality_flags": ["target_has_syntax_warning"]
                        if syntax_diagnostics
                        else [],
                    }
                )
                row["instance_id"] = stable_fingerprint(
                    (row["source"], row["source_revision"], pair)
                )
                accepted.append(row)
                counts["accepted"] += 1
                count_syntax_warnings(warning_summary, syntax_diagnostics, accepted=True)
            except ValueError as error:
                reason = str(error)
                counts["rejected:" + reason.split(":")[0]] += 1
                rejected.append(
                    {
                        "source": source,
                        "upstream_row": offset,
                        "reason": reason,
                        "target_syntax_warnings": syntax_diagnostics,
                    }
                )
            if progress and counts["raw"] % 5000 == 0:
                progress(
                    {"source": source, "counts": dict(counts), "syntax_warnings": warning_summary}
                )
        stats[source] = dict(counts)
        warning_stats[source] = warning_summary
    near_edits.close()
    return accepted, {
        "source_counts": stats,
        "syntax_diagnostics_policy": SYNTAX_DIAGNOSTICS_POLICY,
        "syntax_warning_summary": warning_stats,
        "rejected": rejected,
        "near_duplicate_policy": "exact normalized edit pairs plus within-pool Jaccard >=0.95 (16 minimum-hash shingle candidate heuristic); near benchmark overlap uses token-shingle Jaccard >=0.8; semantic similarity not certified",
    }


def split_connected_records(records, *, seed=17, dev_fraction=0.1):
    """Repo, commit and before-seed connected components are indivisible."""
    if not 0 < dev_fraction < 1:
        raise ValueError("invalid_dev_fraction")
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(records):
        keys = ["before:" + row["before_signature"]]
        keys += ["repo:" + repo for repo in row["canonical_repos"]]
        # Commit hashes shared across forks/sources must remain together.
        keys += ["commit:" + row["commit"]]
        keys += ["structure:" + key for key in row.get("split_keys", [])]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(records):
        components[find(i)].append(row)
    if len(components) < 2:
        raise ValueError("cannot_make_disjoint_split")
    ordered = sorted(
        components.values(),
        key=lambda rows: stable_fingerprint((seed, sorted(r["instance_id"] for r in rows))),
    )
    target = max(1, round(len(records) * dev_fraction))
    dev = []
    train = []
    for rows in ordered:
        destination = dev if len(dev) < target and len(dev) + len(rows) < len(records) else train
        destination.extend(rows)
    if not train or not dev:
        raise ValueError("empty_connected_split")
    for split, rows in (("train", train), ("dev", dev)):
        for row in rows:
            row["pool_split"] = split
    return (
        train,
        dev,
        {
            "policy": "repo_commit_before_seed_connected_components",
            "components": len(components),
            "train": len(train),
            "dev": len(dev),
            "seed": seed,
            "dev_fraction_target": dev_fraction,
        },
    )


def read_arrow_rows(path):
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        try:
            reader = pa.ipc.open_stream(source)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_file(source)
        if hasattr(reader, "num_record_batches"):
            for i in range(reader.num_record_batches):
                yield from reader.get_batch(i).to_pylist()
        else:
            for batch in reader:
                yield from batch.to_pylist()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_profile(
    records,
    encoder,
    decoder,
    *,
    encoder_limit=2048,
    decoder_limit=2048,
    baseline_limit=4608,
    progress=None,
    additional_encoder=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Profile every unique row before freezing one paired/baseline cohort."""
    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.training.repository_sft import tokenize_repository_record

    accepted, rejects, lengths = [], [], []
    for i, row in enumerate(records):
        context = (
            f"### Instruction\n{row['instruction']}\n\n### File: {row['path']}\n{row['before']}"
        )
        item = {**row, "context": context}
        try:
            pair = tokenize_repository_record(
                item,
                EXPERIMENT_SPECS[0],
                encoder,
                decoder,
                encoder_limit=131072,
                decoder_limit=131072,
            )
            baseline = tokenize_repository_record(
                item, EXPERIMENT_SPECS[4], None, decoder, encoder_limit=131072, decoder_limit=131072
            )
            sizes = {
                "encoder": len(pair.encoder_ids),
                "pair_decoder": len(pair.decoder_prefix_ids) + len(pair.target_ids) - 1,
                "target": len(pair.target_ids),
                "baseline_decoder": len(baseline.decoder_prefix_ids) + len(baseline.target_ids) - 1,
            }
            if additional_encoder is not None:
                alternate = len(additional_encoder.encode(context, add_special_tokens=True))
                sizes["encoder_E2B"] = sizes["encoder"]
                sizes["encoder_E4B"] = alternate
                sizes["encoder"] = max(sizes["encoder"], alternate)
            lengths.append(sizes)
            if (
                sizes["encoder"] > encoder_limit
                or sizes["pair_decoder"] > decoder_limit
                or sizes["baseline_decoder"] > baseline_limit
            ):
                rejects.append(
                    {
                        "instance_id": row["instance_id"],
                        "reason": "token_overflow",
                        "lengths": sizes,
                    }
                )
            else:
                row["token_lengths"] = sizes
                accepted.append(row)
        except ValueError as error:
            rejects.append({"instance_id": row["instance_id"], "reason": str(error)})
        if progress and (i + 1) % 5000 == 0:
            progress({"profiled": i + 1, "accepted": len(accepted)})
    quantiles = {}
    if lengths:
        for field in lengths[0]:
            values = sorted(x[field] for x in lengths)
            quantiles[field] = {
                label: values[int((len(values) - 1) * fraction)]
                for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
            }
    return accepted, {
        "profiled": len(records),
        "accepted": len(accepted),
        "rejected": rejects,
        "quantiles_before_limit": quantiles,
        "limits": {
            "encoder": encoder_limit,
            "pair_decoder": decoder_limit,
            "baseline_decoder": baseline_limit,
        },
        "supervised_tokens": sum(r["token_lengths"]["target"] for r in accepted),
    }


def export_pool(records, output, report):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in ("train", "dev"):
        path = root / (split + ".jsonl")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in records:
                if row["pool_split"] == split:
                    # Retain normalized raw fields for the standalone notebook adapter.
                    raw = {
                        **row,
                        "old_contents": row["before"],
                        "new_contents": row["after"],
                        "old_file": row["path"],
                        "new_file": row["path"],
                    }
                    handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
        files[path.name] = file_digest(path)
    manifest = {
        "format": "bifocalgemma_curated_pool_v1",
        "files": files,
        "report": report,
        "record_ids": {
            split: [r["instance_id"] for r in records if r["pool_split"] == split]
            for split in ("train", "dev")
        },
    }
    manifest["fingerprint"] = stable_fingerprint(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
