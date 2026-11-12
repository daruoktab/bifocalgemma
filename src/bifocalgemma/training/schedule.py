"""Immutable, model-independent one-pass schedules and fail-closed manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "bifocalgemma_shared_schedule_v1"
ALGORITHM = "task_source_length_even_rank_sha256_v1"


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Schedule requires nonempty {name}.")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Schedule {name} must be an integer >= {minimum}.")
    return value


def _length_bin(tokens: int) -> str:
    for limit, name in ((2048, "0-2k"), (8192, "2k-8k"), (16384, "8k-16k"), (24576, "16k-24k")):
        if tokens <= limit:
            return name
    return "24k+"


def canonical_schedule_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Bind raw semantic inputs before family-specific routing or tokenization.

    ``token_lengths.encoder`` is the dataset's canonical source length, including
    for decoder-only models. No baseline prefix length or model metadata enters
    this descriptor. Missing canonical lengths fail instead of guessing a bin.
    """
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        identifier = _text(row.get("instance_id"), "instance_id")
        if identifier in seen:
            raise ValueError(f"Duplicate schedule instance_id: {identifier}")
        seen.add(identifier)
        task = _text(row.get("task_type", "repair"), "task_type")
        source = _text(row.get("source"), "source")
        instruction = _text(row.get("instruction"), "instruction")
        # Explicitly present empty values must not fall back to another target.
        target = _text(row.get("target_text", row.get("target_patch")), "target")
        if "context" in row:
            context = _text(row["context"], "context")
        else:
            path = _text(row.get("path"), "path")
            before = row.get("before")
            if not isinstance(before, str):
                raise ValueError("Schedule requires a string before field.")
            context = f"### Instruction\n{instruction}\n\n### File: {path}\n{before}"
        lengths = row.get("token_lengths")
        if not isinstance(lengths, Mapping):
            raise ValueError("Schedule requires canonical token_lengths.encoder.")
        tokens = _integer(lengths.get("encoder"), "encoder tokens", minimum=1)
        result.append(
            {
                "instance_id": identifier,
                "task": task,
                "source": source,
                "encoder_tokens": tokens,
                "length_bin": _length_bin(tokens),
                "canonical_input_sha256": _digest(
                    {
                        "instruction": instruction,
                        "context": context,
                        "task": task,
                        "source": source,
                    }
                ),
                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            }
        )
    return result


@dataclass(frozen=True)
class ScheduleRecord:
    instance_id: str
    task: str
    source: str
    encoder_tokens: int
    length_bin: str
    canonical_input_sha256: str
    target_sha256: str

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> ScheduleRecord:
        values = {name: record.get(name) for name in cls.__dataclass_fields__}
        for name in ("instance_id", "task", "source", "length_bin"):
            _text(values[name], name)
        tokens = _integer(values["encoder_tokens"], "encoder tokens", minimum=1)
        if values["length_bin"] != _length_bin(tokens):
            raise ValueError("Schedule length bin does not match canonical encoder length.")
        for name in ("canonical_input_sha256", "target_sha256"):
            digest = values[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"Invalid schedule {name}.")
        return cls(
            instance_id=_text(values["instance_id"], "instance_id"),
            task=_text(values["task"], "task"),
            source=_text(values["source"], "source"),
            encoder_tokens=tokens,
            length_bin=_text(values["length_bin"], "length_bin"),
            canonical_input_sha256=_text(
                values["canonical_input_sha256"], "canonical_input_sha256"
            ),
            target_sha256=_text(values["target_sha256"], "target_sha256"),
        )


@dataclass(frozen=True)
class SharedSchedule:
    """A fixed sequence; window indices address the original descriptor list.

    Use the builder, not direct construction. Dataset identity is canonical JSON
    rather than a mutable dict; all owned records and index arrays are immutable.
    The terminal cursor is valid for resume, but is not a consumable window.
    """

    records: tuple[ScheduleRecord, ...]
    ordered_indices: tuple[int, ...]
    dataset_identity_json: str
    seed: int
    effective_batch: int
    signature: str

    @property
    def ordered_ids(self) -> tuple[str, ...]:
        return tuple(record.instance_id for record in self.records)

    @property
    def total_examples(self) -> int:
        return len(self.records)

    @property
    def total_updates(self) -> int:
        return (self.total_examples + self.effective_batch - 1) // self.effective_batch

    def validate_cursor(self, global_update: int, *, example_cursor: int | None = None) -> int:
        update = _integer(global_update, "global update")
        if update > self.total_updates:
            raise ValueError("Schedule cursor exceeds the one-pass schedule.")
        offset = min(update * self.effective_batch, self.total_examples)
        if example_cursor is not None:
            _integer(example_cursor, "example cursor")
            if example_cursor != offset:
                raise ValueError("Schedule example cursor disagrees with global update.")
        return offset

    def window(self, global_update: int) -> tuple[int, ...]:
        offset = self.validate_cursor(global_update)
        if global_update == self.total_updates:
            raise ValueError("The completed schedule has no further window; wrapping is forbidden.")
        return self.ordered_indices[offset : offset + self.effective_batch]

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": ALGORITHM,
            "dataset_identity": json.loads(self.dataset_identity_json),
            "seed": self.seed,
            "effective_batch": self.effective_batch,
            "total_examples": self.total_examples,
            "total_updates": self.total_updates,
            "partial_window_size": self.total_examples % self.effective_batch,
            "ordered_examples": [asdict(record) for record in self.records],
        }
        return {**payload, "signature": self.signature}


def build_shared_schedule(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_identity: Mapping[str, Any],
    seed: int = 17,
    effective_batch: int = 4,
) -> SharedSchedule:
    """Distribute task/source/length strata over one pass, without replacement.

    The SHA-256 permutation and exact rational ranks avoid mutable RNG state and
    floating-point ties. Equivalent descriptors in a different input order have
    the same semantic schedule/signature and appropriately remapped indices.
    """
    _integer(seed, "seed")
    _integer(effective_batch, "effective batch", minimum=1)
    if not isinstance(dataset_identity, Mapping) or not dataset_identity:
        raise ValueError("A nonempty pinned dataset identity is required.")
    identity_json = _json(dict(dataset_identity))
    buckets: dict[tuple[str, str, str], list[tuple[int, ScheduleRecord]]] = defaultdict(list)
    seen: set[str] = set()
    for index, raw in enumerate(records):
        record = ScheduleRecord.from_mapping(raw)
        if record.instance_id in seen:
            raise ValueError(f"Duplicate schedule instance_id: {record.instance_id}")
        seen.add(record.instance_id)
        buckets[(record.task, record.source, record.length_bin)].append((index, record))
    if not seen:
        raise ValueError("Cannot build an empty training schedule.")
    ranked: list[tuple[Fraction, str, int, ScheduleRecord]] = []
    for key, bucket in sorted(buckets.items()):
        bucket.sort(
            key=lambda item: (_digest([seed, key, item[1].instance_id]), item[1].instance_id)
        )
        for position, (index, record) in enumerate(bucket):
            ranked.append(
                (Fraction(2 * position + 1, 2 * len(bucket)), record.instance_id, index, record)
            )
    ranked.sort(key=lambda item: (item[0], item[1]))
    schedule = SharedSchedule(
        records=tuple(item[3] for item in ranked),
        ordered_indices=tuple(item[2] for item in ranked),
        dataset_identity_json=identity_json,
        seed=seed,
        effective_batch=effective_batch,
        signature="",
    )
    payload = schedule.to_manifest()
    del payload["signature"]
    return SharedSchedule(
        records=schedule.records,
        ordered_indices=schedule.ordered_indices,
        dataset_identity_json=identity_json,
        seed=seed,
        effective_batch=effective_batch,
        signature=_digest(payload),
    )


def persist_shared_schedule(path: str | Path, schedule: SharedSchedule) -> None:
    """Atomically publish a complete manifest, never replacing an existing one.

    A flushed same-directory temporary file is published with a no-clobber
    Windows rename or POSIX hard link. A concurrent creator must publish the
    exact same manifest. Unsupported filesystems fail rather than overwrite.
    """
    destination = Path(path)
    expected = schedule.to_manifest()
    payload = {key: value for key, value in expected.items() if key != "signature"}
    if _digest(payload) != schedule.signature:
        raise ValueError("Shared schedule signature is invalid.")

    def validate_existing() -> None:
        try:
            actual = json.loads(destination.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ValueError("Existing shared schedule manifest is invalid.") from error
        if actual != expected:
            raise ValueError("Existing shared schedule manifest does not match; create a new run.")

    if destination.exists():
        validate_existing()
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_json(expected) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                # Unlike POSIX rename, Windows rename fails if the target exists.
                os.rename(temporary, destination)
            else:
                os.link(temporary, destination)
        except FileExistsError:
            validate_existing()
    finally:
        temporary.unlink(missing_ok=True)
