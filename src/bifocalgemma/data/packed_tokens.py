"""Lossless int32 token caches: only the current microbatches become Python tuples."""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import numpy as np

from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.curation import file_digest
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.training.stages import atomic_json


class PackedTokenizedDataset(Sequence[TokenizedRepairExample]):
    def __init__(self, directory, split):
        self.directory = Path(directory)
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.rows = index[split]
        self.tokens = np.memmap(self.directory / "tokens.bin", dtype="<i4", mode="r")

    def __len__(self):
        return len(self.rows)

    @overload
    def __getitem__(self, index: int) -> TokenizedRepairExample: ...

    @overload
    def __getitem__(self, index: slice) -> list[TokenizedRepairExample]: ...

    def __getitem__(
        self, index: int | slice
    ) -> TokenizedRepairExample | list[TokenizedRepairExample]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        arrays = [
            tuple(int(v) for v in self.tokens[start : start + length])
            for start, length in row["offsets"]
        ]
        return TokenizedRepairExample(
            row["instance_id"], arrays[0], arrays[1], arrays[2], metadata=row["metadata"]
        )

    def close(self):
        mapping = getattr(self.tokens, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def extreme_examples(self):
        encoder = max(range(len(self.rows)), key=lambda i: self.rows[i]["offsets"][0][1])
        decoder = max(
            range(len(self.rows)), key=lambda i: sum(v[1] for v in self.rows[i]["offsets"][1:])
        )

        def cost(i):
            e = self.rows[i]["offsets"][0][1]
            d = sum(v[1] for v in self.rows[i]["offsets"][1:]) - 1
            return e * e + d * (e + d)

        joint = max(range(len(self.rows)), key=cost)
        return self[encoder], self[decoder], self[joint]


def open_packed_cache(directory, identity):
    directory = Path(directory)
    receipt = json.loads((directory / "READY.json").read_text(encoding="utf-8"))
    if receipt.get("identity") != stable_fingerprint(identity):
        raise ValueError("Packed token cache identity changed.")
    for name in ("tokens.bin", "index.json"):
        if file_digest(directory / name) != receipt["files"][name]:
            raise ValueError("Packed token cache checksum mismatch.")
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    size = (directory / "tokens.bin").stat().st_size // 4
    ids = set()
    for split in ("train", "dev"):
        for row in index[split]:
            if row["instance_id"] in ids or len(row["offsets"]) != 3:
                raise ValueError("Packed token cache duplicate or invalid index.")
            ids.add(row["instance_id"])
            for start, length in row["offsets"]:
                if (
                    type(start) is not int
                    or type(length) is not int
                    or start < 0
                    or length < 0
                    or start + length > size
                ):
                    raise ValueError("Packed token cache invalid offset.")
    return {split: PackedTokenizedDataset(directory, split) for split in ("train", "dev")}, receipt


def build_packed_cache(directory, identity, rows, tokenize, *, progress=None):
    directory = Path(directory)
    if directory.exists():
        return open_packed_cache(directory, identity)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as name:
        stage = Path(name)
        index = {"train": [], "dev": []}
        offset = 0
        with (stage / "tokens.bin").open("wb") as stream:
            for split, records in rows.items():
                for number, row in enumerate(records, 1):
                    item = tokenize(row)
                    offsets = []
                    for values in (item.encoder_ids, item.decoder_prefix_ids, item.target_ids):
                        if values and (min(values) < 0 or max(values) > 2**31 - 1):
                            raise ValueError(
                                "Token IDs do not fit lossless nonnegative int32 storage."
                            )
                        array = np.asarray(values, dtype="<i4")
                        stream.write(array.tobytes())
                        offsets.append([offset, len(values)])
                        offset += len(values)
                    index[split].append(
                        {
                            "instance_id": item.instance_id,
                            "offsets": offsets,
                            "metadata": item.metadata,
                        }
                    )
                    if progress and (number == 1 or number % 250 == 0 or number == len(records)):
                        progress(
                            {
                                "stage": "tokenization",
                                "data_split": split,
                                "items_done": number,
                                "items_total": len(records),
                                "message": f"{split}: {number:,}/{len(records):,}",
                            }
                        )
        atomic_json(stage / "index.json", index)
        receipt = {
            "format": "bifocalgemma_packed_tokens_v1",
            "identity": stable_fingerprint(identity),
            "files": {name: file_digest(stage / name) for name in ("tokens.bin", "index.json")},
            "counts": {split: len(records) for split, records in index.items()},
            "dtype": "little_endian_int32",
            "token_ids_changed": False,
        }
        atomic_json(stage / "READY.json", receipt)
        stage.rename(directory)
    return open_packed_cache(directory, identity)
