"""Packing preserves every token, collator label, and sampler-visible order."""

import pytest
import torch

from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.packed_tokens import build_packed_cache, open_packed_cache


def test_roundtrip_uses_real_sized_ids_without_modulo_and_verifies_hashes(tmp_path):
    examples = {
        "train": [
            TokenizedRepairExample(
                "one", (2, 262100, 111), (2, 200003), (262140, 106), {"source": "fixture"}
            ),
            TokenizedRepairExample(
                "two", (2, 999), (2, 9, 8), (123456, 2, 7, 106), {"source": "fixture"}
            ),
        ],
        "dev": [
            TokenizedRepairExample("three", (2, 333), (2, 8), (220000, 106), {"source": "fixture"})
        ],
    }
    rows = {
        s: [{"instance_id": e.instance_id, "index": i} for i, e in enumerate(v)]
        for s, v in examples.items()
    }
    identity = {"fixture": "exact-token-values"}
    packed, receipt = build_packed_cache(
        tmp_path / "tokens",
        identity,
        rows,
        lambda row: next(
            e for values in examples.values() for e in values if e.instance_id == row["instance_id"]
        ),
    )
    assert receipt["token_ids_changed"] is False
    for split in examples:
        assert list(packed[split]) == examples[split]
    expected = RepairCollator()(examples["train"])
    actual = RepairCollator()(packed["train"][:])
    for name in (
        "encoder_input_ids",
        "decoder_input_ids",
        "labels",
        "encoder_attention_mask",
        "decoder_attention_mask",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    extremes = packed["train"].extreme_examples()
    assert extremes[0].instance_id == "one" and extremes[1].instance_id == "two"
    for dataset in packed.values():
        dataset.close()
    with pytest.raises(ValueError, match="identity"):
        open_packed_cache(tmp_path / "tokens", {**identity, "changed": True})
    path = tmp_path / "tokens/tokens.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        open_packed_cache(tmp_path / "tokens", identity)
