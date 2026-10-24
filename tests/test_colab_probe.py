import importlib
from pathlib import Path

import pytest

from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator


@pytest.fixture
def boundary_builder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("prepare_colab_probe").declared_limit_example


@pytest.mark.parametrize("paired", [True, False])
def test_declared_boundaries_reach_exact_input_length_without_new_target_tokens(
    boundary_builder, paired
):
    actual = TokenizedRepairExample("real", (2, 5, 6) if paired else (), (2, 7), (8, 9, 1))
    encoder_limit, decoder_limit = (32768, 8192) if paired else (0, 41472)
    shape = boundary_builder(actual, encoder_limit, decoder_limit)
    batch = RepairCollator()([shape])
    assert batch.encoder_input_ids.shape == (1, encoder_limit)
    assert batch.decoder_input_ids.shape == (1, decoder_limit)
    assert batch.encoder_attention_mask.all() and batch.decoder_attention_mask.all()
    assert shape.target_ids == actual.target_ids
    assert int((batch.labels != -100).sum()) == len(actual.target_ids)
    assert actual.instance_id == "real" and shape.metadata["shape_only"] is True


@pytest.mark.parametrize("limits", [(2, 10), (10, 2)])
def test_declared_boundaries_cannot_hide_an_overflow_by_truncation(boundary_builder, limits):
    actual = TokenizedRepairExample("overflow", (2, 5, 6), (2, 7), (8, 9, 1))
    with pytest.raises(ValueError, match="must not truncate"):
        boundary_builder(actual, *limits)
