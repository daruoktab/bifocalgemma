import copy

import pytest

from bifocalgemma.training.v2 import MODELS
from bifocalgemma.training.v2_preflight import EXECUTION_VERSION, seal_report
from tests.test_shared_execution import valid_profile
from tools.release_gpu_profile import check_evidence


@pytest.fixture
def completed_evidence():
    profile = valid_profile()
    rows = {}
    for key in MODELS:
        receipt = copy.deepcopy(profile["validation"])
        receipt.update(
            version=EXECUTION_VERSION,
            gpu=profile["gpu"]["name"],
            encoder_tokens=0 if key.endswith("-it") else 32768,
            decoder_tokens=41472 if key.endswith("-it") else 8192,
            optimizer_updates_retained=0,
        )
        rows[key] = {
            "status": "PASS",
            "weights_restored_exactly": True,
            "trainable_sha256_before": "a" * 64,
            "trainable_sha256_after": "a" * 64,
            "validation": seal_report(receipt),
        }
    profile["declared_limit_validation"] = {key: row["validation"] for key, row in rows.items()}
    profile["validation"] = rows["e4b-e4b"]["validation"]
    report = {
        "status": "COMPLETED",
        "diagnostic_only": True,
        "epoch_training_started": False,
        "profile_published": False,
        "failure": None,
        "weights_restored_exactly": True,
        "trainable_sha256_before": "a" * 64,
        "trainable_sha256_after": "a" * 64,
    }
    return seal_report(profile), rows, report


def test_complete_six_model_boundary_evidence_can_be_reviewed_offline(completed_evidence):
    profile, rows, report = completed_evidence
    assert check_evidence(profile, rows, report, profile["contract"]) == profile


@pytest.mark.parametrize("damage", ["missing", "restore", "shape", "headroom", "checksum"])
def test_partial_or_inconsistent_gpu_evidence_cannot_be_released(completed_evidence, damage):
    profile, rows, report = completed_evidence
    row = rows["e2b-it"]
    if damage == "missing":
        rows.pop("e2b-it")
    elif damage == "restore":
        row["trainable_sha256_after"] = "b" * 64
    else:
        receipt = row["validation"]
        if damage in {"shape", "checksum"}:
            receipt["decoder_tokens"] = 6207
        else:
            receipt["phases"]["training"]["estimated_peak_headroom_gib"] = 2
        row["validation"] = receipt if damage == "checksum" else seal_report(receipt)
        profile["declared_limit_validation"]["e2b-it"] = row["validation"]
        profile = seal_report(profile)
    with pytest.raises(ValueError):
        check_evidence(profile, rows, report, profile["contract"])
