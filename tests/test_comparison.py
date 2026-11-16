import json

import pytest

from bifocalgemma.evaluation.comparison import compare_predictions


def test_paired_bootstrap_and_cohort_rejection(tmp_path):
    for name, correct in [("model", [True, True, True]), ("baseline", [True, False, False])]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "benchmark_report.json").write_text(
            json.dumps({"status": "COMPLETED", "protocol_version": "fixture", "total": 3})
        )
        (directory / "predictions.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"instance_id": str(i), "correct": c, "status": "SCORED", "patch_applied": True}
                )
                for i, c in enumerate(correct)
            )
        )
    result = compare_predictions(tmp_path / "model", tmp_path / "baseline", resamples=200)
    assert result["accuracy_delta"] == pytest.approx(2 / 3)
    assert result["paired_bootstrap_95_ci"][0] >= 0
    assert result["model"]["patch_applied_rate"] == 1
    target = tmp_path / "baseline/predictions.jsonl"
    target.write_text(target.read_text().replace('"0"', '"other"'))
    with pytest.raises(ValueError, match="cohort"):
        compare_predictions(tmp_path / "model", tmp_path / "baseline")
