import hashlib
import json

from bifocalgemma.training.benchmark_reset import remote_reset_plan, reset_local_benchmarks
from bifocalgemma.training.stages import StageLedger


def test_reset_archives_evaluation_once_without_changing_training(tmp_path):
    ledger = StageLedger(tmp_path, {"model": "same-training"})
    checkpoint = tmp_path / "training/weights"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"preserve")
    ledger.finish("training", [checkpoint])
    prediction = tmp_path / "benchmarks/test/predictions.jsonl"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("old\n")
    ledger.finish("benchmark:test", [prediction])
    reset_local_benchmarks(tmp_path, "reset-one")
    restored = StageLedger(tmp_path, {"model": "same-training"})
    assert restored.completed("training")
    assert not restored.completed("benchmark:test")
    assert not prediction.exists() and checkpoint.read_bytes() == b"preserve"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("new\n")
    reset_local_benchmarks(tmp_path, "reset-one")
    assert prediction.read_text() == "new\n"
    assert len(list(tmp_path.glob(".benchmark-reset-*"))) == 1


def test_remote_reset_is_scoped_and_receipt_hashes_match():
    prefix = "symmetric/BifocalGemma-E2B-E2B/run/base"
    state = {"stages": {"training": {"status": "COMPLETED"}, "benchmark:a": {}, "final_upload": {}}}
    names = [
        prefix + "/benchmarks/a/predictions.jsonl",
        prefix + "/training/weights.pt",
        "other/benchmarks/a",
    ]
    receipt = {
        "files": {
            "pipeline_state.json": "old",
            "benchmarks/a/predictions.jsonl": "old",
            "training/weights.pt": "untouched",
        }
    }
    deleted, replacements = remote_reset_plan(prefix, names, state, receipt, "reset-one")
    assert deleted == [names[0]]
    new_state = json.loads(replacements[prefix + "/pipeline_state.json"])
    assert new_state["stages"] == {"training": state["stages"]["training"]}
    new_receipt = json.loads(replacements[prefix + "/upload_complete.json"])
    assert new_receipt["files"] == {
        "training/weights.pt": "untouched",
        "pipeline_state.json": hashlib.sha256(
            replacements[prefix + "/pipeline_state.json"]
        ).hexdigest(),
    }
    assert state["stages"].get("benchmark:a") == {}
