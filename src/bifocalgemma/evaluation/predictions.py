"""E02: official SWE-bench JSONL wire format; never include gold artifacts."""

import json
from collections.abc import Iterable
from pathlib import Path

from bifocalgemma.contracts import RepairPrediction


def write_swebench_predictions(predictions: Iterable[RepairPrediction], path: str | Path) -> None:
    rows = []
    seen = set()
    for prediction in predictions:
        if prediction.instance_id in seen:
            raise ValueError(f"Duplicate instance_id: {prediction.instance_id}")
        seen.add(prediction.instance_id)
        rows.append(
            json.dumps(
                {
                    "instance_id": prediction.instance_id,
                    "model_name_or_path": prediction.model_name,
                    "model_patch": prediction.patch,
                },
                ensure_ascii=False,
            )
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
