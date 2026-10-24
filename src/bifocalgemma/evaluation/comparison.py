"""Paired per-example benchmark differences; no composite across unlike tasks."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


def compare_predictions(model_directory, baseline_directory, *, seed=17, resamples=10000):
    model_directory, baseline_directory = Path(model_directory), Path(baseline_directory)
    reports = [
        json.loads((p / "benchmark_report.json").read_text())
        for p in (model_directory, baseline_directory)
    ]
    if any(r["status"] != "COMPLETED" for r in reports):
        raise ValueError("Only complete benchmark reports can be compared.")
    if reports[0]["protocol_version"] != reports[1]["protocol_version"]:
        raise ValueError("Evaluation protocols differ.")
    records = [
        [json.loads(line) for line in (p / "predictions.jsonl").read_text().splitlines()]
        for p in (model_directory, baseline_directory)
    ]
    ids = [[r["instance_id"] for r in rows] for rows in records]
    if not ids[0] or ids[0] != ids[1] or len(set(ids[0])) != len(ids[0]):
        raise ValueError("Paired evaluation requires identical ordered unique cohort IDs.")
    if any(len(rows) != report["total"] for rows, report in zip(records, reports, strict=True)):
        raise ValueError("Prediction count differs from report.")
    if any(type(row["correct"]) is not bool for rows in records for row in rows):
        raise ValueError("Unscored predictions cannot support accuracy comparisons.")
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples.")
    differences = np.asarray(
        [int(a["correct"]) - int(b["correct"]) for a, b in zip(*records, strict=True)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, resamples, 128):
        draws = rng.integers(
            0, len(differences), size=(min(128, resamples - start), len(differences))
        )
        boot.extend(differences[draws].mean(axis=1).tolist())
    summaries = []
    for rows in records:
        entry = {
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
        }
        for key in ("patch_applied", "compile_pass", "functional_pass"):
            if all(key in r for r in rows):
                entry[key + "_rate"] = sum(bool(r[key]) for r in rows) / len(rows)
        summaries.append(entry)
        syntax = [r["syntax_valid"] for r in rows if r.get("syntax_valid") is not None]
        if syntax:
            entry.update(syntax_checked=len(syntax), syntax_success_rate=sum(syntax) / len(syntax))
    return {
        "examples": len(differences),
        "model": summaries[0],
        "baseline": summaries[1],
        "accuracy_delta": float(differences.mean()),
        "paired_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_seed": seed,
        "resamples": resamples,
        "per_example_delta": [
            {"instance_id": i, "delta": int(d)} for i, d in zip(ids[0], differences, strict=True)
        ],
        "limitations": "Per-example resampling; training seed variation and repository-level dependence are not measured. Each benchmark is reported separately.",
    }
