"""Run controlled apples-to-apples evaluations on external baseline models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bifocalgemma.config import EXTERNAL_COMPARISON_MODELS  # noqa: E402
from bifocalgemma.evaluation.external_runner import (  # noqa: E402
    EXTERNAL_CONTEXT_MODES,
    load_hf_external_model_and_tokenizer,
    resolve_external_benchmarks,
    resolve_external_model_spec,
    run_external_benchmark,
)
from bifocalgemma.training.stages import atomic_json  # noqa: E402


def _flatten_csv_args(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for item in values:
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"External model keys or HF IDs (default: all -> {list(EXTERNAL_COMPARISON_MODELS)}).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmark collection names from V4_BENCHMARKS (or 'all', 'reponiah').",
    )
    parser.add_argument(
        "--context-mode",
        choices=EXTERNAL_CONTEXT_MODES,
        default="full",
        help="Prompt context mode for decoder-only comparison ('full' or 'local_only').",
    )
    parser.add_argument(
        "--repair-mode",
        choices=("unified_diff", "full_function"),
        default=None,
        help="Optional override for repair tasks (defaults to per-benchmark config).",
    )
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        default=ROOT / "outputs/v2_cache/benchmarks",
        help="Root directory containing prepared benchmark cohorts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/external_eval_v4",
        help="Output directory for predictions and benchmark reports.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-execution", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model specs, benchmarks, and cohort availability without loading weights.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_tokens = _flatten_csv_args(args.models)
    if not model_tokens or model_tokens == ["all"]:
        model_keys = list(EXTERNAL_COMPARISON_MODELS)
    else:
        model_keys = [resolve_external_model_spec(tok)["key"] for tok in model_tokens]

    bench_tokens = _flatten_csv_args(args.benchmarks)
    benchmarks = resolve_external_benchmarks(bench_tokens)

    summary: dict[str, Any] = {
        "dry_run": args.dry_run,
        "context_mode": args.context_mode,
        "repair_mode": args.repair_mode,
        "models": model_keys,
        "benchmarks": [b["collection"] for b in benchmarks],
        "results": {},
    }

    for model_key in model_keys:
        model_cfg = resolve_external_model_spec(model_key)
        model = None
        tokenizer = None
        if not args.dry_run:
            model, tokenizer = load_hf_external_model_and_tokenizer(
                model_cfg, local_files_only=args.local_files_only
            )
        model_reports: dict[str, Any] = {}
        for bench in benchmarks:
            col = str(bench["collection"])
            bench_out = args.output_dir / model_key / args.context_mode / col
            report = run_external_benchmark(
                model_key,
                bench,
                args.benchmarks_root,
                bench_out,
                context_mode=args.context_mode,
                repair_mode=args.repair_mode,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                allow_execution=not args.no_execution,
                dry_run=args.dry_run,
                local_files_only=args.local_files_only,
                model=model,
                tokenizer=tokenizer,
            )
            model_reports[col] = report
        summary["results"][model_key] = model_reports

    if not args.dry_run:
        atomic_json(args.output_dir / f"summary_{args.context_mode}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
