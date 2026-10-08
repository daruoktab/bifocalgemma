"""Offline repository study preparation, generation and native CPU scoring CLI."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from bifocalgemma.config import EXPERIMENT_NAMES, get_experiment
from bifocalgemma.contracts import RepairExample, RepositoryFile, RepositorySnapshot
from bifocalgemma.data.context import canonical_snapshot_hash
from bifocalgemma.data.repository import (
    SOURCE_REVISIONS,
    GitSnapshotStore,
    prepare_repository_record,
    read_source_rows,
    repository_task_from_row,
)
from bifocalgemma.data.tokenization import load_verified_tokenizer


def snapshot_from_json(path: Path) -> RepositorySnapshot:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RepositorySnapshot(
        data["repo"], data["base_commit"], tuple(RepositoryFile(**file) for file in data["files"])
    )


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )


def source_digest(path: Path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_new_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Output directory already contains a run; choose a new output.")


def fetch_tokenizer_assets(spec, cache: Path, *, token=False):
    """Opt-in metadata/tokenizer fetch only, never model weight download."""
    from huggingface_hub import hf_hub_download

    manifest = json.loads(
        (Path(__file__).parent / "assets/tokenizer_manifest.json").read_text(encoding="utf-8")
    )
    for reference in [spec.decoder] + ([spec.encoder] if spec.encoder else []):
        expected = manifest[reference.model_id]
        directory = cache / "raw" / reference.model_id.replace("/", "__")
        directory.mkdir(parents=True, exist_ok=True)
        for name, digest in expected["files"].items():
            source = Path(
                hf_hub_download(reference.model_id, name, revision=reference.revision, token=token)
            )
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Token asset hash mismatch.")
            (directory / name).write_bytes(data)


def prepare_run(args):
    spec = get_experiment(EXPERIMENT_NAMES[0])
    if args.fetch_tokenizers:
        fetch_tokenizer_assets(spec, args.tokenizer_cache)
    assert spec.encoder is not None
    tokenizer, _ = load_verified_tokenizer(spec.encoder, args.tokenizer_cache)
    config = json.loads(args.study_config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    require_new_output(output)
    store = GitSnapshotStore(args.git_cache, allow_network=args.allow_network)
    inputs, hidden, rejects = [], [], []
    per_repo = Counter()
    seen = set()
    for index, raw in enumerate(read_source_rows(args.rows)):
        if index >= args.raw_limit or len(inputs) >= args.limit:
            break
        try:
            task = repository_task_from_row(args.source, raw)
            if task.language != "python":
                raise ValueError("non_python_task")
            if per_repo[task.repo] >= args.per_repo_limit:
                raise ValueError("per_repo_sampling_limit")
            if task.instance_id in seen:
                raise ValueError("duplicate_instance_id")
            if args.snapshots:
                filename = (
                    hashlib.sha256((task.repo + "@" + task.base_commit).encode()).hexdigest()
                    + ".json"
                )
                snapshot = snapshot_from_json(args.snapshots / filename)
            else:
                snapshot = store.snapshot(task)
            row, private = prepare_repository_record(
                task,
                snapshot,
                tokenizer,
                blocked_repos=set(config["blocked_repos"]),
                aliases=config["repo_aliases"],
                max_context_tokens=args.context_limit,
            )
            digest = canonical_snapshot_hash(snapshot)
            snapshot_path = output / "snapshots" / (digest + ".json")
            write_json(
                snapshot_path,
                {
                    "repo": snapshot.repo,
                    "base_commit": snapshot.base_commit,
                    "files": [{"path": f.path, "content": f.content} for f in snapshot.files],
                },
            )
            row["snapshot_file"] = "snapshots/" + digest + ".json"
            inputs.append(row)
            if private is not None:
                private["snapshot_file"] = row["snapshot_file"]
                hidden.append(private)
            seen.add(task.instance_id)
            per_repo[task.repo] += 1
        except (ValueError, OSError, TimeoutError) as error:
            rejects.append(
                {
                    "row_index": index,
                    "instance_id": raw.get("instance_id", raw.get("id")),
                    "reason": str(error)[:500],
                }
            )
    role = SOURCE_REVISIONS[args.source][2]
    write_jsonl(output / ("train.jsonl" if role == "train" else "eval_inputs.jsonl"), inputs)
    if hidden:
        write_jsonl(output / "eval_private" / "specs.jsonl", hidden)
    report = {
        "source": SOURCE_REVISIONS[args.source],
        "role": role,
        "accepted": len(inputs),
        "processed": len(inputs) + len(rejects),
        "repos": dict(per_repo),
        "filter_counts": dict(Counter(r["reason"].split(":", 1)[0] for r in rejects)),
        "rejections": rejects,
        "context_limit": args.context_limit,
        "quarantine_config_sha256": hashlib.sha256(args.study_config.read_bytes()).hexdigest(),
        "raw_source_sha256": source_digest(args.rows),
        "validation": "static_patch_apply_only",
        "contamination_status": "repo_quarantine_not_full_near_duplicate_audit",
        "snapshot_scope": "UTF8_text_only_native_execution_may_require_real_checkout_and_binary_assets",
    }
    write_json(output / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False))


def prepare_humanevalfix_run(args):
    from bifocalgemma.data.offline_sft import editing_record
    from bifocalgemma.data.repository import RepositoryTask

    spec = get_experiment(EXPERIMENT_NAMES[0])
    assert spec.encoder is not None
    tokenizer, _ = load_verified_tokenizer(spec.encoder, args.tokenizer_cache)
    require_new_output(args.output)
    inputs, hidden = [], []
    for raw in read_source_rows(args.rows):
        before = raw["prompt"] + raw["buggy_solution"]
        after = raw["prompt"] + raw["canonical_solution"]
        commit = hashlib.sha1(before.encode()).hexdigest()
        snapshot = RepositorySnapshot(
            "benchmark/humanevalfix-python", commit, (RepositoryFile("solution.py", before),)
        )
        instruction = "Fix the buggy implementation to satisfy its function specification."
        pair = editing_record(
            {
                "old_contents": before,
                "new_contents": after,
                "old_file": "solution.py",
                "new_file": "solution.py",
                "subject": instruction,
            },
            source="humanevalfix",
        )
        task = RepositoryTask(
            raw["task_id"],
            "bigcode/humanevalpack",
            "9a41762f73a8cb23bb5811b73d5aab164efcf378",
            "eval",
            snapshot.repo,
            commit,
            instruction,
            pair["target_patch"],
            "MIT",
            "from solution import *\n"
            + raw.get("import", "")
            + "\n"
            + raw.get("test_setup", "")
            + "\n"
            + raw["test"],
            None,
            "",
        )
        public, private = prepare_repository_record(
            task, snapshot, tokenizer, blocked_repos=set(), max_context_tokens=args.context_limit
        )
        assert private is not None
        digest = canonical_snapshot_hash(snapshot)
        public["snapshot_file"] = private["snapshot_file"] = "snapshots/" + digest + ".json"
        write_json(
            args.output / public["snapshot_file"],
            {
                "repo": snapshot.repo,
                "base_commit": commit,
                "files": [{"path": "solution.py", "content": before}],
            },
        )
        inputs.append(public)
        hidden.append(private)
    write_jsonl(args.output / "eval_inputs.jsonl", inputs)
    write_jsonl(args.output / "eval_private/specs.jsonl", hidden)
    write_json(
        args.output / "preparation_report.json",
        {
            "source": "humanevalfix-docs-python",
            "accepted": len(inputs),
            "rejections": [],
            "source_file_sha256": source_digest(args.rows),
            "protocol": "native_diff_output_adapter; runtime chosen by caller; published Python3.9.13 score parity not claimed",
        },
    )
    print(json.dumps({"prepared": len(inputs), "source": "humanevalfix-docs-python"}))


def prepare_diagnostic_run(args):
    import gzip

    from bifocalgemma.data.diagnostics import prepare_crosscode_tasks, prepare_repoqa_tasks

    require_new_output(args.output)
    spec = get_experiment(EXPERIMENT_NAMES[0])
    assert spec.encoder is not None
    tokenizer, _ = load_verified_tokenizer(spec.encoder, args.tokenizer_cache)
    if args.source == "repoqa":
        data = json.loads(
            gzip.decompress(args.rows.read_bytes())
            if args.rows.suffix == ".gz"
            else args.rows.read_text(encoding="utf-8")
        )
        public, private = prepare_repoqa_tasks(
            data, tokenizer, max_context_tokens=args.context_limit
        )
    else:
        if any(term in args.rows.name.lower() for term in ("oracle", "reference", "_ref_")):
            raise ValueError("Reference-assisted retrieval is not allowed for main diagnostics.")
        public, private = prepare_crosscode_tasks(
            list(read_source_rows(args.rows)), tokenizer, max_context_tokens=args.context_limit
        )
    write_jsonl(args.output / "eval_inputs.jsonl", public)
    write_jsonl(args.output / "eval_private/specs.jsonl", private)
    write_json(
        args.output / "preparation_report.json",
        {
            "source": args.source,
            "accepted": len(public),
            "rejections": [],
            "source_file_sha256": source_digest(args.rows),
            "official_protocol": False,
            "metric": "adapted_exact_code_match",
            "context_limit": args.context_limit,
        },
    )
    print(json.dumps({"prepared": len(public), "official_protocol": False}))


def score_run(args):
    from bifocalgemma.evaluation.native_python import score_python_patch, summarize_native_results

    root = args.prepared.resolve()
    hidden = list(read_source_rows(root / "eval_private/specs.jsonl"))
    predictions = list(read_source_rows(args.predictions)) if args.predictions else []
    if not args.validate_gold and args.predictions is None:
        raise ValueError("Supply predictions or select gold validation.")
    ids = [row["instance_id"] for row in predictions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate predictions.")
    by_id = {
        row["instance_id"]: row.get("patch", row.get("model_patch", "")) for row in predictions
    }
    expected = {row["instance_id"] for row in hidden}
    if set(by_id) - expected:
        raise ValueError("Predictions contain unknown task IDs.")
    if hidden and hidden[0].get("task_mode"):
        from bifocalgemma.data.diagnostics import score_diagnostics

        if args.validate_gold:
            raise ValueError("Diagnostic gold is a text reference, not an executable repair.")
        report = score_diagnostics(hidden, predictions)
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite diagnostic report.")
        write_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "results"}))
        return
    results = []
    for task in hidden:
        path = (root / task["snapshot_file"]).resolve()
        if not path.is_relative_to(root / "snapshots"):
            raise ValueError("Snapshot path escaped prepared directory.")
        snapshot = snapshot_from_json(path)
        if args.validate_gold:
            prediction = task["gold_patch"]
        else:
            prediction = by_id.get(task["instance_id"], "")
        results.append(
            score_python_patch(
                task["instance_id"],
                snapshot,
                prediction,
                task.get("test_script") or "",
                python_executable=str(args.python_eval),
                expected_snapshot_sha256=task["snapshot_sha256"],
                allow_code_execution=args.allow_code_execution,
                timeout_seconds=args.timeout,
                expected_python_version=task.get("python_environment"),
            )
        )
    report = summarize_native_results(results)
    preparation = json.loads((root / "preparation_report.json").read_text(encoding="utf-8"))
    report["cohort"] = {
        "prepared": len(hidden),
        "preparation_rejected": len(preparation["rejections"]),
        "preparation_report": "preparation_report.json",
    }
    report["purpose"] = (
        "gold_scorer_validation_not_model_score" if args.validate_gold else "prediction_scoring"
    )
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite evaluation report.")
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))


def generate_run(args):
    from bifocalgemma.inference.generation import generate_patch
    from bifocalgemma.models.pretrained import (
        load_local_native_model,
        load_sft_adapters_for_inference,
    )

    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Prediction artifacts already exist.")
    spec = get_experiment(args.experiment)
    model, provenance = load_local_native_model(
        spec, args.decoder_checkpoint, args.encoder_checkpoint
    )
    if args.adapter_checkpoint:
        provenance["sft_checkpoint"] = load_sft_adapters_for_inference(
            model, args.adapter_checkpoint
        )
    predictions = []
    for index, row in enumerate(read_source_rows(args.prepared / "eval_inputs.jsonl")):
        if index >= args.limit:
            break
        if row.get("task_mode"):
            from bifocalgemma.inference.generation import generate_diagnostic_text

            text, stop = generate_diagnostic_text(
                spec,
                model,
                row["instruction"],
                row["context"],
                args.tokenizer_cache,
                max_new_tokens=args.max_new_tokens,
                max_encoder_tokens=args.context_limit,
                max_total_decoder_tokens=args.decoder_limit,
            )
            predictions.append(
                {
                    "instance_id": row["instance_id"],
                    "model_name": spec.name,
                    "text": text,
                    "stop_reason": stop,
                    "task_mode": row["task_mode"],
                }
            )
            continue
        example = RepairExample(
            row["instance_id"],
            row["source"],
            "test",
            RepositorySnapshot(row["repos"], row["commit"], ()),
            row["instruction"],
        )
        result = generate_patch(
            spec,
            example,
            args.max_new_tokens,
            model=model,
            tokenizer_cache=args.tokenizer_cache,
            context=row["context"],
            max_encoder_tokens=args.context_limit,
            max_total_decoder_tokens=args.decoder_limit,
        )
        predictions.append(
            {
                "instance_id": result.instance_id,
                "model_name": result.model_name,
                "patch": result.patch,
            }
        )
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite predictions.")
    write_jsonl(args.output, predictions)
    write_json(
        args.output.with_suffix(".manifest.json"),
        {
            "model": provenance,
            "generated": len(predictions),
            "backend": "eager_greedy_no_temporal_cache",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assets = sub.add_parser("fetch-assets")
    assets.add_argument("--tokenizer-cache", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", choices=tuple(SOURCE_REVISIONS), required=True)
    prepare.add_argument("--rows", type=Path, required=True)
    prepare.add_argument("--snapshots", type=Path)
    prepare.add_argument("--git-cache", type=Path, default=Path("outputs/git_snapshots"))
    prepare.add_argument("--tokenizer-cache", type=Path, required=True)
    prepare.add_argument("--fetch-tokenizers", action="store_true")
    prepare.add_argument("--allow-network", action="store_true")
    prepare.add_argument("--study-config", type=Path, default=Path("configs/repository_study.json"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--context-limit", type=int, default=4096)
    prepare.add_argument("--limit", type=int, default=32)
    prepare.add_argument("--raw-limit", type=int, default=1000)
    prepare.add_argument("--per-repo-limit", type=int, default=8)
    diagnostic = sub.add_parser("prepare-diagnostic")
    diagnostic.add_argument("--source", choices=("repoqa", "crosscodeeval"), required=True)
    diagnostic.add_argument("--rows", type=Path, required=True)
    diagnostic.add_argument("--tokenizer-cache", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--context-limit", type=int, default=4096)
    humaneval = sub.add_parser("prepare-humanevalfix")
    humaneval.add_argument("--rows", type=Path, required=True)
    humaneval.add_argument("--tokenizer-cache", type=Path, required=True)
    humaneval.add_argument("--output", type=Path, required=True)
    humaneval.add_argument("--context-limit", type=int, default=4096)
    score = sub.add_parser("score")
    score.add_argument("--prepared", type=Path, required=True)
    score.add_argument("--predictions", type=Path)
    score.add_argument("--python-eval", type=Path, default=Path(sys.executable))
    score.add_argument("--allow-code-execution", action="store_true")
    score.add_argument("--validate-gold", action="store_true")
    score.add_argument("--timeout", type=float, default=30)
    score.add_argument("--output", type=Path, required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--prepared", type=Path, required=True)
    generate.add_argument("--experiment", choices=EXPERIMENT_NAMES, required=True)
    generate.add_argument("--decoder-checkpoint", type=Path, required=True)
    generate.add_argument("--encoder-checkpoint", type=Path)
    generate.add_argument("--adapter-checkpoint", type=Path)
    generate.add_argument("--tokenizer-cache", type=Path, required=True)
    generate.add_argument("--context-limit", type=int, default=4096)
    generate.add_argument("--decoder-limit", type=int, default=8192)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--limit", type=int, default=4)
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-assets":
            fetch_tokenizer_assets(
                get_experiment("BifocalGemma-E2B-E2B"),
                args.tokenizer_cache,
                token=os.environ.get("HF_TOKEN") or False,
            )
            fetch_tokenizer_assets(
                get_experiment("BifocalGemma-E4B-E4B"),
                args.tokenizer_cache,
                token=os.environ.get("HF_TOKEN") or False,
            )
        elif args.command == "prepare":
            if min(args.limit, args.raw_limit, args.per_repo_limit, args.context_limit) < 1:
                raise ValueError("Preparation budgets must be positive.")
            prepare_run(args)
        elif args.command == "prepare-humanevalfix":
            prepare_humanevalfix_run(args)
        elif args.command == "prepare-diagnostic":
            prepare_diagnostic_run(args)
        elif args.command == "score":
            score_run(args)
        else:
            generate_run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Repository study failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
