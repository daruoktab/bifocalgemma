"""One resident model: full SFT, restartable benchmarks, verified Hub upload, next model."""

import copy
import gc
import json
import math
from pathlib import Path

import torch

from bifocalgemma.config import get_experiment
from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.releases import fetch_collection
from bifocalgemma.evaluation.context import EVALUATION_PROTOCOL, run_context_benchmark
from bifocalgemma.models.memory_efficient import (
    configure_memory_efficient_training,
    run_training_memory_preflight,
)
from bifocalgemma.models.v2.pretrained import load_sft_adapters_for_inference
from bifocalgemma.training.benchmark_reset import reset_local_benchmarks
from bifocalgemma.training.benchmark_upload import BenchmarkUploadQueue
from bifocalgemma.training.curated_sft import prepare_curated_training
from bifocalgemma.training.hub import recover_remote_run, synchronize_run
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.policy import configure_policy_model
from bifocalgemma.training.recipes import resolve_training_cadence
from bifocalgemma.training.stages import StageLedger, atomic_json, latest_checkpoint
from bifocalgemma.training.telemetry import TrainingDashboard, resource_snapshot

FAMILY_SEQUENCES = {
    "symmetric": ("BifocalGemma-E2B-E2B", "BifocalGemma-E4B-E4B"),
    "asymmetric": ("BifocalGemma-E2B-E4B", "BifocalGemma-E4B-E2B"),
    "decoder_only": ("Gemma-4-E2B-IT", "Gemma-4-E4B-IT"),
}


def run_sequential_pipeline(
    *,
    family,
    experiments,
    output_root,
    run_name,
    dataset_repo,
    dataset_revision,
    collection,
    pool_directory,
    tokenizer_cache,
    benchmark_collections,
    benchmark_cache,
    model_factory,
    training_policy,
    memory_policy,
    environment,
    model_repo,
    token,
    mo=None,
    epochs=1,
    batch_size=1,
    eval_batch_size=1,
    accumulation=4,
    max_steps=0,
    eval_every=None,
    save_every=None,
    eval_fraction=0.25,
    save_fraction=0.10,
    clip_norm=1.0,
    seed=17,
    run_training=True,
    run_benchmarks=True,
    upload=True,
    resume_from=None,
    checkpoint_upload=True,
    benchmark_upload_every=500,
    functional_execution=True,
    auto_resume=True,
    recipe_name=None,
    finalize=True,
    benchmark_reset_id=None,
):
    if not experiments or len(set(experiments)) != len(experiments):
        raise ValueError("Choose an ordered sequence with unique experiments.")
    if any(name not in FAMILY_SEQUENCES[family] for name in experiments):
        raise ValueError("Experiment does not belong to this notebook family.")
    if upload and not token:
        raise ValueError("Apply a write HF token before starting the sequential pipeline.")
    initial_rows, pool = load_curated_pool(pool_directory)
    del initial_rows
    prepared = {}
    if run_benchmarks:
        for item in benchmark_collections:
            directory = Path(benchmark_cache) / item["collection"]
            fetch_collection(
                dataset_repo,
                dataset_revision,
                item["collection"],
                directory,
                expected_role="eval",
                include_gold=True,
                token=token,
            )
            prepared[item["collection"]] = directory
    results = []
    device = environment.get("device", "cuda")
    for index, name in enumerate(experiments, 1):
        spec = get_experiment(name)
        root = Path(output_root) / name / run_name
        prefix = f"{family}/{name}/{run_name}"
        if upload and auto_resume:
            recover_remote_run(model_repo, root, prefix, token=token)
        if not auto_resume and root.exists() and any(root.iterdir()):
            raise FileExistsError("AUTO_RESUME=False: choose a new RUN_NAME.")
        reset_local_benchmarks(root, benchmark_reset_id)
        identity = {
            "version": "sequential_context_sft_v1",
            "spec": spec.to_dict(),
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "collection": collection,
            "pool_fingerprint": pool["fingerprint"],
            "training_policy": training_policy,
            "memory_policy": memory_policy,
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "accumulation": accumulation,
            "max_steps": max_steps,
            "clip_norm": clip_norm,
            "seed": seed,
            "cadence_policy": {
                "eval_every": eval_every,
                "save_every": save_every,
                "eval_fraction": eval_fraction,
                "save_fraction": save_fraction,
            },
            "recipe": recipe_name,
            "torch": str(torch.__version__),
            "transformers": environment["transformers"],
        }
        ledger = StageLedger(root, identity)
        train_root = root / "training"
        model = None
        dashboard = TrainingDashboard(
            mo, experiment=f"{name} / {recipe_name}" if recipe_name else name, device=device
        )
        dashboard(
            {
                "stage": "pipeline_model_started",
                "message": f"Model {index}/{len(experiments)}: {name}",
                "model_index": index,
                "model_total": len(experiments),
            }
        )
        active_stage = "training"
        benchmark_upload = BenchmarkUploadQueue(
            root,
            backoff_path=Path(output_root) / ".benchmark-upload-backoff.json",
            progress=dashboard,
        )
        try:
            training_done = ledger.completed("training")
            benchmark_done = (
                all(
                    ledger.completed(
                        "benchmark:" + item["collection"],
                        settings={
                            "benchmark": item,
                            "execution": functional_execution,
                            "protocol": EVALUATION_PROTOCOL,
                        },
                    )
                    for item in benchmark_collections
                )
                if run_benchmarks
                else True
            )
            if not training_done or not benchmark_done:
                if not training_done and not run_training:
                    raise ValueError("No completed training stage; RUN_TRAINING is disabled.")
                dashboard({"stage": "checkpoint_loading", **resource_snapshot(device)})
                model, provenance = model_factory(spec, dashboard)
                if not training_done:
                    if not run_training:
                        raise ValueError("No completed training stage; RUN_TRAINING is disabled.")
                    scope = configure_policy_model(model, training_policy)
                    configure_memory_efficient_training(model, memory_policy)
                    dashboard({"stage": "tokenization", "message": name})
                    train, dev, data_manifest = prepare_curated_training(
                        spec,
                        pool_directory,
                        tokenizer_cache,
                        packed_cache=Path(tokenizer_cache).parent / "packed_training_tokens",
                        progress=dashboard,
                    )
                    full_steps = epochs * math.ceil(len(train) / (batch_size * accumulation))
                    steps = min(max_steps, full_steps) if max_steps else full_steps
                    cadence = resolve_training_cadence(
                        steps,
                        eval_every=eval_every,
                        save_every=save_every,
                        eval_fraction=eval_fraction,
                        save_fraction=save_fraction,
                    )
                    dashboard(
                        {
                            "stage": "training_plan",
                            "message": f"{name} / {recipe_name or 'legacy'}: {steps:,} updates; validation every {cadence['eval_every']:,}, checkpoint every {cadence['save_every']:,}; final validation once.",
                            **cadence,
                        }
                    )
                    selected_resume = (
                        resume_from if index == 1 and resume_from else latest_checkpoint(train_root)
                    )
                    preflight = run_training_memory_preflight(
                        model, train, progress=dashboard, batch_size=batch_size, min_headroom_gib=8
                    )
                    # Dev may contain longer targets than train; probe before any optimizer update.
                    validation_preflight = run_training_memory_preflight(
                        model,
                        dev,
                        progress=dashboard,
                        batch_size=eval_batch_size,
                        min_headroom_gib=8,
                    )
                    compatibility = {
                        "architecture": "sequential_context_sft_v1",
                        "experiment": name,
                        "family": family,
                        "model_spec": spec.to_dict(),
                        "sources": provenance,
                        "data_fingerprint": data_manifest["fingerprint"],
                        "training_policy": training_policy,
                        "memory_policy": memory_policy,
                        "batch_size": batch_size,
                        "eval_batch_size": eval_batch_size,
                        "accumulation": accumulation,
                        "clip_norm": clip_norm,
                        "seed": seed,
                        "torch": str(torch.__version__),
                        "transformers": environment["transformers"],
                    }
                    if (
                        selected_resume is None
                        and train_root.exists()
                        and any(train_root.iterdir())
                    ):
                        config_path = train_root / "run_config.json"
                        if not config_path.exists():
                            raise ValueError(
                                "Unrecognized training artifacts without a resumable checkpoint."
                            )
                        previous = json.loads(config_path.read_text(encoding="utf-8"))
                        if any(previous.get(key) != value for key, value in compatibility.items()):
                            raise ValueError(
                                "Incomplete run compatibility differs; use a new RUN_NAME."
                            )
                        from uuid import uuid4

                        archived = root / (".interrupted-before-checkpoint-" + uuid4().hex)
                        if (
                            train_root.resolve().parent != root.resolve()
                            or archived.resolve().parent != root.resolve()
                        ):
                            raise ValueError("Training restart archive escaped run directory.")
                        train_root.rename(archived)
                        dashboard(
                            {
                                "stage": "training_restart_from_zero",
                                "message": "No complete checkpoint; previous diagnostics archived.",
                            }
                        )
                    # Marker is written outside the helper's training directory, which stays empty for a new run.
                    ledger.start("training")

                    def upload_checkpoint(path):
                        dashboard(
                            {"stage": "checkpoint_upload_started", "message": Path(path).name}
                        )
                        synchronize_run(model_repo, root, prefix, token=token, checkpoint_only=True)
                        dashboard(
                            {"stage": "checkpoint_upload_complete", "message": Path(path).name}
                        )

                    def train_progress(row):
                        if row["step"] == 1 or row["step"] % 10 == 0 or "eval" in row:
                            print(
                                json.dumps(
                                    {
                                        "model": name,
                                        "recipe": recipe_name,
                                        "step": row["step"],
                                        "total_steps": steps,
                                        "train_loss": row["train_loss"],
                                        "dev_loss": row.get("eval", {}).get("loss"),
                                        "learning_rates": row.get("learning_rates"),
                                        "gpu_peak_gib": row.get("gpu_peak_allocated_gib"),
                                    }
                                ),
                                flush=True,
                            )

                    report = fit_offline_sft(
                        model,
                        train,
                        dev,
                        train_root,
                        compatibility,
                        max_steps=steps,
                        epochs=epochs,
                        batch_size=batch_size,
                        eval_batch_size=eval_batch_size,
                        accumulation=accumulation,
                        training_policy=training_policy,
                        clip_norm=clip_norm,
                        eval_every=cadence["eval_every"],
                        save_every=cadence["save_every"],
                        seed=seed,
                        resume_from=selected_resume,
                        event_callback=dashboard,
                        progress=train_progress,
                        data_manifest=data_manifest,
                        save_total_limit=2,
                        checkpoint_callback=upload_checkpoint
                        if upload and checkpoint_upload
                        else None,
                    )
                    report.update(
                        environment=environment,
                        provenance=provenance,
                        scope=scope,
                        memory_preflight=preflight,
                        validation_memory_preflight=validation_preflight,
                        planned_full_epoch_updates=epochs
                        * math.ceil(len(train) / (batch_size * accumulation)),
                        recipe=recipe_name,
                    )
                    atomic_json(train_root / "training_report.json", report)
                    last_saved = report["last_checkpoint"]
                    if not isinstance(last_saved, str):
                        raise ValueError("Completed training has no checkpoint path.")
                    checkpoint = Path(last_saved)
                    ledger.finish(
                        "training",
                        [
                            train_root / "training_report.json",
                            train_root / "run_config.json",
                            checkpoint / "READY.json",
                            checkpoint / "manifest.json",
                            checkpoint / "adapters.safetensors",
                        ],
                        {"global_step": report["global_step"], "checkpoint": checkpoint.name},
                    )
                    for dataset in (train, dev):
                        if hasattr(dataset, "close"):
                            dataset.close()
                    del train, dev, data_manifest
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    report = json.loads(
                        (train_root / "training_report.json").read_text(encoding="utf-8")
                    )
                    checkpoint = train_root / Path(report["last_checkpoint"]).name
                    load_sft_adapters_for_inference(model, checkpoint)
                    configure_memory_efficient_training(model, memory_policy)
            else:
                report = json.loads(
                    (train_root / "training_report.json").read_text(encoding="utf-8")
                )
            if upload and not ledger.completed("training_upload"):
                active_stage = "training_upload"
                ledger.start(active_stage)
                receipt = synchronize_run(model_repo, root, prefix, token=token)
                # Immutable receipt copy: later benchmark uploads replace upload_complete.json.
                receipt_path = train_root / "training_upload_receipt.json"
                atomic_json(receipt_path, receipt)
                ledger.finish(active_stage, [receipt_path], {"revision": receipt["revision"]})
            if run_benchmarks:
                checkpoint = train_root / Path(report["last_checkpoint"]).name
                checkpoint_identity = json.loads(
                    (checkpoint / "READY.json").read_text(encoding="utf-8")
                )
                for item in benchmark_collections:
                    active_stage = "benchmark:" + item["collection"]
                    benchmark_settings = {
                        "benchmark": item,
                        "execution": functional_execution,
                        "protocol": EVALUATION_PROTOCOL,
                    }
                    if ledger.completed(active_stage, settings=benchmark_settings):
                        continue
                    ledger.start(active_stage)
                    benchmark_dir = (
                        root
                        / "benchmarks"
                        / (item["collection"] + "-" + stable_fingerprint(benchmark_settings)[:12])
                    )
                    dashboard.run_dir = benchmark_dir
                    dashboard.run_dir.mkdir(parents=True, exist_ok=True)
                    dashboard({"stage": "benchmark_started", "message": item["collection"]})

                    def save_benchmark_progress():
                        benchmark_upload.attempt(
                            lambda: synchronize_run(model_repo, root, prefix, token=token)
                        )

                    def benchmark_progress(event):
                        if upload and event["stage"] == "benchmark_prediction_saved":
                            benchmark_upload.mark_pending()
                        dashboard(
                            {
                                **event,
                                "benchmark_backup_status": benchmark_upload.state["status"],
                                "backup_pending": benchmark_upload.state["pending"],
                                "backup_next_retry_at": benchmark_upload.state.get("next_retry_at"),
                            }
                        )

                    benchmark = run_context_benchmark(
                        model,
                        spec,
                        prepared[item["collection"]],
                        benchmark_dir,
                        tokenizer_cache,
                        checkpoint_identity=checkpoint_identity,
                        encoder_limit=item.get("encoder_limit", 8192),
                        decoder_limit=item.get("decoder_limit", 2048),
                        max_new_tokens=item.get("max_new_tokens", 2048),
                        allow_execution=functional_execution,
                        progress=benchmark_progress,
                        prediction_callback=save_benchmark_progress if upload else None,
                        upload_every=benchmark_upload_every,
                    )
                    ledger.finish(
                        active_stage,
                        [
                            benchmark_dir / "benchmark_report.json",
                            benchmark_dir / "predictions.jsonl",
                            benchmark_dir / "evaluation_state.json",
                        ],
                        benchmark,
                        settings=benchmark_settings,
                    )
                    if upload:
                        # Publish the completed report/ledger too, not just the last prediction.
                        save_benchmark_progress()
            upload_settings = {
                "benchmarks": benchmark_collections if run_benchmarks else [],
                "execution": functional_execution,
                "protocol": EVALUATION_PROTOCOL,
            }
            if (
                upload
                and finalize
                and (
                    not ledger.completed("final_upload", settings=upload_settings)
                    or benchmark_upload.state.get("pending")
                    or not benchmark_upload.state.get("final_synced", False)
                )
            ):
                active_stage = "final_upload"

                def publish_final():
                    if not ledger.completed("final_upload", settings=upload_settings):
                        ledger.start("final_upload")
                        receipt = synchronize_run(model_repo, root, prefix, token=token)
                        final_receipt = train_root / "final_upload_receipt.json"
                        atomic_json(final_receipt, receipt)
                        ledger.finish(
                            "final_upload",
                            [final_receipt],
                            {"revision": receipt["revision"]},
                            settings=upload_settings,
                        )
                    # Retry this commit even when the local ledger was completed before a failure.
                    return synchronize_run(model_repo, root, prefix, token=token)

                benchmark_upload.attempt(publish_final, final=True)
            pending_upload = bool(upload and finalize and benchmark_upload.state.get("pending"))
            results.append(
                {
                    "experiment": name,
                    "status": "PENDING_UPLOAD" if pending_upload else "COMPLETED",
                    "global_step": report["global_step"],
                    "held_out_eval": report["held_out_eval"],
                    "run_dir": str(root),
                    "stages": copy.deepcopy(ledger.state["stages"]),
                    "benchmark_enabled": run_benchmarks,
                    "upload_enabled": upload,
                    "benchmark_backup": copy.deepcopy(benchmark_upload.state),
                }
            )
            dashboard(
                {
                    "stage": "pipeline_model_complete",
                    "message": f"{name} selesai; model berikutnya",
                    "model_index": index,
                    "model_total": len(experiments),
                }
            )
        except BaseException as error:
            ledger.fail(active_stage, error)
            dashboard({"stage": "failed", "error_type": type(error).__name__, "error": str(error)})
            import traceback

            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "status": "PENDING_UPLOAD"
        if any(r["status"] == "PENDING_UPLOAD" for r in results)
        else "COMPLETED",
        "family": family,
        "sequence": list(experiments),
        "models": results,
    }
