"""One-model Molab workflow, with resumable training and automatic benchmarks."""

import gc
import hashlib
import json
import math
import shutil
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

from bifocalgemma.config import get_v2_experiment
from bifocalgemma.data.releases import fetch_collection
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.evaluation.context import (
    BENCHMARK_RESET_ID,
    V2_EVALUATION_PROTOCOL,
    run_context_benchmark,
)
from bifocalgemma.models.memory_efficient import (
    DEFAULT_MEMORY_POLICY,
    configure_memory_efficient_training,
)
from bifocalgemma.models.pretrained import load_local_native_model, load_sft_adapters_for_inference
from bifocalgemma.training.benchmark_reset import reset_local_benchmarks
from bifocalgemma.training.benchmark_upload import BenchmarkUploadQueue
from bifocalgemma.training.curated_sft import prepare_curated_training
from bifocalgemma.training.hub import recover_remote_run
from bifocalgemma.training.offline_sft import fit_offline_sft
from bifocalgemma.training.policy import configure_policy_model
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.shared_execution import (
    HEADROOM_GIB,
    calibrate_shared_profile,
    publish_with_retry,
    resolve_shared_profile,
    shared_contract,
)
from bifocalgemma.training.source_identity import source_fingerprints
from bifocalgemma.training.stages import StageLedger, atomic_json, latest_checkpoint
from bifocalgemma.training.telemetry import TrainingDashboard, append_event
from bifocalgemma.training.v2_diagnostics import encoder_diagnostics, stratified_subset
from bifocalgemma.training.v2_hub import RemoteHeartbeat, synchronize_preflight, synchronize_v2
from bifocalgemma.training.v2_preflight import EXECUTION_VERSION, select_execution
from bifocalgemma.training.v2_preflight import run_name as execution_run_name

DATASET_REPO = "daruokta/bifocalgemma-data"
DATASET_REVISION = "7972f95d7c87f163c8b5af6a4acd308c0c1d0e23"
COLLECTION = "mixed_context_enc32k_dec8k_v3"
MODELS = {
    "e2b-e2b": ("BifocalGemma-E2B-E2B", "symmetric"),
    "e4b-e2b": ("BifocalGemma-E4B-E2B", "asymmetric"),
    "e2b-e4b": ("BifocalGemma-E2B-E4B", "asymmetric"),
    "e4b-e4b": ("BifocalGemma-E4B-E4B", "symmetric"),
    "e2b-it": ("Gemma-4-E2B-IT", "decoder_only"),
    "e4b-it": ("Gemma-4-E4B-IT", "decoder_only"),
}
BENCHMARKS: list[dict[str, Any]] = [
    {
        "collection": n,
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        **({"max_new_tokens": 8192} if n.startswith(("humaneval", "gsm8k")) else {}),
    }
    for n in (
        "humanevalfix_python_v2",
        "sweqa_oracle_gemma32k_dec8k_v1",
        "sweqa_noisy_oracle_gemma32k_dec8k_v1",
        "crosscodelong_chunk_gemma32k_dec8k_v1",
        "crosscodelong_function_gemma32k_dec8k_v1",
        "longcodeqa_gemma32k_dec8k_v1",
        "gsm8k_zero_shot_v2",
        "gsm8k_eight_shot_v2",
    )
]


def benchmark_jobs(snapshot):
    return [
        (
            item,
            {
                "benchmark": item,
                "execution": True,
                "protocol": V2_EVALUATION_PROTOCOL,
                "snapshot": snapshot,
            },
            "benchmark:" + snapshot + ":" + item["collection"],
        )
        for item in BENCHMARKS
    ]


def benchmark_directory_name(settings):
    """Reuse the original Hub folder even when the evaluation protocol changes.

    The actual protocol remains in the stage and prediction signatures. This
    legacy value determines the storage address only, never evaluation behavior.
    """
    from bifocalgemma.data.offline_sft import stable_fingerprint

    storage = {**settings, "protocol": "greedy_cached_normalized_merged_v2"}
    return (
        settings["snapshot"]
        + "-"
        + settings["benchmark"]["collection"]
        + "-"
        + stable_fingerprint(storage)[:12]
    )


def download_model(spec, cache, token, progress):
    from huggingface_hub import hf_hub_download

    manifest = json.loads(
        files("bifocalgemma.assets").joinpath("tokenizer_manifest.json").read_text()
    )
    weights = []
    provenance = []
    for ref in [spec.decoder] + ([spec.encoder] if spec.encoder else []):
        assets = manifest[ref.model_id]
        if assets["revision"] != ref.revision:
            raise ValueError("Backbone revision is not audited.")
        raw = Path(cache) / "tokenizers/raw" / ref.model_id.replace("/", "__")
        raw.mkdir(parents=True, exist_ok=True)
        for name, digest in assets["files"].items():
            path = Path(
                hf_hub_download(
                    ref.model_id,
                    name,
                    revision=ref.revision,
                    token=token,
                    cache_dir=str(Path(cache) / "hub"),
                )
            )
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Pinned asset hash mismatch: " + name)
            shutil.copyfile(path, raw / name)
        load_verified_tokenizer(ref, Path(cache) / "tokenizers")
        progress({"stage": "checkpoint_download", "message": ref.model_id + " @ " + ref.revision})
        path = hf_hub_download(
            ref.model_id,
            "model.safetensors",
            revision=ref.revision,
            token=token,
            cache_dir=str(Path(cache) / "hub"),
        )
        weights.append(path)
        provenance.append(
            {"model_id": ref.model_id, "revision": ref.revision, "assets": assets["files"]}
        )
    model, source = load_local_native_model(
        spec, weights[0], weights[1] if spec.encoder else None, seed=17, headroom_gib=8
    )
    source.update(source_identity_verified=True, verified_sources=provenance)
    return model, source


def target_fingerprint(examples):
    """Comparable across architectures: no prefix or encoder token contribution."""
    import struct

    digest = hashlib.sha256()
    for item in examples:
        identifier = item.instance_id.encode()
        digest.update(struct.pack("<I", len(identifier)))
        digest.update(identifier)
        digest.update(struct.pack("<I", len(item.target_ids)))
        digest.update(struct.pack("<" + "I" * len(item.target_ids), *item.target_ids))
    return digest.hexdigest()


def source_fingerprint():
    return source_fingerprints()["training_semantics_sha256"]


def _execute_v2(
    model_key,
    *,
    mode,
    token,
    mo=None,
    output_root="outputs/v2",
    cache="outputs/v2_cache",
    baseline_snapshot="trained",
    upload=True,
    monitor=None,
):
    if model_key not in MODELS or mode not in {"pilot", "train", "benchmark"}:
        raise ValueError("Choose a registered model and pilot/train/benchmark mode.")
    if version("transformers") != "5.20.2":
        raise RuntimeError(
            "Molab v2 requires transformers==5.20.2, matching the local ai environment."
        )
    if baseline_snapshot not in {"trained", "initial"}:
        raise ValueError("Invalid baseline snapshot.")
    name, family = MODELS[model_key]
    spec = get_v2_experiment(name)
    policy = unified_recipe(family)
    if baseline_snapshot == "initial" and (spec.encoder is not None or mode != "benchmark"):
        raise ValueError("Initial snapshot is supported only in baseline benchmark mode.")
    if not torch.cuda.is_available():
        raise RuntimeError("Molab CUDA GPU required for production execution.")
    repo = "daruokta/bifocalgemma-v2-" + model_key
    run_name = execution_run_name(mode)
    root = Path(output_root) / model_key / run_name
    prefix = f"{family}/{name}/{run_name}"
    if upload:
        if not token:
            raise ValueError("An HF write token is required.")
        recover_remote_run(repo, root, prefix, token=token)
    reset_local_benchmarks(root, BENCHMARK_RESET_ID)
    identity = {
        "version": "normalized_merged_v2",
        "spec": spec.to_dict(),
        "policy": policy,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "collection": COLLECTION,
        "run_name": run_name,
        "seed": 17,
        "epochs": 1,
        "effective_batch": 4,
        "execution_policy": EXECUTION_VERSION,
        "source_fingerprint": source_fingerprint(),
        "source_fingerprint_policy": "training_semantics_v1",
        "runtime_versions": {
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "huggingface-hub",
                "numpy",
                "pyarrow",
            )
        },
    }
    ledger = StageLedger(root, identity)
    if not ledger.path.exists():
        atomic_json(ledger.path, ledger.state)
    atomic_json(root / "diagnostics/code_provenance.json", source_fingerprints())
    dashboard = (
        monitor if monitor is not None else TrainingDashboard(mo, experiment=name, device="cuda")
    )
    dashboard.run_dir = root
    heartbeat = RemoteHeartbeat(repo, root, prefix, token=token) if upload else None
    guard_queue = None

    def progress(event: dict[str, Any]):
        if upload and guard_queue is not None and guard_queue.state.get("pending"):
            if __import__("time").time() >= guard_queue.state.get("next_retry_at", 0):
                guard_queue.attempt(lambda: synchronize_preflight(repo, root, prefix, token=token))
        event = {
            **event,
            "upload_pending": event.get("upload_pending", False)
            or queue.state.get("pending", False)
            or (guard_queue is not None and guard_queue.state.get("pending", False)),
            "final_synced": queue.state.get("final_synced", False),
        }
        dashboard(event)
        if event.get("stage", "").startswith("preflight_"):
            append_event(root / "diagnostics/preflight_events.jsonl", event)
            print(event.get("message", event["stage"]), flush=True)
        if heartbeat:
            heartbeat(
                event,
                force=event.get("stage")
                in {"training_completed", "pilot_completed", "benchmark_completed"},
            )

    queue = BenchmarkUploadQueue(
        root,
        backoff_path=root / ".upload-backoff.json",
        progress=dashboard,
        interval=1800,
        label="Checkpoint / results",
    )

    def snapshot(*, final=False):
        if not upload:
            return None
        queue.mark_pending()
        result = queue.attempt(
            lambda: synchronize_v2(
                repo, root, prefix, token=token, require_checkpoint=baseline_snapshot != "initial"
            ),
            final=final,
        )
        if result and heartbeat:
            dashboard(
                {
                    "stage": "remote_snapshot_verified",
                    "message": "HF verified @ " + result["revision"],
                    "last_remote_checkpoint": result["retained_checkpoints"][-1]
                    if result["retained_checkpoints"]
                    else None,
                    "remote_revision": result["revision"],
                    "upload_pending": False,
                }
            )
            heartbeat(
                {
                    "remote_revision": result["revision"],
                    "last_remote_checkpoint": result["retained_checkpoints"][-1]
                    if result["retained_checkpoints"]
                    else None,
                    "upload_pending": False,
                    "final_synced": final,
                },
                force=final,
            )
        return result

    progress({"stage": "starting", "message": f"{name} / {mode}; source encoder=base, decoder=IT"})
    training_complete = ledger.completed("training")
    diagnostics_complete = (root / "diagnostics/pilot_report.json").is_file()
    if mode != "benchmark" and training_complete and (mode == "train" or diagnostics_complete):
        snapshot(final=True)
        progress({"stage": "training_completed" if mode == "train" else "pilot_completed"})
        return {
            "status": "COMPLETED" if not queue.state.get("pending") else "PENDING_UPLOAD",
            "mode": mode,
            "model": name,
            "repo": repo,
            "message": "Training already complete; no model loaded or benchmark launched.",
        }
    if mode == "benchmark" and baseline_snapshot == "trained" and not training_complete:
        raise ValueError("No completed v2 training; select train mode first.")
    jobs = benchmark_jobs(baseline_snapshot) if mode == "benchmark" else []
    completed_reports = {
        stage: ledger.state["stages"][stage]["result"]
        for _, settings, stage in jobs
        if ledger.completed(stage, settings=settings)
    }
    if jobs and len(completed_reports) == len(jobs):
        if baseline_snapshot == "trained" and latest_checkpoint(root / "training") is None:
            raise ValueError("Completed run has no valid checkpoint.")
        snapshot(final=True)
        progress(
            {"stage": "benchmark_completed", "items_done": len(jobs), "items_total": len(jobs)}
        )
        return {
            "status": "COMPLETED"
            if not upload or not queue.state.get("pending")
            else "PENDING_UPLOAD",
            "mode": mode,
            "model": name,
            "repo": repo,
            "benchmarks": [completed_reports[stage] for _, _, stage in jobs],
        }
    profile = None
    if mode != "benchmark":
        profile = resolve_shared_profile(
            model_key,
            root,
            shared_contract(identity),
            token=token,
            progress=progress,
        )
    model, provenance = download_model(spec, cache, token, progress)
    benchmark_mode = mode == "benchmark"
    if benchmark_mode:
        if baseline_snapshot == "trained":
            if not ledger.completed("training"):
                raise ValueError("No completed v2 training; select train mode first.")
            checkpoint = latest_checkpoint(root / "training")
            if checkpoint is None:
                raise ValueError("Completed run has no valid checkpoint.")
            load_sft_adapters_for_inference(model, checkpoint)
            checkpoint_identity = json.loads((Path(checkpoint) / "READY.json").read_text())
        else:
            checkpoint_identity = {"initial_model": spec.to_dict()}
        configure_memory_efficient_training(model)
        report_rows = []
        for item, settings, stage in jobs:
            if stage in completed_reports:
                report_rows.append(completed_reports[stage])
                continue
            directory = Path(cache) / "benchmarks" / item["collection"]
            fetch_collection(
                DATASET_REPO,
                DATASET_REVISION,
                item["collection"],
                directory,
                expected_role="eval",
                include_gold=True,
                token=token,
            )
            destination = root / "benchmarks" / benchmark_directory_name(settings)
            ledger.start(stage)
            report = run_context_benchmark(
                model,
                spec,
                directory,
                destination,
                Path(cache) / "tokenizers",
                checkpoint_identity=checkpoint_identity,
                encoder_limit=32768,
                decoder_limit=8192,
                max_new_tokens=item.get("max_new_tokens", 2048),
                allow_execution=True,
                progress=progress,
                prediction_callback=lambda: snapshot(),
                upload_every=100,
                use_cache=True,
            )
            ledger.finish(
                stage,
                [
                    destination / "benchmark_report.json",
                    destination / "predictions.jsonl",
                    destination / "evaluation_state.json",
                ],
                report,
                settings=settings,
            )
            report_rows.append(report)
            # Initial IT evaluation has no trained checkpoint; preserve results using a separate journal upload.
            snapshot(final=True)
        if baseline_snapshot == "initial":
            snapshot(final=True)
        progress(
            {
                "stage": "benchmark_completed",
                "items_done": len(report_rows),
                "items_total": len(BENCHMARKS),
            }
        )
        return {
            "status": "COMPLETED"
            if not upload or not queue.state.get("pending")
            else "PENDING_UPLOAD",
            "mode": mode,
            "model": name,
            "repo": repo,
            "benchmarks": report_rows,
        }
    if training_complete:
        checkpoint = latest_checkpoint(root / "training")
        if checkpoint is None:
            raise ValueError("Completed pilot has no complete checkpoint.")
        load_sft_adapters_for_inference(model, checkpoint)
        configure_memory_efficient_training(model)
        pool = Path(cache) / "dataset" / COLLECTION
        fetch_collection(
            DATASET_REPO, DATASET_REVISION, COLLECTION, pool, expected_role="train", token=token
        )
        _, dev, _ = prepare_curated_training(
            spec,
            pool,
            Path(cache) / "tokenizers",
            packed_cache=Path(cache) / "packed",
            progress=progress,
        )
        diagnostic = encoder_diagnostics(model, stratified_subset(dev, 256), progress=progress)
        diagnostic["full_training_auto_started"] = False
        atomic_json(root / "diagnostics/pilot_report.json", diagnostic)
        snapshot(final=True)
        progress({"stage": "pilot_completed"})
        return {
            "status": "COMPLETED" if not queue.state.get("pending") else "PENDING_UPLOAD",
            "mode": mode,
            "model": name,
            "repo": repo,
            "message": "Training already complete; no benchmark launched.",
        }
    pool = Path(cache) / "dataset" / COLLECTION
    fetch_collection(
        DATASET_REPO, DATASET_REVISION, COLLECTION, pool, expected_role="train", token=token
    )
    train, dev, manifest = prepare_curated_training(
        spec,
        pool,
        Path(cache) / "tokenizers",
        packed_cache=Path(cache) / "packed",
        progress=progress,
    )
    manifest["matched_target_fingerprints"] = {
        "train": target_fingerprint(train),
        "dev": target_fingerprint(dev),
    }
    full_train, full_dev = train, dev
    if mode == "pilot":
        train = stratified_subset(train, 800)
        dev = stratified_subset(dev, 256)
        if len(train) < 800:
            raise ValueError(
                "Pilot requires 800 examples for exactly 200 effective-batch-four steps."
            )
        manifest["pilot_instances"] = {
            "train": [x.instance_id for x in train],
            "dev": [x.instance_id for x in dev],
        }
    configure_policy_model(model, policy)
    selected_resume = latest_checkpoint(root / "training")
    execution_path = root / "diagnostics/execution_preflight.json"
    guard_queue = BenchmarkUploadQueue(
        root / "diagnostics",
        backoff_path=root / ".upload-backoff.json",
        progress=dashboard,
        interval=0,
        label="Memory guard",
    )

    def save_guard():
        if upload:
            guard_queue.mark_pending()
            guard_queue.attempt(lambda: synchronize_preflight(repo, root, prefix, token=token))

    if profile is None:
        profile = calibrate_shared_profile(
            model,
            full_train,
            full_dev,
            policy,
            shared_contract(identity),
            root,
            progress=progress,
            on_saved=save_guard,
        )
        # Followers cannot train until the one selected profile is published.
        profile = publish_with_retry(profile, token=token, progress=progress)
        atomic_json(root / "diagnostics/shared_profile.json", profile)
    execution = select_execution(
        model,
        train,
        dev,
        progress=progress,
        state_path=execution_path,
        identity=identity,
        on_saved=save_guard,
        training_policy=policy,
        execution=profile,
        min_headroom_gib=HEADROOM_GIB,
    )
    atomic_json(execution_path, execution)
    batch_size, accumulation = execution["batch_size"], execution["accumulation"]
    if (batch_size, accumulation, execution["memory_policy"]) != (
        profile["batch_size"],
        profile["accumulation"],
        {**DEFAULT_MEMORY_POLICY, **profile["memory_policy"]},
    ):
        raise ValueError(
            "Execution differs from the shared E4B→E4B profile; per-model tuning is forbidden."
        )
    if selected_resume:
        config = json.loads((root / "training/run_config.json").read_text())
        if (config["batch_size"], config["accumulation"], config["memory_policy"]) != (
            batch_size,
            accumulation,
            model.memory_policy,
        ):
            raise ValueError(
                "Saved training execution differs from the fixed recipe; use a fresh run."
            )
    steps = 200 if mode == "pilot" else math.ceil(len(train) / 4)
    if spec.encoder:
        assert model.encoder is not None
    encoder_width = model.encoder.config.hidden_size if model.encoder is not None else None
    compatibility = {
        "architecture": "normalized_merged_v2",
        "experiment": name,
        "model_spec": spec.to_dict(),
        "sources": provenance,
        "data_fingerprint": manifest["fingerprint"],
        "matched_target_fingerprints": manifest["matched_target_fingerprints"],
        "pilot_instances": manifest.get("pilot_instances"),
        "seed": 17,
        "batch_size": batch_size,
        "accumulation": accumulation,
        "training_policy": policy,
        "memory_policy": model.memory_policy,
        "shared_execution_profile": profile["evidence_sha256"],
        "mode": mode,
        "source_fingerprint": identity["source_fingerprint"],
        "torch": str(torch.__version__),
        "transformers": "5.20.2",
        "alignment": {
            "type": "linear_rmsnorm",
            "epsilon": 1e-6,
            "seed": 17,
            "dtype": "float32",
            "input_width": encoder_width,
            "output_width": model.decoder.config.hidden_size,
            "initialization": "identity"
            if spec.encoder.size == spec.decoder.size
            else "semi_orthogonal",
        }
        if spec.encoder
        else None,
    }
    train_root = root / "training"
    if selected_resume is None and train_root.exists() and any(train_root.iterdir()):
        from uuid import uuid4

        previous = json.loads((train_root / "run_config.json").read_text())
        for key in (
            "architecture",
            "model_spec",
            "data_fingerprint",
            "training_policy",
            "mode",
            "seed",
        ):
            if previous.get(key) != compatibility[key]:
                raise ValueError(
                    "Incomplete run configuration changed; choose a fresh output root."
                )
        archive = root / (".interrupted-before-checkpoint-" + uuid4().hex)
        if (
            train_root.resolve().parent != root.resolve()
            or archive.resolve().parent != root.resolve()
        ):
            raise ValueError("Interrupted training archive escaped run directory.")
        train_root.rename(archive)
    ledger.start("training")
    report = fit_offline_sft(
        model,
        train,
        dev,
        root / "training",
        compatibility,
        max_steps=steps,
        epochs=1,
        batch_size=batch_size,
        accumulation=accumulation,
        eval_batch_size=batch_size,
        training_policy=policy,
        clip_norm=1,
        seed=17,
        eval_every=max(1, steps),
        save_every=max(1, steps),
        save_interval_seconds=900,
        resume_from=selected_resume,
        event_callback=progress,
        data_manifest=manifest,
        checkpoint_callback=lambda path: snapshot(),
        save_total_limit=2,
    )
    report.update(provenance=provenance, execution_preflight=execution, mode=mode)
    atomic_json(root / "training/training_report.json", report)
    ledger.finish(
        "training",
        [
            root / "training/training_report.json",
            root / "training/run_config.json",
            Path(report["last_checkpoint"]) / "READY.json",
        ],
        {"global_step": report["global_step"]},
    )
    if mode == "pilot":
        diagnostic = encoder_diagnostics(model, dev, progress=progress)
        diagnostic.update(
            training_optimizer_diagnostics=[
                r for r in report["history"] if "optimizer_diagnostics" in r
            ],
            full_training_auto_started=False,
        )
        atomic_json(root / "diagnostics/pilot_report.json", diagnostic)
    snapshot(final=True)
    if hasattr(full_train, "close"):
        full_train.close()
    if hasattr(full_dev, "close"):
        full_dev.close()
    progress(
        {
            "stage": "training_completed",
            "step": report["global_step"],
            "upload_pending": queue.state.get("pending", False),
        }
    )
    return {
        "status": "COMPLETED" if not upload or not queue.state.get("pending") else "PENDING_UPLOAD",
        "model": name,
        "mode": mode,
        "repo": repo,
        "global_step": report["global_step"],
        "held_out_eval": report["held_out_eval"],
        "benchmark_auto_started": False,
        "run_dir": str(root),
    }


def run_v2(
    model_key,
    *,
    mode,
    token,
    mo=None,
    output_root="outputs/v2",
    cache="outputs/v2_cache",
    baseline_snapshot="trained",
    upload=True,
    monitor=None,
):
    """Persist interruptions without allowing upload errors to hide the original failure."""
    try:
        return _execute_v2(
            model_key,
            mode=mode,
            token=token,
            mo=mo,
            output_root=output_root,
            cache=cache,
            baseline_snapshot=baseline_snapshot,
            upload=upload,
            monitor=monitor,
        )
    except BaseException as error:
        if model_key in MODELS and mode in {"pilot", "train", "benchmark"}:
            name, family = MODELS[model_key]
            run_name = execution_run_name(mode)
            root = Path(output_root) / model_key / run_name
            event = {
                "stage": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "last_seen": __import__("time").time(),
            }
            atomic_json(root / "failure.json", event)
            path = root / "pipeline_state.json"
            if path.exists():
                state = json.loads(path.read_text())
                for item in state["stages"].values():
                    if item.get("status") == "RUNNING":
                        item.update(
                            status="INTERRUPTED"
                            if isinstance(error, KeyboardInterrupt)
                            else "FAILED",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                atomic_json(path, state)
            if upload and token:
                try:
                    RemoteHeartbeat(
                        "daruokta/bifocalgemma-v2-" + model_key,
                        root,
                        f"{family}/{name}/{run_name}",
                        token=token,
                    )(event, force=True)
                except Exception as upload_error:
                    error.add_note(
                        "Failure status could not be uploaded: " + type(upload_error).__name__
                    )
        raise


def run_training_and_benchmark(
    model_key,
    *,
    token,
    mo=None,
    output_root="outputs/v2",
    cache="outputs/v2_cache",
    upload=True,
    monitor=None,
):
    """Notebook entry point: resume training, then resume trained-model evaluation."""
    kwargs = dict(
        token=token, mo=mo, output_root=output_root, cache=cache, upload=upload, monitor=monitor
    )
    training = run_v2(model_key, mode="train", **kwargs)
    # Release the training model/optimizer before the evaluation loader checks
    # free VRAM. Completed stages already return without loading a model.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluation = run_v2(model_key, mode="benchmark", **kwargs)
    return {
        "status": evaluation["status"],
        "model": evaluation["model"],
        "repo": evaluation["repo"],
        "training": training,
        "benchmarks": evaluation["benchmarks"],
    }
