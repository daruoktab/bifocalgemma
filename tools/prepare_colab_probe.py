"""Build a standalone Colab diagnostic script; never launch an epoch or publish a profile."""

import argparse
import ast
import inspect
import json
from pathlib import Path

from notebook_bundle import source_bundle_bytes

ROOT = Path(__file__).resolve().parents[1]


def declared_limit_example(actual, encoder_limit, decoder_limit):
    from bifocalgemma.contracts import TokenizedRepairExample

    def stretch(tokens, size):
        tokens = tuple(tokens)
        if size < len(tokens):
            raise ValueError("Boundary probe must not truncate the observed maximum.")
        if size and not tokens:
            raise ValueError("Cannot extend an empty token sequence.")
        return (tokens * ((size + len(tokens) - 1) // len(tokens)))[:size] if size else ()

    return TokenizedRepairExample(
        "declared-limit-shape-only",
        stretch(actual.encoder_ids, encoder_limit),
        stretch(actual.decoder_prefix_ids, decoder_limit - len(actual.target_ids) + 1),
        actual.target_ids,
        {"shape_only": True, "limits": {"encoder": encoder_limit, "decoder": decoder_limit}},
    )


def probe(runtime_root):
    import hashlib
    import json
    import os
    import platform
    import random
    import subprocess
    import time
    from importlib.metadata import version
    from pathlib import Path
    from typing import Any

    import torch
    from huggingface_hub import get_token

    from bifocalgemma.config import get_v2_experiment
    from bifocalgemma.data.releases import fetch_collection
    from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
    from bifocalgemma.training.curated_sft import prepare_curated_training
    from bifocalgemma.training.policy import configure_policy_model
    from bifocalgemma.training.recipes import unified_recipe
    from bifocalgemma.training.shared_execution import (
        calibrate_shared_profile,
        numerical_probe,
        probe_optimizer,
        representative_cohort,
        shared_contract,
    )
    from bifocalgemma.training.stages import atomic_json
    from bifocalgemma.training.telemetry import append_event, resource_snapshot
    from bifocalgemma.training.v2 import (
        COLLECTION,
        DATASET_REPO,
        DATASET_REVISION,
        download_model,
        source_fingerprint,
    )
    from bifocalgemma.training.v2_preflight import EXECUTION_VERSION

    root = Path("/content/bifocalgemma-test")
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    started = time.monotonic()

    def progress(event):
        row = append_event(root / "events.jsonl", event)
        atomic_json(root / "status.json", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("The diagnostic requires a CUDA GPU.")
    props = torch.cuda.get_device_properties(0)
    hardware = {
        "name": props.name,
        "total_gib": props.total_memory / 2**30,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": version("triton"),
        "runtime_root": str(runtime_root),
        "nvidia_smi": subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        ).stdout,
    }
    atomic_json(root / "hardware.json", hardware)
    progress({"stage": "hardware", **hardware})
    if "6000" not in props.name or props.total_memory / 2**30 < 85:
        raise RuntimeError("Expected a full RTX PRO 6000 G4; refusing to calibrate another GPU.")
    secret = Path("/content/.bifocalgemma-hf-token")
    if secret.is_file():
        try:
            os.environ["HF_TOKEN"] = secret.read_text().strip()
        finally:
            secret.unlink()
    token = get_token()
    if not token:
        raise RuntimeError("HF authentication is required for the gated base/IT backbones.")
    random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    spec = get_v2_experiment("BifocalGemma-E4B-E4B")
    policy = unified_recipe("symmetric")
    identity = {
        "execution_policy": EXECUTION_VERSION,
        "source_fingerprint": source_fingerprint(),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "collection": COLLECTION,
        "seed": 17,
        "effective_batch": 4,
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
    atomic_json(root / "identity.json", identity)
    progress({"stage": "loading", "message": "Diagnostic E4B base → E4B IT; no epoch training."})
    model, provenance = download_model(spec, cache, token, progress)
    atomic_json(root / "provenance.json", provenance)
    pool = cache / "dataset" / COLLECTION
    fetch_collection(
        DATASET_REPO, DATASET_REVISION, COLLECTION, pool, expected_role="train", token=token
    )
    train, dev, manifest = prepare_curated_training(
        spec, pool, cache / "tokenizers", packed_cache=cache / "packed", progress=progress
    )
    atomic_json(root / "data_manifest.json", manifest)
    configure_policy_model(model, policy)

    def weights_digest():
        digest = hashlib.sha256()
        for name, parameter in sorted(model.named_parameters()):
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(
                    parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
        return digest.hexdigest()

    before = weights_digest()
    outcome: dict[str, Any] = {"failure": None, "status": "COMPLETED"}
    try:
        profile = calibrate_shared_profile(
            model, train, dev, policy, shared_contract(identity), root, progress=progress
        )
        atomic_json(root / "profile.json", profile)
        # Explain strict backend rejections without weakening the acceptance gate.
        cohort = representative_cohort(train)
        item = min(
            cohort,
            key=lambda x: len(x.encoder_ids) + len(x.decoder_prefix_ids) + len(x.target_ids),
        )
        details: dict[str, Any] = {}
        gradient_sets = {}
        with probe_optimizer(model, policy):
            configure_memory_efficient_training(model, {"attention_backend": "tiled_eager_fp32"})
            reference_loss, reference = numerical_probe(model, item)
            for backend in ("tiled_eager_fp32", "flex_attention", "hybrid_flex"):
                progress({"stage": "numerical_details", "message": backend})
                try:
                    configure_memory_efficient_training(model, {"attention_backend": backend})
                    loss, gradients = numerical_probe(model, item)
                    gradient_sets[backend] = gradients
                    rows = []
                    for name, truth in reference.items():
                        actual = gradients[name]
                        difference = actual - truth
                        rows.append(
                            {
                                "parameter": name,
                                "reference_norm": float(truth.norm()),
                                "difference_norm": float(difference.norm()),
                                "relative_l2": float(
                                    difference.norm() / truth.norm().clamp_min(1e-8)
                                ),
                                "max_absolute_error": float(difference.abs().max()),
                            }
                        )
                    details[backend] = {
                        "loss": loss,
                        "reference_loss": reference_loss,
                        "max_relative_l2": max(row["relative_l2"] for row in rows),
                        "global_relative_l2": (
                            sum(row["difference_norm"] ** 2 for row in rows)
                            / max(1e-16, sum(row["reference_norm"] ** 2 for row in rows))
                        )
                        ** 0.5,
                        "worst_parameters": sorted(
                            rows, key=lambda x: x["relative_l2"], reverse=True
                        )[:20],
                    }
                except (RuntimeError, ValueError, KeyError, ImportError) as error:
                    details[backend] = {"error": type(error).__name__ + ": " + str(error)}
                atomic_json(root / "numerical_details.json", details)
            # A full-model FP32 oracle distinguishes BF16 rounding from incorrect math.
            # This is evidence only: it does not relax the production acceptance gate.
            original_dtypes = [(n, p.dtype) for n, p in model.named_parameters()]
            original_buffers = [(n, b.dtype) for n, b in model.named_buffers()]
            try:
                progress(
                    {"stage": "numerical_details", "message": "FP32 oracle; no policy change."}
                )
                model.float()
                configure_memory_efficient_training(
                    model, {"attention_backend": "tiled_eager_fp32"}
                )
                oracle_loss, oracle = numerical_probe(model, item)

                def oracle_error(gradients):
                    rows = []
                    for name, truth in oracle.items():
                        difference = gradients[name] - truth
                        rows.append(
                            {
                                "parameter": name,
                                "reference_norm": float(truth.norm()),
                                "difference_norm": float(difference.norm()),
                                "relative_l2": float(
                                    difference.norm() / truth.norm().clamp_min(1e-8)
                                ),
                            }
                        )
                    return {
                        "global_relative_l2": (
                            sum(r["difference_norm"] ** 2 for r in rows)
                            / max(1e-16, sum(r["reference_norm"] ** 2 for r in rows))
                        )
                        ** 0.5,
                        "max_relative_l2": max(r["relative_l2"] for r in rows),
                        "worst_parameters": sorted(
                            rows, key=lambda r: r["relative_l2"], reverse=True
                        )[:20],
                    }

                details["bf16_vs_fp32_oracle"] = {
                    "oracle_loss": oracle_loss,
                    "backends": {name: oracle_error(g) for name, g in gradient_sets.items()},
                }
                configure_memory_efficient_training(model, {"attention_backend": "hybrid_flex"})
                hybrid_loss, hybrid_gradients = numerical_probe(model, item)
                details["hybrid_fp32_vs_tiled_fp32"] = {
                    "loss": hybrid_loss,
                    **oracle_error(hybrid_gradients),
                }
            except (RuntimeError, ValueError, KeyError, ImportError) as error:
                details["fp32_oracle_error"] = type(error).__name__ + ": " + str(error)
            finally:
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, dtype in original_dtypes:
                        parameter = model.get_parameter(name)
                        parameter.data = parameter.data.to(dtype)
                    for name, dtype in original_buffers:
                        buffer = model.get_buffer(name)
                        buffer.data = buffer.data.to(dtype)
                atomic_json(root / "numerical_details.json", details)
    except BaseException as error:
        outcome["status"] = "FAILED"
        outcome["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        after = weights_digest()
        report = {
            "status": outcome["status"],
            "diagnostic_only": True,
            "epoch_training_started": False,
            "profile_published": False,
            "trainable_sha256_before": before,
            "trainable_sha256_after": after,
            "weights_restored_exactly": before == after,
            "wall_seconds": time.monotonic() - started,
            "failure": outcome["failure"],
            "resources": resource_snapshot("cuda"),
        }
        if before != after:
            report["status"] = "FAILED_WEIGHT_RESTORE"
        atomic_json(root / "report.json", report)
        progress({"stage": "diagnostic_finished", **report})
        if before != after:
            raise RuntimeError("Diagnostic modified trainable weights; profile must not be used.")


def check_followers(runtime_root, *, boundary=False):
    import gc
    import hashlib
    import json
    import os
    from pathlib import Path
    from typing import Any

    import torch
    from huggingface_hub import get_token

    from bifocalgemma.config import get_v2_experiment
    from bifocalgemma.data.releases import fetch_collection
    from bifocalgemma.training.curated_sft import prepare_curated_training
    from bifocalgemma.training.policy import configure_policy_model
    from bifocalgemma.training.recipes import unified_recipe
    from bifocalgemma.training.shared_execution import HEADROOM_GIB, validate_profile
    from bifocalgemma.training.stages import atomic_json
    from bifocalgemma.training.telemetry import append_event
    from bifocalgemma.training.v2 import (
        COLLECTION,
        DATASET_REPO,
        DATASET_REVISION,
        MODELS,
        download_model,
        source_fingerprint,
    )
    from bifocalgemma.training.v2_preflight import maximum_probe, seal_report, select_execution

    root = Path("/content/bifocalgemma-test")
    profile = json.loads((root / "profile.json").read_text())
    validate_profile(profile, profile["contract"])
    if profile["contract"]["source_fingerprint"] != source_fingerprint():
        raise ValueError("Follower diagnostic runtime differs from the calibrated source.")
    secret = Path("/content/.bifocalgemma-hf-token")
    if secret.is_file():
        try:
            os.environ["HF_TOKEN"] = secret.read_text().strip()
        finally:
            secret.unlink()
    token = get_token()
    if not token:
        raise RuntimeError("The follower guards require HF authentication.")
    cache = root / "cache"
    summary: dict[str, dict[str, Any]] = {}
    section = "boundaries" if boundary else "followers"

    def progress(event):
        row = append_event(root / "events.jsonl", event)
        atomic_json(root / "status.json", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    for key, (name, family) in MODELS.items():
        if key == "e4b-e4b" and not boundary:
            continue
        spec = get_v2_experiment(name)
        model = None
        try:
            progress(
                {
                    "stage": "follower_guard",
                    "model": key,
                    "message": "Same shared profile; no tuning, no epoch.",
                }
            )
            model, _ = download_model(spec, cache, token, progress)
            pool = cache / "dataset" / COLLECTION
            fetch_collection(
                DATASET_REPO, DATASET_REVISION, COLLECTION, pool, expected_role="train", token=token
            )
            train, dev, manifest = prepare_curated_training(
                spec, pool, cache / "tokenizers", packed_cache=cache / "packed", progress=progress
            )
            limits = None
            if boundary:
                limits = {
                    "encoder": manifest["limits"]["encoder"] if spec.encoder else 0,
                    "decoder": manifest["limits"][
                        "pair_decoder" if spec.encoder else "baseline_decoder"
                    ],
                }
                actual = maximum_probe(train, dev)

                shape = declared_limit_example(actual, limits["encoder"], limits["decoder"])
                train, dev = [shape], [shape]
            policy = unified_recipe(family)
            configure_policy_model(model, policy)

            def weights_digest(current_model):
                digest = hashlib.sha256()
                for parameter_name, parameter in sorted(current_model.named_parameters()):
                    if parameter.requires_grad:
                        digest.update(parameter_name.encode())
                        digest.update(
                            parameter.detach()
                            .cpu()
                            .contiguous()
                            .view(torch.uint8)
                            .numpy()
                            .tobytes()
                        )
                return digest.hexdigest()

            before = weights_digest(model)
            try:
                receipt = select_execution(
                    model,
                    train,
                    dev,
                    progress=progress,
                    state_path=root / section / key / "guard.json",
                    identity={
                        "contract": profile["contract"],
                        "model_spec": spec.to_dict(),
                        **({"boundary_limits": limits} if boundary else {}),
                    },
                    training_policy=policy,
                    execution=profile,
                    min_headroom_gib=HEADROOM_GIB,
                )
                summary[key] = {"status": "PASS", "validation": receipt}
            finally:
                after = weights_digest(model)
                if before != after:
                    raise RuntimeError("Follower diagnostic failed to restore weights.")
                if key in summary:
                    summary[key].update(
                        weights_restored_exactly=True,
                        trainable_sha256_before=before,
                        trainable_sha256_after=after,
                    )
        except Exception as error:
            summary[key] = {"status": "FAILED", "reason": type(error).__name__ + ": " + str(error)}
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(root / (section + ".json"), summary)
    if not all(row["status"] == "PASS" for row in summary.values()):
        raise RuntimeError("At least one follower rejected the shared profile; do not publish it.")
    if boundary:
        profile = seal_report(
            {
                **profile,
                "observed_data_validation": profile.get(
                    "observed_data_validation", profile["validation"]
                ),
                "validation": summary["e4b-e4b"]["validation"],
                "declared_limit_validation": {
                    key: row["validation"] for key, row in summary.items()
                },
            }
        )
        validate_profile(profile, profile["contract"])
        atomic_json(root / "profile.json", profile)
    progress(
        {"stage": section + "_complete", "models": len(summary), "epoch_training_started": False}
    )


def render(root=ROOT, *, followers=False, boundary=False):
    import hashlib

    manifest = json.loads((root / "notebooks/runtime-manifest.json").read_text())
    if hashlib.sha256(source_bundle_bytes(root)).hexdigest() != manifest["sha256"]:
        raise ValueError("Publish the matching runtime before exporting the Colab diagnostic.")
    loader = (root / "tools/runtime_bootstrap.py").read_text(encoding="utf-8")
    target = check_followers if followers or boundary else probe
    function = inspect.getsource(declared_limit_example) + "\n\n" + inspect.getsource(target)
    ast.parse(function)
    # A fresh subprocess avoids pre-imported Colab Torch/NumPy polluting the pinned environment.
    child = (
        "import runpy, sys\nfrom pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "runpy.run_path(str(root / 'tools/notebook_launcher.py'))['prepare_runtime'](root)\n"
        + function
        + (
            f"\n{target.__name__}(root, boundary=True)\n"
            if boundary
            else f"\n{target.__name__}(root)\n"
        )
    )
    return (
        '"""Generated diagnostic only: no epoch training, checkpoint upload or profile publication."""\n'
        + loader
        + "\nimport subprocess\nimport sys\nfrom pathlib import Path\n"
        + f"manifest = {manifest!r}\n"
        + "root = load_runtime(manifest['repo_id'], manifest['filename'], "
        "manifest['revision'], manifest['sha256'])\n"
        + "child = Path('/content/bifocalgemma_probe_worker.py')\n"
        + f"child.write_text({child!r}, encoding='utf-8')\n"
        + "results = Path('/content/bifocalgemma-test')\n"
        + "results.mkdir(parents=True, exist_ok=True)\n"
        + "try:\n"
        + "    with (results / 'worker.log').open('a', encoding='utf-8') as log:\n"
        + "        with subprocess.Popen([sys.executable, '-u', str(child), str(root)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:\n"
        + "            for line in process.stdout:\n"
        + "                log.write(line)\n"
        + "                log.flush()\n"
        + "                print(line, end='', flush=True)\n"
        + "            if process.wait() != 0:\n"
        + "                raise subprocess.CalledProcessError(process.returncode, process.args)\n"
        + "finally:\n"
        + "    import zipfile\n"
        + "    Path('/content/.bifocalgemma-hf-token').unlink(missing_ok=True)\n"
        + "    with zipfile.ZipFile('/content/bifocalgemma-test.zip', 'w', zipfile.ZIP_DEFLATED) as archive:\n"
        + "        for path in sorted(results.rglob('*')):\n"
        + "            relative = path.relative_to(results)\n"
        + "            if path.is_file() and relative.parts[0] != 'cache' and path.suffix in {'.json', '.jsonl', '.log'}:\n"
        + "                archive.write(path, str(relative))\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--followers", action="store_true", help="Export only guards for the other five models."
    )
    parser.add_argument(
        "--boundary",
        action="store_true",
        help="Export declared-limit shape guards for all six models.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            ROOT
            / "outputs/colab"
            / (
                "boundaries.py"
                if args.boundary
                else "followers.py"
                if args.followers
                else "probe.py"
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(followers=args.followers, boundary=args.boundary), encoding="utf-8", newline="\n"
    )
    print(f"Diagnostic ready: {args.output}")


if __name__ == "__main__":
    main()
