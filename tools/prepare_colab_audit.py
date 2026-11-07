"""Export a read-only trained-checkpoint audit for a dedicated Colab G4 session."""

import json
from pathlib import Path

import prepare_colab_probe


def audit(runtime_root):
    import gc
    import hashlib
    import json
    import os
    import sys
    import time
    from pathlib import Path
    from typing import Any

    import torch
    import torch.nn.functional as functional
    from huggingface_hub import HfApi, get_token, hf_hub_download
    from safetensors.torch import load_file

    from bifocalgemma.config import get_v2_experiment
    from bifocalgemma.data.releases import fetch_collection
    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
    from bifocalgemma.evaluation.context import prompt_ids, score_context_case
    from bifocalgemma.inference.generation import greedy_decode
    from bifocalgemma.models.inference_cache import InferenceCache, cached_logits
    from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
    from bifocalgemma.models.v2.pretrained import load_sft_adapters_for_inference
    from bifocalgemma.training.stages import atomic_json
    from bifocalgemma.training.v2 import DATASET_REPO, DATASET_REVISION, MODELS, download_model

    root = Path("/content/bifocalgemma-test")
    root.mkdir(exist_ok=True)
    cache = root / "cache"
    secret = Path("/content/.bifocalgemma-hf-token")
    if secret.is_file():
        try:
            os.environ["HF_TOKEN"] = secret.read_text().strip()
        finally:
            secret.unlink()
    token = get_token()
    if not token:
        raise RuntimeError("Missing authorized HF credentials.")
    props = torch.cuda.get_device_properties(0)
    if "6000" not in props.name or props.total_memory / 2**30 < 85:
        raise RuntimeError("Expected G4 RTX PRO 6000 with at least 85 GiB.")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    report: dict[str, Any] = {
        "hardware": {
            "gpu": props.name,
            "vram_gib": props.total_memory / 2**30,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "models": {},
        "epoch_training_started": False,
        "hf_writes": False,
    }

    def event(stage, **details):
        payload = {"stage": stage, "time": time.time(), **details}
        atomic_json(root / "status.json", payload)
        print(json.dumps(payload), flush=True)

    def save():
        atomic_json(root / "audit.json", report)

    directory = cache / "benchmarks/humanevalfix_python_v2"
    fetch_collection(
        DATASET_REPO,
        DATASET_REVISION,
        directory.name,
        directory,
        expected_role="eval",
        include_gold=True,
        token=token,
    )
    rows = [json.loads(line) for line in (directory / "eval_inputs.jsonl").read_text().splitlines()]
    gold = {
        r["instance_id"]: r
        for r in [
            json.loads(line)
            for line in (directory / "eval_private/specs.jsonl").read_text().splitlines()
        ]
    }
    selected = [r for r in rows if r["instance_id"] in {"Python/0", "Python/1", "Python/39"}]
    api = HfApi(token=token)
    for key in ("e2b-e2b", "e4b-e4b", "e2b-it", "e4b-it"):
        event("loading", model=key)
        name, family = MODELS[key]
        spec = get_v2_experiment(name)
        repo = "daruokta/bifocalgemma-v2-" + key
        revision = api.model_info(repo).sha
        prefix = f"{family}/{name}/train/training/checkpoint-011102/"
        checkpoint = cache / "checkpoints" / key
        checkpoint.mkdir(parents=True, exist_ok=True)
        for filename in ("READY.json", "manifest.json", "adapters.safetensors"):
            path = hf_hub_download(repo, prefix + filename, revision=revision, token=token)
            import shutil

            shutil.copyfile(path, checkpoint / filename)
        model, provenance = download_model(spec, cache, token, lambda e: None)
        receipt = load_sft_adapters_for_inference(model, checkpoint)
        weights = load_file(str(checkpoint / "adapters.safetensors"))
        params = dict(model.named_parameters())
        mismatches = [
            n for n, value in weights.items() if not torch.equal(params[n].detach().cpu(), value)
        ]
        if mismatches:
            raise RuntimeError(
                "Loaded adapter tensors differ from checkpoint: " + str(mismatches[:3])
            )
        digest = hashlib.sha256((checkpoint / "adapters.safetensors").read_bytes()).hexdigest()
        configure_memory_efficient_training(
            model, {"query_chunk_size": 2048, "gradient_checkpointing": False}
        )
        model.eval()
        decoder, _ = load_verified_tokenizer(spec.decoder, cache / "tokenizers")
        encoder = (
            load_verified_tokenizer(spec.encoder, cache / "tokenizers")[0] if spec.encoder else None
        )
        result = {
            "repo_revision": revision,
            "checkpoint_step": receipt["global_step"],
            "adapters_loaded_exactly": True,
            "adapter_sha256": digest,
            "trainable_tensors": len(weights),
            "cases": [],
        }
        report["models"][key] = result
        del weights, params
        for row in selected:
            event("case", model=key, instance=row["instance_id"])
            enc_ids, prefix_ids = prompt_ids(row, spec, encoder, decoder, encoder_limit=32768)
            target_text = gold[row["instance_id"]]["gold_patch"]
            target = tuple(decoder.encode(target_text, add_special_tokens=False))
            other = next(r for r in selected if r["instance_id"] != row["instance_id"])
            swapped = (
                tuple(encoder.encode(other["context"], add_special_tokens=True)) if encoder else ()
            )
            case = {
                "instance_id": row["instance_id"],
                "source_context": row["context"],
                "encoder_tokens": len(enc_ids),
                "decoder_prefix_tokens": len(prefix_ids),
                "loss": {},
                "generation": {},
            }
            variants = {"correct": enc_ids}
            if encoder:
                variants.update(empty=(), swapped=swapped)
            with torch.inference_mode():
                for variant, context in variants.items():
                    enc = torch.tensor([context], device="cuda", dtype=torch.long)
                    enc_mask = torch.ones_like(enc, dtype=torch.bool)
                    states = model.encode_context(enc, enc_mask) if encoder else None
                    enc_mask = enc_mask if encoder else None
                    full = prefix_ids + target
                    ids = torch.tensor([full[:-1]], device="cuda")
                    logits = model.decode_logits(
                        ids, torch.ones_like(ids, dtype=torch.bool), states, enc_mask
                    )
                    offset = len(prefix_ids) - 1
                    labels = torch.tensor([target], device="cuda")
                    case["loss"][variant] = float(
                        functional.cross_entropy(
                            logits[:, offset:].float().flatten(0, 1), labels.flatten()
                        )
                    )
                    # Cache prefill logits must agree with the reference path on exactly the same tokens.
                    if variant == "correct":
                        first_ids = torch.tensor([prefix_ids], device="cuda")
                        reference = model.decode_next_logits(
                            first_ids,
                            torch.ones_like(first_ids, dtype=torch.bool),
                            states,
                            enc_mask,
                        )
                        cached = cached_logits(model, first_ids, InferenceCache(), states, enc_mask)
                        case["prefill_parity"] = {
                            "max_abs": float((reference.float() - cached.float()).abs().max()),
                            "relative_l2": float(
                                (reference.float() - cached.float()).norm()
                                / reference.float().norm()
                            ),
                            "argmax_equal": bool(reference.argmax(-1).eq(cached.argmax(-1)).all()),
                        }
                        # Teacher-forced self-KV includes the native local-window boundary.
                        forced = (full * ((540 + len(full) - 1) // len(full)))[:540]
                        state = InferenceCache()
                        positions = []
                        for start, stop in (
                            (0, 510),
                            (510, 511),
                            (511, 512),
                            (512, 513),
                            (513, 540),
                        ):
                            chunk = torch.tensor([forced[start:stop]], device="cuda")
                            current = cached_logits(model, chunk, state, states, enc_mask)
                            all_ids = torch.tensor([forced[:stop]], device="cuda")
                            expected = model.decode_next_logits(
                                all_ids,
                                torch.ones_like(all_ids, dtype=torch.bool),
                                states,
                                enc_mask,
                            )
                            positions.append(
                                {
                                    "position": stop,
                                    "relative_l2": float(
                                        (expected.float() - current.float()).norm()
                                        / expected.float().norm()
                                    ),
                                    "argmax_equal": bool(
                                        expected.argmax(-1).eq(current.argmax(-1)).all()
                                    ),
                                }
                            )
                        case["cache_boundary_parity"] = positions
                    del logits
                for use_cache in (True, False):
                    generated = greedy_decode(
                        model,
                        enc_ids,
                        prefix_ids,
                        max_new_tokens=384,
                        max_total_decoder_tokens=41472,
                        use_cache=use_cache,
                    )
                    tokens = generated.token_ids
                    if tokens and tokens[-1] in (1, 106, 50):
                        tokens = tokens[:-1]
                    text = decode_text(decoder, tokens, skip_special_tokens=False)
                    score = score_context_case(
                        row,
                        gold[row["instance_id"]],
                        text,
                        directory=directory,
                        python_executable=sys.executable,
                        allow_execution=False,
                    )
                    case["generation"]["cached" if use_cache else "reference"] = {
                        "tokens": list(generated.token_ids),
                        "text": text,
                        "stop_reason": generated.stop_reason,
                        "patch_applied": score.get("patch_applied"),
                        "reason": score.get("reason"),
                    }
                case["generation_tokens_equal"] = (
                    case["generation"]["cached"]["tokens"]
                    == case["generation"]["reference"]["tokens"]
                )
            result["cases"].append(case)
            save()
            event(
                "case_complete",
                model=key,
                instance=row["instance_id"],
                loss=case["loss"],
                parity=case["prefill_parity"],
                same_generation=case["generation_tokens_equal"],
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        save()
    report["status"] = "COMPLETED"
    save()
    event("audit_complete")


def audit_controls(runtime_root):
    import gc
    import json
    import os
    import sys
    import time
    from pathlib import Path
    from typing import Any

    import torch
    from huggingface_hub import get_token

    from bifocalgemma.config import get_v2_experiment
    from bifocalgemma.data.context_tasks import task_prompt
    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
    from bifocalgemma.evaluation.context import prompt_ids, score_context_case
    from bifocalgemma.inference.generation import greedy_decode
    from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
    from bifocalgemma.models.v2.pretrained import load_sft_adapters_for_inference
    from bifocalgemma.training.stages import atomic_json
    from bifocalgemma.training.v2 import MODELS, download_model

    root = Path("/content/bifocalgemma-test")
    cache = root / "cache"
    secret = Path("/content/.bifocalgemma-hf-token")
    try:
        os.environ["HF_TOKEN"] = secret.read_text().strip()
    finally:
        secret.unlink(missing_ok=True)
    token = get_token()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    directory = cache / "benchmarks/humanevalfix_python_v2"
    rows = [json.loads(x) for x in (directory / "eval_inputs.jsonl").read_text().splitlines()]
    gold = {
        r["instance_id"]: r
        for r in [
            json.loads(x) for x in (directory / "eval_private/specs.jsonl").read_text().splitlines()
        ]
    }
    original = json.loads((root / "audit.json").read_text())
    result: dict[str, Any] = {"models": {}, "training": False, "hf_writes": False}
    for key in ("e2b-e2b", "e4b-e4b"):
        print(json.dumps({"stage": "controls_loading", "model": key}), flush=True)
        spec = get_v2_experiment(MODELS[key][0])
        assert spec.encoder is not None
        model, _ = download_model(spec, cache, token, lambda e: None)
        load_sft_adapters_for_inference(model, cache / "checkpoints" / key)
        configure_memory_efficient_training(
            model, {"query_chunk_size": 2048, "gradient_checkpointing": False}
        )
        model.eval()
        dec, _ = load_verified_tokenizer(spec.decoder, cache / "tokenizers")
        enc, _ = load_verified_tokenizer(spec.encoder, cache / "tokenizers")
        cases = []
        result["models"][key] = cases
        for row in [r for r in rows if r["instance_id"] in {"Python/0", "Python/1", "Python/39"}]:
            enc_ids, prefix = prompt_ids(row, spec, enc, dec, encoder_limit=32768)

            def render(text):
                text = dec.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                return tuple(dec.encode(text, add_special_tokens=False))

            variants = {
                "training_prompt": (enc_ids, render(task_prompt(row, paired=True))),
                "duplicated_context": (enc_ids, render(task_prompt(row, paired=False))),
                "decoder_context_only": ((), render(task_prompt(row, paired=False))),
            }
            case: dict[str, Any] = {"instance_id": row["instance_id"], "variants": {}}
            # Rescore the original tokens after removing EOS exactly as the real benchmark does.
            previous = next(
                c
                for c in original["models"][key]["cases"]
                if c["instance_id"] == row["instance_id"]
            )
            for name in ("cached", "reference"):
                tokens = previous["generation"][name]["tokens"]
                if tokens and tokens[-1] in (1, 106, 50):
                    tokens = tokens[:-1]
                text = decode_text(dec, tokens, skip_special_tokens=False)
                score = score_context_case(
                    row,
                    gold[row["instance_id"]],
                    text,
                    directory=directory,
                    python_executable=sys.executable,
                    allow_execution=True,
                )
                case["variants"]["benchmark_" + name] = {"text": text, "score": score}
            for name, (context, prompt) in variants.items():
                started = time.monotonic()
                generated = greedy_decode(
                    model,
                    context,
                    prompt,
                    max_new_tokens=384,
                    max_total_decoder_tokens=41472,
                    use_cache=True,
                )
                tokens = generated.token_ids
                if tokens and tokens[-1] in (1, 106, 50):
                    tokens = tokens[:-1]
                text = decode_text(dec, tokens, skip_special_tokens=False)
                score = score_context_case(
                    row,
                    gold[row["instance_id"]],
                    text,
                    directory=directory,
                    python_executable=sys.executable,
                    allow_execution=True,
                )
                case["variants"][name] = {
                    "text": text,
                    "score": score,
                    "seconds": time.monotonic() - started,
                    "stop_reason": generated.stop_reason,
                }
            cases.append(case)
            atomic_json(root / "audit-controls.json", result)
            print(
                json.dumps(
                    {
                        "stage": "controls_case_complete",
                        "model": key,
                        "instance": row["instance_id"],
                        "statuses": {k: v["score"]["status"] for k, v in case["variants"].items()},
                    }
                ),
                flush=True,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "COMPLETED"
    atomic_json(root / "audit-controls.json", result)
    print("AUDIT_CONTROLS_COMPLETED", flush=True)


def audit_decoder_control(runtime_root):
    import difflib
    import gc
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from typing import Any

    import torch
    from huggingface_hub import hf_hub_download

    from bifocalgemma.config import get_v2_experiment
    from bifocalgemma.data.context_tasks import task_prompt
    from bifocalgemma.data.patches import extract_unified_diff, patch_paths
    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
    from bifocalgemma.evaluation.context import score_context_case
    from bifocalgemma.inference.generation import greedy_decode
    from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
    from bifocalgemma.models.v2.pretrained import (
        load_local_native_model,
        load_sft_adapters_for_inference,
    )
    from bifocalgemma.repository_study import snapshot_from_json
    from bifocalgemma.training.memory_probe import ProbeLoRALinear
    from bifocalgemma.training.stages import atomic_json
    from bifocalgemma.training.v2 import MODELS

    root = Path("/content/bifocalgemma-test")
    cache = root / "cache"
    directory = cache / "benchmarks/humanevalfix_python_v2"
    rows = {
        r["instance_id"]: r
        for r in [json.loads(x) for x in (directory / "eval_inputs.jsonl").read_text().splitlines()]
    }
    gold = {
        r["instance_id"]: r
        for r in [
            json.loads(x) for x in (directory / "eval_private/specs.jsonl").read_text().splitlines()
        ]
    }
    controls = json.loads((root / "audit-controls.json").read_text())
    report: dict[str, Any] = {
        "decoder_without_lora": {},
        "recount_only": [],
        "training": False,
        "hf_writes": False,
    }
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    for key in ("e2b-e2b", "e4b-e4b"):
        spec = get_v2_experiment(MODELS[key][0])
        assert spec.encoder is not None
        paths = [
            hf_hub_download(
                ref.model_id,
                "model.safetensors",
                revision=ref.revision,
                cache_dir=str(cache / "hub"),
                local_files_only=True,
            )
            for ref in (spec.decoder, spec.encoder)
        ]
        model, _ = load_local_native_model(spec, paths[0], paths[1])
        load_sft_adapters_for_inference(model, cache / "checkpoints" / key)
        configure_memory_efficient_training(
            model, {"query_chunk_size": 2048, "gradient_checkpointing": False}
        )
        dec, _ = load_verified_tokenizer(spec.decoder, cache / "tokenizers")
        modules = [m for m in model.modules() if isinstance(m, ProbeLoRALinear)]
        old = [m.scaling for m in modules]
        try:
            for module in modules:
                module.scaling = 0.0
            cases = []
            for identifier in ("Python/0", "Python/1", "Python/39"):
                row = rows[identifier]
                prompt = dec.apply_chat_template(
                    [{"role": "user", "content": task_prompt(row, paired=False)}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prefix = tuple(dec.encode(prompt, add_special_tokens=False))
                generated = greedy_decode(
                    model,
                    (),
                    prefix,
                    max_new_tokens=384,
                    max_total_decoder_tokens=8192,
                    use_cache=True,
                )
                tokens = (
                    generated.token_ids[:-1]
                    if generated.token_ids[-1] in (1, 106, 50)
                    else generated.token_ids
                )
                text = decode_text(dec, tokens, skip_special_tokens=False)
                score = score_context_case(
                    row,
                    gold[identifier],
                    text,
                    directory=directory,
                    python_executable=sys.executable,
                    allow_execution=True,
                )
                cases.append({"instance_id": identifier, "text": text, "score": score})
                print(
                    json.dumps(
                        {
                            "stage": "decoder_no_lora",
                            "model": key,
                            "instance": identifier,
                            "status": score["status"],
                        }
                    ),
                    flush=True,
                )
            report["decoder_without_lora"][key] = cases
        finally:
            for module, scaling in zip(modules, old, strict=True):
                module.scaling = scaling
        assert all(m.scaling == scaling for m, scaling in zip(modules, old, strict=True))
        del model, modules
        gc.collect()
        torch.cuda.empty_cache()
    # Diagnostic only: repair hunk line counts, never source lines or model tokens.
    for key, cases in controls["models"].items():
        for case in cases:
            identifier = case["instance_id"]
            snapshot = snapshot_from_json(directory / gold[identifier]["snapshot_file"])
            for variant, value in case["variants"].items():
                if value["score"].get("patch_applied"):
                    continue
                item: dict[str, Any] = {"model": key, "instance": identifier, "variant": variant}
                try:
                    patch = extract_unified_diff(value["text"])
                    if set(patch_paths(patch)) != {"solution.py"}:
                        raise ValueError("Recount audit is restricted to solution.py.")
                    with tempfile.TemporaryDirectory(prefix="audit-recount-") as temporary:
                        work = Path(temporary)
                        source = next(f.content for f in snapshot.files if f.path == "solution.py")
                        (work / "solution.py").write_text(source)
                        process = subprocess.run(
                            ["git", "apply", "--recount", "--whitespace=nowarn", "-"],
                            input=patch.encode(),
                            cwd=work,
                            capture_output=True,
                            timeout=10,
                        )
                        item["applied_after_recount"] = process.returncode == 0
                        if process.returncode == 0:
                            changed = (work / "solution.py").read_text()
                            canonical = "diff --git a/solution.py b/solution.py\n" + "".join(
                                difflib.unified_diff(
                                    source.splitlines(keepends=True),
                                    changed.splitlines(keepends=True),
                                    fromfile="a/solution.py",
                                    tofile="b/solution.py",
                                )
                            )
                            item["rescored"] = score_context_case(
                                rows[identifier],
                                gold[identifier],
                                canonical,
                                directory=directory,
                                python_executable=sys.executable,
                                allow_execution=True,
                            )
                except Exception as error:
                    item["error"] = type(error).__name__ + ": " + str(error)
                report["recount_only"].append(item)
    report["lora_scaling_restored"] = True
    atomic_json(root / "audit-decoder-control.json", report)
    print("DECODER_CONTROL_COMPLETE", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--decoder-control", action="store_true")
    args = parser.parse_args()
    path = Path(
        "outputs/colab/audit-decoder-control.py"
        if args.decoder_control
        else "outputs/colab/audit-controls.py"
        if args.controls
        else "outputs/colab/audit.py"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        prepare_colab_probe.render(
            worker=audit_decoder_control
            if args.decoder_control
            else audit_controls
            if args.controls
            else audit
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"script": str(path), "training": False, "hf_writes": False}))


if __name__ == "__main__":
    main()
