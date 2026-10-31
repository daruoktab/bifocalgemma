"""Durable, isolated storage for the Molab base-encoder control pilot."""

import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class Journal:
    def __init__(self, root, identity, on_saved=lambda: None):
        self.root, self.identity, self.on_saved = Path(root), identity, on_saved

    def path(self, key):
        return self.root / (digest(key.encode()) + ".json")

    def get(self, key):
        path = self.path(key)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        data = json.dumps(record["value"], sort_keys=True).encode()
        if (
            record["identity"] != self.identity
            or record["key"] != key
            or digest(data) != record["sha256"]
        ):
            raise ValueError("Evaluation journal identity/checksum mismatch.")
        return record["value"]

    def put(self, key, value):
        write_json(
            self.path(key),
            dict(
                identity=self.identity,
                key=key,
                value=value,
                sha256=digest(json.dumps(value, sort_keys=True).encode()),
            ),
        )
        self.on_saved()


class StageStore:
    def __init__(self, root, identity, on_saved=lambda: None):
        self.root, self.identity, self.on_saved = Path(root), identity, on_saved

    def journal(self):
        return Journal(self.root / "evaluation", self.identity)

    def save(self, model, optimizer, progress):
        folder = self.root / f"step-{progress['step']:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        state = dict(progress)
        state.update(
            weights={
                n: p.detach().float().cpu().clone()
                for n, p in model.named_parameters()
                if p.requires_grad or n.endswith((".lora_A", ".lora_B"))
            },
            optimizer=optimizer.state_dict(),
            python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )
        temporary = folder / "state.tmp"
        torch.save(state, temporary)
        temporary.replace(folder / "state.pt")
        write_json(
            folder / "complete.json",
            {"identity": self.identity, "sha256": digest((folder / "state.pt").read_bytes())},
        )
        self.on_saved()

    def load(self, model, optimizer):
        for folder in sorted(self.root.glob("step-*"), reverse=True):
            receipt = folder / "complete.json"
            if not receipt.is_file() or not (folder / "state.pt").is_file():
                continue
            meta = json.loads(receipt.read_text())
            if meta["identity"] != self.identity:
                raise ValueError("Checkpoint belongs to a different experiment.")
            if digest((folder / "state.pt").read_bytes()) != meta["sha256"]:
                continue
            state = torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
            with torch.no_grad():
                for name, value in state["weights"].items():
                    model.get_parameter(name).copy_(value)
            optimizer.load_state_dict(state["optimizer"])
            random.setstate(state["python_rng"])
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            return state
        return None


class PilotStorage:
    repo = "daruokta/bifocalgemma-pilot-results"
    prefix = "e2b-base"

    def __init__(self, root, token, source_identity):
        from huggingface_hub import HfApi

        self.root, self.token = Path(root), token
        self.api = HfApi(token=token)
        self.identity = source_identity
        self.model: Any = None
        self.last_sync = 0.0
        self.remote_revision = None

    def restore(self):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        # Missing access/auth/network is an error, never interpreted as a fresh run.
        self.api.create_repo(repo_id=self.repo, repo_type="dataset", private=True, exist_ok=True)
        revision = self.api.repo_info(self.repo, repo_type="dataset").sha
        try:
            receipt = hf_hub_download(
                self.repo,
                self.prefix + "/remote.json",
                repo_type="dataset",
                revision=revision,
                token=self.token,
            )
        except EntryNotFoundError:
            return
        meta = json.loads(Path(receipt).read_text())
        if meta["source_identity"] != self.identity:
            raise ValueError("Remote pilot uses another source/configuration; do not overwrite it.")
        for name, checksum in meta["files"].items():
            target = (self.root / name).resolve()
            if not target.is_relative_to(self.root.resolve()):
                raise ValueError("Remote path escapes pilot directory.")
            if target.is_file():
                # Preserve newer local work; its own journal/checkpoint is validated on load.
                continue
            source = Path(
                hf_hub_download(
                    self.repo,
                    self.prefix + "/" + name,
                    repo_type="dataset",
                    revision=revision,
                    token=self.token,
                )
            )
            if digest(source.read_bytes()) != checksum:
                raise ValueError("Remote artifact checksum mismatch.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        self.remote_revision = revision

    @staticmethod
    def cohort(items):
        from dataclasses import asdict

        checksum = hashlib.sha256()
        for item in items:
            checksum.update(json.dumps(asdict(item), sort_keys=True).encode())
            checksum.update(b"\n")
        return {"examples": len(items), "input_target_sha256": checksum.hexdigest()}

    def bind(self, config):
        from importlib.metadata import version

        identity = digest(json.dumps(config, sort_keys=True).encode())
        path = self.root / "manifest.json"
        manifest = {
            "source_identity": self.identity,
            "experiment_identity": identity,
            "config": config,
            "comparison": "encoder source control; A100 versus Molab hardware",
            "training_steps": {"frozen": 100, "joint": 100},
            "dependencies": {
                name: version(name)
                for name in ("torch", "transformers", "marimo", "huggingface-hub")
            },
            "architecture": {
                "attention": "separate_softmax_full_cross",
                "alignment": "linear_rmsnorm_fp32",
                "norm_epsilon": 1e-6,
                "gate_parameter_init": 0.015,
                "gate_tanh_multiplier": 10.0,
                "query_gate_init": 1.0,
            },
        }
        if path.exists() and (
            json.loads(path.read_text())["experiment_identity"] != identity
            or json.loads(path.read_text())["source_identity"] != self.identity
        ):
            raise ValueError("Pilot cohort/configuration changed; refusing resume.")
        write_json(path, manifest)
        self.identity = identity

    def stage(self, name):
        return StageStore(self.root / "checkpoints" / name, self.identity, self.sync)

    def evaluation(self, name):
        self.model.pilot_journal = Journal(
            self.root / "evaluation" / name, self.identity, self.periodic_sync
        )

    def periodic_sync(self):
        if time.monotonic() - self.last_sync > 300:
            self.sync()

    def sync(self, final=False):
        from huggingface_hub import hf_hub_download

        self.last_sync = time.monotonic()

        paths = [
            p
            for p in self.root.rglob("*")
            if p.is_file()
            and "cache" not in p.relative_to(self.root).parts
            and p.suffix not in {".tmp", ".lock"}
            and p.name not in {"remote.json", "upload_status.json"}
        ]
        manifest = self.root / "manifest.json"
        source_id = (
            json.loads(manifest.read_text())["source_identity"]
            if manifest.exists()
            else self.identity
        )
        receipt = {
            "source_identity": source_id,
            "files": {p.relative_to(self.root).as_posix(): digest(p.read_bytes()) for p in paths},
        }
        write_json(self.root / "remote.json", receipt)
        for attempt in range(3):
            try:
                info = self.api.upload_folder(
                    repo_id=self.repo,
                    repo_type="dataset",
                    folder_path=self.root,
                    path_in_repo=self.prefix,
                    allow_patterns=[*receipt["files"], "remote.json"],
                )
                verified = hf_hub_download(
                    self.repo,
                    self.prefix + "/remote.json",
                    repo_type="dataset",
                    revision=info.oid,
                    token=self.token,
                    force_download=True,
                )
                if json.loads(Path(verified).read_text()) != receipt:
                    raise ValueError("Remote receipt verification failed.")
                self.remote_revision = info.oid
                self.last_sync = time.monotonic()
                write_json(
                    self.root / "upload_status.json",
                    {"status": "VERIFIED", "revision": info.oid, "last_seen": time.time()},
                )
                return
            except Exception as error:
                write_json(
                    self.root / "upload_status.json",
                    {
                        "status": "PENDING_UPLOAD",
                        "error_type": type(error).__name__,
                        "last_verified_revision": self.remote_revision,
                        "last_seen": time.time(),
                    },
                )
                if attempt < 2:
                    time.sleep(2**attempt)
        if final:
            raise RuntimeError(
                "Pilot saved locally; final HF upload is still pending. Run again to retry."
            )


def _run(runtime_root, output_root, token, source_identity, monitor=None):
    import os

    from prepare_colab_pilot import pilot_e2b_e2b

    os.environ["HF_TOKEN"] = token
    storage = PilotStorage(output_root, token, source_identity)
    storage.restore()
    manifest_path = storage.root / "manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text())["source_identity"] != source_identity
    ):
        raise ValueError("Local pilot source changed; refusing to reuse results.")
    report_path = storage.root / "pilot_e2b_e2b.json"
    if report_path.exists() and manifest_path.exists():
        finished = json.loads(report_path.read_text())
        if finished.get("status") == "COMPLETED":
            storage.sync(final=True)
            return finished
    from bifocalgemma.models import inference_cache as cache_module
    from bifocalgemma.training import offline_sft, v2_diagnostics

    original_loss = offline_sft.evaluate_sft_loss
    original_diagnostic_loss = v2_diagnostics.evaluate_sft_loss
    original_cached = cache_module.CachedAttention

    def event_callback(event):
        storage.periodic_sync()
        if monitor is not None:
            status_path = storage.root / "upload_status.json"
            upload = json.loads(status_path.read_text()) if status_path.exists() else {}
            display = dict(event)
            if event.get("sub_stage") == "pilot_step":
                display["stage"] = "optimizer_step_complete"
                display["step"] = event["step"] + (100 if not event["freeze_decoder"] else 0)
                display["max_steps"] = 200
            monitor(
                {
                    **display,
                    "remote_revision": storage.remote_revision,
                    "upload_status": upload.get("status", "NOT_UPLOADED"),
                    "upload_pending": upload.get("status") != "VERIFIED",
                    "elapsed_seconds": event.get("active_training_seconds", 0),
                }
            )

    offline_sft.evaluate_sft_loss = journaled_loss(original_loss)
    v2_diagnostics.evaluate_sft_loss = offline_sft.evaluate_sft_loss
    try:
        result = pilot_e2b_e2b(
            runtime_root,
            steps_per_arm=200,
            arms=("full_cross_bf16_frozen_then_joint",),
            humaneval_limit=16,
            output_root=Path(output_root),
            encoder_source="base",
            durable=storage,
            event_callback=event_callback,
        )
    except BaseException:
        storage.sync()
        raise
    finally:
        offline_sft.evaluate_sft_loss = original_loss
        v2_diagnostics.evaluate_sft_loss = original_diagnostic_loss
        cache_module.CachedAttention = original_cached
    storage.sync(final=True)
    return result


def journaled_loss(original):
    """Cache single-example losses; preserve token weighting and diagnostic forwards."""
    from dataclasses import asdict

    def evaluate(model, examples, *, batch_size, device, event_callback=None, step=0):
        journal = getattr(model, "pilot_journal", None)
        collecting = any(
            getattr(layer.self_attn, "collect_encoder_mass", False)
            for layer in model.decoder.layers
        )
        if journal is None or collecting:
            return original(
                model,
                examples,
                batch_size=batch_size,
                device=device,
                event_callback=event_callback,
                step=step,
            )
        rows = []
        for index, example in enumerate(examples):
            key = "loss-" + digest(json.dumps(asdict(example), sort_keys=True).encode())
            value = journal.get(key)
            if value is None:
                value = original(model, [example], batch_size=1, device=device)
                journal.put(key, value)
            rows.append(value)
            if event_callback:
                event_callback(
                    {
                        "stage": "validation_progress",
                        "step": step,
                        "eval_examples_done": index + 1,
                        "eval_examples_total": len(examples),
                    }
                )

        def aggregate(items):
            tokens = sum(x["supervised_tokens"] for x in items)
            return {
                "loss": sum(x["loss"] * x["supervised_tokens"] for x in items) / tokens,
                "supervised_tokens": tokens,
                "examples": sum(x["examples"] for x in items),
            }

        result = aggregate(rows)
        for group in ("by_source", "by_task_type"):
            names = sorted({name for row in rows for name in row[group]})
            result[group] = {
                name: aggregate([row[group][name] for row in rows if name in row[group]])
                for name in names
            }
        return result

    return evaluate


def run(runtime_root, output_root, token, source_identity, monitor=None):
    from filelock import FileLock

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / "pilot.lock", timeout=0):
        return _run(runtime_root, output_root, token, source_identity, monitor)
