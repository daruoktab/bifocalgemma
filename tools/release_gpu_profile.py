"""Review all six Colab boundary receipts before publishing the shared HF profile."""

import argparse
import json
import zipfile
from importlib.metadata import version
from pathlib import Path

from bifocalgemma.models.memory_efficient import DEFAULT_MEMORY_POLICY
from bifocalgemma.training.shared_execution import (
    ASSETS_REPO,
    PROFILE_FILE,
    publish_profile,
    read_remote_profile,
    validate_profile,
)
from bifocalgemma.training.stages import atomic_json
from bifocalgemma.training.v2 import (
    COLLECTION,
    DATASET_REPO,
    DATASET_REVISION,
    MODELS,
    source_fingerprint,
)
from bifocalgemma.training.v2_preflight import EXECUTION_VERSION, verify_report


def check_evidence(profile, boundaries, report, contract):
    validate_profile(profile, contract)
    if set(boundaries) != set(MODELS) or set(profile.get("declared_limit_validation", {})) != set(
        MODELS
    ):
        raise ValueError("All six declared-limit receipts are required.")
    if (
        report.get("status") != "COMPLETED"
        or report.get("diagnostic_only") is not True
        or report.get("epoch_training_started") is not False
        or report.get("profile_published") is not False
        or report.get("failure") is not None
        or report.get("weights_restored_exactly") is not True
        or not report.get("trainable_sha256_before")
        or report.get("trainable_sha256_before") != report.get("trainable_sha256_after")
    ):
        raise ValueError("Diagnostic completion and exact weight restoration are required.")
    policy = {**DEFAULT_MEMORY_POLICY, **profile["memory_policy"]}
    for key, row in boundaries.items():
        if (
            row.get("status") != "PASS"
            or row.get("weights_restored_exactly") is not True
            or not row.get("trainable_sha256_before")
            or row.get("trainable_sha256_before") != row.get("trainable_sha256_after")
        ):
            raise ValueError("Incomplete boundary or weight restoration: " + key)
        receipt = row["validation"]
        verify_report(receipt)
        if receipt != profile["declared_limit_validation"][key]:
            raise ValueError("Profile differs from diagnostic receipt: " + key)
        expected_shape = (0, 41472) if key.endswith("-it") else (32768, 8192)
        if (
            (receipt.get("encoder_tokens"), receipt.get("decoder_tokens")) != expected_shape
            or receipt.get("version") != EXECUTION_VERSION
            or receipt.get("gpu") != profile["gpu"]["name"]
            or receipt.get("status") != "VALIDATED_ON_THIS_GPU"
            or receipt.get("batch_size") != profile["batch_size"]
            or receipt.get("accumulation") != profile["accumulation"]
            or receipt.get("memory_policy") != policy
            or receipt.get("optimizer_updates_retained") != 0
        ):
            raise ValueError("Boundary dimensions or shared configuration differ: " + key)
        for phase in ("training", "evaluation"):
            item = receipt.get("phases", {}).get(phase, {})
            if item.get("status") != "COMPLETED" or item.get("estimated_peak_headroom_gib", 0) < 4:
                raise ValueError("Missing or insufficient-memory boundary phase: " + key)
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError("Diagnostic archive CRC mismatch.")
        profile, boundaries, report = (
            json.loads(archive.read(name))
            for name in ("profile.json", "boundaries.json", "report.json")
        )
    contract = {
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
    check_evidence(profile, boundaries, report, contract)
    result = {
        "status": "VALIDATED_LOCALLY",
        "models": list(boundaries),
        "evidence_sha256": profile["evidence_sha256"],
    }
    if args.publish:
        from huggingface_hub import HfApi, get_token

        token = get_token()
        if not token:
            raise ValueError("HF authentication is required for publication.")
        published = publish_profile(profile, token=token)
        if published != profile:
            raise ValueError("Another valid shared profile already exists; no overwrite performed.")
        revision = HfApi(token=token).repo_info(ASSETS_REPO, repo_type="dataset").sha
        if read_remote_profile(contract, token=token, revision=revision) != profile:
            raise ValueError("Pinned remote verification failed.")
        result.update(
            status="PUBLISHED_AND_VERIFIED",
            repo_id=ASSETS_REPO,
            filename=PROFILE_FILE,
            revision=revision,
        )
        atomic_json(args.archive.parent / "published.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
