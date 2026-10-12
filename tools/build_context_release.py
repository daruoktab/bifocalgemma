"""Package the completed 32K/8K pool, pinned notices and frozen benchmark cohorts."""
# ruff: noqa: E402 -- standalone static-data bootstrap

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ["USE_TORCH"] = "0"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from audit_context_release import audit_pool, audit_views

from bifocalgemma.data.curation import file_digest
from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.releases import build_release
from bifocalgemma.data.repository_corpus import ArchiveSnapshotStore
from bifocalgemma.training.stages import atomic_json


def source_notices(pool, directory):
    """Preserve the exact license bytes referenced by every admitted context file."""
    rows, _ = load_curated_pool(pool)
    grouped = {}
    for row in rows["train"] + rows["dev"]:
        if row.get("record_type") != "context_sft_v1":
            continue
        group = grouped.setdefault((row["repos"], row["commit"]), {})
        for evidence in row["context_source_code_license_evidence"]:
            for item in evidence["licenses"]:
                group[(item["path"], item["sha256"])] = item
    directory.mkdir(parents=True, exist_ok=True)
    store = ArchiveSnapshotStore(ROOT / "outputs/context_sources/repository_corpus/archives")
    notices, inventory = {}, []
    for index, ((repo, commit), entries) in enumerate(grouped.items(), 1):
        snapshot, _ = store.snapshot(SimpleNamespace(repo=repo, base_commit=commit))
        files = {f.path: f.content for f in snapshot.files}
        for (path, sha), item in entries.items():
            name = "source_license_" + sha
            destination = directory / (name + ".txt")
            content = files[path].encode("utf-8")
            import hashlib

            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError("source_notice_hash_changed")
            destination.write_bytes(content)
            notices[name] = {"name": name, "file": destination.relative_to(ROOT).as_posix()}
            inventory.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "path": path,
                    "sha256": sha,
                    "license": item["license"],
                    "notice": name,
                }
            )
        if index % 100 == 0:
            print(
                json.dumps({"stage": "source_notice_audit", "done": index, "total": len(grouped)}),
                flush=True,
            )
    evidence_path = directory / "source_notice_inventory.json"
    atomic_json(evidence_path, inventory)
    return list(notices.values()), evidence_path


def main():
    pool = ROOT / "outputs/mixed_context_enc32k_dec8k_v3"
    release_dir = ROOT / "outputs/dataset_release_context32k_v3"
    if release_dir.exists():
        raise FileExistsError(
            "Choose a new immutable release output; do not overwrite a completed release."
        )
    result = {
        "date": "2026-10-13",
        "pool": audit_pool(pool, ROOT / "outputs/context_sources/repository_corpus/archives"),
    }
    pool_audit = ROOT / "outputs/context_pool_audit_v3.json"
    result["fingerprint"] = stable_fingerprint(result)
    atomic_json(pool_audit, result)
    notices, source_notice_inventory = source_notices(
        pool, ROOT / "outputs/context_source_licenses"
    )
    original = json.loads(
        (ROOT / "outputs/dataset_release_spec_v2_ready.json").read_text(encoding="utf-8")
    )
    collections = [
        {
            "name": "mixed_context_enc32k_dec8k_v3",
            "kind": "curated_pool",
            "directory": pool.relative_to(ROOT).as_posix(),
            "description": "Editing foundation plus licensed pre-fix repository repair/localization/dependency completion. Common six-model cohort at encoder32768, paired shifted decoder8192, baseline41472. Static evidence only; functional training quality unverified.",
        }
    ]
    collections.extend(c for c in original["collections"] if c["kind"] != "curated_pool")
    descriptions = {
        "sweqa_oracle_gemma32k_dec8k_v1": "SWE-QA oracle context, strict-letter accuracy; same question IDs as noisy condition. Upstream-provided relevant context is an oracle condition, not a retrieval result.",
        "sweqa_noisy_oracle_gemma32k_dec8k_v1": "SWE-QA noisy-oracle context, strict-letter accuracy; paired questions with the oracle condition, not independent task count.",
        "crosscodelong_chunk_gemma32k_dec8k_v1": "CrossCodeLongEval chunk completion with released rg1 cross-file/FIM context; common no-crop cohort; adapted whitespace EM, sequence similarity and regex identifier-set F1. Gold-query/oracle retrieval excluded.",
        "crosscodelong_function_gemma32k_dec8k_v1": "CrossCodeLongEval function completion with released rg1 cross-file/FIM context; common no-crop cohort; adapted completion metrics, no functional repair claim.",
        "longcodeqa_gemma32k_dec8k_v1": "Gemma-tokenizer-eligible subset of the released LongCodeQA32K prompts; no context cropping, strict-letter accuracy. Source32K bucket is not an exact Gemma token count.",
    }
    for name, description in descriptions.items():
        collections.append(
            {
                "name": name,
                "kind": "prepared_code_eval",
                "directory": "outputs/context_benchmarks_v1/" + name,
                "description": description,
            }
        )
    pinned_notices = json.loads(
        (ROOT / "outputs/context_notices/notice_sources.json").read_text(encoding="utf-8")
    )
    for item in pinned_notices:
        if file_digest(item["file"]) != item["sha256"]:
            raise ValueError("pinned_notice_integrity")
    spec = {
        **original,
        "name": "bifocalgemma-private-context-data-20261013-v3",
        "date": "2026-10-13",
        "supersedes_revision": "d34a19b3badc0120755495c68227cba038819cd7",
        "collections": collections,
        "notices": original["notices"]
        + [{"name": n["name"], "file": n["file"]} for n in pinned_notices]
        + notices,
        "audits": [
            {"name": "mixed_pool_v3", "file": pool_audit.relative_to(ROOT).as_posix()},
            {
                "name": "context_tokens_32k",
                "file": "outputs/context_token_audit_32k_v1.json",
            },
            {
                "name": "legacy_benchmark_tokens_32k",
                "file": "outputs/context_legacy_benchmark_token_audit_32k_v1.json",
            },
            {"name": "patch_locality_parity", "file": "outputs/context_patch_locality_audit.json"},
            {
                "name": "humanevalfix_gold_native",
                "file": "outputs/humanevalfix_gold_patch_locality_v3.json",
            },
            {
                "name": "source_notice_inventory",
                "file": source_notice_inventory.relative_to(ROOT).as_posix(),
            },
        ],
    }
    spec["benchmark_catalog"] = [
        c for c in original["benchmark_catalog"] if c.get("role") != "train"
    ]
    for name in descriptions:
        preparation = json.loads(
            (ROOT / "outputs/context_benchmarks_v1" / name / "preparation_report.json").read_text(
                encoding="utf-8"
            )
        )
        spec["benchmark_catalog"].append(
            {
                "name": name,
                "collection": name,
                "role": "eval",
                "status": "matched_six_model_cohort_adapted_scorer_ready",
                "count": preparation["count"],
                "protocol": preparation["protocol"],
                "official_protocol": False,
                "model_score": None,
            }
        )
    spec_path = ROOT / "configs/dataset_release.context32k.json"
    atomic_json(spec_path, spec)
    release = build_release(spec_path, release_dir)
    # A full viewer check is required before the release can be published.
    result["release"] = audit_views(release_dir)
    result["fingerprint"] = stable_fingerprint(
        {k: v for k, v in result.items() if k != "fingerprint"}
    )
    atomic_json(ROOT / "outputs/context_release_audit_v3.json", result)
    print(
        json.dumps(
            {"status": "PASS", "release": str(release_dir), "fingerprint": release["fingerprint"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
