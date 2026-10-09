"""Full licensed instruction pool, benchmark quarantine and repository-source audit."""

import gzip
import json
import math
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from bifocalgemma.data.curation import (
    SOURCE_REFS,
    CodeOverlapIndex,
    curate_paired_sources,
    export_pool,
    file_digest,
    read_arrow_rows,
    split_connected_records,
    token_profile,
)
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows, repo_identity

BENCHMARK_FILES = {
    "humanevalfix": (
        "bigcode/humanevalpack",
        "9a41762f73a8cb23bb5811b73d5aab164efcf378",
        "python/test-00000-of-00001.parquet",
    ),
    "canitedit": (
        "nuprl/CanItEdit",
        "3c07f38b1f9385f3214fcea94d4664c79df0d36a",
        "data/test-00000-of-00001.parquet",
    ),
    "resq": ("Qurrent/RES-Q", "c0e850b1c90d1c0eb7d3b2f3c9aa6ec09c2c04fa", "dataset.json"),
}
BENCHMARK_URLS = {
    "repoqa": "https://github.com/evalplus/repoqa_release/releases/download/2024-06-23/repoqa-2024-06-23.json.gz",
    "crosscodeeval": "https://raw.githubusercontent.com/amazon-science/cceval/40c68d2b7ca2a8eae95d901ac80e6a540a84a53d/data/crosscodeeval_data.tar.xz",
}


def fetch_benchmark_files(root, *, token=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (repo, revision, filename) in BENCHMARK_FILES.items():
        paths[key] = Path(
            hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=root / "hf",
                token=token,
            )
        )
    for key, url in BENCHMARK_URLS.items():
        path = root / url.rsplit("/", 1)[-1]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        paths[key] = path
    return paths


def build_benchmark_index(paths, study_config, *, storage_path=None):
    required = set(BENCHMARK_FILES) | set(BENCHMARK_URLS)
    if set(paths) != required:
        raise ValueError("All five benchmark artifacts are mandatory before admission.")
    for key, digest in {
        "repoqa": "c050a2ad90a7df89d9dc1f1c3b3b20683edd20a56293b35fcaae43dec115d681",
        "resq": "43bc9f0644e6fa0347ac58f9d9eb3f7b7235255ca7be4dd27c62d55ecbf342d2",
    }.items():
        if file_digest(paths[key]) != digest:
            raise ValueError("benchmark_artifact_hash_changed:" + key)
    index = CodeOverlapIndex(storage_path=storage_path)
    aliases = study_config["repo_aliases"]
    blocked = set(study_config["blocked_repos"])
    counts = {}
    for key in ("humanevalfix", "canitedit"):
        rows = list(read_source_rows(paths[key]))
        expected = 164 if key == "humanevalfix" else 105
        if len(rows) != expected:
            raise ValueError("unexpected_benchmark_count:" + key)
        counts[key] = len(rows)
        for i, row in enumerate(rows):
            codes = (
                [row["prompt"] + row["buggy_solution"], row["prompt"] + row["canonical_solution"]]
                if key == "humanevalfix"
                else [row["before"], row["after"]]
            )
            for j, code in enumerate(codes):
                index.add(f"{key}:{i}:{j}", code)
    with gzip.open(paths["repoqa"], "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    repos = data["python"]
    count = 0
    for i, repo in enumerate(repos):
        blocked.add(repo_identity(repo["repo"], aliases))
        for name, content in repo["content"].items():
            index.add(f"repoqa:{i}:{name}", content)
        for needle in repo["needles"]:
            content = repo["content"][needle["path"]].encode("utf-8")
            index.add(
                f"repoqa:{i}:needle:{count}",
                content[needle["start_byte"] : needle["end_byte"]].decode("utf-8"),
            )
            count += 1
    if count != 100:
        raise ValueError("unexpected_benchmark_count:repoqa")
    counts["repoqa"] = count
    slugs = set()
    with tarfile.open(paths["crosscodeeval"]) as archive:
        member = archive.getmember("python/line_completion_rg1_bm25.jsonl")
        stream = archive.extractfile(member)
        assert stream is not None
        count = 0
        for line in stream:
            row = json.loads(line)
            slug = row["metadata"]["repository"]
            # Upstream encoding is owner-repository-shortcommit. Match flattened
            # candidate repo identity, without inventing an ambiguous slash.
            slugs.add(__import__("re").sub(r"-[0-9a-f]{7,40}$", "", slug.lower()))
            index.add(
                f"crosscodeeval:{count}:file",
                row["prompt"] + row["groundtruth"] + row.get("right_context", ""),
            )
            for j, chunk in enumerate(row["crossfile_context"].get("list", [])):
                index.add(f"crosscodeeval:{count}:chunk:{j}", chunk["retrieved_chunk"])
            count += 1
    if count != 2665 or len(slugs) != 471:
        raise ValueError("unexpected_crosscodeeval_tasks_or_repository_count")
    counts["crosscodeeval"] = count
    raw = json.loads(Path(paths["resq"]).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("tasks", raw.get("data", raw))
    if not isinstance(raw, list) or len(raw) != 100:
        raise ValueError("unexpected_benchmark_count:resq")
    for row in raw:
        blocked.add(repo_identity(row["repo_url"], aliases))
        for j, file in enumerate(row.get("modified_files", [])):
            if isinstance(file, dict) and isinstance(file.get("content"), str):
                index.add(f"resq:{row['id']}:{j}", file["content"])
    counts["resq"] = len(raw)
    report = {
        "task_counts": counts,
        "crosscodeeval_repository_slugs": sorted(slugs),
        "blocked_repositories": sorted(blocked),
        "artifact_sha256": {k: file_digest(v) for k, v in paths.items()},
        "near_match_threshold": index.threshold,
        "scope": "exact/token-near overlap on released benchmark code, repo family quarantine; no pretraining or semantic contamination certification",
    }
    return index, blocked, slugs, report


def audit_repository_sources(files, study_config, blocked, slugs, progress=None):
    """Full table admission screen only; snapshots and patch roundtrips follow."""
    results = {}
    for key in ("swe_fixer", "swe_rebench_v2"):
        counts = Counter()
        licenses = Counter()
        repos = Counter()
        seen = set()
        for offset, row in enumerate(read_source_rows(files[key][0])):
            counts["raw"] += 1
            try:
                if str(row.get("language", "python")).lower() not in {"python", "py"}:
                    raise ValueError("non_python")
                repo = repo_identity(row["repo"], study_config["repo_aliases"])
                if repo in blocked or repo.replace("/", "-") in slugs:
                    raise ValueError("benchmark_repo_overlap")
                if not all(
                    isinstance(row.get(k), str) and row[k].strip()
                    for k in ("base_commit", "patch", "problem_statement")
                ):
                    raise ValueError("missing_fields")
                signature = stable_fingerprint((repo, row["base_commit"], row["patch"]))
                if signature in seen:
                    raise ValueError("duplicate_lineage")
                seen.add(signature)
                licenses[str(row.get("license", "unknown"))] += 1
                repos[repo] += 1
                counts["table_eligible_before_snapshot_license_gate"] += 1
            except (ValueError, KeyError) as error:
                counts["rejected:" + str(error)] += 1
            if progress and (offset + 1) % 20000 == 0:
                progress({"repository_source": key, "counts": dict(counts)})
        results[key] = {
            "counts": dict(counts),
            "source_code_licenses": dict(licenses),
            "repositories": dict(repos),
            "file_sha256": file_digest(files[key][0]),
            "status": "not_training_ready_until_snapshot_license_patch_and_context_checks",
        }
    return results


def build_full_pool(
    source_files,
    benchmark_files,
    encoder,
    decoder,
    study_config,
    output,
    *,
    encoder_limit=2048,
    decoder_limit=2048,
    baseline_limit=4608,
    progress=None,
    additional_encoder=None,
):
    def rows(key):
        for path in source_files[key]:
            yield from (
                read_arrow_rows(path) if Path(path).suffix == ".arrow" else read_source_rows(path)
            )

    index, blocked, slugs, benchmark_report = build_benchmark_index(
        benchmark_files,
        study_config,
        storage_path=Path(output).parent / "curation_indexes" / "benchmarks.sqlite",
    )

    records, admission = curate_paired_sources(
        [(key, rows(key)) for key in SOURCE_REFS],
        index,
        blocked_repos=blocked,
        blocked_repo_slugs=slugs,
        near_index_path=Path(output).parent / "curation_indexes" / "edits.sqlite",
        aliases=study_config["repo_aliases"],
        progress=progress,
    )
    eligible, profile = token_profile(
        records,
        encoder,
        decoder,
        encoder_limit=encoder_limit,
        decoder_limit=decoder_limit,
        baseline_limit=baseline_limit,
        progress=progress,
        additional_encoder=additional_encoder,
    )
    index.close()
    train, dev, split = split_connected_records(eligible)
    report = {
        "sources": SOURCE_REFS,
        "source_file_sha256": {k: [file_digest(p) for p in source_files[k]] for k in SOURCE_REFS},
        "benchmark_quarantine": benchmark_report,
        "admission": admission,
        "tokens": profile,
        "split": split,
        "training_plan": {
            "epochs": 1,
            "accumulation": 4,
            "batch_size": 1,
            "updates_full_epoch": math.ceil(len(train) / 4),
            "supervised_train_tokens": sum(r["token_lengths"]["target"] for r in train),
            "supervised_dev_tokens": sum(r["token_lengths"]["target"] for r in dev),
        },
        "model_cohort_policy": "same records for all six; both base encoders profiled; IT tokenizer behavior assets verified identical across E2B/E4B",
        "auxiliary_role": "NextCoder is synthetic instruction editing, not repository snapshot repair",
        "functional_quality_verified": False,
    }
    manifest = export_pool(train + dev, output, report)
    if progress:
        progress(
            {
                "exported_pool": str(output),
                "train": len(train),
                "dev": len(dev),
                "updates_full_epoch": report["training_plan"]["updates_full_epoch"],
            }
        )
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("outputs/full_corpus_raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_training_pool_v1"))
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, default=Path("configs/repository_study.json"))
    parser.add_argument(
        "--fetch", action="store_true", help="Explicit public dataset downloads; no model weights."
    )
    parser.add_argument("--audit-repository", action="store_true")
    args = parser.parse_args()
    if not args.fetch:
        raise ValueError("Use --fetch to authorize pinned public dataset downloads.")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a curated pool.")
    config = json.loads(args.study_config.read_text(encoding="utf-8"))
    args.cache.mkdir(parents=True, exist_ok=True)

    def progress(row):
        print(json.dumps(row), flush=True)
        (args.cache / "progress.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    prepare_full_corpus(
        args.cache,
        args.output,
        args.tokenizer_cache,
        config,
        audit_repository=args.audit_repository,
        progress=progress,
    )


def prepare_full_corpus(
    cache,
    output,
    tokenizer_cache,
    config,
    *,
    token=False,
    audit_repository=True,
    encoder_limit=2048,
    decoder_limit=2048,
    progress=None,
):
    """Library entry point for the same complete curation as the CLI."""
    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.data.tokenization import load_verified_tokenizer

    cache = Path(cache)
    output = Path(output)
    tokenizer_cache = Path(tokenizer_cache)
    if output.exists():
        raise FileExistsError("Curated output exists; validate/reuse it or choose a new directory.")
    cache.mkdir(parents=True, exist_ok=True)
    if progress is None:

        def progress(row):
            print(json.dumps(row), flush=True)

    cache.mkdir(parents=True, exist_ok=True)

    benchmarks = fetch_benchmark_files(cache / "benchmarks", token=token)
    progress({"stage": "benchmark_artifacts_cached"})
    source_files = {}
    for key, ref in SOURCE_REFS.items():
        files = {
            "editpackft": ["data/train-00000-of-00001.parquet"],
            "commitpackft_python": ["data/python/data.jsonl"],
            "nextcoder": [f"data-{i:05d}-of-00006.arrow" for i in range(6)],
        }[key]
        source_files[key] = []
        for filename in files:
            source_files[key].append(
                hf_hub_download(
                    ref["id"],
                    filename,
                    revision=ref["revision"],
                    repo_type="dataset",
                    cache_dir=cache / "hf",
                    token=token,
                )
            )
            progress({"stage": "source_cached", "source": key, "file": filename})
    assets = {}
    manifests = {}
    for spec in EXPERIMENT_SPECS:
        for ref in (spec.encoder, spec.decoder):
            if ref is not None and ref.model_id not in assets:
                assets[ref.model_id], manifests[ref.model_id] = load_verified_tokenizer(
                    ref, tokenizer_cache
                )
    # Equivalence is checked using all tokenizer behavior assets, not assumed from model size.
    for family in ("", "-it"):
        left = manifests["google/gemma-4-E2B" + family]["files"]
        right = manifests["google/gemma-4-E4B" + family]["files"]
        names = {"tokenizer.json"} | (
            {"tokenizer_config.json", "chat_template.jinja"} if family else set()
        )
        if any(left[k] != right.get(k) for k in names):
            raise ValueError(
                "Tokenizer equivalence gate failed; profile all configurations separately."
            )
    manifest = build_full_pool(
        source_files,
        benchmarks,
        assets["google/gemma-4-E2B"],
        assets["google/gemma-4-E2B-it"],
        config,
        output,
        encoder_limit=encoder_limit,
        decoder_limit=decoder_limit,
        baseline_limit=encoder_limit + decoder_limit + 512,
        progress=progress,
        additional_encoder=assets["google/gemma-4-E4B"],
    )
    if audit_repository:
        for key, repo, revision, filename in (
            (
                "swe_fixer",
                "internlm/SWE-Fixer-Train-110K",
                config["sources"]["internlm/SWE-Fixer-Train-110K"],
                "swe-fixer-train-110k.jsonl",
            ),
            (
                "swe_rebench_v2",
                "nebius/SWE-rebench-V2",
                config["sources"]["nebius/SWE-rebench-V2"],
                "data/train-00000-of-00001.parquet",
            ),
        ):
            source_files[key] = [
                hf_hub_download(
                    repo,
                    filename,
                    revision=revision,
                    repo_type="dataset",
                    cache_dir=cache / "hf",
                    token=token,
                )
            ]
        quarantine = manifest["report"]["benchmark_quarantine"]
        report = audit_repository_sources(
            source_files,
            config,
            set(quarantine["blocked_repositories"]),
            set(quarantine["crosscodeeval_repository_slugs"]),
            progress=progress,
        )
        (cache / "repository_table_audit.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    progress({"stage": "completed", "pool": str(output), "fingerprint": manifest["fingerprint"]})

    return manifest


if __name__ == "__main__":
    main()
