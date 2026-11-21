"""Build pinned, audited context benchmarks and a repository SFT pool locally."""
# ruff: noqa: E402 -- standalone tool bootstraps the local src package before imports

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ["USE_TORCH"] = "0"  # Static data/tokenizer preparation needs no tensor runtime.
sys.path.insert(0, str(ROOT / "src"))
ENCODER_LIMIT = 32768
DECODER_LIMIT = 8192
BASELINE_LIMIT = ENCODER_LIMIT + DECODER_LIMIT + 512

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.data.context_benchmarks import (
    matched_eval_cohort,
    prepare_crosscodelong,
    prepare_longcodeqa,
    prepare_sweqa,
)
from bifocalgemma.data.context_tasks import (
    BLOCKED_V4_DATASETS,
    V4_DATASET_REGISTRY,
    is_permissive_license,
    normalize_context_task,
    repair_preimage_coverage,
    tokenize_context_task,
)
from bifocalgemma.data.curation import code_signature, export_pool, split_connected_records
from bifocalgemma.data.curriculum import V4_DEFAULT_MIX, build_v4_curriculum
from bifocalgemma.data.grounded_tasks import generate_grounded_pool_tasks
from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.patches import patch_paths
from bifocalgemma.data.repository import (
    SOURCE_REVISIONS,
    read_source_rows,
    repo_identity,
    repository_task_from_row,
)
from bifocalgemma.data.repository_corpus import (
    ArchiveSnapshotStore,
    build_repository_corpus,
    context_license_evidence,
    v4_repository_task_from_row,
)
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.training.stages import atomic_json


def log(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def benchmarks(output, cache, tokenizer_cache):
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "sweqa_oracle_v1/preparation_report.json").exists():
        log({"stage": "sweqa_preparation"})
        prepare_sweqa(cache, output)
    if not (output / "crosscodelong_function_source_v1/preparation_report.json").exists():
        log({"stage": "crosscodelong_preparation"})
        prepare_crosscodelong(cache, output)
    if not (output / "longcodeqa_32k_source_v1/preparation_report.json").exists():
        log({"stage": "longcodeqa_preparation"})
        prepare_longcodeqa(cache, output)
    sources = [
        ("sweqa_oracle_v1", "sweqa_oracle_gemma32k_dec8k_v1", ENCODER_LIMIT),
        ("sweqa_noisy_oracle_v1", "sweqa_noisy_oracle_gemma32k_dec8k_v1", ENCODER_LIMIT),
        ("crosscodelong_chunk_source_v1", "crosscodelong_chunk_gemma32k_dec8k_v1", ENCODER_LIMIT),
        (
            "crosscodelong_function_source_v1",
            "crosscodelong_function_gemma32k_dec8k_v1",
            ENCODER_LIMIT,
        ),
        ("longcodeqa_32k_source_v1", "longcodeqa_gemma32k_dec8k_v1", ENCODER_LIMIT),
    ]
    for source, name, limit in sources:
        if not (output / name / "preparation_report.json").exists():
            log({"stage": "matched_cohort", "collection": name})
            r = matched_eval_cohort(
                output / source, output / name, tokenizer_cache, encoder_limit=limit
            )
            log({"collection": name, "count": r["count"], "rejected": len(r["rejections"])})
    repos = set()
    slugs = set()
    for directory in output.iterdir():
        report = directory / "preparation_report.json"
        if not report.is_file():
            continue
        r = json.loads(report.read_text(encoding="utf-8"))
        for name in r["repositories"]:
            if "/" in name:
                repos.add(name)
            else:
                # Released filenames use owner-repo-commit slugs; exact slug prefixes are retained.
                slugs.add(name)
        repos.update(r.get("protocol", {}).get("all_upstream_repositories_quarantined", []))
    atomic_json(
        output / "benchmark_quarantine.json",
        {"repositories": sorted(repos), "slugs": sorted(slugs)},
    )


def prepare_core(
    output,
    cache,
    foundation,
    tokenizers,
    benchmarks_root,
    target_count,
    per_repo_limit,
    *,
    prefetch=True,
    source_key="swe_rebench_v2",
    disjoint_context=False,
):
    from huggingface_hub import hf_hub_download

    from bifocalgemma.data.full_pool import build_benchmark_index, fetch_benchmark_files

    if source_key not in {"swe_rebench_v2", "swe_fixer"}:
        raise ValueError(f"Unsupported prepare_core source_key: {source_key!r}")
    study = json.loads((ROOT / "configs/repository_study.json").read_text(encoding="utf-8"))
    quarantine = json.loads(
        (benchmarks_root / "benchmark_quarantine.json").read_text(encoding="utf-8")
    )
    study["blocked_repos"] = sorted(set(study["blocked_repos"]) | set(quarantine["repositories"]))
    _, manifest = load_curated_pool(foundation)
    blocked = set(study["blocked_repos"]) | set(
        manifest["report"]["benchmark_quarantine"]["blocked_repositories"]
    )
    existing_slugs = set(
        manifest["report"]["benchmark_quarantine"]["crosscodeeval_repository_slugs"]
    )
    source, revision, _ = SOURCE_REVISIONS[source_key]
    filename = (
        "swe-fixer-train-110k.jsonl"
        if source_key == "swe_fixer"
        else "data/train-00000-of-00001.parquet"
    )
    source_path = hf_hub_download(
        source,
        filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache / "hf",
    )
    candidates = []
    per_repo = Counter()
    counts = Counter()
    for offset, row in enumerate(read_source_rows(source_path)):
        counts["source_rows"] += 1
        try:
            task = repository_task_from_row(source_key, row)
            if task.language not in {"python", "py"}:
                raise ValueError("non_python")
            if not source_python_paths(task.target_patch):
                raise ValueError("no_non_test_python_change")
            if task.repo in blocked or task.repo.replace("/", "-") in existing_slugs:
                raise ValueError("benchmark_repository")
            if any(
                slug.startswith(task.repo.replace("/", "-") + "-") for slug in quarantine["slugs"]
            ):
                raise ValueError("new_crosscode_repository")
            if per_repo_limit and per_repo[task.repo] >= per_repo_limit:
                raise ValueError("balanced_repository_quota")
            candidates.append({**row, "original_upstream_row": offset})
            per_repo[task.repo] += 1
        except (ValueError, KeyError) as error:
            counts[str(error).split(":")[0]] += 1
    candidate_file = cache / "repository_candidates.jsonl"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    with candidate_file.open("w", encoding="utf-8", newline="\n") as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(
        cache / "candidate_selection.json",
        {
            "counts": dict(counts),
            "candidates": len(candidates),
            "per_repository_limit": per_repo_limit,
            "target_admitted_count": target_count or "all_eligible_candidates",
            "policy": "pinned_upstream_order_with_repository_quota_before_model_results",
        },
    )
    log({"stage": "snapshot_prefetch", "candidates": len(candidates)})
    archives = cache / "repository_corpus/archives"
    store = ArchiveSnapshotStore(archives, allow_network=True)

    def fetch(row):
        try:
            store.fetch_archive(repository_task_from_row(source_key, row))
            return "cached"
        except Exception as exc:
            return type(exc).__name__

    # Bounded concurrency; no model, repository code execution, or package installation.
    totals = Counter()
    unique_snapshots = {}
    for row in candidates:
        unique_snapshots.setdefault((row["repo"], row["base_commit"]), row)
    if prefetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for index, status in enumerate(executor.map(fetch, unique_snapshots.values()), 1):
                totals[status] += 1
                if index % 25 == 0 or index == len(unique_snapshots):
                    log(
                        {
                            "stage": "snapshot_prefetch",
                            "done": index,
                            "total": len(unique_snapshots),
                            "outcomes": dict(totals),
                        }
                    )
    else:
        log({"stage": "reuse_cached_snapshots", "missing_archives_are_rejections": True})
    log({"stage": "repository_admission"})
    benchmark_index_path = cache / "benchmark_index_current_normalization.sqlite"
    benchmark_evidence_path = cache / "benchmark_index_evidence.json"
    from bifocalgemma.data.curation import file_digest

    if not benchmark_index_path.exists() and not benchmark_evidence_path.exists():
        log({"stage": "rebuild_benchmark_index"})
        from uuid import uuid4

        temporary_index = cache / (".benchmark-index-" + uuid4().hex + ".sqlite")
        index, _, _, evidence = build_benchmark_index(
            fetch_benchmark_files(ROOT / "outputs/full_corpus_raw/benchmarks"),
            study,
            storage_path=temporary_index,
        )
        index.close()
        temporary_index.replace(benchmark_index_path)
        evidence["index_sha256"] = file_digest(benchmark_index_path)
        atomic_json(benchmark_evidence_path, evidence)
    elif not benchmark_evidence_path.exists() or not benchmark_index_path.exists():
        raise ValueError("Benchmark index preparation is incomplete; inspect before reuse.")
    else:
        evidence = json.loads(benchmark_evidence_path.read_text(encoding="utf-8"))
        if evidence.get("index_sha256") != file_digest(benchmark_index_path):
            raise ValueError("Completed benchmark index hash changed.")
    return build_repository_corpus(
        source_key,
        candidate_file,
        foundation,
        tokenizers,
        study,
        output,
        cache / "repository_corpus",
        target_count=target_count or len(candidates) + 1,
        per_repo_limit=per_repo_limit or len(candidates) + 1,
        context_limit=ENCODER_LIMIT,
        decoder_limit=DECODER_LIMIT,
        disk_budget_gib=80,
        allow_network=False,
        progress=log,
        benchmark_index_path=benchmark_index_path,
        disjoint_context=disjoint_context,
    )


def completions_from_snapshot(
    row,
    snapshot,
    *,
    max_per_file: int | None = None,
    max_total: int | None = None,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Mask all non-overlapping unchanged functions calling symbols imported from sibling files."""
    files = {f.path: f.content for f in snapshot.files}
    changed = set(patch_paths(row["target_patch"]))
    results: list[tuple[str, str, str, dict[str, Any]]] = []
    for path in sorted(files):
        if (
            not path.endswith(".py")
            or path in changed
            or "/test" in path
            or path.startswith("test")
        ):
            continue
        content = files[path]
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        imports = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            suffix = node.module.replace(".", "/")
            if node.level:
                parent = Path(path).parent
                for _ in range(node.level - 1):
                    parent = parent.parent
                module = (parent / suffix).as_posix()
                possible = [p for p in files if p in {module + ".py", module + "/__init__.py"}]
            else:
                possible = [
                    p
                    for p in files
                    if p.endswith(suffix + ".py") or p.endswith(suffix + "/__init__.py")
                ]
            if len(possible) == 1 and possible[0] != path:
                for alias in node.names:
                    if alias.name != "*":
                        imports[alias.asname or alias.name] = possible[0]
        lines = content.splitlines(keepends=True)
        fn_nodes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.body and n.end_lineno
        ]
        fn_nodes.sort(key=lambda n: (n.lineno, n.col_offset))
        occupied: list[tuple[int, int]] = []
        file_count = 0
        for node in fn_nodes:
            a, b = node.body[0].lineno - 1, node.end_lineno
            assert b is not None
            if any(not (b <= prev_a or a >= prev_b) for prev_a, prev_b in occupied):
                continue
            calls = {
                n.id
                for statement in node.body
                for n in ast.walk(statement)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            helpers = sorted({imports[n] for n in calls if n in imports})
            helpers = [p for p in helpers if p not in changed]
            if not helpers:
                continue
            target = "".join(lines[a:b])
            if len(target) < 64 or target.count("\n") < 2:
                continue
            masked = (
                "".join(lines[:a])
                + " " * node.body[0].col_offset
                + "# <MISSING_CODE>\n"
                + "".join(lines[b:])
            )
            visible = {path: masked, **{h: files[h] for h in helpers}}
            if any(target in text for text in visible.values()):
                continue
            instruction = (
                f"Complete the missing body of {node.name} in {path} using the imported helpers."
            )
            context = (
                "### Instruction\n"
                + instruction
                + "\n\n"
                + "\n\n".join(f"### File: {p}\n{v}" for p, v in sorted(visible.items()))
            )
            selection = {
                "gold_used": False,
                "policy": "static_import_dependency_masking_training_v1",
                "windows": [{"path": p} for p in sorted(visible)],
            }
            occupied.append((a, b))
            results.append((instruction, context, target, selection))
            file_count += 1
            if max_total is not None and len(results) >= max_total:
                return results
            if max_per_file is not None and file_count >= max_per_file:
                break
    return results


def completion_from_snapshot(row, snapshot):
    """Mask an unchanged function that calls a symbol imported from another source file."""
    extracted = completions_from_snapshot(row, snapshot, max_total=1)
    return extracted[0] if extracted else None


def source_python_paths(patch):
    from pathlib import PurePosixPath

    return [
        p
        for p in patch_paths(patch)
        if p.endswith(".py")
        and not ({"test", "tests"} & set(PurePosixPath(p).parts))
        and not PurePosixPath(p).name.startswith("test_")
        and not PurePosixPath(p).name.endswith("_test.py")
    ]


def context_pool(
    core,
    output,
    cache,
    tokenizers,
    *,
    disjoint_context: bool = False,
    max_completions_per_file: int | None = None,
    max_completions_per_snapshot: int | None = None,
):
    rows, manifest = load_curated_pool(core)
    candidate_rows = {
        raw["instance_id"]: {"candidate_row": offset, "upstream_row": raw["original_upstream_row"]}
        for offset, raw in enumerate(read_source_rows(cache / "repository_candidates.jsonl"))
    }
    store = ArchiveSnapshotStore(cache / "repository_corpus/archives", allow_network=False)
    references = {
        ref.model_id: ref for s in EXPERIMENT_SPECS for ref in (s.encoder, s.decoder) if ref
    }
    assets = {name: load_verified_tokenizer(ref, tokenizers)[0] for name, ref in references.items()}
    encoders = {
        s.encoder.model_id: assets[s.encoder.model_id] for s in EXPERIMENT_SPECS if s.encoder
    }
    decoders = {s.decoder.model_id: assets[s.decoder.model_id] for s in EXPERIMENT_SPECS}
    effective_signatures = {
        name: stable_fingerprint(
            {
                "backend_sha256": hashlib.sha256(
                    asset.backend_tokenizer.to_str().encode()
                ).hexdigest(),
                "template": asset.chat_template,
                "special_tokens": asset.special_tokens_map,
            }
        )
        for name, asset in assets.items()
    }
    accepted, rejected, seen = [], Counter(), set()
    all_rows = rows["train"] + rows["dev"]
    for offset, row in enumerate(all_rows, 1):
        if row["instance_id"] not in candidate_rows:
            raise ValueError("Admitted task is absent from the pinned candidate inventory.")
        row: dict[str, Any] = {**row, **candidate_rows[row["instance_id"]]}
        if offset == 1 or offset % 25 == 0:
            log(
                {
                    "stage": "context_task_admission",
                    "done": offset - 1,
                    "total": len(all_rows),
                    "accepted": len(accepted),
                    "rejected": dict(rejected),
                }
            )
        if not source_python_paths(row["target_patch"]):
            rejected["no_non_test_python_change"] += 1
            continue
        if row.get("target_syntax_warnings"):
            rejected["target_syntax_warning"] += 1
            continue
        variants: list[tuple[str, str, str, str, dict[str, Any], Any]] = []
        if repair_preimage_coverage(row["target_patch"], row["context_selection"]):
            variants.append(
                (
                    "repair",
                    row["instruction"],
                    row["context"],
                    row["target_patch"],
                    row["context_selection"],
                    None,
                )
            )
        else:
            rejected["repair_preimage_not_fully_visible_in_issue_selected_context"] += 1
        visible = {w["path"] for w in row["context_selection"]["windows"]}
        locations = sorted(
            set(patch_paths(row["target_patch"])) - {p for p in visible if p.startswith("tests/")}
        )
        if locations and set(locations) <= visible:
            variants.append(
                (
                    "localization",
                    row["instruction"],
                    row["context"],
                    json.dumps(locations),
                    row["context_selection"],
                    None,
                )
            )
        task = v4_repository_task_from_row(
            row.get("source_key", "swe_rebench_v2"),
            {
                "repo": row["repos"],
                "base_commit": row["commit"],
                "problem_statement": row["instruction"],
                "patch": row["target_patch"],
                "instance_id": row["instance_id"],
            },
        )
        snapshot, _ = store.snapshot(task)
        for instruction, context, target, selection in completions_from_snapshot(
            row,
            snapshot,
            max_per_file=max_completions_per_file,
            max_total=max_completions_per_snapshot,
        ):
            try:
                evidence = context_license_evidence(snapshot, selection)
                variants.append(("completion", instruction, context, target, selection, evidence))
            except ValueError as exc:
                rejected["completion_license:" + str(exc)] += 1
        for kind, instruction, context, target, selection, evidence in variants:
            payload = {
                **row,
                "record_type": "context_sft_v1",
                "research_role": "repository_dependency_multitask_sft",
                "task_scope": "cross_file_context"
                if len({w["path"] for w in selection["windows"]}) > 1
                else "single_file_context_with_snapshot",
                "task_type": kind,
                "instance_id": "ctx-"
                + stable_fingerprint([row["instance_id"], kind, context, target]),
                "instruction": instruction,
                "context": context,
                "target_text": target,
                "context_selection": selection,
                "context_source_code_license_evidence": evidence
                or row["context_source_code_license_evidence"],
            }
            if disjoint_context:
                payload["disjoint_context"] = True
                payload["context_format"] = "v4"
            record = normalize_context_task(payload)
            record.pop("target_patch", None)
            record["before_signature"] = code_signature(context)
            record["normalized_pair_hash"] = record["pair_hash"]
            record["split_keys"] = ["source-task:" + row["instance_id"]]
            signature = stable_fingerprint([kind, context, target])
            if signature in seen:
                rejected["duplicate_task"] += 1
                continue
            try:
                lengths = []
                row_tokens = {}
                for spec in EXPERIMENT_SPECS:
                    key = (
                        effective_signatures[spec.encoder.model_id] if spec.encoder else None,
                        effective_signatures[spec.decoder.model_id],
                    )
                    if key not in row_tokens:
                        row_tokens[key] = tokenize_context_task(
                            record,
                            spec,
                            encoders[spec.encoder.model_id] if spec.encoder else None,
                            decoders[spec.decoder.model_id],
                            encoder_limit=ENCODER_LIMIT,
                            decoder_limit=DECODER_LIMIT if spec.encoder else BASELINE_LIMIT,
                        )
                    item = row_tokens[key]
                    lengths.append(
                        {
                            "encoder": len(item.encoder_ids),
                            "decoder": len(item.decoder_prefix_ids) + len(item.target_ids) - 1,
                            "target": len(item.target_ids),
                        }
                    )
                record["token_lengths"] = {
                    "encoder": max(v["encoder"] for v in lengths),
                    "pair_decoder": max(v["decoder"] for v in lengths[:4]),
                    "baseline_decoder": max(v["decoder"] for v in lengths[4:]),
                    "target": max(v["target"] for v in lengths),
                }
                accepted.append(record)
                seen.add(signature)
            except ValueError as exc:
                rejected[kind + ":" + str(exc)] += 1
    train, dev, split = split_connected_records(accepted)
    report = {
        **manifest["report"],
        "research_role": "repository_dependency_multitask_sft",
        "task_counts": dict(Counter(r["task_type"] for r in accepted)),
        "task_rejections": dict(rejected),
        "effective_tokenizer_signatures": effective_signatures,
        "repair_eligibility": "all existing-file hunk preimages visible in issue-selected windows; no oracle refilling; new-file patches excluded",
        "split": split,
        "training_plan": {
            "epochs": 1,
            "batch_size": 1,
            "accumulation": 4,
            "updates_full_epoch": (len(train) + 3) // 4,
            "supervised_train_tokens": sum(r["token_lengths"]["target"] for r in train),
        },
    }
    result = export_pool(train + dev, output, report)
    log(
        {
            "stage": "context_pool_complete",
            "train": len(train),
            "dev": len(dev),
            "tasks": report["task_counts"],
        }
    )
    return result


def code_fragment_signatures(text):
    """Exact normalized code/subtree fingerprints; do not claim semantic near-dedup."""
    code = textwrap.dedent(text)
    result = {code_signature(code)} if len(code) >= 64 else set()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return result
    lines = code.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            fragments = [ast.get_source_segment(code, node)]
            if node.body and node.end_lineno:
                fragments.append("".join(lines[node.body[0].lineno - 1 : node.end_lineno]))
            result.update(
                code_signature(textwrap.dedent(f)) for f in fragments if f and len(f) >= 64
            )
    return result


def mixed_pool(foundation, context, benchmark_root, output):
    old, original = load_curated_pool(foundation)
    new, repository = load_curated_pool(context)
    if repository["report"]["benchmark_quarantine"] != original["report"]["benchmark_quarantine"]:
        raise ValueError(
            "Repository core and published foundation use different quarantine registries."
        )
    quarantine = json.loads(
        (benchmark_root / "benchmark_quarantine.json").read_text(encoding="utf-8")
    )
    blocked = set(quarantine["repositories"])
    slugs = quarantine["slugs"]
    signatures = set()
    # Evaluate raw released completion targets, not only the token-eligible subset.
    for directory in benchmark_root.glob("crosscodelong_*_source_v1"):
        for row in read_source_rows(directory / "eval_private/specs.jsonl"):
            signatures.update(code_fragment_signatures(row["reference"]))
    import re
    import zipfile

    from huggingface_hub import hf_hub_download

    from bifocalgemma.data.context_benchmarks import LONGCODE_REVISION, SWEQA_REVISION

    sweqa = hf_hub_download(
        "lailaelkoussy/swe-qa",
        "data/oracle-00000-of-00001.parquet",
        repo_type="dataset",
        revision=SWEQA_REVISION,
        cache_dir=ROOT / "outputs/context_sources/hf",
        local_files_only=True,
    )
    for row in read_source_rows(sweqa):
        for chunk in row["chunks"]:
            signatures.update(code_fragment_signatures(chunk["content"]))
    longcode = hf_hub_download(
        "Steefano/LCB",
        "LongCodeQA.zip",
        repo_type="dataset",
        revision=LONGCODE_REVISION,
        cache_dir=ROOT / "outputs/context_sources/hf",
        local_files_only=True,
    )
    with zipfile.ZipFile(longcode) as archive:
        for row in json.loads(archive.read("LQA/32K.json")):
            for code in re.findall(
                r"\[start of [^\n]+\.py\]\n(.*?)\n\[end of [^\n]+\.py\]", row["repo_text"], re.S
            ):
                signatures.update(code_fragment_signatures(code))
    accepted = []
    rejected = []
    counts = Counter()
    seen = set()
    for index, row in enumerate(old["train"] + old["dev"], 1):
        canonical = row.get("canonical_repos", [])
        reason = None
        if set(canonical) & blocked or any(
            any(s.startswith(repo.replace("/", "-") + "-") for s in slugs) for repo in canonical
        ):
            reason = "new_benchmark_repository"
        elif (
            code_fragment_signatures(row["before"]) | code_fragment_signatures(row["after"])
        ) & signatures:
            reason = "new_benchmark_normalized_code_fragment"
        if reason:
            counts[reason] += 1
            rejected.append({"instance_id": row["instance_id"], "reason": reason})
            continue
        accepted.append({**row, "task_type": "repair"})
        seen.add(row["normalized_pair_hash"])
        if index % 2500 == 0:
            log(
                {
                    "stage": "foundation_new_benchmark_audit",
                    "done": index,
                    "retained": len(accepted),
                }
            )
    from bifocalgemma.data.curation import token_profile

    first_ref, second_ref = EXPERIMENT_SPECS[0].encoder, EXPERIMENT_SPECS[1].encoder
    assert first_ref is not None and second_ref is not None
    encoder = load_verified_tokenizer(first_ref, ROOT / "outputs/context_tokenizers")[0]
    other_encoder = load_verified_tokenizer(second_ref, ROOT / "outputs/context_tokenizers")[0]
    decoder = load_verified_tokenizer(
        EXPERIMENT_SPECS[0].decoder, ROOT / "outputs/context_tokenizers"
    )[0]
    other_decoder = load_verified_tokenizer(
        EXPERIMENT_SPECS[1].decoder, ROOT / "outputs/context_tokenizers"
    )[0]
    decoder_equivalence = (
        decoder.backend_tokenizer.to_str() == other_decoder.backend_tokenizer.to_str()
        and decoder.chat_template == other_decoder.chat_template
        and decoder.special_tokens_map == other_decoder.special_tokens_map
    )
    if not decoder_equivalence:
        raise ValueError("IT tokenizer equivalence changed; profile each decoder separately.")
    accepted, fresh_profile = token_profile(
        accepted,
        encoder,
        decoder,
        encoder_limit=ENCODER_LIMIT,
        decoder_limit=DECODER_LIMIT,
        baseline_limit=BASELINE_LIMIT,
        additional_encoder=other_encoder,
        progress=log,
    )
    # Both before/after editing functions and new dependency targets must pass the same exact-fragment gate.
    for row in new["train"] + new["dev"]:
        if row["task_type"] == "completion":
            from bifocalgemma.data.python_quality import inspect_python_target

            try:
                diagnostics = inspect_python_target(
                    "def _sft_target_body_():\n" + row["target_text"],
                    filename=row["instance_id"],
                )
            except ValueError:
                counts["completion_target_body_invalid_syntax"] += 1
                continue
            if diagnostics:
                counts["completion_target_body_syntax_warning"] += 1
                continue
            row = {
                **row,
                "completion_body_quality_policy": "ast_only_warning_free_body_in_function_wrapper_v1",
            }
        if code_fragment_signatures(row["target_text"]) & signatures:
            counts["context_target_benchmark_overlap"] += 1
            continue
        fragments = set()
        for code in re.split(r"^### File: [^\n]+\n", row["context"], flags=re.M)[1:]:
            fragments.update(code_fragment_signatures(code))
        if fragments & signatures:
            counts["context_input_benchmark_code_fragment"] += 1
            continue
        if row["normalized_pair_hash"] in seen:
            counts["mixed_duplicate_pair"] += 1
            continue
        seen.add(row["normalized_pair_hash"])
        accepted.append(row)
    train, dev, split = split_connected_records(accepted)
    report = {
        "research_role": "editing_and_repository_dependency_multitask_sft",
        "snapshot_policy": "mixed; repository rows use immutable pre-fix archives",
        "context_source_license_policy": "nearest_license_for_each_visible_file_v1",
        "tokens": {
            "limits": {
                "encoder": ENCODER_LIMIT,
                "pair_decoder": DECODER_LIMIT,
                "baseline_decoder": BASELINE_LIMIT,
            },
            "foundation_fresh_profile": fresh_profile,
            "decoder_equivalence": {
                "E2B_IT_E4B_IT_effective_backend_template_special_tokens_identical": decoder_equivalence
            },
            "context_profile": repository["report"]["tokens"],
            "length_measurement": "real audited tokenizers; common six-model cohort",
        },
        "split": split,
        "task_counts": dict(Counter(r.get("task_type", "repair") for r in accepted)),
        "source_counts": dict(Counter(r["source"] for r in accepted)),
        "foundation_fingerprint": original["fingerprint"],
        "context_fingerprint": repository["fingerprint"],
        "repository_quarantine_registry_matches_published_foundation": True,
        "repository_core_foundation_reference": repository["report"]["foundation_fingerprint"],
        "new_benchmark_audit": {
            "policy": "repo_quarantine_and_exact_normalized_code_subtrees_v1",
            "fragment_signatures": len(signatures),
            "rejections": dict(counts),
            "full_semantic_contamination_certification": False,
            "exact_fragment_fields_checked": [
                "foundation_before",
                "foundation_after",
                "context_target",
                "visible_context_code_sections",
            ],
            "partial_nonparsable_window_subtrees_exhaustively_checked": False,
            "unresolved_original_repository_rows": sum(
                not r.get("canonical_repos") for r in accepted
            ),
        },
        "benchmark_quarantine": {
            **original["report"]["benchmark_quarantine"],
            "additional": quarantine,
        },
        "training_plan": {
            "epochs": 1,
            "batch_size": 1,
            "accumulation": 4,
            "updates_full_epoch": (len(train) + 3) // 4,
            "supervised_train_tokens": sum(r["token_lengths"]["target"] for r in train),
        },
    }
    # Foundation rows have fresh 32K/8K profiling; new task rows retain all-six-model evidence.
    result = export_pool(train + dev, output, report)
    atomic_json(output / "new_benchmark_exclusions.json", rejected)
    log(
        {
            "stage": "mixed_pool_complete",
            "train": len(train),
            "dev": len(dev),
            "tasks": report["task_counts"],
        }
    )
    return result


def normalize_math_question(text: str) -> str:
    """Normalize math/QA prompt text for exact GSM8K deduplication."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    return re.sub(r"[^a-z0-9 ]+", "", cleaned).strip()


def build_gsm8k_fingerprints(questions) -> set[str]:
    """Build normalized question fingerprints for GSM8K test-set deduplication."""
    return {
        fp
        for q in questions
        if (fp := normalize_math_question(q))
    }


def _extract_row_license(raw: dict[str, Any], source_key: str) -> str:
    raw_meta = raw.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    for candidate in (
        raw.get("code_license"),
        raw.get("repo_license"),
        raw.get("license"),
        raw.get("license_name"),
        meta.get("license"),
        meta.get("repo_license"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    entry = V4_DATASET_REGISTRY[source_key]
    return str(entry["dataset_license"]).lower()


def _extract_repo_from_v4_row(raw: dict[str, Any]) -> str:
    raw_meta = raw.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    canonical = raw.get("canonical_repos")
    first_canonical = canonical[0] if isinstance(canonical, list) and canonical else ""
    for val in (
        raw.get("repo"),
        raw.get("repo_name"),
        raw.get("repos"),
        first_canonical,
        meta.get("repository_url"),
        meta.get("repo"),
    ):
        if isinstance(val, str) and val.strip():
            cleaned = val.strip().removeprefix("https://github.com/").removeprefix("http://github.com/").removesuffix(".git").strip("/")
            if "/" in cleaned:
                parts = cleaned.split("/")
                return f"{parts[0]}/{parts[1]}".lower()
            return cleaned.lower()
    return ""


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _all_code_fragment_signatures(text: str) -> set[str]:
    if not text:
        return set()
    sigs = set(code_fragment_signatures(text))
    for section in re.split(
        r"^### (?:File: [^\n]+|CROSS-FILE CONTEXT|Cross-File Context|PREFIX|SUFFIX|Target File Prefix[^\n]*|Target File Suffix[^\n]*|Local Cursor[^\n]*)\n",
        text,
        flags=re.M,
    )[1:]:
        sigs.update(code_fragment_signatures(section))
    return sigs


def _resolve_v4_license_evidence(
    raw_license: str,
    *,
    code_only: bool,
) -> tuple[str, list[str], str]:
    """Normalize compound SPDX OR/AND/comma expressions into valid code evidence licenses."""
    from bifocalgemma.data.context_tasks import (
        PERMITTED_V4_CODE_LICENSES,
        PERMITTED_V4_DATASET_LICENSES,
    )

    normalized = raw_license.strip().lower().replace("_", "-")
    allowed = PERMITTED_V4_CODE_LICENSES if code_only else PERMITTED_V4_DATASET_LICENSES
    or_branches = [b.strip("() \t") for b in re.split(r"\s+or\s+", normalized) if b.strip("() \t")]
    chosen_parts: list[str] = []
    for branch in or_branches:
        parts = [p.strip("() \t") for p in re.split(r"\s+and\s+|,", branch) if p.strip("() \t")]
        if parts and all(part in allowed for part in parts):
            chosen_parts = parts
            break
    if not chosen_parts:
        chosen_parts = [
            p.strip("() \t")
            for p in re.split(r"\s+and\s+|\s+or\s+|,", normalized)
            if p.strip("() \t")
        ]
    code_parts = list(
        dict.fromkeys(part for part in chosen_parts if part in PERMITTED_V4_CODE_LICENSES)
    )
    if not code_parts and not code_only:
        code_parts = ["apache-2.0"]
    evidence_lic = " and ".join(code_parts) if code_parts else normalized
    return evidence_lic, code_parts, normalized


def _conversation_to_instruction_context_target(raw: dict[str, Any]) -> tuple[str, str, str]:
    messages = raw.get("messages") or raw.get("conversations")
    if isinstance(messages, list) and len(messages) >= 2:
        user_turns = [
            str(m.get("content") or m.get("value") or "")
            for m in messages
            if str(m.get("role") or m.get("from") or "").lower() in {"user", "human"}
        ]
        assistant_turns = [
            str(m.get("content") or m.get("value") or "")
            for m in messages
            if str(m.get("role") or m.get("from") or "").lower() in {"assistant", "gpt", "model"}
        ]
        if user_turns and assistant_turns:
            full_user = "\n\n".join(t for t in user_turns if t.strip())
            target = assistant_turns[-1]
            if "\n\n" in full_user and len(full_user) > 2048:
                ctx, _, instr = full_user.rpartition("\n\n")
                if len(instr.strip()) >= 8:
                    return instr.strip(), ctx.strip(), target
            return full_user.strip(), full_user.strip(), target
    instr = str(
        raw.get("instruction")
        or raw.get("problem_statement")
        or raw.get("question")
        or raw.get("problem")
        or raw.get("prompt")
        or raw.get("input")
        or ""
    ).strip()
    ctx = str(
        raw.get("context")
        or raw.get("before")
        or raw.get("old_contents")
        or raw.get("input")
        or instr
    ).strip()
    target = str(
        raw.get("target_text")
        or raw.get("target_patch")
        or raw.get("patch")
        or raw.get("model_patch")
        or raw.get("output")
        or raw.get("response")
        or raw.get("completion")
        or raw.get("solution")
        or raw.get("generated_solution")
        or raw.get("answer")
        or ""
    )
    return instr, ctx, target


def adapt_v4_row(
    source_key: str,
    raw: dict[str, Any],
    *,
    blocked_repos: set[str] | frozenset[str] = frozenset(),
    blocked_slugs: set[str] | frozenset[str] = frozenset(),
    code_signatures: set[str] | frozenset[str] = frozenset(),
    gsm8k_fingerprints: set[str] | frozenset[str] = frozenset(),
    repo_aliases: dict[str, str] | None = None,
    max_encoder_tokens: int = ENCODER_LIMIT,
) -> dict[str, Any]:
    """Adapt and strictly validate a vetted v4 dataset row against license & quarantine gates."""
    raw_source = str(raw.get("source") or "")
    if raw_source in BLOCKED_V4_DATASETS or source_key == "nextcoder":
        raise ValueError("blocked_v4_dataset")
    if source_key not in V4_DATASET_REGISTRY:
        raise ValueError(f"unvetted_v4_source:{source_key}")
    entry = V4_DATASET_REGISTRY[source_key]
    if entry["dataset_id"] in BLOCKED_V4_DATASETS:
        raise ValueError("blocked_v4_dataset")

    raw_license = _extract_row_license(raw, source_key)
    code_only = bool(entry.get("requires_row_license_check", False))
    if not is_permissive_license(raw_license, code_only=code_only):
        raise ValueError(f"non_permissive_license:{raw_license}")
    evidence_lic, evidence_code_licenses, normalized_lic = _resolve_v4_license_evidence(
        raw_license, code_only=code_only
    )

    if "min_test_score" in entry:
        score = raw.get("average_test_score")
        if score is None or float(score) < float(entry["min_test_score"]):
            raise ValueError("opencodeinstruct_low_test_score")

    repo_str = _extract_repo_from_v4_row(raw)
    canonical_repo = ""
    if repo_str and "/" in repo_str:
        try:
            canonical_repo = repo_identity(repo_str, repo_aliases)
        except ValueError:
            canonical_repo = repo_str
    if entry.get("quarantine_repos", True) and canonical_repo:
        slug_prefix = canonical_repo.replace("/", "-")
        if canonical_repo in blocked_repos or slug_prefix in blocked_slugs or any(
            s.startswith(slug_prefix + "-") for s in blocked_slugs
        ):
            raise ValueError("benchmark_repository_quarantine")

    task_type = str(raw.get("task_type") or entry["default_task_type"])
    if source_key == "fim_midtraining":
        raw_meta = raw.get("metadata")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        prefix = str(raw.get("prefix") or raw.get("prompt") or "")
        suffix = str(raw.get("suffix") or raw.get("right_context") or "")
        target = str(raw.get("middle") or raw.get("completion") or raw.get("output") or "")
        if not prefix or not target:
            raise ValueError("missing_fim_fields")
        file_path = str(meta.get("file_path") or raw.get("path") or "module.py")
        cross = str(meta.get("cross_file_context") or raw.get("cross_file_context") or "").strip()
        instr = str(raw.get("instruction") or f"Complete the missing code in {file_path}.").strip()
        parts = [f"### Instruction\n{instr}"]
        if cross:
            parts.append(f"### File: {meta.get('cross_file_path', 'sibling.py')}\n{cross}")
        parts.append(f"### File: {file_path}\n{prefix}\n# <MISSING_CODE>\n{suffix}")
        context = "\n\n".join(parts)
        visible_paths = [file_path] + (
            [str(meta.get("cross_file_path", "sibling.py"))] if cross else []
        )
    else:
        instr, context, target = _conversation_to_instruction_context_target(raw)
        if not instr or not context or not target:
            raise ValueError("missing_v4_fields")
        if task_type == "completion" and "# <MISSING_CODE>" not in context:
            context = f"{context}\n# <MISSING_CODE>"
        if task_type == "completion" and target in context:
            raise ValueError("completion_target_visible_in_context")
        path_hint = str(raw.get("path") or raw.get("old_file") or "")
        if path_hint and "### File:" not in context and task_type in {"repair", "completion", "localization"}:
            context = f"### Instruction\n{instr}\n\n### File: {path_hint}\n{context}"
        visible_paths = [
            m.group(1).strip()
            for m in re.finditer(r"^### File:\s*([^\n(]+)", context, re.M)
        ] or [path_hint or "context.txt"]

    if entry.get("dedup_gsm8k") and gsm8k_fingerprints:
        q_norm = normalize_math_question(instr)
        if q_norm in gsm8k_fingerprints or any(fp in q_norm for fp in gsm8k_fingerprints if len(fp) >= 24):
            raise ValueError("gsm8k_test_overlap")

    if code_signatures:
        row_sigs = (
            _all_code_fragment_signatures(target)
            | _all_code_fragment_signatures(context)
            | _all_code_fragment_signatures(str(raw.get("after") or raw.get("new_contents") or ""))
        )
        if row_sigs & code_signatures:
            raise ValueError("benchmark_code_overlap")

    token_limit = int(entry.get("max_tokens") or max_encoder_tokens)
    raw_token_lengths = raw.get("token_lengths")
    raw_tl_enc = (
        raw_token_lengths.get("encoder")
        if isinstance(raw_token_lengths, dict) and isinstance(raw_token_lengths.get("encoder"), int)
        else None
    )
    est_enc_tokens = int(raw.get("encoder_tokens") or raw_tl_enc or _estimate_tokens(context))
    if est_enc_tokens > token_limit:
        raise ValueError("v4_context_exceeds_max_tokens")

    snap_sha = hashlib.sha256(context.encode("utf-8")).hexdigest()
    raw_id = str(raw.get("instance_id") or "")
    inst_id = raw_id if raw_id else f"v4-{source_key}-" + stable_fingerprint([source_key, instr, context, target])[:24]
    lic_sha = hashlib.sha256(f"{entry['dataset_id']}:{raw_license}".encode("utf-8")).hexdigest()
    repo_label = canonical_repo or f"dataset/{source_key}"
    record: dict[str, Any] = {
        "record_type": "context_sft_v1",
        "context_format": "v4",
        "disjoint_context": True,
        "v4_bucket": entry["bucket"],
        "role": "train",
        "task_type": task_type,
        "instance_id": inst_id,
        "group": repo_label if canonical_repo else f"{repo_label}:{inst_id}",
        "canonical_repos": [canonical_repo] if canonical_repo else [],
        "instruction": instr,
        "context": context,
        "target_text": target,
        "source": entry["dataset_id"],
        "source_key": source_key,
        "source_revision": str(raw.get("source_revision") or "pinned"),
        "snapshot_sha256": snap_sha,
        "commit": str(raw.get("base_commit") or raw.get("commit_hash") or raw.get("commit") or snap_sha[:40]),
        "repos": repo_label,
        "license": evidence_lic,
        "dataset_license": normalized_lic,
        "context_selection": {
            "gold_used": False,
            "policy": "v4_disjoint_curated_adapter",
            "windows": [{"path": p} for p in dict.fromkeys(visible_paths)],
        },
        "context_source_code_license_evidence": [
            {
                "path": p,
                "licenses": [{"license": lic, "sha256": lic_sha} for lic in evidence_code_licenses],
            }
            for p in dict.fromkeys(visible_paths)
        ],
        "token_lengths": {
            "encoder": est_enc_tokens,
            "target": _estimate_tokens(target),
        },
    }
    normalized = normalize_context_task(record)
    normalized["before_signature"] = code_signature(context)
    normalized["normalized_pair_hash"] = normalized["pair_hash"]
    return normalized


def _row_repo_candidates(
    row: dict[str, Any], repo_aliases: dict[str, str] | None = None
) -> set[str]:
    candidates: set[str] = set()
    raw_vals: list[Any] = [row.get("repos"), row.get("repo"), row.get("repo_name")]
    canonical_list = row.get("canonical_repos")
    if isinstance(canonical_list, list):
        raw_vals.extend(canonical_list)
    for val in raw_vals:
        if isinstance(val, str) and val.strip():
            cleaned = (
                val.strip()
                .removeprefix("https://github.com/")
                .removeprefix("http://github.com/")
                .removesuffix(".git")
                .strip("/")
            )
            if "/" in cleaned:
                owner, name = cleaned.split("/", 1)
                repo_slug = f"{owner}/{name}".lower()
                try:
                    candidates.add(repo_identity(repo_slug, repo_aliases))
                except ValueError:
                    candidates.add(repo_slug)
            else:
                candidates.add(cleaned.lower())
    return candidates


def _is_quarantined_repo(
    candidates: set[str],
    blocked: set[str],
    slugs: set[str],
) -> bool:
    for canonical_repo in candidates:
        slug_prefix = canonical_repo.replace("/", "-")
        if (
            canonical_repo in blocked
            or slug_prefix in slugs
            or any(s.startswith(slug_prefix + "-") for s in slugs)
        ):
            return True
    return False


def _collect_benchmark_quarantine_assets(
    benchmark_root: Path | None,
    *,
    cache_hf: Path | None = None,
    gsm8k_dir: Path | None = None,
) -> tuple[set[str], list[str]]:
    """Collect code fragment signatures and GSM8K test questions locally without network access."""
    import zipfile

    signatures: set[str] = set()
    gsm8k_questions: list[str] = []

    if benchmark_root is not None and Path(benchmark_root).is_dir():
        root_dir = Path(benchmark_root)
        for directory in sorted(root_dir.iterdir()):
            if not directory.is_dir():
                continue
            for spec_path in (
                directory / "eval_private/specs.jsonl",
                directory / "hidden_gold.jsonl",
            ):
                if spec_path.is_file():
                    for row in read_source_rows(spec_path):
                        for field in ("reference", "test_script", "canonical_solution"):
                            val = row.get(field)
                            if isinstance(val, str) and val.strip():
                                signatures.update(_all_code_fragment_signatures(val))
            for inp_path in (directory / "eval_inputs.jsonl", directory / "inputs.jsonl"):
                if inp_path.is_file():
                    for row in read_source_rows(inp_path):
                        if "gsm8k" in directory.name.lower() or str(row.get("benchmark") or "").lower() == "gsm8k":
                            q = row.get("question")
                            if isinstance(q, str) and q.strip():
                                gsm8k_questions.append(q)
                        elif isinstance(row.get("context"), str) and row["context"].strip():
                            signatures.update(_all_code_fragment_signatures(row["context"]))

    eff_gsm_dir = gsm8k_dir if gsm8k_dir is not None else (ROOT / "outputs/gsm8k_sources")
    gsm_test = eff_gsm_dir / "test.jsonl"
    if gsm_test.is_file():
        for row in read_source_rows(gsm_test):
            q = row.get("question")
            if isinstance(q, str) and q.strip():
                gsm8k_questions.append(q)

    if cache_hf is not None and Path(cache_hf).is_dir():
        try:
            from huggingface_hub import hf_hub_download

            from bifocalgemma.data.context_benchmarks import LONGCODE_REVISION, SWEQA_REVISION

            try:
                sweqa = hf_hub_download(
                    "lailaelkoussy/swe-qa",
                    "data/oracle-00000-of-00001.parquet",
                    repo_type="dataset",
                    revision=SWEQA_REVISION,
                    cache_dir=cache_hf,
                    local_files_only=True,
                )
                for row in read_source_rows(sweqa):
                    for chunk in row.get("chunks", []):
                        content = chunk.get("content") if isinstance(chunk, dict) else None
                        if isinstance(content, str) and content.strip():
                            signatures.update(_all_code_fragment_signatures(content))
            except Exception:
                pass
            try:
                longcode = hf_hub_download(
                    "Steefano/LCB",
                    "LongCodeQA.zip",
                    repo_type="dataset",
                    revision=LONGCODE_REVISION,
                    cache_dir=cache_hf,
                    local_files_only=True,
                )
                with zipfile.ZipFile(longcode) as archive:
                    for row in json.loads(archive.read("LQA/32K.json")):
                        for code in re.findall(
                            r"\[start of [^\n]+\.py\]\n(.*?)\n\[end of [^\n]+\.py\]",
                            row.get("repo_text", ""),
                            re.S,
                        ):
                            signatures.update(code_fragment_signatures(code))
            except Exception:
                pass
        except ImportError:
            pass

    return signatures, list(dict.fromkeys(gsm8k_questions))


def mixed_pool_v4(
    v4_sources: dict[str, Any],
    benchmark_root: Path | None,
    output: Path,
    *,
    foundation: Path | None = None,
    context: Path | None = None,
    blocked_repos: set[str] | None = None,
    blocked_slugs: set[str] | None = None,
    code_signatures: set[str] | None = None,
    gsm8k_questions: tuple[str, ...] | list[str] = (),
    foundation_edit_cap: int = 4000,
    grounded_max_parents: int = 500,
    target_count: int | None = None,
    min_long_context_ratio: float = 0.40,
    seed: int = 17,
) -> dict[str, Any]:
    """Build the v4 disjoint-context curated pool (`mixed_context_enc32k_dec8k_v4`)."""
    blocked = set(blocked_repos or ())
    slugs = set(blocked_slugs or ())
    sigs = set(code_signatures or ())
    eff_gsm_questions = list(gsm8k_questions)
    if benchmark_root is not None:
        bench_path = Path(benchmark_root)
        if (bench_path / "benchmark_quarantine.json").is_file():
            quarantine = json.loads(
                (bench_path / "benchmark_quarantine.json").read_text(encoding="utf-8")
            )
            blocked |= set(quarantine.get("repositories", []))
            slugs |= set(quarantine.get("slugs", []))
        if code_signatures is None or not eff_gsm_questions:
            auto_sigs, auto_gsm = _collect_benchmark_quarantine_assets(bench_path)
            if code_signatures is None:
                sigs |= auto_sigs
            if not eff_gsm_questions:
                eff_gsm_questions = auto_gsm

    gsm8k_fps = build_gsm8k_fingerprints(eff_gsm_questions)
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    # 1. Optionally carry over non-NextCoder foundation rows (capped) & context pool rows
    if foundation is not None:
        old_rows, _ = load_curated_pool(foundation)
        cap_counts: Counter[str] = Counter()
        for row in old_rows["train"] + old_rows["dev"]:
            src = str(row.get("source") or "")
            src_lower = src.lower()
            if src == "microsoft/NextCoderDataset" or src_lower == "nextcoder" or src in BLOCKED_V4_DATASETS:
                rejection_counts["dropped_nextcoder_or_blocked"] += 1
                continue
            if cap_counts[src] >= foundation_edit_cap:
                rejection_counts["foundation_edit_capped"] += 1
                continue
            src_key = (
                "editpackft"
                if "editpackft" in src_lower
                else ("commitpackft" if "commitpackft" in src_lower else "")
            )
            if not src_key:
                continue
            try:
                adapted = adapt_v4_row(
                    src_key,
                    {**row, "task_type": "repair"},
                    blocked_repos=blocked,
                    blocked_slugs=slugs,
                    code_signatures=sigs,
                )
            except ValueError as err:
                reason = str(err)
                rejection_counts[reason.split(":")[0]] += 1
                rejected.append(
                    {"source_key": f"foundation:{src_key}", "instance_id": row.get("instance_id"), "reason": reason}
                )
                continue
            if adapted["instance_id"] not in seen_ids:
                seen_ids.add(adapted["instance_id"])
                admitted.append(adapted)
                cap_counts[src] += 1

    if context is not None:
        ctx_rows, _ = load_curated_pool(context)
        for row in ctx_rows["train"] + ctx_rows["dev"]:
            src = str(row.get("source") or "")
            src_key = str(row.get("source_key") or "")
            if (
                src in BLOCKED_V4_DATASETS
                or src.lower() == "nextcoder"
                or src_key == "nextcoder"
            ):
                rejection_counts["blocked_v4_dataset"] += 1
                rejected.append(
                    {"source_key": f"context:{src_key or src}", "instance_id": row.get("instance_id"), "reason": "blocked_v4_dataset"}
                )
                continue
            if _is_quarantined_repo(_row_repo_candidates(row), blocked, slugs):
                rejection_counts["benchmark_repository_quarantine"] += 1
                rejected.append(
                    {"source_key": f"context:{src_key or src}", "instance_id": row.get("instance_id"), "reason": "benchmark_repository_quarantine"}
                )
                continue
            if sigs:
                ctx_sigs = (
                    _all_code_fragment_signatures(str(row.get("target_text") or row.get("target_patch") or ""))
                    | _all_code_fragment_signatures(str(row.get("before") or ""))
                    | _all_code_fragment_signatures(str(row.get("after") or ""))
                    | _all_code_fragment_signatures(str(row.get("context") or ""))
                )
                if ctx_sigs & sigs:
                    rejection_counts["benchmark_code_overlap"] += 1
                    rejected.append(
                        {"source_key": f"context:{src_key or src}", "instance_id": row.get("instance_id"), "reason": "benchmark_code_overlap"}
                    )
                    continue
            enriched: dict[str, Any] = {**row, "context_format": "v4", "disjoint_context": True}
            before_val = str(enriched.get("before") or enriched.get("context") or "")
            enriched.setdefault("before", before_val)
            enriched.setdefault("after", "")
            enriched.setdefault("path", "repository_context")
            enriched.setdefault("before_signature", code_signature(before_val))
            enriched.setdefault(
                "normalized_pair_hash",
                enriched.get("pair_hash")
                or hashlib.sha256(
                    (
                        str(enriched.get("context") or "")
                        + "\0"
                        + str(enriched.get("target_text") or enriched.get("target_patch") or "")
                    ).encode("utf-8")
                ).hexdigest(),
            )
            inst_id = str(enriched["instance_id"])
            if inst_id not in seen_ids:
                seen_ids.add(inst_id)
                admitted.append(enriched)

    # 2. Adapt all vetted v4 sources
    for source_key, raw_iter in v4_sources.items():
        for idx, raw in enumerate(raw_iter):
            try:
                adapted = adapt_v4_row(
                    source_key,
                    raw,
                    blocked_repos=blocked,
                    blocked_slugs=slugs,
                    code_signatures=sigs,
                    gsm8k_fingerprints=gsm8k_fps,
                )
            except ValueError as err:
                reason = str(err)
                rejection_counts[reason.split(":")[0]] += 1
                rejected.append(
                    {"source_key": source_key, "offset": idx, "reason": reason}
                )
                continue
            if adapted["instance_id"] in seen_ids:
                rejection_counts["duplicate_instance_id"] += 1
                continue
            seen_ids.add(adapted["instance_id"])
            admitted.append(adapted)

    if len(admitted) < 2:
        raise ValueError("insufficient_v4_admitted_records")

    # 3. Connected split BEFORE grounded derivation so train/dev stay strictly isolated
    train_base, dev_base, split_info = split_connected_records(admitted, seed=seed)

    # 4. Wire grounded_tasks.py on admitted parents (preserving parent split)
    grounded = generate_grounded_pool_tasks(
        train_base + dev_base,
        seed=seed,
        max_parents=grounded_max_parents,
    )
    for g in grounded:
        if g["instance_id"] not in seen_ids:
            seen_ids.add(g["instance_id"])
            if g.get("pool_split") == "dev":
                dev_base.append(g)
            else:
                train_base.append(g)

    # 5. Wire curriculum.py for 35/25/25/15 target mix & >=40% 8k-32k rows
    train_target = target_count if target_count is not None else len(train_base)
    train_selected, train_curriculum_stats = build_v4_curriculum(
        train_base,
        target_count=train_target,
        target_mix=V4_DEFAULT_MIX,
        min_long_context_ratio=min_long_context_ratio,
        seed=seed,
    )
    for row in train_selected:
        row["pool_split"] = "train"
    for row in dev_base:
        row["pool_split"] = "dev"

    all_selected = train_selected + dev_base
    report = {
        "research_role": "v4_disjoint_context_multitask_sft",
        "pool_recipe": "mixed_context_enc32k_dec8k_v4",
        "context_format": "v4",
        "disjoint_context": True,
        "snapshot_policy": "mixed_v4; repo snapshots immutable; grounded tasks derived post-split",
        "context_source_license_policy": "nearest_license_for_each_visible_file_v1",
        "dropped_sources": ["microsoft/NextCoderDataset"],
        "tokens": {
            "limits": {
                "encoder": ENCODER_LIMIT,
                "pair_decoder": DECODER_LIMIT,
                "baseline_decoder": BASELINE_LIMIT,
            }
        },
        "split": {**split_info, "train": len(train_selected), "dev": len(dev_base)},
        "task_counts": dict(Counter(r.get("task_type", "repair") for r in all_selected)),
        "bucket_counts": train_curriculum_stats["bucket_counts"],
        "curriculum_stats": train_curriculum_stats,
        "grounded_tasks_count": len(grounded),
        "source_counts": dict(Counter(r["source"] for r in all_selected)),
        "rejections": dict(rejection_counts),
        "benchmark_quarantine": {
            "blocked_repositories": sorted(blocked),
            "crosscodeeval_repository_slugs": sorted(slugs),
        },
        "training_plan": {
            "epochs": 1,
            "batch_size": 1,
            "accumulation": 4,
            "updates_full_epoch": (len(train_selected) + 3) // 4,
            "supervised_train_tokens": sum(
                r.get("token_lengths", {}).get("target", 1) for r in train_selected
            ),
        },
    }
    result = export_pool(all_selected, output, report)
    atomic_json(Path(output) / "v4_exclusions.json", rejected)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["benchmarks", "repository", "tasks", "mix", "mix-v4"]
    )
    parser.add_argument("--source-key", choices=["swe_rebench_v2", "swe_fixer"], default="swe_rebench_v2")
    parser.add_argument("--target-count", type=int, default=0, help="0 = all admitted candidates")
    parser.add_argument("--per-repo-limit", type=int, default=0, help="0 = no repository quota")
    parser.add_argument(
        "--cached-snapshots-only",
        action="store_true",
        help="Resume admission without repeating snapshot downloads; missing archives remain explicit exclusions.",
    )
    args = parser.parse_args()
    cache = ROOT / "outputs/context_sources"
    output = ROOT / "outputs/context_benchmarks_v1"
    tokenizers = ROOT / "outputs/full_corpus_raw/tokenizers"
    if not tokenizers.exists():
        tokenizers = ROOT / "outputs/repository_corpus_probe_v1/tokenizers"
    # Use the already audited tokenizer cache; no model weights are downloaded.
    from bifocalgemma.repository_study import fetch_tokenizer_assets

    tokenizers = ROOT / "outputs/context_tokenizers"
    for spec in EXPERIMENT_SPECS[:2]:
        fetch_tokenizer_assets(spec, tokenizers)
    if args.stage == "benchmarks":
        benchmarks(output, cache / "hf", tokenizers)
    elif args.stage == "repository":
        prepare_core(
            ROOT / "outputs/repository_enc32k_dec8k_context_v1",
            cache,
            ROOT / "outputs/instruction_enc8k_dec2k_v2_ready",
            tokenizers,
            output,
            args.target_count,
            args.per_repo_limit,
            prefetch=not args.cached_snapshots_only,
            source_key=args.source_key,
            disjoint_context=True,
        )
    elif args.stage == "tasks":
        context_pool(
            ROOT / "outputs/repository_enc32k_dec8k_context_v1",
            ROOT / "outputs/context_sft_enc32k_dec8k_v1",
            cache,
            tokenizers,
            disjoint_context=True,
        )
    elif args.stage == "mix-v4":
        code_sigs, gsm8k_qs = _collect_benchmark_quarantine_assets(
            output, cache_hf=cache / "hf"
        )
        mixed_pool_v4(
            {},
            output,
            ROOT / "outputs/mixed_context_enc32k_dec8k_v4",
            foundation=ROOT / "outputs/instruction_enc8k_dec2k_v2_ready",
            context=ROOT / "outputs/context_sft_enc32k_dec8k_v1",
            code_signatures=code_sigs,
            gsm8k_questions=gsm8k_qs,
        )
    else:
        mixed_pool(
            ROOT / "outputs/instruction_enc8k_dec2k_v2_ready",
            ROOT / "outputs/context_sft_enc32k_dec8k_v1",
            output,
            ROOT / "outputs/mixed_context_enc32k_dec8k_v3",
        )


if __name__ == "__main__":
    main()
