"""Build pinned, audited context benchmarks and a repository SFT pool locally."""
# ruff: noqa: E402 -- standalone tool bootstraps the local src package before imports

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
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
    normalize_context_task,
    repair_preimage_coverage,
    tokenize_context_task,
)
from bifocalgemma.data.curation import code_signature, export_pool, split_connected_records
from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint
from bifocalgemma.data.patches import patch_paths
from bifocalgemma.data.repository import (
    SOURCE_REVISIONS,
    read_source_rows,
    repository_task_from_row,
)
from bifocalgemma.data.repository_corpus import (
    ArchiveSnapshotStore,
    build_repository_corpus,
    context_license_evidence,
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
):
    from huggingface_hub import hf_hub_download

    from bifocalgemma.data.full_pool import build_benchmark_index, fetch_benchmark_files

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
    source, revision, _ = SOURCE_REVISIONS["swe_rebench_v2"]
    source_path = hf_hub_download(
        source,
        "data/train-00000-of-00001.parquet",
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
            task = repository_task_from_row("swe_rebench_v2", row)
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
            store.fetch_archive(repository_task_from_row("swe_rebench_v2", row))
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
        "swe_rebench_v2",
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
    )


def completion_from_snapshot(row, snapshot):
    """Mask an unchanged function that calls a symbol imported from another source file."""
    files = {f.path: f.content for f in snapshot.files}
    changed = set(patch_paths(row["target_patch"]))
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
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
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
            a, b = node.body[0].lineno - 1, node.end_lineno
            assert b is not None
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
            return instruction, context, target, selection
    return None


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


def context_pool(core, output, cache, tokenizers):
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
        task = repository_task_from_row(
            "swe_rebench_v2",
            {
                "repo": row["repos"],
                "base_commit": row["commit"],
                "problem_statement": row["instruction"],
                "patch": row["target_patch"],
                "instance_id": row["instance_id"],
            },
        )
        snapshot, _ = store.snapshot(task)
        completion = completion_from_snapshot(row, snapshot)
        if completion:
            instruction, context, target, selection = completion
            try:
                evidence = context_license_evidence(snapshot, selection)
                variants.append(("completion", instruction, context, target, selection, evidence))
            except ValueError as exc:
                rejected["completion_license:" + str(exc)] += 1
        for kind, instruction, context, target, selection, evidence in variants:
            record = normalize_context_task(
                {
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
            )
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["benchmarks", "repository", "tasks", "mix"])
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
        )
    elif args.stage == "tasks":
        context_pool(
            ROOT / "outputs/repository_enc32k_dec8k_context_v1",
            ROOT / "outputs/context_sft_enc32k_dec8k_v1",
            cache,
            tokenizers,
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
