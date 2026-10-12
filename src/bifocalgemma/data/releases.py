"""Frozen data releases for a private HF dataset repo, separate from model weights."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from bifocalgemma.data.offline_sft import load_curated_pool, stable_fingerprint

FORMAT = "bifocalgemma_dataset_release_v1"
NAME = re.compile(r"[a-z][a-z0-9_-]{0,79}")
REVISION = re.compile(r"[0-9a-f]{40}")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in stream if line.strip())


def archive_files(root, names, output):
    """Deterministic ZIP of an explicit allowlist; no caches, weights or run logs."""
    inventory = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(names):
            source = root / name
            if not source.resolve().is_relative_to(root.resolve()):
                raise ValueError("release_source_path")
            if not source.is_file() or source.is_symlink():
                raise ValueError("release_missing_or_linked_file:" + name)
            inventory[name] = {"sha256": digest(source), "bytes": source.stat().st_size}
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with source.open("rb") as incoming, archive.open(info, "w") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    return {
        "path": output.name,
        "sha256": digest(output),
        "bytes": output.stat().st_size,
        "members": inventory,
    }


def write_view(path, records, *, training, protocol=None, shots=None):
    """Lossless public rows plus explicit viewer fields; eval gold stays separate."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = [
        "instance_id",
        "role",
        "source",
        "instruction",
        "context",
        "provenance_json",
        "question",
        "shots",
        "protocol_json",
        "task_scope",
        "research_role",
        "edit_kind",
        "task_type",
        "condition",
        "completion_kind",
        "max_encoder_tokens",
        "max_new_tokens",
        "path",
        "before",
        "after",
        "record_json",
    ]
    if training:
        columns.extend(["target_patch", "target_text"])
    schema = pa.schema([(name, pa.string()) for name in columns])
    batch = []
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for row in records:
            item = {
                "instance_id": row["instance_id"],
                "role": row["role"],
                "source": str(row.get("source", row.get("benchmark", ""))),
                "instruction": row.get("instruction", ""),
                "question": row.get("question", ""),
                "shots": str(shots) if shots is not None else "",
                "protocol_json": json.dumps(protocol or {}, sort_keys=True),
                "task_scope": row.get(
                    "task_scope", "math_question" if "question" in row else "single_file_context"
                ),
                "research_role": row.get(
                    "research_role",
                    "evaluation" if not training else "auxiliary_single_file_editing",
                ),
                "edit_kind": row.get("edit_kind", ""),
                "task_type": row.get("task_type", "repair"),
                "condition": row.get("condition", ""),
                "completion_kind": row.get("completion_kind", ""),
                "max_encoder_tokens": str(row.get("max_encoder_tokens", "")),
                "max_new_tokens": str(row.get("max_new_tokens", "")),
                "path": row.get("path", ""),
                "before": row.get("before", "") if training else "",
                "after": row.get("after", "") if training else "",
                "record_json": json.dumps(row, sort_keys=True, ensure_ascii=False),
                "context": row.get("context", ""),
                "provenance_json": json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "source_revision",
                            "dataset_license",
                            "code_license",
                            "code_license_verification",
                            "provenance_type",
                            "git_commit",
                            "seed_program_hash",
                            "commit_kind",
                            "normalization_policy",
                            "patch_line_policy",
                            "upstream_row",
                            "canonical_repos",
                            "commit",
                            "pool_split",
                            "quality_gate",
                            "quality_flags",
                            "context_policy",
                            "official_protocol",
                        )
                        if key in row
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            if training:
                if not item["context"]:
                    item["context"] = (
                        f"### Instruction\n{row['instruction']}\n\n"
                        f"### File: {row['path']}\n{row['before']}"
                    )
                item["target_patch"] = row.get("target_patch", "")
                item["target_text"] = row.get("target_text", row.get("target_patch", ""))
                item["task_type"] = row.get("task_type", "repair")
            batch.append(item)
            if len(batch) == 256:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema), row_group_size=256)
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema), row_group_size=256)


def build_release(spec_path, output):
    """Only complete, validated pools can enter a release; partial logs are not data."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new immutable release output directory.")
    # Stage separately so a failed export cannot look like a published release.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        collections = {}
        configs = []
        for entry in spec["collections"]:
            name = entry["name"]
            if not NAME.fullmatch(name) or name in collections:
                raise ValueError("invalid_or_duplicate_collection")
            root = Path(entry["directory"])
            directory = stage / "collections" / name
            directory.mkdir(parents=True)
            views = directory / "view"
            views.mkdir()
            if entry["kind"] == "curated_pool":
                split_rows, manifest = load_curated_pool(root)
                audit_split_disjointness(split_rows)
                archives = {
                    "public": archive_files(
                        root,
                        ["train.jsonl", "dev.jsonl", "manifest.json"]
                        + (
                            ["remediation_ledger.jsonl"]
                            if (root / "remediation_ledger.jsonl").exists()
                            else []
                        ),
                        directory / "pool.zip",
                    )
                }
                counts = {key: len(value) for key, value in split_rows.items()}
                role = "train"
                limits = manifest["report"]["tokens"]["limits"]
                sources = dict(
                    Counter(row["source"] for group in split_rows.values() for row in group)
                )
                licenses = dict(
                    Counter(
                        row.get("code_license", row.get("license", "unknown"))
                        for group in split_rows.values()
                        for row in group
                    )
                )
                for split, records in split_rows.items():
                    write_view(views / (split + ".parquet"), records, training=True)
                configs.append(
                    {
                        "config_name": name,
                        "data_files": [
                            {"split": "train", "path": f"collections/{name}/view/train.parquet"},
                            {"split": "validation", "path": f"collections/{name}/view/dev.parquet"},
                        ],
                    }
                )
                evidence = {
                    "pool_fingerprint": manifest["fingerprint"],
                    "token_limits": limits,
                    "source_counts": sources,
                    "code_licenses": licenses,
                    "functional_quality_verified": False,
                    "research_role": manifest["report"].get(
                        "research_role", "auxiliary_single_file_editing"
                    ),
                    "remediation": manifest["report"].get("remediation"),
                }
            elif entry["kind"] in {"gsm8k", "prepared_code_eval"}:
                if entry["kind"] == "gsm8k":
                    from bifocalgemma.evaluation.gsm8k import load_gsm8k

                    public, private, manifest = load_gsm8k(root, include_gold=True)
                    public_names = ["inputs.jsonl", "manifest.json"]
                    gold_names = ["hidden_gold.jsonl"]
                    evidence = {
                        "manifest_fingerprint": manifest["fingerprint"],
                        "protocol": manifest["protocol"],
                        "shots": manifest["shots"],
                    }
                else:
                    public = list(rows(root / "eval_inputs.jsonl"))
                    private = list(rows(root / "eval_private/specs.jsonl"))
                    manifest = json.loads(
                        (root / "preparation_report.json").read_text(encoding="utf-8")
                    )
                    public_names = ["eval_inputs.jsonl", "preparation_report.json"]
                    public_names += sorted(
                        {row["snapshot_file"] for row in public if "snapshot_file" in row}
                    )
                    gold_names = ["eval_private/specs.jsonl"]
                    evidence = {
                        "preparation_report": manifest,
                        "official_protocol": False,
                        "protocol": manifest.get(
                            "protocol",
                            {
                                "official_protocol": False,
                                "adaptation": "native_diff_output_adapter",
                            },
                        ),
                        "snapshot_patch_checks": validate_code_eval_snapshots(
                            root, public, private
                        ),
                    }
                public_ids = [row["instance_id"] for row in public]
                if (
                    not public
                    or len(set(public_ids)) != len(public)
                    or public_ids != [row["instance_id"] for row in private]
                ):
                    raise ValueError("eval_identity_mismatch")
                if any(
                    row.get("role") != "eval"
                    or any(
                        key in row
                        for key in (
                            "gold_patch",
                            "target_patch",
                            "answer",
                            "reference",
                            "test_code",
                            "test_script",
                            "correct_answer",
                            "correct_letter",
                            "target_text",
                            "groundtruth",
                            "is_hard",
                        )
                    )
                    for row in public
                ):
                    raise ValueError("eval_gold_in_public_input_or_wrong_role")
                archives = {
                    "public": archive_files(root, public_names, directory / "public.zip"),
                    "gold": archive_files(root, gold_names, directory / "gold.zip"),
                }
                write_view(
                    views / "test.parquet",
                    public,
                    training=False,
                    protocol=evidence.get(
                        "protocol",
                        {"official_protocol": False, "adaptation": "native_diff_output_adapter"},
                    ),
                    shots=evidence.get("shots"),
                )
                counts, role = {"test": len(public)}, "eval"
                configs.append(
                    {
                        "config_name": name,
                        "data_files": [
                            {"split": "test", "path": f"collections/{name}/view/test.parquet"}
                        ],
                    }
                )
            else:
                raise ValueError("unsupported_release_kind")
            for archive in archives.values():
                archive["path"] = f"collections/{name}/" + archive["path"]
            collections[name] = {
                "role": role,
                "kind": entry["kind"],
                "counts": counts,
                "description": entry["description"],
                "archives": archives,
                "evidence": evidence,
            }
        if not collections:
            raise ValueError("empty_release")
        # Notices are exact upstream bytes, never user credentials or whole caches.
        for notice in spec.get("notices", []):
            name = notice["name"]
            if not NAME.fullmatch(name):
                raise ValueError("invalid_notice_name")
            destination = stage / "notices" / (name + ".txt")
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(notice["file"], destination)
        audit_artifacts = {}
        for entry in spec.get("audits", []):
            name = entry["name"]
            if not NAME.fullmatch(name) or name in audit_artifacts:
                raise ValueError("invalid_audit_artifact_name")
            source = Path(entry["file"])
            json.loads(source.read_text(encoding="utf-8"))
            destination = stage / "audits" / (name + ".json")
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            audit_artifacts[name] = {
                "path": destination.relative_to(stage).as_posix(),
                "sha256": digest(destination),
                "bytes": destination.stat().st_size,
            }
        release = {
            "format": FORMAT,
            "viewer_schema_version": 3,
            "audits": audit_artifacts,
            "supersedes_revision": spec.get("supersedes_revision"),
            "name": spec["name"],
            "collections": collections,
            "benchmark_catalog": spec.get("benchmark_catalog", []),
            "curator": spec["curator"],
            "date": spec["date"],
            "limitations": [
                "Derived curation of credited upstream datasets, not original task authorship.",
                "Static parse/patch checks do not establish functional repair quality.",
                "No certification of pretraining or semantic contamination removal.",
                "Private gold is separated by download policy, not access-control isolation.",
            ],
        }
        release["fingerprint"] = stable_fingerprint(release)
        write_json(stage / "release.json", release)
        write_card(stage, release, configs)
        inventory = {
            p.relative_to(stage).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        write_json(stage / "inventory.json", inventory)
        stage.rename(output)
    return release


def validate_code_eval_snapshots(root, public, private):
    """Bind gold to the same public before snapshot and check patch application."""
    from bifocalgemma.data.context import canonical_snapshot_hash
    from bifocalgemma.data.patches import apply_patch_to_snapshot
    from bifocalgemma.repository_study import snapshot_from_json

    checked = 0
    for visible, gold in zip(public, private, strict=True):
        if "snapshot_file" not in visible:
            continue
        name = visible["snapshot_file"]
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("eval_snapshot_path")
        snapshot = snapshot_from_json(path)
        if (
            canonical_snapshot_hash(snapshot) != visible["snapshot_sha256"]
            or gold.get("snapshot_sha256") != visible["snapshot_sha256"]
            or gold.get("snapshot_file") != name
        ):
            raise ValueError("eval_snapshot_identity")
        if "gold_patch" in gold:
            apply_patch_to_snapshot(snapshot, gold["gold_patch"])
        checked += 1
    return checked


def audit_split_disjointness(split_rows):
    """Recheck connected identities before export, independent of row file hashes."""
    keys = {}
    for split, records in split_rows.items():
        current = set()
        for row in records:
            current.update("repo:" + value for value in row.get("canonical_repos", []))
            current.update("structure:" + value for value in row.get("split_keys", []))
            for key in ("commit", "before_signature", "normalized_pair_hash"):
                if row.get(key):
                    current.add(key + ":" + row[key])
        keys[split] = current
    if keys["train"] & keys["dev"]:
        raise ValueError("release_train_dev_lineage_overlap")


def write_card(root, release, configs):
    import yaml

    metadata = {
        "language": ["en", "code"],
        "license": "other",
        "license_name": "upstream-per-record-licenses",
        "license_link": f"https://huggingface.co/datasets/{release['curator']}/bifocalgemma-data/blob/main/LICENSES.md",
        "task_categories": ["text-generation"],
        "configs": configs,
        "pretty_name": "BifocalGemma curated data",
    }
    lines = [
        "---",
        yaml.safe_dump(metadata, sort_keys=False).strip(),
        "---",
        "# BifocalGemma curated data",
        "",
        "Private, frozen training and evaluation artifacts. Code/model repo: "
        "https://github.com/daruoktab/bifocalgemma",
        "",
        "| Collection | Role | Counts |",
        "| --- | --- | --- |",
    ]
    for name, item in release["collections"].items():
        lines.append(f"| {name} | {item['role']} | {json.dumps(item['counts'])} |")
    for name, item in release["collections"].items():
        lines.extend(["", f"**{name}**: {item['description']}"])
    lines.extend(
        [
            "",
            "Use a full Hub commit SHA and verify release/file hashes before loading.",
            "Training downloads never select eval collections; eval gold needs explicit "
            "include_gold=True. All gold is still accessible to repo collaborators.",
            "Viewer schema v3 exposes task_type, condition, budgets, question, shots, protocol and lossless record_json. "
            "Native ZIPs preserve scorer/pool schemas, IDs and bytes. GSM8K protocols "
            "share 1,319 test questions intentionally; they are not 2,638 unique tasks.",
            "Training collection roles distinguish editing foundation from repository dependency tasks. "
            "NextCoder seed code licensing and original repository lineage are unresolved.",
            "Benchmark configs are adaptations; official protocol parity and model quality "
            "are not established. Registry-only benchmarks are not ready evaluators.",
            "See release.json for status and limits; do not relabel a 2k cohort as 8k.",
            "",
            "## Limitations",
            "",
        ]
        + release["limitations"]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "LICENSES.md").write_text(
        "# Upstream licenses and attribution\n\n"
        "This collection has no blanket code license. Dataset metadata licensing does not "
        "replace underlying source-code licenses. Keep per-record provenance, pinned revisions "
        "and upstream notices when redistributing. notices/ preserves fetched source cards and "
        "license texts; copyright notice coverage for individual GitHub files is not certified. "
        "Private staging is not public-release clearance. Do not publish this repo publicly "
        "without a separate attribution review.\n\n"
        "EditPackFT, CommitPackFT, NextCoderDataset and HumanEvalPack declare MIT dataset "
        "licensing. GSM8K source is MIT, copyright OpenAI. Other benchmark sources remain "
        "referenced in the catalog unless their payload and license are independently admitted.\n",
        encoding="utf-8",
    )


def verify_release(directory):
    root = Path(directory)
    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    if release.get("format") != FORMAT or stable_fingerprint(
        {key: value for key, value in release.items() if key != "fingerprint"}
    ) != release.get("fingerprint"):
        raise ValueError("release_manifest_integrity")
    inventory = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
    for name, entry in inventory.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("release_inventory_path")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("release_file_integrity:" + name)
    return release


def unpack_archive(archive_path, descriptor, destination):
    if digest(archive_path) != descriptor["sha256"]:
        raise ValueError("release_archive_integrity")
    root = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or set(names) != set(descriptor["members"]):
            raise ValueError("release_archive_members")
        for info in archive.infolist():
            path = root / info.filename
            item = descriptor["members"][info.filename]
            if (
                not path.resolve().is_relative_to(root)
                or info.is_dir()
                or info.file_size != item["bytes"]
                or info.file_size > 2_000_000_000
            ):
                raise ValueError("release_archive_path_or_size")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if digest(path) != item["sha256"]:
                raise ValueError("release_member_integrity")


def fetch_collection(
    repo_id, revision, collection, output, *, expected_role="train", include_gold=False, token=None
):
    """Download one named immutable config only. No training-time upstream curation."""
    from huggingface_hub import hf_hub_download

    if not REVISION.fullmatch(revision):
        raise ValueError("Dataset revision must be a full immutable 40-character commit SHA.")
    if expected_role not in {"train", "eval"} or (expected_role == "train" and include_gold):
        raise ValueError("invalid_dataset_role_or_gold")
    source = hf_hub_download(
        repo_id, "release.json", repo_type="dataset", revision=revision, token=token
    )
    release = json.loads(Path(source).read_text(encoding="utf-8"))
    if release.get("format") != FORMAT or stable_fingerprint(
        {key: value for key, value in release.items() if key != "fingerprint"}
    ) != release.get("fingerprint"):
        raise ValueError("release_manifest_integrity")
    if collection not in release["collections"]:
        ready = [
            name for name, entry in release["collections"].items() if entry["role"] == expected_role
        ]
        raise ValueError(
            f"Dataset collection {collection!r} is not published at this revision. Ready {expected_role} collections: {ready}"
        )
    item = release["collections"][collection]
    if item["role"] != expected_role:
        raise ValueError("dataset_collection_role_mismatch")
    output = Path(output)
    if output.exists():
        if expected_role == "train":
            _, manifest = load_curated_pool(output)
            if manifest["fingerprint"] != item["evidence"]["pool_fingerprint"]:
                raise ValueError("existing_pool_has_another_fingerprint")
            return manifest
        for key in ("public", "gold") if include_gold else ("public",):
            for name, evidence in item["archives"][key]["members"].items():
                path = (output / name).resolve()
                if not path.is_relative_to(output.resolve()) or not path.is_file():
                    raise ValueError("existing_eval_collection_missing_file")
                if digest(path) != evidence["sha256"]:
                    raise ValueError("existing_eval_collection_integrity")
        return {"collection": collection, "role": "eval", "include_gold": include_gold}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        for key in ("public", "gold") if include_gold else ("public",):
            descriptor = item["archives"][key]
            file = hf_hub_download(
                repo_id, descriptor["path"], repo_type="dataset", revision=revision, token=token
            )
            unpack_archive(file, descriptor, stage)
        if expected_role == "train":
            _, manifest = load_curated_pool(stage)
            if manifest["fingerprint"] != item["evidence"]["pool_fingerprint"]:
                raise ValueError("downloaded_pool_identity")
        else:
            manifest = {"collection": collection, "role": "eval", "include_gold": include_gold}
        stage.rename(output)
    return manifest


def publish_release(directory, repo_id, *, replace_current=False, expected_parent=None):
    """Create/upload only a private dataset owned by the authenticated account."""
    from huggingface_hub import HfApi

    release = verify_release(directory)
    if expected_parent is not None and not REVISION.fullmatch(expected_parent):
        raise ValueError("expected_parent_must_be_full_commit_sha")
    api = HfApi()
    owner = api.whoami()["name"]
    if repo_id.split("/")[0] != owner or len(repo_id.split("/")) != 2:
        raise ValueError("Dataset repo must belong to the authenticated personal account.")
    if api.repo_exists(repo_id, repo_type="dataset"):
        if not api.dataset_info(repo_id).private:
            raise ValueError("Refusing to upload private release into a public repo.")
    else:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=False)
    current = api.dataset_info(repo_id)
    if not current.private:
        raise ValueError("Dataset privacy verification failed.")
    if expected_parent is not None and current.sha != expected_parent:
        raise ValueError("dataset_head_changed_before_publish")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=directory,
        commit_message="Publish corrected frozen dataset release",
        delete_patterns=["collections/**", "notices/**"] if replace_current else None,
        parent_commit=expected_parent,
        allow_patterns=list(
            json.loads((Path(directory) / "inventory.json").read_text(encoding="utf-8"))
        )
        + ["inventory.json"],
    )
    info = api.dataset_info(repo_id, revision=commit.oid)
    if not info.private:
        raise ValueError("Dataset is not private after upload.")
    return {
        "repo_id": repo_id,
        "revision": commit.oid,
        "private": info.private,
        "release_fingerprint": release["fingerprint"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--directory", type=Path, required=True)
    publish.add_argument("--repo", required=True)
    publish.add_argument("--replace-current", action="store_true")
    publish.add_argument("--expected-parent")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--collection", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--role", choices=["train", "eval"], default="train")
    fetch.add_argument("--include-gold", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_release(args.spec, args.output)
        print(
            json.dumps(
                {"fingerprint": result["fingerprint"], "collections": list(result["collections"])}
            )
        )
    elif args.command == "publish":
        print(
            json.dumps(
                publish_release(
                    args.directory,
                    args.repo,
                    replace_current=args.replace_current,
                    expected_parent=args.expected_parent,
                )
            )
        )
    else:
        result = fetch_collection(
            args.repo,
            args.revision,
            args.collection,
            args.output,
            expected_role=args.role,
            include_gold=args.include_gold,
        )
        print(json.dumps({"fingerprint": result.get("fingerprint"), "output": str(args.output)}))


if __name__ == "__main__":
    main()
