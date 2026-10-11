"""Full local release audit: native rows, real tokenizer, all patches and viewer parity."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.data.curation import token_profile
from bifocalgemma.data.offline_sft import editing_record, load_curated_pool, stable_fingerprint
from bifocalgemma.data.patches import git_environment
from bifocalgemma.data.releases import verify_release
from bifocalgemma.data.tokenization import load_verified_tokenizer


def emit(value):
    print(json.dumps(value), flush=True)


def validate_all_patches(records, batch_size=256):
    checked = 0
    for start in range(0, len(records), batch_size):
        group = records[start : start + batch_size]
        with tempfile.TemporaryDirectory(prefix="bifocal-v2-patch-audit-") as temporary:
            root = Path(temporary).resolve()
            patches = []
            for i, row in enumerate(group):
                aliases = {
                    "old_contents": row["before"],
                    "new_contents": row["after"],
                    "old_file": row["path"],
                    "new_file": row["path"],
                }
                if any(key in row and row[key] != value for key, value in aliases.items()):
                    raise ValueError("legacy_alias_mismatch")
                original = editing_record({**row, **aliases}, source="editpackft")
                for key in (
                    "before",
                    "after",
                    "instruction",
                    "path",
                    "target_patch",
                    "pair_hash",
                    "instruction_hash",
                ):
                    if original[key] != row[key]:
                        raise ValueError(
                            "canonical_patch_or_hash_mismatch:" + row["instance_id"] + ":" + key
                        )
                patch = row["target_patch"].splitlines(keepends=True)
                expected = [
                    f"diff --git a/{row['path']} b/{row['path']}\n",
                    f"--- a/{row['path']}\n",
                    f"+++ b/{row['path']}\n",
                ]
                if patch[:3] != expected:
                    raise ValueError("patch_header_mismatch")
                name = str(i) + ".py"
                (root / name).write_bytes(row["before"].encode("utf-8"))
                patches.append(
                    f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n"
                    + "".join(patch[3:])
                )
            payload = "".join(patches).encode("utf-8")
            command = ["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn"]
            for check in (True, False):
                result = subprocess.run(
                    command + (["--check"] if check else []) + ["-"],
                    cwd=root,
                    input=payload,
                    env=git_environment(),
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode:
                    raise ValueError(
                        f"git_patch_batch_failed:start={start}:"
                        + result.stderr.decode("utf-8", "replace")[:600]
                    )
            for i, row in enumerate(group):
                if (root / (str(i) + ".py")).read_bytes() != row["after"].encode("utf-8"):
                    raise ValueError("patch_after_mismatch:" + row["instance_id"])
            checked += len(group)
        if checked % (batch_size * 8) == 0:
            emit({"stage": "git_patch_roundtrip", "checked": checked})
    return checked


def quantiles(values):
    values = sorted(values)
    return {
        label: values[int((len(values) - 1) * fraction)]
        for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
    }


def audit(pool, release_dir, tokenizer_cache, *, reuse_validated=False):
    splits, manifest = load_curated_pool(pool)
    records = splits["train"] + splits["dev"]
    report = {
        "pool_fingerprint": manifest["fingerprint"],
        "counts": {k: len(v) for k, v in splits.items()},
        "sources": {},
        "licenses": dict(Counter(row["code_license"] for row in records)),
        "edit_kinds": dict(Counter(row["edit_kind"] for row in records)),
        "all_pairs_changed": all(row["before"] != row["after"] for row in records),
        "git_patch_roundtrips": 0,
    }
    if reuse_validated:
        proof = manifest["report"]["git_patch_validation"]
        identity = stable_fingerprint(
            [(row["instance_id"], row["target_patch"]) for row in records]
        )
        if (
            identity != proof["retained_identity"]
            or proof["passed"] != proof["checked"]
            or proof["retained_subset"] != len(records)
        ):
            raise ValueError("reused_git_proof_identity_mismatch")
        report["git_patch_roundtrips"] = len(records)
        report["git_validation_evidence"] = proof
    else:
        report["git_patch_roundtrips"] = validate_all_patches(records)
    for source in sorted({row["source_key"] for row in records}):
        parts = {
            split: [row for row in rows if row["source_key"] == source]
            for split, rows in splits.items()
        }
        report["sources"][source] = {
            split: {
                "records": len(rows),
                "supervised_tokens": sum(row["token_lengths"]["target"] for row in rows),
            }
            for split, rows in parts.items()
        }
    tokenizers, token_manifests = {}, {}
    for spec in EXPERIMENT_SPECS:
        for ref in (spec.encoder, spec.decoder):
            if ref and ref.model_id not in tokenizers:
                tokenizers[ref.model_id], token_manifests[ref.model_id] = load_verified_tokenizer(
                    ref, tokenizer_cache
                )
    # The E2B/E4B behavior assets must match before sharing profile results across six experiments.
    for suffix in ("", "-it"):
        a = token_manifests["google/gemma-4-E2B" + suffix]["files"]
        b = token_manifests["google/gemma-4-E4B" + suffix]["files"]
        for filename in (
            ("tokenizer.json",)
            if not suffix
            else ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        ):
            if a[filename] != b[filename]:
                raise ValueError("six_configuration_tokenizer_equivalence_failed")
    if reuse_validated:
        profile = manifest["report"]["tokens"]
        if profile["accepted"] != len(records) or profile.get("rejected"):
            raise ValueError("reused_token_profile_cohort_mismatch")
        for key, limit in (
            ("encoder", profile["limits"]["encoder"]),
            ("pair_decoder", profile["limits"]["pair_decoder"]),
            ("baseline_decoder", profile["limits"]["baseline_decoder"]),
        ):
            if any(row["token_lengths"][key] > limit for row in records):
                raise ValueError("reused_token_profile_overflow")
        report["tokenizer_profiles"] = profile
        report["tokenizer_profile_evidence"] = (
            "all_retained_rows_in_freshly_profiled_45303_candidates; unchanged_subset_after_compile_gate"
        )
    else:
        originals = {row["instance_id"]: dict(row["token_lengths"]) for row in records}
        limits = manifest["report"]["tokens"]["limits"]
        accepted, profile = token_profile(
            records,
            tokenizers["google/gemma-4-E2B"],
            tokenizers["google/gemma-4-E2B-it"],
            additional_encoder=tokenizers["google/gemma-4-E4B"],
            encoder_limit=limits["encoder"],
            decoder_limit=limits["pair_decoder"],
            baseline_limit=limits["baseline_decoder"],
            progress=emit,
        )
        if len(accepted) != len(records) or profile["rejected"]:
            raise ValueError("v2_token_budget_overflow")
        if any(row["token_lengths"] != originals[row["instance_id"]] for row in records):
            raise ValueError("original_token_lengths_changed")
        report["tokenizer_profiles"] = profile
    report["tokenizer_assets"] = token_manifests
    report["tokenizer_equivalence_six_configurations"] = True
    report["release_views"] = {}
    if release_dir:
        import pyarrow.parquet as pq

        release = verify_release(release_dir)
        root = Path(release_dir)
        for name, item in release["collections"].items():
            directory = root / "collections" / name / "view"
            native = {}
            if item["role"] == "train":
                native = {
                    split: {row["instance_id"]: row for row in rows}
                    for split, rows in splits.items()
                }
            else:
                with zipfile.ZipFile(root / item["archives"]["public"]["path"]) as archive:
                    filename = "inputs.jsonl" if item["kind"] == "gsm8k" else "eval_inputs.jsonl"
                    native["test"] = {
                        row["instance_id"]: row
                        for row in (
                            json.loads(line) for line in archive.read(filename).splitlines() if line
                        )
                    }
            for path in sorted(directory.glob("*.parquet")):
                expected = native[path.stem]
                seen_ids, seen_rows, seen_inputs = set(), set(), set()
                count = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        raw = json.loads(row["record_json"])
                        if raw != expected.get(row["instance_id"]):
                            raise ValueError("viewer_row_differs_from_native_archive")
                        key = hashlib.sha256(row["record_json"].encode()).hexdigest()
                        if row["instance_id"] in seen_ids or key in seen_rows:
                            raise ValueError("duplicate_view_row")
                        seen_ids.add(row["instance_id"])
                        seen_rows.add(key)
                        count += 1
                        if row["instance_id"] != raw["instance_id"]:
                            raise ValueError("view_native_identity")
                        if item["role"] == "train":
                            if (
                                row["target_patch"] != raw["target_patch"]
                                or row["before"] != raw["before"]
                            ):
                                raise ValueError("lossless_view_mismatch")
                            seen_inputs.add(
                                (row["instruction"], row["context"], row["target_patch"])
                            )
                        else:
                            if any(
                                k in raw
                                for k in (
                                    "answer",
                                    "gold_patch",
                                    "target_patch",
                                    "test_code",
                                    "reference",
                                    "solution",
                                )
                            ):
                                raise ValueError("gold_in_public_view")
                            if "question" in raw and (
                                row["question"] != raw["question"]
                                or row["shots"] != str(item["evidence"]["shots"])
                            ):
                                raise ValueError("gsm_protocol_view_mismatch")
                if seen_ids != set(expected):
                    raise ValueError("view_native_cohort_mismatch")
                if item["role"] == "train" and len(seen_inputs) != count:
                    raise ValueError("duplicate_model_input_target")
                report["release_views"][path.relative_to(root).as_posix()] = {
                    "records": count,
                    "duplicate_rows": 0,
                    "bytes": path.stat().st_size,
                    "record_json_matches_native": True,
                }
        report["release_fingerprint"] = release["fingerprint"]
        report["release_bytes"] = sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-validated-pool-audits",
        action="store_true",
        help="Verify frozen audit identity/coverage instead of repeating completed Git/token checks.",
    )
    args = parser.parse_args()
    report = audit(
        args.pool,
        args.release,
        args.tokenizer_cache,
        reuse_validated=args.reuse_validated_pool_audits,
    )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    emit({"stage": "complete", "counts": report["counts"], "output": str(args.output)})


if __name__ == "__main__":
    main()
