"""V2 remediation of a frozen auxiliary editing corpus; no model/code execution."""

import ast
import copy
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from bifocalgemma.data.curation import code_signature, export_pool, split_connected_records
from bifocalgemma.data.offline_sft import editing_record, load_curated_pool, stable_fingerprint
from bifocalgemma.data.python_quality import summarize_admitted_warning_records


class WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    if isinstance(first.value.value, str):
                        node.body = node.body[1:]
        return node


@lru_cache(maxsize=4096)
def inspect_structure(code):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            node = ast.parse(code, type_comments=True)
            compile_error = None
            try:
                compile(node, "<dataset-static-audit>", "exec", dont_inherit=True)
            except (SyntaxError, ValueError, RecursionError) as error:
                compile_error = str(error)
        full = stable_fingerprint(ast.dump(node, include_attributes=False))
        stripped = WithoutDocstrings().visit(node)
        body = bool(stripped.body)
        core = stable_fingerprint(ast.dump(stripped, include_attributes=False))
        return {
            "full": full,
            "core": core,
            "has_body": body,
            "compile_error": compile_error,
            "warnings": sum(issubclass(w.category, SyntaxWarning) for w in caught),
        }
    except (SyntaxError, ValueError, RecursionError):
        return None


def event_key(row):
    if row["source_key"] == "nextcoder":
        return ("synthetic", row["source"], row["commit"], row["path"], row["instruction"])
    return ("git", row["commit"], row["path"])


def preference(row):
    # CommitPack retains the upstream export's module documentation; all winners
    # remain static exports, never falsely marked as verified Git checkouts.
    return (
        row["source_key"] != "commitpackft_python",
        -len(row["before"]) - len(row["after"]),
        row["instance_id"],
    )


def merge_lineage(winner, group):
    """Keep discarded export identities available to splitting and audit."""
    aliases = []
    for row in group:
        aliases.extend(row.get("duplicate_source_records", []))
        if row is not winner:
            aliases.append(
                {
                    key: row.get(key)
                    for key in (
                        "instance_id",
                        "source",
                        "upstream_row",
                        "commit",
                        "canonical_repos",
                    )
                }
            )
    winner["duplicate_source_records"] = sorted(aliases, key=lambda x: x["instance_id"])
    winner["canonical_repos"] = sorted(
        {repo for row in group for repo in row.get("canonical_repos", [])}
    )
    winner["duplicate_commit_keys"] = sorted(
        {row["commit"] for row in group}
        | {item["commit"] for item in aliases if item.get("commit")}
    )


def remediate_records(records, *, progress=None):
    ledger = []
    counts = Counter()
    structures = {}
    events = defaultdict(list)

    def reject(row, reason, retained=None):
        counts[reason] += 1
        ledger.append(
            {
                "instance_id": row["instance_id"],
                "source": row["source"],
                "reason": reason,
                "retained_id": retained,
            }
        )

    for index, incoming in enumerate(records):
        if progress and index % 2000 == 0:
            progress({"stage": "remediation", "examined": index, "total": len(records)})
        row = copy.deepcopy(incoming)
        if row.get("role") != "train":
            raise ValueError("evaluation_record_cannot_enter_training")
        before, after = inspect_structure(row["before"]), inspect_structure(row["after"])
        if after is None:
            reject(row, "after_invalid_python")
            continue
        if after["compile_error"]:
            reject(row, "after_compile_error")
            continue
        if after["warnings"]:
            reject(row, "after_syntax_warning")
            continue
        if before and before["core"] == after["core"]:
            reject(row, "documentation_or_format_only")
            continue
        structures[row["instance_id"]] = (before, after)
        events[event_key(row)].append(row)

    winners = []
    for group in events.values():
        pairs = {
            (
                structures[r["instance_id"]][0]["core"]
                if structures[r["instance_id"]][0]
                else code_signature(r["before"]),
                structures[r["instance_id"]][1]["core"],
            )
            for r in group
        }
        if len(group) > 1 and len(pairs) != 1:
            for row in group:
                reject(row, "ambiguous_git_event_variants")
            continue
        winner = sorted(group, key=preference)[0]
        merge_lineage(winner, group)
        for row in group:
            if row is not winner:
                reject(row, "duplicate_git_or_seed_event", winner["instance_id"])
        winners.append(winner)

    # Equal visible inputs with different code targets are quarantined rather
    # than silently choosing a supposedly correct alternative without tests.
    inputs = defaultdict(list)
    for row in winners:
        inputs[stable_fingerprint((row["instruction"], row["path"], row["before"]))].append(row)
    coherent = []
    for group in inputs.values():
        outputs = {structures[r["instance_id"]][1]["core"] for r in group}
        if len(outputs) > 1:
            for row in group:
                reject(row, "ambiguous_visible_input")
            continue
        coherent.extend(group)

    equivalents = defaultdict(list)
    for row in coherent:
        before, after = structures[row["instance_id"]]
        key = (before["core"] if before else code_signature(row["before"]), after["core"])
        equivalents[key].append(row)
    selected = []
    for group in equivalents.values():
        winner = sorted(group, key=preference)[0]
        merge_lineage(winner, group)
        for row in group:
            if row is not winner:
                reject(row, "duplicate_structural_edit", winner["instance_id"])
        selected.append(winner)

    for row in selected:
        before, after = structures[row["instance_id"]]
        row["before_signature"] = code_signature(row["before"])
        row["normalized_pair_hash"] = stable_fingerprint(
            (row["before_signature"], code_signature(row["after"]))
        )
        row["structural_edit_hash"] = stable_fingerprint(
            (before["core"] if before else row["before_signature"], after["core"])
        )
        row["split_keys"] = [
            "alias_commit:" + value for value in row.get("duplicate_commit_keys", [])
        ]
        for label, item in (("before_ast", before), ("after_ast", after)):
            if item and item["has_body"]:
                row["split_keys"].append("code_ast:" + item["core"])
        row["provenance_type"] = (
            "synthetic_edit_from_unresolved_seed"
            if row["source_key"] == "nextcoder"
            else "upstream_git_export"
        )
        row["git_commit"] = None if row["source_key"] == "nextcoder" else row["commit"]
        row["seed_program_hash"] = row["commit"] if row["source_key"] == "nextcoder" else None
        row["commit_kind"] = (
            "synthetic_seed_hash_legacy_group_key"
            if row["source_key"] == "nextcoder"
            else "git_commit"
        )
        row["code_license_verification"] = (
            "unverified_original_seed"
            if row["source_key"] == "nextcoder"
            else "upstream_row_declared_not_checkout_verified"
        )
        if row["source_key"] == "nextcoder":
            row["code_license"] = "unknown"
            row["license"] = "unknown"
        row["dataset_schema_version"] = 2
        row["research_role"] = "auxiliary_single_file_editing"
        row["task_scope"] = "single_file_context"
        row["edit_kind"] = (
            "syntax_adaptation_candidate"
            if before is None or before["compile_error"]
            else "substantive_edit_candidate"
        )
        row["quality_gate"] = "ast_compile_warning_free_git_patch_static_not_functional_validation"
        row["functional_quality_verified"] = False
        row["normalization_policy"] = "python_structure_preserving_v2"
    return selected, ledger, dict(counts)


def audit_rows(split_rows):
    keys = {split: defaultdict(set) for split in split_rows}
    all_ids = set()
    counters = Counter()
    for split, rows in split_rows.items():
        local = defaultdict(set)
        for row in rows:
            identifier = row["instance_id"]
            if identifier in all_ids:
                raise ValueError("duplicate_instance_id")
            all_ids.add(identifier)
            before, after = inspect_structure(row["before"]), inspect_structure(row["after"])
            if after is None or after["warnings"] or after["compile_error"]:
                raise ValueError("invalid_or_warning_target")
            if before and before["core"] == after["core"]:
                raise ValueError("non_substantive_edit")
            values = {
                "event": stable_fingerprint(event_key(row)),
                "structural_edit": row["structural_edit_hash"],
                "visible_input": stable_fingerprint(
                    (row["instruction"], row["path"], row["before"])
                ),
                "input_target": stable_fingerprint(
                    (row["instruction"], row["path"], row["before"], row["target_patch"])
                ),
            }
            for name, key in values.items():
                if key in local[name]:
                    raise ValueError("within_split_duplicate:" + name)
                local[name].add(key)
                keys[split][name].add(key)
            for value in row.get("canonical_repos", []):
                keys[split]["repo"].add(value)
            keys[split]["commit_or_seed"].add(row["commit"])
            keys[split]["before_signature"].add(row["before_signature"])
            for key in row["split_keys"]:
                keys[split]["structural_code"].add(key)
            if row["role"] != "train" or row["pool_split"] != split:
                raise ValueError("invalid_role")
            counters[row["source_key"]] += 1
    intersections = {key: len(keys["train"][key] & keys["dev"][key]) for key in keys["train"]}
    if any(intersections.values()):
        raise ValueError("cross_split_overlap:" + str(intersections))
    return {
        "cross_split_overlap": intersections,
        "source_counts": dict(counters),
        "unique_ids": len(all_ids),
        "within_split_duplicate_events": 0,
        "within_split_duplicate_structural_edits": 0,
        "ambiguous_visible_inputs": 0,
    }


def refresh_patch_targets(records):
    """Preserve source bytes; regenerate diffs using Git's LF-only line counting."""
    changed = []
    for row in records:
        corrected = editing_record(
            {
                **row,
                "old_contents": row["before"],
                "new_contents": row["after"],
                "old_file": row["path"],
                "new_file": row["path"],
            },
            source="editpackft",
        )["target_patch"]
        if corrected != row["target_patch"]:
            changed.append(
                {
                    "instance_id": row["instance_id"],
                    "old_sha256": hashlib.sha256(row["target_patch"].encode()).hexdigest(),
                    "new_sha256": hashlib.sha256(corrected.encode()).hexdigest(),
                }
            )
            row["target_patch"] = corrected
        row["patch_line_policy"] = "git_lf_delimiters_v2"
    return changed


def retokenize_records(records, tokenizer_cache, limits, *, progress=None):
    """Verify all four asset bundles before profiling the common six-model cohort."""
    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.data.curation import token_profile
    from bifocalgemma.data.tokenization import load_verified_tokenizer

    assets, manifests = {}, {}
    for spec in EXPERIMENT_SPECS:
        for ref in (spec.encoder, spec.decoder):
            if ref and ref.model_id not in assets:
                assets[ref.model_id], manifests[ref.model_id] = load_verified_tokenizer(
                    ref, Path(tokenizer_cache)
                )
    for suffix in ("", "-it"):
        left = manifests["google/gemma-4-E2B" + suffix]["files"]
        right = manifests["google/gemma-4-E4B" + suffix]["files"]
        names = (
            ("tokenizer.json",)
            if not suffix
            else ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        )
        if any(left[name] != right[name] for name in names):
            raise ValueError("six_configuration_tokenizer_equivalence_failed")
    return token_profile(
        records,
        assets["google/gemma-4-E2B"],
        assets["google/gemma-4-E2B-it"],
        additional_encoder=assets["google/gemma-4-E4B"],
        encoder_limit=limits["encoder"],
        decoder_limit=limits["pair_decoder"],
        baseline_limit=limits["baseline_decoder"],
        progress=progress,
    )


def build_remediated_pool(source, output, *, progress=None, tokenizer_cache=None):
    rows, old = load_curated_pool(source)
    selected, ledger, rejected = remediate_records(rows["train"] + rows["dev"], progress=progress)
    patch_changes = refresh_patch_targets(selected)
    fresh_profile = None
    if patch_changes:
        if tokenizer_cache is None:
            raise ValueError("corrected_patches_require_verified_tokenizer_cache")
        selected, fresh_profile = retokenize_records(
            selected, tokenizer_cache, old["report"]["tokens"]["limits"], progress=progress
        )
        for item in fresh_profile["rejected"]:
            ledger.append({**item, "reason": "corrected_patch_token_overflow"})
        rejected["corrected_patch_token_overflow"] = len(fresh_profile["rejected"])
    if progress:
        progress({"stage": "split_and_full_audit", "retained": len(selected)})
    train, dev, split = split_connected_records(selected)
    audit = audit_rows({"train": train, "dev": dev})
    report = copy.deepcopy(old["report"])
    # V1 admission/profile reports are retained as historical evidence, not
    # falsely rebranded as fresh upstream admission for the corrected cohort.
    report["parent_pool_fingerprint"] = old["fingerprint"]
    report["research_role"] = "auxiliary_single_file_editing"
    report["remediation"] = {
        "policy": "event_and_structure_remediation_v2",
        "input_records": sum(map(len, rows.values())),
        "retained_records": len(selected),
        "rejected": rejected,
        "audit": audit,
        "functional_validation": "not_performed",
        "source_license_policy": "unknown_synthetic_seed_explicit_private_only",
    }
    report["split"] = split
    report["split"]["policy"] = "repo_commit_seed_and_structural_code_components_v2"
    report["training_plan"] = {
        "epochs": 1,
        "batch_size": 1,
        "accumulation": 4,
        "updates_full_epoch": math.ceil(len(train) / 4),
        "supervised_train_tokens": sum(r["token_lengths"]["target"] for r in train),
        "supervised_dev_tokens": sum(r["token_lengths"]["target"] for r in dev),
    }
    parent_tokens = report["tokens"]
    quantiles = {}
    for field in selected[0]["token_lengths"]:
        values = sorted(row["token_lengths"][field] for row in selected)
        quantiles[field] = {
            label: values[int((len(values) - 1) * fraction)]
            for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
        }
    report["tokens"] = {
        "limits": parent_tokens["limits"],
        "profiled": len(selected),
        "accepted": len(selected),
        "rejected": [],
        "quantiles_after_remediation": quantiles,
        "supervised_tokens": sum(r["token_lengths"]["target"] for r in selected),
        "length_measurement": "inherited_verified_token_lengths_of_byte_unchanged_inputs_targets",
        "parent_profile_before_remediation": parent_tokens,
    }
    report["parent_admission_before_remediation"] = report.pop("admission", None)
    report["parent_final_cohort_syntax_warnings"] = report.pop("final_cohort_syntax_warnings", None)
    report["final_cohort_syntax_warnings"] = {
        split: summarize_admitted_warning_records(items)
        for split, items in (("train", train), ("dev", dev))
    }
    report["syntax_diagnostics_policy"] = "v2_ast_and_compile_warning_free_targets"
    report["admitted_target_syntax_warnings"] = {"records": 0, "warnings": 0}
    if fresh_profile is not None:
        fresh_profile["quantiles_after_remediation"] = fresh_profile.pop("quantiles_before_limit")
        fresh_profile["length_measurement"] = "fresh_offline_real_gemma_tokenizers_all_records"
        report["tokens"] = fresh_profile
    report["patch_line_remediation"] = {
        "policy": "git_lf_delimiters_v2",
        "regenerated": len(selected),
        "changed": patch_changes,
    }
    manifest = export_pool(train + dev, output, report)
    Path(output, "remediation_ledger.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in ledger), encoding="utf-8"
    )
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args()
    result = build_remediated_pool(
        args.source,
        args.output,
        progress=lambda value: print(json.dumps(value), flush=True),
        tokenizer_cache=args.tokenizer_cache,
    )
    print(
        json.dumps(
            {
                "fingerprint": result["fingerprint"],
                "remediation": result["report"]["remediation"],
                "split": result["report"]["split"],
            }
        )
    )


if __name__ == "__main__":
    main()
