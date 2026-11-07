"""Profile raw Python repair targets without snapshots, execution or training admission."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.data.curation import file_digest
from bifocalgemma.data.repository import SOURCE_REVISIONS, read_source_rows
from bifocalgemma.data.tokenization import load_verified_tokenizer


def distribution(values):
    ordered = sorted(values)
    return (
        {
            "count": len(ordered),
            "total": sum(ordered),
            **{
                f"p{percent}": ordered[max(0, math.ceil(len(ordered) * percent / 100) - 1)]
                for percent in (50, 90, 95, 99)
            },
            "max": ordered[-1],
        }
        if ordered
        else {"count": 0}
    )


def profile(source, rows_file, tokenizer_cache):
    decoder, assets = load_verified_tokenizer(EXPERIMENT_SPECS[0].decoder, tokenizer_cache)
    counts = Counter()
    lengths = {"target": [], "pair_decoder": [], "issue": []}
    coverage = Counter()
    stream = hashlib.sha256()
    for offset, raw in enumerate(read_source_rows(rows_file)):
        counts["raw_rows"] += 1
        if str(raw.get("language") or "python").lower() not in {"python", "py"}:
            counts["non_python"] += 1
            continue
        counts["python_rows"] += 1
        issue, patch = raw.get("problem_statement"), raw.get("patch")
        if not all(isinstance(value, str) and value.strip() for value in (issue, patch)):
            counts["missing_issue_or_patch"] += 1
            continue
        assert isinstance(issue, str) and isinstance(patch, str)
        messages = [
            {
                "role": "user",
                "content": (
                    "Return only a unified diff fulfilling the instruction.\n\n"
                    + issue.strip()
                ),
            }
        ]
        prefix = decoder.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full = decoder.apply_chat_template(
            messages + [{"role": "assistant", "content": patch}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix_ids = decoder.encode(prefix, add_special_tokens=False)
        if not isinstance(full, str):
            raise ValueError("Single-conversation template must return text.")
        full_ids = decoder.encode(full.rstrip(), add_special_tokens=False)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            counts["prefix_target_boundary"] += 1
            continue
        target = full_ids[len(prefix_ids) :]
        if not target or target[-1] != 106 or prefix_ids[0] != 2:
            raise ValueError("Unexpected audited special-token boundary")
        row_lengths = {
            "target": len(target),
            "pair_decoder": len(full_ids) - 1,
            "issue": len(decoder.encode(issue.strip(), add_special_tokens=False)),
        }
        for key, value in row_lengths.items():
            lengths[key].append(value)
        for cap in (2048, 4096, 8192, 16384):
            if row_lengths["pair_decoder"] <= cap:
                coverage[str(cap)] += 1
        stream.update(
            json.dumps([offset, raw.get("instance_id"), row_lengths], separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )
        counts["profiled"] += 1
    return {
        "format": "raw_repository_target_profile_v1",
        "source": source,
        "source_reference": SOURCE_REVISIONS[source],
        "source_file_sha256": file_digest(rows_file),
        "tokenizer_assets": assets,
        "scope": "All raw Python rows with issue/patch; not training admission. No snapshot context, license, overlap, patch validity or functional validation is implied.",
        "decoder_policy": "Official patch-only IT, no thinking/tools; pair length includes issue prefix and closing turn, with one shift.",
        "language_policy": "Rows are selected by declared language; missing language defaults to Python as in the repository task adapter. This is not per-file Python verification.",
        "counts": dict(counts),
        "quantiles": {k: distribution(v) for k, v in lengths.items()},
        "pair_decoder_coverage": dict(coverage),
        "profile_stream_sha256": stream.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("swe_rebench_v2", "swe_fixer"), required=True)
    parser.add_argument("--rows-file", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite profiling evidence")
    result = profile(args.source, args.rows_file, args.tokenizer_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: result[key] for key in ("counts", "quantiles", "pair_decoder_coverage")})
    )


if __name__ == "__main__":
    main()
