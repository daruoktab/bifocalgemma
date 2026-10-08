"""Text-only reader diagnostics; adapted protocol, never official benchmark scores."""

import re


def prepare_repoqa_tasks(dataset, tokenizer, *, max_context_tokens=4096):
    public, private = [], []
    for repo in dataset["python"]:
        parts = []
        offsets = {}
        for path, content in sorted(repo["content"].items()):
            header = f"\n### File: {path}\n"
            offsets[path] = sum(len(p) for p in parts) + len(header)
            parts.extend((header, content))
        full = "".join(parts)
        for index, needle in enumerate(repo["needles"]):
            source = repo["content"][needle["path"]].encode("utf-8")
            before = source[: needle["start_byte"]].decode("utf-8")
            reference = source[needle["start_byte"] : needle["end_byte"]].decode("utf-8")
            start = offsets[needle["path"]] + len(before)
            if full[start : start + len(reference)] != reference:
                raise ValueError("RepoQA source span mismatch.")
            instruction = (
                "Retrieve and repeat the exact function described below, returning only its code.\n"
                + needle["description"]
            )
            header = "### Request\n" + needle["description"] + "\n\n"
            context = reference
            if (
                len(tokenizer.encode(header + context, add_special_tokens=True))
                > max_context_tokens
            ):
                raise ValueError("RepoQA complete needle exceeds explicit context budget.")
            if not reference.strip():
                raise ValueError("RepoQA empty needle.")
            left_active = right_active = True
            left = full[:start].splitlines(keepends=True)
            right = full[start + len(reference) :].splitlines(keepends=True)
            # Explicit, line-aligned construction; task answer is intentionally in context.
            for distance in range(max(len(left), len(right))):
                if left_active and distance < len(left):
                    candidate = left[-1 - distance] + context
                    if (
                        len(tokenizer.encode(header + candidate, add_special_tokens=True))
                        <= max_context_tokens
                    ):
                        context = candidate
                    else:
                        left_active = False
                else:
                    left_active = False
                if right_active and distance < len(right):
                    candidate = context + right[distance]
                    if (
                        len(tokenizer.encode(header + candidate, add_special_tokens=True))
                        <= max_context_tokens
                    ):
                        context = candidate
                    else:
                        right_active = False
                else:
                    right_active = False
                if not left_active and not right_active:
                    break
                if (
                    len(tokenizer.encode(header + context, add_special_tokens=True))
                    >= max_context_tokens - 32
                ):
                    break
                # Stop at the nearest boundaries; never jump over omitted lines.
                if distance >= 2000:
                    break
            identifier = f"python::{repo['repo']}::{needle['name']}"
            public.append(
                {
                    "instance_id": identifier,
                    "source": "evalplus/repoqa",
                    "source_revision": "2024-06-23",
                    "role": "eval",
                    "task_mode": "function_retrieval",
                    "instruction": instruction,
                    "context": header + context,
                    "context_policy": "adapted_Gemma_budget_lexicographic_line_windows_v1",
                    "official_protocol": False,
                }
            )
            private.append(
                {
                    "instance_id": identifier,
                    "task_mode": "function_retrieval",
                    "reference": reference,
                }
            )
    return public, private


def prepare_crosscode_tasks(rows, tokenizer, *, max_context_tokens=4096):
    public, private = [], []
    for row in rows:
        cross = row.get("crossfile_context", {}).get("text", "")
        context = cross + "\n" + row["prompt"]
        if len(tokenizer.encode(context, add_special_tokens=True)) > max_context_tokens:
            raise ValueError("CrossCodeEval input overflow; no silent truncation.")
        identifier = str(row["metadata"]["task_id"])
        public.append(
            {
                "instance_id": identifier,
                "source": "amazon-science/cceval",
                "source_revision": "caller_pinned_preprocessed_file",
                "role": "eval",
                "task_mode": "line_completion",
                "instruction": "Complete the next code statement after the supplied prefix. Return only the continuation, without repeating the prefix.",
                "context": context,
                "context_policy": "precomputed_non_reference_retrieval_no_truncation",
                "official_protocol": False,
            }
        )
        private.append(
            {
                "instance_id": identifier,
                "task_mode": "line_completion",
                "reference": row["groundtruth"],
            }
        )
    return public, private


def normalized_code(text):
    text = text.strip()
    fence = re.fullmatch(r"```(?:python|py|text)?\s*\n(.*?)\n```", text, re.S)
    if fence:
        text = fence.group(1)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def score_diagnostics(specs, predictions):
    if len({r["instance_id"] for r in specs}) != len(specs):
        raise ValueError("Duplicate diagnostic tasks.")
    if len({r["instance_id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate diagnostic predictions.")
    expected = {r["instance_id"] for r in specs}
    by_id = {r["instance_id"]: r.get("text", "") for r in predictions}
    if set(by_id) - expected:
        raise ValueError("Unknown diagnostic prediction IDs.")
    results = [
        {
            "instance_id": r["instance_id"],
            "present": r["instance_id"] in by_id,
            "exact_match": r["instance_id"] in by_id
            and normalized_code(by_id[r["instance_id"]]) == normalized_code(r["reference"]),
        }
        for r in specs
    ]
    return {
        "metric": "adapted_exact_code_match",
        "official_benchmark_score": False,
        "attempted": len(specs),
        "matched": sum(r["exact_match"] for r in results),
        "exact_match_rate": sum(r["exact_match"] for r in results) / len(specs) if specs else None,
        "missing_predictions": sum(not r["present"] for r in results),
        "results": results,
        "limitations": "No RepoQA syntax-similarity threshold, CrossCodeEval statement/parser/comment normalization or identifier metrics. Use upstream scorer for published-score comparisons.",
    }
