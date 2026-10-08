"""Target-free deterministic lexical context windows shared by all six models."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from bifocalgemma.contracts import RepositorySnapshot


@dataclass(frozen=True)
class ContextWindow:
    path: str
    start_line: int
    end_line: int
    content: str
    score: float


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", text.lower())


def render_window(window: ContextWindow) -> str:
    return (
        f"### File: {window.path} (lines {window.start_line}-{window.end_line})\n{window.content}"
    )


def select_repository_context(
    snapshot: RepositorySnapshot,
    issue: str,
    tokenizer,
    *,
    max_tokens: int = 4096,
    window_lines: int = 80,
    max_windows: int = 12,
) -> tuple[str, dict]:
    """Only public issue/before text is accepted; gold paths/patch are never arguments."""
    if min(max_tokens, window_lines, max_windows) < 1 or not issue.strip():
        raise ValueError("invalid_context_selection_settings")
    query = Counter(_terms(issue))
    candidates = []
    seen = set()
    for file in sorted(snapshot.files, key=lambda f: f.path):
        if file.path in seen:
            raise ValueError("duplicate_snapshot_path")
        seen.add(file.path)
        # Do not render hidden tests/config secrets or generated cache directories.
        parts = file.path.lower().split("/")
        if any(
            p.startswith(".") or p in {"tests", "test", "node_modules", "vendor"} for p in parts
        ):
            continue
        if not file.path.endswith((".py", ".js", ".ts", ".md", ".toml", ".cfg", ".txt")):
            continue
        lines = file.content.splitlines(keepends=True)
        for start in range(0, len(lines), window_lines):
            text = "".join(lines[start : start + window_lines])
            candidates.append((file.path, start + 1, min(start + window_lines, len(lines)), text))
    if not candidates:
        raise ValueError("no_context_windows")
    term_counts = [Counter(_terms(path + "\n" + text)) for path, _, _, text in candidates]
    frequencies = Counter(term for count in term_counts for term in count)
    average_length = sum(sum(count.values()) for count in term_counts) / len(term_counts) or 1
    ranked = []
    for item, counts in zip(candidates, term_counts, strict=True):
        length = sum(counts.values())
        score = 0.0
        for term, query_count in query.items():
            freq = counts[term]
            idf = math.log(
                1 + (len(candidates) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
            )
            score += (
                query_count
                * idf
                * freq
                * 2.2
                / (freq + 1.2 * (0.25 + 0.75 * length / average_length))
            )
        if item[0].lower() in issue.lower():
            score += 5.0
        ranked.append(ContextWindow(*item, score))
    ranked.sort(key=lambda w: (-w.score, w.path, w.start_line))
    context = f"### Issue\n{issue}"
    if len(tokenizer.encode(context, add_special_tokens=True)) > max_tokens:
        raise ValueError("issue_exceeds_context_budget")
    selected = []
    skipped = 0
    for window in ranked:
        candidate = context + "\n\n" + render_window(window)
        if len(tokenizer.encode(candidate, add_special_tokens=True)) > max_tokens:
            skipped += 1
            continue
        selected.append(window)
        context = candidate
        if len(selected) == max_windows:
            break
    if not selected:
        raise ValueError("no_complete_window_fits_context_budget")
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
    return context, {
        "policy": "public_issue_bm25_windows_v1",
        "gold_used": False,
        "max_tokens": max_tokens,
        "actual_tokens": len(tokenizer.encode(context, add_special_tokens=True)),
        "window_lines": window_lines,
        "max_windows": max_windows,
        "skipped_over_budget": skipped,
        "context_sha256": digest,
        "windows": [asdict(window) | {"content": None} for window in selected],
    }


def canonical_snapshot_hash(snapshot: RepositorySnapshot) -> str:
    payload = [(f.path, f.content) for f in sorted(snapshot.files, key=lambda f: f.path)]
    if len({path for path, _ in payload}) != len(payload):
        raise ValueError("duplicate_snapshot_path")
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
