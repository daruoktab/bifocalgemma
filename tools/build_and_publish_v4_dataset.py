"""Fetch vetted permissive v4 sources, build mixed_context_enc32k_dec8k_v4, verify, and publish."""

# ruff: noqa: E402 -- standalone data preparation and release script

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("USE_TORCH", "0")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from audit_context_release import audit_views
from prepare_context_data import (
    BASELINE_LIMIT,
    DECODER_LIMIT,
    ENCODER_LIMIT,
    _collect_benchmark_quarantine_assets,
    mixed_pool_v4,
)

from bifocalgemma.config import EXPERIMENT_SPECS, V4_BENCHMARKS
from bifocalgemma.data.context_tasks import (
    V4_DATASET_REGISTRY,
    is_permissive_license,
)
from bifocalgemma.data.curation import export_pool, file_digest
from bifocalgemma.data.curriculum import classify_v4_bucket
from bifocalgemma.data.offline_sft import load_curated_pool
from bifocalgemma.data.releases import build_release, publish_release, verify_release
from bifocalgemma.data.repository_corpus import ArchiveSnapshotStore
from bifocalgemma.data.tokenization import load_verified_tokenizer
from bifocalgemma.evaluation.external_runner import _build_reponiah_cohort
from bifocalgemma.training.repository_sft import tokenize_repository_record
from bifocalgemma.training.stages import atomic_json

V4_FETCH_DIR = ROOT / "outputs/v4_fetched_sources"
V4_CONTEXT_POOL_FULL = ROOT / "outputs/context_sft_enc32k_dec8k_v4_full"
V4_CONTEXT_POOL = ROOT / "outputs/context_sft_enc32k_dec8k_v4_local"
V4_FOUNDATION_POOL = ROOT / "outputs/instruction_enc8k_dec2k_v2_ready"
V4_BENCHMARK_ROOT = ROOT / "outputs/context_benchmarks_v1"
V4_REPONIAH_DIR = ROOT / "outputs/context_benchmarks_v1/reponiah_gemma32k_dec8k_v1"
V4_TOKENIZERS = ROOT / "outputs/context_tokenizers"
V4_MIXED_POOL = ROOT / "outputs/mixed_context_enc32k_dec8k_v4"
V4_RELEASE_DIR = ROOT / "outputs/dataset_release_context32k_v4"
V4_SPEC_PATH = ROOT / "configs/dataset_release.context32k.v4.json"
V4_PUBLICATION_RECEIPT = ROOT / "outputs/context_release_publication_v4.json"
V4_NOTICE_DIR = ROOT / "outputs/v4_dataset_notices"
HF_DATASET_REPO = "daruokta/bifocalgemma-data"
EXPECTED_PARENT_REVISION = "7972f95d7c87f163c8b5af6a4acd308c0c1d0e23"


def _log(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _load_gemma4_tokenizers() -> tuple[Any, Any, Any, Any]:
    paired_spec = EXPERIMENT_SPECS[0]
    baseline_spec = EXPERIMENT_SPECS[4]
    assert paired_spec.encoder is not None
    encoder, _ = load_verified_tokenizer(paired_spec.encoder, V4_TOKENIZERS)
    decoder, _ = load_verified_tokenizer(paired_spec.decoder, V4_TOKENIZERS)
    return paired_spec, baseline_spec, encoder, decoder


def _measure_exact_tokens(
    row: dict[str, Any],
    paired_spec: Any,
    baseline_spec: Any,
    encoder: Any,
    decoder: Any,
) -> dict[str, int] | None:
    """Validate row against both paired and baseline limits with real Gemma-4 tokenizers."""
    enriched = {
        **row,
        "role": "train",
        "record_type": "context_sft_v1",
        "context_format": "v4",
        "disjoint_context": True,
    }
    try:
        paired_item = tokenize_repository_record(
            enriched,
            paired_spec,
            encoder,
            decoder,
            encoder_limit=ENCODER_LIMIT,
            decoder_limit=DECODER_LIMIT,
        )
        base_item = tokenize_repository_record(
            enriched,
            baseline_spec,
            None,
            decoder,
            encoder_limit=ENCODER_LIMIT,
            decoder_limit=BASELINE_LIMIT,
        )
    except Exception:
        return None
    return {
        "encoder": len(paired_item.encoder_ids),
        "pair_decoder": len(paired_item.decoder_prefix_ids) + len(paired_item.target_ids) - 1,
        "baseline_decoder": len(base_item.decoder_prefix_ids) + len(base_item.target_ids) - 1,
        "target": len(paired_item.target_ids),
    }


def _split_long_user_prompt(user_text: str) -> tuple[str, str]:
    """Split a long single-turn prompt into (instruction, context) without dropping content."""
    text = user_text.strip()
    if len(text) <= 2048:
        return text, text
    if "\n\n" in text:
        head, _, tail = text.rpartition("\n\n")
        if 12 <= len(tail.strip()) <= 2000 and len(head.strip()) >= 256:
            return tail.strip(), head.strip()
    if "\n" in text:
        head, _, tail = text.rpartition("\n")
        if 12 <= len(tail.strip()) <= 2000 and len(head.strip()) >= 256:
            return tail.strip(), head.strip()
    return (
        "Read the provided long context carefully and answer the request embedded in it.",
        text,
    )


def _extract_python_fim_from_code(
    code: str,
    *,
    file_path: str,
) -> tuple[str, str, str] | None:
    """Mask one real Python function body (4..90 lines) into a leak-free # <MISSING_CODE> task."""
    if not code or "# <MISSING_CODE>" in code or len(code) > 140_000:
        return None
    normalized = code.replace("\r\n", "\n")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(normalized)
    except SyntaxError:
        return None
    lines = normalized.splitlines()
    candidates: list[tuple[int, int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            body_start = node.body[0].lineno
            body_end = getattr(node, "end_lineno", None)
            if not isinstance(body_end, int):
                continue
            body_len = body_end - body_start + 1
            if 4 <= body_len <= 90 and body_start > node.lineno:
                candidates.append((body_len, body_start, body_end, node.name))
    if not candidates:
        return None
    # Prefer medium-substantial functions near the middle of the file.
    candidates.sort(key=lambda item: (abs(item[0] - 18), abs(item[1] - len(lines) // 2)))
    for _, body_start, body_end, func_name in candidates[:6]:
        target_lines = lines[body_start - 1 : body_end]
        target = "\n".join(target_lines)
        if not target.strip() or "# <MISSING_CODE>" in target:
            continue
        first_non_empty = next((ln for ln in target_lines if ln.strip()), target_lines[0])
        indent = first_non_empty[: len(first_non_empty) - len(first_non_empty.lstrip(" \t"))]
        masked_lines = [
            *lines[: body_start - 1],
            f"{indent}# <MISSING_CODE>",
            *lines[body_end:],
        ]
        masked_code = "\n".join(masked_lines)
        if target in masked_code or masked_code.count("# <MISSING_CODE>") != 1:
            continue
        instruction = f"Complete the missing Python function body for `{func_name}` in `{file_path}`."
        context = f"### Instruction\n{instruction}\n\n### File: {file_path}\n{masked_code}"
        if target in context or context.count("# <MISSING_CODE>") != 1:
            continue
        return instruction, context, target
    return None


def _write_fetched_source(
    source_key: str,
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> Path:
    V4_FETCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = V4_FETCH_DIR / f"{source_key}.jsonl"
    tmp_path = V4_FETCH_DIR / f".{source_key}.jsonl.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
    atomic_json(
        V4_FETCH_DIR / f"{source_key}.done.json",
        {
            "source_key": source_key,
            "count": len(rows),
            "long_ge_8k": sum(1 for r in rows if r.get("encoder_tokens", 0) >= 8192),
            "sha256": file_digest(out_path),
            **meta,
        },
    )
    return out_path


def _source_ready(source_key: str) -> bool:
    return (V4_FETCH_DIR / f"{source_key}.jsonl").is_file() and (
        V4_FETCH_DIR / f"{source_key}.done.json"
    ).is_file()


def _probe_and_append(
    rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    source_key: str,
    paired_spec: Any,
    baseline_spec: Any,
    encoder: Any,
    decoder: Any,
    *,
    min_enc_tokens: int = 0,
    max_enc_tokens: int = ENCODER_LIMIT,
    max_target_tokens: int = 4096,
) -> bool:
    entry = V4_DATASET_REGISTRY[source_key]
    probe_row = {
        "instance_id": candidate.get("instance_id", f"probe-{source_key}-{len(rows)}"),
        "source": entry["dataset_id"],
        "source_key": source_key,
        "task_type": candidate["task_type"],
        "instruction": candidate["instruction"],
        "context": candidate["context"],
        "target_text": candidate["target_text"],
    }
    lengths = _measure_exact_tokens(probe_row, paired_spec, baseline_spec, encoder, decoder)
    if lengths is None:
        return False
    if (
        lengths["encoder"] < min_enc_tokens
        or lengths["encoder"] > max_enc_tokens
        or lengths["target"] > max_target_tokens
    ):
        return False
    candidate["encoder_tokens"] = lengths["encoder"]
    candidate["token_lengths"] = lengths
    rows.append(candidate)
    return True


def _stream_hf_jsonl(repo_id: str, filename: str):
    """Stream a remote Hugging Face dataset JSONL file line-by-line with constant O(1) memory."""
    import requests
    from huggingface_hub import get_token, hf_hub_url

    url = hf_hub_url(repo_id, filename, repo_type="dataset")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                yield json.loads(raw_line)
            except json.JSONDecodeError:
                continue


def fetch_all_v4_sources(*, force: bool = False) -> dict[str, Path]:
    """Stream, filter, and exact-tokenize every vetted HF source into outputs/v4_fetched_sources."""
    from datasets import load_dataset

    paired_spec, baseline_spec, encoder, decoder = _load_gemma4_tokenizers()
    paths: dict[str, Path] = {}

    # 1. opc_sft_stage2 (MIT, repair_loc bucket, task_type="instruction")
    if force or not _source_ready("opc_sft_stage2"):
        rows: list[dict[str, Any]] = []
        configs = [
            ("evol_instruct", 2500),
            ("package_instruct", 2500),
            ("educational_instruct", 1500),
        ]
        for cfg_name, limit in configs:
            ds = load_dataset(
                "OpenCoder-LLM/opc-sft-stage2", cfg_name, split="train", streaming=True
            )
            taken = 0
            for idx, item in enumerate(ds):
                if taken >= limit or idx >= limit * 4:
                    break
                instr = str(item.get("instruction") or "").strip()
                out = str(item.get("output") or "").strip()
                if len(instr) < 16 or len(out) < 16 or len(instr) > 24000 or len(out) > 16000:
                    continue
                cand = {
                    "instance_id": f"v4-opc2-{cfg_name}-{item.get('seq_id', idx)}",
                    "task_type": "instruction",
                    "instruction": instr,
                    "context": instr,
                    "target_text": out,
                    "license": "mit",
                }
                if _probe_and_append(
                    rows, cand, "opc_sft_stage2", paired_spec, baseline_spec, encoder, decoder
                ):
                    taken += 1
            _log({"stage": "fetch", "source": "opc_sft_stage2", "config": cfg_name, "taken": taken})
        paths["opc_sft_stage2"] = _write_fetched_source(
            "opc_sft_stage2", rows, {"dataset_id": "OpenCoder-LLM/opc-sft-stage2", "license": "mit"}
        )
    else:
        paths["opc_sft_stage2"] = V4_FETCH_DIR / "opc_sft_stage2.jsonl"

    # 2. self_oss_instruct (ODC-BY, repair_loc bucket, task_type="instruction")
    if force or not _source_ready("self_oss_instruct"):
        rows = []
        ds = load_dataset(
            "bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train", streaming=True
        )
        for idx, item in enumerate(ds):
            if len(rows) >= 6000 or idx >= 24000:
                break
            instr = str(item.get("instruction") or "").strip()
            resp = str(item.get("response") or "").strip()
            if len(instr) < 16 or len(resp) < 16 or len(instr) > 24000 or len(resp) > 16000:
                continue
            cand = {
                "instance_id": f"v4-selfoss-{idx}",
                "task_type": "instruction",
                "instruction": instr,
                "context": instr,
                "target_text": resp,
                "license": "odc-by",
            }
            _probe_and_append(
                rows, cand, "self_oss_instruct", paired_spec, baseline_spec, encoder, decoder
            )
        _log({"stage": "fetch", "source": "self_oss_instruct", "taken": len(rows)})
        paths["self_oss_instruct"] = _write_fetched_source(
            "self_oss_instruct",
            rows,
            {
                "dataset_id": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
                "license": "odc-by",
            },
        )
    else:
        paths["self_oss_instruct"] = V4_FETCH_DIR / "self_oss_instruct.jsonl"

    # 3. high_coder (MIT, Python AST function-body FIM -> fim_completion bucket)
    if force or not _source_ready("high_coder"):
        long_rows: list[dict[str, Any]] = []
        med_rows: list[dict[str, Any]] = []
        target_long = 5500
        target_med = 4500
        ds = _stream_hf_jsonl("Crownelius/High-Coder-SFT-Medium", "dataset.jsonl")
        scanned = 0
        for idx, item in enumerate(ds):
            scanned = idx + 1
            if len(long_rows) >= target_long and len(med_rows) >= target_med:
                break
            if scanned >= 180_000:
                break
            if str(item.get("language") or "").lower() != "python":
                continue
            lic_obj = item.get("licensing")
            if not isinstance(lic_obj, dict) or lic_obj.get("redistribution_allowed") is not True:
                continue
            lic = str(lic_obj.get("license_spdx") or "")
            if not is_permissive_license(lic, code_only=True):
                continue
            content_obj = item.get("content")
            code = str(content_obj.get("text") or "") if isinstance(content_obj, dict) else ""
            # Fast char filter: need at least ~6k chars for medium, ~24k chars for >=8k long
            if len(code) < 5500 or len(code) > 130_000:
                continue
            if len(med_rows) >= target_med and len(code) < 22_000:
                continue
            sample_id = str(item.get("sample_id") or idx)
            file_path = f"src/module_{sample_id[:12]}.py"
            extracted = _extract_python_fim_from_code(code, file_path=file_path)
            if extracted is None:
                continue
            instr, ctx, target = extracted
            cand: dict[str, Any] = {
                "instance_id": f"v4-highcoder-{sample_id}",
                "task_type": "completion",
                "instruction": instr,
                "context": ctx,
                "target_text": target,
                "path": file_path,
                "license": lic,
            }
            bucket_list: list[dict[str, Any]] = []
            if _probe_and_append(
                bucket_list,
                cand,
                "high_coder",
                paired_spec,
                baseline_spec,
                encoder,
                decoder,
                min_enc_tokens=1500,
                max_enc_tokens=ENCODER_LIMIT,
                max_target_tokens=3072,
            ):
                accepted = bucket_list[0]
                if accepted["encoder_tokens"] >= 8192:
                    if len(long_rows) < target_long:
                        long_rows.append(accepted)
                elif len(med_rows) < target_med:
                    med_rows.append(accepted)
            if scanned % 10000 == 0:
                _log(
                    {
                        "stage": "fetch_progress",
                        "source": "high_coder",
                        "scanned": scanned,
                        "long": len(long_rows),
                        "med": len(med_rows),
                    }
                )
        rows = long_rows + med_rows
        _log(
            {
                "stage": "fetch",
                "source": "high_coder",
                "scanned": scanned,
                "long": len(long_rows),
                "med": len(med_rows),
                "total": len(rows),
            }
        )
        paths["high_coder"] = _write_fetched_source(
            "high_coder",
            rows,
            {"dataset_id": "Crownelius/High-Coder-SFT-Medium", "license": "mit"},
        )
    else:
        paths["high_coder"] = V4_FETCH_DIR / "high_coder.jsonl"

    # 4. longalign (Apache-2.0, long_context_qa bucket)
    if force or not _source_ready("longalign"):
        rows = []
        ds = load_dataset("zai-org/LongAlign-10k", split="train", streaming=True)
        for idx, item in enumerate(ds):
            if len(rows) >= 8500:
                break
            reported_len = item.get("length")
            if isinstance(reported_len, int) and reported_len > 36000:
                continue
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "user"
            ]
            asst_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "assistant"
            ]
            if not user_parts or not asst_parts:
                continue
            target = asst_parts[-1].strip()
            if len(target) < 4 or len(target) > 16000:
                continue
            if len(user_parts) > 1:
                ctx = "\n\n".join(user_parts[:-1]).strip()
                instr = user_parts[-1].strip()
            else:
                instr, ctx = _split_long_user_prompt(user_parts[0])
            if not instr or not ctx:
                continue
            cand = {
                "instance_id": f"v4-longalign-{item.get('id', idx)}",
                "task_type": "long_context_qa",
                "instruction": instr,
                "context": ctx,
                "target_text": target,
                "license": "apache-2.0",
            }
            _probe_and_append(
                rows,
                cand,
                "longalign",
                paired_spec,
                baseline_spec,
                encoder,
                decoder,
                min_enc_tokens=1024,
                max_enc_tokens=ENCODER_LIMIT,
                max_target_tokens=4096,
            )
            if (idx + 1) % 2000 == 0:
                _log({"stage": "fetch_progress", "source": "longalign", "scanned": idx + 1, "taken": len(rows)})
        _log({"stage": "fetch", "source": "longalign", "taken": len(rows)})
        paths["longalign"] = _write_fetched_source(
            "longalign", rows, {"dataset_id": "zai-org/LongAlign-10k", "license": "apache-2.0"}
        )
    else:
        paths["longalign"] = V4_FETCH_DIR / "longalign.jsonl"

    # 5. smoltalk_longalign (Apache-2.0, long_context_qa bucket)
    if force or not _source_ready("smoltalk_longalign"):
        rows = []
        ds = load_dataset("HuggingFaceTB/smoltalk", "longalign", split="train", streaming=True)
        for idx, item in enumerate(ds):
            if len(rows) >= 3500:
                break
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "user"
            ]
            asst_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "assistant"
            ]
            if not user_parts or not asst_parts:
                continue
            target = asst_parts[-1].strip()
            if len(target) < 4 or len(target) > 16000:
                continue
            if len(user_parts) > 1:
                ctx = "\n\n".join(user_parts[:-1]).strip()
                instr = user_parts[-1].strip()
            else:
                instr, ctx = _split_long_user_prompt(user_parts[0])
            cand = {
                "instance_id": f"v4-smoltalk-longalign-{idx}",
                "task_type": "long_context_qa",
                "instruction": instr,
                "context": ctx,
                "target_text": target,
                "license": "apache-2.0",
            }
            _probe_and_append(
                rows,
                cand,
                "smoltalk_longalign",
                paired_spec,
                baseline_spec,
                encoder,
                decoder,
                min_enc_tokens=1024,
                max_enc_tokens=ENCODER_LIMIT,
                max_target_tokens=4096,
            )
        _log({"stage": "fetch", "source": "smoltalk_longalign", "taken": len(rows)})
        paths["smoltalk_longalign"] = _write_fetched_source(
            "smoltalk_longalign",
            rows,
            {"dataset_id": "HuggingFaceTB/smoltalk", "config": "longalign", "license": "apache-2.0"},
        )
    else:
        paths["smoltalk_longalign"] = V4_FETCH_DIR / "smoltalk_longalign.jsonl"

    # 6. ultradata_agent (Apache-2.0, Code-Agent long trajectories -> long_context_qa bucket)
    if force or not _source_ready("ultradata_agent"):
        rows = []
        ds = load_dataset(
            "openbmb/UltraData-SFT-Agent-2609", "Code-Agent", split="train", streaming=True
        )
        for idx, item in enumerate(ds):
            if len(rows) >= 4500 or idx >= 30000:
                break
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 4:
                continue
            last_asst_idx = next(
                (
                    i
                    for i in range(len(messages) - 1, -1, -1)
                    if str(messages[i].get("role") or "").lower() == "assistant"
                    and str(messages[i].get("content") or "").strip()
                ),
                None,
            )
            if last_asst_idx is None or last_asst_idx < 2:
                continue
            target = str(messages[last_asst_idx].get("content") or "").strip()
            if len(target) < 16 or len(target) > 12000:
                continue
            prev_msg = messages[last_asst_idx - 1]
            instr = str(prev_msg.get("content") or "").strip()
            if len(instr) < 8 or len(instr) > 8000:
                continue
            history_parts = []
            for turn in messages[: last_asst_idx - 1]:
                role = str(turn.get("role") or "context").strip()
                content = str(turn.get("content") or "").strip()
                if content:
                    history_parts.append(f"[{role}]\n{content}")
            ctx = "\n\n".join(history_parts)
            if len(ctx) < 4000 or len(ctx) > 130_000:
                continue
            cand = {
                "instance_id": f"v4-ultradata-agent-{item.get('uuid', idx)}",
                "task_type": "long_context_qa",
                "instruction": instr,
                "context": ctx,
                "target_text": target,
                "license": "apache-2.0",
            }
            _probe_and_append(
                rows,
                cand,
                "ultradata_agent",
                paired_spec,
                baseline_spec,
                encoder,
                decoder,
                min_enc_tokens=2048,
                max_enc_tokens=ENCODER_LIMIT,
                max_target_tokens=3072,
            )
            if (idx + 1) % 2000 == 0:
                _log(
                    {
                        "stage": "fetch_progress",
                        "source": "ultradata_agent",
                        "scanned": idx + 1,
                        "taken": len(rows),
                    }
                )
        _log({"stage": "fetch", "source": "ultradata_agent", "taken": len(rows)})
        paths["ultradata_agent"] = _write_fetched_source(
            "ultradata_agent",
            rows,
            {
                "dataset_id": "openbmb/UltraData-SFT-Agent-2609",
                "config": "Code-Agent",
                "license": "apache-2.0",
            },
        )
    else:
        paths["ultradata_agent"] = V4_FETCH_DIR / "ultradata_agent.jsonl"

    # 7. smoltalk_constraints (Apache-2.0, retention bucket)
    if force or not _source_ready("smoltalk_constraints"):
        rows = []
        ds = load_dataset(
            "HuggingFaceTB/smoltalk", "smol-constraints", split="train", streaming=True
        )
        for idx, item in enumerate(ds):
            if len(rows) >= 3500 or idx >= 12000:
                break
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "user"
            ]
            asst_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "assistant"
            ]
            if not user_parts or not asst_parts:
                continue
            instr = "\n\n".join(user_parts).strip()
            target = asst_parts[-1].strip()
            if len(instr) < 12 or len(target) < 8 or len(instr) > 12000 or len(target) > 12000:
                continue
            cand = {
                "instance_id": f"v4-smoltalk-constraints-{idx}",
                "task_type": "retention",
                "instruction": instr,
                "context": instr,
                "target_text": target,
                "license": "apache-2.0",
            }
            _probe_and_append(
                rows, cand, "smoltalk_constraints", paired_spec, baseline_spec, encoder, decoder
            )
        _log({"stage": "fetch", "source": "smoltalk_constraints", "taken": len(rows)})
        paths["smoltalk_constraints"] = _write_fetched_source(
            "smoltalk_constraints",
            rows,
            {
                "dataset_id": "HuggingFaceTB/smoltalk",
                "config": "smol-constraints",
                "license": "apache-2.0",
            },
        )
    else:
        paths["smoltalk_constraints"] = V4_FETCH_DIR / "smoltalk_constraints.jsonl"

    # 8. smoltalk_metamathqa (Apache-2.0, retention bucket, GSM8K-deduped in mix)
    if force or not _source_ready("smoltalk_metamathqa"):
        rows = []
        ds = load_dataset(
            "HuggingFaceTB/smoltalk", "metamathqa-50k", split="train", streaming=True
        )
        for idx, item in enumerate(ds):
            if len(rows) >= 4000 or idx >= 15000:
                break
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "user"
            ]
            asst_parts = [
                str(m.get("content") or "")
                for m in messages
                if str(m.get("role") or "").lower() == "assistant"
            ]
            if not user_parts or not asst_parts:
                continue
            instr = "\n\n".join(user_parts).strip()
            target = asst_parts[-1].strip()
            if len(instr) < 12 or len(target) < 8 or len(instr) > 8000 or len(target) > 8000:
                continue
            cand = {
                "instance_id": f"v4-smoltalk-metamath-{idx}",
                "task_type": "retention",
                "instruction": instr,
                "context": instr,
                "target_text": target,
                "license": "apache-2.0",
            }
            _probe_and_append(
                rows, cand, "smoltalk_metamathqa", paired_spec, baseline_spec, encoder, decoder
            )
        _log({"stage": "fetch", "source": "smoltalk_metamathqa", "taken": len(rows)})
        paths["smoltalk_metamathqa"] = _write_fetched_source(
            "smoltalk_metamathqa",
            rows,
            {
                "dataset_id": "HuggingFaceTB/smoltalk",
                "config": "metamathqa-50k",
                "license": "apache-2.0",
            },
        )
    else:
        paths["smoltalk_metamathqa"] = V4_FETCH_DIR / "smoltalk_metamathqa.jsonl"

    # 9. openmathinstruct2 (CC-BY-4.0, retention bucket, GSM8K-deduped in mix)
    if force or not _source_ready("openmathinstruct2"):
        rows = []
        ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True)
        for idx, item in enumerate(ds):
            if len(rows) >= 4000 or idx >= 15000:
                break
            prob = str(item.get("problem") or "").strip()
            sol = str(item.get("generated_solution") or "").strip()
            if len(prob) < 12 or len(sol) < 8 or len(prob) > 8000 or len(sol) > 8000:
                continue
            cand = {
                "instance_id": f"v4-openmath2-{idx}",
                "task_type": "retention",
                "instruction": prob,
                "context": prob,
                "target_text": sol,
                "license": "cc-by-4.0",
            }
            _probe_and_append(
                rows, cand, "openmathinstruct2", paired_spec, baseline_spec, encoder, decoder
            )
        _log({"stage": "fetch", "source": "openmathinstruct2", "taken": len(rows)})
        paths["openmathinstruct2"] = _write_fetched_source(
            "openmathinstruct2",
            rows,
            {"dataset_id": "nvidia/OpenMathInstruct-2", "license": "cc-by-4.0"},
        )
    else:
        paths["openmathinstruct2"] = V4_FETCH_DIR / "openmathinstruct2.jsonl"

    return paths


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def build_and_verify_mixed_pool(
    source_paths: dict[str, Path],
    *,
    target_count: int = 48000,
    foundation_edit_cap: int = 4000,
    grounded_max_parents: int = 2500,
) -> dict[str, Any]:
    """Run mixed_pool_v4 and enforce 100%-row real Gemma-4 tokenization parity (zero overflow)."""
    context_pool_dir = V4_CONTEXT_POOL_FULL if V4_CONTEXT_POOL_FULL.is_dir() else V4_CONTEXT_POOL
    if not context_pool_dir.is_dir():
        raise FileNotFoundError(f"Missing local v4 context pool: {context_pool_dir}")
    v4_sources = {
        key: rows for key, path in source_paths.items() if (rows := _iter_jsonl(path))
    }
    auto_sigs, auto_gsm = _collect_benchmark_quarantine_assets(
        V4_BENCHMARK_ROOT,
        cache_hf=ROOT / "outputs/context_sources/hf_cache",
        gsm8k_dir=ROOT / "outputs/gsm8k_sources",
    )
    # Also include HumanEvalFix & GSM8K prepared dirs for complete quarantine
    hef_sigs, hef_gsm = _collect_benchmark_quarantine_assets(ROOT / "outputs")
    all_sigs = auto_sigs | hef_sigs
    all_gsm = list(dict.fromkeys([*auto_gsm, *hef_gsm]))
    _log(
        {
            "stage": "quarantine_loaded",
            "code_signatures": len(all_sigs),
            "gsm8k_questions": len(all_gsm),
        }
    )

    prelim_dir = ROOT / "outputs/_mixed_context_enc32k_dec8k_v4_prelim"
    mixed_pool_v4(
        v4_sources,
        V4_BENCHMARK_ROOT,
        prelim_dir,
        foundation=V4_FOUNDATION_POOL,
        context=context_pool_dir,
        code_signatures=all_sigs,
        gsm8k_questions=all_gsm,
        foundation_edit_cap=foundation_edit_cap,
        grounded_max_parents=grounded_max_parents,
        target_count=target_count,
        min_long_context_ratio=0.40,
        seed=17,
    )

    # Strict 100%-row exact Gemma-4 tokenization verification & exact token_lengths backfill
    rows, prelim_manifest = load_curated_pool(prelim_dir)
    paired_spec, baseline_spec, encoder, decoder = _load_gemma4_tokenizers()
    verified_rows: list[dict[str, Any]] = []
    overflow_dropped: Counter[str] = Counter()
    for split_name in ("train", "dev"):
        for idx, row in enumerate(rows[split_name], 1):
            lengths = _measure_exact_tokens(row, paired_spec, baseline_spec, encoder, decoder)
            if lengths is None:
                overflow_dropped[f"{split_name}:{row.get('source_key', row.get('source'))}"] += 1
                continue
            verified = {**row, "pool_split": split_name, "token_lengths": lengths}
            verified_rows.append(verified)
            if idx % 5000 == 0:
                _log(
                    {
                        "stage": "exact_token_verify",
                        "split": split_name,
                        "done": idx,
                        "total": len(rows[split_name]),
                        "dropped": sum(overflow_dropped.values()),
                    }
                )

    train_rows = [r for r in verified_rows if r["pool_split"] == "train"]
    dev_rows = [r for r in verified_rows if r["pool_split"] == "dev"]
    bucket_counts = Counter(classify_v4_bucket(r) for r in train_rows)
    long_train = sum(1 for r in train_rows if r["token_lengths"]["encoder"] >= 8192)
    report = {
        **prelim_manifest["report"],
        "exact_token_verified": True,
        "exact_token_overflow_dropped": dict(overflow_dropped),
        "split": {
            **prelim_manifest["report"]["split"],
            "train": len(train_rows),
            "dev": len(dev_rows),
        },
        "task_counts": dict(Counter(r.get("task_type", "repair") for r in verified_rows)),
        "bucket_counts": dict(bucket_counts),
        "bucket_ratios": {
            b: round(c / max(1, len(train_rows)), 4) for b, c in sorted(bucket_counts.items())
        },
        "long_context_train_count": long_train,
        "long_context_train_ratio": round(long_train / max(1, len(train_rows)), 4),
        "source_counts": dict(Counter(r["source"] for r in verified_rows)),
        "training_plan": {
            "epochs": 1.5,
            "batch_size": 2,
            "accumulation": 8,
            "effective_batch": 16,
            "updates_1_epoch": (len(train_rows) + 15) // 16,
            "updates_1_5_epochs": (round(len(train_rows) * 1.5) + 15) // 16,
            "supervised_train_tokens": sum(r["token_lengths"]["target"] for r in train_rows),
        },
    }
    exported = export_pool(verified_rows, V4_MIXED_POOL, report)
    _log(
        {
            "stage": "mixed_pool_v4_verified",
            "train": len(train_rows),
            "dev": len(dev_rows),
            "overflow_dropped": dict(overflow_dropped),
            "bucket_ratios": report["bucket_ratios"],
            "long_context_train_ratio": report["long_context_train_ratio"],
            "fingerprint": exported["fingerprint"],
        }
    )
    return exported


def _build_v4_notices(pool_dir: Path) -> tuple[list[dict[str, str]], Path]:
    """Generate dataset-level notices for v4 sources and archive notices for repo snapshots."""
    V4_NOTICE_DIR.mkdir(parents=True, exist_ok=True)
    dataset_notice_specs = {
        "v4_opc_sft_stage2": (
            "Dataset: OpenCoder-LLM/opc-sft-stage2\nLicense: MIT\n"
            "Source: https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2\n"
        ),
        "v4_self_oss_instruct": (
            "Dataset: bigcode/self-oss-instruct-sc2-exec-filter-50k\nLicense: ODC-BY\n"
            "Source: https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k\n"
        ),
        "v4_longalign": (
            "Dataset: zai-org/LongAlign-10k\nLicense: Apache-2.0\n"
            "Source: https://huggingface.co/datasets/zai-org/LongAlign-10k\n"
        ),
        "v4_smoltalk": (
            "Dataset: HuggingFaceTB/smoltalk (configs: longalign, smol-constraints, metamathqa-50k)\nLicense: Apache-2.0\n"
            "Source: https://huggingface.co/datasets/HuggingFaceTB/smoltalk\n"
        ),
        "v4_ultradata_agent": (
            "Dataset: openbmb/UltraData-SFT-Agent-2609 (config: Code-Agent)\nLicense: Apache-2.0\n"
            "Source: https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609\n"
        ),
        "v4_openmathinstruct2": (
            "Dataset: nvidia/OpenMathInstruct-2\nLicense: CC-BY-4.0\n"
            "Source: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2\n"
        ),
    }
    notices: list[dict[str, str]] = []
    for name, body in dataset_notice_specs.items():
        path = V4_NOTICE_DIR / f"{name}.txt"
        path.write_text(body, encoding="utf-8")
        notices.append({"name": name, "file": path.relative_to(ROOT).as_posix()})

    # Preserve exact archive license bytes for SWE-rebench-V2 snapshot rows
    rows, _ = load_curated_pool(pool_dir)
    grouped: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}
    for row in rows["train"] + rows["dev"]:
        if row.get("source") not in {"nebius/SWE-rebench-V2", "internlm/SWE-Fixer-Train-110K"}:
            continue
        group = grouped.setdefault((row["repos"], row["commit"]), {})
        for evidence in row.get("context_source_code_license_evidence", ()):
            for item in evidence.get("licenses", ()):
                if "path" in item and "sha256" in item:
                    group[(item["path"], item["sha256"])] = item

    src_lic_dir = ROOT / "outputs/context_source_licenses_v4"
    src_lic_dir.mkdir(parents=True, exist_ok=True)
    store = ArchiveSnapshotStore(ROOT / "outputs/context_sources/repository_corpus/archives")
    seen_notices: dict[str, dict[str, str]] = {}
    inventory: list[dict[str, Any]] = []
    for (repo, commit), entries in grouped.items():
        snapshot, _ = store.snapshot(SimpleNamespace(repo=repo, base_commit=commit))
        files = {f.path: f.content for f in snapshot.files}
        for (path, sha), item in entries.items():
            name = "source_license_" + sha
            dest = src_lic_dir / (name + ".txt")
            content = files[path].encode("utf-8")
            if hashlib.sha256(content).hexdigest() != sha:
                raise ValueError("source_notice_hash_changed")
            dest.write_bytes(content)
            seen_notices[name] = {"name": name, "file": dest.relative_to(ROOT).as_posix()}
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
    inv_path = src_lic_dir / "source_notice_inventory_v4.json"
    atomic_json(inv_path, inventory)
    notices.extend(seen_notices.values())
    return notices, inv_path


def _prepare_reponiah_collection() -> Path:
    """Build deterministic frozen RepoNIAH (4k-32k) collection with real Gemma-4 tokenizer."""
    reponiah_item = next(
        item for item in V4_BENCHMARKS if item["collection"] == "reponiah_gemma32k_dec8k_v1"
    )
    _, _, _, decoder = _load_gemma4_tokenizers()
    public, gold = _build_reponiah_cohort(
        reponiah_item,
        tokenizer=decoder,
        max_position_embeddings=32768,
        enable_thinking=False,
    )
    V4_REPONIAH_DIR.mkdir(parents=True, exist_ok=True)
    (V4_REPONIAH_DIR / "eval_private").mkdir(parents=True, exist_ok=True)
    with (V4_REPONIAH_DIR / "eval_inputs.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in public:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (V4_REPONIAH_DIR / "eval_private/specs.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        for row in gold:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(
        V4_REPONIAH_DIR / "preparation_report.json",
        {
            "collection": "reponiah_gemma32k_dec8k_v1",
            "count": len(public),
            "depths": list(reponiah_item["depths"]),
            "depth_ratios": list(reponiah_item["depth_ratios"]),
            "cases_per_depth": reponiah_item["cases_per_depth"],
            "seed": 42,
            "protocol": "reponiah_exact_value_retrieval_4k_32k_v1",
            "official_protocol": False,
        },
    )
    _log({"stage": "reponiah_prepared", "count": len(public), "dir": str(V4_REPONIAH_DIR)})
    return V4_REPONIAH_DIR


def build_and_verify_release() -> dict[str, Any]:
    """Package mixed_context_enc32k_dec8k_v4 plus all 9 eval cohorts (incl. RepoNIAH) and verify 100% view parity."""
    _prepare_reponiah_collection()
    v3_spec = json.loads((ROOT / "configs/dataset_release.context32k.json").read_text(encoding="utf-8"))
    eval_collections = [c for c in v3_spec["collections"] if c["kind"] != "curated_pool"]
    if not any(c["name"] == "reponiah_gemma32k_dec8k_v1" for c in eval_collections):
        eval_collections.append(
            {
                "name": "reponiah_gemma32k_dec8k_v1",
                "kind": "prepared_code_eval",
                "directory": V4_REPONIAH_DIR.relative_to(ROOT).as_posix(),
                "description": (
                    "Frozen RepoNIAH needle-in-a-codebase-haystack benchmark across 4k/8k/16k/32k "
                    "Gemma-4 tokens and 5 depth positions (0.1/0.25/0.5/0.75/0.9, 100 cases)."
                ),
            }
        )
    collections = [
        {
            "name": "mixed_context_enc32k_dec8k_v4",
            "kind": "curated_pool",
            "directory": V4_MIXED_POOL.relative_to(ROOT).as_posix(),
            "description": (
                "V4 disjoint-context 32K/8K curated pool (35% repair/loc/code, 25% multi-function FIM, "
                "25% long-context QA/grounded, 15% retention; >=40% >=8k tokens; 0% NextCoder; "
                "100%-row exact Gemma-4 tokenizer verified at encoder<=32768, pair_decoder<=8192, baseline_decoder<=41472)."
            ),
        },
        *eval_collections,
    ]
    base_notices = [
        n
        for n in v3_spec["notices"]
        if not str(n["name"]).startswith("source_license_") and n["name"] != "nextcoder"
    ]
    v4_notices, inv_path = _build_v4_notices(V4_MIXED_POOL)
    # Deduplicate notices by name
    merged_notices: dict[str, dict[str, str]] = {}
    for n in [*base_notices, *v4_notices]:
        merged_notices[n["name"]] = n

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    spec = {
        **v3_spec,
        "name": f"bifocalgemma-private-context-data-{today.replace('-', '')}-v4",
        "date": today,
        "supersedes_revision": EXPECTED_PARENT_REVISION,
        "collections": collections,
        "notices": list(merged_notices.values()),
        "audits": [
            *[a for a in v3_spec.get("audits", []) if a["name"] not in {"mixed_pool_v3", "source_notice_inventory"}],
            {"name": "source_notice_inventory_v4", "file": inv_path.relative_to(ROOT).as_posix()},
        ],
    }
    atomic_json(V4_SPEC_PATH, spec)
    release = build_release(V4_SPEC_PATH, V4_RELEASE_DIR)
    verify_release(V4_RELEASE_DIR)
    view_audit = audit_views(V4_RELEASE_DIR)
    audit_path = ROOT / "outputs/context_release_audit_v4.json"
    atomic_json(
        audit_path,
        {
            "date": today,
            "release_fingerprint": release["fingerprint"],
            "views": view_audit,
        },
    )
    _log(
        {
            "stage": "release_verified",
            "release_dir": str(V4_RELEASE_DIR),
            "fingerprint": release["fingerprint"],
            "collections": list(release["collections"].keys()),
        }
    )
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--mix-only", action="store_true")
    parser.add_argument("--release-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--target-count", type=int, default=48000)
    args = parser.parse_args()

    run_all = not (args.fetch_only or args.mix_only or args.release_only)
    source_paths: dict[str, Path] = {}
    if run_all or args.fetch_only:
        source_paths = fetch_all_v4_sources(force=args.force_fetch)
    if args.fetch_only:
        return

    if not source_paths:
        for key in (
            "opc_sft_stage2",
            "self_oss_instruct",
            "high_coder",
            "longalign",
            "smoltalk_longalign",
            "ultradata_agent",
            "smoltalk_constraints",
            "smoltalk_metamathqa",
            "openmathinstruct2",
        ):
            p = V4_FETCH_DIR / f"{key}.jsonl"
            if p.is_file():
                source_paths[key] = p

    if run_all or args.mix_only:
        build_and_verify_mixed_pool(source_paths, target_count=args.target_count)
    if args.mix_only:
        return

    if run_all or args.release_only:
        build_and_verify_release()

    if args.publish:
        receipt = publish_release(
            V4_RELEASE_DIR,
            HF_DATASET_REPO,
            replace_current=False,
            expected_parent=EXPECTED_PARENT_REVISION,
        )
        atomic_json(V4_PUBLICATION_RECEIPT, receipt)
        _log({"stage": "published", **receipt})


if __name__ == "__main__":
    main()
