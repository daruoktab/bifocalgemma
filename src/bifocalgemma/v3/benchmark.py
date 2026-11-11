"""V3 batched evaluation and the eight-benchmark suite."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.data.curation import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.evaluation.context import (
    EVALUATION_PROTOCOL,
    V2_EVALUATION_PROTOCOL,
    decoder_prompt,
    prompt_ids,
    score_context_case,
)
from bifocalgemma.inference.generation import GreedyGeneration, greedy_decode
from bifocalgemma.models import inference_cache as inference_cache_mod
from bifocalgemma.training.stages import (
    atomic_json,
)
from bifocalgemma.training.telemetry import resource_snapshot
from bifocalgemma.v3.attention import configure_full_rank_cross_attention
from bifocalgemma.v3.constants import (
    BENCHMARK_SAMPLE_SEED,
    BENCHMARK_TIER_CONFIG,
    MODEL_EVAL_FOOTPRINT,
)


def infer_benchmark_weight_tier(benchmark_item: dict[str, Any] | None) -> str:
    """Classify benchmark evaluation weight tier from collection name and token budgets."""
    if not benchmark_item:
        return "sft_eval"
    if "weight_tier" in benchmark_item and benchmark_item["weight_tier"] in BENCHMARK_TIER_CONFIG:
        return benchmark_item["weight_tier"]
    name = str(benchmark_item.get("collection", "")).lower()
    if "gsm8k" in name:
        return "light"
    if "humanevalfix" in name:
        return "medium"
    if "crosscodelong" in name:
        return "heavy"
    if "longcodeqa" in name or "sweqa" in name:
        return "medium_long"
    enc_lim = int(benchmark_item.get("encoder_limit", 32768))
    if enc_lim <= 4096:
        return "light"
    return "medium_long"


def resolve_optimal_eval_batch_size(
    model_key: str,
    benchmark_item: dict[str, Any] | None = None,
    *,
    free_vram_gib: float | None = None,
    device: Any = None,
) -> int:
    """Compute the highest safe evaluation/benchmark batch size for any of the 10 models.

    Accounts for:
    1. Model size and architecture (`e2b-it` > `2B-2B` > `e4b-it` / asymmetric > `4B-4B`).
    2. Benchmark sequence length & generation weight (`light` > `medium` > `medium_long` > `heavy`).
    3. Available GPU VRAM headroom (probed live or supplied via `free_vram_gib`).
    """
    profile = MODEL_EVAL_FOOTPRINT.get(model_key, {"weights_gib": 14.0, "scale_factor": 1.0})
    tier = infer_benchmark_weight_tier(benchmark_item)
    tier_cfg = BENCHMARK_TIER_CONFIG[tier]
    scale_factor = float(profile["scale_factor"])
    per_sample_gib = float(tier_cfg["per_sample_gib"]) / scale_factor
    max_cap = round(float(tier_cfg["max_batch"]) * min(1.25, max(0.65, scale_factor)))
    max_cap = max(2, max_cap)

    if free_vram_gib is None and torch.cuda.is_available():
        try:
            dev_idx = (
                device.index
                if isinstance(device, torch.device) and device.index is not None
                else torch.cuda.current_device()
            )
            free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
            free_vram_gib = float(free_bytes) / (1024.0**3)
        except Exception:
            free_vram_gib = None

    if free_vram_gib is not None:
        usable_gib = max(0.0, float(free_vram_gib) - 6.0)
        vram_batch = int(usable_gib // max(0.5, per_sample_gib))
        return max(1, min(max_cap, vram_batch))

    # Deterministic default for 80 GiB Molab GPU when running on CPU / offline inspection
    assumed_usable_gib = max(8.0, 80.0 - float(profile["weights_gib"]) - 8.0)
    heuristic_batch = int(
        min(
            round(float(tier_cfg["base_batch"]) * scale_factor),
            assumed_usable_gib // max(0.5, per_sample_gib),
        )
    )
    return max(1, min(max_cap, heuristic_batch))


def prepare_sampled_benchmark_collection(
    collection_dir: Path | str,
    *,
    sample_limit: int | None = None,
    sample_seed: int = BENCHMARK_SAMPLE_SEED,
    output_dir: Path | str | None = None,
) -> Path:
    """Create a deterministic sampled benchmark directory when `sample_limit` is set (e.g., 100 cases).

    Guarantees identical sampled `instance_id` cohorts across all 10 models.
    """
    src_dir = Path(collection_dir).resolve()
    if sample_limit is None or sample_limit <= 0:
        return src_dir

    limit = sample_limit
    inputs_name = (
        "eval_inputs.jsonl" if (src_dir / "eval_inputs.jsonl").exists() else "inputs.jsonl"
    )
    gold_name = (
        "eval_private/specs.jsonl"
        if (src_dir / "eval_private/specs.jsonl").exists()
        else "hidden_gold.jsonl"
    )
    inputs_path = src_dir / inputs_name
    gold_path = src_dir / gold_name
    if not inputs_path.exists() or not gold_path.exists():
        return src_dir

    public_rows = list(read_source_rows(inputs_path))
    gold_rows = list(read_source_rows(gold_path))
    if len(public_rows) <= limit:
        return src_dir

    dest_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else src_dir.parent / f"{src_dir.name}__sample{limit}_seed{sample_seed}"
    )
    sample_manifest_path = dest_dir / "sample_manifest.json"
    rng = random.Random(f"bifocalgemma_benchmark_sample_{src_dir.name}_{sample_seed}_{limit}")
    chosen_indices = sorted(rng.sample(range(len(public_rows)), limit))
    sampled_public = [public_rows[i] for i in chosen_indices]
    sampled_gold = [gold_rows[i] for i in chosen_indices]
    sampled_ids = [r["instance_id"] for r in sampled_public]

    if sample_manifest_path.is_file():
        try:
            existing_meta = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
            if (
                existing_meta.get("sampled_instance_ids") == sampled_ids
                and (dest_dir / inputs_name).is_file()
                and (dest_dir / gold_name).is_file()
            ):
                return dest_dir
        except Exception:
            pass

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_inputs = dest_dir / inputs_name
    dest_gold = dest_dir / gold_name
    dest_inputs.parent.mkdir(parents=True, exist_ok=True)
    dest_gold.parent.mkdir(parents=True, exist_ok=True)

    with dest_inputs.open("w", encoding="utf-8", newline="\n") as f_in:
        for row in sampled_public:
            f_in.write(json.dumps(row, ensure_ascii=False) + "\n")
    with dest_gold.open("w", encoding="utf-8", newline="\n") as f_gold:
        for row in sampled_gold:
            f_gold.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Symlink or copy any auxiliary snapshot files referenced by gold rows
    for g_row in sampled_gold:
        snap_rel = g_row.get("snapshot_file")
        if snap_rel:
            src_snap = (src_dir / snap_rel).resolve()
            dst_snap = (dest_dir / snap_rel).resolve()
            if src_snap.is_file() and not dst_snap.exists():
                dst_snap.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_snap, dst_snap)

    atomic_json(
        sample_manifest_path,
        {
            "collection": src_dir.name,
            "sample_limit": limit,
            "sample_seed": sample_seed,
            "total_population": len(public_rows),
            "sampled_count": len(sampled_ids),
            "sampled_instance_ids": sampled_ids,
        },
    )
    return dest_dir


def batched_greedy_decode(
    model: Any,
    batch_encoder_ids: list[tuple[int, ...]],
    batch_prefix_ids: list[tuple[int, ...]],
    *,
    max_new_tokens_list: list[int] | int,
    eos_ids: tuple[int, ...] = (1, 106, 50),
    max_total_decoder_tokens: int = 4096,
    progress: Any = None,
    use_cache: bool = True,
) -> list[GreedyGeneration]:
    """Batched greedy generation across B >= 1 items with left-padded causal KV cache and static cross-KV."""
    bsz = len(batch_prefix_ids)
    if bsz == 0:
        return []
    if isinstance(max_new_tokens_list, int):
        budgets = [max_new_tokens_list] * bsz
    else:
        budgets = list(max_new_tokens_list)

    if bsz == 1:
        if (
            getattr(inference_cache_mod.CachedAttention, "__name__", "")
            != "FullCrossCachedAttention"
        ):
            from bifocalgemma.v3.attention import FullCrossCachedAttention

            if getattr(model, "encoder", None) is None or hasattr(
                model.decoder.layers[0].self_attn, "cross_q_norm"
            ):
                setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)
            else:
                configure_full_rank_cross_attention(model)
        return [
            greedy_decode(
                model,
                batch_encoder_ids[0],
                batch_prefix_ids[0],
                max_new_tokens=budgets[0],
                eos_ids=eos_ids,
                max_total_decoder_tokens=max_total_decoder_tokens,
                progress=progress,
                use_cache=use_cache,
            )
        ]

    # Validate budgets and vocab bounds up front matching greedy_decode contracts
    vocabulary = model.decoder.config.vocab_size
    for enc_ids, pref_ids, budget in zip(batch_encoder_ids, batch_prefix_ids, budgets, strict=True):
        if budget < 1 or not pref_ids or len(pref_ids) + budget > max_total_decoder_tokens:
            raise ValueError("generation_budget_exceeded")
        if min((*enc_ids, *pref_ids)) < 0 or max((*enc_ids, *pref_ids)) >= vocabulary:
            raise ValueError("tokenizer_ids_do_not_fit_model_vocabulary")
        if getattr(model, "encoder", None) is None and enc_ids:
            raise ValueError("baseline_cannot_receive_encoder_tokens")

    device = next(model.parameters()).device
    if device.type == "meta":
        raise ValueError("generation_requires_materialized_weights")

    if not use_cache:
        return [
            greedy_decode(
                model,
                batch_encoder_ids[i],
                batch_prefix_ids[i],
                max_new_tokens=budgets[i],
                eos_ids=eos_ids,
                max_total_decoder_tokens=max_total_decoder_tokens,
                progress=progress,
                use_cache=False,
            )
            for i in range(bsz)
        ]

    previous_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            if getattr(model, "encoder", None) is not None:
                max_enc_len = max((len(enc_ids) for enc_ids in batch_encoder_ids), default=0)
                if max_enc_len > 0:
                    enc_tensor = torch.zeros((bsz, max_enc_len), dtype=torch.long, device=device)
                    enc_mask = torch.zeros((bsz, max_enc_len), dtype=torch.bool, device=device)
                    for b, enc_ids in enumerate(batch_encoder_ids):
                        if enc_ids:
                            enc_tensor[b, : len(enc_ids)] = torch.tensor(
                                enc_ids, dtype=torch.long, device=device
                            )
                            enc_mask[b, : len(enc_ids)] = True
                    states = model.encode_context(enc_tensor, enc_mask)
                else:
                    states = model.decoder.embed_tokens.weight.new_empty(
                        bsz, 0, model.decoder.config.hidden_size
                    )
                    enc_mask = torch.zeros((bsz, 0), dtype=torch.bool, device=device)
            else:
                states = None
                enc_mask = None

            max_pref_len = max(len(pref_ids) for pref_ids in batch_prefix_ids)
            pad_lens = [max_pref_len - len(pref_ids) for pref_ids in batch_prefix_ids]
            pad_offsets = torch.tensor(pad_lens, dtype=torch.long, device=device)
            prefill_valid = torch.zeros((bsz, max_pref_len), dtype=torch.bool, device=device)
            sequences: list[list[int]] = []
            for b, (pref_ids, pad_len) in enumerate(zip(batch_prefix_ids, pad_lens, strict=True)):
                pad_tok = pref_ids[0]
                sequences.append([pad_tok] * pad_len + list(pref_ids))
                prefill_valid[b, pad_len:] = True

            outputs: list[list[int]] = [[] for _ in range(bsz)]
            reasons = ["length_limit" for _ in range(bsz)]
            finished = [False for _ in range(bsz)]
            max_steps = max(budgets)

            if (
                getattr(inference_cache_mod.CachedAttention, "__name__", "")
                != "FullCrossCachedAttention"
            ):
                from bifocalgemma.v3.attention import FullCrossCachedAttention

                if getattr(model, "encoder", None) is None or hasattr(
                    model.decoder.layers[0].self_attn, "cross_q_norm"
                ):
                    setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)
                else:
                    configure_full_rank_cross_attention(model)
            cache = inference_cache_mod.InferenceCache()
            original_cls = [layer.self_attn.__class__ for layer in model.decoder.layers]
            try:
                for layer in model.decoder.layers:
                    layer.self_attn.__class__ = inference_cache_mod.CachedAttention
                for step_idx in range(max_steps):
                    is_prefill = cache.offset == 0
                    ids = torch.tensor(
                        [seq if is_prefill else seq[-1:] for seq in sequences],
                        dtype=torch.long,
                        device=device,
                    )
                    raw_positions = torch.arange(
                        cache.offset, cache.offset + ids.shape[1], device=device
                    )[None, :]
                    positions = (raw_positions - pad_offsets[:, None]).clamp_min(0)
                    setattr(
                        cache,
                        "step_valid_mask",
                        prefill_valid
                        if is_prefill
                        else torch.ones((bsz, 1), dtype=torch.bool, device=device),
                    )
                    hidden = model.decoder(
                        input_ids=ids,
                        position_ids=positions,
                        attention_mask={"full_attention": None, "sliding_attention": None},
                        use_cache=False,
                        inference_cache=cache,
                        encoder_hidden_states=states,
                        encoder_valid=enc_mask,
                    ).last_hidden_state[:, -1:]
                    logits = model.lm_head(hidden)
                    cap = model.decoder.config.final_logit_softcapping
                    if cap is not None:
                        logits = torch.tanh(logits / cap) * cap
                    cache.offset += ids.shape[1]
                    next_logits = logits[:, -1]
                    if not bool(torch.isfinite(next_logits).all()):
                        raise ValueError("nonfinite_generation_logits")
                    next_tokens = next_logits.argmax(-1).tolist()
                    for b in range(bsz):
                        if finished[b]:
                            sequences[b].append(eos_ids[0])
                            continue
                        tok = int(next_tokens[b])
                        outputs[b].append(tok)
                        sequences[b].append(tok)
                        if tok in eos_ids:
                            reasons[b] = "tool_boundary" if tok == 50 else "eos"
                            finished[b] = True
                        elif len(outputs[b]) >= budgets[b]:
                            reasons[b] = "length_limit"
                            finished[b] = True
                    if progress:
                        progress(
                            {
                                "stage": "benchmark_decoding",
                                "generated_tokens": step_idx + 1,
                                "batch_size": bsz,
                            }
                        )
                    if all(finished):
                        break
            finally:
                for layer, cls in zip(model.decoder.layers, original_cls, strict=True):
                    layer.self_attn.__class__ = cls
    finally:
        model.train(previous_mode)

    return [
        GreedyGeneration(
            tuple(outputs[b]),
            reasons[b],
            len(batch_encoder_ids[b]),
            len(batch_prefix_ids[b]),
        )
        for b in range(bsz)
    ]


def run_batched_context_benchmark(
    model: Any,
    spec: ExperimentSpec,
    directory: Path | str,
    output: Path | str,
    tokenizer_cache: Path | str,
    *,
    checkpoint_identity: dict[str, Any],
    model_key: str | None = None,
    benchmark_item: dict[str, Any] | None = None,
    batch_size: int | None = None,
    encoder_limit: int = 8192,
    decoder_limit: int = 2048,
    max_new_tokens: int = 2048,
    allow_execution: bool = True,
    python_executable: str | None = None,
    progress: Any = None,
    prediction_callback: Any = None,
    upload_every: int = 100,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run context benchmark using dynamic optimal batching and deterministic sampling with OOM backoff."""
    from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
    from bifocalgemma.training.timing import DurableClock

    sample_limit = benchmark_item.get("sample_limit") if benchmark_item else None
    sample_seed = (
        int(benchmark_item.get("sample_seed", BENCHMARK_SAMPLE_SEED))
        if benchmark_item
        else BENCHMARK_SAMPLE_SEED
    )
    effective_dir = prepare_sampled_benchmark_collection(
        directory, sample_limit=sample_limit, sample_seed=sample_seed
    )

    device = next(model.parameters()).device
    resolved_batch_size = (
        batch_size
        if batch_size is not None and batch_size >= 1
        else resolve_optimal_eval_batch_size(
            model_key or "4base-4it", benchmark_item, device=device
        )
    )

    root, out_dir = Path(effective_dir), Path(output)
    inputs_path = (
        root / "eval_inputs.jsonl"
        if (root / "eval_inputs.jsonl").exists()
        else root / "inputs.jsonl"
    )
    gold_path = (
        root / "eval_private/specs.jsonl"
        if (root / "eval_private/specs.jsonl").exists()
        else root / "hidden_gold.jsonl"
    )
    public = list(read_source_rows(inputs_path))
    hidden = list(read_source_rows(gold_path))
    ids = [r["instance_id"] for r in public]
    if not ids or len(set(ids)) != len(ids) or ids != [r["instance_id"] for r in hidden]:
        raise ValueError("benchmark_public_gold_cohort_mismatch")
    gold = {r["instance_id"]: r for r in hidden}
    protocol_version = V2_EVALUATION_PROTOCOL if use_cache else EVALUATION_PROTOCOL
    signature = stable_fingerprint(
        {
            "model": spec.to_dict(),
            "checkpoint": checkpoint_identity,
            "inputs": public,
            "gold": hidden,
            "encoder_limit": encoder_limit,
            "decoder_limit": decoder_limit,
            "max_new_tokens": max_new_tokens,
            "execution": allow_execution,
            "protocol": protocol_version,
            "generation_budget_policy": "actual_paired_prompt_mcq256_v2",
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "evaluation_state.json"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")).get("signature") != signature:
            raise ValueError("Benchmark/checkpoint/protocol changed; choose a new output.")
    else:
        atomic_json(state_path, {"signature": signature, "total": len(ids)})

    predictions = out_dir / "predictions.jsonl"
    completed = []
    if predictions.exists():
        with predictions.open("rb+") as stream:
            offset = 0
            for line in stream:
                if not line.endswith(b"\n"):
                    if stream.read():
                        raise ValueError("Corrupt prediction journal before final record.")
                    stream.truncate(offset)
                    break
                try:
                    value = json.loads(line)
                except ValueError:
                    if stream.read():
                        raise ValueError("Corrupt prediction journal before final record.")
                    stream.truncate(offset)
                    break
                completed.append(value)
                offset += len(line)
    completed_ids = [r["instance_id"] for r in completed]
    if completed_ids != ids[: len(completed_ids)]:
        raise ValueError("Prediction journal identity/order mismatch.")

    decoder, _ = load_verified_tokenizer(spec.decoder, Path(tokenizer_cache))
    encoder = (
        load_verified_tokenizer(spec.encoder, Path(tokenizer_cache))[0] if spec.encoder else None
    )
    started = time.perf_counter()
    clock = DurableClock(out_dir / "timing.json", started=started)
    segment_initial_count = len(completed)
    heartbeat_at = 0.0
    heartbeat_stage = None

    def activity(event: dict[str, Any], *, force: bool = False) -> None:
        nonlocal heartbeat_at, heartbeat_stage
        now = time.monotonic()
        if not force and event.get("stage") == heartbeat_stage and now - heartbeat_at < 5:
            return
        heartbeat_at = now
        heartbeat_stage = event.get("stage")
        payload = {
            "updated_at": time.time(),
            "benchmark": Path(directory).name,
            "eval_batch_size": current_batch_size,
            **event,
        }
        atomic_json(out_dir / "activity.json", payload)
        clock.persist(force=force)
        if progress:
            progress(payload)

    remaining_rows = public[len(completed) :]
    current_batch_size = max(1, resolved_batch_size)
    cursor = 0

    while cursor < len(remaining_rows):
        chunk_rows = remaining_rows[cursor : cursor + current_batch_size]
        chunk_start_index = len(completed) + 1
        chunk_started = time.monotonic()

        batch_enc: list[tuple[int, ...]] = []
        batch_pref: list[tuple[int, ...]] = []
        batch_budgets: list[int] = []
        max_total_dec = 0

        for row in chunk_rows:
            cap = min(encoder_limit, row.get("max_encoder_tokens", encoder_limit))
            enc, prefix = prompt_ids(row, spec, encoder, decoder, encoder_limit=cap)
            paired_prompt = decoder_prompt(row, paired=True)
            try:
                paired_rendered = decoder.apply_chat_template(
                    [{"role": "user", "content": paired_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                paired_rendered = decoder.apply_chat_template(
                    [{"role": "user", "content": paired_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            paired_prefix_size = len(decoder.encode(paired_rendered, add_special_tokens=False))
            row_budget = int(row.get("max_new_tokens", max_new_tokens))
            if (
                str(row.get("task_type", "")).lower() in {"mcq", "qa", "retrieval"}
                and max_new_tokens >= 256
            ):
                row_budget = max(row_budget, 256)
            new_tokens = min(row_budget, decoder_limit - paired_prefix_size)
            if new_tokens < 1:
                raise ValueError("benchmark_decoder_prefix_exceeds_budget")
            total = decoder_limit if spec.encoder else cap + decoder_limit + 512
            if len(prefix) + new_tokens > total:
                raise ValueError("benchmark_decoder_prefix_exceeds_budget")
            batch_enc.append(enc)
            batch_pref.append(prefix)
            batch_budgets.append(new_tokens)
            max_total_dec = max(max_total_dec, total)

        activity(
            {
                "instance_id": chunk_rows[0]["instance_id"],
                "items_done": len(completed),
                "items_total": len(ids),
                "active_case": chunk_start_index,
                "stage": "benchmark_case_started",
                "generated_tokens": 0,
                "eval_batch_size": len(chunk_rows),
            },
            force=True,
        )

        try:
            generations = batched_greedy_decode(
                model,
                batch_enc,
                batch_pref,
                max_new_tokens_list=batch_budgets,
                max_total_decoder_tokens=max_total_dec,
                progress=lambda event: activity(
                    {
                        "instance_id": chunk_rows[0]["instance_id"],
                        "items_done": len(completed),
                        "items_total": len(ids),
                        "active_case": chunk_start_index,
                        **event,
                        "case_elapsed_seconds": time.monotonic() - chunk_started,
                    }
                ),
                use_cache=use_cache,
            )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                continue
            raise

        per_item_seconds = (time.monotonic() - chunk_started) / max(1, len(chunk_rows))
        for row, generated, enc, prefix in zip(
            chunk_rows, generations, batch_enc, batch_pref, strict=True
        ):
            index = len(completed) + 1
            tokens = generated.token_ids
            if tokens and tokens[-1] in (1, 106, 50):
                tokens = tokens[:-1]
            completion = decode_text(decoder, tokens, skip_special_tokens=False)
            scoring = score_context_case(
                row,
                gold[row["instance_id"]],
                completion,
                directory=root,
                python_executable=python_executable or sys.executable,
                allow_execution=allow_execution,
                **({"include_syntax": True} if use_cache else {}),
            )
            record = {
                "instance_id": row["instance_id"],
                "completion": completion,
                "stop_reason": generated.stop_reason,
                "encoder_tokens": len(enc),
                "decoder_prefix_tokens": len(prefix),
                "generated_tokens": len(tokens),
                "case_seconds": per_item_seconds,
                "eval_batch_size": len(chunk_rows),
                **scoring,
            }
            with predictions.open("a", encoding="utf-8", newline="\n") as out_stream:
                out_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                out_stream.flush()
                os.fsync(out_stream.fileno())
            completed.append(record)
            if prediction_callback and (index % upload_every == 0 or index == len(ids)):
                prediction_callback()
            if progress:
                progress(
                    {
                        "stage": "benchmark_progress",
                        "items_done": index,
                        "items_total": len(ids),
                        "message": f"{Path(directory).name}: {index:,}/{len(ids):,} (batch={len(chunk_rows)})",
                        "benchmark": Path(directory).name,
                        "eval_batch_size": len(chunk_rows),
                        "benchmark_elapsed_seconds": clock.elapsed(),
                        "unobserved_gap_seconds": clock.gap,
                        "benchmark_eta_seconds": (time.perf_counter() - started)
                        / max(1, index - segment_initial_count)
                        * max(0, len(ids) - index),
                        "benchmark_correct": sum(r["correct"] is True for r in completed),
                        "last_prediction": record,
                        **resource_snapshot(device),
                    }
                )
        cursor += len(chunk_rows)

    scored = all(r["correct"] is not None for r in completed)
    clock.close()
    report: dict[str, Any] = {
        "status": "COMPLETED",
        "signature": signature,
        "experiment": spec.name,
        "benchmark": Path(directory).name,
        "total": len(ids),
        "sample_limit": sample_limit,
        "eval_batch_size": current_batch_size,
        "correct": sum(bool(r["correct"]) for r in completed) if scored else None,
        "accuracy": sum(bool(r["correct"]) for r in completed) / len(ids) if scored else None,
        "score_available": scored,
        "statuses": dict(Counter(r["status"] for r in completed)),
        "generation_length_limit": sum(r["stop_reason"] == "length_limit" for r in completed),
        "segment_seconds": time.perf_counter() - started,
        "timing": clock.summary(),
        "official_protocol": False,
        "protocol_version": protocol_version,
        "protocol": ("greedy_cached" if use_cache else "greedy_no_temporal_cache")
        + "; fixed eligible cohort; no context truncation",
        "resources": resource_snapshot(device),
    }
    for key in ("exact_match", "sequence_similarity", "regex_identifier_set_f1"):
        if completed and key in completed[0]:
            report[key] = sum(float(r[key]) for r in completed) / len(ids)
    if completed and "patch_applied" in completed[0]:
        report["patch_apply_rate"] = sum(bool(r["patch_applied"]) for r in completed) / len(ids)
        checked = [r for r in completed if r.get("syntax_valid") is not None]
        report["syntax_checked"] = len(checked)
        report["syntax_success_rate"] = (
            sum(bool(r["syntax_valid"]) for r in checked) / len(checked) if checked else None
        )
    atomic_json(out_dir / "benchmark_report.json", report)
    activity(
        {"stage": "benchmark_completed", "items_done": len(ids), "items_total": len(ids)},
        force=True,
    )
    return report
