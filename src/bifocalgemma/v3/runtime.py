"""V3 runtime guards, length temperature, and tokenization patch."""

from __future__ import annotations

import math
import os
from typing import Any


def ensure_hardware_memory_guards() -> dict[str, Any]:
    """Configure CUDA expandable segments and return active hardware guard settings."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return {
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "sm86_head_dim256_tile_clamp": clamped_triton_kernel_options(256),
    }


def clamped_triton_kernel_options(head_dim: int) -> dict[str, int]:
    """Clamp Triton tile dimensions for head_dim >= 256 to respect the 101,376B sm86 SRAM limit."""
    if int(head_dim) >= 256:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "bwd_BLOCK_M": 16,
            "bwd_BLOCK_N": 16,
            "num_stages": 1,
            "bwd_num_stages": 1,
        }
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "bwd_BLOCK_M": 32,
        "bwd_BLOCK_N": 32,
        "num_stages": 2,
        "bwd_num_stages": 1,
    }


def critical_attention_beta(n_enc: int, n_0: int = 2048) -> float:
    """Compute critical attention length temperature scaling factor beta_N for long contexts."""
    if n_enc <= n_0 or n_0 <= 0:
        return 1.0
    return float(0.1 * math.log(float(n_enc) / float(n_0)) + 1.0)


def format_query_injected_context(context: str, instruction: str | None = None) -> str:
    """Format query-injected encoder context so bidirectional encoder self-attention sees task intent."""
    if not instruction or not instruction.strip():
        return context
    clean_query = instruction.strip()
    if context.startswith("[QUERY]"):
        return context
    return f"[QUERY] {clean_query}\n{context}"


def patch_matrix_tokenization() -> None:
    """Ensure that IT encoders prepend BOS token during tokenization, matching Base encoder invariants.

    Also supports optional query-prefix injection (`row.get("inject_query_prefix")`) into encoder context
    while preserving strict token-limit safety and backward compatibility.
    """
    import bifocalgemma.data.context_tasks as ct_mod
    import bifocalgemma.data.tokenization as tok_mod
    import bifocalgemma.evaluation.context as eval_ctx_mod
    import bifocalgemma.evaluation.gsm8k as gsm8k_mod
    import bifocalgemma.training.curated_sft as cs_mod
    import bifocalgemma.training.repository_sft as rs_mod

    # 1. Wrap load_verified_tokenizer so any loaded tokenizer's encode() safely prepends BOS when add_special_tokens=True
    if not getattr(tok_mod, "_matrix_bos_patched", False):
        orig_load = tok_mod.load_verified_tokenizer

        def safe_load_verified_tokenizer(checkpoint, tokenizer_cache):
            tokenizer, expected = orig_load(checkpoint, tokenizer_cache)
            if not getattr(tokenizer, "_matrix_bos_wrapped", False):
                orig_encode = tokenizer.encode

                def safe_encode(text, *args, **kwargs):
                    ids = orig_encode(text, *args, **kwargs)
                    add_special = kwargs.get(
                        "add_special_tokens", True if len(args) < 2 else args[1]
                    )
                    if (
                        add_special
                        and ids
                        and tokenizer.bos_token_id is not None
                        and ids[0] != tokenizer.bos_token_id
                    ):
                        if isinstance(ids, list):
                            ids = [tokenizer.bos_token_id] + ids
                        elif isinstance(ids, tuple):
                            ids = (tokenizer.bos_token_id, *ids)
                    return ids

                tokenizer.encode = safe_encode
                tokenizer._matrix_bos_wrapped = True
            return tokenizer, expected

        setattr(tok_mod, "load_verified_tokenizer", safe_load_verified_tokenizer)
        setattr(cs_mod, "load_verified_tokenizer", safe_load_verified_tokenizer)
        if hasattr(eval_ctx_mod, "load_verified_tokenizer"):
            setattr(eval_ctx_mod, "load_verified_tokenizer", safe_load_verified_tokenizer)
        if hasattr(gsm8k_mod, "load_verified_tokenizer"):
            setattr(gsm8k_mod, "load_verified_tokenizer", safe_load_verified_tokenizer)
        setattr(tok_mod, "_matrix_bos_patched", True)

    # 2. Defense-in-depth: patch tokenize_context_task in context_tasks and repository_sft
    if not getattr(ct_mod, "_matrix_bos_patched", False):

        def safe_tokenize_context_task(
            row, spec, encoder, decoder, *, encoder_limit, decoder_limit
        ):
            context = row["context"]
            if row.get("inject_query_prefix") and row.get("instruction"):
                candidate_ctx = format_query_injected_context(context, row.get("instruction"))
            else:
                candidate_ctx = context
            if spec.encoder:
                raw_enc = tuple(encoder.encode(candidate_ctx, add_special_tokens=True))
                enc = (
                    (encoder.bos_token_id, *raw_enc)
                    if (raw_enc and raw_enc[0] != encoder.bos_token_id)
                    else raw_enc
                )
                if len(enc) > encoder_limit and candidate_ctx != context:
                    raw_enc = tuple(encoder.encode(context, add_special_tokens=True))
                    enc = (
                        (encoder.bos_token_id, *raw_enc)
                        if (raw_enc and raw_enc[0] != encoder.bos_token_id)
                        else raw_enc
                    )
            else:
                enc = ()
            messages = [
                {
                    "role": "user",
                    "content": ct_mod.task_prompt(row, paired=spec.encoder is not None),
                }
            ]
            prefix = decoder.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            target_text = row.get("target_text", row.get("target_patch"))
            if not isinstance(target_text, str) or not target_text:
                raise ValueError("empty_sft_target")
            full = decoder.apply_chat_template(
                messages + [{"role": "assistant", "content": target_text}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
            full_ids = tuple(decoder.encode(full.rstrip(), add_special_tokens=False))
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError("prefix_target_boundary")
            target = full_ids[len(prefix_ids) :]
            if (
                not prefix_ids
                or prefix_ids[0] != decoder.bos_token_id
                or not target
                or target[-1] != decoder.convert_tokens_to_ids("<turn|>")
                or (enc and enc[0] != encoder.bos_token_id)
            ):
                raise ValueError("context_task_special_tokens")
            if len(enc) > encoder_limit or len(full_ids) - 1 > decoder_limit:
                raise ValueError("context_task_token_overflow")
            return ct_mod.TokenizedRepairExample(
                row["instance_id"],
                enc,
                prefix_ids,
                target,
                metadata={
                    "source": row["source"],
                    "task_type": row.get("task_type", "repair"),
                },
            )

        setattr(ct_mod, "tokenize_context_task", safe_tokenize_context_task)
        if hasattr(rs_mod, "tokenize_context_task"):
            setattr(rs_mod, "tokenize_context_task", safe_tokenize_context_task)
        setattr(ct_mod, "_matrix_bos_patched", True)

    # 3. Defense-in-depth: patch prompt_ids in evaluation/context.py
    if not getattr(eval_ctx_mod, "_matrix_bos_patched", False):
        orig_prompt_ids = eval_ctx_mod.prompt_ids

        def safe_prompt_ids(row, spec, encoder, decoder, *, encoder_limit):
            enc, prefix = orig_prompt_ids(row, spec, encoder, decoder, encoder_limit=encoder_limit)
            if enc and encoder is not None and enc[0] != encoder.bos_token_id:
                enc = (encoder.bos_token_id, *enc)
                if len(enc) > encoder_limit:
                    raise ValueError("encoder_context_overflow")
            return enc, prefix

        setattr(eval_ctx_mod, "prompt_ids", safe_prompt_ids)
        setattr(eval_ctx_mod, "_matrix_bos_patched", True)


patch_matrix_tokenization()
