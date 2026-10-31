"""Molab Full-Matrix Training and Evaluation Support.

Encapsulates:
1. Matrix specification for the 8 experimental models (Base vs IT encoder,
   symmetric and asymmetric configurations).
2. Architecture: Decoupled full-rank BF16 cross-attention + learnable gates.
3. Two-stage curriculum: 5,551 steps frozen decoder + 5,551 steps joint SFT = 11,102 steps.
4. Resumable checkpointing: checkpoint-<step> with READY.json, manifest.json,
   adapters.safetensors, trainer_state.pt.
5. Hugging Face synchronization to daruokta/bifocalgemma-v2-<model_key>.
6. Complete 8-benchmark evaluation suite without pilot cutoffs.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if "torchvision" not in sys.modules or sys.modules["torchvision"] is not None:
    try:
        import torch
        import torchvision

        torchvision.ops.nms(
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            0.5,
        )
    except Exception:
        sys.modules["torchvision"] = None
        sys.modules["torchvision.transforms"] = None

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from bifocalgemma.config import CheckpointRef, ExperimentSpec
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.curation import file_digest
from bifocalgemma.data.releases import fetch_collection
from bifocalgemma.evaluation.context import V2_EVALUATION_PROTOCOL, run_context_benchmark
from bifocalgemma.models import inference_cache as inference_cache_mod
from bifocalgemma.models.memory_efficient import (
    TiledMergedAttention,
    attention_tile,
    configure_memory_efficient_training,
    tiled_attention,
)
from bifocalgemma.training.benchmark_upload import BenchmarkUploadQueue
from bifocalgemma.training.curated_sft import prepare_curated_training
from bifocalgemma.training.hub import recover_remote_run
from bifocalgemma.training.memory_probe import ProbeLoRALinear
from bifocalgemma.training.offline_sft import evaluate_sft_loss
from bifocalgemma.training.policy import configure_policy_model, update_policy_learning_rates
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.stages import (
    StageLedger,
    atomic_json,
    latest_checkpoint,
    validate_checkpoint_files,
)
from bifocalgemma.training.telemetry import TrainingDashboard
from bifocalgemma.training.v2 import benchmark_directory_name, download_model, target_fingerprint
from bifocalgemma.training.v2_hub import RemoteHeartbeat, synchronize_v2


def patch_matrix_tokenization() -> None:
    """Ensure that IT encoders prepend BOS token during tokenization, matching Base encoder invariants.

    The official Gemma IT tokenizer configuration does not include a BOS token in single-sequence
    post-processing (since IT models usually rely on chat templates). In BifocalGemma, however, the
    encoder context must start with bos_token_id (2) to satisfy the architecture and data contracts.
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

        tok_mod.load_verified_tokenizer = safe_load_verified_tokenizer
        cs_mod.load_verified_tokenizer = safe_load_verified_tokenizer
        if hasattr(eval_ctx_mod, "load_verified_tokenizer"):
            eval_ctx_mod.load_verified_tokenizer = safe_load_verified_tokenizer
        if hasattr(gsm8k_mod, "load_verified_tokenizer"):
            gsm8k_mod.load_verified_tokenizer = safe_load_verified_tokenizer
        tok_mod._matrix_bos_patched = True

    # 2. Defense-in-depth: patch tokenize_context_task in context_tasks and repository_sft
    if not getattr(ct_mod, "_matrix_bos_patched", False):

        def safe_tokenize_context_task(
            row, spec, encoder, decoder, *, encoder_limit, decoder_limit
        ):
            context = row["context"]
            if spec.encoder:
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

        ct_mod.tokenize_context_task = safe_tokenize_context_task
        rs_mod.tokenize_context_task = safe_tokenize_context_task
        ct_mod._matrix_bos_patched = True

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

        eval_ctx_mod.prompt_ids = safe_prompt_ids
        eval_ctx_mod._matrix_bos_patched = True


patch_matrix_tokenization()

DATASET_REPO = "daruokta/bifocalgemma-data"
DATASET_REVISION = "7972f95d7c87f163c8b5af6a4acd308c0c1d0e23"
COLLECTION = "mixed_context_enc32k_dec8k_v3"

TOTAL_STEPS = 11102
STAGE1_STEPS = 5551
STAGE2_STEPS = 5551
EFFECTIVE_BATCH = 4
BATCH_SIZE = 1
ACCUMULATION = 4
TOTAL_EXAMPLES = 44408

MATRIX_MODELS: dict[str, dict[str, Any]] = {
    "2base-2it": {
        "title": "BifocalGemma · 2Base → 2IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E2B",
        "encoder_revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "encoder_size": "E2B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-2base-2it",
    },
    "2it-2it": {
        "title": "BifocalGemma · 2IT → 2IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "encoder_size": "E2B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-2it-2it",
    },
    "4base-4it": {
        "title": "BifocalGemma · 4Base → 4IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E4B",
        "encoder_revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "encoder_size": "E4B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-4base-4it",
    },
    "4it-4it": {
        "title": "BifocalGemma · 4IT → 4IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "encoder_size": "E4B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-4it-4it",
    },
    "4base-2it": {
        "title": "BifocalGemma · 4Base → 2IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E4B",
        "encoder_revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "encoder_size": "E4B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-4base-2it",
    },
    "4it-2it": {
        "title": "BifocalGemma · 4IT → 2IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "encoder_size": "E4B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-4it-2it",
    },
    "2base-4it": {
        "title": "BifocalGemma · 2Base → 4IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E2B",
        "encoder_revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "encoder_size": "E2B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-2base-4it",
    },
    "2it-4it": {
        "title": "BifocalGemma · 2IT → 4IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "encoder_size": "E2B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-v2-2it-4it",
    },
}

BENCHMARKS: list[dict[str, Any]] = [
    {
        "collection": "humanevalfix_python_v2",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 8192,
    },
    {
        "collection": "sweqa_oracle_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
    },
    {
        "collection": "sweqa_noisy_oracle_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
    },
    {
        "collection": "crosscodelong_chunk_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
    },
    {
        "collection": "crosscodelong_function_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
    },
    {
        "collection": "longcodeqa_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
    },
    {
        "collection": "gsm8k_zero_shot_v2",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 8192,
    },
    {
        "collection": "gsm8k_eight_shot_v2",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 8192,
    },
]


def make_experiment_spec(model_key: str) -> ExperimentSpec:
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    encoder_ref = CheckpointRef(
        model_id=cfg["encoder_id"],
        revision=cfg["encoder_revision"],
        size=cfg["encoder_size"],
        role=cfg["encoder_role"],
    )
    decoder_ref = CheckpointRef(
        model_id=cfg["decoder_id"],
        revision=cfg["decoder_revision"],
        size=cfg["decoder_size"],
        role=cfg["decoder_role"],
    )
    name = f"BifocalGemma-{cfg['encoder_size']}-{cfg['decoder_size']}-{model_key}"
    return ExperimentSpec(
        name=name,
        encoder=encoder_ref,
        decoder=decoder_ref,
        attention_mode="merged",
        tie_encoder_decoder_words=False,
        training_recipe="direct_sft",
        distillation=False,
        share_ple=False,
        architecture_version="normalized_merged_v2",
    )


def is_cross_bridge_parameter(name: str) -> bool:
    """Return True if parameter belongs to encoder_projection or the full-rank cross-attention bridge."""
    return name.startswith("encoder_projection.") or any(
        tok in name
        for tok in (
            ".cross_k_proj.",
            ".cross_v_proj.",
            ".cross_k_norm.",
            ".cross_v_norm.",
            ".cross_q_norm.",
            ".cross_gate",
        )
    )


def configure_full_rank_cross_attention(model: Any) -> dict[str, Any]:
    """Attach full-rank 16-bit (bfloat16 compute) cross-attention K/V projections, unrotated cross-query norm, and learnable gates."""
    orig_cached_cls = getattr(
        inference_cache_mod, "_orig_cached_attention", inference_cache_mod.CachedAttention
    )
    setattr(inference_cache_mod, "_orig_cached_attention", orig_cached_cls)

    class Bf16FullLinear(torch.nn.Module):
        """Full-rank linear projection with 16-bit bfloat16 compute and FP32 master weight."""

        def __init__(self, base_layer: torch.nn.Module) -> None:
            super().__init__()
            base: Any = getattr(base_layer, "base", base_layer)
            w = base.weight.detach().float().clone()
            self.in_features = int(w.shape[1])
            self.out_features = int(w.shape[0])
            self.weight = torch.nn.Parameter(w)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            compute_dtype = (
                torch.bfloat16
                if (x.device.type == "cuda" or x.dtype == torch.bfloat16)
                else x.dtype
            )
            return F.linear(x.to(compute_dtype), self.weight.to(compute_dtype)).to(x.dtype)

    class FullCrossTiledAttention(TiledMergedAttention):
        """Decoupled causal self-attention and unrotated full-rank 16-bit gated cross-attention."""

        def forward(
            self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Any,
            shared_kv_states: dict[str, Any],
            past_key_values: Any = None,
            *,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_valid: torch.Tensor | None = None,
            decoder_valid: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            if past_key_values is not None:
                raise ValueError("Tiled training attention does not support temporal cache.")
            if encoder_hidden_states is None or encoder_valid is None or decoder_valid is None:
                raise ValueError("Merged attention requires encoder states and validity masks.")
            batch, length, _ = hidden_states.shape
            shape = (batch, length, -1, self.head_dim)
            cos, sin = position_embeddings
            raw_q = self.q_proj(hidden_states).view(shape)
            query_self = apply_rotary_pos_emb(
                self.q_norm(raw_q), cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            query_cross = self.cross_q_norm(raw_q).to(raw_q.dtype).transpose(1, 2)

            has_encoder_tokens = encoder_hidden_states.shape[1] > 0 and bool(encoder_valid.any())
            if self.is_kv_shared_layer:
                key_self, value_self, cross_key, cross_value = shared_kv_states[self.layer_type]
            else:
                if self.v_proj is None:
                    raise ValueError("Merged branch requires separate value projection.")
                key_self = self.k_norm(self.k_proj(hidden_states).view(shape))
                key_self = apply_rotary_pos_emb(key_self, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                value_self = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
                if has_encoder_tokens:
                    cross_shape = (
                        batch,
                        encoder_hidden_states.shape[1],
                        self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                        self.head_dim,
                    )
                    cross_key = (
                        self.cross_k_norm(
                            self.cross_k_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2)
                    )
                    cross_value = (
                        self.cross_v_norm(
                            self.cross_v_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2)
                    )
                else:
                    cross_key, cross_value = None, None
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (
                    key_self,
                    value_self,
                    cross_key,
                    cross_value,
                )

            chunk_sz = getattr(self, "query_chunk_size", 128)
            backend_name = getattr(self, "attention_backend", "tiled_eager_fp32")
            self_mask = {"valid": decoder_valid, "bidirectional": False}
            self_out = tiled_attention(
                query_self,
                key_self,
                value_self,
                chunk_size=chunk_sz,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                attention_mask=self_mask,
                window=self.sliding_window,
                backend=backend_name,
            )

            if has_encoder_tokens and cross_key is not None and cross_value is not None:
                pieces = []
                for start in range(0, length, chunk_sz):
                    stop = min(length, start + chunk_sz)
                    cross_mask = (
                        encoder_valid[:, None, None, :].bool()
                        & decoder_valid[:, None, start:stop, None].bool()
                    )
                    pieces.append(
                        attention_tile(
                            query_cross[:, :, start:stop],
                            cross_key,
                            cross_value,
                            cross_mask,
                            self.scaling,
                            0.0,
                            True,
                        )
                    )
                cross_out = torch.cat(pieces, dim=2)
                gate_logits = torch.einsum(
                    "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
                ) * (4.0 * (self.head_dim**-0.5))
                query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
                head_gate = torch.tanh(self.cross_gate.float() * 10.0).view(1, -1, 1, 1)
                effective_gate = (head_gate * query_gate).to(self_out.dtype)
                result = self_out + effective_gate * cross_out
            else:
                result = self_out

            result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
            return self.o_proj(result), None

    class FullCrossCachedAttention(orig_cached_cls):
        """Cached inference attention matching FullCrossTiledAttention."""

        def forward(
            self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Any,
            shared_kv_states: dict[str, Any],
            past_key_values: Any = None,
            *,
            inference_cache: Any = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_valid: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, None]:
            if not getattr(self, "full_rank_cross_attention", False):
                return super().forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    shared_kv_states,
                    past_key_values=past_key_values,
                    inference_cache=inference_cache,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_valid=encoder_valid,
                    **kwargs,
                )
            if self.training or torch.is_grad_enabled():
                raise ValueError("Inference cache requires eval and inference_mode/no_grad.")
            batch, length, _ = hidden_states.shape
            shape = (batch, length, -1, self.head_dim)
            cos, sin = position_embeddings
            raw_q = self.q_proj(hidden_states).view(shape)
            query_self = apply_rotary_pos_emb(
                self.q_norm(raw_q), cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            query_cross = self.cross_q_norm(raw_q).to(raw_q.dtype).transpose(1, 2)
            cache = inference_cache
            if cache is None:
                raise ValueError("Explicit inference cache required.")
            positions = torch.arange(cache.offset, cache.offset + length, device=query_self.device)
            if self.is_kv_shared_layer:
                key_self, value_self, cross = shared_kv_states[self.layer_type]
                key_positions, self_length = cache.shared_positions[self.layer_type]
            else:
                raw_key = self.k_proj(hidden_states).view(shape)
                raw_value = (
                    self.v_proj(hidden_states).view(shape) if self.v_proj is not None else raw_key
                )
                key_self = apply_rotary_pos_emb(
                    self.k_norm(raw_key), cos, sin, unsqueeze_dim=2
                ).transpose(1, 2)
                value_self = self.v_norm(raw_value).transpose(1, 2)
                previous = cache.layers.get(self.layer_idx)
                key_positions = positions
                if previous is not None:
                    key_self = torch.cat((previous[0], key_self), dim=2)
                    value_self = torch.cat((previous[1], value_self), dim=2)
                    key_positions = torch.cat((previous[2], positions))
                keep = self.sliding_window if self.sliding_window else key_self.shape[2]
                stored_key, stored_value, stored_positions = (
                    key_self[:, :, -keep:],
                    value_self[:, :, -keep:],
                    key_positions[-keep:],
                )
                cross = previous[3] if previous is not None else None
                if (
                    encoder_hidden_states is not None
                    and encoder_hidden_states.shape[1] > 0
                    and cross is None
                ):
                    cross_shape = (
                        batch,
                        encoder_hidden_states.shape[1],
                        self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                        self.head_dim,
                    )
                    cross = (
                        self.cross_k_norm(
                            self.cross_k_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2),
                        self.cross_v_norm(
                            self.cross_v_proj(encoder_hidden_states).view(cross_shape)
                        )
                        .to(query_self.dtype)
                        .transpose(1, 2),
                    )
                cache.layers[self.layer_idx] = (
                    stored_key,
                    stored_value,
                    stored_positions,
                    cross,
                )
                self_length = key_self.shape[2]
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (key_self, value_self, cross)
                cache.shared_positions[self.layer_type] = (key_positions, self_length)

            self_pieces = []
            for start in range(0, length, 128):
                stop = min(length, start + 128)
                visible = key_positions[None, :] <= positions[start:stop, None]
                if self.sliding_window:
                    visible &= (
                        key_positions[None, :] > positions[start:stop, None] - self.sliding_window
                    )
                allowed = visible[None, None].expand(batch, 1, -1, -1)
                self_pieces.append(
                    F.scaled_dot_product_attention(
                        query_self[:, :, start:stop],
                        key_self,
                        value_self,
                        attn_mask=allowed,
                        dropout_p=0.0,
                        scale=self.scaling,
                        enable_gqa=True,
                    )
                )
            self_out = torch.cat(self_pieces, dim=2)

            if cross is not None and encoder_valid is not None and bool(encoder_valid.any()):
                cross_pieces = []
                for start in range(0, length, 128):
                    stop = min(length, start + 128)
                    cross_mask = (
                        encoder_valid[:, None, None, :].bool().expand(batch, 1, stop - start, -1)
                    )
                    cross_pieces.append(
                        attention_tile(
                            query_cross[:, :, start:stop],
                            cross[0],
                            cross[1],
                            cross_mask,
                            self.scaling,
                            0.0,
                            True,
                        )
                    )
                cross_out = torch.cat(cross_pieces, dim=2)
                gate_logits = torch.einsum(
                    "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
                ) * (4.0 * (self.head_dim**-0.5))
                query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
                head_gate = torch.tanh(self.cross_gate.float() * 10.0).view(1, -1, 1, 1)
                effective_gate = (head_gate * query_gate).to(self_out.dtype)
                result = self_out + effective_gate * cross_out
            else:
                result = self_out

            result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
            return self.o_proj(result), None

    setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)
    kv_owner_layers = 0
    total_cross_params = 0
    for layer in model.decoder.layers:
        attn = layer.self_attn
        q_base = getattr(attn.q_proj, "base", attn.q_proj)
        dev = q_base.weight.device
        num_heads = int(q_base.weight.shape[0] // attn.head_dim)
        attn.cross_q_norm = copy.deepcopy(attn.q_norm).float().to(dev)
        attn.cross_gate = torch.nn.Parameter(
            torch.full((num_heads,), 0.015, dtype=torch.float32, device=dev)
        )
        attn.cross_gate_proj = torch.nn.Parameter(
            torch.zeros((num_heads, attn.head_dim), dtype=torch.float32, device=dev)
        )
        if not attn.is_kv_shared_layer:
            attn.cross_k_proj = Bf16FullLinear(attn.k_proj).to(dev)
            attn.cross_v_proj = Bf16FullLinear(attn.v_proj).to(dev)
            attn.cross_k_norm = copy.deepcopy(attn.k_norm).float().to(dev)
            attn.cross_v_norm = copy.deepcopy(attn.v_norm).float().to(dev)
            kv_owner_layers += 1
        attn.query_chunk_size = getattr(attn, "query_chunk_size", 128)
        attn.attention_backend = getattr(attn, "attention_backend", "tiled_eager_fp32")
        attn.full_rank_cross_attention = True
        attn.__class__ = FullCrossTiledAttention

    for n, p in model.named_parameters():
        if is_cross_bridge_parameter(n):
            total_cross_params += int(p.numel())
    return {
        "mode": "full_rank_cross_bf16",
        "kv_owner_layers": kv_owner_layers,
        "total_decoder_layers": len(model.decoder.layers),
        "total_cross_bridge_parameters": total_cross_params,
    }


def configure_matrix_stage(
    model: Any,
    policy: dict[str, Any],
    *,
    freeze_decoder: bool,
    total_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Configure trainable groups for Stage 1 (frozen decoder) or Stage 2 (joint SFT)."""
    if total_steps < 1:
        raise ValueError("Stage horizon must be a positive integer.")
    default_scaling = policy["alpha"] / policy["rank"]

    for name, module in model.named_modules():
        if isinstance(module, ProbeLoRALinear):
            if name.startswith("decoder."):
                module.scaling = 0.0 if freeze_decoder else default_scaling
            else:
                module.scaling = default_scaling

    for name, param in model.named_parameters():
        if is_cross_bridge_parameter(name):
            param.requires_grad_(True)
        elif name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(True)
        elif name.startswith("decoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(not freeze_decoder)
        else:
            param.requires_grad_(False)

    active_names = (
        ("encoder", "alignment") if freeze_decoder else ("encoder", "decoder", "alignment")
    )
    entries: dict[str, list[tuple[str, Any]]] = {name: [] for name in active_names}
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        if is_cross_bridge_parameter(name):
            entries["alignment"].append((name, param))
        elif name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            entries["encoder"].append((name, param))
        elif (
            not freeze_decoder
            and name.startswith("decoder.")
            and name.endswith((".lora_A", ".lora_B"))
        ):
            entries["decoder"].append((name, param))
        else:
            raise ValueError(f"Unexpected trainable parameter in matrix stage: {name}")

    groups: list[dict[str, Any]] = []
    for group_name in active_names:
        if not entries[group_name]:
            raise ValueError(f"Empty trainable group in matrix stage: {group_name}")
        settings = policy["groups"][group_name]
        warmup_ratio = settings.get("warmup_ratio", 0.03)
        warmup_steps = min(
            total_steps, max(settings.get("warmup_steps", 0), math.ceil(total_steps * warmup_ratio))
        )
        groups.append(
            {
                "name": group_name,
                "params": [p for _, p in entries[group_name]],
                "param_names": [n for n, _ in entries[group_name]],
                "lr": settings["lr"],
                "base_lr": settings["lr"],
                "warmup_steps": warmup_steps,
                "scheduler": settings.get("scheduler", "cosine"),
                "total_steps": total_steps,
                "warmup_ratio": warmup_ratio,
                "min_lr_ratio": settings.get("min_lr_ratio", 0.1),
            }
        )
    summary = {
        "freeze_decoder": freeze_decoder,
        "active_groups": list(active_names),
        "trainable_parameters": {
            g["name"]: sum(int(p.numel()) for p in g["params"]) for g in groups
        },
        "decoder_trainable_tensors": sum(
            1
            for n, p in model.named_parameters()
            if n.startswith("decoder.") and not is_cross_bridge_parameter(n) and p.requires_grad
        ),
    }
    return groups, summary


def save_matrix_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_dir: Path,
    global_step: int,
    stage_name: str,
    compatibility: dict[str, Any],
    training_state: dict[str, Any],
) -> Path:
    """Save checkpoint with READY.json, manifest.json, adapters.safetensors, and trainer_state.pt."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    stage_dir = checkpoint_dir.with_name("." + checkpoint_dir.name + ".tmp")
    stage_dir.mkdir(parents=True, exist_ok=True)

    parameters = {
        name: param.detach().cpu().contiguous()
        for name, param in model.named_parameters()
        if is_cross_bridge_parameter(name) or name.endswith((".lora_A", ".lora_B"))
    }
    save_file(parameters, str(stage_dir / "adapters.safetensors"))

    trainer_payload = {
        "optimizer": optimizer.state_dict(),
        "training_state": {**training_state, "global_step": global_step, "stage": stage_name},
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(trainer_payload, stage_dir / "trainer_state.pt")

    manifest = {
        "format": "bifocalgemma_full_matrix_v1",
        "global_step": global_step,
        "stage": stage_name,
        "compatibility": compatibility,
        "trainable_names": sorted(parameters),
    }
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ready_marker = {
        name: file_digest(stage_dir / name)
        for name in ("adapters.safetensors", "trainer_state.pt", "manifest.json")
    }
    (stage_dir / "READY.json").write_text(json.dumps(ready_marker, indent=2), encoding="utf-8")

    from bifocalgemma.training.stages import retry_windows_file_operation

    retry_windows_file_operation(lambda: stage_dir.rename(checkpoint_dir))
    return checkpoint_dir


def restore_matrix_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Restore checkpoint and verified parameters; load trainer state and RNG."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    validate_checkpoint_files(checkpoint_dir)
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))

    weights = load_file(str(checkpoint_dir / "adapters.safetensors"), device="cpu")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in weights.items():
            if name in parameters:
                parameters[name].copy_(tensor.to(parameters[name].device))
            else:
                raise ValueError(f"Adapter parameter {name} not found in model.")

    trainer_payload = torch.load(
        checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    if "optimizer" in trainer_payload and optimizer is not None:
        try:
            optimizer.load_state_dict(trainer_payload["optimizer"])
        except Exception:
            # Group structure may vary between stages (e.g. Stage 1 -> Stage 2 adds decoder).
            pass
    if "python_rng" in trainer_payload:
        random.setstate(trainer_payload["python_rng"])
    if "torch_rng" in trainer_payload:
        torch.set_rng_state(trainer_payload["torch_rng"])
    if trainer_payload.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(trainer_payload["cuda_rng"])

    return {
        "global_step": manifest["global_step"],
        "stage": manifest.get("stage", "unknown"),
        "training_state": trainer_payload.get("training_state", {}),
        "manifest": manifest,
    }


def prune_matrix_checkpoints(train_dir: Path, keep_limit: int = 2) -> list[str]:
    """Keep the milestone checkpoint-005551 and the last keep_limit checkpoints."""
    train_dir = Path(train_dir).resolve()
    candidates = sorted(
        [p for p in train_dir.glob("checkpoint-*") if (p / "READY.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if len(candidates) <= keep_limit:
        return [p.name for p in candidates]
    milestone = {
        p
        for p in candidates
        if p.name in {f"checkpoint-{STAGE1_STEPS:06d}", f"checkpoint-{TOTAL_STEPS:06d}"}
    }
    latest = set(candidates[-keep_limit:])
    keep = milestone | latest
    for p in candidates:
        if p not in keep:
            try:
                shutil.rmtree(p)
            except Exception:
                pass
    return sorted(p.name for p in keep)


def load_matrix_adapters_for_inference(model: Any, checkpoint_dir: Path | str) -> dict[str, Any]:
    """Load adapters for post-training benchmark evaluation; unfreeze scaling."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    manifest = validate_checkpoint_files(checkpoint_dir)
    safetensors_path = checkpoint_dir / "adapters.safetensors"
    weights = load_file(str(safetensors_path), device="cpu")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in weights.items():
            if name in parameters:
                parameters[name].copy_(tensor.to(parameters[name].device))
            else:
                raise ValueError(f"Unknown adapter parameter in checkpoint: {name}")

    policy = manifest.get("compatibility", {}).get("training_policy", {})
    alpha = policy.get("alpha", 16)
    rank = policy.get("rank", 16)
    scaling = alpha / rank if rank else 1.0
    for module in model.modules():
        if isinstance(module, ProbeLoRALinear):
            module.scaling = scaling

    if hasattr(model, "_tie_words"):
        model._tie_words()
    model.eval()
    return manifest


def run_matrix_stage(
    model: Any,
    train_examples: list[Any],
    dev_examples: list[Any],
    stage_name: str,
    *,
    policy: dict[str, Any],
    max_steps: int,
    freeze_decoder: bool,
    run_dir: Path,
    compatibility: dict[str, Any],
    start_global_step: int,
    batch_size: int = 1,
    accumulation: int = 4,
    clip_norm: float = 1.0,
    seed: int = 17,
    progress: Any = None,
    checkpoint_callback: Any = None,
    save_interval_seconds: float = 900.0,
    save_every_steps: int = 500,
    resume_from_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    """Execute one training stage (Stage 1 frozen or Stage 2 joint) with AdamW."""
    run_dir = Path(run_dir).resolve()
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    groups, stage_summary = configure_matrix_stage(
        model, policy, freeze_decoder=freeze_decoder, total_steps=max_steps
    )
    opt_cfg = policy.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        eps=float(opt_cfg.get("eps", 1e-8)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        foreach=False,
    )
    trainable_params = [p for g in groups for p in g["params"]]

    decoder_snapshot = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if n.startswith("decoder.")
        and not is_cross_bridge_parameter(n)
        and n.endswith((".lora_A", ".lora_B"))
    }

    start_step = 0
    order = list(range(len(train_examples)))
    random.Random(seed + (1 if stage_name == "joint" else 0)).shuffle(order)
    cursor = 0
    epoch = 0
    history: list[dict[str, Any]] = []

    if resume_from_checkpoint is not None:
        restored = restore_matrix_checkpoint(model, optimizer, Path(resume_from_checkpoint))
        state = restored.get("training_state", {})
        if state.get("stage") == stage_name:
            start_step = state.get("step", 0)
            order = state.get("order", order)
            cursor = state.get("cursor", cursor)
            epoch = state.get("epoch", epoch)
            history = state.get("history", [])

    model.train()
    last_save_time = time.monotonic()

    for step in range(start_step, max_steps):
        global_step = start_global_step + step + 1
        step_started = time.perf_counter()

        micro_batches = []
        for _ in range(accumulation):
            if cursor >= len(order):
                epoch += 1
                order = list(range(len(train_examples)))
                random.Random(seed + epoch).shuffle(order)
                cursor = 0
            indices = order[cursor : cursor + batch_size]
            cursor += len(indices)
            items = [train_examples[i] for i in indices]
            micro_batches.append(RepairCollator()(items))

        token_count = sum(int((b.labels != -100).sum()) for b in micro_batches)
        if not token_count:
            token_count = 1

        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for batch_cpu in micro_batches:
            batch = batch_cpu.to(device)
            count = int((batch_cpu.labels != -100).sum())
            scale = count / token_count if token_count else (1.0 / len(micro_batches))
            output = model.forward_loss(batch) if hasattr(model, "forward_loss") else model(batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError(f"Nonfinite loss at step {step + 1} (global {global_step})")
            (output.loss * scale).backward()
            mean_loss += output.loss.detach().item() * scale
            del output, batch

        if freeze_decoder:
            leaked = [
                n
                for n, p in model.named_parameters()
                if n.startswith("decoder.")
                and not is_cross_bridge_parameter(n)
                and p.grad is not None
                and bool(p.grad.abs().sum() > 0)
            ]
            if leaked:
                raise RuntimeError(
                    f"Frozen decoder received nonzero parameter gradients: {leaked[:3]}"
                )

        grads = [p.grad for p in trainable_params if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError(f"Missing or nonfinite gradients at step {global_step}.")

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm, foreach=False)
        )
        rates = update_policy_learning_rates(optimizer, step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda" and torch.cuda.memory_reserved(device) > 28 * 2**30:
            torch.cuda.empty_cache()

        step_seconds = time.perf_counter() - step_started
        row = {
            "step": global_step,
            "stage_step": step + 1,
            "stage": stage_name,
            "freeze_decoder": freeze_decoder,
            "max_steps": TOTAL_STEPS,
            "train_loss": mean_loss,
            "target_tokens": token_count,
            "gradient_norm": grad_norm,
            "learning_rates": rates,
            "step_seconds": step_seconds,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            ),
        }
        history.append(row)

        if progress:
            progress({"stage": "matrix_training_step", **row})

        is_stage_end = step + 1 == max_steps
        is_interval_step = global_step % save_every_steps == 0
        is_time_up = time.monotonic() - last_save_time >= save_interval_seconds

        if is_stage_end or is_interval_step or is_time_up:
            ckpt_path = save_matrix_checkpoint(
                model,
                optimizer,
                train_dir / f"checkpoint-{global_step:06d}",
                global_step=global_step,
                stage_name=stage_name,
                compatibility=compatibility,
                training_state={
                    "step": step + 1,
                    "global_step": global_step,
                    "order": order,
                    "cursor": cursor,
                    "epoch": epoch,
                    "history": history[-100:],
                },
            )
            last_save_time = time.monotonic()
            prune_matrix_checkpoints(train_dir, keep_limit=2)
            if checkpoint_callback:
                checkpoint_callback(ckpt_path)

    if freeze_decoder:
        max_decoder_delta = max(
            (
                float((model.get_parameter(n).detach().cpu() - before_t).abs().max())
                for n, before_t in decoder_snapshot.items()
            ),
            default=0.0,
        )
        if max_decoder_delta != 0.0:
            raise RuntimeError(f"Frozen decoder weights mutated during stage: {max_decoder_delta}")

    held_out = None
    if dev_examples:
        held_out = evaluate_sft_loss(
            model, dev_examples, batch_size=batch_size, device=device, event_callback=progress
        )

    return {
        "stage": stage_name,
        "completed_steps": max_steps,
        "end_global_step": start_global_step + max_steps,
        "held_out_loss": held_out,
        "last_checkpoint": str(train_dir / f"checkpoint-{(start_global_step + max_steps):06d}"),
    }


def run_matrix_training(
    model_key: str,
    *,
    token: str,
    mo: Any = None,
    output_root: str | Path = "outputs/v2",
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
) -> dict[str, Any]:
    """Execute complete two-stage matrix training run (5,551 frozen + 5,551 joint = 11,102 steps)."""
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    policy = unified_recipe(cfg["family"])
    repo = cfg["repo"]

    root = Path(output_root) / model_key / "train"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{cfg['family']}/{spec.name}/train"

    dashboard = (
        monitor
        if monitor is not None
        else TrainingDashboard(mo, experiment=cfg["title"], device="cuda")
    )
    dashboard.run_dir = root

    if upload and token:
        recover_remote_run(repo, root, prefix, token=token)

    identity = {
        "format": "bifocalgemma_pipeline_v1",
        "version": "normalized_merged_v2",
        "spec": spec.to_dict(),
        "policy": policy,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "collection": COLLECTION,
        "run_name": "train",
        "seed": 17,
        "total_steps": TOTAL_STEPS,
        "stage1_steps": STAGE1_STEPS,
        "stage2_steps": STAGE2_STEPS,
        "model_key": model_key,
    }
    ledger = StageLedger(root, identity)
    if not ledger.path.exists():
        atomic_json(ledger.path, ledger.state)

    heartbeat = RemoteHeartbeat(repo, root, prefix, token=token) if upload and token else None
    queue = BenchmarkUploadQueue(
        root,
        backoff_path=root / ".upload-backoff.json",
        progress=dashboard,
        interval=1800,
        label="Checkpoint / results",
    )

    def progress(event: dict[str, Any]):
        event = {
            **event,
            "upload_pending": queue.state.get("pending", False),
            "final_synced": queue.state.get("final_synced", False),
        }
        dashboard(event)
        if heartbeat:
            heartbeat(
                event,
                force=event.get("stage")
                in {"training_completed", "benchmark_completed", "stage_completed"},
            )

    def snapshot(*, final: bool = False):
        if not upload or not token:
            return None
        queue.mark_pending()
        result = queue.attempt(
            lambda: synchronize_v2(repo, root, prefix, token=token, require_checkpoint=True),
            final=final,
        )
        if result and heartbeat:
            dashboard(
                {
                    "stage": "remote_snapshot_verified",
                    "message": "HF verified @ " + result["revision"],
                    "last_remote_checkpoint": result["retained_checkpoints"][-1]
                    if result["retained_checkpoints"]
                    else None,
                    "remote_revision": result["revision"],
                    "upload_pending": False,
                }
            )
            heartbeat(
                {
                    "remote_revision": result["revision"],
                    "last_remote_checkpoint": result["retained_checkpoints"][-1]
                    if result["retained_checkpoints"]
                    else None,
                    "upload_pending": False,
                    "final_synced": final,
                },
                force=final,
            )
        return result

    train_dir = root / "training"
    existing_ckpt = latest_checkpoint(train_dir)
    completed_report_path = root / "training/training_report.json"
    if completed_report_path.is_file() and existing_ckpt:
        rep = json.loads(completed_report_path.read_text(encoding="utf-8"))
        if rep.get("global_step") == TOTAL_STEPS:
            progress({"stage": "training_completed", "step": TOTAL_STEPS})
            snapshot(final=True)
            return {
                "status": "COMPLETED",
                "model_key": model_key,
                "global_step": TOTAL_STEPS,
                "message": "Full matrix training already completed.",
            }

    progress({"stage": "starting_training", "message": f"Starting {cfg['title']} ({model_key})"})

    # Download model backbone
    model, provenance = download_model(spec, cache, token, progress)
    configure_policy_model(model, copy.deepcopy(policy))
    configure_memory_efficient_training(model)
    cross_info = configure_full_rank_cross_attention(model)
    progress({"stage": "configured_cross_attention", **cross_info})

    # Prepare curated dataset
    pool = Path(cache) / "dataset" / COLLECTION
    fetch_collection(
        DATASET_REPO, DATASET_REVISION, COLLECTION, pool, expected_role="train", token=token
    )
    train_data, dev_data, manifest = prepare_curated_training(
        spec,
        pool,
        Path(cache) / "tokenizers",
        packed_cache=Path(cache) / "packed",
        progress=progress,
    )
    manifest["matched_target_fingerprints"] = {
        "train": target_fingerprint(train_data),
        "dev": target_fingerprint(dev_data),
    }

    compatibility = {
        "architecture": "decoupled_full_cross_bf16_gated_v1",
        "experiment": spec.name,
        "model_key": model_key,
        "model_spec": spec.to_dict(),
        "sources": provenance,
        "data_fingerprint": manifest["fingerprint"],
        "matched_target_fingerprints": manifest["matched_target_fingerprints"],
        "seed": 17,
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
        "training_policy": policy,
        "memory_policy": model.memory_policy,
        "total_steps": TOTAL_STEPS,
        "stage1_steps": STAGE1_STEPS,
        "stage2_steps": STAGE2_STEPS,
    }

    # Determine resume point
    resume_ckpt = latest_checkpoint(train_dir)
    current_global_step = 0
    if resume_ckpt:
        manifest_meta = json.loads(
            (Path(resume_ckpt) / "manifest.json").read_text(encoding="utf-8")
        )
        current_global_step = manifest_meta.get("global_step", 0)

    stage1_report = None
    if current_global_step < STAGE1_STEPS:
        progress({"stage": "starting_stage1_frozen", "step": current_global_step})
        stage1_report = run_matrix_stage(
            model,
            train_data,
            dev_data,
            stage_name="frozen_decoder",
            policy=copy.deepcopy(policy),
            max_steps=STAGE1_STEPS,
            freeze_decoder=True,
            run_dir=root,
            compatibility=compatibility,
            start_global_step=0,
            batch_size=BATCH_SIZE,
            accumulation=ACCUMULATION,
            clip_norm=1.0,
            seed=17,
            progress=progress,
            checkpoint_callback=lambda p: snapshot(),
            resume_from_checkpoint=resume_ckpt if current_global_step > 0 else None,
        )
        progress({"stage": "stage1_completed", "last_checkpoint": stage1_report["last_checkpoint"]})
        stage1_files = [p for p in Path(stage1_report["last_checkpoint"]).glob("*") if p.is_file()]
        ledger.finish("stage1_frozen", stage1_files, stage1_report)
        snapshot(final=True)
        current_global_step = STAGE1_STEPS
        resume_ckpt = stage1_report["last_checkpoint"]

    stage2_report = None
    if current_global_step < TOTAL_STEPS:
        progress({"stage": "starting_stage2_joint", "step": current_global_step})
        stage2_steps_remaining = TOTAL_STEPS - STAGE1_STEPS
        stage2_report = run_matrix_stage(
            model,
            train_data,
            dev_data,
            stage_name="joint",
            policy=copy.deepcopy(policy),
            max_steps=stage2_steps_remaining,
            freeze_decoder=False,
            run_dir=root,
            compatibility=compatibility,
            start_global_step=STAGE1_STEPS,
            batch_size=BATCH_SIZE,
            accumulation=ACCUMULATION,
            clip_norm=1.0,
            seed=17,
            progress=progress,
            checkpoint_callback=lambda p: snapshot(),
            resume_from_checkpoint=resume_ckpt,
        )
        progress({"stage": "stage2_completed", "last_checkpoint": stage2_report["last_checkpoint"]})
        stage2_files = [p for p in Path(stage2_report["last_checkpoint"]).glob("*") if p.is_file()]
        ledger.finish("stage2_joint", stage2_files, stage2_report)
        snapshot(final=True)

    final_report = {
        "status": "COMPLETED",
        "model_key": model_key,
        "spec": spec.to_dict(),
        "global_step": TOTAL_STEPS,
        "provenance": provenance,
        "stage1": stage1_report,
        "stage2": stage2_report,
        "last_checkpoint": str(train_dir / f"checkpoint-{TOTAL_STEPS:06d}"),
    }
    atomic_json(root / "training/training_report.json", final_report)
    snapshot(final=True)
    progress({"stage": "training_completed", "step": TOTAL_STEPS})

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_report


def run_matrix_benchmark(
    model_key: str,
    *,
    token: str,
    mo: Any = None,
    output_root: str | Path = "outputs/v2",
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
) -> dict[str, Any]:
    """Execute complete 8-benchmark suite without pilot cutoffs on the trained matrix model."""
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    policy = unified_recipe(cfg["family"])
    repo = cfg["repo"]

    root = Path(output_root) / model_key / "train"
    prefix = f"{cfg['family']}/{spec.name}/train"

    dashboard = (
        monitor
        if monitor is not None
        else TrainingDashboard(mo, experiment=cfg["title"], device="cuda")
    )
    dashboard.run_dir = root

    if upload and token:
        recover_remote_run(repo, root, prefix, token=token)

    identity = {
        "format": "bifocalgemma_pipeline_v1",
        "version": "normalized_merged_v2",
        "spec": spec.to_dict(),
        "policy": policy,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "collection": COLLECTION,
        "run_name": "train",
        "seed": 17,
        "total_steps": TOTAL_STEPS,
        "stage1_steps": STAGE1_STEPS,
        "stage2_steps": STAGE2_STEPS,
        "model_key": model_key,
    }
    ledger = StageLedger(root, identity)
    if not ledger.path.exists():
        atomic_json(ledger.path, ledger.state)

    heartbeat = RemoteHeartbeat(repo, root, prefix, token=token) if upload and token else None
    queue = BenchmarkUploadQueue(
        root,
        backoff_path=root / ".upload-backoff.json",
        progress=dashboard,
        interval=1800,
        label="Benchmark upload",
    )

    def progress(event: dict[str, Any]):
        dashboard(event)
        if heartbeat:
            heartbeat(event, force=event.get("stage") in {"benchmark_completed"})

    def snapshot(*, final: bool = False):
        if not upload or not token:
            return None
        queue.mark_pending()
        result = queue.attempt(
            lambda: synchronize_v2(repo, root, prefix, token=token, require_checkpoint=True),
            final=final,
        )
        if result and heartbeat:
            heartbeat(
                {
                    "remote_revision": result["revision"],
                    "upload_pending": False,
                    "final_synced": final,
                },
                force=final,
            )
        return result

    checkpoint_path = latest_checkpoint(root / "training")
    if not checkpoint_path:
        raise ValueError(f"No completed training checkpoint found in {root / 'training'}.")

    checkpoint_identity = json.loads(
        (Path(checkpoint_path) / "READY.json").read_text(encoding="utf-8")
    )

    reports = []
    missing_jobs = []

    for item in BENCHMARKS:
        settings = {
            "benchmark": item,
            "execution": True,
            "protocol": V2_EVALUATION_PROTOCOL,
            "snapshot": "trained",
        }
        dest = root / "benchmarks" / benchmark_directory_name(settings)
        report_file = dest / "benchmark_report.json"
        if report_file.is_file():
            try:
                rep = json.loads(report_file.read_text(encoding="utf-8"))
                reports.append(rep)
                continue
            except Exception:
                pass
        missing_jobs.append((item, settings, dest))

    if not missing_jobs:
        snapshot(final=True)
        progress(
            {
                "stage": "benchmark_completed",
                "items_done": len(reports),
                "items_total": len(BENCHMARKS),
            }
        )
        return {
            "status": "COMPLETED",
            "model": spec.name,
            "repo": repo,
            "benchmarks": reports,
        }

    # Load model for inference
    model, _ = download_model(spec, cache, token, progress)
    configure_policy_model(model, copy.deepcopy(policy))
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    load_matrix_adapters_for_inference(model, checkpoint_path)

    for item, settings, destination in missing_jobs:
        collection_dir = Path(cache) / "benchmarks" / item["collection"]
        fetch_collection(
            DATASET_REPO,
            DATASET_REVISION,
            item["collection"],
            collection_dir,
            expected_role="eval",
            include_gold=True,
            token=token,
        )
        report = run_context_benchmark(
            model,
            spec,
            collection_dir,
            destination,
            Path(cache) / "tokenizers",
            checkpoint_identity=checkpoint_identity,
            encoder_limit=item["encoder_limit"],
            decoder_limit=item["decoder_limit"],
            max_new_tokens=item["max_new_tokens"],
            allow_execution=True,
            progress=progress,
            prediction_callback=lambda: snapshot(),
            upload_every=100,
            use_cache=True,
        )
        reports.append(report)
        ledger.finish(
            f"benchmark_{item['collection']}",
            [
                destination / "benchmark_report.json",
                destination / "predictions.jsonl",
                destination / "evaluation_state.json",
            ],
            report,
            settings=settings,
        )
        snapshot(final=False)

    snapshot(final=True)
    progress(
        {
            "stage": "benchmark_completed",
            "items_done": len(reports),
            "items_total": len(BENCHMARKS),
        }
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "COMPLETED" if not queue.state.get("pending") else "PENDING_UPLOAD",
        "model": spec.name,
        "repo": repo,
        "benchmarks": reports,
    }


def run_matrix_training_and_benchmark(
    model_key: str,
    *,
    token: str,
    mo: Any = None,
    output_root: str | Path = "outputs/v2",
    cache: str | Path = "outputs/v2_cache",
    upload: bool = True,
    monitor: Any = None,
) -> dict[str, Any]:
    """Notebook entry point: execute matrix training curriculum, then run 8-benchmark suite."""
    kwargs = dict(
        token=token,
        mo=mo,
        output_root=output_root,
        cache=cache,
        upload=upload,
        monitor=monitor,
    )
    training_res = run_matrix_training(model_key, **kwargs)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluation_res = run_matrix_benchmark(model_key, **kwargs)
    return {
        "status": evaluation_res["status"],
        "model": evaluation_res["model"],
        "model_key": model_key,
        "repo": evaluation_res["repo"],
        "training": training_res,
        "benchmarks": evaluation_res["benchmarks"],
    }
