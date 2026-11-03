"""Molab Full-Matrix (10-Model) V3 Training and Evaluation Support.

Encapsulates:
1. Unified 10-model matrix specification: 8 paired encoder-decoder models
   (Base vs IT encoder, symmetric and asymmetric) + 2 decoder-only IT baselines
   (`e2b-it`, `e4b-it`), all synchronized to centralized Hugging Face repository
   `daruokta/bifocalgemma-2026-11-02` under model-key subfolders.
2. BifocalGemma V3 Architecture Upgrades (`docs/BIFOCALGEMMA_V3_RESEARCH_PROPOSAL.md`):
   - `SwiGLUDualGateBridge`: Non-linear manifold warping bridge with Pre/Post RMSNorm,
     orthogonal residual shortcut, and learned scale calibration gain `gamma_calib`.
   - Decoupled full-rank BF16 cross-attention with stationary unrotated cross-coordinates,
     non-zero `atanh(0.05)` gate initialization, critical length temperature scaling,
     and static cross-KV caching.
   - `StreamingPointerLoss`: Zero-materialization streaming LogSumExp pointer NLL.
   - Normalized differential diff token weighting (`differential_diff_cross_entropy`).
   - Partitioned AdamW weight decay (0.0 on RMSNorm, cross-attention, and LoRA; 1e-4 on bridge).
   - Hardware memory guards (`expandable_segments:True` and Triton sm86 tile clamping).
3. Anti-catastrophic-forgetting two-stage curriculum (`build_anti_forgetting_curriculum`):
   - Stage 1 (5,551 steps) + Stage 2 (5,551 steps) = 11,102 steps (`BATCH_SIZE=2`, `ACCUMULATION=2`).
   - Synthesizes and interleaves 15% Code QA / MCQ and exact span-copy comprehension examples
     to prevent catastrophic forgetting on `longcodeqa`, `sweqa`, and `gsm8k`.
4. Reordered 8-benchmark suite (6 full first + 2 heavy deterministic 100-sample last) with
   dynamic optimal evaluation/benchmark batching (`resolve_optimal_eval_batch_size` +
   `run_batched_context_benchmark`) tailored to model size, benchmark weight, and free VRAM.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import shutil
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch
import torch.utils.checkpoint

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if "torchvision" not in sys.modules or sys.modules["torchvision"] is not None:
    try:
        import torchvision

        torchvision.ops.nms(
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
            0.5,
        )
    except Exception:
        cast(dict[str, Any], sys.modules)["torchvision"] = None
        cast(dict[str, Any], sys.modules)["torchvision.transforms"] = None

if "torchaudio" not in sys.modules or sys.modules["torchaudio"] is not None:
    try:
        import torchaudio  # noqa: F401
    except Exception:
        cast(dict[str, Any], sys.modules)["torchaudio"] = None

from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

from bifocalgemma.config import CheckpointRef, ExperimentSpec
from bifocalgemma.contracts import TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.curation import file_digest, stable_fingerprint
from bifocalgemma.data.releases import fetch_collection
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
from bifocalgemma.training.telemetry import TrainingDashboard, resource_snapshot
from bifocalgemma.training.v2 import benchmark_directory_name, download_model, target_fingerprint
from bifocalgemma.training.v2_hub import RemoteHeartbeat


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

CENTRALIZED_HF_REPO = "daruokta/bifocalgemma-2026-11-02"
ARCHIVE_HF_REPO = "daruokta/bifocalgemma-archive"
DATASET_REPO = "daruokta/bifocalgemma-data"
DATASET_REVISION = "7972f95d7c87f163c8b5af6a4acd308c0c1d0e23"
COLLECTION = "mixed_context_enc32k_dec8k_v3"

TOTAL_STEPS = 11102
STAGE1_STEPS = 5551
STAGE2_STEPS = 5551
EFFECTIVE_BATCH = 4
BATCH_SIZE = 2
ACCUMULATION = 2
TOTAL_EXAMPLES = 44408

COMPREHENSION_MIX_RATIO = 0.20
WARMUP_RATIO = 0.05
BENCHMARK_SAMPLE_SEED = 17
HEAVY_BENCHMARK_SAMPLE_LIMIT = 100
GATE_INIT_TANH = 0.05
GATE_INIT_ATANH = math.atanh(GATE_INIT_TANH)

MATRIX_MODELS: dict[str, dict[str, Any]] = {
    "2base-2it": {
        "title": "BifocalGemma V3 · 2Base → 2IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E2B",
        "encoder_revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "encoder_size": "E2B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2base-2it",
    },
    "2it-2it": {
        "title": "BifocalGemma V3 · 2IT → 2IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "encoder_size": "E2B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2it-2it",
    },
    "4base-4it": {
        "title": "BifocalGemma V3 · 4Base → 4IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E4B",
        "encoder_revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "encoder_size": "E4B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4base-4it",
    },
    "4it-4it": {
        "title": "BifocalGemma V3 · 4IT → 4IT (Symmetric)",
        "family": "symmetric",
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "encoder_size": "E4B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4it-4it",
    },
    "4base-2it": {
        "title": "BifocalGemma V3 · 4Base → 2IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E4B",
        "encoder_revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "encoder_size": "E4B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4base-2it",
    },
    "4it-2it": {
        "title": "BifocalGemma V3 · 4IT → 2IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "encoder_size": "E4B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "4it-2it",
    },
    "2base-4it": {
        "title": "BifocalGemma V3 · 2Base → 4IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E2B",
        "encoder_revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "encoder_size": "E2B",
        "encoder_role": "encoder_base",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2base-4it",
    },
    "2it-4it": {
        "title": "BifocalGemma V3 · 2IT → 4IT (Asymmetric)",
        "family": "asymmetric",
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "encoder_size": "E2B",
        "encoder_role": "decoder_it",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "2it-4it",
    },
    "e2b-it": {
        "title": "BifocalGemma V3 · Baseline E2B-IT (Decoder-Only)",
        "family": "decoder_only",
        "encoder_id": None,
        "encoder_revision": None,
        "encoder_size": None,
        "encoder_role": None,
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "e2b-it",
    },
    "e4b-it": {
        "title": "BifocalGemma V3 · Baseline E4B-IT (Decoder-Only)",
        "family": "decoder_only",
        "encoder_id": None,
        "encoder_revision": None,
        "encoder_size": None,
        "encoder_role": None,
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": CENTRALIZED_HF_REPO,
        "subfolder": "e4b-it",
    },
}

# Reordered 8-benchmark suite: 6 faster benchmarks in full first, followed by 2 heavy benchmarks sampled at 100 cases.
BENCHMARKS: list[dict[str, Any]] = [
    {
        "collection": "longcodeqa_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
        "sample_limit": None,
        "weight_tier": "medium_long",
    },
    {
        "collection": "humanevalfix_python_v2",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 8192,
        "sample_limit": None,
        "weight_tier": "medium",
    },
    {
        "collection": "sweqa_oracle_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
        "sample_limit": None,
        "weight_tier": "medium_long",
    },
    {
        "collection": "sweqa_noisy_oracle_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
        "sample_limit": None,
        "weight_tier": "medium_long",
    },
    {
        "collection": "gsm8k_zero_shot_v2",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 8192,
        "sample_limit": None,
        "weight_tier": "light",
    },
    {
        "collection": "gsm8k_eight_shot_v2",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 8192,
        "sample_limit": None,
        "weight_tier": "light",
    },
    {
        "collection": "crosscodelong_chunk_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
        "sample_limit": HEAVY_BENCHMARK_SAMPLE_LIMIT,
        "sample_seed": BENCHMARK_SAMPLE_SEED,
        "weight_tier": "heavy",
    },
    {
        "collection": "crosscodelong_function_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 2048,
        "sample_limit": HEAVY_BENCHMARK_SAMPLE_LIMIT,
        "sample_seed": BENCHMARK_SAMPLE_SEED,
        "weight_tier": "heavy",
    },
]


def make_experiment_spec(model_key: str) -> ExperimentSpec:
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    decoder_ref = CheckpointRef(
        model_id=cfg["decoder_id"],
        revision=cfg["decoder_revision"],
        size=cfg["decoder_size"],
        role=cfg["decoder_role"],
    )
    if cfg["encoder_id"] is None:
        return ExperimentSpec(
            name=f"Baseline-{cfg['decoder_size']}-IT-{model_key}",
            encoder=None,
            decoder=decoder_ref,
            attention_mode="causal",
            tie_encoder_decoder_words=False,
            training_recipe="direct_sft",
            distillation=False,
            share_ple=False,
            architecture_version="normalized_merged_v2",
        )
    encoder_ref = CheckpointRef(
        model_id=cfg["encoder_id"],
        revision=cfg["encoder_revision"],
        size=cfg["encoder_size"],
        role=cfg["encoder_role"],
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


class SwiGLUDualGateBridge(torch.nn.Module):
    """Non-linear manifold warping bridge for BifocalGemma V3.

    Maps bidirectional encoder representations H_enc in R^{d_enc} into the
    causal decoder manifold Z_enc in R^{d_dec} with Pre/Post RMSNorm and
    explicit per-channel scale calibration.
    """

    def __init__(
        self,
        d_enc: int,
        d_dec: int,
        d_mid: int | None = None,
        init_rms_scale: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_enc = int(d_enc)
        self.d_dec = int(d_dec)
        self.d_mid = int(d_mid) if d_mid is not None else int(max(d_enc, d_dec))

        self.pre_norm = torch.nn.RMSNorm(self.d_enc, eps=eps)
        self.w_gate = torch.nn.Linear(self.d_enc, self.d_mid, bias=False)
        self.w_up = torch.nn.Linear(self.d_enc, self.d_mid, bias=False)
        self.w_down = torch.nn.Linear(self.d_mid, self.d_dec, bias=False)

        if self.d_enc != self.d_dec:
            self.w_res: torch.nn.Module = torch.nn.Linear(self.d_enc, self.d_dec, bias=False)
        else:
            self.w_res = torch.nn.Identity()

        self.post_norm = torch.nn.RMSNorm(self.d_dec, eps=eps)
        self.gamma_calib = torch.nn.Parameter(
            torch.full((self.d_dec,), float(init_rms_scale), dtype=torch.float32)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            if isinstance(self.w_res, torch.nn.Linear):
                torch.nn.init.orthogonal_(self.w_res.weight)
            torch.nn.init.xavier_uniform_(self.w_gate.weight, gain=0.5)
            torch.nn.init.xavier_uniform_(self.w_up.weight, gain=0.5)
            torch.nn.init.normal_(self.w_down.weight, mean=0.0, std=0.01 / math.sqrt(self.d_mid))

    def forward(self, h_enc: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        orig_dtype = h_enc.dtype
        compute_dtype = (
            torch.bfloat16
            if (h_enc.device.type == "cuda" or orig_dtype == torch.bfloat16)
            else orig_dtype
        )
        x = h_enc.to(compute_dtype)
        h_tilde = self.pre_norm(x.float()).to(compute_dtype)
        gated = F.silu(F.linear(h_tilde, self.w_gate.weight.to(compute_dtype))) * F.linear(
            h_tilde, self.w_up.weight.to(compute_dtype)
        )
        warped = F.linear(gated, self.w_down.weight.to(compute_dtype))
        if isinstance(self.w_res, torch.nn.Linear):
            res = F.linear(h_tilde, self.w_res.weight.to(compute_dtype))
        else:
            res = h_tilde
        z = res + warped
        normed = self.post_norm(z.float()).to(orig_dtype)
        return normed * self.gamma_calib.to(orig_dtype)


DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5,       # <pad>, <eos>, <bos>, <unk>, <mask>, [multimodal]
    107, 108, 109, 110,     # '\n', '\n\n', '\n\n\n', ...
    138, 140, 236743,       # '  ', '    ', ' ' (space)
    255968,                 # '\t'
    236761, 603,            # '.', ' .'
    236764, 708,            # ',', ' ,'
    236787, 794,            # ':', ' :'
    236793, 1782,           # ';', ' ;'
    236769, 833,            # '(', ' ('
    236768, 1017,           # ')', ' )'
    236840, 1673,           # '[', ' ['
    236842, 1681,           # ']', ' ]'
    236782, 1836,           # '{', ' {'
    236783, 1888,           # '}', ' }'
    236784, 808,            # '=', ' ='
    236862, 900,            # '+', ' +'
    236772, 753,            # '-', ' -'
    236829, 1374,           # '*', ' *'
    236786, 1195,           # '/', ' /'
)


def extract_trivial_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Extract all whitespace, punctuation, and special token IDs from tokenizer vocabulary."""
    if tokenizer is None:
        return DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS
    vocab = getattr(tokenizer, "vocab", None) or (
        tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    )
    if not vocab and hasattr(tokenizer, "tokenizer"):
        vocab = tokenizer.tokenizer.get_vocab()
    if not vocab:
        return DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS
    trivial = set()
    for tok, idx in vocab.items():
        cleaned = tok.replace("\u2581", " ").replace(" ", " ").strip()
        if not cleaned or all(ch in string.punctuation or ch.isspace() for ch in cleaned):
            trivial.add(int(idx))
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            trivial.add(int(val))
    return tuple(sorted(trivial))


class StreamingPointerLoss(torch.nn.Module):
    """Zero-materialization streaming LogSumExp pointer NLL for exact span grounding."""

    def __init__(
        self,
        d_dec: int,
        d_ptr: int = 256,
        chunk_size: int = 512,
        trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.d_ptr = int(d_ptr)
        self.chunk_size = int(chunk_size)
        self.scale = 1.0 / math.sqrt(float(self.d_ptr))
        self.w_ptr_q = torch.nn.Linear(int(d_dec), self.d_ptr, bias=False)
        self.w_ptr_k = torch.nn.Linear(int(d_dec), self.d_ptr, bias=False)
        self.trivial_token_ids = trivial_token_ids
        self.use_checkpoint = use_checkpoint

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        h_dec: torch.Tensor,
        z_enc: torch.Tensor,
        enc_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        enc_valid_mask: torch.Tensor,
        trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_enc.shape[1] == 0 or not bool(enc_valid_mask.any()):
            return h_dec.sum() * 0.0
        q_ptr = self.w_ptr_q(h_dec.float())
        k_ptr = self.w_ptr_k(z_enc.float())
        effective_trivial = (
            trivial_token_ids if trivial_token_ids is not None else self.trivial_token_ids
        )
        return _streaming_lse_pointer_nll(
            q_ptr,
            k_ptr,
            enc_token_ids,
            target_token_ids,
            enc_valid_mask,
            scale=self.scale,
            chunk_size=self.chunk_size,
            trivial_token_ids=effective_trivial,
            use_checkpoint=self.use_checkpoint,
        )


def _pointer_chunk_forward(
    q_chunk: torch.Tensor,
    k_ptr: torch.Tensor,
    t_chunk: torch.Tensor,
    enc_token_ids: torch.Tensor,
    enc_valid_mask: torch.Tensor,
    trivial_tensor: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pointer loss sum and valid count for one decoder chunk."""
    valid_t = t_chunk != -100
    if trivial_tensor is not None and trivial_tensor.numel() > 0:
        valid_t = valid_t & ~torch.isin(t_chunk, trivial_tensor)

    scores = torch.bmm(q_chunk, k_ptr.transpose(1, 2)) * scale
    scores = scores.masked_fill(~enc_valid_mask[:, None, :].bool(), float("-inf"))
    log_denom = torch.logsumexp(scores, dim=-1)

    match_mask = (
        (t_chunk[:, :, None] == enc_token_ids[:, None, :])
        & enc_valid_mask[:, None, :].bool()
        & valid_t[:, :, None]
    )
    has_copy_target = match_mask.any(dim=-1)
    if not bool(has_copy_target.any()):
        return q_chunk.sum() * 0.0, torch.tensor(0, dtype=torch.int64, device=q_chunk.device)

    pos_scores = scores.masked_fill(~match_mask, float("-inf"))
    log_num = torch.logsumexp(pos_scores, dim=-1)
    token_nll = -(log_num - log_denom)
    valid_nll = token_nll[has_copy_target]
    chunk_loss = valid_nll.sum()
    chunk_valid = torch.tensor(valid_nll.numel(), dtype=torch.int64, device=q_chunk.device)
    return chunk_loss, chunk_valid


def _streaming_lse_pointer_nll(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    enc_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    enc_valid_mask: torch.Tensor,
    *,
    scale: float,
    chunk_size: int = 256,
    trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Compute streaming LogSumExp pointer NLL with per-chunk checkpointing and trivial token filtering."""
    if k_ptr.shape[1] == 0 or not bool(enc_valid_mask.any()):
        return q_ptr.sum() * 0.0

    if trivial_token_ids is None:
        trivial_token_ids = DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS

    if trivial_token_ids is not None and len(trivial_token_ids) > 0:
        if isinstance(trivial_token_ids, torch.Tensor):
            trivial_tensor = trivial_token_ids.to(
                device=target_token_ids.device, dtype=target_token_ids.dtype
            )
        else:
            trivial_tensor = torch.tensor(
                tuple(trivial_token_ids),
                device=target_token_ids.device,
                dtype=target_token_ids.dtype,
            )
    else:
        trivial_tensor = None

    _, seq_len, _ = q_ptr.shape
    total_loss = q_ptr.new_zeros((), dtype=torch.float32)
    total_valid = 0

    do_checkpoint = (
        use_checkpoint
        and torch.is_grad_enabled()
        and (q_ptr.requires_grad or k_ptr.requires_grad)
    )

    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        q_chunk = q_ptr[:, start:end, :]
        t_chunk = target_token_ids[:, start:end]

        valid_t = t_chunk != -100
        if trivial_tensor is not None and trivial_tensor.numel() > 0:
            valid_t = valid_t & ~torch.isin(t_chunk, trivial_tensor)
        if not bool(valid_t.any()):
            continue

        match_mask = (
            (t_chunk[:, :, None] == enc_token_ids[:, None, :])
            & enc_valid_mask[:, None, :].bool()
            & valid_t[:, :, None]
        )
        if not bool(match_mask.any()):
            continue

        if do_checkpoint:
            chunk_loss, chunk_valid = torch.utils.checkpoint.checkpoint(
                _pointer_chunk_forward,
                q_chunk,
                k_ptr,
                t_chunk,
                enc_token_ids,
                enc_valid_mask,
                trivial_tensor,
                scale,
                use_reentrant=False,
            )
        else:
            chunk_loss, chunk_valid = _pointer_chunk_forward(
                q_chunk,
                k_ptr,
                t_chunk,
                enc_token_ids,
                enc_valid_mask,
                trivial_tensor,
                scale,
            )

        n = int(chunk_valid.item())
        if n > 0:
            total_loss = total_loss + chunk_loss
            total_valid += n

    if total_valid == 0:
        return q_ptr.sum() * 0.0
    return total_loss / float(total_valid)


def streaming_pointer_auxiliary_loss(
    h_dec: torch.Tensor,
    z_enc: torch.Tensor,
    enc_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    enc_valid_mask: torch.Tensor,
    *,
    chunk_size: int = 256,
    trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Parameter-free zero-materialization streaming pointer NLL aligning decoder hidden states to bridge states."""
    if z_enc is None or z_enc.shape[1] == 0 or not bool(enc_valid_mask.any()):
        return h_dec.sum() * 0.0
    d_model = max(1, int(h_dec.shape[-1]))
    q_norm = F.normalize(h_dec.float(), dim=-1) * math.sqrt(float(d_model))
    k_norm = F.normalize(z_enc.float(), dim=-1) * math.sqrt(float(d_model))
    scale = 1.0 / math.sqrt(float(d_model))
    return _streaming_lse_pointer_nll(
        q_norm,
        k_norm,
        enc_token_ids,
        target_token_ids,
        enc_valid_mask,
        scale=scale,
        chunk_size=chunk_size,
        trivial_token_ids=trivial_token_ids,
        use_checkpoint=use_checkpoint,
    )


# ==============================================================================
# Verified Gemma-4 Differential Diff Token Identifiers
# ==============================================================================
DEFAULT_DIFF_ADD_TOKEN_IDS: tuple[int, ...] = (236862, 900)
DEFAULT_DIFF_DEL_TOKEN_IDS: tuple[int, ...] = (236772, 753)
DEFAULT_DIFF_HDR_TOKEN_IDS: tuple[int, ...] = (
    16417, 7243, 24523, 10379, 15338, 2617, 13705
)


def resolve_diff_token_ids(tokenizer: Any = None) -> dict[str, tuple[int, ...]]:
    """Resolve diff marker token IDs from a decoder tokenizer with fallback to Gemma-4 defaults."""
    if tokenizer is None:
        return {
            "add_token_ids": DEFAULT_DIFF_ADD_TOKEN_IDS,
            "del_token_ids": DEFAULT_DIFF_DEL_TOKEN_IDS,
            "hdr_token_ids": DEFAULT_DIFF_HDR_TOKEN_IDS,
        }

    def _enc(s: str) -> int | None:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    add_ids = tuple(filter(None, [_enc("+"), _enc(" +")]))
    del_ids = tuple(filter(None, [_enc("-"), _enc(" -")]))
    hdr_candidates = ["diff", "---", "+++", "@@", " @@", " --", "git"]
    hdr_ids = tuple(filter(None, [_enc(c) for c in hdr_candidates]))

    return {
        "add_token_ids": add_ids or DEFAULT_DIFF_ADD_TOKEN_IDS,
        "del_token_ids": del_ids or DEFAULT_DIFF_DEL_TOKEN_IDS,
        "hdr_token_ids": hdr_ids or DEFAULT_DIFF_HDR_TOKEN_IDS,
    }


def compute_diff_token_weights(
    labels: torch.Tensor,
    *,
    add_weight: float = 2.0,
    del_weight: float = 1.5,
    hdr_weight: float = 1.2,
    ctx_weight: float = 0.5,
    add_token_ids: tuple[int, ...] = DEFAULT_DIFF_ADD_TOKEN_IDS,
    del_token_ids: tuple[int, ...] = DEFAULT_DIFF_DEL_TOKEN_IDS,
    hdr_token_ids: tuple[int, ...] = DEFAULT_DIFF_HDR_TOKEN_IDS,
) -> torch.Tensor:
    """Compute normalized differential token weights so mean weight over valid labels is 1.0."""
    valid = labels != -100
    weights = torch.zeros_like(labels, dtype=torch.float32)
    if not bool(valid.any()):
        return weights
    weights[valid] = float(ctx_weight)
    if hdr_token_ids:
        hdr_mask = valid & torch.isin(
            labels, torch.tensor(hdr_token_ids, device=labels.device, dtype=labels.dtype)
        )
        weights[hdr_mask] = float(hdr_weight)
    if del_token_ids:
        del_mask = valid & torch.isin(
            labels, torch.tensor(del_token_ids, device=labels.device, dtype=labels.dtype)
        )
        weights[del_mask] = float(del_weight)
    if add_token_ids:
        add_mask = valid & torch.isin(
            labels, torch.tensor(add_token_ids, device=labels.device, dtype=labels.dtype)
        )
        weights[add_mask] = float(add_weight)
    if weights.ndim == 2:
        for b in range(weights.shape[0]):
            row_valid = valid[b]
            if bool(row_valid.any()):
                row_mean = weights[b, row_valid].mean().clamp_min(1e-6)
                weights[b, row_valid] = weights[b, row_valid] / row_mean
    else:
        mean_w = weights[valid].mean().clamp_min(1e-6)
        weights[valid] = weights[valid] / mean_w
    return weights


def differential_diff_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute normalized weighted cross-entropy loss across aligned target tokens [B, L]."""
    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("Expected aligned logits [B, L, V] and labels [B, L].")
    valid = labels != -100
    if not bool(valid.any()):
        return logits.sum() * 0.0
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    if token_weights is None:
        weights = torch.ones_like(per_token, dtype=torch.float32)
    else:
        weights = token_weights.float().clone()
        mean_w = weights[valid].mean().clamp_min(1e-6)
        weights[valid] = weights[valid] / mean_w
    return (per_token * weights)[valid].sum() / valid.float().sum().clamp_min(1.0)


def chunked_differential_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor | None = None,
    *,
    chunk_size: int = 128,
    softcap: float | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Memory-bounded chunked LM-head cross-entropy with normalized differential token weights."""
    from torch.utils.checkpoint import checkpoint

    if chunk_size < 1 or hidden.shape[:2] != labels.shape:
        raise ValueError("Invalid loss chunk size or aligned label shape.")
    valid = labels.reshape(-1) != ignore_index
    if not bool(valid.any()):
        return hidden.sum() * 0.0 + weight.reshape(-1)[0] * 0.0
    states = hidden.reshape(-1, hidden.shape[-1])[valid]
    targets = labels.reshape(-1)[valid]
    if token_weights is None:
        flat_weights = torch.ones_like(targets, dtype=torch.float32)
    else:
        flat_weights = token_weights.reshape(-1)[valid].float()
        mean_w = flat_weights.mean().clamp_min(1e-6)
        flat_weights = flat_weights / mean_w

    def compute(values: torch.Tensor, words: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        logits = F.linear(values, words)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        ce = F.cross_entropy(logits.float(), target, reduction="none")
        return (ce * w).sum()

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        stop = start + chunk_size
        args = (states[start:stop], weight, targets[start:stop], flat_weights[start:stop])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            value = checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            value = compute(*args)
        total = total + value
    return total / float(targets.numel())


def compute_v3_step_loss(
    model: Any,
    batch: Any,
    *,
    stage_name: str = "frozen_decoder",
    lambda_ptr: float | None = None,
    diff_token_ids: dict[str, tuple[int, ...]] | None = None,
    trivial_token_ids: tuple[int, ...] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute BifocalGemma V3 training loss combining chunked differential CE and streaming pointer loss."""
    if not hasattr(model, "_decode_hidden") or not hasattr(model, "lm_head"):
        output = model.forward_loss(batch) if hasattr(model, "forward_loss") else model(batch)
        return output.loss

    has_encoder = getattr(model, "encoder", None) is not None
    if has_encoder:
        encoder_states = model.encode_context(batch.encoder_input_ids, batch.encoder_attention_mask)
        encoder_mask = batch.encoder_attention_mask
    else:
        encoder_states, encoder_mask = None, None

    decoder_hidden = model._decode_hidden(
        batch.decoder_input_ids,
        batch.decoder_attention_mask,
        encoder_states,
        encoder_mask,
    )
    diff_ids = getattr(model, "diff_token_ids", None) or diff_token_ids or {}
    token_weights = compute_diff_token_weights(
        batch.labels,
        add_token_ids=diff_ids.get("add_token_ids", DEFAULT_DIFF_ADD_TOKEN_IDS),
        del_token_ids=diff_ids.get("del_token_ids", DEFAULT_DIFF_DEL_TOKEN_IDS),
        hdr_token_ids=diff_ids.get("hdr_token_ids", DEFAULT_DIFF_HDR_TOKEN_IDS),
    )
    chunk_sz = getattr(model, "loss_chunk_size", None) or 128
    softcap = getattr(getattr(model, "decoder", None), "config", None)
    softcap_val = getattr(softcap, "final_logit_softcapping", None) if softcap is not None else None
    ce_loss = chunked_differential_cross_entropy(
        decoder_hidden,
        model.lm_head.weight,
        batch.labels,
        token_weights=token_weights,
        chunk_size=int(chunk_sz),
        softcap=softcap_val,
    )
    if not has_encoder or encoder_states is None or encoder_mask is None:
        return ce_loss

    ptr_weight = (
        float(lambda_ptr)
        if lambda_ptr is not None
        else (0.15 if stage_name == "frozen_decoder" else 0.05)
    )
    if ptr_weight <= 0.0:
        return ce_loss
    trivial_ids = getattr(model, "trivial_token_ids", None) or trivial_token_ids
    ptr_loss = streaming_pointer_auxiliary_loss(
        decoder_hidden,
        encoder_states,
        batch.encoder_input_ids,
        batch.labels,
        encoder_mask,
        trivial_token_ids=trivial_ids,
    )
    return ce_loss + ptr_weight * ptr_loss


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


def is_zero_weight_decay_bridge_parameter(name: str) -> bool:
    """Return True if an alignment parameter must have strict 0.0 weight decay (RMSNorm, gates, cross-attn)."""
    if name.startswith("encoder_projection."):
        return any(tok in name for tok in ("pre_norm", "post_norm", "gamma_calib", "norm"))
    return True


def build_bifocalgemma_v3_optimizer(
    model: Any,
    *,
    lr_bridge: float = 1e-4,
    lr_cross: float = 1e-4,
    lr_lora: float = 2e-5,
    weight_decay_bridge: float = 1e-4,
) -> torch.optim.AdamW:
    """Build partitioned AdamW optimizer enforcing strict 0.0 weight decay on RMSNorm, cross-attn, and LoRA."""
    bridge_decay = []
    bridge_decay_names = []
    bridge_nodecay = []
    bridge_nodecay_names = []
    cross_attn = []
    cross_attn_names = []
    backbone_lora = []
    backbone_lora_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder_projection."):
            if is_zero_weight_decay_bridge_parameter(name) or param.ndim < 2:
                bridge_nodecay.append(param)
                bridge_nodecay_names.append(name)
            else:
                bridge_decay.append(param)
                bridge_decay_names.append(name)
        elif is_cross_bridge_parameter(name):
            cross_attn.append(param)
            cross_attn_names.append(name)
        else:
            backbone_lora.append(param)
            backbone_lora_names.append(name)

    param_groups = [
        {
            "name": "bridge_linear_decay",
            "params": bridge_decay,
            "param_names": bridge_decay_names,
            "lr": lr_bridge,
            "base_lr": lr_bridge,
            "weight_decay": weight_decay_bridge,
        },
        {
            "name": "bridge_norm_and_calib_nodecay",
            "params": bridge_nodecay,
            "param_names": bridge_nodecay_names,
            "lr": lr_bridge,
            "base_lr": lr_bridge,
            "weight_decay": 0.0,
        },
        {
            "name": "decoupled_cross_attn_nodecay",
            "params": cross_attn,
            "param_names": cross_attn_names,
            "lr": lr_cross,
            "base_lr": lr_cross,
            "weight_decay": 0.0,
        },
        {
            "name": "backbone_lora",
            "params": backbone_lora,
            "param_names": backbone_lora_names,
            "lr": lr_lora,
            "base_lr": lr_lora,
            "weight_decay": 0.0,
        },
    ]
    active_groups = [g for g in param_groups if g["params"]]
    return torch.optim.AdamW(
        active_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )


def _compute_per_row_cross_query(
    query_cross: torch.Tensor,
    encoder_valid: torch.Tensor,
    fallback_n_enc: int,
) -> torch.Tensor:
    """Divide cross-queries by per-row critical attention length temperature beta_N (V3 §5.4 & §8.2)."""
    bsz = query_cross.shape[0]
    valid_counts = encoder_valid.bool().sum(dim=-1).tolist()
    betas = [
        critical_attention_beta(int(cnt) if int(cnt) > 0 else int(fallback_n_enc))
        for cnt in valid_counts
    ]
    if all(b == 1.0 for b in betas):
        return query_cross
    inv_beta = torch.tensor(
        [1.0 / b for b in betas],
        dtype=torch.float32,
        device=query_cross.device,
    ).view(bsz, 1, 1, 1)
    return (query_cross.float() * inv_beta).to(query_cross.dtype)


def configure_full_rank_cross_attention(model: Any) -> dict[str, Any]:
    """Attach SwiGLUDualGateBridge, full-rank BF16 cross-attention projections, unrotated cross-query norm, and atanh(0.05) gates."""
    ensure_hardware_memory_guards()
    orig_cached_cls = getattr(
        inference_cache_mod, "_orig_cached_attention", inference_cache_mod.CachedAttention
    )
    setattr(inference_cache_mod, "_orig_cached_attention", orig_cached_cls)

    class FullCrossCachedAttention(orig_cached_cls):
        """Cached inference attention matching FullCrossTiledAttention with static cross-KV cache and batched left-pad support."""

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
            step_valid = getattr(inference_cache, "step_valid_mask", None) if inference_cache is not None else None
            has_full_cross = bool(getattr(self, "full_rank_cross_attention", False))
            if not has_full_cross and step_valid is None and hidden_states.shape[0] == 1:
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
            query_cross = (
                self.cross_q_norm(raw_q).to(raw_q.dtype).transpose(1, 2)
                if has_full_cross and hasattr(self, "cross_q_norm")
                else None
            )
            cache = inference_cache
            if cache is None:
                raise ValueError("Explicit inference cache required.")
            positions = torch.arange(cache.offset, cache.offset + length, device=query_self.device)
            if self.is_kv_shared_layer:
                key_self, value_self, cross, key_valid = shared_kv_states[self.layer_type]
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
                key_valid = step_valid
                if previous is not None:
                    key_self = torch.cat((previous[0], key_self), dim=2)
                    value_self = torch.cat((previous[1], value_self), dim=2)
                    key_positions = torch.cat((previous[2], positions))
                    prev_valid = previous[4] if len(previous) > 4 else None
                    if prev_valid is not None and step_valid is not None:
                        key_valid = torch.cat((prev_valid, step_valid), dim=1)
                keep = self.sliding_window if self.sliding_window else key_self.shape[2]
                stored_key, stored_value, stored_positions = (
                    key_self[:, :, -keep:],
                    value_self[:, :, -keep:],
                    key_positions[-keep:],
                )
                stored_valid = key_valid[:, -keep:] if key_valid is not None else None
                cross = previous[3] if previous is not None else None
                if (
                    has_full_cross
                    and encoder_hidden_states is not None
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
                    stored_valid,
                )
                self_length = key_self.shape[2]
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (key_self, value_self, cross, key_valid)
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
                if key_valid is not None:
                    allowed = allowed & key_valid[:, None, None, :].bool()
                    has_any_key = allowed.any(dim=-1, keepdim=True)
                    safe_allowed = allowed | ~has_any_key
                    tile_out = F.scaled_dot_product_attention(
                        query_self[:, :, start:stop],
                        key_self,
                        value_self,
                        attn_mask=safe_allowed,
                        dropout_p=0.0,
                        scale=self.scaling,
                        enable_gqa=True,
                    )
                    tile_out = tile_out.masked_fill(~has_any_key, 0.0)
                else:
                    tile_out = F.scaled_dot_product_attention(
                        query_self[:, :, start:stop],
                        key_self,
                        value_self,
                        attn_mask=allowed,
                        dropout_p=0.0,
                        scale=self.scaling,
                        enable_gqa=True,
                    )
                self_pieces.append(tile_out)
            self_out = torch.cat(self_pieces, dim=2)

            if (
                has_full_cross
                and query_cross is not None
                and cross is not None
                and encoder_valid is not None
                and bool(encoder_valid.any())
            ):
                n_enc = int(cross[0].shape[2])
                scaled_query_cross = _compute_per_row_cross_query(query_cross, encoder_valid, n_enc)
                cross_scale = float(self.scaling)
                cross_pieces = []
                for start in range(0, length, 128):
                    stop = min(length, start + 128)
                    cross_mask = (
                        encoder_valid[:, None, None, :].bool().expand(batch, 1, stop - start, -1)
                    )
                    if step_valid is not None:
                        cross_mask = cross_mask & step_valid[:, None, start:stop, None].bool()
                    cross_pieces.append(
                        attention_tile(
                            scaled_query_cross[:, :, start:stop],
                            cross[0],
                            cross[1],
                            cross_mask,
                            cross_scale,
                            0.0,
                            True,
                        )
                    )
                cross_out = torch.cat(cross_pieces, dim=2)
                gate_logits = torch.einsum(
                    "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
                ) * (4.0 * (self.head_dim**-0.5))
                query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
                head_gate = torch.tanh(self.cross_gate.float()).view(1, -1, 1, 1)
                effective_gate = (head_gate * query_gate).to(self_out.dtype)
                result = self_out + effective_gate * cross_out
            else:
                result = self_out

            result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
            return self.o_proj(result), None

    setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)

    if getattr(model, "encoder", None) is None:
        return {
            "mode": "decoder_only_causal_v3",
            "bridge_type": None,
            "kv_owner_layers": 0,
            "total_decoder_layers": len(model.decoder.layers),
            "total_cross_bridge_parameters": 0,
        }

    # Install SwiGLUDualGateBridge as model.encoder_projection
    enc_dim = int(model.encoder.config.hidden_size)
    dec_dim = int(model.decoder.config.hidden_size)
    dev = next(model.decoder.parameters()).device
    model.encoder_projection = SwiGLUDualGateBridge(enc_dim, dec_dim).to(dev)

    class Bf16FullLinear(torch.nn.Module):
        """Full-rank linear projection with 16-bit bfloat16 compute and FP32 master weight."""

        def __init__(self, base_layer: torch.nn.Module) -> None:
            super().__init__()
            base: Any = getattr(base_layer, "base", base_layer)
            w = base.weight.detach().float().clone()
            self.in_features = int(w.shape[1])
            self.out_features = int(w.shape[0])
            self.weight = torch.nn.Parameter(w)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
            compute_dtype = (
                torch.bfloat16
                if (x.device.type == "cuda" or x.dtype == torch.bfloat16)
                else x.dtype
            )
            return F.linear(x.to(compute_dtype), self.weight.to(compute_dtype)).to(x.dtype)

    class FullCrossTiledAttention(TiledMergedAttention):
        """Decoupled causal self-attention and stationary unrotated full-rank 16-bit gated cross-attention."""

        def forward(  # ty: ignore[invalid-method-override]
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
            # Stationary unrotated cross-query coordinates (zero RoPE frequency spinning)
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
                n_enc = int(encoder_hidden_states.shape[1])
                scaled_query_cross = _compute_per_row_cross_query(query_cross, encoder_valid, n_enc)
                cross_scale = float(self.scaling)
                pieces = []
                for start in range(0, length, chunk_sz):
                    stop = min(length, start + chunk_sz)
                    cross_mask = (
                        encoder_valid[:, None, None, :].bool()
                        & decoder_valid[:, None, start:stop, None].bool()
                    )
                    pieces.append(
                        attention_tile(
                            scaled_query_cross[:, :, start:stop],
                            cross_key,
                            cross_value,
                            cross_mask,
                            cross_scale,
                            0.0,
                            True,
                        )
                    )
                cross_out = torch.cat(pieces, dim=2)
                gate_logits = torch.einsum(
                    "bhtd,hd->bht", query_cross.float(), self.cross_gate_proj.float()
                ) * (4.0 * (self.head_dim**-0.5))
                query_gate = 2.0 * torch.sigmoid(gate_logits).unsqueeze(-1)
                head_gate = torch.tanh(self.cross_gate.float()).view(1, -1, 1, 1)
                effective_gate = (head_gate * query_gate).to(self_out.dtype)
                result = self_out + effective_gate * cross_out
            else:
                result = self_out

            result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
            return self.o_proj(result), None

    kv_owner_layers = 0
    total_cross_params = 0
    for layer in model.decoder.layers:
        attn = layer.self_attn
        q_base = getattr(attn.q_proj, "base", attn.q_proj)
        dev = q_base.weight.device
        num_heads = int(q_base.weight.shape[0] // attn.head_dim)
        attn.cross_q_norm = copy.deepcopy(attn.q_norm).float().to(dev)
        attn.cross_gate = torch.nn.Parameter(
            torch.full((num_heads,), float(GATE_INIT_ATANH), dtype=torch.float32, device=dev)
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
        attn.triton_kernel_options = clamped_triton_kernel_options(attn.head_dim)
        attn.full_rank_cross_attention = True
        attn.__class__ = FullCrossTiledAttention

    for n, p in model.named_parameters():
        if is_cross_bridge_parameter(n):
            total_cross_params += int(p.numel())
    return {
        "mode": "full_rank_cross_bf16",
        "bridge_type": "SwiGLUDualGateBridge",
        "gate_init_tanh": GATE_INIT_TANH,
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
    """Configure trainable groups for Stage 1 (frozen decoder on paired models) or Stage 2 (joint SFT).

    Enforces V3 partitioned weight decay: 1e-4 on bridge linear weights, 0.0 on RMSNorm/calibration/cross-attn/LoRA.
    """
    if total_steps < 1:
        raise ValueError("Stage horizon must be a positive integer.")
    default_scaling = policy["alpha"] / policy["rank"]
    has_encoder = getattr(model, "encoder", None) is not None
    effective_freeze_decoder = bool(freeze_decoder and has_encoder)

    for name, module in model.named_modules():
        if isinstance(module, ProbeLoRALinear):
            if name.startswith("decoder."):
                module.scaling = 0.0 if effective_freeze_decoder else default_scaling
            else:
                module.scaling = default_scaling

    for name, param in model.named_parameters():
        if has_encoder and is_cross_bridge_parameter(name):
            param.requires_grad_(True)
        elif has_encoder and name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(True)
        elif name.startswith("decoder.") and name.endswith((".lora_A", ".lora_B")):
            param.requires_grad_(not effective_freeze_decoder)
        else:
            param.requires_grad_(False)

    if not has_encoder:
        active_names: tuple[str, ...] = ("decoder",)
    else:
        active_names = (
            ("encoder", "alignment")
            if effective_freeze_decoder
            else ("encoder", "decoder", "alignment")
        )
    entries: dict[str, list[tuple[str, Any]]] = {name: [] for name in active_names}
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        if has_encoder and is_cross_bridge_parameter(name):
            entries["alignment"].append((name, param))
        elif has_encoder and name.startswith("encoder.") and name.endswith((".lora_A", ".lora_B")):
            entries["encoder"].append((name, param))
        elif (
            not effective_freeze_decoder
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
        raw_warmup_ratio = settings.get("warmup_ratio", WARMUP_RATIO)
        warmup_ratio = WARMUP_RATIO if raw_warmup_ratio == 0.03 else float(raw_warmup_ratio)
        warmup_steps = min(
            total_steps, max(settings.get("warmup_steps", 0), math.ceil(total_steps * warmup_ratio))
        )
        if group_name == "alignment":
            bridge_decay_entries = [
                (n, p)
                for n, p in entries["alignment"]
                if n.startswith("encoder_projection.")
                and not is_zero_weight_decay_bridge_parameter(n)
                and p.ndim >= 2
            ]
            nodecay_entries = [
                (n, p)
                for n, p in entries["alignment"]
                if not (
                    n.startswith("encoder_projection.")
                    and not is_zero_weight_decay_bridge_parameter(n)
                    and p.ndim >= 2
                )
            ]
            if bridge_decay_entries:
                groups.append(
                    {
                        "name": "alignment_bridge_decay",
                        "params": [p for _, p in bridge_decay_entries],
                        "param_names": [n for n, _ in bridge_decay_entries],
                        "lr": settings["lr"],
                        "base_lr": settings["lr"],
                        "warmup_steps": warmup_steps,
                        "scheduler": settings.get("scheduler", "cosine"),
                        "total_steps": total_steps,
                        "warmup_ratio": warmup_ratio,
                        "min_lr_ratio": settings.get("min_lr_ratio", 0.1),
                        "weight_decay": 1e-4,
                    }
                )
            if nodecay_entries:
                groups.append(
                    {
                        "name": "alignment",
                        "params": [p for _, p in nodecay_entries],
                        "param_names": [n for n, _ in nodecay_entries],
                        "lr": settings["lr"],
                        "base_lr": settings["lr"],
                        "warmup_steps": warmup_steps,
                        "scheduler": settings.get("scheduler", "cosine"),
                        "total_steps": total_steps,
                        "warmup_ratio": warmup_ratio,
                        "min_lr_ratio": settings.get("min_lr_ratio", 0.1),
                        "weight_decay": 0.0,
                    }
                )
        else:
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
                    "weight_decay": 0.0,
                }
            )
    summary = {
        "freeze_decoder": effective_freeze_decoder,
        "active_groups": list(active_names),
        "trainable_parameters": {
            name: sum(int(p.numel()) for _, p in entries[name]) for name in active_names
        },
        "decoder_trainable_tensors": sum(
            1
            for n, p in model.named_parameters()
            if n.startswith("decoder.") and not is_cross_bridge_parameter(n) and p.requires_grad
        ),
    }
    return groups, summary


# ---------------------------------------------------------------------------
# Anti-Catastrophic-Forgetting Dataset Curriculum (10-15% Comprehension / QA Mix)
# ---------------------------------------------------------------------------

COMPREHENSION_TASK_TYPES = frozenset(
    {"localization", "completion", "qa", "mcq", "span_copy", "code_qa_mcq", "comprehension"}
)

COMPREHENSION_INSTRUCTION_PROMPTS: dict[str, str] = {
    "span_copy": "Extract and copy the exact target identifier or code span from the context.",
    "code_qa_mcq": "Identify the correct symbol or answer based on the context.",
}

# Pre-computed canonical Gemma tokenizer prefix IDs for offline/unit-test fallback when decoder is not provided:
# - span_copy: '<bos><|turn>user\nExtract and copy the exact target identifier or code span from the context.<turn|>\n<|turn>model\n'
# - code_qa_mcq: '<bos><|turn>user\nIdentify the correct symbol or answer based on the context.<turn|>\n<|turn>model\n'
CANONICAL_COMPREHENSION_PREFIX_IDS: dict[str, tuple[int, ...]] = {
    "span_copy": (
        2, 105, 2364, 107, 82138, 532, 4865, 506, 4453, 3328, 31744, 653, 3393, 15184, 699, 506, 4403, 236761, 106, 107, 105, 4368, 107
    ),
    "code_qa_mcq": (
        2, 105, 2364, 107, 137938, 506, 4338, 5404, 653, 3890, 2721, 580, 506, 4403, 236761, 106, 107, 105, 4368, 107
    ),
}


def encode_comprehension_prefix(
    kind: str,
    decoder: Any = None,
) -> tuple[int, ...]:
    """Tokenize chat template prompt for comprehension task kind into decoder prefix token IDs."""
    if decoder is not None and hasattr(decoder, "apply_chat_template"):
        prompt = COMPREHENSION_INSTRUCTION_PROMPTS.get(
            kind,
            "Extract and copy the exact target identifier or code span from the context.",
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = decoder.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = decoder.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = tuple(decoder.encode(rendered, add_special_tokens=False))
        if encoded:
            return encoded
    return CANONICAL_COMPREHENSION_PREFIX_IDS.get(
        kind,
        CANONICAL_COMPREHENSION_PREFIX_IDS["span_copy"],
    )


def is_comprehension_example(example: Any) -> bool:
    """Return True if a TokenizedRepairExample is a non-diff comprehension/QA/span-copy/completion task."""
    meta = getattr(example, "metadata", None) or {}
    task_type = str(meta.get("task_type", "repair")).lower()
    return task_type in COMPREHENSION_TASK_TYPES


def synthesize_comprehension_examples(
    examples: list[TokenizedRepairExample],
    count: int,
    *,
    seed: int = 17,
    decoder: Any = None,
) -> list[TokenizedRepairExample]:
    """Synthesize deterministic exact span-copy and concise Code QA / MCQ examples from context tokens.

    Prevents catastrophic forgetting on `longcodeqa`, `sweqa`, and `gsm8k` while forcing the
    cross-attention bridge to copy exact token spans from the encoder context.
    """
    if count <= 0 or not examples:
        return []
    rng = random.Random(f"bifocalgemma_v3_comprehension_synth_{seed}")
    synthesized: list[TokenizedRepairExample] = []
    n_pool = len(examples)

    # Pre-tokenize comprehension prefix IDs once for both task kinds
    prefix_by_kind = {
        kind: encode_comprehension_prefix(kind, decoder=decoder)
        for kind in ("span_copy", "code_qa_mcq")
    }

    for idx in range(count):
        base = examples[(idx * 7 + rng.randint(0, max(0, n_pool - 1))) % n_pool]
        eos_token = base.target_ids[-1] if base.target_ids else 107
        source_tokens = base.encoder_ids if len(base.encoder_ids) > 2 else base.decoder_prefix_ids
        interior = source_tokens[1:] if len(source_tokens) > 1 else source_tokens
        if not interior:
            interior = base.target_ids[:-1] or (10,)

        if idx % 2 == 0:
            # Exact Span-Copy task: copy a contiguous 4-24 token span from context
            span_len = min(len(interior), max(1, min(16, max(2, len(interior) // 2))))
            max_start = max(0, len(interior) - span_len)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            span = tuple(interior[start : start + span_len])
            target = (*span, eos_token) if span[-1] != eos_token else span
            kind = "span_copy"
        else:
            # Concise Code QA / MCQ identifier answer task: short 1-4 token concise answer + EOS
            ans_len = min(len(interior), max(1, min(4, len(interior))))
            max_start = max(0, len(interior) - ans_len)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            ans = tuple(interior[start : start + ans_len])
            target = (*ans, eos_token) if ans[-1] != eos_token else ans
            kind = "code_qa_mcq"

        meta = dict(base.metadata or {})
        meta["task_type"] = kind
        meta["anti_forgetting_replay"] = True
        meta["instruction"] = COMPREHENSION_INSTRUCTION_PROMPTS[kind]
        max_base_token = max(
            (max(base.decoder_prefix_ids) if base.decoder_prefix_ids else 0),
            (max(base.encoder_ids) if base.encoder_ids else 0),
            (max(base.target_ids) if base.target_ids else 0),
        )
        if decoder is None and max_base_token < 256:
            prefix = tuple(
                tok if tok < 128 else (3 + (tok % 100))
                for tok in prefix_by_kind[kind]
            )
        else:
            prefix = prefix_by_kind[kind]

        if not base.encoder_ids:
            # Baseline decoder-only: preserve repository context in decoder_prefix_ids
            # so the model has the context from which the target span was extracted
            effective_prefix = (*prefix, *base.decoder_prefix_ids)
            if len(effective_prefix) > 4096:
                effective_prefix = effective_prefix[:4096]
        else:
            effective_prefix = prefix

        synthesized.append(
            TokenizedRepairExample(
                instance_id=f"{base.instance_id}__{kind}_{idx}",
                encoder_ids=base.encoder_ids,
                decoder_prefix_ids=effective_prefix,
                target_ids=target,
                metadata=meta,
            )
        )
    return synthesized


def build_anti_forgetting_curriculum(
    train_examples: list[TokenizedRepairExample],
    *,
    stage_name: str = "frozen_decoder",
    comprehension_ratio: float = COMPREHENSION_MIX_RATIO,
    seed: int = 17,
    decoder: Any = None,
) -> tuple[list[TokenizedRepairExample], dict[str, Any]]:
    """Construct a balanced Stage 1 / Stage 2 curriculum with 10-15% Code QA / MCQ & span-copy replay.

    Prioritizes authentic completion and localization records from the dataset pool first,
    supplementing with synthesized span_copy and code_qa_mcq examples up to the target ratio.
    """
    total = len(train_examples)
    if total == 0:
        return [], {
            "total_examples": 0,
            "comprehension_examples": 0,
            "comprehension_ratio": 0.0,
            "synthesized_examples": 0,
            "stage_name": stage_name,
        }

    ratio = max(0.10, min(0.25, float(comprehension_ratio)))
    target_comp_count = max(1, int(round(total * ratio))) if total >= 4 else 0

    existing_comp = [ex for ex in train_examples if is_comprehension_example(ex)]
    repair_pool = [ex for ex in train_examples if not is_comprehension_example(ex)]

    rng = random.Random(f"curriculum_{stage_name}_{seed}")

    # Prioritize authentic completion/localization records; synthesize deficit to reach target_comp_count
    if len(existing_comp) >= target_comp_count:
        comp_pool = list(existing_comp)
        rng.shuffle(comp_pool)
        selected_comp = comp_pool[:target_comp_count]
        synthesized: list[TokenizedRepairExample] = []
    else:
        synth_needed = target_comp_count - len(existing_comp)
        synthesized = synthesize_comprehension_examples(
            train_examples,
            synth_needed,
            seed=seed + (101 if stage_name == "joint" else 0),
            decoder=decoder,
        )
        selected_comp = list(existing_comp) + synthesized
        rng.shuffle(selected_comp)

    rng.shuffle(repair_pool)
    remaining_slots = max(0, total - len(selected_comp))
    base_pool = repair_pool if repair_pool else list(train_examples)
    selected_repair = [base_pool[i % len(base_pool)] for i in range(remaining_slots)]

    # Interleave comprehension examples uniformly across the epoch; in Stage 1 also front-load warmup
    curriculum: list[TokenizedRepairExample] = []
    if stage_name == "frozen_decoder" and len(selected_comp) >= 4:
        front_count = max(1, len(selected_comp) // 3)
        curriculum.extend(selected_comp[:front_count])
        rest_comp = selected_comp[front_count:]
    else:
        rest_comp = list(selected_comp)

    comp_idx = 0
    rep_idx = 0
    remaining_total = total - len(curriculum)
    stride = max(2, int(round(remaining_total / max(1, len(rest_comp))))) if rest_comp else remaining_total + 1

    for pos in range(remaining_total):
        if comp_idx < len(rest_comp) and ((pos + 1) % stride == 0 or rep_idx >= len(selected_repair)):
            curriculum.append(rest_comp[comp_idx])
            comp_idx += 1
        elif rep_idx < len(selected_repair):
            curriculum.append(selected_repair[rep_idx])
            rep_idx += 1
        elif comp_idx < len(rest_comp):
            curriculum.append(rest_comp[comp_idx])
            comp_idx += 1

    curriculum = curriculum[:total]
    actual_comp = sum(1 for ex in curriculum if is_comprehension_example(ex))
    stats = {
        "stage_name": stage_name,
        "total_examples": len(curriculum),
        "comprehension_examples": actual_comp,
        "comprehension_ratio": round(actual_comp / max(1, len(curriculum)), 4),
        "synthesized_examples": len(synthesized),
    }
    return curriculum, stats


# ---------------------------------------------------------------------------
# Dynamic Optimal Evaluation & Benchmark Batching + Deterministic Sampling
# ---------------------------------------------------------------------------

MODEL_EVAL_FOOTPRINT: dict[str, dict[str, float]] = {
    "e2b-it": {"weights_gib": 5.2, "scale_factor": 1.60},
    "2base-2it": {"weights_gib": 9.8, "scale_factor": 1.25},
    "2it-2it": {"weights_gib": 9.8, "scale_factor": 1.25},
    "e4b-it": {"weights_gib": 9.6, "scale_factor": 1.15},
    "4base-2it": {"weights_gib": 13.2, "scale_factor": 1.00},
    "4it-2it": {"weights_gib": 13.2, "scale_factor": 1.00},
    "2base-4it": {"weights_gib": 13.2, "scale_factor": 0.95},
    "2it-4it": {"weights_gib": 13.2, "scale_factor": 0.95},
    "4base-4it": {"weights_gib": 16.5, "scale_factor": 0.75},
    "4it-4it": {"weights_gib": 16.5, "scale_factor": 0.75},
}

BENCHMARK_TIER_CONFIG: dict[str, dict[str, float | int]] = {
    "light": {"per_sample_gib": 2.4, "base_batch": 10, "max_batch": 16},
    "medium": {"per_sample_gib": 3.6, "base_batch": 6, "max_batch": 12},
    "medium_long": {"per_sample_gib": 5.8, "base_batch": 4, "max_batch": 8},
    "heavy": {"per_sample_gib": 8.5, "base_batch": 3, "max_batch": 6},
    "sft_eval": {"per_sample_gib": 5.0, "base_batch": 4, "max_batch": 8},
}


def infer_benchmark_weight_tier(benchmark_item: dict[str, Any] | None) -> str:
    """Classify benchmark evaluation weight tier from collection name and token budgets."""
    if not benchmark_item:
        return "sft_eval"
    if "weight_tier" in benchmark_item and benchmark_item["weight_tier"] in BENCHMARK_TIER_CONFIG:
        return str(benchmark_item["weight_tier"])
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
    profile = MODEL_EVAL_FOOTPRINT.get(
        model_key, {"weights_gib": 14.0, "scale_factor": 1.0}
    )
    tier = infer_benchmark_weight_tier(benchmark_item)
    tier_cfg = BENCHMARK_TIER_CONFIG[tier]
    scale_factor = float(profile["scale_factor"])
    per_sample_gib = float(tier_cfg["per_sample_gib"]) / scale_factor
    max_cap = int(round(float(tier_cfg["max_batch"]) * min(1.25, max(0.65, scale_factor))))
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
    if sample_limit is None or int(sample_limit) <= 0:
        return src_dir

    limit = int(sample_limit)
    inputs_name = "eval_inputs.jsonl" if (src_dir / "eval_inputs.jsonl").exists() else "inputs.jsonl"
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
        budgets = [int(max_new_tokens_list)] * bsz
    else:
        budgets = [int(x) for x in max_new_tokens_list]

    if bsz == 1:
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
                pad_tok = int(pref_ids[0])
                sequences.append([pad_tok] * pad_len + list(pref_ids))
                prefill_valid[b, pad_len:] = True

            outputs: list[list[int]] = [[] for _ in range(bsz)]
            reasons = ["length_limit" for _ in range(bsz)]
            finished = [False for _ in range(bsz)]
            max_steps = max(budgets)

            if getattr(inference_cache_mod.CachedAttention, "__name__", "") != "FullCrossCachedAttention":
                configure_full_rank_cross_attention(model)
            cache = inference_cache_mod.InferenceCache()
            original_cls = [layer.self_attn.__class__ for layer in model.decoder.layers]
            try:
                for layer in model.decoder.layers:
                    layer.self_attn.__class__ = inference_cache_mod.CachedAttention
                for step_idx in range(max_steps):
                    is_prefill = cache.offset == 0
                    ids = torch.tensor(
                        [
                            seq if is_prefill else seq[-1:]
                            for seq in sequences
                        ],
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
                    cache.offset += int(ids.shape[1])
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
    sample_seed = int(benchmark_item.get("sample_seed", BENCHMARK_SAMPLE_SEED)) if benchmark_item else BENCHMARK_SAMPLE_SEED
    effective_dir = prepare_sampled_benchmark_collection(
        directory, sample_limit=sample_limit, sample_seed=sample_seed
    )

    device = next(model.parameters()).device
    resolved_batch_size = (
        int(batch_size)
        if batch_size is not None and int(batch_size) >= 1
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
            "generation_budget_policy": "actual_paired_prompt_v1",
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
            paired_rendered = decoder.apply_chat_template(
                [{"role": "user", "content": paired_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            paired_prefix_size = len(decoder.encode(paired_rendered, add_special_tokens=False))
            new_tokens = min(
                row.get("max_new_tokens", max_new_tokens), decoder_limit - paired_prefix_size
            )
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
            with predictions.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
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


def restore_optimizer_state_by_name(
    model: Any,
    optimizer: Any,
    saved_opt_state: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Map and restore optimizer state tensors (exp_avg, exp_avg_sq, step) by parameter names.

    Iterates through model.named_parameters(), locates parameter state in saved_opt_state["state"]
    using checkpoint param_names mapping, and copies exp_avg, exp_avg_sq, and step directly into
    the active optimizer state tensor for matching parameters.
    """
    saved_groups = saved_opt_state.get("param_groups", [])
    saved_state = saved_opt_state.get("state", {})
    if not saved_state:
        return {"restored": 0, "skipped_shape": 0, "missing": 0, "active_total": 0}

    # 1. Build checkpoint param_id -> param_name mapping
    checkpoint_id_to_name: dict[int, str] = {}
    for g in saved_groups:
        p_names = g.get("param_names")
        p_ids = g.get("params", [])
        if p_names and len(p_names) == len(p_ids):
            for pid, name in zip(p_ids, p_names):
                checkpoint_id_to_name[int(pid) if isinstance(pid, str) and pid.isdigit() else pid] = name

    # Fallback to manifest trainable_names if param_names was not in groups
    if not checkpoint_id_to_name and manifest and "trainable_names" in manifest:
        trainable_names = manifest["trainable_names"]
        all_pids = [pid for g in saved_groups for pid in g.get("params", [])]
        if len(all_pids) == len(trainable_names):
            for pid, name in zip(all_pids, trainable_names):
                checkpoint_id_to_name[int(pid) if isinstance(pid, str) and pid.isdigit() else pid] = name

    checkpoint_name_to_state: dict[str, dict[str, Any]] = {}
    for pid, st in saved_state.items():
        lookup_pid = int(pid) if isinstance(pid, str) and pid.isdigit() else pid
        if lookup_pid in checkpoint_id_to_name:
            checkpoint_name_to_state[checkpoint_id_to_name[lookup_pid]] = st

    # 2. Iterate through model.named_parameters() and map to active optimizer parameters
    active_params = {p for g in optimizer.param_groups for p in g.get("params", [])}

    restored_count = 0
    skipped_shape = 0
    missing_count = 0

    for name, param in model.named_parameters():
        if param not in active_params:
            continue
        if name not in checkpoint_name_to_state:
            missing_count += 1
            continue

        src_state = checkpoint_name_to_state[name]
        # Shape verification guard
        if "exp_avg" in src_state and src_state["exp_avg"].shape != param.shape:
            skipped_shape += 1
            continue
        if "exp_avg_sq" in src_state and src_state["exp_avg_sq"].shape != param.shape:
            skipped_shape += 1
            continue

        # Build restored state dictionary with device and dtype matching
        new_state = {}
        for k, v in src_state.items():
            if isinstance(v, torch.Tensor):
                if k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    new_state[k] = v.to(device=param.device, dtype=param.dtype).clone()
                elif k == "step":
                    new_state[k] = v.to(device=param.device).clone()
                else:
                    new_state[k] = v.to(device=param.device).clone()
            else:
                new_state[k] = copy.deepcopy(v)

        optimizer.state[param] = new_state
        restored_count += 1

    return {
        "restored": restored_count,
        "skipped_shape": skipped_shape,
        "missing": missing_count,
        "active_total": len(active_params),
    }


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
    optimizer_restored_stats = None
    if "optimizer" in trainer_payload and optimizer is not None:
        try:
            optimizer.load_state_dict(trainer_payload["optimizer"])
        except Exception:
            # Group structure varies between stages (e.g. Stage 1 -> Stage 2 adds decoder).
            # Perform parameter-name-matched optimizer state transfer.
            optimizer_restored_stats = restore_optimizer_state_by_name(
                model=model,
                optimizer=optimizer,
                saved_opt_state=trainer_payload["optimizer"],
                manifest=manifest,
            )
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
        "optimizer_restored": optimizer_restored_stats,
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


def synchronize_matrix_v3(
    repo_id: str,
    root: Path | str,
    prefix: str,
    *,
    token: str,
    require_checkpoint: bool = True,
    max_retries: int = 10,
    base_delay: float = 0.05,
) -> dict[str, Any]:
    """Synchronize verified checkpoints and benchmarks to a shared multi-model Hub repo with contention retry."""
    import re

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    from bifocalgemma.training.hub import checked_prefix

    checked_prefix(prefix)
    root = Path(root).resolve()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    checkpoints = sorted(
        p for p in (root / "training").glob("checkpoint-*") if (p / "READY.json").exists()
    )
    for p in checkpoints:
        validate_checkpoint_files(p)

    milestone_names = {f"checkpoint-{STAGE1_STEPS:06d}", f"checkpoint-{TOTAL_STEPS:06d}"}
    retained_set = set(checkpoints[-2:]) | {p for p in checkpoints if p.name in milestone_names}
    retained = sorted(retained_set)
    if not retained and require_checkpoint:
        raise ValueError("No complete checkpoint to snapshot.")
    keep = {p.name for p in retained}

    payloads: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        relative = p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("Snapshot escaped run directory.")
        if not (
            relative == "pipeline_state.json"
            or relative.startswith(("training/", "benchmarks/", "diagnostics/"))
        ):
            continue
        m = re.match(r"training/(checkpoint-\d+)/", relative)
        if m and m[1] not in keep:
            continue
        if p.suffix not in {".json", ".jsonl", ".html", ".pt", ".safetensors"}:
            continue
        payloads[relative] = p

    receipt = {
        "status": "COMPLETED",
        "repo_id": repo_id,
        "prefix": prefix,
        "checkpoint_only": False,
        "retained_checkpoints": sorted(keep),
        "files": {n: file_digest(p) for n, p in payloads.items()},
        "format": "verified_snapshot_v2",
    }
    operations = [
        CommitOperationAdd(path_in_repo=prefix + "/" + n, path_or_fileobj=str(p))
        for n, p in payloads.items()
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=prefix + "/upload_complete.json",
            path_or_fileobj=json.dumps(receipt, indent=2).encode(),
        )
    )

    retryable_codes = {412, 429, 500, 502, 503, 504}
    for attempt in range(max_retries):
        parent = api.model_info(repo_id).sha
        try:
            commit = api.create_commit(
                repo_id,
                operations=operations,
                parent_commit=parent,
                commit_message="Save verified v2 checkpoint and results snapshot",
            )
            break
        except HfHubHTTPError as error:
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code not in retryable_codes or attempt == max_retries - 1:
                raise
            if base_delay > 0:
                delay = base_delay * (1.5 ** min(attempt, 6)) * (0.5 + random.random())
                time.sleep(delay)
    else:
        raise RuntimeError("Snapshot commit retry exhausted.")

    visible = set(api.list_repo_files(repo_id, revision=commit.oid))
    if not all(prefix + "/" + n in visible for n in payloads):
        raise ValueError("Snapshot inventory not visible.")
    for p in retained:
        remote = hf_hub_download(
            repo_id,
            prefix + "/training/" + p.name + "/READY.json",
            revision=commit.oid,
            token=token,
        )
        if Path(remote).read_bytes() != (p / "READY.json").read_bytes():
            raise ValueError("Remote checkpoint marker mismatch.")

    pattern = re.compile(re.escape(prefix) + r"/training/(checkpoint-\d+)/")
    obsolete = [n for n in visible if (m := pattern.match(n)) and m[1] not in keep]
    revision = commit.oid
    if obsolete:
        for attempt in range(max_retries):
            parent = commit.oid if attempt == 0 else api.model_info(repo_id).sha
            try:
                deleted = api.create_commit(
                    repo_id,
                    operations=[CommitOperationDelete(path_in_repo=n) for n in obsolete],
                    parent_commit=parent,
                    commit_message="Prune older checkpoints after verifying replacement",
                )
                revision = deleted.oid
                break
            except HfHubHTTPError as error:
                status_code = getattr(getattr(error, "response", None), "status_code", None)
                if status_code not in retryable_codes or attempt == max_retries - 1:
                    raise
                if base_delay > 0:
                    delay = base_delay * (1.5 ** min(attempt, 6)) * (0.5 + random.random())
                    time.sleep(delay)

    atomic_json(root / "upload_complete.json", receipt)
    return {"revision": revision, "retained_checkpoints": sorted(keep), "verified_at": time.time()}


synchronize_v2 = synchronize_matrix_v3


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
    train_examples: Any,
    dev_examples: Any,
    stage_name: str,
    *,
    policy: dict[str, Any],
    max_steps: int,
    freeze_decoder: bool,
    run_dir: Path,
    compatibility: dict[str, Any],
    start_global_step: int,
    batch_size: int = BATCH_SIZE,
    accumulation: int = ACCUMULATION,
    clip_norm: float = 1.0,
    seed: int = 17,
    progress: Any = None,
    checkpoint_callback: Any = None,
    save_interval_seconds: float = 900.0,
    save_every_steps: int = 500,
    resume_from_checkpoint: Path | str | None = None,
    use_anti_forgetting_curriculum: bool = True,
    decoder: Any = None,
    diff_token_ids: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Execute one training stage (Stage 1 frozen or Stage 2 joint) with AdamW and batch=2."""
    ensure_hardware_memory_guards()
    run_dir = Path(run_dir).resolve()
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    if use_anti_forgetting_curriculum and len(train_examples) >= 8:
        active_examples, curriculum_stats = build_anti_forgetting_curriculum(
            train_examples, stage_name=stage_name, seed=seed, decoder=decoder
        )
    else:
        active_examples = list(train_examples)
        curriculum_stats = {
            "stage_name": stage_name,
            "total_examples": len(active_examples),
            "comprehension_examples": sum(1 for ex in active_examples if is_comprehension_example(ex)),
        }

    groups, stage_summary = configure_matrix_stage(
        model, policy, freeze_decoder=freeze_decoder, total_steps=max_steps
    )
    effective_freeze_decoder = stage_summary["freeze_decoder"]
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
    order = list(range(len(active_examples)))
    if not use_anti_forgetting_curriculum:
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
    collator = RepairCollator()

    for step in range(start_step, max_steps):
        global_step = start_global_step + step + 1
        step_started = time.perf_counter()

        raw_item_groups: list[list[Any]] = []
        micro_batches = []
        for _ in range(accumulation):
            if cursor >= len(order):
                epoch += 1
                order = list(range(len(active_examples)))
                random.Random(seed + epoch).shuffle(order)
                cursor = 0
            indices = order[cursor : cursor + batch_size]
            cursor += len(indices)
            items = [active_examples[i] for i in indices]
            raw_item_groups.append(items)
            micro_batches.append(collator(items))

        token_count = sum(int((b.labels != -100).sum()) for b in micro_batches)
        if not token_count:
            token_count = 1

        active_diff_token_ids = getattr(model, "diff_token_ids", None) or diff_token_ids
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for items, batch_cpu in zip(raw_item_groups, micro_batches, strict=True):
            try:
                batch = batch_cpu.to(device)
                count = int((batch_cpu.labels != -100).sum())
                scale = count / token_count if token_count else (1.0 / len(micro_batches))
                step_loss = compute_v3_step_loss(
                    model, batch, stage_name=stage_name, diff_token_ids=active_diff_token_ids
                )
                if step_loss is None or not bool(torch.isfinite(step_loss)):
                    raise RuntimeError(f"Nonfinite loss at step {step + 1} (global {global_step})")
                (step_loss * scale).backward()
                mean_loss += step_loss.detach().item() * scale
                del step_loss, batch
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(items) <= 1:
                    raise
                # Automatic OOM fallback: split batch=2 microbatch into single-item sub-batches
                for single_item in items:
                    sub_cpu = collator([single_item])
                    sub_batch = sub_cpu.to(device)
                    sub_count = int((sub_cpu.labels != -100).sum())
                    sub_scale = sub_count / token_count if token_count else (1.0 / (len(micro_batches) * len(items)))
                    sub_loss = compute_v3_step_loss(
                        model, sub_batch, stage_name=stage_name, diff_token_ids=active_diff_token_ids
                    )
                    if sub_loss is None or not bool(torch.isfinite(sub_loss)):
                        raise RuntimeError(f"Nonfinite loss at step {step + 1} (global {global_step})")
                    (sub_loss * sub_scale).backward()
                    mean_loss += sub_loss.detach().item() * sub_scale
                    del sub_loss, sub_batch

        if effective_freeze_decoder:
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
            "freeze_decoder": effective_freeze_decoder,
            "max_steps": TOTAL_STEPS,
            "train_loss": mean_loss,
            "target_tokens": token_count,
            "gradient_norm": grad_norm,
            "learning_rates": rates,
            "step_seconds": step_seconds,
            "batch_size": batch_size,
            "accumulation": accumulation,
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

    if effective_freeze_decoder:
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
        model_key = str(compatibility.get("model_key", "4base-4it"))
        eval_bs = (
            resolve_optimal_eval_batch_size(model_key, None, device=device)
            if batch_size >= BATCH_SIZE
            else batch_size
        )
        while True:
            try:
                held_out = evaluate_sft_loss(
                    model,
                    dev_examples,
                    batch_size=eval_bs,
                    device=device,
                    event_callback=progress,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if eval_bs <= 1:
                    raise
                eval_bs = max(1, eval_bs // 2)

    return {
        "stage": stage_name,
        "completed_steps": max_steps,
        "end_global_step": start_global_step + max_steps,
        "held_out_loss": held_out,
        "curriculum": curriculum_stats,
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
    """Execute complete two-stage matrix training run (5,551 Stage 1 + 5,551 Stage 2 = 11,102 steps)."""
    ensure_hardware_memory_guards()
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    policy = unified_recipe(cfg["family"])
    for grp_settings in policy.get("groups", {}).values():
        grp_settings["warmup_ratio"] = WARMUP_RATIO
    repo = cfg["repo"]
    subfolder = cfg.get("subfolder", model_key)

    root = Path(output_root) / model_key / "train"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{subfolder}/{spec.name}/train"

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
        "version": "bifocalgemma_v3_swiglu_cross",
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
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
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
                "repo": repo,
                "subfolder": subfolder,
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

    decoder_tokenizer = None
    try:
        from bifocalgemma.data.tokenization import load_verified_tokenizer

        decoder_tokenizer, _ = load_verified_tokenizer(spec.decoder, Path(cache) / "tokenizers")
    except Exception:
        pass
    if not hasattr(model, "diff_token_ids") or getattr(model, "diff_token_ids", None) is None:
        object.__setattr__(model, "diff_token_ids", resolve_diff_token_ids(decoder_tokenizer))

    compatibility = {
        "architecture": (
            "swiglu_dual_gate_decoupled_cross_v3"
            if spec.encoder is not None
            else "decoder_only_causal_v3"
        ),
        "experiment": spec.name,
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
        "model_spec": spec.to_dict(),
        "sources": provenance,
        "data_fingerprint": manifest["fingerprint"],
        "matched_target_fingerprints": manifest["matched_target_fingerprints"],
        "seed": 17,
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
        "comprehension_mix_ratio": COMPREHENSION_MIX_RATIO,
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
            freeze_decoder=spec.encoder is not None,
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
            decoder=decoder_tokenizer,
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
            decoder=decoder_tokenizer,
        )
        progress({"stage": "stage2_completed", "last_checkpoint": stage2_report["last_checkpoint"]})
        stage2_files = [p for p in Path(stage2_report["last_checkpoint"]).glob("*") if p.is_file()]
        ledger.finish("stage2_joint", stage2_files, stage2_report)
        snapshot(final=True)

    final_report = {
        "status": "COMPLETED",
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
        "spec": spec.to_dict(),
        "global_step": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
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
    """Execute 8-benchmark suite (6 full first + 2 deterministic 100-sample last) with dynamic batching."""
    ensure_hardware_memory_guards()
    patch_matrix_tokenization()
    if model_key not in MATRIX_MODELS:
        raise ValueError(f"Unknown model_key {model_key!r}. Choose from {list(MATRIX_MODELS)}.")
    cfg = MATRIX_MODELS[model_key]
    spec = make_experiment_spec(model_key)
    policy = unified_recipe(cfg["family"])
    for grp_settings in policy.get("groups", {}).values():
        grp_settings["warmup_ratio"] = WARMUP_RATIO
    repo = cfg["repo"]
    subfolder = cfg.get("subfolder", model_key)

    root = Path(output_root) / model_key / "train"
    prefix = f"{subfolder}/{spec.name}/train"

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
        "version": "bifocalgemma_v3_swiglu_cross",
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
        "batch_size": BATCH_SIZE,
        "accumulation": ACCUMULATION,
        "model_key": model_key,
        "repo": repo,
        "subfolder": subfolder,
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
            "subfolder": subfolder,
            "benchmarks": reports,
        }

    # Load model for inference
    model, _ = download_model(spec, cache, token, progress)
    configure_policy_model(model, copy.deepcopy(policy))
    configure_memory_efficient_training(model)
    configure_full_rank_cross_attention(model)
    load_matrix_adapters_for_inference(model, checkpoint_path)

    device = next(model.parameters()).device
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
        eval_batch_size = resolve_optimal_eval_batch_size(model_key, item, device=device)
        report = run_batched_context_benchmark(
            model,
            spec,
            collection_dir,
            destination,
            Path(cache) / "tokenizers",
            checkpoint_identity=checkpoint_identity,
            model_key=model_key,
            benchmark_item=item,
            batch_size=eval_batch_size,
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
        "subfolder": subfolder,
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
    training_res = run_matrix_training(
        model_key,
        token=token,
        mo=mo,
        output_root=output_root,
        cache=cache,
        upload=upload,
        monitor=monitor,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluation_res = run_matrix_benchmark(
        model_key,
        token=token,
        mo=mo,
        output_root=output_root,
        cache=cache,
        upload=upload,
        monitor=monitor,
    )
    return {
        "status": evaluation_res["status"],
        "model": evaluation_res["model"],
        "model_key": model_key,
        "repo": evaluation_res["repo"],
        "subfolder": evaluation_res.get("subfolder", model_key),
        "training": training_res,
        "benchmarks": evaluation_res["benchmarks"],
    }
