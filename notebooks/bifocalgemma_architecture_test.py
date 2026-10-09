# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "marimo==0.25.1",
#     "torch==2.14.1",
#     "transformers==5.19.0",
#     "pytest==8.4.2",
# ]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
# Standalone Molab notebook. Core definitions mirror the audited repo modules.
# Random initialization only. No checkpoint/data download, RL, VM or upload.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # BifocalGemma · Architecture Test
    Notebook kecil untuk memeriksa wiring Gemma 4: PLE/RoPE, merged attention,
    word tying, patch loss, dan backward. Control center ada di cell berikutnya.
    Default: random **tiny full-depth**. Dimensi asli diperiksa pada meta;
    forward/backward dimensi asli dapat diaktifkan untuk GPU Molab 96 GB.
    """)
    return (mo,)


@app.cell
def _(mo):
    # 1. CONTROL CENTER — cukup ubah bagian ini.
    EXPERIMENT = "BifocalGemma-E2B-E2B"  # Empat pasangan atau Gemma-4-E2B-IT/E4B-IT.
    DEVICE = "auto"  # auto / cpu / cuda
    PRECISION = "auto"  # auto = BF16 pada CUDA, FP32 pada CPU
    RUN_MATRIX = True  # Tiny tests keenam konfigurasi
    RUN_FULL_SIZE = False  # Opsional: satu full-size random forward/backward
    SEED = 17
    ENCODER_TOKENS = 16
    TARGET_TOKENS = 4
    SAVE_REPORT = True
    OUTPUT_DIR = "outputs/marimo_architecture"
    mo.md("## 1 · Control center\nSemua konfigurasi eksperimen berada dalam satu cell.")
    return (
        EXPERIMENT,
        DEVICE,
        PRECISION,
        RUN_MATRIX,
        RUN_FULL_SIZE,
        SEED,
        ENCODER_TOKENS,
        TARGET_TOKENS,
        SAVE_REPORT,
        OUTPUT_DIR,
    )


@app.cell
def _(DEVICE, PRECISION, mo):
    # 2. ENVIRONMENT — pakai dependency yang sudah terpasang; tidak ada pip di cell.
    import copy
    import gc
    import json
    import os
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import version
    from pathlib import Path
    from types import SimpleNamespace
    from typing import cast

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import Gemma4TextConfig, Gemma4TextModel
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextAttention,
        Gemma4TextDecoderLayer,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    assert version("transformers") == "5.19.0", "Gunakan transformers==5.19.0."
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_float32_matmul_precision("highest")
    device_request = os.environ.get("BIFOCALGEMMA_NOTEBOOK_DEVICE", DEVICE)
    precision_request = os.environ.get("BIFOCALGEMMA_NOTEBOOK_PRECISION", PRECISION)
    assert device_request in {"auto", "cpu", "cuda"}
    assert precision_request in {"auto", "float32", "bfloat16"}
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_request == "auto"
        else device_request
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia.")
    dtype = (
        torch.bfloat16
        if precision_request == "bfloat16" or (precision_request == "auto" and device == "cuda")
        else torch.float32
    )
    if dtype == torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU/runtime ini belum mendukung BF16.")
    environment = {
        "python": platform.python_version(),
        "marimo": version("marimo"),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda_build": torch.version.cuda,
        "device": device,
        "precision": str(dtype),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30
        if device == "cuda"
        else None,
        "compute_capability": list(torch.cuda.get_device_capability(0))
        if device == "cuda"
        else None,
    }
    mo.vstack([mo.md("## 2 · Environment"), mo.ui.table([environment], selection=None)])
    return (
        copy,
        gc,
        json,
        os,
        datetime,
        timezone,
        version,
        Path,
        SimpleNamespace,
        cast,
        torch,
        nn,
        F,
        Gemma4TextConfig,
        Gemma4TextModel,
        Gemma4TextAttention,
        Gemma4TextDecoderLayer,
        apply_rotary_pos_emb,
        repeat_kv,
        device,
        dtype,
        environment,
    )


@app.cell
def _(SimpleNamespace, copy, version, Gemma4TextConfig, mo):
    # 3. CONFIG NATIVE — metadata config yang sudah diaudit, tanpa network.
    AUDITED_CONFIGS = {
        "E2B": {
            "model_id": "google/gemma-4-E2B-it",
            "source_url": "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/config.json",
            "verified_on": "2026-10-05",
            "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "upstream_transformers_version": "5.5.0.dev0",
            "text_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "attention_k_eq_v": False,
                "bos_token_id": 2,
                "dtype": "bfloat16",
                "enable_moe_block": False,
                "eos_token_id": 1,
                "expert_intermediate_size": None,
                "final_logit_softcapping": 30.0,
                "global_head_dim": 512,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 1536,
                "hidden_size_per_layer_input": 256,
                "initializer_range": 0.02,
                "intermediate_size": 6144,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "max_position_embeddings": 131072,
                "model_type": "gemma4_text",
                "num_attention_heads": 8,
                "num_experts": None,
                "num_global_key_value_heads": None,
                "num_hidden_layers": 35,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 20,
                "pad_token_id": 0,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional",
                    },
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                },
                "sliding_window": 512,
                "tie_word_embeddings": True,
                "top_k_experts": None,
                "use_bidirectional_attention": None,
                "use_cache": True,
                "use_double_wide_mlp": True,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
            },
        },
        "E4B": {
            "model_id": "google/gemma-4-E4B-it",
            "source_url": "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/config.json",
            "verified_on": "2026-10-05",
            "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
            "upstream_transformers_version": "5.5.0.dev0",
            "text_config": {
                "attention_bias": False,
                "attention_dropout": 0.0,
                "attention_k_eq_v": False,
                "bos_token_id": 2,
                "dtype": "bfloat16",
                "enable_moe_block": False,
                "eos_token_id": 1,
                "expert_intermediate_size": None,
                "final_logit_softcapping": 30.0,
                "global_head_dim": 512,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "hidden_size": 2560,
                "hidden_size_per_layer_input": 256,
                "initializer_range": 0.02,
                "intermediate_size": 10240,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "max_position_embeddings": 131072,
                "model_type": "gemma4_text",
                "num_attention_heads": 8,
                "num_experts": None,
                "num_global_key_value_heads": None,
                "num_hidden_layers": 42,
                "num_key_value_heads": 2,
                "num_kv_shared_layers": 18,
                "pad_token_id": 0,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional",
                    },
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                },
                "sliding_window": 512,
                "tie_word_embeddings": True,
                "top_k_experts": None,
                "use_bidirectional_attention": None,
                "use_cache": True,
                "use_double_wide_mlp": False,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
            },
        },
    }

    PAIRS = (
        ("BifocalGemma-E2B-E2B", "E2B", "E2B"),
        ("BifocalGemma-E4B-E4B", "E4B", "E4B"),
        ("BifocalGemma-E2B-E4B", "E2B", "E4B"),
        ("BifocalGemma-E4B-E2B", "E4B", "E2B"),
        ("Gemma-4-E2B-IT", None, "E2B"),
        ("Gemma-4-E4B-IT", None, "E4B"),
    )

    def experiment_spec(name):
        for candidate, encoder_size, decoder_size in PAIRS:
            if candidate == name:
                return SimpleNamespace(
                    name=name,
                    encoder=SimpleNamespace(size=encoder_size) if encoder_size else None,
                    decoder=SimpleNamespace(size=decoder_size),
                    tie_encoder_decoder_words=encoder_size == decoder_size,
                )
        raise ValueError(f"Unknown experiment: {name}")

    def native_config(size: str, *, tiny: bool = True, encoder: bool = False):
        """Full dimensions are for meta inspection only; factory always uses tiny."""
        if version("transformers") != "5.19.0":
            raise RuntimeError("Native integration is validated against transformers==5.19.0.")
        values = copy.deepcopy(AUDITED_CONFIGS[size]["text_config"])
        if tiny:
            width = 32 if size == "E2B" else 48
            values.update(
                vocab_size=128,
                vocab_size_per_layer_input=128,
                hidden_size=width,
                intermediate_size=4 * width,
                hidden_size_per_layer_input=4,
                head_dim=8,
                global_head_dim=16,
            )
        values.update(use_cache=False, use_bidirectional_attention="all" if encoder else None)
        config = Gemma4TextConfig.from_dict(values)
        config._attn_implementation = "eager"
        return config

    mo.md(
        "## 3 · Config Gemma\n35/42 layers; local/global heads, KV sharing, PLE, dan varian MLP dipertahankan."
    )
    return (
        AUDITED_CONFIGS,
        PAIRS,
        experiment_spec,
        native_config,
    )


@app.cell
def _(AUDITED_CONFIGS, Gemma4TextModel, PAIRS, native_config, torch, mo):
    # 4. DIMENSI ASLI — hanya meta, tanpa alokasi bobot GPU.
    text_parameter_counts = {}
    meta_rows = []
    for meta_size in ("E2B", "E4B"):
        with torch.device("meta"):
            meta_model = Gemma4TextModel(native_config(meta_size, tiny=False))
        assert all(meta_parameter.is_meta for meta_parameter in meta_model.parameters())
        text_parameter_counts[meta_size] = sum(
            meta_parameter.numel() for meta_parameter in meta_model.parameters()
        )
        meta_rows.append(
            {
                "size": meta_size,
                "layers": len(meta_model.layers),
                "hidden": meta_model.config.hidden_size,
                "ordinary_words": str(tuple(meta_model.embed_tokens.weight.shape)),
                "PLE": str(tuple(meta_model.embed_tokens_per_layer.weight.shape)),
                "text_parameters": text_parameter_counts[meta_size],
                "revision_it_config": AUDITED_CONFIGS[meta_size]["revision"],
                "device": "meta",
            }
        )
        del meta_model
    full_parameter_counts = {}
    for meta_name, meta_encoder, meta_decoder in PAIRS:
        meta_pair_count = text_parameter_counts[meta_decoder] + (
            text_parameter_counts[meta_encoder] if meta_encoder else 0
        )
        if meta_encoder == meta_decoder:
            meta_word_config = AUDITED_CONFIGS[meta_decoder]["text_config"]
            meta_pair_count -= meta_word_config["vocab_size"] * meta_word_config["hidden_size"]
        elif meta_encoder:
            meta_pair_count += (
                AUDITED_CONFIGS[meta_encoder]["text_config"]["hidden_size"]
                * AUDITED_CONFIGS[meta_decoder]["text_config"]["hidden_size"]
            )
        full_parameter_counts[meta_name] = meta_pair_count
    mo.vstack([mo.md("## 4 · Dimensi asli pada meta"), mo.ui.table(meta_rows, selection=None)])
    return (
        text_parameter_counts,
        meta_rows,
        full_parameter_counts,
    )


@app.cell
def _(torch, F, mo):
    from dataclasses import dataclass, field

    # 5. BATCH & LOSS — synthetic IDs, shift sekali, prefix/padding label -100.
    @dataclass(frozen=True)
    class TokenizedRepairExample:
        """Prefix is already IT-formatted; target contains the closing target token."""

        instance_id: str
        encoder_ids: tuple[int, ...]
        decoder_prefix_ids: tuple[int, ...]
        target_ids: tuple[int, ...]
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Seq2SeqBatch:
        encoder_input_ids: torch.Tensor
        encoder_attention_mask: torch.Tensor
        decoder_input_ids: torch.Tensor
        decoder_attention_mask: torch.Tensor
        labels: torch.Tensor

        def to(self, device: str | torch.device) -> "Seq2SeqBatch":
            return Seq2SeqBatch(
                self.encoder_input_ids.to(device),
                self.encoder_attention_mask.to(device),
                self.decoder_input_ids.to(device),
                self.decoder_attention_mask.to(device),
                self.labels.to(device),
            )

        @property
        def encoder_position_ids(self) -> torch.Tensor:
            return (self.encoder_attention_mask.long().cumsum(-1) - 1).clamp_min(0)

        @property
        def decoder_position_ids(self) -> torch.Tensor:
            return (self.decoder_attention_mask.long().cumsum(-1) - 1).clamp_min(0)

    @dataclass
    class ModelOutput:
        logits: torch.Tensor
        loss: torch.Tensor | None

    class RepairCollator:
        def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
            self.pad_token_id = pad_token_id
            self.ignore_index = ignore_index

        def __call__(self, examples: list[TokenizedRepairExample]) -> Seq2SeqBatch:
            if not examples:
                raise ValueError("Cannot collate an empty batch.")
            encoder_presence = [bool(example.encoder_ids) for example in examples]
            if any(encoder_presence) and not all(encoder_presence):
                raise ValueError(
                    "Do not mix decoder-only and encoder-decoder examples in one batch."
                )
            for example in examples:
                if not example.decoder_prefix_ids or not example.target_ids:
                    raise ValueError("Decoder prefix and target must each contain tokens.")
                if (
                    min((*example.encoder_ids, *example.decoder_prefix_ids, *example.target_ids))
                    < 0
                ):
                    raise ValueError("Input token IDs must be nonnegative.")
            enc_len = max(len(example.encoder_ids) for example in examples)
            dec_len = max(
                len(example.decoder_prefix_ids) + len(example.target_ids) - 1
                for example in examples
            )
            size = len(examples)
            encoder = torch.full((size, enc_len), self.pad_token_id, dtype=torch.long)
            decoder = torch.full((size, dec_len), self.pad_token_id, dtype=torch.long)
            enc_mask = torch.zeros_like(encoder, dtype=torch.bool)
            dec_mask = torch.zeros_like(decoder, dtype=torch.bool)
            labels = torch.full_like(decoder, self.ignore_index)
            for row, example in enumerate(examples):
                encoder[row, -len(example.encoder_ids) :] = torch.tensor(example.encoder_ids)
                enc_mask[row, -len(example.encoder_ids) :] = True
                combined = example.decoder_prefix_ids + example.target_ids
                inputs = combined[:-1]
                targets = list(combined[1:])
                targets[: len(example.decoder_prefix_ids) - 1] = [self.ignore_index] * (
                    len(example.decoder_prefix_ids) - 1
                )
                decoder[row, -len(inputs) :] = torch.tensor(inputs)
                dec_mask[row, -len(inputs) :] = True
                labels[row, -len(targets) :] = torch.tensor(targets)
            return Seq2SeqBatch(encoder, enc_mask, decoder, dec_mask, labels)

    def patch_cross_entropy(
        logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
            raise ValueError("Expected logits [B,L,V] and already aligned labels [B,L].")
        if not bool((labels != ignore_index).any()):
            return logits.sum() * 0
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )

    def synthetic_batch(spec, encoder_tokens, target_tokens, device):
        if not 2 <= encoder_tokens <= 128 or not 2 <= target_tokens <= 32:
            raise ValueError("Gunakan encoder_tokens 2..128 dan target_tokens 2..32 untuk smoke.")
        context = tuple([2] + [3 + index % 80 for index in range(encoder_tokens - 1)])
        target = tuple([18 + index % 60 for index in range(target_tokens - 1)] + [106])
        examples = [
            TokenizedRepairExample(
                "synthetic-1", context if spec.encoder else (), (2, 105, 7), target
            ),
            TokenizedRepairExample(
                "synthetic-2",
                context[:-1] if spec.encoder else (),
                (2, 105, 8),
                target[:-2] + (106,),
            ),
        ]
        batch = RepairCollator()(examples).to(device)
        first_target = 2
        assert batch.labels[0, first_target].item() == target[0]
        assert (batch.labels[0, :first_target] == -100).all()
        return batch

    mo.md(
        "## 5 · Batch dan patch loss\nIDs 0..127 bersifat synthetic; tokenizer asli tetap diuji terpisah."
    )
    return (
        TokenizedRepairExample,
        Seq2SeqBatch,
        ModelOutput,
        RepairCollator,
        patch_cross_entropy,
        synthetic_batch,
    )


@app.cell
def _(Gemma4TextAttention, apply_rotary_pos_emb, repeat_kv, nn, torch, mo):
    # 6. MERGED ATTENTION — native proyeksi/norm + satu joint softmax.
    def build_joint_visibility(
        encoder_valid: torch.Tensor,
        decoder_valid: torch.Tensor,
        self_window: int | None = None,
    ) -> torch.Tensor:
        """Keys are [decoder, encoder]; decoder causal/windowed, encoder globally visible."""
        if encoder_valid.ndim != 2 or decoder_valid.ndim != 2:
            raise ValueError("Validity masks must have [batch, sequence] shape.")
        if encoder_valid.shape[0] != decoder_valid.shape[0]:
            raise ValueError("Encoder and decoder batch sizes differ.")
        if self_window is not None and self_window <= 0:
            raise ValueError("Self-attention window must be positive.")
        length = decoder_valid.shape[1]
        positions = torch.arange(length, device=decoder_valid.device)
        visibility = positions[None, :] <= positions[:, None]
        if self_window is not None:
            visibility &= positions[None, :] > positions[:, None] - self_window
        self_visible = visibility[None] & decoder_valid.bool()[:, None, :]
        cross_visible = encoder_valid.bool()[:, None, :].expand(-1, length, -1)
        return (
            torch.cat((self_visible, cross_visible), dim=-1)[:, None]
            & decoder_valid.bool()[:, None, :, None]
        )

    class NativeMergedAttention(Gemma4TextAttention):
        """Preserve native parameter names and KV owners; concatenate after self RoPE."""

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            shared_kv_states,
            past_key_values=None,
            *,
            encoder_hidden_states=None,
            encoder_valid=None,
            decoder_valid=None,
            **kwargs,
        ):
            if past_key_values is not None:
                raise ValueError("Native tiny merged attention does not support temporal cache.")
            if encoder_hidden_states is None or encoder_valid is None or decoder_valid is None:
                raise ValueError(
                    "Merged attention requires explicit encoder states and validity masks."
                )
            batch, length, _ = hidden_states.shape
            shape = (batch, length, -1, self.head_dim)
            cos, sin = position_embeddings
            query = self.q_norm(self.q_proj(hidden_states).view(shape))
            query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            if self.is_kv_shared_layer:
                key, value = shared_kv_states[self.layer_type]
            else:
                assert self.v_proj is not None
                key = self.k_norm(self.k_proj(hidden_states).view(shape))
                key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                value = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
                cross_shape = (
                    batch,
                    encoder_hidden_states.shape[1],
                    self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                    self.head_dim,
                )
                cross_key = self.k_norm(self.k_proj(encoder_hidden_states).view(cross_shape))
                cross_value = self.v_norm(self.v_proj(encoder_hidden_states).view(cross_shape))
                key = torch.cat((key, cross_key.transpose(1, 2)), dim=2)
                value = torch.cat((value, cross_value.transpose(1, 2)), dim=2)
            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = (key, value)
            allowed = build_joint_visibility(encoder_valid, decoder_valid, self.sliding_window)
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
            scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
            scores = scores.masked_fill(~allowed, float("-inf"))
            scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
            probabilities = probabilities.masked_fill(~allowed, 0)
            probabilities = nn.functional.dropout(
                probabilities, p=float(self.attention_dropout or 0.0), training=self.training
            )
            result = (probabilities @ value).transpose(1, 2).contiguous().reshape(batch, length, -1)
            return self.o_proj(result), None

    mo.md(
        "## 6 · Merged attention\nSelf keys causal/windowed; encoder keys global. Shared layers menggunakan KV pemiliknya."
    )
    return (
        build_joint_visibility,
        NativeMergedAttention,
    )


@app.cell
def _(
    Gemma4TextModel,
    Gemma4TextDecoderLayer,
    ModelOutput,
    NativeMergedAttention,
    Seq2SeqBatch,
    cast,
    native_config,
    nn,
    patch_cross_entropy,
    torch,
    mo,
):
    # 7. MODEL BUILDER — encoder bidirectional + decoder native; tying simetris.
    class ArchitectureModel(nn.Module):
        """Native full-depth text stacks at reduced width; never a pretrained model."""

        def __init__(self, spec, *, tiny=True):
            super().__init__()
            self.spec = spec
            self.decoder = Gemma4TextModel(native_config(spec.decoder.size, tiny=tiny))
            self.encoder = (
                Gemma4TextModel(native_config(spec.encoder.size, tiny=tiny, encoder=True))
                if spec.encoder
                else None
            )
            width = self.decoder.config.hidden_size
            self.encoder_projection = None
            if self.encoder is not None:
                enc_width = self.encoder.config.hidden_size
                self.encoder_projection = (
                    nn.Identity() if enc_width == width else nn.Linear(enc_width, width, bias=False)
                )
                for index, layer in enumerate(self.decoder.layers):
                    layer = cast(Gemma4TextDecoderLayer, layer)
                    attention = NativeMergedAttention(self.decoder.config, index)
                    attention.load_state_dict(layer.self_attn.state_dict(), strict=True)
                    layer.self_attn = attention
            self.lm_head = nn.Linear(width, self.decoder.config.vocab_size, bias=False)
            self._tie_words()

        def _tie_words(self):
            self.lm_head.weight = self.decoder.embed_tokens.weight
            if self.encoder is not None and self.spec.tie_encoder_decoder_words:
                self.encoder.embed_tokens.weight = self.decoder.embed_tokens.weight

        def _apply(self, fn, recurse=True):
            super()._apply(fn, recurse=recurse)
            self._tie_words()
            return self

        def load_state_dict(self, state_dict, strict=True, assign=False):
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
            self._tie_words()
            return result

        @staticmethod
        def _positions(mask):
            return (mask.long().cumsum(-1) - 1).clamp_min(0)

        def encode_context(self, input_ids, attention_mask):
            if self.encoder is None or self.encoder_projection is None:
                raise ValueError("Decoder-only baseline has no encoder.")
            if input_ids.shape[1] == 0:
                return self.decoder.embed_tokens.weight.new_empty(
                    input_ids.shape[0], 0, self.decoder.config.hidden_size
                )
            hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=self._positions(attention_mask),
                use_cache=False,
            ).last_hidden_state
            # No padding states enter K/V, including batches with wholly masked context.
            return self.encoder_projection(hidden).masked_fill(~attention_mask.bool()[..., None], 0)

        def decode_logits(self, input_ids, decoder_mask, encoder_states=None, encoder_mask=None):
            kwargs = {}
            if self.encoder is not None:
                if encoder_states is None:
                    encoder_states = self.decoder.embed_tokens.weight.new_empty(
                        input_ids.shape[0], 0, self.decoder.config.hidden_size
                    )
                    encoder_mask = decoder_mask.new_zeros((input_ids.shape[0], 0))
                if encoder_mask is None or encoder_mask.shape != encoder_states.shape[:2]:
                    raise ValueError("Encoder mask must match encoder states.")
                kwargs = dict(
                    encoder_hidden_states=encoder_states,
                    encoder_valid=encoder_mask,
                    decoder_valid=decoder_mask,
                )
            elif encoder_states is not None or encoder_mask is not None:
                raise ValueError("Decoder-only baseline cannot receive encoder states.")
            hidden = self.decoder(
                input_ids=input_ids,
                attention_mask=decoder_mask,
                position_ids=self._positions(decoder_mask),
                use_cache=False,
                **kwargs,
            ).last_hidden_state
            logits = self.lm_head(hidden)
            cap = self.decoder.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap
            return logits

        def forward(self, batch: Seq2SeqBatch) -> ModelOutput:
            if self.encoder is None:
                if batch.encoder_input_ids.shape[1] != 0:
                    raise ValueError("Baseline requires an empty encoder sequence.")
                hidden, mask = None, None
            else:
                hidden = self.encode_context(batch.encoder_input_ids, batch.encoder_attention_mask)
                mask = batch.encoder_attention_mask
            logits = self.decode_logits(
                batch.decoder_input_ids, batch.decoder_attention_mask, hidden, mask
            )
            return ModelOutput(logits, patch_cross_entropy(logits, batch.labels))

    def build_model(spec, device, dtype, *, tiny=True):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            # Full-size optional run allocates directly on CUDA in BF16.
            with torch.device(device):
                model = ArchitectureModel(spec, tiny=tiny)
        finally:
            torch.set_default_dtype(previous_dtype)
        return model

    mo.md(
        "## 7 · Perakitan model\nSatu alignment projection asimetris; ordinary words simetris tied, PLE terpisah."
    )
    return (
        ArchitectureModel,
        build_model,
    )


@app.cell
def _(Gemma4TextAttention, build_model, copy, nn, synthetic_batch, torch, mo):
    # 8. TEST HELPERS — assertions harus lulus; error tetap terlihat di cell.
    def check_tiny_logic(spec, device, dtype, encoder_tokens, target_tokens, seed):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_model(spec, device, dtype).eval()
        batch = synthetic_batch(spec, encoder_tokens, target_tokens, device)
        output = model(batch)
        assert output.loss is not None
        assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert all(g.abs().sum() > 0 for g in gradients)
        assert model.lm_head.weight is model.decoder.embed_tokens.weight
        if spec.encoder:
            assert model.encoder is not None
            assert (
                model.encoder.embed_tokens.weight is model.decoder.embed_tokens.weight
            ) == spec.tie_encoder_decoder_words
            assert (
                model.encoder.embed_tokens_per_layer.weight
                is not model.decoder.embed_tokens_per_layer.weight
            )
            assert (
                sum(a.self_attn.is_kv_shared_layer for a in model.decoder.layers)
                == model.decoder.config.num_kv_shared_layers
            )
            assert all(
                not hasattr(layer.self_attn, "k_proj")
                for layer in model.decoder.layers
                if layer.self_attn.is_kv_shared_layer
            )
            if spec.tie_encoder_decoder_words:
                assert isinstance(model.encoder_projection, nn.Identity)
            else:
                assert isinstance(model.encoder_projection, nn.Linear)
        parameter_ids = [id(p) for p in model.parameters()]
        assert len(parameter_ids) == len(set(parameter_ids))
        checks = [
            "prefix_loss_mask",
            "single_shift",
            "finite_forward_backward",
            "all_gradients_nonzero",
            "embedding_aliases",
            "unique_parameters",
        ]
        # Semantic parity is checked in FP32, separately from the requested
        # forward/backward precision. BF16 rounding accumulates at full depth.
        model = model.to(dtype=torch.float32)
        tolerance = 2e-5
        with torch.no_grad():
            # No-context result must reproduce a decoder using original native attention.
            # Preserve nonpersistent native RoPE/embedding buffers too: creating
            # a fresh FP32 model + load_state_dict loses their BF16 initialization.
            native = copy.deepcopy(model.decoder)
            for index, layer in enumerate(native.layers):
                attention = Gemma4TextAttention(native.config, index).to(
                    device=device, dtype=torch.float32
                )
                attention.load_state_dict(layer.self_attn.state_dict())
                layer.self_attn = attention
            native.eval()
            ids, mask = batch.decoder_input_ids, batch.decoder_attention_mask
            expected = model.lm_head(
                native(
                    ids, attention_mask=mask, position_ids=model._positions(mask), use_cache=False
                ).last_hidden_state
            )
            cap = model.decoder.config.final_logit_softcapping
            if cap is not None:
                expected = torch.tanh(expected / cap) * cap
            empty = model.decode_logits(ids, mask)
            try:
                torch.testing.assert_close(
                    empty[mask], expected[mask], atol=tolerance, rtol=tolerance
                )
            except AssertionError as error:
                raise AssertionError(f"{spec.name}: native_decoder_parity: {error}") from error
            del native
            checks.append("native_decoder_parity_without_context")
            changed = ids.clone()
            changed[:, -1] = 99
            future = model.decode_logits(changed, mask)
            try:
                torch.testing.assert_close(
                    empty[:, :-1][mask[:, :-1]],
                    future[:, :-1][mask[:, :-1]],
                    atol=tolerance,
                    rtol=0,
                )
            except AssertionError as error:
                raise AssertionError(f"{spec.name}: decoder_causality: {error}") from error
            checks.append("decoder_causality")
            if spec.encoder:
                h = model.encode_context(batch.encoder_input_ids, batch.encoder_attention_mask)
                with_context = model.decode_logits(ids, mask, h, batch.encoder_attention_mask)
                changed_h = h.clone()
                changed_h[:, -1] = 0
                changed_result = model.decode_logits(
                    ids, mask, changed_h, batch.encoder_attention_mask
                )
                assert not torch.equal(with_context[mask], changed_result[mask])
                no_keys = torch.zeros_like(batch.encoder_attention_mask)
                masked = model.decode_logits(ids, mask, h, no_keys)
                try:
                    torch.testing.assert_close(
                        masked[mask], empty[mask], atol=tolerance, rtol=tolerance
                    )
                except AssertionError as error:
                    raise AssertionError(f"{spec.name}: masked_encoder_parity: {error}") from error
                checks.extend(
                    ["encoder_context_influence", "masked_encoder_parity", "KV_sharing_structure"]
                )
        result = {
            "model": spec.name,
            "precision": str(dtype),
            "semantic_check_precision": "float32",
            "status": "PASS",
            "checks": checks,
            "check_count": len(checks),
            "loss": output.loss.detach().item(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "encoder_layers": len(model.encoder.layers) if model.encoder else 0,
            "decoder_layers": len(model.decoder.layers),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
            if device == "cuda"
            else None,
            "pretrained_weights_loaded": False,
        }
        return result

    mo.md(
        "## 8 · Test logika\nForward/backward, loss masking, aliases, native parity, causality, dan pengaruh encoder."
    )
    return (check_tiny_logic,)


@app.cell
def _(
    NativeMergedAttention,
    build_joint_visibility,
    experiment_spec,
    native_config,
    patch_cross_entropy,
    synthetic_batch,
    torch,
):
    def test_local_self_window_keeps_global_encoder_keys():
        encoder_valid = torch.ones(1, 2, dtype=torch.bool)
        decoder_valid = torch.ones(1, 513, dtype=torch.bool)
        visible = build_joint_visibility(encoder_valid, decoder_valid, self_window=512)
        assert not visible[0, 0, -1, 0]
        assert visible[0, 0, -1, 1:513].all()
        assert visible[0, 0, -1, 513:].all()
        assert not visible[0, 0, 0, 1:513].any()

    def test_prefix_and_padding_receive_zero_loss_gradient():
        spec = experiment_spec("BifocalGemma-E2B-E2B")
        batch = synthetic_batch(spec, 4, 4, "cpu")
        logits = torch.randn(*batch.labels.shape, 128, requires_grad=True)
        loss = patch_cross_entropy(logits, batch.labels)
        loss.backward()
        assert logits.grad is not None
        assert (logits.grad[batch.labels == -100] == 0).all()
        assert logits.grad[batch.labels != -100].abs().sum() > 0

    def test_fully_masked_attention_has_finite_backward():
        attention = NativeMergedAttention(native_config("E2B"), 0)
        states = torch.randn(1, 3, 32, requires_grad=True)
        context = torch.randn(1, 2, 32, requires_grad=True)
        positions = (torch.ones(1, 3, 8), torch.zeros(1, 3, 8))
        output, _ = attention(
            states,
            positions,
            None,
            {},
            encoder_hidden_states=context,
            encoder_valid=torch.zeros(1, 2, dtype=torch.bool),
            decoder_valid=torch.zeros(1, 3, dtype=torch.bool),
        )
        assert (output == 0).all()
        output.sum().backward()
        assert states.grad is not None and torch.isfinite(states.grad).all()
        assert context.grad is not None and torch.isfinite(context.grad).all()

    return (
        test_local_self_window_keeps_global_encoder_keys,
        test_prefix_and_padding_receive_zero_loss_gradient,
        test_fully_masked_attention_has_finite_backward,
    )


@app.cell
def _(
    ENCODER_TOKENS,
    EXPERIMENT,
    PAIRS,
    RUN_MATRIX,
    SEED,
    TARGET_TOKENS,
    check_tiny_logic,
    device,
    dtype,
    experiment_spec,
    gc,
    mo,
    torch,
):
    # 9. RUN TINY TESTS — keenam konfigurasi berurutan agar memori tidak menumpuk.
    selected_experiments = [name for name, _, _ in PAIRS] if RUN_MATRIX else [EXPERIMENT]
    # Validate the selected experiment even when running the matrix.
    experiment_spec(EXPERIMENT)
    tiny_results = []
    for matrix_name in selected_experiments:
        matrix_report = check_tiny_logic(
            experiment_spec(matrix_name), device, dtype, ENCODER_TOKENS, TARGET_TOKENS, SEED
        )
        tiny_results.append(matrix_report)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    mo.vstack([mo.md("## 9 · Hasil tiny tests"), mo.ui.table(tiny_results, selection=None)])
    return (tiny_results,)


@app.cell
def _(
    ENCODER_TOKENS,
    EXPERIMENT,
    RUN_FULL_SIZE,
    SEED,
    TARGET_TOKENS,
    build_model,
    device,
    experiment_spec,
    full_parameter_counts,
    gc,
    mo,
    synthetic_batch,
    tiny_results,
    torch,
):
    # 10. OPTIONAL FULL-SIZE RANDOM FORWARD/BACKWARD — untuk GPU Molab 96 GB.
    full_result = {"status": "DISABLED", "pretrained_weights_loaded": False}
    if RUN_FULL_SIZE:
        assert all(row["status"] == "PASS" for row in tiny_results)
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Full-size probe membutuhkan CUDA dengan BF16.")
        gc.collect()
        torch.cuda.empty_cache()
        full_free_bytes, full_total_bytes = torch.cuda.mem_get_info()
        # BF16 weights + full gradients + conservative fixed workspace headroom.
        full_minimum_bytes = full_parameter_counts[EXPERIMENT] * 4 + 8 * 2**30
        if full_free_bytes < full_minimum_bytes:
            raise RuntimeError(
                f"Free VRAM {full_free_bytes / 2**30:.1f} GiB; probe ini membutuhkan perkiraan {full_minimum_bytes / 2**30:.1f} GiB."
            )
        torch.manual_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        full_model = build_model(experiment_spec(EXPERIMENT), "cuda", torch.bfloat16, tiny=False)
        full_batch = synthetic_batch(
            experiment_spec(EXPERIMENT), ENCODER_TOKENS, TARGET_TOKENS, "cuda"
        )
        full_output = full_model(full_batch)
        assert full_output.loss is not None
        assert torch.isfinite(full_output.logits).all() and torch.isfinite(full_output.loss)
        full_output.loss.backward()
        full_gradients = [
            full_parameter.grad
            for full_parameter in full_model.parameters()
            if full_parameter.grad is not None
        ]
        assert full_gradients and all(
            torch.isfinite(full_gradient).all() for full_gradient in full_gradients
        )
        full_result = {
            "status": "PASS",
            "model": EXPERIMENT,
            "mode": "full_dimensions_random_forward_backward",
            "precision": "bfloat16",
            "parameters": sum(full_parameter.numel() for full_parameter in full_model.parameters()),
            "loss": full_output.loss.detach().item(),
            "logits_shape": list(full_output.logits.shape),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "pretrained_weights_loaded": False,
            "optimizer_created": False,
        }
        del full_output, full_gradients, full_model, full_batch
        gc.collect()
        torch.cuda.empty_cache()
    mo.vstack(
        [mo.md("## 10 · Full-size probe opsional"), mo.ui.table([full_result], selection=None)]
    )
    return (full_result,)


@app.cell
def _(
    ENCODER_TOKENS,
    EXPERIMENT,
    OUTPUT_DIR,
    Path,
    RUN_MATRIX,
    SAVE_REPORT,
    SEED,
    TARGET_TOKENS,
    datetime,
    environment,
    full_result,
    json,
    meta_rows,
    mo,
    tiny_results,
    timezone,
):
    # 11. SUMMARY & REPORT — hanya hasil test, tidak ada upload.
    summary = {
        "environment": environment,
        "config": {
            "experiment": EXPERIMENT,
            "run_matrix": RUN_MATRIX,
            "seed": SEED,
            "encoder_tokens": ENCODER_TOKENS,
            "target_tokens": TARGET_TOKENS,
        },
        "meta": meta_rows,
        "tiny": tiny_results,
        "full_size": full_result,
        "training_run": False,
        "benchmark_run": False,
    }
    report_path = None
    if SAVE_REPORT:
        report_folder = Path(OUTPUT_DIR)
        report_folder.mkdir(parents=True, exist_ok=True)
        report_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_path = report_folder / f"architecture_{environment['device']}_{report_stamp}.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_check_count = sum(row["check_count"] for row in tiny_results)
    mo.md(f"""
    ## 11 · Ringkasan
    **{len(tiny_results)} model · {summary_check_count} checks PASS** · device `{environment["device"]}`.
    Full-size probe: **{full_result["status"]}**.
    Report: `{report_path or "tidak disimpan"}`.

    Hasil ini memeriksa wiring arsitektur pada bobot acak. Adaptasi checkpoint base/IT,
    training SFT, tokenizer/dataset, cache autoregresif dan benchmark repair adalah tahap berikutnya.
    """)
    return (
        summary,
        report_path,
    )


if __name__ == "__main__":
    app.run()
