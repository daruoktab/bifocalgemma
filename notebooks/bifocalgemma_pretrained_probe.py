# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "marimo==0.25.1",
#     "torch==2.14.1",
#     "transformers==5.19.0",
#     "safetensors==0.8.0",
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
# Standalone Molab pretrained + co-resident training memory probe.
# Pinned checkpoint downloads execute only after the GPU/headroom guard passes.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("""
    # BifocalGemma · Pretrained & GPU Memory Probe
    Checkpoint asli: encoder **base**, decoder **IT**, text-only, BF16.
    Pilot LoRA pada attention dengan dua optimizer steps dan satu contoh code-repair sintetis.
    Model dimuat independen. Concurrency probe menahan beberapa model dan forward graphs
    bersama, kemudian backward: pengukuran tekanan memori, bukan klaim CUDA kernels paralel.
    Cell configuration menentukan model dan lengths. Semua hasil tersimpan sebagai JSON lokal.
    """)
    return (mo,)


@app.cell
def _(mo):
    # 1. CONTROL CENTER
    DEVICE = "auto"
    PRECISION = "auto"
    RUN_PRETRAINED = True  # GPU guard harus lulus sebelum download/model allocation.
    RUN_CONCURRENCY = True
    SINGLE_EXPERIMENT = "BifocalGemma-E2B-E2B"
    CONCURRENT_EXPERIMENTS = (
        "BifocalGemma-E2B-E2B",
        "BifocalGemma-E2B-E4B",
        "BifocalGemma-E4B-E2B",
        "BifocalGemma-E4B-E4B",
    )
    TRAINING_POLICY = "lora"  # lora / full; actual optimizer dtype dilaporkan
    LORA_RANK = 16
    STEPS = 2
    BATCH_SIZE = 1
    ENCODER_CONTEXT_TOKENS = 512
    MAX_DECODER_TOKENS = 128
    HEADROOM_GIB = 8
    SEED = 17
    CACHE_DIR = "outputs/hf_cache"
    OUTPUT_DIR = "outputs/marimo_pretrained"
    # Token widget dan Apply/verify berada pada cell 2a/2b.
    mo.md(
        "## 1 · Control center\nDefault: LoRA BF16, 1 contoh per batch, 512 encoder tokens, 2 langkah."
    )
    return (
        DEVICE,
        PRECISION,
        RUN_PRETRAINED,
        RUN_CONCURRENCY,
        SINGLE_EXPERIMENT,
        CONCURRENT_EXPERIMENTS,
        TRAINING_POLICY,
        LORA_RANK,
        STEPS,
        BATCH_SIZE,
        ENCODER_CONTEXT_TOKENS,
        MAX_DECODER_TOKENS,
        HEADROOM_GIB,
        SEED,
        CACHE_DIR,
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
def _(mo):
    # 2a. HF TOKEN INPUT — submit form avoids rerunning models per keystroke.
    hf_token_form = mo.ui.dictionary(
        {
            "token": mo.ui.text(
                label="Hugging Face token (read access cukup)",
                kind="password",
                placeholder="hf_...",
                full_width=True,
            )
        }
    ).form(submit_button_label="Apply HF token")
    mo.vstack([mo.md("## 2a · Hugging Face API token"), hf_token_form])
    return (hf_token_form,)


@app.cell
def _(hf_token_form, mo, os):
    # 2b. APPLY / VERIFY — explicit token dependency for every Hub/tokenizer call.
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    _token_value = ((hf_token_form.value or {}).get("token", "") or "").strip()
    token_source = "widget" if _token_value else "environment"
    if not _token_value:
        _token_value = os.environ.get("HF_TOKEN", "").strip()
    hf_download_token = _token_value or False
    if _token_value:
        try:
            _identity = HfApi(token=_token_value).whoami()
        except HfHubHTTPError:
            raise RuntimeError(
                "HF token gagal diverifikasi; cek token/akses lalu Apply ulang."
            ) from None
        auth_info = {
            "mode": "authenticated",
            "source": token_source,
            "username": _identity.get("name", "unknown"),
            "verified": True,
        }
        auth_message = "HF token terverifikasi. Siap digunakan oleh loader dan tokenizer."
    else:
        auth_info = {"mode": "anonymous", "source": "none", "verified": False}
        auth_message = "Tanpa token: akses anonim untuk checkpoint publik. Isi token lalu klik Apply bila diperlukan."
    mo.md("## 2b · Apply / verifikasi HF token\n" + auth_message)
    return (hf_download_token, auth_info)


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
def _(
    ArchitectureModel,
    Gemma4TextModel,
    PAIRS,
    Path,
    RepairCollator,
    TokenizedRepairExample,
    build_model,
    copy,
    experiment_spec,
    gc,
    json,
    nn,
    hf_download_token,
    torch,
    F,
    mo,
):
    # 8. PRETRAINED LOADER & PROFILER — no automatic downloads in definitions.
    import hashlib
    import re
    import time

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    CHECKPOINTS = {
        "E2B": {
            "encoder": {
                "model_id": "google/gemma-4-E2B",
                "revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
                "files": {
                    "tokenizer.json": "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
                    "tokenizer_config.json": "12754e3442e47c1a1c55d550b617fc47ae227e8156ed1b1de0d8428195768b5c",
                    "config.json": "e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8",
                    "generation_config.json": "de16fa1d777387d5fe6df54a9565d9c0f1213f7b31f2a2cf44805e27686d9c7c",
                    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
                },
            },
            "decoder": {
                "model_id": "google/gemma-4-E2B-it",
                "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                "files": {
                    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
                    "tokenizer_config.json": "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
                    "chat_template.jinja": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
                    "config.json": "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
                    "generation_config.json": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
                    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
                },
            },
        },
        "E4B": {
            "encoder": {
                "model_id": "google/gemma-4-E4B",
                "revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
                "files": {
                    "tokenizer.json": "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
                    "tokenizer_config.json": "6a9383197f000d2723684efd2210f5bf217bc29fa2cf360ae1574201b47af060",
                    "config.json": "f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20",
                    "generation_config.json": "de16fa1d777387d5fe6df54a9565d9c0f1213f7b31f2a2cf44805e27686d9c7c",
                    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
                },
            },
            "decoder": {
                "model_id": "google/gemma-4-E4B-it",
                "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                "files": {
                    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
                    "tokenizer_config.json": "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
                    "chat_template.jinja": "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
                    "config.json": "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
                    "generation_config.json": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
                    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
                },
            },
        },
    }

    def transfer_text_checkpoint(backbone, path: Path, *, skip_words: bool = False) -> dict:
        prefix = "model.language_model."
        target = backbone.state_dict()
        expected = {
            key: value
            for key, value in target.items()
            if not (skip_words and key == "embed_tokens.weight")
        }
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            available = set(reader.keys())
            missing = [prefix + key for key in expected if prefix + key not in available]
            if missing:
                raise ValueError(f"Missing checkpoint tensors: {missing[:5]}")
            for key, value in expected.items():
                shape = reader.get_slice(prefix + key).get_shape()
                if tuple(shape) != tuple(value.shape):
                    raise ValueError(
                        f"Shape mismatch for {key}: source {shape}, target {tuple(value.shape)}"
                    )
            if any(value.is_meta for value in expected.values()):
                raise ValueError("Initialize destination storage before checkpoint transfer.")
            ignored_shared = []
            skipped = {prefix + "embed_tokens.weight"} if skip_words else set()
            for source_key in available:
                if not source_key.startswith(prefix) or source_key in skipped:
                    continue
                key = source_key[len(prefix) :]
                if key in expected:
                    continue
                match = re.fullmatch(
                    r"layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm|v_norm)\.weight", key
                )
                layers = getattr(backbone, "layers", ())
                if (
                    match
                    and int(match[1]) < len(layers)
                    and layers[int(match[1])].self_attn.is_kv_shared_layer
                ):
                    ignored_shared.append(key)
                else:
                    raise ValueError(f"Unmapped text tensor: {key}")
            with torch.no_grad():
                for key, destination in expected.items():
                    source = reader.get_tensor(prefix + key)
                    # Direct CPU-mmap -> destination copy avoids a second large GPU PLE tensor.
                    destination.copy_(source)
        return {
            "text_prefix": prefix,
            "tensors_copied": len(expected),
            "encoder_words_skipped": skip_words,
            "ignored_shared_kv_tensors": sorted(ignored_shared),
            "nontext_tensors_ignored": sum(not key.startswith(prefix) for key in available),
        }

    class ProbeLoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int = 16):
            super().__init__()
            if rank <= 0:
                raise ValueError("LoRA rank must be positive.")
            self.base = base
            self.base.requires_grad_(False)
            self.rank = rank
            self.scaling = 2.0  # alpha = 2 * rank
            self.lora_A = nn.Parameter(
                torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
            )
            nn.init.normal_(self.lora_A, std=0.02)

        def forward(self, values):
            delta = F.linear(F.linear(values.float(), self.lora_A), self.lora_B)
            return self.base(values) + delta.to(values.dtype) * self.scaling

    def configure_probe_training(model, *, policy: str = "lora", rank: int = 16) -> dict:
        if policy not in {"lora", "full"}:
            raise ValueError("Training policy must be lora or full.")
        model.requires_grad_(policy == "full")
        adapted = []
        if policy == "lora":
            for name, module in list(model.named_modules()):
                if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                }:
                    parent_name, attribute = name.rsplit(".", 1)
                    setattr(
                        model.get_submodule(parent_name), attribute, ProbeLoRALinear(module, rank)
                    )
                    adapted.append(name)
            # Asymmetric alignment is new, so it must remain trainable too.
            projection = getattr(model, "encoder_projection", None)
            if isinstance(projection, nn.Linear):
                projection.requires_grad_(True)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters were selected.")
        return {
            "policy": policy,
            "rank": rank if policy == "lora" else None,
            "adapter_modules": adapted,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "trainable_bytes": tensor_bytes(trainable),
        }

    def tensor_bytes(tensors) -> int:
        seen = set()
        total = 0
        for tensor in tensors:
            # Parameters are unique across the iterator; object identity also dedups optimizer aliases.
            if id(tensor) not in seen:
                seen.add(id(tensor))
                total += tensor.numel() * tensor.element_size()
        return total

    def optimizer_bytes(optimizers) -> int:
        return tensor_bytes(
            value
            for optimizer in optimizers
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )

    def gpu_memory(device) -> dict:
        if torch.device(device).type != "cuda":
            return dict(
                allocated_bytes=None,
                reserved_bytes=None,
                free_bytes=None,
                total_bytes=None,
                peak_allocated_bytes=None,
                peak_reserved_bytes=None,
            )
        free, total = torch.cuda.mem_get_info(device)
        return dict(
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device),
            free_bytes=free,
            total_bytes=total,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )

    def profile_training_group(
        models, batches, *, steps=2, mode="overlap_graphs", learning_rate=1e-4
    ):
        if not models or len(models) != len(batches) or steps < 1:
            raise ValueError("Supply one batch per model and at least one optimization step.")
        if mode not in {"round_robin", "overlap_graphs"}:
            raise ValueError("Mode must be round_robin or overlap_graphs.")
        parameter_groups = [[p for p in model.parameters() if p.requires_grad] for model in models]
        parameter_ids = [id(p) for group in parameter_groups for p in group]
        if not all(parameter_groups) or len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("Models need independent nonempty trainable parameter sets.")
        device = next(models[0].parameters()).device
        if any(p.device != device for model in models for p in model.parameters()):
            raise ValueError("All models must be resident on the same device.")
        optimizers = [
            torch.optim.AdamW(group, lr=learning_rate, foreach=False) for group in parameter_groups
        ]
        for model in models:
            model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        before = gpu_memory(device)
        started = time.perf_counter()
        losses = []
        snapshots = []
        success = False
        try:
            for step in range(steps):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                if mode == "overlap_graphs":
                    outputs = [model(batch) for model, batch in zip(models, batches, strict=True)]
                    graphs = [output.loss for output in outputs]
                    if any(loss is None for loss in graphs):
                        raise ValueError("Model did not supply a supervised loss.")
                    snapshots.append(
                        {"step": step, "stage": "all_forward_graphs", **gpu_memory(device)}
                    )
                    for loss in graphs:
                        assert loss is not None
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite loss.")
                        loss.backward()
                        losses.append(loss.detach().item())
                    del loss, graphs, outputs
                else:
                    for model, batch in zip(models, batches, strict=True):
                        output = model(batch)
                        if output.loss is None or not bool(torch.isfinite(output.loss)):
                            raise RuntimeError("Missing/nonfinite loss.")
                        output.loss.backward()
                        losses.append(output.loss.detach().item())
                        del output
                for group, optimizer in zip(parameter_groups, optimizers, strict=True):
                    gradients = [p.grad for p in group if p.grad is not None]
                    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                        raise RuntimeError("Missing/nonfinite gradients.")
                    if not any(bool(g.abs().sum() > 0) for g in gradients):
                        raise RuntimeError("All trainable gradients are zero.")
                    optimizer.step()
                    del gradients
                snapshots.append({"step": step, "stage": "optimizer_step", **gpu_memory(device)})
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            states = [
                value
                for optimizer in optimizers
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ]
            result = {
                "status": "PASS",
                "model_count": len(models),
                "mode": mode,
                "steps_per_model": steps,
                "optimizer": "torch.optim.AdamW(foreach=False)",
                "optimizer_tensor_dtypes": sorted({str(t.dtype) for t in states}),
                "optimizer_state_bytes": optimizer_bytes(optimizers),
                "optimizer_gpu_state_bytes": tensor_bytes(
                    t for t in states if t.device.type == "cuda"
                ),
                "parameter_dtypes": sorted(
                    {str(p.dtype) for model in models for p in model.parameters()}
                ),
                "parameter_bytes": tensor_bytes(p for model in models for p in model.parameters()),
                "trainable_bytes": tensor_bytes(p for group in parameter_groups for p in group),
                "seconds": time.perf_counter() - started,
                "losses": losses,
                "before": before,
                "after": gpu_memory(device),
                "snapshots": snapshots,
                "independent_weights": True,
                "concurrent_cuda_kernels_claimed": False,
            }
            success = True
            return result
        finally:
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
            if not success:
                # Caller owns models and may release them after OOM; no success is fabricated.
                for model in models:
                    model.zero_grad(set_to_none=True)

    def probe_estimate(spec, policy, rank, headroom_gib):
        with torch.device("meta"):
            blueprint = build_model(spec, "meta", torch.bfloat16, tiny=False)
        scope = configure_probe_training(blueprint, policy=policy, rank=rank)
        weights = tensor_bytes(blueprint.parameters())
        trainable = tensor_bytes(p for p in blueprint.parameters() if p.requires_grad)
        return {
            "weight_bytes": weights,
            "trainable_bytes": trainable,
            "estimated_gradient_and_adam_bytes": trainable * 3,
            "guard_bytes": weights + trainable * 3 + int(headroom_gib * 2**30),
            "trainable_parameters": scope["trainable_parameters"],
        }

    def gpu_guard(estimate, device):
        if device != "cuda":
            return "CUDA is required; no checkpoint downloaded."
        if not torch.cuda.is_bf16_supported():
            return "BF16 CUDA support is required; no checkpoint downloaded."
        free, total = torch.cuda.mem_get_info()
        if free < estimate["guard_bytes"]:
            return f"Free VRAM {free / 2**30:.2f} GiB < guard {estimate['guard_bytes'] / 2**30:.2f} GiB."
        return None

    def verified_assets(checkpoint, cache_dir):
        token = hf_download_token
        paths = {}
        for name, digest in checkpoint["files"].items():
            path = Path(
                hf_hub_download(
                    checkpoint["model_id"],
                    name,
                    revision=checkpoint["revision"],
                    cache_dir=cache_dir,
                    token=token,
                )
            )
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Pinned asset hash mismatch: {checkpoint['model_id']}/{name}")
            paths[name] = path
        return paths

    def load_pretrained_probe(spec, cache_dir, seed):
        torch.manual_seed(seed)
        decoder_ref = CHECKPOINTS[spec.decoder.size]["decoder"]
        encoder_ref = CHECKPOINTS[spec.encoder.size]["encoder"] if spec.encoder else None
        refs = [decoder_ref] + ([encoder_ref] if encoder_ref else [])
        source_files = []
        # Cache/config validation happens before destination GPU allocation.
        for ref in refs:
            assets = verified_assets(ref, cache_dir)
            print(f"Downloading/caching checkpoint {ref['model_id']} @ {ref['revision']} ...")
            weights = Path(
                hf_hub_download(
                    ref["model_id"],
                    "model.safetensors",
                    revision=ref["revision"],
                    cache_dir=cache_dir,
                    token=hf_download_token,
                )
            )
            source_files.append((ref, assets, weights))
        model = build_model(spec, "cuda", torch.bfloat16, tiny=False)
        mappings = []
        mappings.append(
            {
                "role": "decoder_it",
                "source": decoder_ref,
                **transfer_text_checkpoint(model.decoder, source_files[0][2]),
            }
        )
        if spec.encoder:
            assert model.encoder is not None
            mappings.append(
                {
                    "role": "encoder_base",
                    "source": encoder_ref,
                    **transfer_text_checkpoint(
                        model.encoder, source_files[1][2], skip_words=spec.tie_encoder_decoder_words
                    ),
                }
            )
        model._tie_words()
        return model, {
            "experiment": spec.name,
            "checkpoint_mapping": mappings,
            "encoder_words_anchor": "decoder_it"
            if spec.tie_encoder_decoder_words
            else "encoder_base",
            "PLE_shared": False,
            "new_alignment_projection": bool(spec.encoder and not spec.tie_encoder_decoder_words),
            "pretrained_weights_loaded": True,
            "checkpoint_format": "single_safetensors",
        }

    def repair_probe_batch(spec, cache_dir, encoder_tokens, decoder_limit, batch_size):
        if (
            not 64 <= encoder_tokens <= 8192
            or not 64 <= decoder_limit <= 4096
            or not 1 <= batch_size <= 8
        ):
            raise ValueError("Probe limits: encoder 64..8192, decoder 64..4096, batch 1..8.")
        decoder_ref = CHECKPOINTS[spec.decoder.size]["decoder"]
        decoder = AutoTokenizer.from_pretrained(
            decoder_ref["model_id"],
            revision=decoder_ref["revision"],
            cache_dir=cache_dir,
            token=hf_download_token,
            trust_remote_code=False,
        )
        assert decoder is not None
        issue = "Fix add so that it returns the sum of its arguments."
        context = "### Issue\n" + issue + "\n### File: maths.py\ndef add(a, b):\n    return a - b\n"
        # Explicit synthetic context padding for a reproducible memory stress length.
        context += "# context padding, unrelated comment\n" * encoder_tokens
        encoder_ids = ()
        encoder_policy = "no_encoder"
        if spec.encoder:
            ref = CHECKPOINTS[spec.encoder.size]["encoder"]
            encoder = AutoTokenizer.from_pretrained(
                ref["model_id"],
                revision=ref["revision"],
                cache_dir=cache_dir,
                token=hf_download_token,
                trust_remote_code=False,
            )
            assert encoder is not None
            encoder_ids = tuple(encoder.encode(context, add_special_tokens=True)[:encoder_tokens])
            assert encoder_ids[0] == 2 and len(encoder_ids) == encoder_tokens
            prompt = (
                f"Return only a unified diff fixing this issue using the encoder context.\n{issue}"
            )
            encoder_policy = "synthetic_comment_padding_then_explicit_token_crop"
        else:
            # Baseline receives equivalent selected context, rather than the entire padding tail.
            encoded = decoder.encode(context, add_special_tokens=False)[:encoder_tokens]
            decoded_context = decoder.decode(encoded)
            assert isinstance(decoded_context, str)
            prompt = "Return only a unified diff fixing the issue.\n" + decoded_context
        patch = "diff --git a/maths.py b/maths.py\n--- a/maths.py\n+++ b/maths.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
        messages = [{"role": "user", "content": prompt}]
        prefix = decoder.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        complete = decoder.apply_chat_template(
            messages + [{"role": "assistant", "content": patch}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        assert isinstance(prefix, str) and isinstance(complete, str)
        prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
        full_ids = tuple(decoder.encode(complete.rstrip(), add_special_tokens=False))
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Tokenizer crossed prefix/target boundary.")
        target = full_ids[len(prefix_ids) :]
        if not target or target[-1] != 106:
            raise ValueError("Missing canonical IT closing turn.")
        if len(full_ids) - 1 > decoder_limit:
            raise ValueError("Decoder budget exceeded; increase MAX_DECODER_TOKENS explicitly.")
        examples = [
            TokenizedRepairExample(f"synthetic-repair-{i}", encoder_ids, prefix_ids, target)
            for i in range(batch_size)
        ]
        batch = RepairCollator()(examples).to("cuda")
        return batch, {
            "synthetic": True,
            "batch_size": batch_size,
            "encoder_tokens_actual": len(encoder_ids),
            "decoder_tokens_actual": len(full_ids) - 1,
            "prefix_tokens": len(prefix_ids),
            "target_tokens": len(target),
            "context_policy": encoder_policy,
            "example": "maths.py arithmetic repair",
            "real_dataset_training": False,
        }

    def run_single_pretrained(
        name,
        policy,
        rank,
        steps,
        cache_dir,
        encoder_tokens,
        decoder_limit,
        batch_size,
        headroom,
        seed,
        device,
    ):
        spec = experiment_spec(name)
        estimate = probe_estimate(spec, policy, rank, headroom)
        reason = gpu_guard(estimate, device)
        if reason:
            return {
                "status": "SKIPPED_HEADROOM",
                "reason": reason,
                "estimate": estimate,
                "pretrained_weights_loaded": False,
            }
        model = None
        stage = "load_checkpoint"
        try:
            model, provenance = load_pretrained_probe(spec, cache_dir, seed)
            scope = configure_probe_training(model, policy=policy, rank=rank)
            stage = "optimizer_probe"
            batch, data = repair_probe_batch(
                spec, cache_dir, encoder_tokens, decoder_limit, batch_size
            )
            loaded = gpu_memory("cuda")
            result = profile_training_group([model], [batch], steps=steps, mode="round_robin")
            return {
                "status": "PASS",
                "experiment": name,
                "scope": scope,
                "data": data,
                "provenance": provenance,
                "loaded_memory": loaded,
                "training": result,
            }
        except torch.cuda.OutOfMemoryError:
            return {
                "status": "OOM",
                "stage": stage,
                "experiment": name,
                "estimate": estimate,
                "memory": gpu_memory(device),
                "pretrained_weights_loaded": model is not None,
            }
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def run_resident_group(
        names,
        policy,
        rank,
        steps,
        cache_dir,
        encoder_tokens,
        decoder_limit,
        batch_size,
        headroom,
        seed,
        device,
    ):
        models, batches, rows, provenance = [], [], [], []
        attempted_count = 0
        phase = "initialization"
        try:
            for name in names:
                attempted_count = len(models) + 1
                phase = "load_checkpoint"
                spec = experiment_spec(name)
                estimate = probe_estimate(spec, policy, rank, headroom)
                # Previous optimizers have been released; account for their states/gradients
                # when they will all be recreated for the next cumulative training probe.
                resident_trainable = sum(
                    tensor_bytes(p for p in m.parameters() if p.requires_grad) for m in models
                )
                estimate["guard_bytes"] += 3 * resident_trainable
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                reason = gpu_guard(estimate, device)
                if reason:
                    rows.append(
                        {
                            "status": "SKIPPED_HEADROOM",
                            "model_count_requested": len(models) + 1,
                            "next_model": name,
                            "reason": reason,
                            "estimate": estimate,
                        }
                    )
                    break
                model, mapping = load_pretrained_probe(spec, cache_dir, seed)
                configure_probe_training(model, policy=policy, rank=rank)
                batch, data = repair_probe_batch(
                    spec, cache_dir, encoder_tokens, decoder_limit, batch_size
                )
                models.append(model)
                batches.append(batch)
                provenance.append(mapping)
                loaded = gpu_memory("cuda")
                # Every model has independent trainable and frozen parameters across experiments.
                ids = [id(p) for m in models for p in m.parameters()]
                if len(ids) != len(set(ids)):
                    raise ValueError(
                        "Models must not share checkpoint parameter objects across experiments."
                    )
                for mode in ("round_robin", "overlap_graphs"):
                    phase = mode
                    result = profile_training_group(models, batches, steps=steps, mode=mode)
                    rows.append(
                        {
                            "status": "PASS",
                            "experiments": list(names[: len(models)]),
                            "scope_policy": policy,
                            "last_batch": data,
                            "loaded_memory": loaded,
                            "training": result,
                        }
                    )
            return {
                "rows": rows,
                "provenance": provenance,
                "models_loaded": len(models),
                "max_models_passed": max(
                    (row["training"]["model_count"] for row in rows if row["status"] == "PASS"),
                    default=0,
                ),
                "independent_weights": True,
            }
        except torch.cuda.OutOfMemoryError:
            rows.append(
                {
                    "status": "OOM",
                    "model_count_attempted": attempted_count,
                    "phase": phase,
                    "memory": gpu_memory(device),
                }
            )
            return {
                "rows": rows,
                "provenance": provenance,
                "models_loaded": len(models),
                "max_models_passed": max(
                    (row["training"]["model_count"] for row in rows if row["status"] == "PASS"),
                    default=0,
                ),
                "independent_weights": True,
            }
        finally:
            models.clear()
            batches.clear()
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    mo.md(
        "## 8 · Loader dan profiler\nStrict tensor mapping, base/IT provenance, attention LoRA, AdamW dan allocator snapshots."
    )
    return (
        CHECKPOINTS,
        ProbeLoRALinear,
        configure_probe_training,
        transfer_text_checkpoint,
        tensor_bytes,
        optimizer_bytes,
        gpu_memory,
        profile_training_group,
        probe_estimate,
        gpu_guard,
        run_single_pretrained,
        run_resident_group,
    )


@app.cell
def _(
    BATCH_SIZE,
    CACHE_DIR,
    ENCODER_CONTEXT_TOKENS,
    HEADROOM_GIB,
    LORA_RANK,
    MAX_DECODER_TOKENS,
    RUN_PRETRAINED,
    SEED,
    SINGLE_EXPERIMENT,
    STEPS,
    TRAINING_POLICY,
    device,
    mo,
    run_single_pretrained,
):
    # 9. SINGLE PRETRAINED PILOT
    single_result = {"status": "DISABLED", "pretrained_weights_loaded": False}
    if RUN_PRETRAINED:
        single_result = run_single_pretrained(
            SINGLE_EXPERIMENT,
            TRAINING_POLICY,
            LORA_RANK,
            STEPS,
            CACHE_DIR,
            ENCODER_CONTEXT_TOKENS,
            MAX_DECODER_TOKENS,
            BATCH_SIZE,
            HEADROOM_GIB,
            SEED,
            device,
        )
    mo.vstack([mo.md("## 9 · Pilot model asli"), mo.ui.table([single_result], selection=None)])
    return (single_result,)


@app.cell
def _(
    BATCH_SIZE,
    CACHE_DIR,
    CONCURRENT_EXPERIMENTS,
    ENCODER_CONTEXT_TOKENS,
    HEADROOM_GIB,
    LORA_RANK,
    MAX_DECODER_TOKENS,
    RUN_CONCURRENCY,
    RUN_PRETRAINED,
    SEED,
    STEPS,
    TRAINING_POLICY,
    device,
    mo,
    run_resident_group,
    single_result,
):
    # 10. CUMULATIVE GPU RESIDENCY — 1, 2, 3, 4 independent models.
    concurrency_result: dict = {"rows": [], "status": "DISABLED"}
    if RUN_PRETRAINED and RUN_CONCURRENCY and single_result["status"] == "PASS":
        concurrency_result = run_resident_group(
            CONCURRENT_EXPERIMENTS,
            TRAINING_POLICY,
            LORA_RANK,
            STEPS,
            CACHE_DIR,
            ENCODER_CONTEXT_TOKENS,
            MAX_DECODER_TOKENS,
            BATCH_SIZE,
            HEADROOM_GIB,
            SEED,
            device,
        )
    concurrency_table = []
    for row in concurrency_result["rows"]:
        training = row.get("training", {})
        memory = training.get("after", {})
        concurrency_table.append(
            {
                "status": row["status"],
                "models": training.get(
                    "model_count",
                    row.get("model_count_requested", row.get("model_count_attempted")),
                ),
                "mode": training.get("mode"),
                "experiments": row.get("experiments", [row.get("next_model")]),
                "peak_allocated_gib": (memory["peak_allocated_bytes"] / 2**30)
                if memory.get("peak_allocated_bytes") is not None
                else None,
                "peak_reserved_gib": (memory["peak_reserved_bytes"] / 2**30)
                if memory.get("peak_reserved_bytes") is not None
                else None,
                "optimizer_gib": training.get("optimizer_state_bytes", 0) / 2**30,
                "seconds": training.get("seconds"),
                "reason": row.get("reason"),
            }
        )
    mo.vstack(
        [
            mo.md("## 10 · Beberapa model resident / overlap graphs"),
            mo.ui.table(concurrency_table, selection=None),
        ]
    )
    return (
        concurrency_result,
        concurrency_table,
    )


@app.cell
def _(
    auth_info,
    BATCH_SIZE,
    CONCURRENT_EXPERIMENTS,
    ENCODER_CONTEXT_TOKENS,
    LORA_RANK,
    MAX_DECODER_TOKENS,
    OUTPUT_DIR,
    Path,
    SEED,
    SINGLE_EXPERIMENT,
    STEPS,
    TRAINING_POLICY,
    concurrency_result,
    concurrency_table,
    datetime,
    environment,
    json,
    mo,
    single_result,
    timezone,
):
    # 11. SAVE REPORT
    report = {
        "environment": environment,
        "authentication": auth_info,
        "config": {
            "single_experiment": SINGLE_EXPERIMENT,
            "concurrent_experiments": list(CONCURRENT_EXPERIMENTS),
            "training_policy": TRAINING_POLICY,
            "checkpoint_precision": "bfloat16",
            "lora_rank": LORA_RANK,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "encoder_context_tokens": ENCODER_CONTEXT_TOKENS,
            "max_decoder_tokens": MAX_DECODER_TOKENS,
            "seed": SEED,
        },
        "single": single_result,
        "concurrency": concurrency_result,
        "throughput_improvement_claimed": False,
        "dataset_benchmark_run": False,
    }
    folder = Path(OUTPUT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = folder / f"pretrained_memory_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mo.md(f"""
    ## 11 · Report
    Pilot: **{single_result["status"]}**. Report: `{report_path}`.
    PASS menunjukkan optimizer steps dan VRAM yang benar-benar diukur di runtime ini.
    SKIPPED_HEADROOM menunjukkan guard, bukan pengukuran model yang berhasil dimuat.
    Probe memakai contoh repair sintetis dan tidak mengukur kualitas model/benchmark.
    Round-robin dan overlap graphs berbagi GPU residency; CUDA kernel speedup tidak diasumsikan.
    """)
    return (
        report,
        report_path,
    )


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


if __name__ == "__main__":
    app.run()
