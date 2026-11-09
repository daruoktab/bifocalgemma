"""Pinned metadata only; no pretrained weights or tokenizer payloads.

Includes in-memory fallback definitions and automatic disk self-healing so that
runtime extraction or partial container restarts in Molab never raise
FileNotFoundError when reading asset JSONs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

EMBEDDED_ASSETS: dict[str, dict[str, Any]] = {
    "gemma4_e2b.json": {
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
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
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
    "gemma4_e4b.json": {
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
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
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
    "tokenizer_manifest.json": {
        "google/gemma-4-E2B": {
            "revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
            "files": {
                "tokenizer.json": "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
                "tokenizer_config.json": "12754e3442e47c1a1c55d550b617fc47ae227e8156ed1b1de0d8428195768b5c",
                "config.json": "e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8",
                "generation_config.json": "de16fa1d777387d5fe6df54a9565d9c0f1213f7b31f2a2cf44805e27686d9c7c",
                "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
            },
        },
        "google/gemma-4-E4B": {
            "revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
            "files": {
                "tokenizer.json": "12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6",
                "tokenizer_config.json": "6a9383197f000d2723684efd2210f5bf217bc29fa2cf360ae1574201b47af060",
                "config.json": "f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20",
                "generation_config.json": "de16fa1d777387d5fe6df54a9565d9c0f1213f7b31f2a2cf44805e27686d9c7c",
                "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
            },
        },
        "google/gemma-4-E2B-it": {
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
        "google/gemma-4-E4B-it": {
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


def ensure_assets_on_disk() -> None:
    """Ensure asset JSON files exist on disk in the package directory if deleted or missing."""
    try:
        assets_dir = Path(__file__).resolve().parent
        assets_dir.mkdir(parents=True, exist_ok=True)
        for filename, payload in EMBEDDED_ASSETS.items():
            target = assets_dir / filename
            if not target.is_file():
                target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def load_asset_json(filename: str) -> dict[str, Any]:
    """Load an asset JSON by filename with automatic disk self-healing and in-memory fallback."""
    ensure_assets_on_disk()
    try:
        from importlib.resources import files

        asset = files("bifocalgemma.assets").joinpath(filename)
        return json.loads(asset.read_text(encoding="utf-8"))
    except Exception:
        if filename in EMBEDDED_ASSETS:
            return copy.deepcopy(EMBEDDED_ASSETS[filename])
        raise


ensure_assets_on_disk()
