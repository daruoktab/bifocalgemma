"""V3 matrix constants: repositories, step budget, model matrix, benchmarks."""

from __future__ import annotations

import math
from typing import Any

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
