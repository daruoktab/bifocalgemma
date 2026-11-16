"""Canonical six-model registry, ten-model matrix registry, and benchmark constants."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

ModelSize = Literal["E2B", "E4B"]


@dataclass(frozen=True)
class BackboneProfile:
    size: ModelSize
    hidden_size: int
    layers: int
    kv_heads: int
    kv_shared_layers: int
    base_revision: str
    it_revision: str
    query_heads: int = 8
    local_head_dim: int = 256
    global_head_dim: int = 512
    sliding_window: int = 512
    global_layers: int = 7
    context_limit: int = 131072
    vocabulary_size: int = 262144
    ple_width: int = 256
    double_wide_mlp: bool = False


PROFILES: dict[ModelSize, BackboneProfile] = {
    "E2B": BackboneProfile(
        "E2B",
        1536,
        35,
        1,
        20,
        "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        double_wide_mlp=True,
    ),
    "E4B": BackboneProfile(
        "E4B",
        2560,
        42,
        2,
        18,
        "411aa17b749aa952df1359d2dcea73917a544d9a",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
    ),
}


@dataclass(frozen=True)
class CheckpointRef:
    model_id: str
    revision: str
    size: ModelSize
    role: Literal["encoder_base", "decoder_it"]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    encoder: CheckpointRef | None
    decoder: CheckpointRef
    attention_mode: Literal["merged", "causal"]
    tie_encoder_decoder_words: bool
    training_recipe: Literal["direct_sft"] = "direct_sft"
    distillation: bool = False
    share_ple: bool = False
    architecture_version: str = "merged_v1"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.architecture_version == "merged_v1":
            result.pop("architecture_version")
        return result


def _ref(size: ModelSize, *, encoder: bool) -> CheckpointRef:
    profile = PROFILES[size]
    return CheckpointRef(
        f"google/gemma-4-{size}" + ("" if encoder else "-it"),
        profile.base_revision if encoder else profile.it_revision,
        size,
        "encoder_base" if encoder else "decoder_it",
    )


def _pair(encoder: ModelSize, decoder: ModelSize) -> ExperimentSpec:
    return ExperimentSpec(
        f"BifocalGemma-{encoder}-{decoder}",
        _ref(encoder, encoder=True),
        _ref(decoder, encoder=False),
        "merged",
        encoder == decoder,
    )


EXPERIMENT_SPECS = (
    _pair("E2B", "E2B"),
    _pair("E4B", "E4B"),
    _pair("E2B", "E4B"),
    _pair("E4B", "E2B"),
    ExperimentSpec("Gemma-4-E2B-IT", None, _ref("E2B", encoder=False), "causal", False),
    ExperimentSpec("Gemma-4-E4B-IT", None, _ref("E4B", encoder=False), "causal", False),
)
EXPERIMENT_NAMES = tuple(spec.name for spec in EXPERIMENT_SPECS)


def get_experiment(name: str) -> ExperimentSpec:
    for spec in EXPERIMENT_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown experiment {name!r}. Choose from {EXPERIMENT_NAMES}.")


def get_v2_experiment(name: str) -> ExperimentSpec:
    """Fresh base encoder / IT decoder, with independent encoder word tables."""
    return replace(
        get_experiment(name),
        tie_encoder_decoder_words=False,
        architecture_version="normalized_merged_v2",
    )


def validate_experiment_matrix(path: str | Path) -> tuple[ExperimentSpec, ...]:
    """Check the checked-in JSON against the registry; never fetch checkpoints."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "training_recipe": "direct_sft",
        "distillation": False,
        "experiments": [spec.to_dict() for spec in EXPERIMENT_SPECS],
    }
    if payload != expected:
        raise ValueError("Experiment matrix differs from the canonical direct-SFT registry.")
    return EXPERIMENT_SPECS


# ---------------------------------------------------------------------------
# Ten-Model Matrix Constants, Step Budgets, and Benchmark Suite
# ---------------------------------------------------------------------------

CENTRALIZED_HF_REPO = "daruokta/bifocalgemma-2026-11-02"
CLEAN_RECIPE_VERSION = "bifocalgemma_shared_schedule_2026_11_12"
CLEAN_DEFAULT_HF_REPO = "daruokta/bifocalgemma-clean"
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

# Legacy curriculum helpers only; the main matrix performs no replacements.
COMPREHENSION_MIX_RATIO = 0.20
WARMUP_RATIO = 0.05
BENCHMARK_SAMPLE_SEED = 17
HEAVY_BENCHMARK_SAMPLE_LIMIT = 100
GATE_INIT_TANH = 0.05
GATE_INIT_ATANH = math.atanh(GATE_INIT_TANH)

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
        "repo": "daruokta/bifocalgemma-2base-2it",
        "subfolder": "latest",
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
        "repo": "daruokta/bifocalgemma-2it-2it",
        "subfolder": "latest",
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
        "repo": "daruokta/bifocalgemma-4base-4it",
        "subfolder": "latest",
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
        "repo": "daruokta/bifocalgemma-4it-4it",
        "subfolder": "latest",
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
        "repo": "daruokta/bifocalgemma-4base-2it",
        "subfolder": "latest",
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
        "repo": "daruokta/bifocalgemma-4it-2it",
        "subfolder": "latest",
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
        "repo": "daruokta/bifocalgemma-2base-4it",
        "subfolder": "latest",
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
        "repo": "daruokta/bifocalgemma-2it-4it",
        "subfolder": "latest",
    },
    "e2b-it": {
        "title": "BifocalGemma · Baseline E2B-IT (Decoder-Only)",
        "family": "decoder_only",
        "encoder_id": None,
        "encoder_revision": None,
        "encoder_size": None,
        "encoder_role": None,
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_size": "E2B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-e2b-it",
        "subfolder": "latest",
    },
    "e4b-it": {
        "title": "BifocalGemma · Baseline E4B-IT (Decoder-Only)",
        "family": "decoder_only",
        "encoder_id": None,
        "encoder_revision": None,
        "encoder_size": None,
        "encoder_role": None,
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_size": "E4B",
        "decoder_role": "decoder_it",
        "repo": "daruokta/bifocalgemma-e4b-it",
        "subfolder": "latest",
    },
}


def make_experiment_spec(model_key: str) -> ExperimentSpec:
    """Construct the canonical ExperimentSpec for one of the ten matrix models."""
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


# Reordered 8-benchmark suite: 6 faster benchmarks in full first, followed by 2 heavy benchmarks sampled at 100 cases.
BENCHMARKS: list[dict[str, Any]] = [
    {
        "collection": "longcodeqa_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 256,
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
        "max_new_tokens": 256,
        "sample_limit": None,
        "weight_tier": "medium_long",
    },
    {
        "collection": "sweqa_noisy_oracle_gemma32k_dec8k_v1",
        "encoder_limit": 32768,
        "decoder_limit": 8192,
        "max_new_tokens": 256,
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
