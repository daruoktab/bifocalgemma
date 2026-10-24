"""C01: canonical six-model registry. See docs/ARCHITECTURE.md and BUILD_GUIDE.md."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

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

    def to_dict(self) -> dict:
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
