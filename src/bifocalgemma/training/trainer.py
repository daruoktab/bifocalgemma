"""T02: direct SFT driver placeholder. See docs/TRAINING.md."""

from dataclasses import dataclass
from pathlib import Path

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.scaffold import FeatureNotReadyError


@dataclass(frozen=True)
class SFTConfig:
    learning_rate: float | None = None
    micro_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    max_steps: int | None = None
    precision: str | None = None
    peft_strategy: str | None = None
    teacher_required: bool = False


def train_direct_sft(
    spec: ExperimentSpec, data_manifest: Path, config: SFTConfig, output_dir: Path
) -> None:
    """Future: profile -> select trainable parameters -> CE -> checkpoint.

    D03/D04 and M04/M05 must be ready before optimizer/data loading. The default
    path never launches a teacher. Null hyperparameters require measured profiling.
    """
    raise FeatureNotReadyError(
        "T02", "Wire data, native model, PEFT, optimizer and checkpoint resume."
    )
