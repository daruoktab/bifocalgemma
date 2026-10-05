"""D01: interfaces between construction blocks. See docs/DATA_PIPELINE.md."""

from dataclasses import dataclass
from pathlib import PurePosixPath

import torch


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    content: str

    def __post_init__(self):
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path or ":" in self.path:
            raise ValueError("Repository paths must be relative POSIX paths inside the snapshot.")


@dataclass(frozen=True)
class RepositorySnapshot:
    repo: str
    base_commit: str
    files: tuple[RepositoryFile, ...]


@dataclass(frozen=True)
class RepairExample:
    instance_id: str
    source: str
    split: str
    snapshot: RepositorySnapshot
    issue: str
    target_patch: str | None = None


@dataclass(frozen=True)
class TokenizedRepairExample:
    """Prefix is already IT-formatted; target contains the closing target token."""

    instance_id: str
    encoder_ids: tuple[int, ...]
    decoder_prefix_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


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


@dataclass(frozen=True)
class RepairPrediction:
    instance_id: str
    model_name: str
    patch: str


@dataclass(frozen=True)
class EvaluationReport:
    benchmark: str
    run_id: str
    resolved: int
    attempted: int
    infrastructure_failures: int
