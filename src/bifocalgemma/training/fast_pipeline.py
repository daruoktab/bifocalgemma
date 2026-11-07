"""Fast, focused from-scratch training pipeline for BifocalGemma-2B and Baseline-2B.

Eliminates slow multi-model matrix iteration, disables copy-biasing pointer loss
(lambda_ptr=0.0), introduces learnable cross-attention sharpness scaling, and automatically
exports an intact, standalone Hugging Face PreTrainedModel at training completion.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
from bifocalgemma.contracts import Seq2SeqBatch, TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.modeling_bifocalgemma import BifocalGemmaForCausalLM
from bifocalgemma.training.memory_probe import configure_probe_training
from bifocalgemma.v3.objectives import (
    build_bifocalgemma_v3_optimizer,
    compute_bifocalgemma_v3_loss,
    is_cross_bridge_parameter,
)

logger = logging.getLogger(__name__)


@dataclass
class FastTrainingConfig:
    """Hyperparameters for the fast, two-stage from-scratch training pipeline."""

    model_choice: str = "bifocalgemma-2b"  # "bifocalgemma-2b" or "baseline-2b"
    total_steps: int = 2000
    stage1_steps: int = 1000
    stage2_steps: int = 1000
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    lr_bridge: float = 1e-4
    lr_cross: float = 1e-4
    lr_lora: float = 2e-5  # Conservative Stage 2 decoder LR to avoid catastrophic interference
    lambda_ptr: float = 0.0  # Clean cross-entropy; eliminates copy-bias on code repair
    warmup_steps: int = 100
    weight_decay_bridge: float = 1e-4
    eval_interval: int = 200
    save_interval: int = 500
    output_dir: str = "outputs/bifocalgemma_2b"
    data_dir: str | None = None  # Path to real repair dataset (jsonl or directory)
    pretrained_backbone_dir: str | None = None  # Local directory containing backbone safetensors
    decoder_backbone: str = "google/gemma-4-E2B-it"
    encoder_backbone: str = "google/gemma-4-E2B"
    tokenizer_path: str = "outputs/context_tokenizers/raw/google__gemma-4-E2B-it"
    push_to_hub: bool = False
    hub_repo: str | None = None
    hf_token: str | None = None
    auto_download_curated: bool = True
    collection: str = "mixed_context_enc32k_dec8k_v3"
    dataset_repo: str = "daruokta/bifocalgemma-data"
    dataset_revision: str = "7972f95d7c87f163c8b5af6a4acd308c0c1d0e23"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    tiny: bool = False  # Set True for instant CI/smoke validation

    def __post_init__(self) -> None:
        if self.model_choice == "baseline-2b":
            self.stage1_steps = 0
            self.stage2_steps = self.total_steps
        else:
            if self.stage1_steps + self.stage2_steps != self.total_steps:
                if self.stage1_steps == 1000 and self.stage2_steps == 1000:
                    self.stage1_steps = self.total_steps // 2
                    self.stage2_steps = self.total_steps - self.stage1_steps
                else:
                    self.total_steps = self.stage1_steps + self.stage2_steps
            if self.stage1_steps + self.stage2_steps != self.total_steps:
                self.stage2_steps = self.total_steps - self.stage1_steps


class JSONLRepairDataset(Dataset):
    """Loads and tokenizes real code repair examples from a JSONL file or directory."""

    def __init__(
        self,
        file_or_dir: str | Path,
        tokenizer: Any = None,
        is_baseline: bool = False,
        split: str | None = None,
        max_samples: int | None = None,
        max_vocab_size: int | None = None,
    ) -> None:
        self.examples: list[TokenizedRepairExample] = []
        self.tokenizer = tokenizer
        self.max_vocab_size = max_vocab_size
        path = Path(file_or_dir)
        files: list[Path] = []
        if path.is_file():
            files = [path]
        elif path.is_dir():
            if split:
                split_files = [path / f"{split}.jsonl", path / f"{split}.json"]
                existing = [f for f in split_files if f.is_file()]
                if existing:
                    files = existing
            if not files:
                files = sorted(path.glob("*.jsonl"))
                if not files:
                    files = sorted(path.glob("*.json"))

        raw_rows: list[dict[str, Any]] = []
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        raw_rows.append(json.loads(line))
                    except Exception:
                        continue
                if max_samples and len(raw_rows) >= max_samples:
                    break
            if max_samples and len(raw_rows) >= max_samples:
                break

        for idx, row in enumerate(raw_rows):
            example = self._row_to_example(row, idx, tokenizer, is_baseline)
            if example is not None:
                self.examples.append(example)

    def _row_to_example(
        self, row: dict[str, Any], idx: int, tokenizer: Any, is_baseline: bool
    ) -> TokenizedRepairExample | None:
        # Pretokenized format
        if "decoder_prefix_ids" in row and "target_ids" in row:
            enc_ids = () if is_baseline else tuple(row.get("encoder_ids", ()))
            prefix_ids = tuple(row["decoder_prefix_ids"])
            target_ids = tuple(row["target_ids"])
            if self.max_vocab_size is not None:
                limit = max(1, self.max_vocab_size - 10)
                enc_ids = tuple((t % limit) + 1 for t in enc_ids)
                prefix_ids = tuple((t % limit) + 1 for t in prefix_ids)
                target_ids = tuple((t % limit) + 1 for t in target_ids)
            return TokenizedRepairExample(
                instance_id=str(row.get("instance_id", f"example-{idx}")),
                encoder_ids=enc_ids,
                decoder_prefix_ids=prefix_ids,
                target_ids=target_ids,
            )

        # Raw text format
        instance_id = str(row.get("instance_id", f"example-{idx}"))
        context = row.get("context") or (
            f"### Instruction\n{row.get('instruction', '')}\n\n### File: {row.get('path', '')}\n{row.get('before', '')}"
        )
        target_patch = (
            row.get("target_text")
            or row.get("target_patch")
            or row.get("after")
            or row.get("patch")
            or ""
        )
        task_kind = str(row.get("task_type", "repair")).lower()
        task_instruction_header = {
            "repair": "Return only a unified diff fulfilling the instruction.",
            "localization": "Return only a JSON list of repository file paths relevant to fixing the issue.",
            "completion": "Return only the missing Python code, preserving its indentation.",
        }.get(task_kind, "Return only a unified diff fulfilling the instruction.")
        instruction_body = row.get("instruction") or "Repair the bug."

        if tokenizer is not None:
            if is_baseline:
                enc_ids = ()
                prompt = f"{task_instruction_header}\n\n{context}"
            else:
                enc_ids = tuple(tokenizer.encode(context, add_special_tokens=True))
                prompt = f"{task_instruction_header}\n\n{instruction_body}"

            messages = [{"role": "user", "content": prompt}]
            try:
                prefix = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                full = tokenizer.apply_chat_template(
                    messages + [{"role": "assistant", "content": target_patch}],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                prefix_ids = tuple(tokenizer.encode(prefix, add_special_tokens=False))
                full_ids = tuple(tokenizer.encode(full.rstrip(), add_special_tokens=False))
                target = full_ids[len(prefix_ids) :] if len(full_ids) >= len(prefix_ids) else (106,)
            except Exception:
                prefix_ids = tuple(
                    tokenizer.encode(f"<user>\n{prompt}\n<assistant>\n", add_special_tokens=True)
                )
                target = tuple(tokenizer.encode(target_patch, add_special_tokens=False)) + (106,)

            if self.max_vocab_size is not None:
                limit = max(1, self.max_vocab_size - 10)
                enc_ids = tuple((t % limit) + 1 for t in enc_ids)
                prefix_ids = tuple((t % limit) + 1 for t in prefix_ids)
                target = tuple((t % limit) + 1 for t in target)

            return TokenizedRepairExample(
                instance_id=instance_id,
                encoder_ids=enc_ids,
                decoder_prefix_ids=prefix_ids or (2,),
                target_ids=target or (106,),
            )
        else:
            limit = max(1, self.max_vocab_size - 20) if self.max_vocab_size else 100
            enc_ids = () if is_baseline else tuple((idx * 7 + i) % limit + 10 for i in range(16))
            prefix_ids = (2, (idx * 3) % limit + 10, (idx * 5) % limit + 10)
            target = (
                (idx * 11) % limit + 10,
                min(106, (self.max_vocab_size - 1) if self.max_vocab_size else 106),
            )
            return TokenizedRepairExample(
                instance_id=instance_id,
                encoder_ids=enc_ids,
                decoder_prefix_ids=prefix_ids,
                target_ids=target,
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedRepairExample:
        return self.examples[idx]


class SyntheticRepairDataset(Dataset):
    """Fallback synthetic dataset for fast verification and unit tests."""

    def __init__(
        self,
        num_samples: int = 128,
        enc_len: int = 64,
        dec_len: int = 32,
        vocab_size: int = 128,
    ) -> None:
        self.num_samples = num_samples
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(idx)
        enc_ids = torch.randint(10, self.vocab_size, (self.enc_len,))
        dec_ids = torch.randint(10, self.vocab_size, (max(2, self.dec_len),))
        inputs = dec_ids[:-1]
        targets = dec_ids[1:].clone()
        prefix_len = inputs.shape[0] // 2
        targets[:prefix_len] = -100
        return {
            "encoder_input_ids": enc_ids,
            "encoder_attention_mask": torch.ones_like(enc_ids),
            "decoder_input_ids": inputs,
            "decoder_attention_mask": torch.ones_like(inputs),
            "labels": targets,
        }


def collate_fast_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


class FastTrainingPipeline:
    """Orchestrates Stage 1 (Bridge Alignment) and Stage 2 (Joint SFT) training."""

    def __init__(
        self,
        config: FastTrainingConfig,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.cfg = config
        self.callback = callback
        torch.manual_seed(self.cfg.seed)

        # 1. Instantiate BifocalGemma PreTrainedModel
        is_baseline = self.cfg.model_choice == "baseline-2b"
        self.model_config = BifocalGemmaConfig(
            encoder_size="none" if is_baseline else "E2B",
            decoder_size="E2B",
            tiny=self.cfg.tiny,
            gate_init_tanh=0.05,
            init_rms_scale=1.0,
            learnable_cross_scale=True,
        )
        if self.cfg.tiny:
            self.model = BifocalGemmaForCausalLM(self.model_config).to(self.cfg.device)
        else:
            dec_src = self.cfg.pretrained_backbone_dir or self.cfg.decoder_backbone
            enc_src = (
                (self.cfg.pretrained_backbone_dir or self.cfg.encoder_backbone)
                if not is_baseline
                else None
            )
            backbone_dtype = (
                torch.bfloat16
                if self.cfg.device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else None
            )
            self.model = BifocalGemmaForCausalLM.from_pretrained_backbones(
                config=self.model_config,
                decoder_path_or_id=dec_src,
                encoder_path_or_id=enc_src,
                pretrained_backbone_dir=self.cfg.pretrained_backbone_dir,
                torch_dtype=backbone_dtype,
                device=self.cfg.device,
            )

        self.model.enable_memory_efficient_training(gradient_checkpointing=not self.cfg.tiny)

        # 2. Setup tokenizer if available (only for full-scale models; tiny uses synthetic vocab_size)
        self.tokenizer = None
        if not self.cfg.tiny and self.cfg.tokenizer_path and Path(self.cfg.tokenizer_path).is_dir():
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg.tokenizer_path, local_files_only=True
                )
            except Exception as _exc:
                logger.debug("Local tokenizer load skipped: %s", _exc)

        # 3. Setup dataloaders
        vocab = self.model_config.vocab_size
        max_vocab = vocab if self.cfg.tiny else None
        if train_dataloader is None:
            data_path = Path(self.cfg.data_dir) if self.cfg.data_dir else None
            if data_path is None and not self.cfg.tiny and self.cfg.auto_download_curated:
                pool_dir = Path("outputs/v2_cache/dataset") / self.cfg.collection
                try:
                    from bifocalgemma.data.releases import fetch_collection

                    fetch_collection(
                        self.cfg.dataset_repo,
                        self.cfg.dataset_revision,
                        self.cfg.collection,
                        pool_dir,
                        expected_role="train",
                        token=self.cfg.hf_token,
                    )
                    if pool_dir.exists():
                        data_path = pool_dir
                except Exception as _exc:
                    logger.warning("Curated collection fetch skipped: %s", _exc)

            if data_path is not None and data_path.exists():
                train_file = (
                    data_path / "train.jsonl"
                    if (data_path / "train.jsonl").is_file()
                    else data_path
                )
                train_ds: Dataset = JSONLRepairDataset(
                    train_file,
                    tokenizer=self.tokenizer,
                    is_baseline=is_baseline,
                    split="train",
                    max_vocab_size=max_vocab,
                )
                val_file = (
                    data_path / "val.jsonl"
                    if (data_path / "val.jsonl").is_file()
                    else (data_path / "dev.jsonl" if (data_path / "dev.jsonl").is_file() else None)
                )
                if val_file is not None:
                    val_ds: Dataset = JSONLRepairDataset(
                        val_file,
                        tokenizer=self.tokenizer,
                        is_baseline=is_baseline,
                        split="val",
                        max_vocab_size=max_vocab,
                    )
                else:
                    if len(train_ds) > 1:
                        val_len = max(1, int(len(train_ds) * 0.1))
                        train_len = len(train_ds) - val_len
                        train_ds, val_ds = torch.utils.data.random_split(
                            train_ds,
                            [train_len, val_len],
                            generator=torch.Generator().manual_seed(self.cfg.seed),
                        )
                    else:
                        val_ds = train_ds

                collator = RepairCollator(pad_token_id=0, ignore_index=-100)
                self.train_loader = DataLoader(
                    train_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    collate_fn=collator,
                )
                self.val_loader = DataLoader(
                    val_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    collate_fn=collator,
                )
            elif not self.cfg.tiny and (self.cfg.auto_download_curated or self.cfg.data_dir):
                raise RuntimeError(
                    "No dataset available for non-tiny fast training. "
                    "Provide data_dir or enable auto_download_curated with network access."
                )
            else:
                train_ds = SyntheticRepairDataset(num_samples=256, vocab_size=vocab)
                self.train_loader = DataLoader(
                    train_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    collate_fn=collate_fast_batch,
                )
                val_ds = SyntheticRepairDataset(num_samples=32, vocab_size=vocab)
                self.val_loader = DataLoader(
                    val_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    collate_fn=collate_fast_batch,
                )
        else:
            self.train_loader = train_dataloader
            self.val_loader = val_dataloader if val_dataloader is not None else train_dataloader

        self.out_path = Path(self.cfg.output_dir)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def configure_stage1(self) -> torch.optim.Optimizer:
        """Stage 1: Freeze decoder backbone; train SwiGLU bridge and decoupled cross-attention."""
        if self.model.encoder is None:
            raise ValueError(
                "Stage 1 Bridge Alignment requires an encoder; baseline decoder-only model has no bridge."
            )
        # Freeze all decoder parameters except cross bridge
        for name, param in self.model.decoder.named_parameters():
            if is_cross_bridge_parameter(name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        if self.model.encoder is not None:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        if self.model.encoder_projection is not None:
            for param in self.model.encoder_projection.parameters():
                param.requires_grad = True

        self.model.lm_head.weight.requires_grad = False

        return build_bifocalgemma_v3_optimizer(
            self.model,
            lr_bridge=self.cfg.lr_bridge,
            lr_cross=self.cfg.lr_cross,
            lr_lora=0.0,
            weight_decay_bridge=self.cfg.weight_decay_bridge,
        )

    def configure_stage2(self) -> torch.optim.Optimizer:
        """Stage 2: Unfreeze LoRA on encoder and decoder with conservative learning rate."""
        # Attach probe LoRA to attention projections
        configure_probe_training(self.model, policy="lora", rank=16)

        # Ensure cross bridge and projection remain trainable
        if self.model.encoder_projection is not None:
            for param in self.model.encoder_projection.parameters():
                param.requires_grad = True

        for name, param in self.model.decoder.named_parameters():
            if is_cross_bridge_parameter(name):
                param.requires_grad = True

        return build_bifocalgemma_v3_optimizer(
            self.model,
            lr_bridge=self.cfg.lr_bridge * 0.5,
            lr_cross=self.cfg.lr_cross * 0.5,
            lr_lora=self.cfg.lr_lora,
            weight_decay_bridge=self.cfg.weight_decay_bridge,
        )

    def train_step(
        self, batch: dict[str, torch.Tensor] | Seq2SeqBatch, stage_name: str
    ) -> torch.Tensor:
        """Execute single forward pass with clean chunked Differential Cross-Entropy."""
        if isinstance(batch, Seq2SeqBatch):
            seq2seq_batch = batch.to(self.cfg.device)
            if self.model.encoder is None and seq2seq_batch.encoder_input_ids.shape[1] > 0:
                bsz = seq2seq_batch.decoder_input_ids.shape[0]
                seq2seq_batch = Seq2SeqBatch(
                    encoder_input_ids=torch.empty(
                        (bsz, 0), dtype=torch.long, device=self.cfg.device
                    ),
                    encoder_attention_mask=torch.empty(
                        (bsz, 0), dtype=torch.bool, device=self.cfg.device
                    ),
                    decoder_input_ids=seq2seq_batch.decoder_input_ids,
                    decoder_attention_mask=seq2seq_batch.decoder_attention_mask,
                    labels=seq2seq_batch.labels,
                )
        else:
            bsz = batch["decoder_input_ids"].shape[0]
            device = self.cfg.device
            dec_ids = batch["decoder_input_ids"].to(device)
            dec_mask = (
                batch["decoder_attention_mask"].to(device)
                if "decoder_attention_mask" in batch and batch["decoder_attention_mask"] is not None
                else torch.ones_like(dec_ids, dtype=torch.bool)
            )
            labels = (
                batch.get("labels") if "labels" in batch else batch.get("targets", dec_ids)
            ).to(device)

            if (
                "encoder_input_ids" in batch
                and batch["encoder_input_ids"] is not None
                and batch["encoder_input_ids"].shape[1] > 0
                and self.model.encoder is not None
            ):
                enc_ids = batch["encoder_input_ids"].to(device)
                enc_mask = (
                    batch["encoder_attention_mask"].to(device)
                    if "encoder_attention_mask" in batch
                    and batch["encoder_attention_mask"] is not None
                    else torch.ones_like(enc_ids, dtype=torch.bool)
                )
            else:
                enc_ids = torch.empty((bsz, 0), dtype=torch.long, device=device)
                enc_mask = torch.empty((bsz, 0), dtype=torch.bool, device=device)

            seq2seq_batch = Seq2SeqBatch(
                encoder_input_ids=enc_ids,
                encoder_attention_mask=enc_mask,
                decoder_input_ids=dec_ids,
                decoder_attention_mask=dec_mask,
                labels=labels,
            )

        return compute_bifocalgemma_v3_loss(
            self.model,
            seq2seq_batch,
            stage_name=stage_name,
            lambda_ptr=self.cfg.lambda_ptr,
        )

    def evaluate(self) -> float:
        """Compute validation cross-entropy loss."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.train_step(batch, stage_name="eval")
                total_loss += float(loss.item())
                count += 1
        self.model.train()
        return total_loss / max(1, count)

    def export_pretrained(
        self,
        export_dir: str | Path | None = None,
        push_to_hub: bool | None = None,
        hub_repo: str | None = None,
        token: str | None = None,
    ) -> Path:
        """Export standalone PreTrainedModel, merging LoRA weights and optionally uploading to Hub."""
        target_dir = (
            Path(export_dir) if export_dir is not None else self.out_path / "final_pretrained_model"
        )
        should_push = self.cfg.push_to_hub if push_to_hub is None else push_to_hub
        repo = self.cfg.hub_repo if hub_repo is None else hub_repo
        hf_token = self.cfg.hf_token if token is None else token

        self.model.save_pretrained(
            target_dir,
            push_to_hub=should_push,
            repo_id=repo,
            token=hf_token,
            tokenizer=self.tokenizer,
        )
        return target_dir

    def run(self) -> dict[str, Any]:
        """Execute complete fast two-stage training and export standalone model."""
        start_time = time.time()
        global_step = 0
        data_iter = iter(self.train_loader)

        # -------------------------------------------------------------
        # Training Execution (Single-Stage Baseline or Two-Stage Paired)
        # -------------------------------------------------------------
        if self.model.encoder is None:
            # Baseline decoder-only model: single continuous full Decoder SFT pass
            full_steps = self.cfg.total_steps
            logger.info("Starting Full Decoder SFT (Baseline Decoder-Only, %d steps)", full_steps)
            opt = self.configure_stage2()

            while global_step < full_steps:
                self.model.train()
                opt.zero_grad()
                accum_loss = 0.0

                for _ in range(self.cfg.gradient_accumulation_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    loss = (
                        self.train_step(batch, stage_name="full_decoder_sft")
                        / self.cfg.gradient_accumulation_steps
                    )
                    loss.backward()
                    accum_loss += float(loss.item())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                global_step += 1

                if global_step % 50 == 0 or global_step == full_steps:
                    val_loss = (
                        self.evaluate() if global_step % self.cfg.eval_interval == 0 else None
                    )
                    record = {
                        "stage": "full_decoder_sft",
                        "step": global_step,
                        "loss": accum_loss * self.cfg.gradient_accumulation_steps,
                        "val_loss": val_loss,
                        "elapsed_sec": time.time() - start_time,
                    }
                    self.history.append(record)
                    if self.callback:
                        self.callback(record)
        else:
            # Paired BifocalGemma model: Two-Stage Training (Stage 1 Bridge Alignment + Stage 2 Joint SFT)
            # -------------------------------------------------------------
            # STAGE 1: Bridge Manifold Alignment
            # -------------------------------------------------------------
            logger.info("Starting Stage 1: Bridge Alignment (%d steps)", self.cfg.stage1_steps)
            opt1 = self.configure_stage1()

            while global_step < self.cfg.stage1_steps:
                self.model.train()
                opt1.zero_grad()
                accum_loss = 0.0

                for _ in range(self.cfg.gradient_accumulation_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    loss = (
                        self.train_step(batch, stage_name="stage1")
                        / self.cfg.gradient_accumulation_steps
                    )
                    loss.backward()
                    accum_loss += float(loss.item())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt1.step()
                global_step += 1

                if global_step % 50 == 0 or global_step == self.cfg.stage1_steps:
                    val_loss = (
                        self.evaluate() if global_step % self.cfg.eval_interval == 0 else None
                    )
                    record = {
                        "stage": "stage1_bridge_align",
                        "step": global_step,
                        "loss": accum_loss * self.cfg.gradient_accumulation_steps,
                        "val_loss": val_loss,
                        "elapsed_sec": time.time() - start_time,
                    }
                    self.history.append(record)
                    if self.callback:
                        self.callback(record)

            # -------------------------------------------------------------
            # STAGE 2: Joint LoRA Fine-Tuning
            # -------------------------------------------------------------
            stage2_target = self.cfg.stage1_steps + self.cfg.stage2_steps
            remaining_stage2_steps = stage2_target - global_step
            logger.info("Starting Stage 2: Joint LoRA SFT (%d steps)", remaining_stage2_steps)
            opt2 = self.configure_stage2()

            while global_step < stage2_target:
                self.model.train()
                opt2.zero_grad()
                accum_loss = 0.0

                for _ in range(self.cfg.gradient_accumulation_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.train_loader)
                        batch = next(data_iter)

                    loss = (
                        self.train_step(batch, stage_name="stage2")
                        / self.cfg.gradient_accumulation_steps
                    )
                    loss.backward()
                    accum_loss += float(loss.item())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt2.step()
                global_step += 1

                if global_step % 50 == 0 or global_step == stage2_target:
                    val_loss = (
                        self.evaluate() if global_step % self.cfg.eval_interval == 0 else None
                    )
                    record = {
                        "stage": "stage2_joint_sft",
                        "step": global_step,
                        "loss": accum_loss * self.cfg.gradient_accumulation_steps,
                        "val_loss": val_loss,
                        "elapsed_sec": time.time() - start_time,
                    }
                    self.history.append(record)
                    if self.callback:
                        self.callback(record)

        # -------------------------------------------------------------
        # Automatic Standalone PreTrainedModel Export
        # -------------------------------------------------------------
        final_val_loss = self.evaluate()
        export_dir = self.out_path / "final_pretrained_model"
        logger.info("Training complete. Exporting standalone Hugging Face model to %s", export_dir)

        self.export_pretrained(export_dir)

        # Save training summary
        summary = {
            "model_choice": self.cfg.model_choice,
            "total_steps": global_step,
            "final_val_loss": final_val_loss,
            "export_dir": str(export_dir),
            "config": asdict(self.cfg),
            "history": self.history[-10:],
            "completed_at": time.time(),
        }
        (self.out_path / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary


def run_fast_training(argv: list[str] | None = None) -> int:
    """CLI entrypoint for fast from-scratch training."""
    parser = argparse.ArgumentParser(description="Fast From-Scratch BifocalGemma Training")
    parser.add_argument(
        "--model",
        choices=["bifocalgemma-2b", "baseline-2b"],
        default="bifocalgemma-2b",
        help="Model architecture: primary BifocalGemma-2B or Baseline-2B",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--output-dir", type=str, default="outputs/bifocalgemma_2b")
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Directory containing training data"
    )
    parser.add_argument(
        "--pretrained-backbone-dir",
        type=str,
        default=None,
        help="Local directory containing pretrained backbone safetensors",
    )
    parser.add_argument(
        "--decoder-backbone",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Hugging Face repo id or local path for decoder backbone",
    )
    parser.add_argument(
        "--encoder-backbone",
        type=str,
        default="google/gemma-4-E2B",
        help="Hugging Face repo id or local path for encoder backbone",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="outputs/context_tokenizers/raw/google__gemma-4-E2B-it",
        help="Path to verified local tokenizer assets",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", type=str, default=None)
    parser.add_argument("--tiny", action="store_true", help="Run tiny smoke training for CI")

    args = parser.parse_args(argv)
    stage1 = 0 if args.model == "baseline-2b" else args.steps // 2
    stage2 = args.steps if args.model == "baseline-2b" else args.steps - stage1
    cfg = FastTrainingConfig(
        model_choice=args.model,
        total_steps=args.steps,
        stage1_steps=stage1,
        stage2_steps=stage2,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        pretrained_backbone_dir=args.pretrained_backbone_dir,
        decoder_backbone=args.decoder_backbone,
        encoder_backbone=args.encoder_backbone,
        tokenizer_path=args.tokenizer_path,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
        tiny=args.tiny,
    )
    pipeline = FastTrainingPipeline(cfg)
    summary = pipeline.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run_fast_training())
