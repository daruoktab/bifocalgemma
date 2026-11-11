"""Fast, focused from-scratch training pipeline for BifocalGemma-2B and Baseline-2B.

Eliminates slow multi-model matrix iteration, disables copy-biasing pointer loss
(lambda_ptr=0.0), introduces learnable cross-attention sharpness scaling, and automatically
exports an intact, standalone Hugging Face PreTrainedModel at training completion.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch.utils.data import DataLoader, Dataset

from bifocalgemma.assets import ensure_assets_on_disk
from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
from bifocalgemma.contracts import Seq2SeqBatch, TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.modeling_bifocalgemma import BifocalGemmaForCausalLM
from bifocalgemma.training.memory_probe import ProbeLoRALinear, configure_probe_training
from bifocalgemma.v3.curriculum import build_anti_forgetting_curriculum
from bifocalgemma.v3.objectives import (
    build_bifocalgemma_v3_optimizer,
    compute_v3_step_loss,
    is_cross_bridge_parameter,
)

logger = logging.getLogger(__name__)

DEFAULT_FAST_HF_REPO = "daruokta/bifocalgemma-2026-11-02"


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
    stage1_lambda_ptr: float = 0.15  # Stage 1 pointer grounding loss
    gate_init_tanh: float = 0.10  # Initial tanh cross-attention gate value
    stage1_encoder_lora: bool = True  # Enable ProbeLoRALinear on encoder during Stage 1
    use_comprehension_curriculum: bool = True  # Mix anti-forgetting comprehension replay in Stage 1
    inject_query_into_encoder: bool = (
        True  # Inject task instruction [QUERY] into bidirectional encoder
    )
    warmup_steps: int = 100
    weight_decay_bridge: float = 1e-4
    eval_interval: int = 200
    save_interval: int = 250
    log_interval: int = 25
    output_dir: str = "outputs/bifocalgemma_2b"
    data_dir: str | None = None  # Path to real repair dataset (jsonl or directory)
    pretrained_backbone_dir: str | None = None  # Local directory containing backbone safetensors
    decoder_backbone: str = "google/gemma-4-E2B-it"
    encoder_backbone: str = "google/gemma-4-E2B"
    tokenizer_path: str = "outputs/context_tokenizers/raw/google__gemma-4-E2B-it"
    push_to_hub: bool = False
    hub_repo: str | None = DEFAULT_FAST_HF_REPO
    hub_subfolder: str | None = None
    hf_token: str | None = None
    auto_download_curated: bool = True
    resume_if_available: bool = True
    collection: str = "mixed_context_enc32k_dec8k_v3"
    dataset_repo: str = "daruokta/bifocalgemma-data"
    dataset_revision: str = "7972f95d7c87f163c8b5af6a4acd308c0c1d0e23"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    tiny: bool = False  # Set True for instant CI/smoke validation
    run_benchmarks: bool = True  # Automatically run the 8-benchmark suite after training

    def __post_init__(self) -> None:
        if self.hub_subfolder is None:
            self.hub_subfolder = f"fast-{self.model_choice}"
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
        inject_query_into_encoder: bool = True,
    ) -> None:
        self.examples: list[TokenizedRepairExample] = []
        self.tokenizer = tokenizer
        self.max_vocab_size = max_vocab_size
        self.is_baseline = is_baseline
        self.inject_query_into_encoder = inject_query_into_encoder
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
            task_kind = str(row.get("task_type") or row.get("task_kind") or "repair")
            return TokenizedRepairExample(
                instance_id=str(row.get("instance_id", f"example-{idx}")),
                encoder_ids=enc_ids,
                decoder_prefix_ids=prefix_ids,
                target_ids=target_ids,
                metadata={"task_type": task_kind},
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
        instruction = row.get("instruction") or ""

        encoder_context = context
        if (
            not is_baseline
            and self.inject_query_into_encoder
            and instruction
            and (instruction not in context)
        ):
            encoder_context = f"[QUERY] {instruction}\n\n{context}"

        if tokenizer is not None:
            if is_baseline:
                enc_ids = ()
                prompt = f"{task_instruction_header}\n\n{context}"
            else:
                enc_ids = tuple(tokenizer.encode(encoder_context, add_special_tokens=True))
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

            max_baseline_tokens = 32768 + 8192 + 512
            if is_baseline and len(prefix_ids) + len(target) > max_baseline_tokens:
                max_prefix = max(64, max_baseline_tokens - len(target))
                if len(prefix_ids) > max_prefix:
                    prefix_ids = prefix_ids[:32] + prefix_ids[-(max_prefix - 32) :]
            elif not is_baseline and len(enc_ids) > 32768:
                enc_ids = enc_ids[:32768]

            if self.max_vocab_size is not None:
                limit = max(1, self.max_vocab_size - 10)
                enc_ids = tuple((t % limit) + 1 for t in enc_ids)
                prefix_ids = tuple((t % limit) + 1 for t in prefix_ids)
                target = tuple((t % limit) + 1 for t in target)

            task_kind = str(row.get("task_type") or row.get("task_kind") or "repair")
            return TokenizedRepairExample(
                instance_id=instance_id,
                encoder_ids=enc_ids,
                decoder_prefix_ids=prefix_ids or (2,),
                target_ids=target or (106,),
                metadata={"task_type": task_kind},
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

    def __getitem__(self, index: int) -> TokenizedRepairExample:
        return self.examples[index]


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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx = index
        g = torch.Generator().manual_seed(idx)
        enc_ids = torch.randint(10, self.vocab_size, (self.enc_len,), generator=g)
        dec_ids = torch.randint(10, self.vocab_size, (max(2, self.dec_len),), generator=g)
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
        ensure_assets_on_disk()
        self.cfg = config
        self.callback = callback
        torch.manual_seed(self.cfg.seed)

        if self.cfg.hf_token:
            import os

            os.environ["HF_TOKEN"] = self.cfg.hf_token
            try:
                from huggingface_hub import login

                login(token=self.cfg.hf_token, add_to_git_credential=False)
            except Exception as _login_exc:
                logger.debug("HF login skipped: %s", _login_exc)

        # 1. Instantiate BifocalGemma PreTrainedModel
        self.model = self._build_model()

        # 2. Setup tokenizer if available (only for full-scale models; tiny uses synthetic vocab_size)
        self.tokenizer = None
        if not self.cfg.tiny:
            tok_src = (
                self.cfg.tokenizer_path
                if self.cfg.tokenizer_path and Path(self.cfg.tokenizer_path).is_dir()
                else self.cfg.decoder_backbone
            )
            try:
                from transformers.models.gemma.tokenization_gemma import GemmaTokenizer

                self.tokenizer = GemmaTokenizer.from_pretrained(tok_src, token=self.cfg.hf_token)
            except Exception:
                try:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(tok_src, token=self.cfg.hf_token)
                except Exception as _exc:
                    logger.debug("Tokenizer load skipped: %s", _exc)

        # 3. Setup dataloaders
        self.train_loader, self.val_loader = self._prepare_dataloaders(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

        self.out_path = Path(self.cfg.output_dir)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def _build_model(self) -> BifocalGemmaForCausalLM:
        """Instantiate BifocalGemma PreTrainedModel."""
        is_baseline = self.cfg.model_choice == "baseline-2b"
        self.model_config = BifocalGemmaConfig(
            encoder_size="none" if is_baseline else "E2B",
            decoder_size="E2B",
            tiny=self.cfg.tiny,
            gate_init_tanh=self.cfg.gate_init_tanh,
            init_rms_scale=1.0,
            learnable_cross_scale=True,
        )
        if self.cfg.tiny:
            model = cast(Any, BifocalGemmaForCausalLM(self.model_config)).to(self.cfg.device)
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
            model = BifocalGemmaForCausalLM.from_pretrained_backbones(
                config=self.model_config,
                decoder_path_or_id=dec_src,
                encoder_path_or_id=enc_src,
                pretrained_backbone_dir=self.cfg.pretrained_backbone_dir,
                torch_dtype=backbone_dtype,
                device=self.cfg.device,
                token=self.cfg.hf_token,
            )

        model.enable_memory_efficient_training(gradient_checkpointing=not self.cfg.tiny)
        if model.decoder is not None and hasattr(model.decoder, "layers"):
            for layer in model.decoder.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and hasattr(attn, "cross_v_norm"):
                    if not hasattr(attn.cross_v_norm, "weight") or attn.cross_v_norm.weight is None:
                        dev = (
                            next(attn.parameters()).device
                            if list(attn.parameters())
                            else model.device
                        )
                        attn.cross_v_norm.weight = torch.nn.Parameter(
                            torch.ones(
                                getattr(attn, "head_dim", 32), dtype=torch.float32, device=dev
                            )
                        )
        return model

    def _prepare_dataloaders(
        self,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Set up training and validation dataloaders with curriculum replay support."""
        vocab = self.model_config.vocab_size
        max_vocab = vocab if self.cfg.tiny else None
        is_baseline = self.cfg.model_choice == "baseline-2b"

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
                    if not self.cfg.hf_token:
                        raise RuntimeError(
                            f"No dataset available for non-tiny fast training: gagal mengunduh dataset privat '{self.cfg.dataset_repo}'. "
                            "HF_TOKEN belum diisi atau bernilai kosong. "
                            "Masukkan Hugging Face Token Anda (dengan hak akses Read) untuk mengunduh dataset privat ini."
                        ) from _exc
                    else:
                        raise RuntimeError(
                            f"No dataset available for non-tiny fast training: gagal mengunduh dataset '{self.cfg.dataset_repo}'. "
                            f"Pastikan token Hugging Face Anda memiliki hak akses Read ke repositori privat ini. (Detail: {_exc})"
                        ) from _exc

            if data_path is not None and data_path.exists():
                train_file = (
                    data_path / "train.jsonl"
                    if (data_path / "train.jsonl").is_file()
                    else data_path
                )
                train_ds: Any = JSONLRepairDataset(
                    train_file,
                    tokenizer=self.tokenizer,
                    is_baseline=is_baseline,
                    split="train",
                    max_vocab_size=max_vocab,
                    inject_query_into_encoder=self.cfg.inject_query_into_encoder,
                )
                if (
                    isinstance(train_ds, JSONLRepairDataset)
                    and hasattr(train_ds, "examples")
                    and len(train_ds.examples) >= 8
                    and self.cfg.use_comprehension_curriculum
                ):
                    curriculum_examples, _ = build_anti_forgetting_curriculum(
                        train_ds.examples,
                        stage_name="stage1",
                        seed=self.cfg.seed,
                        decoder=self.tokenizer,
                    )
                    train_ds.examples = curriculum_examples

                val_file = (
                    data_path / "val.jsonl"
                    if (data_path / "val.jsonl").is_file()
                    else (data_path / "dev.jsonl" if (data_path / "dev.jsonl").is_file() else None)
                )
                if val_file is not None:
                    val_ds: Any = JSONLRepairDataset(
                        val_file,
                        tokenizer=self.tokenizer,
                        is_baseline=is_baseline,
                        split="val",
                        max_vocab_size=max_vocab,
                        inject_query_into_encoder=self.cfg.inject_query_into_encoder,
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
                train_loader = DataLoader(
                    train_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    collate_fn=collator,
                )
                val_loader = DataLoader(
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
                train_loader = DataLoader(
                    train_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    collate_fn=collate_fast_batch,
                )
                val_ds = SyntheticRepairDataset(num_samples=32, vocab_size=vocab)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    collate_fn=collate_fast_batch,
                )
        else:
            train_loader = train_dataloader
            val_loader = val_dataloader if val_dataloader is not None else train_dataloader

        return train_loader, val_loader

    def configure_stage1(self) -> torch.optim.Optimizer:
        """Stage 1: Freeze decoder backbone; train SwiGLU bridge, cross-attention, and optional encoder LoRA."""
        if self.model.encoder is None:
            raise ValueError(
                "Stage 1 Bridge Alignment requires an encoder; baseline decoder-only model has no bridge."
            )

        if self.cfg.stage1_encoder_lora:
            if not any(isinstance(m, ProbeLoRALinear) for m in self.model.modules()):
                configure_probe_training(self.model, policy="lora", rank=16)

            for name, module in self.model.named_modules():
                if isinstance(module, ProbeLoRALinear):
                    if name.startswith("encoder.") or ".encoder." in name:
                        module.scaling = 2.0
                    elif name.startswith("decoder.") or ".decoder." in name:
                        module.scaling = 0.0

            for name, param in self.model.named_parameters():
                if name.startswith("encoder_projection."):
                    param.requires_grad = True
                elif is_cross_bridge_parameter(name):
                    param.requires_grad = True
                elif (name.startswith("encoder.") or ".encoder." in name) and (
                    name.endswith(".lora_A") or name.endswith(".lora_B")
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
                self.model.lm_head.weight.requires_grad = False

            return build_bifocalgemma_v3_optimizer(
                self.model,
                lr_bridge=self.cfg.lr_bridge,
                lr_cross=self.cfg.lr_cross,
                lr_lora=self.cfg.lr_lora,
                weight_decay_bridge=self.cfg.weight_decay_bridge,
            )
        else:
            for name, param in self.model.decoder.named_parameters():
                if is_cross_bridge_parameter(name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

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
        if not any(isinstance(m, ProbeLoRALinear) for m in self.model.modules()):
            configure_probe_training(self.model, policy="lora", rank=16)

        for name, module in self.model.named_modules():
            if isinstance(module, ProbeLoRALinear):
                module.scaling = 2.0

        for name, param in self.model.named_parameters():
            if name.startswith("encoder_projection."):
                param.requires_grad = True
            elif is_cross_bridge_parameter(name):
                param.requires_grad = True
            elif name.endswith(".lora_A") or name.endswith(".lora_B"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            self.model.lm_head.weight.requires_grad = False

        return build_bifocalgemma_v3_optimizer(
            self.model,
            lr_bridge=self.cfg.lr_bridge * 0.5,
            lr_cross=self.cfg.lr_cross * 0.5,
            lr_lora=self.cfg.lr_lora,
            weight_decay_bridge=self.cfg.weight_decay_bridge,
        )

    def _update_learning_rates(self, optimizer: torch.optim.Optimizer, global_step: int) -> None:
        """Apply linear warmup and cosine decay to all optimizer parameter groups."""
        warmup_steps = max(1, self.cfg.warmup_steps)
        total_steps = max(warmup_steps + 1, self.cfg.total_steps)
        min_lr_ratio = 0.10

        if global_step < warmup_steps:
            factor = float(global_step) / float(warmup_steps)
        else:
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(global_step - warmup_steps) / float(decay_steps)))
            factor = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        for group in optimizer.param_groups:
            base_lr = group.get("base_lr", group["lr"])
            if "base_lr" not in group:
                group["base_lr"] = base_lr
            group["lr"] = base_lr * factor

    def save_checkpoint(
        self,
        global_step: int,
        stage_name: str,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Path:
        """Save lightweight trainable state locally and sync to Hugging Face Hub if enabled."""
        ckpt_dir = self.out_path / "checkpoint_latest"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "checkpoint_state.pt"
        trainable_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        saved_tensors = {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
            if (
                k in trainable_names
                or "lora_" in k
                or "encoder_projection" in k
                or is_cross_bridge_parameter(k)
            )
        }
        has_lora = any(isinstance(m, ProbeLoRALinear) for m in self.model.modules())
        active_opt = optimizer if optimizer is not None else getattr(self, "current_optimizer", None)
        opt_state = None
        if active_opt is not None:
            try:
                opt_state = active_opt.state_dict()
            except Exception as _opt_exc:
                logger.warning("Could not serialize optimizer state: %s", _opt_exc)
        payload = {
            "model_choice": self.cfg.model_choice,
            "global_step": global_step,
            "stage": stage_name,
            "has_lora": has_lora,
            "history": self.history[-200:],
            "tensors": saved_tensors,
            "optimizer": opt_state,
            "saved_at": time.time(),
        }
        tmp_file = ckpt_dir / "checkpoint_state.pt.tmp"
        torch.save(payload, str(tmp_file))
        tmp_file.replace(ckpt_file)

        # Also save versioned milestone checkpoint locally
        milestone_file = ckpt_dir / f"checkpoint_step_{global_step}.pt"
        try:
            torch.save(payload, str(milestone_file))
        except Exception as _ms_exc:
            logger.warning("Could not write milestone checkpoint file: %s", _ms_exc)

        if self.cfg.push_to_hub and self.cfg.hub_repo and self.cfg.hf_token:
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=self.cfg.hf_token)
                api.create_repo(repo_id=self.cfg.hub_repo, repo_type="model", exist_ok=True)
                sub = (self.cfg.hub_subfolder or f"fast-{self.cfg.model_choice}").strip("/")
                api.upload_file(
                    path_or_fileobj=str(ckpt_file),
                    path_in_repo=f"{sub}/checkpoint_latest/checkpoint_state.pt",
                    repo_id=self.cfg.hub_repo,
                    repo_type="model",
                    commit_message=f"[{self.cfg.model_choice}] Save checkpoint at step {global_step}",
                )
                if (
                    global_step % (self.cfg.save_interval * 2) == 0
                    or global_step == self.cfg.total_steps
                ):
                    api.upload_file(
                        path_or_fileobj=str(milestone_file),
                        path_in_repo=f"{sub}/checkpoint_latest/checkpoint_step_{global_step}.pt",
                        repo_id=self.cfg.hub_repo,
                        repo_type="model",
                        commit_message=f"[{self.cfg.model_choice}] Save milestone checkpoint at step {global_step}",
                    )
            except Exception as _exc:
                logger.warning("Remote checkpoint upload skipped at step %d: %s", global_step, _exc)
        return ckpt_file

    def try_resume_checkpoint(self) -> int:
        """Resume from local or remote Hugging Face checkpoint if available and compatible."""
        if not self.cfg.resume_if_available:
            return 0

        ckpt_dir = self.out_path / "checkpoint_latest"
        ckpt_file = ckpt_dir / "checkpoint_state.pt"

        local_step = 0
        if ckpt_file.is_file():
            try:
                local_payload = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
                if local_payload.get("model_choice") == self.cfg.model_choice:
                    local_step = int(local_payload.get("global_step", 0))
            except Exception as _l_exc:
                logger.warning("Could not read local checkpoint %s: %s", ckpt_file, _l_exc)

        # Check remote checkpoint on Hugging Face if enabled
        if (
            not self.cfg.tiny
            and self.cfg.push_to_hub
            and self.cfg.hub_repo
            and self.cfg.hf_token
        ):
            try:
                from huggingface_hub import hf_hub_download

                sub = (self.cfg.hub_subfolder or f"fast-{self.cfg.model_choice}").strip("/")
                remote_file = hf_hub_download(
                    repo_id=self.cfg.hub_repo,
                    filename=f"{sub}/checkpoint_latest/checkpoint_state.pt",
                    repo_type="model",
                    token=self.cfg.hf_token,
                )
                remote_payload = torch.load(remote_file, map_location="cpu", weights_only=False)
                if remote_payload.get("model_choice") == self.cfg.model_choice:
                    remote_step = int(remote_payload.get("global_step", 0))
                    if remote_step > local_step:
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        import shutil

                        shutil.copy2(remote_file, ckpt_file)
                        print(
                            f"[{time.strftime('%X')}] 📥 Mengunduh checkpoint terbaru dari Hub (step {remote_step} > local {local_step})...",
                            flush=True,
                        )
                    else:
                        logger.info(
                            "Local checkpoint step %d >= remote step %d; keeping local.",
                            local_step,
                            remote_step,
                        )
            except Exception as _hub_exc:
                logger.info("No remote checkpoint downloaded from Hub: %s", _hub_exc)

        if ckpt_file.is_file():
            try:
                payload = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
                if payload.get("model_choice") == self.cfg.model_choice:
                    saved_step = int(payload.get("global_step", 0))
                    if saved_step > 0:
                        has_lora = payload.get("has_lora", False)
                        if has_lora and not any(
                            isinstance(m, ProbeLoRALinear) for m in self.model.modules()
                        ):
                            configure_probe_training(self.model, policy="lora", rank=16)

                        stage = str(payload.get("stage", ""))
                        is_stage1 = (saved_step < self.cfg.stage1_steps) or ("stage1" in stage)
                        if has_lora:
                            if (
                                is_stage1
                                and self.cfg.stage1_encoder_lora
                                and self.model.encoder is not None
                            ):
                                for name, module in self.model.named_modules():
                                    if isinstance(module, ProbeLoRALinear):
                                        if name.startswith("encoder.") or ".encoder." in name:
                                            module.scaling = 2.0
                                        elif name.startswith("decoder.") or ".decoder." in name:
                                            module.scaling = 0.0
                            else:
                                for name, module in self.model.named_modules():
                                    if isinstance(module, ProbeLoRALinear):
                                        module.scaling = 2.0

                        tensors = payload.get("tensors", {})
                        if tensors:
                            self.model.load_state_dict(tensors, strict=False)
                            cast(Any, self.model).to(self.cfg.device)
                        self.history = list(payload.get("history", []))
                        self._resumed_optimizer_state = payload.get("optimizer")
                        print(
                            f"[{time.strftime('%X')}] 🔄 Melanjutkan otomatis dari checkpoint step {saved_step}/{self.cfg.total_steps} ({self.cfg.model_choice})...",
                            flush=True,
                        )
                        return saved_step
            except Exception as _exc:
                logger.warning("Could not resume from %s: %s", ckpt_file, _exc)

        # Fallback: check if a completed final_pretrained_model already exists locally or on HF Hub
        local_final = self.out_path / "final_pretrained_model" / "model.safetensors"
        if (
            not local_final.is_file()
            and self.cfg.push_to_hub
            and self.cfg.hub_repo
            and self.cfg.hf_token
            and self._is_remote_pretrained_uploaded()
        ):
            try:
                import shutil

                from huggingface_hub import hf_hub_download

                sub = (self.cfg.hub_subfolder or f"fast-{self.cfg.model_choice}").strip("/")
                downloaded_model = hf_hub_download(
                    repo_id=self.cfg.hub_repo,
                    filename=f"{sub}/pretrained_model/model.safetensors",
                    repo_type="model",
                    token=self.cfg.hf_token,
                )
                local_final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded_model, local_final)
            except Exception:
                pass

        if local_final.is_file():
            try:
                from safetensors.torch import load_file

                self.model.merge_and_unload()
                state = load_file(str(local_final), device="cpu")
                self.model.load_state_dict(state, strict=False)
                cast(Any, self.model).to(self.cfg.device)
                print(
                    f"[{time.strftime('%X')}] ✅ Memuat model selesai ({self.cfg.total_steps}/{self.cfg.total_steps} steps) dari {local_final}...",
                    flush=True,
                )
                return self.cfg.total_steps
            except Exception as _exc:
                logger.warning(
                    "Could not load completed pretrained model %s: %s", local_final, _exc
                )

        return 0

    load_checkpoint_if_available = try_resume_checkpoint

    def _sync_existing_legacy_export_if_present(self) -> None:
        """If a completed 2000-step model exists at outputs/fast_bifocalgemma/final_pretrained_model, upload it to Hub."""
        if not (self.cfg.push_to_hub and self.cfg.hub_repo and self.cfg.hf_token):
            return
        legacy_dir = Path("outputs/fast_bifocalgemma/final_pretrained_model")
        marker = legacy_dir / ".uploaded_to_hub"
        if (
            self.cfg.model_choice == "bifocalgemma-2b"
            and (legacy_dir / "model.safetensors").is_file()
            and not marker.is_file()
        ):
            try:
                from huggingface_hub import HfApi

                sub = (self.cfg.hub_subfolder or "fast-bifocalgemma-2b").strip("/")
                print(
                    f"[{time.strftime('%X')}] ☁️ Menemukan model selesai di {legacy_dir}, mengunggah otomatis ke {self.cfg.hub_repo} ({sub}/pretrained_model)...",
                    flush=True,
                )
                api = HfApi(token=self.cfg.hf_token)
                api.create_repo(repo_id=self.cfg.hub_repo, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=str(legacy_dir),
                    repo_id=self.cfg.hub_repo,
                    repo_type="model",
                    path_in_repo=f"{sub}/pretrained_model",
                )
                marker.write_text(self.cfg.hub_repo, encoding="utf-8")
                print(
                    f"[{time.strftime('%X')}] ✅ Berhasil mengunggah model selesai ke Hugging Face: {self.cfg.hub_repo}/{sub}/pretrained_model",
                    flush=True,
                )
            except Exception as _exc:
                logger.warning("Legacy model auto-upload skipped: %s", _exc)

    def train_step(
        self,
        batch: dict[str, torch.Tensor] | Seq2SeqBatch,
        stage_name: str,
        lambda_ptr: float | None = None,
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
            lbl_raw = batch.get("labels") if "labels" in batch else batch.get("targets", dec_ids)
            labels = (lbl_raw if lbl_raw is not None else dec_ids).to(device)

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

        if lambda_ptr is None:
            if stage_name in ("stage1_bridge_warmup", "stage1_bridge_align", "stage1"):
                active_lambda_ptr = (
                    self.cfg.lambda_ptr if self.cfg.lambda_ptr > 0 else self.cfg.stage1_lambda_ptr
                )
            else:
                active_lambda_ptr = self.cfg.lambda_ptr
        else:
            active_lambda_ptr = lambda_ptr

        return compute_v3_step_loss(
            self.model,
            seq2seq_batch,
            stage_name=stage_name,
            lambda_ptr=active_lambda_ptr,
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
        sub = (self.cfg.hub_subfolder or f"fast-{self.cfg.model_choice}").strip("/")
        path_in_repo = f"{sub}/pretrained_model" if sub else None

        self.model.save_pretrained(
            target_dir,
            push_to_hub=should_push,
            repo_id=repo,
            path_in_repo=path_in_repo,
            token=hf_token,
            tokenizer=self.tokenizer,
        )
        return target_dir

    def _is_remote_pretrained_uploaded(self) -> bool:
        """Check whether the standalone pretrained model already exists on Hugging Face Hub."""
        if not (self.cfg.push_to_hub and self.cfg.hub_repo and self.cfg.hf_token):
            return False
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.cfg.hf_token)
            sub = (self.cfg.hub_subfolder or f"fast-{self.cfg.model_choice}").strip("/")
            return api.file_exists(
                repo_id=self.cfg.hub_repo,
                filename=f"{sub}/pretrained_model/model.safetensors",
                repo_type="model",
            )
        except Exception:
            return False

    def _ensure_verified_tokenizers(self, spec: Any, tokenizer_cache: Path) -> None:
        """Ensure audited tokenizer assets exist in tokenizer_cache/raw/... for benchmark evaluation."""
        import hashlib
        import shutil

        from huggingface_hub import hf_hub_download

        from bifocalgemma.assets import load_asset_json
        from bifocalgemma.data.tokenization import load_verified_tokenizer

        manifest = load_asset_json("tokenizer_manifest.json")
        refs = [spec.decoder] + ([spec.encoder] if spec.encoder else [])
        for ref in refs:
            assets = manifest[ref.model_id]
            raw = tokenizer_cache / "raw" / ref.model_id.replace("/", "__")
            raw.mkdir(parents=True, exist_ok=True)
            all_present = True
            for name, digest in assets["files"].items():
                target_file = raw / name
                if (
                    not target_file.is_file()
                    or hashlib.sha256(target_file.read_bytes()).hexdigest() != digest
                ):
                    all_present = False
                    break
            if not all_present:
                for name, digest in assets["files"].items():
                    downloaded = Path(
                        hf_hub_download(
                            ref.model_id,
                            name,
                            revision=ref.revision,
                            token=self.cfg.hf_token,
                        )
                    )
                    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Pinned tokenizer asset hash mismatch: {name}")
                    shutil.copyfile(downloaded, raw / name)
            load_verified_tokenizer(ref, tokenizer_cache)

    def run_benchmark_suite(
        self, summary_base: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute the 8-benchmark evaluation suite with automatic per-benchmark & intra-benchmark resume and Hub sync."""
        if self.cfg.tiny:
            return []

        # 1. Clear residual training memory and cached allocator blocks
        import gc
        import shutil

        from bifocalgemma.data.releases import fetch_collection
        from bifocalgemma.models import inference_cache as inference_cache_mod
        from bifocalgemma.v3.attention import FullCrossCachedAttention
        from bifocalgemma.v3.benchmark import run_batched_context_benchmark
        from bifocalgemma.v3.constants import BENCHMARKS
        from bifocalgemma.v3.matrix import make_experiment_spec

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Consolidate LoRA adapters and configure model for fast memory-efficient inference
        self.model.merge_and_unload()
        self.model.enable_memory_efficient_training(gradient_checkpointing=False)
        self.model.eval()
        setattr(inference_cache_mod, "CachedAttention", FullCrossCachedAttention)

        model_key = "e2b-it" if self.cfg.model_choice == "baseline-2b" else "2base-2it"
        spec = make_experiment_spec(model_key)
        tokenizer_cache = Path("outputs/v2_cache/tokenizers")
        self._ensure_verified_tokenizers(spec, tokenizer_cache)

        sub = (self.cfg.hub_subfolder or f"fast-{self.cfg.model_choice}").strip("/")
        bench_root = self.out_path / "benchmarks"
        bench_root.mkdir(parents=True, exist_ok=True)
        checkpoint_identity = {
            "model_choice": self.cfg.model_choice,
            "total_steps": self.cfg.total_steps,
        }

        total_bench = len(BENCHMARKS)
        print(
            f"[{time.strftime('%X')}] 📊 Memulai Benchmark Suite ({total_bench} benchmarks) untuk {self.cfg.model_choice}...",
            flush=True,
        )

        reports: list[dict[str, Any]] = []
        for idx, item in enumerate(BENCHMARKS, start=1):
            collection_name = str(item["collection"])
            bench_out_dir = bench_root / collection_name
            bench_out_dir.mkdir(parents=True, exist_ok=True)
            report_file = bench_out_dir / "benchmark_report.json"
            remote_bench_prefix = f"{sub}/benchmarks/{collection_name}"

            # Try restoring completed or partial benchmark state from Hugging Face Hub
            if (
                not report_file.is_file()
                and self.cfg.push_to_hub
                and self.cfg.hub_repo
                and self.cfg.hf_token
            ):
                try:
                    from huggingface_hub import hf_hub_download

                    dl_report = hf_hub_download(
                        repo_id=self.cfg.hub_repo,
                        filename=f"{remote_bench_prefix}/benchmark_report.json",
                        repo_type="model",
                        token=self.cfg.hf_token,
                    )
                    shutil.copy2(dl_report, report_file)
                except Exception:
                    for partial_name in (
                        "evaluation_state.json",
                        "predictions.jsonl",
                        "timing.json",
                    ):
                        target_partial = bench_out_dir / partial_name
                        if not target_partial.is_file():
                            try:
                                from huggingface_hub import hf_hub_download

                                dl_part = hf_hub_download(
                                    repo_id=self.cfg.hub_repo,
                                    filename=f"{remote_bench_prefix}/{partial_name}",
                                    repo_type="model",
                                    token=self.cfg.hf_token,
                                )
                                shutil.copy2(dl_part, target_partial)
                            except Exception:
                                pass

            if report_file.is_file():
                try:
                    existing_report = json.loads(report_file.read_text(encoding="utf-8"))
                    eval_state_file = bench_out_dir / "evaluation_state.json"
                    state_sig = None
                    if eval_state_file.is_file():
                        try:
                            state_sig = json.loads(eval_state_file.read_text(encoding="utf-8")).get("signature")
                        except Exception:
                            pass
                    is_valid_sig = True
                    if state_sig and existing_report.get("signature"):
                        is_valid_sig = (state_sig == existing_report.get("signature"))

                    if (
                        existing_report.get("status") == "COMPLETED"
                        and existing_report.get("experiment") == spec.name
                        and is_valid_sig
                    ):
                        acc = existing_report.get("accuracy")
                        acc_str = f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) else "N/A"
                        print(
                            f"[{time.strftime('%X')}] ✅ Benchmark [{idx}/{total_bench}] {collection_name} sudah selesai sebelumnya (accuracy={acc_str}). Melewati...",
                            flush=True,
                        )
                        reports.append(existing_report)
                        continue
                except Exception:
                    pass

            print(
                f"[{time.strftime('%X')}] 🔎 Menjalankan Benchmark [{idx}/{total_bench}]: {collection_name}...",
                flush=True,
            )
            bench_data_dir = Path("outputs/v2_cache/benchmarks") / collection_name
            fetch_collection(
                self.cfg.dataset_repo,
                self.cfg.dataset_revision,
                collection_name,
                bench_data_dir,
                expected_role="eval",
                include_gold=True,
                token=self.cfg.hf_token,
            )

            last_log_done = {"count": -1, "ts": 0.0}

            def on_bench_progress(
                event: dict[str, Any],
                _idx: int = idx,
                _col: str = collection_name,
                _tracker: dict[str, Any] = last_log_done,
            ) -> None:
                done = int(event.get("items_done", 0))
                total = int(event.get("items_total", 0))
                correct = int(event.get("benchmark_correct", 0))
                now = time.time()
                if (
                    event.get("stage") == "benchmark_progress"
                    and done > _tracker["count"]
                    and (done % 25 == 0 or done == total or now - _tracker["ts"] >= 60.0)
                ):
                    _tracker["count"] = done
                    _tracker["ts"] = now
                    live_acc = (correct / max(1, done)) * 100.0
                    print(
                        f"[{time.strftime('%X')}]   ↳ [{_idx}/{total_bench}] {_col}: {done}/{total} selesai (live acc={live_acc:.2f}%)",
                        flush=True,
                    )
                if self.callback and total > 0:
                    self.callback(
                        {
                            "stage": f"bench {_idx}/{total_bench}: {_col}",
                            "step": self.cfg.total_steps,
                            "loss": 0.0,
                            "val_loss": None,
                            "bench_done": done,
                            "bench_total": total,
                            "bench_acc": (correct / max(1, done)) if done > 0 else 0.0,
                        }
                    )

            def sync_partial_benchmark(
                _dir: Path = bench_out_dir,
                _prefix: str = remote_bench_prefix,
                _col: str = collection_name,
            ) -> None:
                if not (self.cfg.push_to_hub and self.cfg.hub_repo and self.cfg.hf_token):
                    return
                try:
                    from huggingface_hub import HfApi

                    api = HfApi(token=self.cfg.hf_token)
                    for fname in ("evaluation_state.json", "predictions.jsonl", "timing.json"):
                        fpath = _dir / fname
                        if fpath.is_file():
                            api.upload_file(
                                path_or_fileobj=str(fpath),
                                path_in_repo=f"{_prefix}/{fname}",
                                repo_id=self.cfg.hub_repo,
                                repo_type="model",
                                commit_message=f"[{self.cfg.model_choice}] Sync benchmark progress ({_col})",
                            )
                except Exception as _exc:
                    logger.debug("Partial benchmark upload skipped: %s", _exc)

            try:
                report = run_batched_context_benchmark(
                    self.model,
                    spec,
                    bench_data_dir,
                    bench_out_dir,
                    tokenizer_cache,
                    checkpoint_identity=checkpoint_identity,
                    model_key=model_key,
                    benchmark_item=item,
                    encoder_limit=int(item["encoder_limit"]),
                    decoder_limit=int(item["decoder_limit"]),
                    max_new_tokens=int(item.get("max_new_tokens", 2048)),
                    allow_execution=True,
                    progress=on_bench_progress,
                    prediction_callback=sync_partial_benchmark,
                    upload_every=50,
                    use_cache=True,
                )
            except ValueError as _val_exc:
                if "Benchmark/checkpoint/protocol changed" in str(_val_exc):
                    shutil.rmtree(bench_out_dir, ignore_errors=True)
                    bench_out_dir.mkdir(parents=True, exist_ok=True)
                    report = run_batched_context_benchmark(
                        self.model,
                        spec,
                        bench_data_dir,
                        bench_out_dir,
                        tokenizer_cache,
                        checkpoint_identity=checkpoint_identity,
                        model_key=model_key,
                        benchmark_item=item,
                        encoder_limit=int(item["encoder_limit"]),
                        decoder_limit=int(item["decoder_limit"]),
                        max_new_tokens=int(item.get("max_new_tokens", 2048)),
                        allow_execution=True,
                        progress=on_bench_progress,
                        prediction_callback=sync_partial_benchmark,
                        upload_every=50,
                        use_cache=True,
                    )
                else:
                    raise

            reports.append(report)
            acc = report.get("accuracy")
            acc_str = f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) else "N/A"
            print(
                f"[{time.strftime('%X')}] 📊 Benchmark [{idx}/{total_bench}] {collection_name} selesai -> accuracy={acc_str} ({report.get('correct')}/{report.get('total')})",
                flush=True,
            )

            # Save and upload incremental benchmark summary + benchmark folder
            bench_summary_payload = {
                "model_choice": self.cfg.model_choice,
                "completed_benchmarks": len(reports),
                "total_benchmarks": total_bench,
                "benchmarks": reports,
                "updated_at": time.time(),
            }
            bench_summary_file = self.out_path / "benchmark_summary.json"
            bench_summary_file.write_text(
                json.dumps(bench_summary_payload, indent=2), encoding="utf-8"
            )
            if summary_base is not None:
                summary_base["benchmarks"] = reports
                (self.out_path / "training_summary.json").write_text(
                    json.dumps(summary_base, indent=2), encoding="utf-8"
                )

            if self.cfg.push_to_hub and self.cfg.hub_repo and self.cfg.hf_token:
                try:
                    from huggingface_hub import HfApi

                    api = HfApi(token=self.cfg.hf_token)
                    api.upload_folder(
                        folder_path=str(bench_out_dir),
                        repo_id=self.cfg.hub_repo,
                        repo_type="model",
                        path_in_repo=remote_bench_prefix,
                        commit_message=f"[{self.cfg.model_choice}] Benchmark {collection_name} completed (acc={acc_str})",
                    )
                    api.upload_file(
                        path_or_fileobj=str(bench_summary_file),
                        path_in_repo=f"{sub}/benchmark_summary.json",
                        repo_id=self.cfg.hub_repo,
                        repo_type="model",
                        commit_message=f"[{self.cfg.model_choice}] Update benchmark_summary.json ({len(reports)}/{total_bench})",
                    )
                except Exception as _exc:
                    logger.warning("Benchmark upload skipped for %s: %s", collection_name, _exc)

        return reports

    def _run_stage(
        self,
        stage_name: str,
        optimizer: torch.optim.Optimizer,
        start_step: int,
        target_step: int,
        data_iter: Any,
        start_time: float,
    ) -> tuple[int, Any]:
        """Execute one training stage with warmup + cosine scheduling, evaluation, and checkpointing."""
        self.current_stage = stage_name
        self.current_optimizer = optimizer
        global_step = start_step
        total_stage_steps = target_step - start_step
        log_step = max(1, min(self.cfg.log_interval, max(1, total_stage_steps // 20)))
        save_every = max(1, self.cfg.save_interval)

        while global_step < target_step:
            self.model.train()
            optimizer.zero_grad()
            accum_loss = 0.0

            if stage_name in ("stage1_bridge_warmup", "stage1_bridge_align", "stage1"):
                active_lambda_ptr = (
                    self.cfg.lambda_ptr if self.cfg.lambda_ptr > 0 else self.cfg.stage1_lambda_ptr
                )
            else:
                active_lambda_ptr = self.cfg.lambda_ptr

            for _ in range(self.cfg.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                loss = (
                    self.train_step(batch, stage_name=stage_name, lambda_ptr=active_lambda_ptr)
                    / self.cfg.gradient_accumulation_steps
                )
                loss.backward()
                accum_loss += float(loss.item())

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._update_learning_rates(optimizer, global_step)
            optimizer.step()
            global_step += 1

            step_loss = accum_loss * self.cfg.gradient_accumulation_steps
            val_loss = (
                self.evaluate()
                if (global_step % self.cfg.eval_interval == 0 or global_step == target_step)
                else None
            )
            record = {
                "stage": stage_name,
                "step": global_step,
                "loss": step_loss,
                "val_loss": val_loss,
                "elapsed_sec": time.time() - start_time,
            }
            if global_step % log_step == 0 or global_step == target_step:
                self.history.append(record)
                val_str = f" | val_loss={val_loss:.4f}" if val_loss is not None else ""
                print(
                    f"[{time.strftime('%X')}] Step {global_step}/{target_step} [{stage_name}] - loss={record['loss']:.4f}{val_str}",
                    flush=True,
                )
            if not self.cfg.tiny and (global_step % save_every == 0 or global_step == target_step):
                self.save_checkpoint(global_step, stage_name, optimizer=optimizer)
            if self.callback:
                self.callback(record)

        return global_step, data_iter

    def run(self) -> dict[str, Any]:
        """Execute complete fast two-stage training, export standalone model, and run benchmark suite."""
        start_time = time.time()
        self._sync_existing_legacy_export_if_present()
        global_step = self.try_resume_checkpoint()
        already_completed = global_step >= self.cfg.total_steps
        data_iter = iter(self.train_loader)
        current_stage = "completed" if already_completed else "starting"
        self.current_stage = current_stage

        try:
            if already_completed:
                last_loss = float(self.history[-1].get("loss", 0.0)) if self.history else 0.0
                last_val = next(
                    (
                        h["val_loss"]
                        for h in reversed(self.history)
                        if h.get("val_loss") is not None
                    ),
                    None,
                )
                print(
                    f"[{time.strftime('%X')}] ✅ Pelatihan {self.cfg.model_choice} terdeteksi sudah selesai ({global_step}/{self.cfg.total_steps} steps). Melewati tahap training...",
                    flush=True,
                )
                if self.callback:
                    self.callback(
                        {
                            "stage": "completed",
                            "step": global_step,
                            "loss": last_loss,
                            "val_loss": last_val,
                            "elapsed_sec": 0.0,
                        }
                    )
            # -------------------------------------------------------------
            # Training Execution (Single-Stage Baseline or Two-Stage Paired)
            # -------------------------------------------------------------
            elif self.model.encoder is None:
                # Baseline decoder-only model: single continuous full Decoder SFT pass
                full_steps = self.cfg.total_steps
                current_stage = "full_decoder_sft"
                logger.info(
                    "Starting Full Decoder SFT (Baseline Decoder-Only, %d steps)", full_steps
                )
                print(
                    f"[{time.strftime('%X')}] 🚀 Memulai Full Decoder SFT (Baseline-2B, {full_steps} steps)...",
                    flush=True,
                )
                opt = self.configure_stage2()
                resumed_opt_state = getattr(self, "_resumed_optimizer_state", None)
                if resumed_opt_state is not None:
                    try:
                        opt.load_state_dict(resumed_opt_state)
                        logger.info("Restored optimizer state into baseline SFT optimizer.")
                    except Exception as _opt_r_exc:
                        logger.warning("Could not restore baseline optimizer state: %s", _opt_r_exc)
                    self._resumed_optimizer_state = None
                global_step, data_iter = self._run_stage(
                    stage_name="full_decoder_sft",
                    optimizer=opt,
                    start_step=global_step,
                    target_step=full_steps,
                    data_iter=data_iter,
                    start_time=start_time,
                )
            else:
                # Paired BifocalGemma model: Two-Stage Training (Stage 1 Bridge Alignment + Stage 2 Joint SFT)
                # -------------------------------------------------------------
                # STAGE 1: Bridge Manifold Alignment
                # -------------------------------------------------------------
                opt1 = None
                if global_step < self.cfg.stage1_steps:
                    current_stage = "stage1_bridge_warmup"
                    logger.info(
                        "Starting Stage 1: Bridge Alignment (%d steps)", self.cfg.stage1_steps
                    )
                    print(
                        f"[{time.strftime('%X')}] 🚀 Memulai Stage 1: Bridge Alignment ({self.cfg.stage1_steps} steps)...",
                        flush=True,
                    )
                    opt1 = self.configure_stage1()
                    resumed_opt_state = getattr(self, "_resumed_optimizer_state", None)
                    if resumed_opt_state is not None:
                        try:
                            opt1.load_state_dict(resumed_opt_state)
                            logger.info("Restored optimizer state into Stage 1 optimizer.")
                        except Exception as _opt_r_exc:
                            logger.warning("Could not restore Stage 1 optimizer state: %s", _opt_r_exc)
                        self._resumed_optimizer_state = None
                    global_step, data_iter = self._run_stage(
                        stage_name="stage1_bridge_warmup",
                        optimizer=opt1,
                        start_step=global_step,
                        target_step=self.cfg.stage1_steps,
                        data_iter=data_iter,
                        start_time=start_time,
                    )

                # -------------------------------------------------------------
                # STAGE 2: Joint LoRA Fine-Tuning
                # -------------------------------------------------------------
                stage2_target = self.cfg.stage1_steps + self.cfg.stage2_steps
                if global_step < stage2_target:
                    current_stage = "stage2_joint_sft"
                    remaining_stage2_steps = stage2_target - global_step
                    logger.info(
                        "Starting Stage 2: Joint LoRA SFT (%d steps)", remaining_stage2_steps
                    )
                    print(
                        f"[{time.strftime('%X')}] 🚀 Memulai Stage 2: Joint LoRA SFT ({remaining_stage2_steps} steps)...",
                        flush=True,
                    )
                    opt2 = self.configure_stage2()

                    resumed_opt_state = getattr(self, "_resumed_optimizer_state", None)
                    if resumed_opt_state is not None:
                        try:
                            opt2.load_state_dict(resumed_opt_state)
                            logger.info("Restored optimizer state into Stage 2 optimizer.")
                        except Exception as _opt_r_exc:
                            logger.warning("Could not restore Stage 2 optimizer state: %s", _opt_r_exc)
                        self._resumed_optimizer_state = None
                    # Preserve AdamW momentum state (exp_avg, exp_avg_sq, step) from Stage 1 into Stage 2
                    elif opt1 is not None and hasattr(opt1, "state") and opt1.state:
                        opt2_params = {p for group in opt2.param_groups for p in group["params"]}
                        transferred = 0
                        for p, s in opt1.state.items():
                            if p in opt2_params:
                                opt2.state[p] = {
                                    k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                    for k, v in s.items()
                                }
                                transferred += 1
                        logger.info(
                            "Preserved AdamW momentum state for %d parameters across Stage 1 -> Stage 2",
                            transferred,
                        )

                    global_step, data_iter = self._run_stage(
                        stage_name="stage2_joint_sft",
                        optimizer=opt2,
                        start_step=global_step,
                        target_step=stage2_target,
                        data_iter=data_iter,
                        start_time=start_time,
                    )
        except KeyboardInterrupt:
            print(
                f"\n[{time.strftime('%X')}] ⏸️ Pelatihan/Evaluasi dihentikan pada step {global_step}. Menyimpan checkpoint otomatis...",
                flush=True,
            )
            if not already_completed:
                self.save_checkpoint(
                    global_step,
                    getattr(self, "current_stage", current_stage),
                    optimizer=getattr(self, "current_optimizer", None),
                )
            last_val = next(
                (h["val_loss"] for h in reversed(self.history) if h.get("val_loss") is not None),
                0.0,
            )
            safe_cfg_int = {
                k: ("***REDACTED***" if k == "hf_token" and v else v)
                for k, v in asdict(self.cfg).items()
            }
            interrupted_summary = {
                "status": "INTERRUPTED_SAVED",
                "model_choice": self.cfg.model_choice,
                "total_steps": global_step,
                "final_val_loss": float(last_val or 0.0),
                "export_dir": str(self.out_path / "checkpoint_latest"),
                "hub_repo": self.cfg.hub_repo,
                "hub_subfolder": self.cfg.hub_subfolder,
                "config": safe_cfg_int,
                "history": self.history[-10:],
                "completed_at": time.time(),
                "elapsed_time_sec": time.time() - start_time,
            }
            (self.out_path / "training_summary.json").write_text(
                json.dumps(interrupted_summary, indent=2), encoding="utf-8"
            )
            return interrupted_summary

        # -------------------------------------------------------------
        # Automatic Standalone PreTrainedModel Export
        # -------------------------------------------------------------
        cached_val = next(
            (h["val_loss"] for h in reversed(self.history) if h.get("val_loss") is not None),
            None,
        )
        final_val_loss = (
            float(cached_val) if (already_completed and cached_val is not None) else self.evaluate()
        )
        export_dir = self.out_path / "final_pretrained_model"
        if already_completed and (
            (export_dir / "model.safetensors").is_file() or self._is_remote_pretrained_uploaded()
        ):
            self.model.merge_and_unload()
            print(
                f"[{time.strftime('%X')}] ✅ Standalone model {self.cfg.model_choice} sudah tersimpan di Hugging Face Hub. Lanjut ke Benchmark Suite...",
                flush=True,
            )
        else:
            logger.info(
                "Training complete. Exporting standalone Hugging Face model to %s", export_dir
            )
            print(
                f"[{time.strftime('%X')}] 💾 Training complete. Exporting standalone Hugging Face model to {export_dir}...",
                flush=True,
            )
            self.export_pretrained(export_dir)

        # Save training summary (redact hf_token so Hugging Face secret scanner allows upload)
        safe_cfg = {
            k: ("***REDACTED***" if k == "hf_token" and v else v)
            for k, v in asdict(self.cfg).items()
        }
        summary: dict[str, Any] = {
            "status": "COMPLETED",
            "model_choice": self.cfg.model_choice,
            "total_steps": global_step,
            "final_val_loss": final_val_loss,
            "export_dir": str(export_dir),
            "hub_repo": self.cfg.hub_repo,
            "hub_subfolder": self.cfg.hub_subfolder,
            "config": safe_cfg,
            "history": self.history[-10:],
            "benchmarks": [],
            "completed_at": time.time(),
            "elapsed_time_sec": time.time() - start_time,
        }
        summary_file = self.out_path / "training_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # -------------------------------------------------------------
        # Automatic Benchmark Suite Execution
        # -------------------------------------------------------------
        if self.cfg.run_benchmarks and not self.cfg.tiny:
            try:
                benchmarks = self.run_benchmark_suite(summary_base=summary)
                summary["benchmarks"] = benchmarks
                summary["elapsed_time_sec"] = time.time() - start_time
                summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except KeyboardInterrupt:
                print(
                    f"\n[{time.strftime('%X')}] ⏸️ Benchmark dijeda. Semua benchmark yang sudah selesai tetap tersimpan dan otomatis dilanjutkan saat dijalankan ulang.",
                    flush=True,
                )
                summary["status"] = "BENCHMARK_PAUSED"
                summary["elapsed_time_sec"] = time.time() - start_time
                summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if self.cfg.push_to_hub and self.cfg.hub_repo and self.cfg.hf_token:
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=self.cfg.hf_token)
                sub = (self.cfg.hub_subfolder or f"fast-{self.cfg.model_choice}").strip("/")
                api.upload_file(
                    path_or_fileobj=str(summary_file),
                    path_in_repo=f"{sub}/training_summary.json",
                    repo_id=self.cfg.hub_repo,
                    repo_type="model",
                    commit_message=f"[{self.cfg.model_choice}] Upload training & benchmark summary (val_loss={final_val_loss:.4f})",
                )
            except Exception as _exc:
                logger.warning("Training summary upload skipped: %s", _exc)

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
    parser.add_argument("--hub-repo", type=str, default=DEFAULT_FAST_HF_REPO)
    parser.add_argument("--tiny", action="store_true", help="Run tiny smoke training for CI")
    parser.add_argument(
        "--no-benchmarks",
        action="store_true",
        help="Skip automatic benchmark suite execution after training",
    )
    parser.add_argument("--stage1-lambda-ptr", type=float, default=0.15)
    parser.add_argument("--gate-init-tanh", type=float, default=0.10)
    parser.add_argument("--no-stage1-encoder-lora", action="store_true")
    parser.add_argument("--no-curriculum", action="store_true")

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
        stage1_lambda_ptr=args.stage1_lambda_ptr,
        gate_init_tanh=args.gate_init_tanh,
        stage1_encoder_lora=not args.no_stage1_encoder_lora,
        use_comprehension_curriculum=not args.no_curriculum,
        tiny=args.tiny,
        run_benchmarks=not args.no_benchmarks,
    )
    pipeline = FastTrainingPipeline(cfg)
    summary = pipeline.run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run_fast_training())
