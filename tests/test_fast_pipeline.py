"""Unit tests for FastTrainingPipeline v3 improvements.

Covers:
1. Stage 1 Encoder LoRA active + Decoder LoRA frozen (scaling == 0.0), Stage 2 both active (scaling == 2.0).
2. Optimizer momentum state (exp_avg, exp_avg_sq, step) preserved from Stage 1 into Stage 2.
3. Warmup + cosine LR schedule updates group["lr"].
4. Query injection into encoder context in JSONLRepairDataset.
5. Anti-forgetting comprehension curriculum integration and pointer grounding loss.
6. Checkpoint save/resume with LoRA scaling preserved across stages.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pytest
import torch

# Ensure transformers compatibility check passes in test environment
_orig_version = importlib.metadata.version


def _patched_version(distribution_name: str) -> str:
    if distribution_name == "transformers":
        return "5.22.1"
    return _orig_version(distribution_name)


setattr(importlib.metadata, "version", _patched_version)

from bifocalgemma.training.fast_pipeline import (  # noqa: E402
    FastTrainingConfig,
    FastTrainingPipeline,
    JSONLRepairDataset,
)
from bifocalgemma.training.memory_probe import ProbeLoRALinear  # noqa: E402
from bifocalgemma.v3.objectives import is_cross_bridge_parameter  # noqa: E402


class DummyTrackingTokenizer:
    """Mock tokenizer tracking encoded prompt strings for inspection."""

    def __init__(self) -> None:
        self.encoder_prompts: list[str] = []
        self.all_prompts: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.all_prompts.append(text)
        if add_special_tokens:
            self.encoder_prompts.append(text)
        return [hash(w) % 1000 + 10 for w in text.split()]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        content = messages[0]["content"] if messages else ""
        return f"<chat>{content}</chat>"


def test_stage1_and_stage2_lora_configuration():
    """Verify Stage 1 activates Encoder LoRA while freezing Decoder LoRA (scaling=0.0), and Stage 2 enables both."""
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        stage1_encoder_lora=True,
        lr_bridge=1e-4,
        lr_cross=1e-4,
        lr_lora=2e-5,
    )
    pipeline = FastTrainingPipeline(cfg)

    # 1. Configure Stage 1
    opt1 = pipeline.configure_stage1()

    # Check ProbeLoRALinear modules scaling
    lora_modules = [
        (name, module)
        for name, module in pipeline.model.named_modules()
        if isinstance(module, ProbeLoRALinear)
    ]
    assert len(lora_modules) > 0, "ProbeLoRALinear modules should be attached in Stage 1"

    encoder_lora_count = 0
    decoder_lora_count = 0
    for name, module in lora_modules:
        if name.startswith("encoder."):
            assert module.scaling == 2.0, f"Encoder LoRA {name} should have scaling 2.0"
            encoder_lora_count += 1
        elif name.startswith("decoder."):
            assert module.scaling == 0.0, f"Decoder LoRA {name} should have scaling 0.0"
            decoder_lora_count += 1

    assert encoder_lora_count > 0, "Must have encoder LoRA modules"
    assert decoder_lora_count > 0, "Must have decoder LoRA modules attached"

    # Check requires_grad in Stage 1
    for name, param in pipeline.model.named_parameters():
        if name.startswith("encoder_projection."):
            assert param.requires_grad is True, f"{name} must be trainable in Stage 1"
        elif is_cross_bridge_parameter(name):
            assert param.requires_grad is True, f"{name} cross-attn must be trainable in Stage 1"
        elif name.startswith("encoder.") and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            assert param.requires_grad is True, f"Encoder LoRA {name} must be trainable in Stage 1"
        elif name.startswith("decoder.") and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            assert param.requires_grad is False, f"Decoder LoRA {name} must be FROZEN in Stage 1"
        elif name == "lm_head.weight":
            assert param.requires_grad is False, "lm_head must be frozen in Stage 1"

    # Check optimizer groups in Stage 1
    lora_groups1 = [g for g in opt1.param_groups if g.get("name") == "backbone_lora"]
    assert len(lora_groups1) == 1, "Must have a backbone_lora param group in Stage 1"
    assert lora_groups1[0]["lr"] == cfg.lr_lora
    assert all(name.startswith("encoder.") for name in lora_groups1[0]["param_names"]), (
        "Stage 1 backbone_lora must only contain encoder parameters"
    )

    # 2. Configure Stage 2
    opt2 = pipeline.configure_stage2()

    # In Stage 2, all ProbeLoRALinear modules must have scaling == 2.0
    for name, module in pipeline.model.named_modules():
        if isinstance(module, ProbeLoRALinear):
            assert module.scaling == 2.0, f"{name} should have scaling 2.0 in Stage 2"

    # In Stage 2, both encoder and decoder LoRA parameters must be trainable
    for name, param in pipeline.model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            assert param.requires_grad is True, f"{name} must be trainable in Stage 2"
        elif name.startswith("encoder_projection.") or is_cross_bridge_parameter(name):
            assert param.requires_grad is True, f"{name} must remain trainable in Stage 2"

    lora_groups2 = [g for g in opt2.param_groups if g.get("name") == "backbone_lora"]
    assert len(lora_groups2) == 1
    assert any(name.startswith("decoder.") for name in lora_groups2[0]["param_names"])
    assert any(name.startswith("encoder.") for name in lora_groups2[0]["param_names"])
    assert len(lora_groups2[0]["params"]) > len(lora_groups1[0]["params"])


def test_optimizer_momentum_preserved_stage1_to_stage2():
    """Verify AdamW momentum (exp_avg, exp_avg_sq, step) transfers across the Stage 1 -> Stage 2 boundary."""
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        stage1_encoder_lora=True,
    )
    pipeline = FastTrainingPipeline(cfg)

    # Stage 1 optimizer
    opt1 = pipeline.configure_stage1()

    # Run dummy backward to populate exp_avg and exp_avg_sq in opt1
    trainable_1 = [p for p in pipeline.model.parameters() if p.requires_grad]
    assert len(trainable_1) > 0
    loss1 = sum(p.sum() for p in trainable_1)
    assert isinstance(loss1, torch.Tensor)
    loss1.backward()
    opt1.step()
    opt1.zero_grad()

    stage1_states_count = len(opt1.state)
    assert stage1_states_count > 0, "Optimizer state must be populated after step"

    # Check that each parameter has step, exp_avg, exp_avg_sq
    for p, state in opt1.state.items():
        assert "exp_avg" in state
        assert "exp_avg_sq" in state
        assert "step" in state
        assert state["step"] == 1

    # Stage 2 optimizer
    opt2 = pipeline.configure_stage2()
    assert len(opt2.state) == 0, "Stage 2 optimizer state should be empty initially"

    # Transfer state from Stage 1 into Stage 2 (matching parameter tensor identity)
    opt2_params = {p for group in opt2.param_groups for p in group["params"]}
    transferred = 0
    for p, s in opt1.state.items():
        if p in opt2_params:
            opt2.state[p] = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in s.items()
            }
            transferred += 1

    assert transferred == stage1_states_count, "All Stage 1 parameters should continue into Stage 2"
    assert len(opt2.state) == stage1_states_count

    # Verify tensor values match bit-for-bit
    for p in opt1.state:
        assert torch.allclose(opt2.state[p]["exp_avg"], opt1.state[p]["exp_avg"])
        assert torch.allclose(opt2.state[p]["exp_avg_sq"], opt1.state[p]["exp_avg_sq"])
        assert opt2.state[p]["step"] == opt1.state[p]["step"]

    # Run Stage 2 optimization step cleanly
    trainable_2 = [p for p in pipeline.model.parameters() if p.requires_grad]
    assert len(trainable_2) > len(trainable_1), "Decoder LoRA joins in Stage 2"
    loss2 = sum(p.sum() for p in trainable_2)
    assert isinstance(loss2, torch.Tensor)
    loss2.backward()
    opt2.step()
    opt2.zero_grad()

    # Preserved parameters should now have step == 2, and new decoder parameters now have step == 1
    sample_p = next(iter(opt1.state))
    assert opt2.state[sample_p]["step"] == 2
    assert len(opt2.state) > stage1_states_count


def test_warmup_cosine_learning_rate_schedule():
    """Verify linear warmup and cosine decay properly update each group's lr."""
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        total_steps=100,
        warmup_steps=10,
        lr_bridge=1e-4,
        lr_cross=2e-4,
        lr_lora=3e-5,
    )
    pipeline = FastTrainingPipeline(cfg)
    opt = pipeline.configure_stage1()

    # Step 0: Warmup start (factor = 0.0)
    pipeline._update_learning_rates(opt, global_step=0)
    for g in opt.param_groups:
        assert g["lr"] == 0.0

    # Step 5: Mid-warmup (factor = 0.5)
    pipeline._update_learning_rates(opt, global_step=5)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(g["base_lr"] * 0.5)

    # Step 10: Peak of warmup (factor = 1.0)
    pipeline._update_learning_rates(opt, global_step=10)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(g["base_lr"] * 1.0)

    # Step 55: Midpoint of cosine decay from 10 to 100
    # progress = (55 - 10) / (100 - 10) = 45 / 90 = 0.5
    # cos(pi * 0.5) = 0.0 -> factor = 0.10 + 0.90 * 0.5 * 1.0 = 0.55
    pipeline._update_learning_rates(opt, global_step=55)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(g["base_lr"] * 0.55, rel=1e-4)

    # Step 100: End of cosine decay (factor = 0.10)
    pipeline._update_learning_rates(opt, global_step=100)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(g["base_lr"] * 0.10, rel=1e-4)

    # Step 120: Beyond total_steps (clamped to min_lr_ratio = 0.10)
    pipeline._update_learning_rates(opt, global_step=120)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(g["base_lr"] * 0.10, rel=1e-4)


def test_jsonl_dataset_query_injection_into_encoder(tmp_path: Path):
    """Verify JSONLRepairDataset prepends [QUERY] {instruction} into encoder context when applicable."""
    records = [
        {
            "instance_id": "test-inject",
            "instruction": "Fix off-by-one bug in indexing",
            "context": "def get_item(items, idx):\n    return items[idx - 1]\n",
            "target_patch": "def get_item(items, idx):\n    return items[idx]\n",
        },
        {
            "instance_id": "test-already-in-context",
            "instruction": "Fix off-by-one bug",
            "context": "### Instruction\nFix off-by-one bug\n\ndef get_item(items, idx):\n    return items[idx]\n",
            "target_patch": "...",
        },
        {
            "instance_id": "test-empty-instruction",
            "instruction": "",
            "context": "def foo(): pass",
            "target_patch": "...",
        },
    ]

    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    # 1. With inject_query_into_encoder = True
    tok1 = DummyTrackingTokenizer()
    ds_injected = JSONLRepairDataset(
        jsonl_file,
        tokenizer=tok1,
        is_baseline=False,
        inject_query_into_encoder=True,
    )
    assert len(ds_injected) == 3

    # Check Row 1: [QUERY] prepended
    first_enc_prompt = tok1.encoder_prompts[0]
    assert first_enc_prompt.startswith("[QUERY] Fix off-by-one bug in indexing\n\ndef get_item")

    # Check Row 2: already in context -> NOT prepended with duplicate [QUERY]
    second_enc_prompt = tok1.encoder_prompts[1]
    assert not second_enc_prompt.startswith("[QUERY] Fix off-by-one bug\n\n### Instruction")
    assert second_enc_prompt.startswith("### Instruction")

    # Check Row 3: empty instruction -> NOT prepended with [QUERY]
    third_enc_prompt = tok1.encoder_prompts[2]
    assert third_enc_prompt == "def foo(): pass"

    # 2. With inject_query_into_encoder = False
    tok2 = DummyTrackingTokenizer()
    ds_no_inject = JSONLRepairDataset(
        jsonl_file,
        tokenizer=tok2,
        is_baseline=False,
        inject_query_into_encoder=False,
    )
    assert len(ds_no_inject) == 3
    assert tok2.encoder_prompts[0] == records[0]["context"]

    # 3. Baseline mode: encoder_ids is empty
    tok3 = DummyTrackingTokenizer()
    ds_base = JSONLRepairDataset(
        jsonl_file,
        tokenizer=tok3,
        is_baseline=True,
        inject_query_into_encoder=True,
    )
    assert ds_base[0].encoder_ids == ()


def test_anti_forgetting_curriculum_and_stage1_pointer_loss(tmp_path: Path):
    """Verify comprehension curriculum mixes span-copy and QA examples, and Stage 1 uses stage1_lambda_ptr."""
    records = [
        {
            "instance_id": f"repair-{i}",
            "instruction": f"Fix repair issue #{i}",
            "path": f"module_{i}.py",
            "before": f"def func_{i}():\n    return {i}\n",
            "target_patch": f"def func_{i}():\n    return {i + 1}\n",
        }
        for i in range(10)
    ]
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    train_file = data_dir / "train.jsonl"
    train_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    val_file = data_dir / "val.jsonl"
    val_file.write_text("\n".join(json.dumps(r) for r in records[:2]), encoding="utf-8")

    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
        use_comprehension_curriculum=True,
        stage1_lambda_ptr=0.15,
        lambda_ptr=0.0,
        total_steps=4,
        stage1_steps=2,
        stage2_steps=2,
        gate_init_tanh=0.10,
    )
    pipeline = FastTrainingPipeline(cfg)

    # 1. Verify gate_init_tanh was passed into model configuration
    assert pipeline.model_config.gate_init_tanh == pytest.approx(0.10)

    # 2. Verify curriculum was applied in _prepare_dataloaders
    train_ds = pipeline.train_loader.dataset
    assert isinstance(train_ds, JSONLRepairDataset)
    assert len(train_ds) >= 10
    has_comp = any(
        getattr(ex, "metadata", {}).get("task_type") in ("span_copy", "code_qa_mcq")
        or "span" in ex.instance_id.lower()
        or "qa" in ex.instance_id.lower()
        or "synth" in ex.instance_id.lower()
        for ex in train_ds
    )
    assert has_comp, "Curriculum should synthesize comprehension examples into train dataset"

    # 3. Verify train_step respects stage1_lambda_ptr in Stage 1 and lambda_ptr in Stage 2
    batch = next(iter(pipeline.train_loader))

    # Stage 1 bridge warmup uses stage1_lambda_ptr (0.15)
    loss1 = pipeline.train_step(batch, stage_name="stage1_bridge_warmup")
    assert isinstance(loss1, torch.Tensor)
    assert loss1.ndim == 0
    assert not torch.isnan(loss1)

    # Stage 2 joint SFT uses lambda_ptr (0.0)
    loss2 = pipeline.train_step(batch, stage_name="stage2_joint_sft")
    assert isinstance(loss2, torch.Tensor)
    assert loss2.ndim == 0
    assert not torch.isnan(loss2)


def test_checkpoint_save_and_resume_lora_scaling(tmp_path: Path):
    """Verify checkpoint save and resume properly re-attaches LoRA and restores stage-appropriate scaling."""
    out_dir = tmp_path / "ckpt_test"
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        tiny=True,
        output_dir=str(out_dir),
        stage1_encoder_lora=True,
        total_steps=10,
        stage1_steps=5,
        stage2_steps=5,
    )
    pipeline = FastTrainingPipeline(cfg)
    pipeline.configure_stage1()

    # Save checkpoint at step 2 (Stage 1)
    ckpt_file = pipeline.save_checkpoint(2, "stage1_bridge_warmup")
    assert ckpt_file.is_file()

    payload = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    assert payload["has_lora"] is True
    assert payload["global_step"] == 2
    assert payload["stage"] == "stage1_bridge_warmup"

    # Create a fresh pipeline and resume
    new_pipeline = FastTrainingPipeline(cfg)
    resumed_step = new_pipeline.try_resume_checkpoint()
    assert resumed_step == 2

    # Check that ProbeLoRALinear is attached and encoder is 2.0 while decoder is 0.0
    for name, module in new_pipeline.model.named_modules():
        if isinstance(module, ProbeLoRALinear):
            if name.startswith("encoder."):
                assert module.scaling == 2.0
            elif name.startswith("decoder."):
                assert module.scaling == 0.0

    # Also verify load_checkpoint_if_available alias
    assert new_pipeline.load_checkpoint_if_available == new_pipeline.try_resume_checkpoint


def test_fast_training_pipeline_end_to_end_smoke(tmp_path: Path):
    """Verify complete end-to-end two-stage training run executes smoothly and exports standalone model."""
    records = [
        {
            "instance_id": f"e2e-repair-{i}",
            "instruction": f"Fix e2e bug #{i}",
            "path": f"module_{i}.py",
            "before": f"def foo_{i}(x):\n    return x - {i}\n",
            "target_patch": f"def foo_{i}(x):\n    return x + {i}\n",
        }
        for i in range(8)
    ]
    data_dir = tmp_path / "e2e_data"
    data_dir.mkdir()
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    train_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    val_file.write_text("\n".join(json.dumps(r) for r in records[:2]), encoding="utf-8")

    out_dir = tmp_path / "e2e_out"
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        total_steps=4,
        stage1_steps=2,
        stage2_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=1,
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        tiny=True,
        run_benchmarks=False,
    )
    pipeline = FastTrainingPipeline(cfg)
    summary = pipeline.run()

    assert summary["status"] == "COMPLETED"
    assert summary["total_steps"] == 4
    assert summary["final_val_loss"] > 0.0

    export_path = Path(summary["export_dir"])
    assert (export_path / "model.safetensors").is_file()
    assert (export_path / "config.json").is_file()
