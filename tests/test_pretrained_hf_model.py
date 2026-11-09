"""Comprehensive test suite for standard Hugging Face PreTrainedModel BifocalGemma."""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
from bifocalgemma.modeling_bifocalgemma import (
    BifocalGemmaForCausalLM,
)
from bifocalgemma.training.memory_probe import ProbeLoRALinear


def test_bifocalgemma_config_initialization_and_serialization():
    config = BifocalGemmaConfig(
        encoder_size="E2B",
        decoder_size="E2B",
        tiny=True,
        init_rms_scale=1.1,
        gate_init_tanh=0.05,
    )
    assert config.model_type == "bifocalgemma"
    assert config.d_enc == 32
    assert config.d_dec == 32
    assert config.init_rms_scale == 1.1
    assert config.architectures == ["BifocalGemmaForCausalLM"]

    # Serialization
    cfg_dict = config.to_dict()
    assert cfg_dict["model_type"] == "bifocalgemma"
    assert cfg_dict["d_enc"] == 32
    assert cfg_dict["d_dec"] == 32

    reloaded = BifocalGemmaConfig.from_dict(cfg_dict)
    assert reloaded.d_enc == 32
    assert reloaded.init_rms_scale == 1.1


def test_bifocalgemma_for_causal_lm_forward_loss_and_generation():
    config = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    model = BifocalGemmaForCausalLM(config)

    assert model.encoder is not None
    assert model.decoder is not None
    assert model.encoder_projection is not None

    batch_size = 2
    enc_len = 8
    dec_len = 6
    vocab_size = config.vocab_size

    encoder_input_ids = torch.randint(10, 100, (batch_size, enc_len))
    decoder_input_ids = torch.randint(10, 100, (batch_size, dec_len))
    labels = decoder_input_ids.clone()

    # Forward with labels
    output = model(
        input_ids=decoder_input_ids,
        encoder_input_ids=encoder_input_ids,
        labels=labels,
    )
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert output.loss.item() > 0.0
    assert output.logits.shape == (batch_size, dec_len, vocab_size)

    # Autoregressive generation
    gen = model.generate(
        input_ids=decoder_input_ids[:, :2],
        encoder_input_ids=encoder_input_ids,
        max_new_tokens=4,
    )
    assert gen.shape == (batch_size, 6)


def test_bifocalgemma_merge_and_unload():
    config = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    model = BifocalGemmaForCausalLM(config)

    # Wrap a decoder projection with ProbeLoRALinear
    first_layer = model.decoder.layers[0]
    base_linear = first_layer.mlp.gate_proj
    lora_module = ProbeLoRALinear(base_linear, rank=4)
    # Give non-zero weights to lora_A and lora_B
    lora_module.lora_A.data.fill_(0.1)
    lora_module.lora_B.data.fill_(0.2)
    first_layer.mlp.gate_proj = lora_module

    w_before = base_linear.weight.clone()
    assert isinstance(first_layer.mlp.gate_proj, ProbeLoRALinear)

    # Merge and unload
    model.merge_and_unload()
    assert not isinstance(first_layer.mlp.gate_proj, ProbeLoRALinear)
    assert isinstance(first_layer.mlp.gate_proj, nn.Linear)

    w_after = first_layer.mlp.gate_proj.weight
    # Check that weights were actually changed by lora delta
    assert not torch.allclose(w_before, w_after)


def test_bifocalgemma_save_and_from_pretrained_roundtrip(tmp_path):
    config = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    model = BifocalGemmaForCausalLM(config)

    # Attach LoRA to verify automatic merge on save
    first_layer = model.decoder.layers[0]
    first_layer.mlp.gate_proj = ProbeLoRALinear(first_layer.mlp.gate_proj, rank=4)
    assert any(isinstance(m, ProbeLoRALinear) for m in model.modules())

    export_dir = tmp_path / "bifocalgemma_export"
    model.save_pretrained(export_dir)

    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "README.md").is_file()

    # Load back via from_pretrained
    reloaded = BifocalGemmaForCausalLM.from_pretrained(export_dir)
    assert isinstance(reloaded, BifocalGemmaForCausalLM)
    assert reloaded.config.d_enc == config.d_enc
    assert reloaded.config.d_dec == config.d_dec

    # Verify no LoRA modules remain in the reloaded pre-trained model
    assert not any(isinstance(m, ProbeLoRALinear) for m in reloaded.modules())

    # Verify numerical execution on reloaded model
    batch_size = 1
    enc_ids = torch.randint(10, 100, (batch_size, 4))
    dec_ids = torch.randint(10, 100, (batch_size, 4))
    out = reloaded(input_ids=dec_ids, encoder_input_ids=enc_ids)
    assert out.logits.shape == (batch_size, 4, config.vocab_size)

    # Test AutoModelForCausalLM loading from exported directory
    hf_model = AutoModelForCausalLM.from_pretrained(export_dir)
    assert isinstance(hf_model, BifocalGemmaForCausalLM)
    hf_out = hf_model(input_ids=dec_ids, encoder_input_ids=enc_ids)
    assert hf_out.logits.shape == (batch_size, 4, config.vocab_size)

    # Test tokenizer resolution if tokenizer files were saved
    if (export_dir / "tokenizer.json").is_file():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(export_dir)
        tokens = tokenizer.encode("def add(a, b): return a + b")
        assert len(tokens) > 0


def test_bifocalgemma_bin_serialization_and_dtype(tmp_path):
    config = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    model = BifocalGemmaForCausalLM(config)

    export_dir = tmp_path / "bifocalgemma_bin_export"
    model.save_pretrained(export_dir, safe_serialization=False)

    assert (export_dir / "pytorch_model.bin").is_file()
    assert (export_dir / "config.json").is_file()

    reloaded = BifocalGemmaForCausalLM.from_pretrained(
        export_dir, torch_dtype=torch.float32, device_map="cpu"
    )
    assert isinstance(reloaded, BifocalGemmaForCausalLM)
    assert next(reloaded.parameters()).dtype == torch.float32


def test_auto_classes_registration():
    config = AutoConfig.for_model("bifocalgemma", tiny=True)
    assert isinstance(config, BifocalGemmaConfig)


def test_from_pretrained_backbones_loading(tmp_path):
    from safetensors.torch import save_file

    config = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    base_model = BifocalGemmaForCausalLM(config)

    # Save mock backbone safetensors with official Gemma-4 prefix "model.language_model."
    dec_state = {
        f"model.language_model.{k}": v.clone()
        for k, v in base_model.decoder.state_dict().items()
    }
    enc_state = {
        f"model.language_model.{k}": v.clone()
        for k, v in base_model.encoder.state_dict().items()
    }

    dec_file = tmp_path / "dec_model.safetensors"
    enc_file = tmp_path / "enc_model.safetensors"
    save_file(dec_state, str(dec_file))
    save_file(enc_state, str(enc_file))

    # 1. Load from explicit backbone files
    model = BifocalGemmaForCausalLM.from_pretrained_backbones(
        config=config,
        decoder_path_or_id=dec_file,
        encoder_path_or_id=enc_file,
        device="cpu",
    )

    assert model.decoder is not None
    assert model.encoder is not None
    assert model.encoder_projection is not None
    assert model.decoder.embed_tokens.weight is model.lm_head.weight

    # Test forward and generation
    enc_ids = torch.randint(10, 100, (1, 6))
    dec_ids = torch.randint(10, 100, (1, 4))
    out = model(input_ids=dec_ids, encoder_input_ids=enc_ids)
    assert out.logits.shape == (1, 4, config.vocab_size)

    gen = model.generate(input_ids=dec_ids, encoder_input_ids=enc_ids, max_new_tokens=3)
    assert gen.shape == (1, 7)

    # 2. Load from backbone directories containing model.safetensors
    dec_dir = tmp_path / "mock_decoder_dir"
    enc_dir = tmp_path / "mock_encoder_dir"
    dec_dir.mkdir(parents=True, exist_ok=True)
    enc_dir.mkdir(parents=True, exist_ok=True)
    save_file(dec_state, str(dec_dir / "model.safetensors"))
    save_file(enc_state, str(enc_dir / "model.safetensors"))

    model_from_dirs = BifocalGemmaForCausalLM.from_pretrained_backbones(
        config=config,
        decoder_path_or_id=dec_dir,
        encoder_path_or_id=enc_dir,
        device="cpu",
    )
    assert model_from_dirs.decoder is not None
    assert model_from_dirs.encoder is not None
    out_dirs = model_from_dirs(input_ids=dec_ids, encoder_input_ids=enc_ids)
    assert out_dirs.logits.shape == (1, 4, config.vocab_size)

    # 3. Load from pretrained_backbone_dir containing decoder/ and encoder/ subdirectories
    backbone_unified = tmp_path / "backbone_unified"
    (backbone_unified / "decoder").mkdir(parents=True, exist_ok=True)
    (backbone_unified / "encoder").mkdir(parents=True, exist_ok=True)
    save_file(dec_state, str(backbone_unified / "decoder/model.safetensors"))
    save_file(enc_state, str(backbone_unified / "encoder/model.safetensors"))

    model_from_unified = BifocalGemmaForCausalLM.from_pretrained_backbones(
        config=config,
        pretrained_backbone_dir=backbone_unified,
        device="cpu",
    )
    assert model_from_unified.decoder is not None
    assert model_from_unified.encoder is not None
    out_unified = model_from_unified(input_ids=dec_ids, encoder_input_ids=enc_ids)
    assert out_unified.logits.shape == (1, 4, config.vocab_size)


def test_fast_training_pipeline_with_jsonl_data_dir(tmp_path):
    import json
    from pathlib import Path

    from bifocalgemma.contracts import Seq2SeqBatch
    from bifocalgemma.training.fast_pipeline import (
        FastTrainingConfig,
        FastTrainingPipeline,
        JSONLRepairDataset,
    )

    # 1. Create realistic train and validation JSONL files
    data_dir = tmp_path / "repair_data"
    data_dir.mkdir(parents=True)
    train_records = [
        {
            "instance_id": f"repo-train-{i}",
            "instruction": f"Fix bug #{i} in repository module",
            "path": f"src/module_{i}.py",
            "before": f"def foo_{i}(x):\n    return x - 1\n",
            "target_patch": f"def foo_{i}(x):\n    return x + 1\n",
        }
        for i in range(8)
    ]
    val_records = [
        {
            "instance_id": f"repo-val-{i}",
            "instruction": f"Fix validation bug #{i}",
            "path": f"src/val_{i}.py",
            "before": f"def val_func_{i}():\n    pass\n",
            "target_patch": f"def val_func_{i}():\n    return True\n",
        }
        for i in range(4)
    ]

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    train_file.write_text("\n".join(json.dumps(r) for r in train_records), encoding="utf-8")
    val_file.write_text("\n".join(json.dumps(r) for r in val_records), encoding="utf-8")

    # 2. Test JSONLRepairDataset directly and verify neutral prompt guarantees
    ds = JSONLRepairDataset(train_file, is_baseline=False)
    assert len(ds) == 8
    ex = ds[0]
    assert ex.instance_id == "repo-train-0"
    assert len(ex.encoder_ids) > 0
    assert len(ex.decoder_prefix_ids) > 0

    # Ensure zero occurrence of prompt leakage phrase in paired and baseline datasets
    baseline_ds = JSONLRepairDataset(train_file, is_baseline=True)
    assert len(baseline_ds) == 8

    # 3. Configure and run fast training pipeline with real data_dir
    out_dir = tmp_path / "fast_pipeline_output"
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        total_steps=2,
        stage1_steps=1,
        stage2_steps=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        data_dir=str(data_dir),
        output_dir=str(out_dir),
        tiny=True,
    )
    pipeline = FastTrainingPipeline(cfg)

    # Verify prompt neutrality with pipeline tokenizer
    tok_ds = JSONLRepairDataset(train_file, tokenizer=pipeline.tokenizer, is_baseline=False)
    tok_base_ds = JSONLRepairDataset(train_file, tokenizer=pipeline.tokenizer, is_baseline=True)
    if pipeline.tokenizer is not None:
        paired_text = pipeline.tokenizer.decode(
            list(tok_ds[0].decoder_prefix_ids), skip_special_tokens=False
        )
        assert "using the encoder context" not in paired_text.lower(), (
            f"Prompt leakage detected in paired decoder prefix: {paired_text!r}"
        )
        assert "Return only a unified diff fulfilling the instruction." in paired_text

        base_text = pipeline.tokenizer.decode(
            list(tok_base_ds[0].decoder_prefix_ids), skip_special_tokens=False
        )
        assert "using the encoder context" not in base_text.lower()

    # Verify dataloaders yield Seq2SeqBatch
    first_batch = next(iter(pipeline.train_loader))
    assert isinstance(first_batch, Seq2SeqBatch)
    assert first_batch.decoder_input_ids.shape[0] == 2

    # Run pipeline through Stage 1, Stage 2, and automatic standalone model export
    summary = pipeline.run()
    assert summary["total_steps"] == 2
    assert summary["final_val_loss"] > 0.0

    export_dir = Path(summary["export_dir"])
    assert export_dir.name == "final_pretrained_model"
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "generation_config.json").is_file()
    assert (export_dir / "configuration_bifocalgemma.py").is_file()
    assert (export_dir / "modeling_bifocalgemma.py").is_file()
    assert (export_dir / "README.md").is_file()

    # Verify exported model loads back cleanly
    loaded = BifocalGemmaForCausalLM.from_pretrained(export_dir)
    assert isinstance(loaded, BifocalGemmaForCausalLM)
    gen = loaded.generate(
        input_ids=torch.tensor([[2, 10, 11]]),
        encoder_input_ids=torch.tensor([[10, 11, 12]]),
        max_new_tokens=2,
    )
    assert gen.shape == (1, 5)


def test_export_matrix_pretrained_model(tmp_path):
    from bifocalgemma.modeling_bifocalgemma import (
        BifocalGemmaForCausalLM,
        export_matrix_pretrained_model,
    )
    from bifocalgemma.models.v2.native import build_native_tiny_model
    from bifocalgemma.training.memory_probe import configure_probe_training
    from bifocalgemma.v3.attention import configure_full_rank_cross_attention
    from bifocalgemma.v3.matrix import make_experiment_spec

    # Build NativeTinyModel matching matrix runner
    spec = make_experiment_spec("2base-2it")
    model = build_native_tiny_model(spec)
    configure_full_rank_cross_attention(model)

    # Attach LoRA adapters to test automatic merge_and_unload
    configure_probe_training(model, policy="lora", rank=4)

    export_dir = tmp_path / "matrix_pretrained_export"
    exported_path = export_matrix_pretrained_model(
        model=model,
        spec=spec,
        export_dir=export_dir,
    )
    assert exported_path == export_dir
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "generation_config.json").is_file()
    assert (export_dir / "configuration_bifocalgemma.py").is_file()
    assert (export_dir / "modeling_bifocalgemma.py").is_file()
    assert (export_dir / "README.md").is_file()

    # Load back using standard Hugging Face PreTrainedModel
    loaded = BifocalGemmaForCausalLM.from_pretrained(export_dir)
    assert isinstance(loaded, BifocalGemmaForCausalLM)

    enc_ids = torch.randint(10, 100, (1, 6))
    dec_ids = torch.randint(10, 100, (1, 4))
    out = loaded(input_ids=dec_ids, encoder_input_ids=enc_ids)
    assert out.logits.shape == (1, 4, loaded.config.vocab_size)

    # Test AutoModelForCausalLM loading with trust_remote_code=True
    hf_loaded = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)
    assert isinstance(hf_loaded, BifocalGemmaForCausalLM)
    hf_out = hf_loaded(input_ids=dec_ids, encoder_input_ids=enc_ids)
    assert hf_out.logits.shape == (1, 4, hf_loaded.config.vocab_size)


def test_export_matrix_pretrained_model_baseline(tmp_path):
    from bifocalgemma.modeling_bifocalgemma import (
        BifocalGemmaForCausalLM,
        export_matrix_pretrained_model,
    )
    from bifocalgemma.models.v2.native import build_native_tiny_model
    from bifocalgemma.training.memory_probe import configure_probe_training
    from bifocalgemma.v3.matrix import make_experiment_spec

    # Build NativeTinyModel for baseline (no encoder)
    spec = make_experiment_spec("e2b-it")
    model = build_native_tiny_model(spec)
    configure_probe_training(model, policy="lora", rank=4)

    export_dir = tmp_path / "baseline_matrix_export"
    exported_path = export_matrix_pretrained_model(
        model=model,
        spec=spec,
        export_dir=export_dir,
    )
    assert exported_path == export_dir
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "generation_config.json").is_file()
    assert (export_dir / "configuration_bifocalgemma.py").is_file()
    assert (export_dir / "modeling_bifocalgemma.py").is_file()
    assert (export_dir / "README.md").is_file()

    # Load back using standard Hugging Face PreTrainedModel
    loaded = BifocalGemmaForCausalLM.from_pretrained(export_dir)
    assert isinstance(loaded, BifocalGemmaForCausalLM)
    assert loaded.encoder is None
    dec_ids = torch.randint(10, 100, (1, 4))
    out = loaded(input_ids=dec_ids)
    assert out.logits.shape == (1, 4, loaded.config.vocab_size)


def test_from_pretrained_backbones_baseline(tmp_path):
    from safetensors.torch import save_file

    config = BifocalGemmaConfig(encoder_size="none", decoder_size="E2B", tiny=True)
    base_model = BifocalGemmaForCausalLM(config)
    assert base_model.encoder is None

    dec_state = {
        f"model.language_model.{k}": v.clone()
        for k, v in base_model.decoder.state_dict().items()
    }
    backbone_dir = tmp_path / "baseline_backbone_dir"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    save_file(dec_state, str(backbone_dir / "model.safetensors"))

    model = BifocalGemmaForCausalLM.from_pretrained_backbones(
        config=config,
        decoder_path_or_id=backbone_dir,
        device="cpu",
    )
    assert model.decoder is not None
    assert model.encoder is None
    dec_ids = torch.randint(10, 100, (1, 4))
    out = model(input_ids=dec_ids)
    assert out.logits.shape == (1, 4, config.vocab_size)


def test_fast_training_pipeline_baseline_2b(tmp_path):
    from bifocalgemma.training.fast_pipeline import (
        FastTrainingConfig,
        FastTrainingPipeline,
    )

    out_dir = tmp_path / "baseline_fast_output"
    cfg = FastTrainingConfig(
        model_choice="baseline-2b",
        total_steps=4,
        stage1_steps=2,
        stage2_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        output_dir=str(out_dir),
        tiny=True,
    )
    pipeline = FastTrainingPipeline(cfg)
    assert pipeline.model.encoder is None
    summary = pipeline.run()

    assert summary["model_choice"] == "baseline-2b"
    assert summary["total_steps"] == 4
    assert len(summary["history"]) > 0

    export_dir = out_dir / "final_pretrained_model"
    assert (export_dir / "model.safetensors").is_file()
    assert (export_dir / "config.json").is_file()

    # Load back using standard Hugging Face PreTrainedModel
    loaded = BifocalGemmaForCausalLM.from_pretrained(export_dir)
    assert isinstance(loaded, BifocalGemmaForCausalLM)
    assert loaded.encoder is None
    assert loaded.decoder is not None
    dec_ids = torch.randint(10, 100, (1, 4))
    out = loaded(input_ids=dec_ids)
    assert out.logits.shape == (1, 4, loaded.config.vocab_size)


def test_train_fast_cli_baseline_2b(tmp_path):
    from bifocalgemma.__main__ import main

    out_dir = tmp_path / "cli_fast_baseline"
    exit_code = main(
        [
            "train-fast",
            "--model",
            "baseline-2b",
            "--steps",
            "2",
            "--tiny",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "final_pretrained_model" / "model.safetensors").is_file()
    assert (out_dir / "training_summary.json").is_file()


def test_enable_memory_efficient_training_and_softcapping():
    from bifocalgemma.models.memory_efficient import TiledEncoderAttention

    # Test softcapping in decode_logits
    cfg = BifocalGemmaConfig(encoder_size="none", decoder_size="E2B", tiny=True)
    cfg.decoder_config["final_logit_softcapping"] = 30.0
    model = BifocalGemmaForCausalLM(cfg)
    assert model.decoder.config.final_logit_softcapping == 30.0

    dec_ids = torch.randint(10, 100, (2, 4))
    logits = model.decode_logits(dec_ids, None, None, None)
    # Logits capped by tanh * 30.0 must be in [-30, 30]
    assert logits.abs().max().item() <= 30.0 + 1e-4

    # Test enable_memory_efficient_training for baseline (decoder-only)
    model.enable_memory_efficient_training(
        query_chunk_size=64, loss_chunk_size=64, gradient_checkpointing=False
    )
    assert isinstance(model.decoder.layers[0].self_attn, TiledEncoderAttention)
    assert model.decoder.layers[0].self_attn.query_chunk_size == 64
    assert model.loss_chunk_size == 64
    assert not model.decoder.config.use_cache

    # Test enable_memory_efficient_training for paired model
    paired_cfg = BifocalGemmaConfig(encoder_size="E2B", decoder_size="E2B", tiny=True)
    paired_model = BifocalGemmaForCausalLM(paired_cfg)
    paired_model.enable_memory_efficient_training(
        query_chunk_size=64, loss_chunk_size=64, gradient_checkpointing=False
    )
    assert isinstance(paired_model.encoder.layers[0].self_attn, TiledEncoderAttention)
    assert paired_model.loss_chunk_size == 64
    assert not paired_model.encoder.config.use_cache
    assert not paired_model.decoder.config.use_cache


def test_fast_pipeline_batch_edge_cases(tmp_path):
    import pytest

    from bifocalgemma.contracts import Seq2SeqBatch
    from bifocalgemma.training.fast_pipeline import (
        FastTrainingConfig,
        FastTrainingPipeline,
    )

    cfg = FastTrainingConfig(
        model_choice="baseline-2b",
        total_steps=2,
        stage1_steps=1,
        stage2_steps=1,
        tiny=True,
    )
    pipeline = FastTrainingPipeline(cfg)

    # 1. Calling configure_stage1 on baseline model must raise clear ValueError
    with pytest.raises(ValueError, match="Stage 1 Bridge Alignment requires an encoder"):
        pipeline.configure_stage1()

    # 2. train_step with dict missing decoder_attention_mask should fall back gracefully
    dict_batch = {
        "decoder_input_ids": torch.randint(10, 100, (2, 8)),
        "labels": torch.randint(10, 100, (2, 8)),
    }
    loss = pipeline.train_step(dict_batch, stage_name="stage2")
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

    # 3. train_step on baseline model with a Seq2SeqBatch containing encoder tokens strips them
    seq_batch = Seq2SeqBatch(
        encoder_input_ids=torch.randint(10, 100, (2, 16)),
        encoder_attention_mask=torch.ones((2, 16), dtype=torch.bool),
        decoder_input_ids=torch.randint(10, 100, (2, 8)),
        decoder_attention_mask=torch.ones((2, 8), dtype=torch.bool),
        labels=torch.randint(10, 100, (2, 8)),
    )
    loss_seq = pipeline.train_step(seq_batch, stage_name="stage2")
    assert isinstance(loss_seq, torch.Tensor)
    assert loss_seq.ndim == 0
    assert not torch.isnan(loss_seq)


def test_fast_training_config_step_reconciliation():
    from bifocalgemma.training.fast_pipeline import FastTrainingConfig

    # 1. Setting total_steps overrides default stage steps
    cfg = FastTrainingConfig(total_steps=500)
    assert cfg.stage1_steps == 250
    assert cfg.stage2_steps == 250
    assert cfg.total_steps == 500

    # 2. Odd total_steps partitions without losing a step
    cfg_odd = FastTrainingConfig(total_steps=5)
    assert cfg_odd.stage1_steps == 2
    assert cfg_odd.stage2_steps == 3
    assert cfg_odd.total_steps == 5

    # 3. Explicit stage1 and stage2 steps update total_steps
    cfg_explicit = FastTrainingConfig(stage1_steps=150, stage2_steps=350)
    assert cfg_explicit.total_steps == 500


def test_fast_pipeline_skips_training_when_already_completed(tmp_path, monkeypatch):
    from bifocalgemma.training.fast_pipeline import (
        FastTrainingConfig,
        FastTrainingPipeline,
    )

    for model_choice in ("bifocalgemma-2b", "baseline-2b"):
        out_dir = tmp_path / f"completed_{model_choice}"
        cfg = FastTrainingConfig(
            model_choice=model_choice,
            total_steps=4,
            output_dir=str(out_dir),
            tiny=True,
        )
        p1 = FastTrainingPipeline(cfg)
        p1.save_checkpoint(4, "completed")

        p2 = FastTrainingPipeline(cfg)
        monkeypatch.setattr(p2, "try_resume_checkpoint", lambda: 4)
        called_steps = []
        monkeypatch.setattr(
            p2,
            "train_step",
            lambda batch, stage_name: called_steps.append(stage_name)
            or torch.tensor(0.5, requires_grad=True),
        )
        summary = p2.run()
        assert summary["status"] == "COMPLETED"
        assert summary["total_steps"] == 4
        assert "stage1" not in called_steps
        assert "stage2" not in called_steps
        assert "full_decoder_sft" not in called_steps
