# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo==0.25.2", "torch==2.15.0", "torchvision==0.29.2",
#     "transformers==5.22.1", "huggingface-hub==2.4.0", "safetensors==0.8.0",
#     "tokenizers==0.23.3", "pyarrow==26.0.0", "numpy==2.5.4", "pytest==9.2.0",
#     "psutil==7.2.2", "PyYAML>=6.0",
# ]
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu132" }]
# torchvision = [{ index = "pytorch-cu132" }]
# [[tool.uv.index]]
# name = "pytorch-cu132"
# url = "https://download.pytorch.org/whl/cu132"
# explicit = true
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # BifocalGemma · Fast From-Scratch Training Pipeline

        Pipeline pelatihan terfokus dari awal (from-scratch) untuk **BifocalGemma-2B** (Encoder 32k → Causal Decoder 8k) dan **Baseline-2B**.

        ### Keunggulan Arsitektur & Training:
        1. **Eliminasi Copy Bias**: Menonaktifkan Streaming Pointer Loss ($\lambda_{\text{ptr}} = 0.0$) demi Cross-Entropy bersih selaras evaluasi.
        2. **Learnable Sharpness Scaling**: Memperbaiki attention dilution pada cross-attention panjang.
        3. **BifocalGemma-2B (2 Tahap)**: Stage 1 Bridge Manifold Alignment (decoder dibekukan) + Stage 2 Joint LoRA Fine-Tuning ($2 \times 10^{-5}$).
        4. **Baseline-2B (1 Tahap Penuh)**: Full Decoder SFT langsung dari awal hingga akhir tanpa pembagian tahap.
        5. **Ekspor Hugging Face Utuh Otomatis**: Penggabungan bobot LoRA (`merge_and_unload`) langsung ke satu artefak `model.safetensors` + `config.json` utuh.
        """
    )
    return (mo,)


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from bifocalgemma.training.fast_pipeline import FastTrainingConfig, FastTrainingPipeline

    return FastTrainingConfig, FastTrainingPipeline, Path, os, root, src_dir, sys


@app.cell
def _(FastTrainingConfig, mo, os):
    import torch

    cuda_available = torch.cuda.is_available()
    model_choice = mo.ui.dropdown(
        options=["bifocalgemma-2b", "baseline-2b"],
        value="bifocalgemma-2b",
        label="Model Target",
    )
    total_steps = mo.ui.slider(
        start=10,
        stop=5000,
        step=50,
        value=2000 if cuda_available else 200,
        label="Total Steps",
    )
    output_dir = mo.ui.text(value="outputs/fast_bifocalgemma", label="Output Directory")
    pretrained_backbone_dir = mo.ui.text(
        value="", label="Pretrained Backbone Dir (Opsional, direktori safetensors lokal)"
    )
    data_dir = mo.ui.text(
        value="",
        label=(
            "Data Directory (JSONL / Curated Pool, kosongkan untuk auto-download "
            "mixed_context_enc32k_dec8k_v3 di GPU atau synthetic di Tiny)"
        ),
    )
    tiny_mode = mo.ui.checkbox(
        value=not cuda_available,
        label="Tiny Mode (CI / Verifikasi Cepat tanpa unduhan 2B)",
    )
    push_hub = mo.ui.checkbox(value=False, label="Push model utuh ke Hugging Face Hub")
    hub_repo = mo.ui.text(value="daruokta/bifocalgemma-fast", label="Hub Repository ID")
    hf_token = mo.ui.text(
        kind="password",
        value=os.environ.get("HF_TOKEN", ""),
        label="Hugging Face Token (opsional bila HF_TOKEN di environment)",
    )

    mo.vstack(
        [
            model_choice,
            total_steps,
            output_dir,
            pretrained_backbone_dir,
            data_dir,
            tiny_mode,
            push_hub,
            hub_repo,
            hf_token,
        ]
    )
    return (
        data_dir,
        hf_token,
        hub_repo,
        model_choice,
        output_dir,
        pretrained_backbone_dir,
        push_hub,
        tiny_mode,
        total_steps,
    )


@app.cell
def _(mo):
    train_button = mo.ui.button(label="Mulai Training Dari Awal", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(
    FastTrainingConfig,
    FastTrainingPipeline,
    data_dir,
    hf_token,
    hub_repo,
    mo,
    model_choice,
    os,
    output_dir,
    pretrained_backbone_dir,
    push_hub,
    tiny_mode,
    total_steps,
    train_button,
):
    if not train_button.value:
        mo.stop()

    steps = int(total_steps.value)
    if model_choice.value == "baseline-2b":
        stage1 = 0
        stage2 = steps
    else:
        stage1 = steps // 2
        stage2 = steps - stage1
    token = hf_token.value.strip() or os.environ.get("HF_TOKEN", "").strip() or None
    cfg = FastTrainingConfig(
        model_choice=model_choice.value,
        total_steps=steps,
        stage1_steps=stage1,
        stage2_steps=stage2,
        output_dir=output_dir.value.strip() or "outputs/fast_bifocalgemma",
        data_dir=data_dir.value.strip() or None,
        pretrained_backbone_dir=pretrained_backbone_dir.value.strip() or None,
        tiny=bool(tiny_mode.value),
        push_to_hub=push_hub.value,
        hub_repo=hub_repo.value.strip() if push_hub.value else None,
        hf_token=token,
        auto_download_curated=True,
    )

    if model_choice.value == "baseline-2b":
        mo.output.replace(
            mo.md(
                f"🚀 Memulai Full Decoder SFT {cfg.model_choice} ({steps} langkah penuh tanpa pembagian tahap)..."
            )
        )
    else:
        mo.output.replace(
            mo.md(
                f"🚀 Memulai pelatihan {cfg.model_choice} (Stage 1: {stage1} steps, Stage 2: {stage2} steps)..."
            )
        )
    pipeline = FastTrainingPipeline(cfg)
    summary = pipeline.run()

    mo.output.replace(
        mo.md(
            f"""
            ### ✅ Pelatihan Selesai!
            - **Model:** `{summary["model_choice"]}`
            - **Langkah Selesai:** `{summary["total_steps"]}`
            - **Final Validation Loss:** `{summary["final_val_loss"]:.4f}`
            - **Artefak Hugging Face Utuh Disimpan di:** `{summary["export_dir"]}`
            """
        )
    )
    return cfg, pipeline, steps, summary


if __name__ == "__main__":
    app.run()
