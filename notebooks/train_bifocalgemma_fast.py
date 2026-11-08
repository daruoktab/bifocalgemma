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
    import io
    import os
    import sys
    import time
    import urllib.request
    import zipfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    src_dir = root / "src"
    if not (src_dir / "bifocalgemma").is_dir():
        runtime_dir = Path("outputs/runtime").resolve()
        if not (runtime_dir / "src/bifocalgemma").is_dir():
            url = "https://huggingface.co/daruokta/bifocalgemma-2026-11-02/resolve/main/runtime/latest.zip"
            with urllib.request.urlopen(url, timeout=60) as resp:
                with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                    zf.extractall(runtime_dir)
        src_dir = runtime_dir / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from bifocalgemma.training.fast_pipeline import FastTrainingConfig, FastTrainingPipeline

    return FastTrainingConfig, FastTrainingPipeline, Path, os, root, src_dir, sys, time


@app.cell
def _(mo, os):
    hf_token_input = mo.ui.text(
        value=os.environ.get("HF_TOKEN", ""),
        kind="password",
        label="Hugging Face Token",
        placeholder="hf_...",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 🔑 1. Autentikasi Hugging Face
                Masukkan token Hugging Face Anda (izin *Read* untuk unduh dataset privat `daruokta/bifocalgemma-data` & model backbone Gemma-4, atau izin *Write* untuk push model ke Hub).
                """
            ),
            hf_token_input,
        ]
    )
    return (hf_token_input,)


@app.cell
def _(hf_token_input, mo, os):
    hf_token = hf_token_input.value.strip() or os.environ.get("HF_TOKEN", "").strip() or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        auth_banner = mo.md("✅ **Hugging Face Token aktif.** Kredensial siap digunakan.")
    else:
        auth_banner = mo.md(
            "⚠️ **Token belum diisi.** Diperlukan untuk mode non-tiny dengan unduhan otomatis dataset/backbone."
        )
    auth_banner
    return auth_banner, hf_token


@app.cell
def _(mo):
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

    training_settings = mo.vstack(
        [
            mo.md("### ⚙️ 2. Pengaturan Pelatihan"),
            model_choice,
            total_steps,
            tiny_mode,
            output_dir,
            push_hub,
            hub_repo,
            pretrained_backbone_dir,
            data_dir,
        ]
    )
    training_settings
    return (
        data_dir,
        hub_repo,
        model_choice,
        output_dir,
        pretrained_backbone_dir,
        push_hub,
        tiny_mode,
        total_steps,
        training_settings,
    )


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Mulai Training Dari Awal", kind="success")
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
    output_dir,
    pretrained_backbone_dir,
    push_hub,
    time,
    tiny_mode,
    total_steps,
    train_button,
):
    mo.stop(
        not train_button.value,
        mo.md("Tekan tombol **Mulai Training Dari Awal** di atas untuk menjalankan pelatihan."),
    )

    is_tiny = bool(tiny_mode.value)
    has_local_data = bool(data_dir.value.strip())

    if not is_tiny and not has_local_data and not hf_token:
        mo.stop(
            True,
            mo.md(
                """
                ### ⚠️ Hugging Face Token Diperlukan!
                Anda menjalankan mode **Full Scale (2B)** dengan unduhan dataset otomatis.
                Dataset kurasi `daruokta/bifocalgemma-data` dan model backbone Gemma adalah repositori privat/gated yang **memerlukan HF_TOKEN**.

                👉 **Silakan masukkan token Hugging Face Anda pada Langkah 1 di atas, lalu tekan tombol 'Mulai Training Dari Awal' kembali.**
                *(Atau centang 'Tiny Mode' jika Anda hanya ingin uji coba cepat tanpa unduhan 2B).*
                """
            ),
        )

    steps = int(total_steps.value)
    if model_choice.value == "baseline-2b":
        stage1 = 0
        stage2 = steps
    else:
        stage1 = steps // 2
        stage2 = steps - stage1
    cfg = FastTrainingConfig(
        model_choice=model_choice.value,
        total_steps=steps,
        stage1_steps=stage1,
        stage2_steps=stage2,
        output_dir=output_dir.value.strip() or "outputs/fast_bifocalgemma",
        data_dir=data_dir.value.strip() or None,
        pretrained_backbone_dir=pretrained_backbone_dir.value.strip() or None,
        tiny=is_tiny,
        push_to_hub=push_hub.value,
        hub_repo=hub_repo.value.strip() if push_hub.value else None,
        hf_token=hf_token,
        auto_download_curated=True,
    )

    progress_tracker = {"last_step": 0}
    with mo.status.progress_bar(total=steps, title=f"Melatih {cfg.model_choice}") as bar:

        def on_step(record):
            curr_step = record["step"]
            delta = max(0, curr_step - progress_tracker["last_step"])
            progress_tracker["last_step"] = curr_step
            stage = record.get("stage", "training")
            loss = record.get("loss", 0.0)
            val_loss = record.get("val_loss")
            val_str = f" | Val: {val_loss:.4f}" if val_loss is not None else ""
            bar.update(
                increment=delta,
                subtitle=f"[{stage}] Step {curr_step}/{steps} | Loss: {loss:.4f}{val_str}",
            )

        pipeline = FastTrainingPipeline(cfg, callback=on_step)
        summary = pipeline.run()

    history_rows = [
        {
            "Step": h["step"],
            "Stage": h["stage"],
            "Train Loss": f"{h['loss']:.4f}",
            "Val Loss": f"{h['val_loss']:.4f}" if h.get("val_loss") is not None else "-",
            "Durasi": f"{h.get('elapsed_sec', 0):.1f}s",
        }
        for h in pipeline.history
    ]

    hub_info = (
        f"- **Hugging Face Hub:** [{cfg.hub_repo}](https://huggingface.co/{cfg.hub_repo})\n"
        if cfg.push_to_hub and cfg.hub_repo
        else ""
    )

    mo.output.replace(
        mo.vstack(
            [
                mo.md(
                    f"""
                    ### 🎉 Pelatihan Selesai!
                    - **Model Target:** `{summary['model_choice']}`
                    - **Total Langkah:** `{summary['total_steps']}`
                    - **Final Validation Loss:** `{summary['final_val_loss']:.4f}`
                    - **Total Waktu:** `{summary.get('elapsed_time_sec', 0):.1f} detik`
                    - **Artefak Model Utuh Disimpan di:** `{summary['export_dir']}`
                    {hub_info}
                    """
                ),
                mo.ui.table(history_rows, label="Riwayat Pelatihan (Milestones)")
                if history_rows
                else mo.md(""),
            ]
        )
    )
    return cfg, pipeline, steps, summary


if __name__ == "__main__":
    app.run()
