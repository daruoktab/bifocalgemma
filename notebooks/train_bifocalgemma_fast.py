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

        Pipeline pelatihan otomatis untuk **BifocalGemma-2B** (Encoder 32k → Causal Decoder 8k) dan **Baseline-2B**.
        - **Hugging Face Repo Otomatis:** [`daruokta/bifocalgemma-2026-11-02`](https://huggingface.co/daruokta/bifocalgemma-2026-11-02) (folder `fast-bifocalgemma-2b/` atau `fast-baseline-2b/`)
        - **Auto-Save & Auto-Resume:** Checkpoint disimpan otomatis setiap 250 steps dan saat dihentikan di tengah jalan, serta otomatis dilanjutkan saat dijalankan ulang.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _():
    import io
    import os
    import sys
    import time
    import urllib.request
    import zipfile
    from pathlib import Path

    RUNTIME_BUNDLE_VERSION = "2026-11-10-v3-fp32-fix"

    def ensure_fast_runtime(force_refresh: bool = False) -> Path:
        root = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
        src_dir = root / "src"
        if not (src_dir / "bifocalgemma/assets/gemma4_e2b.json").is_file():
            runtime_dir = Path("outputs/runtime").resolve()
            version_file = runtime_dir / ".runtime_version"
            current_ver = (
                version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else ""
            )
            if (
                force_refresh
                or not (runtime_dir / "src/bifocalgemma/assets/gemma4_e2b.json").is_file()
                or current_ver != RUNTIME_BUNDLE_VERSION
            ):
                runtime_dir.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/daruokta/bifocalgemma-2026-11-02/resolve/main/runtime/latest.zip?t={int(time.time())}"
                req = urllib.request.Request(url, headers={"User-Agent": "bifocalgemma-runtime"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                        zf.extractall(runtime_dir)
                version_file.write_text(RUNTIME_BUNDLE_VERSION, encoding="utf-8")
                for mod_name in [
                    k
                    for k in list(sys.modules)
                    if k == "bifocalgemma" or k.startswith("bifocalgemma.")
                ]:
                    sys.modules.pop(mod_name, None)
            src_dir = runtime_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        try:
            from bifocalgemma.assets import ensure_assets_on_disk

            ensure_assets_on_disk()
        except Exception:
            pass
        return src_dir

    src_dir = ensure_fast_runtime()
    from bifocalgemma.training.fast_pipeline import (
        DEFAULT_FAST_HF_REPO,
        FastTrainingConfig,
        FastTrainingPipeline,
    )

    return (
        DEFAULT_FAST_HF_REPO,
        FastTrainingConfig,
        FastTrainingPipeline,
        Path,
        ensure_fast_runtime,
        os,
        src_dir,
        sys,
        time,
    )


@app.cell
def _(DEFAULT_FAST_HF_REPO, mo, os):
    import torch

    cuda_available = torch.cuda.is_available()
    hf_token_input = mo.ui.text(
        value=os.environ.get("HF_TOKEN", ""),
        kind="password",
        label="Hugging Face Token (Read & Write)",
        placeholder="hf_...",
        full_width=True,
    )
    model_choice = mo.ui.dropdown(
        options=["bifocalgemma-2b", "baseline-2b"],
        value="bifocalgemma-2b",
        label="Model Target",
    )
    total_steps = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=2000 if cuda_available else 200,
        label="Total Steps",
    )
    train_button = mo.ui.run_button(label="🚀 Jalankan / Lanjutkan Training", kind="success")

    mo.vstack(
        [
            mo.md(
                f"""
                ### 🔑 Konfigurasi Pelatihan & Hugging Face Otomatis
                Masukkan `HF_TOKEN` Anda di bawah. Semua hasil dan checkpoint otomatis diunggah ke:
                - **`{DEFAULT_FAST_HF_REPO}`** → subfolder **`fast-bifocalgemma-2b/`** atau **`fast-baseline-2b/`**
                """
            ),
            hf_token_input,
            mo.hstack([model_choice, total_steps], justify="start", gap=2),
            train_button,
        ]
    )
    return cuda_available, hf_token_input, model_choice, total_steps, train_button


@app.cell
def _(
    DEFAULT_FAST_HF_REPO,
    FastTrainingConfig,
    FastTrainingPipeline,
    cuda_available,
    ensure_fast_runtime,
    hf_token_input,
    mo,
    model_choice,
    os,
    total_steps,
    train_button,
):
    hf_token = hf_token_input.value.strip() or os.environ.get("HF_TOKEN", "").strip() or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    mo.stop(
        not train_button.value,
        mo.md(
            f"Siap melatih **`{model_choice.value}`** ({int(total_steps.value)} steps) "
            f"dengan auto-upload ke [`{DEFAULT_FAST_HF_REPO}/fast-{model_choice.value}`]"
            f"(https://huggingface.co/{DEFAULT_FAST_HF_REPO}/tree/main/fast-{model_choice.value}). "
            "Tekan tombol **🚀 Jalankan / Lanjutkan Training** di atas."
        ),
    )

    is_tiny = not cuda_available
    if not is_tiny and not hf_token:
        mo.stop(
            True,
            mo.md(
                """
                ### ⚠️ Hugging Face Token Diperlukan!
                Masukkan token Hugging Face Anda pada kolom di atas lalu klik **🚀 Jalankan / Lanjutkan Training**.
                """
            ),
        )

    ensure_fast_runtime(force_refresh=True)
    from bifocalgemma.training.fast_pipeline import (
        FastTrainingConfig as _LiveFastTrainingConfig,
    )
    from bifocalgemma.training.fast_pipeline import (
        FastTrainingPipeline as _LiveFastTrainingPipeline,
    )

    steps = int(total_steps.value)
    subfolder = f"fast-{model_choice.value}"
    cfg = _LiveFastTrainingConfig(
        model_choice=model_choice.value,
        total_steps=steps,
        output_dir=f"outputs/fast_bifocalgemma/{model_choice.value}",
        tiny=is_tiny,
        push_to_hub=bool(hf_token),
        hub_repo=DEFAULT_FAST_HF_REPO,
        hub_subfolder=subfolder,
        hf_token=hf_token,
        auto_download_curated=True,
        resume_if_available=True,
        run_benchmarks=not is_tiny,
    )

    progress_tracker = {"last_step": 0}
    with mo.status.progress_bar(
        total=steps, title=f"Melatih & Evaluasi Benchmark {cfg.model_choice}"
    ) as bar:

        def on_step(record):
            curr_step = min(steps, int(record.get("step", 0)))
            delta = max(0, curr_step - progress_tracker["last_step"])
            progress_tracker["last_step"] = curr_step
            stage = record.get("stage", "training")
            if "bench_total" in record:
                b_done = record.get("bench_done", 0)
                b_tot = record.get("bench_total", 0)
                b_acc = record.get("bench_acc", 0.0) * 100.0
                bar.update(
                    increment=delta,
                    subtitle=f"[{stage}] {b_done}/{b_tot} kasus | Live Acc: {b_acc:.2f}%",
                )
            else:
                loss = record.get("loss", 0.0)
                val_loss = record.get("val_loss")
                val_str = f" | Val: {val_loss:.4f}" if val_loss is not None else ""
                bar.update(
                    increment=delta,
                    subtitle=f"[{stage}] Step {curr_step}/{steps} | Loss: {loss:.4f}{val_str}",
                )

        pipeline = _LiveFastTrainingPipeline(cfg, callback=on_step)
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

    benchmark_rows = [
        {
            "Benchmark": b.get("benchmark", "-"),
            "Total": b.get("total", 0),
            "Benar": b.get("correct", 0),
            "Akurasi": (
                f"{b['accuracy'] * 100:.2f}%"
                if isinstance(b.get("accuracy"), (int, float))
                else "-"
            ),
            "Patch Apply": (
                f"{b['patch_apply_rate'] * 100:.1f}%"
                if isinstance(b.get("patch_apply_rate"), (int, float))
                else "-"
            ),
            "Syntax Valid": (
                f"{b['syntax_success_rate'] * 100:.1f}%"
                if isinstance(b.get("syntax_success_rate"), (int, float))
                else "-"
            ),
            "Durasi": f"{b.get('segment_seconds', 0):.1f}s",
        }
        for b in summary.get("benchmarks", [])
    ]

    status_title = (
        "⏸️ Pelatihan/Evaluasi Dijeda & Checkpoint Tersimpan!"
        if summary.get("status") in {"INTERRUPTED_SAVED", "BENCHMARK_PAUSED"}
        else "🎉 Pelatihan & Benchmark Selesai (Terunggah ke Hugging Face)!"
    )
    hub_url = f"https://huggingface.co/{DEFAULT_FAST_HF_REPO}/tree/main/{subfolder}"

    mo.output.replace(
        mo.vstack(
            [
                mo.md(
                    f"""
                    ### {status_title}
                    - **Model Target:** `{summary["model_choice"]}`
                    - **Langkah Tercapai:** `{summary["total_steps"]} / {steps}`
                    - **Validation Loss Terakhir:** `{summary["final_val_loss"]:.4f}`
                    - **Benchmark Selesai:** `{len(benchmark_rows)} / 8`
                    - **Total Waktu Sesi Ini:** `{summary.get("elapsed_time_sec", 0):.1f} detik`
                    - **Folder Hugging Face:** [`{DEFAULT_FAST_HF_REPO}/{subfolder}`]({hub_url})
                    - **Folder Lokal:** `{summary["export_dir"]}`
                    """
                ),
                mo.ui.table(benchmark_rows, label="📊 Hasil Evaluasi Benchmark Suite")
                if benchmark_rows
                else mo.md(""),
                mo.ui.table(history_rows, label="📈 Riwayat Pelatihan (Milestones)")
                if history_rows
                else mo.md(""),
            ]
        )
    )
    return cfg, pipeline, steps, summary


if __name__ == "__main__":
    app.run()
