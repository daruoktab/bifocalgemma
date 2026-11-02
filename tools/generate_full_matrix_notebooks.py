"""Generate and verify the 10 experimental Marimo training notebooks for the Molab V3 matrix run."""

import argparse
import ast
import base64
import hashlib
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.notebook_bundle import source_bundle_bytes  # noqa: E402

CENTRALIZED_HF_REPO = "daruokta/bifocalgemma-2026-11-02"

MATRIX_NOTEBOOK_SPECS = {
    "2base-2it": {
        "title": "BifocalGemma V3 · 2Base → 2IT (Symmetric)",
        "config_type": "Symmetric (2B -> 2B)",
        "encoder_id": "google/gemma-4-E2B",
        "encoder_rev": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "enc_dim": 1536,
        "dec_dim": 1536,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "2base-2it",
    },
    "2it-2it": {
        "title": "BifocalGemma V3 · 2IT → 2IT (Symmetric)",
        "config_type": "Symmetric (2B -> 2B)",
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "enc_dim": 1536,
        "dec_dim": 1536,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "2it-2it",
    },
    "4base-4it": {
        "title": "BifocalGemma V3 · 4Base → 4IT (Symmetric)",
        "config_type": "Symmetric (4B -> 4B)",
        "encoder_id": "google/gemma-4-E4B",
        "encoder_rev": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "enc_dim": 2560,
        "dec_dim": 2560,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "4base-4it",
    },
    "4it-4it": {
        "title": "BifocalGemma V3 · 4IT → 4IT (Symmetric)",
        "config_type": "Symmetric (4B -> 4B)",
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "enc_dim": 2560,
        "dec_dim": 2560,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "4it-4it",
    },
    "4base-2it": {
        "title": "BifocalGemma V3 · 4Base → 2IT (Asymmetric)",
        "config_type": "Asymmetric (4B -> 2B)",
        "encoder_id": "google/gemma-4-E4B",
        "encoder_rev": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "enc_dim": 2560,
        "dec_dim": 1536,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "4base-2it",
    },
    "4it-2it": {
        "title": "BifocalGemma V3 · 4IT → 2IT (Asymmetric)",
        "config_type": "Asymmetric (4B -> 2B)",
        "encoder_id": "google/gemma-4-E4B-it",
        "encoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "enc_dim": 2560,
        "dec_dim": 1536,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "4it-2it",
    },
    "2base-4it": {
        "title": "BifocalGemma V3 · 2Base → 4IT (Asymmetric)",
        "config_type": "Asymmetric (2B -> 4B)",
        "encoder_id": "google/gemma-4-E2B",
        "encoder_rev": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "enc_dim": 1536,
        "dec_dim": 2560,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "2base-4it",
    },
    "2it-4it": {
        "title": "BifocalGemma V3 · 2IT → 4IT (Asymmetric)",
        "config_type": "Asymmetric (2B -> 4B)",
        "encoder_id": "google/gemma-4-E2B-it",
        "encoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "enc_dim": 1536,
        "dec_dim": 2560,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "2it-4it",
    },
    "e2b-it": {
        "title": "BifocalGemma V3 · Baseline E2B-IT (Decoder-Only)",
        "config_type": "Baseline Decoder-Only (2B)",
        "encoder_id": "None",
        "encoder_rev": "None",
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "enc_dim": 0,
        "dec_dim": 1536,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "e2b-it",
    },
    "e4b-it": {
        "title": "BifocalGemma V3 · Baseline E4B-IT (Decoder-Only)",
        "config_type": "Baseline Decoder-Only (4B)",
        "encoder_id": "None",
        "encoder_rev": "None",
        "decoder_id": "google/gemma-4-E4B-it",
        "decoder_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "enc_dim": 0,
        "dec_dim": 2560,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "e4b-it",
    },
}


def matrix_bundle(root: Path) -> tuple[str, str]:
    buffer = io.BytesIO(source_bundle_bytes(root))
    matrix_support = (root / "tools/molab_matrix_support.py").read_text(encoding="utf-8")
    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED) as archive:
        item = zipfile.ZipInfo("tools/molab_matrix_support.py", date_time=(1980, 1, 1, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        item.create_system = 3
        item.external_attr = 0o100644 << 16
        archive.writestr(item, matrix_support.replace("\r\n", "\n"))
    payload = buffer.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    encoded = base64.b85encode(payload).decode("ascii")
    literal = (
        "(\n"
        + "\n".join("        " + repr(encoded[i : i + 100]) for i in range(0, len(encoded), 100))
        + "\n    )"
    )
    return literal, checksum


def render_notebook(model_key: str, bundle_literal: str, bundle_sha: str) -> str:
    spec = MATRIX_NOTEBOOK_SPECS[model_key]
    title = spec["title"]
    encoder_id = spec["encoder_id"]
    encoder_rev = spec["encoder_rev"]
    decoder_id = spec["decoder_id"]
    decoder_rev = spec["decoder_rev"]
    config_type = spec["config_type"]
    enc_dim = spec["enc_dim"]
    dec_dim = spec["dec_dim"]
    repo_id = spec["repo_id"]
    subfolder = spec["subfolder"]

    code = f'''# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo==0.25.2", "torch==2.14.1", "torchvision==0.29.1",
#     "transformers==5.20.2", "huggingface-hub==2.2.0", "safetensors==0.8.0",
#     "tokenizers==0.23.3", "pyarrow==26.0.0", "numpy==2.5.4", "pytest==8.4.2",
#     "psutil==7.2.2", "PyYAML>=6.0",
#     "wigglystuff==0.5.36",
#     "triton-windows==3.8.0.post29; sys_platform == 'win32'",
#     "triton==3.8.0; sys_platform == 'linux'",
# ]
# [tool.uv.sources]
# torch = [{{ index = "pytorch-cu130" }}]
# torchvision = [{{ index = "pytorch-cu130" }}]
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
# Generated by tools/generate_full_matrix_notebooks.py.
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # {title}

        Notebook pelatihan dan evaluasi terpadu Molab (10-Model Matrix V3) dengan arsitektur **SwiGLUDualGateBridge + Decoupled Full-Rank Cross-Attention**, kurikulum anti-catastrophic-forgetting (15% Code QA / MCQ & Span-Copy replay), `BATCH_SIZE = 2`, serta dynamic optimal evaluation/benchmark batching.

        ### Spesifikasi Model:
        - **Model Key:** `{model_key}`
        - **Encoder Backbone:** `{encoder_id}` (rev `{encoder_rev}`)
        - **Decoder Backbone:** `{decoder_id}` (rev `{decoder_rev}`)
        - **Konfigurasi Dimensi:** `{config_type}` (Encoder `{enc_dim}` -> Decoder `{dec_dim}`)
        - **Arsitektur V3:** `SwiGLUDualGateBridge` + Decoupled Full-Rank BF16 Cross-Attention (`atanh(0.05)` gate init, unrotated cross-coordinates, critical length scaling)
        - **Centralized Hugging Face Repository:** [{repo_id}](https://huggingface.co/{repo_id}) (Subfolder: `{subfolder}/`)

        ### Kurikulum Dua Tahap Anti-Forgetting (11.102 Steps / 44.408 Contoh · Batch = 2, Accum = 2):
        1. **Stage 1 (5.551 steps):**
           - Alignment & Cross-Attention Warmup (Decoder LoRA beku untuk model berpasangan; aktif untuk baseline decoder-only)
           - 15% Code QA / MCQ & Exact Span-Copy comprehension replay untuk mencegah catastrophic forgetting
        2. **Stage 2 (5.551 steps):**
           - Joint SFT Fine-Tuning (Decoder LoRA + Bridge + Cross-Attention + Encoder LoRA)
           - 15% Code QA / MCQ & Exact Span-Copy replay dipertahankan sepanjang epoch

        ### Evaluasi 8 Benchmark Terurut (6 Full Cepat + 2 Heavy Sampled-100) dengan Dynamic Batching:
        1. `longcodeqa_gemma32k_dec8k_v1` (Full)
        2. `humanevalfix_python_v2` (Full)
        3. `sweqa_oracle_gemma32k_dec8k_v1` (Full)
        4. `sweqa_noisy_oracle_gemma32k_dec8k_v1` (Full)
        5. `gsm8k_zero_shot_v2` (Full)
        6. `gsm8k_eight_shot_v2` (Full)
        7. `crosscodelong_chunk_gemma32k_dec8k_v1` (100 deterministic sampled cases, `seed=17`)
        8. `crosscodelong_function_gemma32k_dec8k_v1` (100 deterministic sampled cases, `seed=17`)
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    import base64
    import hashlib
    import io
    import runpy
    import sys
    import zipfile
    from pathlib import Path

    SOURCE_BUNDLE = {bundle_literal}
    BUNDLE_SHA256 = "{bundle_sha}"

    def activate_runtime(payload_b85: str, expected_sha: str, notebook_file=None):
        root = (Path("outputs/v2/runtime") / expected_sha[:16]).resolve()
        ready_flag = root / ".bundle_ready"
        if not ready_flag.is_file():
            data = base64.b85decode(payload_b85)
            if hashlib.sha256(data).hexdigest() != expected_sha:
                raise ValueError("Source bundle SHA256 checksum mismatch.")
            root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for entry in archive.infolist():
                    dest = (root / entry.filename).resolve()
                    if not dest.is_relative_to(root):
                        raise ValueError("Unsafe bundle path escape detected.")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.read(entry))
            ready_flag.write_text("ok", encoding="utf-8")

        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        if notebook_file:
            repo_root = Path(notebook_file).resolve().parents[1]
            if (repo_root / "src/bifocalgemma").is_dir() and str(repo_root / "src") not in sys.path:
                sys.path.insert(0, str(repo_root / "src"))
                sys.path.insert(0, str(repo_root))

        launcher = runpy.run_path(str(root / "tools/notebook_launcher.py"))
        dependency_runtime = launcher["prepare_runtime"](root)
        return dependency_runtime, root

    dependency_runtime, runtime_root = activate_runtime(
        SOURCE_BUNDLE,
        BUNDLE_SHA256,
        notebook_file=globals().get("__file__"),
    )
    mo.md(
        f"**Runtime V3 siap** · bundle `{{BUNDLE_SHA256[:12]}}` · SwiGLUDualGateBridge & dynamic benchmark batching aktif."
    )
    return BUNDLE_SHA256, dependency_runtime, runtime_root


@app.cell
def _(dependency_runtime, mo, runtime_root):
    assert dependency_runtime["status"] == "READY"
    MODEL_KEY = "{model_key}"
    HF_REPO_ID = "{repo_id}"
    HF_SUBFOLDER = "{subfolder}"
    token_form = mo.ui.dictionary(
        {{
            "token": mo.ui.text(
                kind="password", label="HF write token (atau HF_TOKEN environment)"
            ),
        }}
    ).form(submit_button_label="Jalankan")
    mo.vstack(
        [
            mo.md(
                f"**Model:** `{{MODEL_KEY}}` · **HF:** `{{HF_REPO_ID}}` (subfolder `{{HF_SUBFOLDER}}`)\\n"
                "Training 11.102 steps (Batch=2, Accum=2) dan 8 benchmark (6 Full + 2 Sampled-100) berjalan otomatis.\\n"
                "Token di bawah hanya diperlukan jika `HF_TOKEN` belum tersedia di environment."
            ),
            token_form,
        ]
    )
    return HF_REPO_ID, HF_SUBFOLDER, MODEL_KEY, token_form


@app.cell
def _(MODEL_KEY, dependency_runtime, mo):
    assert dependency_runtime["status"] == "READY"
    WigglyDashboard = (
        __import__("importlib")
        .import_module("bifocalgemma.training.wiggly_dashboard")
        .WigglyDashboard
    )

    live_monitor = WigglyDashboard(mo, experiment=MODEL_KEY, device="cuda")
    live_monitor.layout()
    return (live_monitor,)


@app.cell
def _(MODEL_KEY, live_monitor, mo, token_form):
    import os

    token = os.environ.get("HF_TOKEN", "").strip()
    if token_form.value is not None:
        token = token_form.value["token"].strip() or token
    mo.stop(
        not token,
        mo.md(
            "Isi HF token di atas dan tekan **Jalankan**, atau sediakan `HF_TOKEN` lalu Run.\\n"
            "Training akan menyelesaikan Stage 1 (5.551 steps) + Stage 2 (5.551 steps) = 11.102 steps "
            "dan dilanjutkan dengan 8 benchmark (6 full + 2 sampled-100)."
        ),
    )
    ms = __import__("importlib").import_module("tools.molab_matrix_support")

    result = ms.run_matrix_training_and_benchmark(
        MODEL_KEY, token=token, mo=mo, monitor=live_monitor
    )
    mo.vstack(
        [
            mo.md(
                f"## Hasil sesi\\n**{{result['model']}}** · **{{result['status']}}**\\n"
                f"HF: `{{result['repo']}}` (`{{result.get('subfolder', MODEL_KEY)}}`)"
            ),
            mo.ui.table(result["benchmarks"], selection=None),
        ]
    )
    return (result,)


if __name__ == "__main__":
    app.run()
'''
    return code


def format_code(code: str, filename: str) -> str:
    formatter = shutil.which("ruff")
    if not formatter:
        candidate = Path.home() / ".local/bin" / ("ruff.exe" if shutil.which("cmd") else "ruff")
        if candidate.is_file():
            formatter = str(candidate)
    if formatter:
        try:
            formatted = subprocess.run(
                [formatter, "check", "--select", "I", "--fix", "--stdin-filename", filename, "-"],
                input=code,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=True,
            ).stdout
            return subprocess.run(
                [formatter, "format", "--stdin-filename", filename, "-"],
                input=formatted,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=True,
            ).stdout
        except subprocess.SubprocessError:
            pass
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check existing files match generated code")
    args = parser.parse_args()

    print("Building deterministic source bundle...")
    bundle_literal, bundle_sha = matrix_bundle(ROOT)
    print(f"Source bundle hash: {bundle_sha}")

    notebooks_dir = ROOT / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    generated_files = []
    for model_key in MATRIX_NOTEBOOK_SPECS:
        filename = f"train_{model_key.replace('-', '_')}.py"
        target_path = notebooks_dir / filename
        print(f"Rendering {filename} for model {model_key}...")

        raw_code = render_notebook(model_key, bundle_literal, bundle_sha)
        formatted_code = format_code(raw_code, filename)

        # Validate syntax
        ast.parse(formatted_code, filename=str(target_path))

        if args.check:
            if not target_path.exists():
                raise SystemExit(f"Missing notebook file: {target_path}")
            current = target_path.read_text(encoding="utf-8")
            if current != formatted_code:
                raise SystemExit(f"Notebook out of sync: {target_path}")
        else:
            target_path.write_text(formatted_code, encoding="utf-8")
            print(f"  Wrote {target_path} ({len(formatted_code)} bytes)")
        generated_files.append(target_path)

    print(f"\nSuccessfully generated and validated {len(generated_files)} matrix notebooks!")


if __name__ == "__main__":
    main()
