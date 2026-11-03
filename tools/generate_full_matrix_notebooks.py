"""Generate and verify the 10 experimental Marimo training notebooks for the Molab V3 matrix run."""

import argparse
import ast
import hashlib
import io
import json
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


def matrix_bundle(root: Path) -> tuple[bytes, str]:
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
    return payload, checksum


def render_notebook(model_key: str, manifest: dict[str, str]) -> str:
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
    manifest_repr = json.dumps(manifest, indent=8)

    code = f'''# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo==0.25.2", "torch==2.15.0", "torchvision==0.29.2",
#     "transformers==5.22.1", "huggingface-hub==2.4.0", "safetensors==0.8.0",
#     "tokenizers==0.23.3", "pyarrow==26.0.0", "numpy==2.5.4", "pytest==9.2.0",
#     "psutil==7.2.2", "PyYAML>=6.0",
#     "wigglystuff==0.5.36",
#     "triton-windows==3.9.0.post30; sys_platform == 'win32'",
#     "triton==3.9.0; sys_platform == 'linux'",
# ]
# [tool.uv.sources]
# torch = [{{ index = "pytorch-cu132" }}]
# torchvision = [{{ index = "pytorch-cu132" }}]
# [[tool.uv.index]]
# name = "pytorch-cu132"
# url = "https://download.pytorch.org/whl/cu132"
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
    import runpy

    def load_runtime(repo_id, filename, revision, expected_sha, notebook_file=None):
        import hashlib
        import io
        import sys
        import urllib.request
        import zipfile
        from pathlib import Path
        from uuid import uuid4

        root = (Path("outputs/v3/runtime") / expected_sha[:16]).resolve()
        local = (
            Path(notebook_file).resolve().parent / "runtime" / Path(filename).name
            if notebook_file
            else None
        )
        cached = root / "runtime.zip"
        source = local if local is not None and local.is_file() else cached
        data = source.read_bytes() if source.is_file() else b""
        if hashlib.sha256(data).hexdigest() != expected_sha:
            urls = [
                f"https://huggingface.co/{{repo_id}}/resolve/{{revision}}/{{filename}}",
                f"https://huggingface.co/datasets/{{repo_id}}/resolve/{{revision}}/{{filename}}",
            ]
            downloaded = False
            for url in urls:
                try:
                    req = urllib.request.Request(
                        url, headers={{"User-Agent": "bifocalgemma-runtime"}}
                    )
                    with urllib.request.urlopen(req, timeout=60) as response:
                        data = response.read(64 * 2**20 + 1)
                    if hashlib.sha256(data).hexdigest() == expected_sha:
                        downloaded = True
                        break
                except Exception:
                    continue
            if not downloaded and (
                len(data) > 64 * 2**20 or hashlib.sha256(data).hexdigest() != expected_sha
            ):
                raise ValueError("Runtime archive checksum mismatch; no code was activated.")

        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                target = (root / entry.filename).resolve()
                if (
                    not target.is_relative_to(root)
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Runtime archive contains an unsafe path.")
                if (
                    entry.file_size > 32 * 2**20
                    or sum(e.file_size for e in archive.infolist()) > 128 * 2**20
                ):
                    raise ValueError("Runtime archive is too large.")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))

        temporary = root / (".download-" + uuid4().hex)
        temporary.write_bytes(data)
        temporary.replace(cached)

        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        if notebook_file:
            repo_root = Path(notebook_file).resolve().parents[1]
            if (repo_root / "src/bifocalgemma").is_dir() and str(repo_root / "src") not in sys.path:
                sys.path.insert(0, str(repo_root / "src"))
                sys.path.insert(0, str(repo_root))

        return root

    runtime_manifest = {manifest_repr}
    runtime_root = load_runtime(
        runtime_manifest["repo_id"],
        runtime_manifest["filename"],
        runtime_manifest["revision"],
        runtime_manifest["sha256"],
        notebook_file=globals().get("__file__"),
    )
    launcher = runpy.run_path(str(runtime_root / "tools/notebook_launcher.py"))
    dependency_runtime = launcher["prepare_runtime"](runtime_root)
    mo.md(
        f"**Runtime V3 siap** · bundle `{{runtime_manifest['sha256'][:12]}}` · SwiGLUDualGateBridge & dynamic benchmark batching aktif."
    )
    return dependency_runtime, runtime_manifest, runtime_root


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
        for candidate in [
            Path(sys.executable).parent / ("ruff.exe" if sys.platform == "win32" else "ruff"),
            Path(sys.executable).parent / "Scripts" / ("ruff.exe" if sys.platform == "win32" else "ruff"),
            Path.home() / ".local/bin" / ("ruff.exe" if shutil.which("cmd") else "ruff"),
        ]:
            if candidate.is_file():
                formatter = str(candidate)
                break
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
    parser.add_argument(
        "--publish", action="store_true", help="Publish immutable runtime bundle to Hugging Face"
    )
    args = parser.parse_args()

    print("Building deterministic source bundle...")
    payload, bundle_sha = matrix_bundle(ROOT)
    print(f"Source bundle hash: {bundle_sha}")

    notebooks_dir = ROOT / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    manifest_path = notebooks_dir / "runtime-manifest.json"

    if args.publish:
        from huggingface_hub import HfApi, hf_hub_download

        filename = f"runtime/runtime-{bundle_sha[:16]}.zip"
        repo_id = CENTRALIZED_HF_REPO
        api = HfApi()
        print(f"Publishing runtime bundle to {repo_id}:{filename}...")
        commit = api.upload_file(
            repo_id=repo_id,
            path_in_repo=filename,
            path_or_fileobj=payload,
            commit_message="Publish immutable BifocalGemma V3 matrix notebook runtime",
        )
        print(f"Uploaded commit: {commit.oid}")
        remote = hf_hub_download(repo_id, filename, revision=commit.oid)
        if hashlib.sha256(Path(remote).read_bytes()).hexdigest() != bundle_sha:
            raise ValueError("Published runtime checksum mismatch.")
        manifest = {
            "repo_id": repo_id,
            "filename": filename,
            "revision": commit.oid,
            "sha256": bundle_sha,
        }
        runtime_dir = notebooks_dir / "runtime"
        runtime_dir.mkdir(exist_ok=True)
        (runtime_dir / f"runtime-{bundle_sha[:16]}.zip").write_bytes(payload)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {manifest_path}")
    else:
        if not manifest_path.is_file():
            raise SystemExit(f"Missing runtime manifest: {manifest_path}. Use --publish first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != bundle_sha:
            raise SystemExit(
                f"Source bundle hash ({bundle_sha}) does not match manifest ({manifest.get('sha256')}). "
                "Use --publish to upload the new bundle, or update the manifest."
            )

    generated_files = []
    for model_key in MATRIX_NOTEBOOK_SPECS:
        filename = f"train_{model_key.replace('-', '_')}.py"
        target_path = notebooks_dir / filename
        print(f"Rendering {filename} for model {model_key}...")

        raw_code = render_notebook(model_key, manifest)
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
            print(
                f"  Wrote {target_path} ({len(formatted_code)} bytes, {len(formatted_code.splitlines())} lines)"
            )
        generated_files.append(target_path)

    print(f"\nSuccessfully generated and validated {len(generated_files)} matrix notebooks!")


if __name__ == "__main__":
    main()
