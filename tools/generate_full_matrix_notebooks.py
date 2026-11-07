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
        "title": "BifocalGemma · 2Base → 2IT (Symmetric)",
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
        "title": "BifocalGemma · 2IT → 2IT (Symmetric)",
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
        "title": "BifocalGemma · 4Base → 4IT (Symmetric)",
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
        "title": "BifocalGemma · 4IT → 4IT (Symmetric)",
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
        "title": "BifocalGemma · 4Base → 2IT (Asymmetric)",
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
        "title": "BifocalGemma · 4IT → 2IT (Asymmetric)",
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
        "title": "BifocalGemma · 2Base → 4IT (Asymmetric)",
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
        "title": "BifocalGemma · 2IT → 4IT (Asymmetric)",
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
        "title": "BifocalGemma · Baseline E2B-IT (Decoder-Only)",
        "config_type": "Baseline Decoder-Only (2B)",
        "encoder_id": None,
        "encoder_rev": None,
        "decoder_id": "google/gemma-4-E2B-it",
        "decoder_rev": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "enc_dim": 0,
        "dec_dim": 1536,
        "repo_id": CENTRALIZED_HF_REPO,
        "subfolder": "e2b-it",
    },
    "e4b-it": {
        "title": "BifocalGemma · Baseline E4B-IT (Decoder-Only)",
        "config_type": "Baseline Decoder-Only (4B)",
        "encoder_id": None,
        "encoder_rev": None,
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
    is_baseline = encoder_id is None
    if is_baseline:
        cell1_md = f"""        # {title}

        Pelatihan dan evaluasi terpadu Molab: Baseline Decoder-Only Causal LM (Full Decoder SFT).
        - **Model:** `{model_key}` ({config_type}: Decoder `{dec_dim}`) · **HF:** [{repo_id}](https://huggingface.co/{repo_id}) (`{subfolder}/`)
        - **Backbone:** Decoder `{decoder_id}` (rev `{decoder_rev}`)
        - **Kurikulum:** Full Decoder SFT = 11,102 steps terbuka penuh dari awal tanpa pembagian tahap (Batch=2, Accum=2)
        - **Benchmarks:** 6 Full (longcodeqa, humanevalfix, sweqa oracle/noisy, gsm8k 0/8-shot) + 2 Sampled-100 (crosscodelong chunk/func)"""
        ready_message = "**Runtime Baseline siap** · Full Decoder SFT & dynamic benchmark batching aktif."
    else:
        cell1_md = f"""        # {title}

        Pelatihan dan evaluasi terpadu Molab: SwiGLUDualGateBridge + Decoupled Full-Rank Cross-Attention.
        - **Model:** `{model_key}` ({config_type}: {enc_dim} -> {dec_dim}) · **HF:** [{repo_id}](https://huggingface.co/{repo_id}) (`{subfolder}/`)
        - **Backbones:** Encoder `{encoder_id}` (rev `{encoder_rev}`) | Decoder `{decoder_id}` (rev `{decoder_rev}`)
        - **Kurikulum:** Stage 1 Warmup + Stage 2 Joint SFT = 11,102 steps (Batch=2, Accum=2)
        - **Benchmarks:** 6 Full (longcodeqa, humanevalfix, sweqa oracle/noisy, gsm8k 0/8-shot) + 2 Sampled-100 (crosscodelong chunk/func)"""
        ready_message = "**Runtime BifocalGemma siap** · SwiGLUDualGateBridge & dynamic benchmark batching aktif."

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
{cell1_md}
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
        "{ready_message}"
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
            "token": mo.ui.text(kind="password", label="HF write token (atau HF_TOKEN environment)"),
            "repo_id": mo.ui.text(value="daruokta/bifocalgemma-clean", label="Target HF Repo"),
            "standalone_repo_id": mo.ui.text(value="", label="Standalone HF Repo (opsional)"),
            "total_steps": mo.ui.number(value=11102, start=10, stop=50000, step=100, label="Total Steps"),
            "force_fresh": mo.ui.checkbox(value=False, label="Force Fresh (abaikan checkpoint lama)"),
        }}
    ).form(submit_button_label="Jalankan")
    mo.vstack(
        [
            mo.md(
                f"**Model:** `{{MODEL_KEY}}` · **Default HF:** `{{HF_REPO_ID}}` (subfolder `{{HF_SUBFOLDER}}`)\\n"
                f"Pelatihan {{MODEL_KEY}} dan 8 benchmark berjalan terpadu.\\n"
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
        __import__("importlib").import_module("bifocalgemma.training.wiggly_dashboard").WigglyDashboard
    )
    live_monitor = WigglyDashboard(mo, experiment=MODEL_KEY, device="cuda")
    live_monitor.layout()
    return (live_monitor,)


@app.cell
def _(MODEL_KEY, live_monitor, mo, token_form):
    import os

    form = token_form.value or {{}}
    token = form.get("token", "").strip() or os.environ.get("HF_TOKEN", "").strip()
    repo_id = form.get("repo_id", "").strip() or "daruokta/bifocalgemma-clean"
    standalone_repo_id = form.get("standalone_repo_id", "").strip() or None
    total_steps = int(form.get("total_steps") or 11102)
    force_fresh = bool(form.get("force_fresh", False))

    mo.stop(
        not token,
        mo.md(
            "Isi HF token di atas dan tekan **Jalankan**, atau sediakan `HF_TOKEN` lalu Run.\\n"
            f"Training: {{total_steps:,}} steps dan dilanjutkan 8 benchmark."
        ),
    )
    ms = __import__("importlib").import_module("tools.molab_matrix_support")

    result = ms.run_matrix_training_and_benchmark(
        MODEL_KEY,
        token=token,
        repo_id=repo_id,
        standalone_repo_id=standalone_repo_id,
        total_steps=total_steps,
        force_fresh=force_fresh,
        mo=mo,
        monitor=live_monitor,
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
            Path(sys.executable).parent
            / "Scripts"
            / ("ruff.exe" if sys.platform == "win32" else "ruff"),
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
    parser.add_argument(
        "--check", action="store_true", help="Check existing files match generated code"
    )
    parser.add_argument(
        "--publish", action="store_true", help="Publish immutable runtime bundle to Hugging Face"
    )
    args = parser.parse_args()

    print("Building deterministic source bundle...")
    payload, bundle_sha = matrix_bundle(ROOT)
    print(f"Source bundle hash: {bundle_sha}")

    notebooks_dir = ROOT / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    runtime_dir = notebooks_dir / "runtime"
    runtime_dir.mkdir(exist_ok=True)
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
            commit_message="Publish immutable BifocalGemma matrix notebook runtime",
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
        for old_zip in runtime_dir.glob("runtime-*.zip"):
            try:
                old_zip.unlink()
            except OSError:
                pass
        (runtime_dir / f"runtime-{bundle_sha[:16]}.zip").write_bytes(payload)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {manifest_path}")
    elif args.check:
        if not manifest_path.is_file():
            raise SystemExit(f"Missing runtime manifest: {manifest_path}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != bundle_sha:
            raise SystemExit(
                f"Source bundle hash ({bundle_sha}) does not match manifest ({manifest.get('sha256')}). "
                "Run without --check to regenerate."
            )
    else:
        filename = f"runtime/runtime-{bundle_sha[:16]}.zip"
        for old_zip in runtime_dir.glob("runtime-*.zip"):
            try:
                old_zip.unlink()
            except OSError:
                pass
        (runtime_dir / f"runtime-{bundle_sha[:16]}.zip").write_bytes(payload)
        existing_rev = "LOCAL-PREVIEW"
        if manifest_path.is_file():
            try:
                prev = json.loads(manifest_path.read_text(encoding="utf-8"))
                if prev.get("sha256") == bundle_sha and prev.get("revision"):
                    existing_rev = prev["revision"]
            except Exception:
                pass
        manifest = {
            "repo_id": CENTRALIZED_HF_REPO,
            "filename": filename,
            "revision": existing_rev,
            "sha256": bundle_sha,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Updated local runtime manifest: {manifest_path}")

    generated_files = []
    for model_key in MATRIX_NOTEBOOK_SPECS:
        filename = f"train_{model_key.replace('-', '_')}.py"
        target_path = notebooks_dir / filename
        print(f"Rendering {filename} for model {model_key}...")

        raw_code = render_notebook(model_key, manifest)
        formatted_code = format_code(raw_code, filename)

        # Validate syntax
        ast.parse(formatted_code, filename=str(target_path))

        lines_count = len(formatted_code.splitlines())
        if lines_count >= 250:
            raise ValueError(
                f"Notebook {filename} has {lines_count} lines, which exceeds the limit of 250"
            )

        if args.check:
            if not target_path.exists():
                raise SystemExit(f"Missing notebook file: {target_path}")
            current = target_path.read_text(encoding="utf-8")
            if current != formatted_code:
                raise SystemExit(f"Notebook out of sync: {target_path}")
        else:
            target_path.write_text(formatted_code, encoding="utf-8")
            print(f"  Wrote {target_path} ({len(formatted_code)} bytes, {lines_count} lines)")
        generated_files.append(target_path)

    print(f"\nSuccessfully generated and validated {len(generated_files)} matrix notebooks!")


if __name__ == "__main__":
    main()
