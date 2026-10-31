"""Launch the BifocalGemma pilot on an active Colab session with live streaming."""

import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import get_token


def run_cmd(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else "pilot-a100"
    arm = sys.argv[2] if len(sys.argv) > 2 else "full_cross_bf16_frozen_then_joint"
    steps = sys.argv[3] if len(sys.argv) > 3 else "200"

    token = get_token()
    if not token:
        print("Error: No local Hugging Face token found.")
        sys.exit(1)

    script_path = Path("outputs/colab/pilot_e2b_e2b.py").resolve()
    if not script_path.is_file():
        print(f"Error: Script {script_path} not found.")
        sys.exit(1)

    tag = arm if arm else "all"

    # 1. Check if session exists; if not, create it
    print(f"[Launcher] Checking active Colab sessions for '{session_id}'...", flush=True)
    sessions_check = run_cmd(["colab", "sessions"])
    if session_id not in sessions_check.stdout:
        print(f"[Launcher] Creating new Colab session '{session_id}' with A100 GPU...", flush=True)
        new_res = run_cmd(["colab", "new", "--gpu", "A100", "-s", session_id])
        print(f"[Launcher] Session create: {new_res.stdout.strip()} {new_res.stderr.strip()}", flush=True)

    # 2. Write token file onto VM so it survives subshell boundaries
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(token.strip())
        temp_tok = tf.name
    try:
        run_cmd(["colab", "upload", "-s", session_id, temp_tok, "/content/.bifocalgemma-hf-token"])
    finally:
        Path(temp_tok).unlink(missing_ok=True)

    # 3. Build command
    cmd = [
        "colab",
        "exec",
        "-s",
        session_id,
        "--env",
        f"HF_TOKEN={token}",
    ]
    if arm:
        cmd.extend(["--env", f"BIFOCAL_PILOT_ARMS={arm}"])
    if steps:
        cmd.extend(["--env", f"BIFOCAL_PILOT_STEPS={steps}"])
    cmd.extend(
        [
            "--timeout",
            "86400",
            "-f",
            str(script_path),
        ]
    )

    print(
        f"[Launcher] Starting pilot arm '{tag}' (steps={steps}) on Colab session {session_id} (timeout=86400s)...",
        flush=True,
    )
    print(f"[Launcher] Live notifications: https://ntfy.sh/bifocalgemma-daru", flush=True)
    print(f"[Launcher] Auto-sync dataset: https://huggingface.co/datasets/daruokta/bifocalgemma-pilot-results", flush=True)
    print(f"[Launcher] Auto-shutdown: enabled via runtime.unassign()", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_path = Path(f"outputs/colab/pilot_{tag}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

    ret = proc.wait()
    print(f"[Launcher] Process completed with returncode: {ret}", flush=True)

    # 4. Download results
    zip_target = Path(f"outputs/colab/{tag}_results.zip")
    print(f"[Launcher] Checking for local result download...", flush=True)
    dl = subprocess.run(
        [
            "colab",
            "download",
            "-s",
            session_id,
            "/content/bifocalgemma-test.zip",
            str(zip_target),
        ],
        capture_output=True,
        text=True,
    )
    if (not zip_target.is_file() or zip_target.stat().st_size == 0) and token:
        print("[Launcher] Checking Hugging Face dataset backup...", flush=True)
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="daruokta/bifocalgemma-pilot-results",
                repo_type="dataset",
                filename="bifocalgemma-test.zip",
                local_dir=str(zip_target.parent),
                token=token,
            )
            hf_zip = zip_target.parent / "bifocalgemma-test.zip"
            if hf_zip.is_file():
                if zip_target.is_file():
                    zip_target.unlink()
                hf_zip.rename(zip_target)
                print(f"[Launcher] Restored results from HF dataset to {zip_target}", flush=True)
        except Exception as e:
            print(f"[Launcher] HF download note: {e}", flush=True)

    print(f"[Launcher] Ensuring Colab session {session_id} is stopped...", flush=True)
    run_cmd(["colab", "stop", "-s", session_id])
    sys.exit(ret)


if __name__ == "__main__":
    main()
