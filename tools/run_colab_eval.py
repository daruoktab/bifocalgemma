"""Run grounding evaluation script on Colab session and stream output."""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import get_token


def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else "eval-a100"
    arm = sys.argv[2] if len(sys.argv) > 2 else "it_encoder_frozen_then_joint"

    token = get_token()
    if not token:
        print("Error: No local Hugging Face token found.")
        sys.exit(1)

    cmd = [
        "colab",
        "exec",
        "-s",
        session_id,
        "--env",
        f"HF_TOKEN={token}",
        "--timeout",
        "7200",
        "-f",
        "outputs/colab/eval_grounding.py",
    ]

    print(
        f"[Launcher] Running grounding evaluation for arm '{arm}' on Colab session {session_id}...",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_path = Path(f"outputs/colab/grounding_eval_{arm}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

    ret = proc.wait()
    print(f"[Launcher] Evaluation completed with returncode: {ret}", flush=True)

    if ret == 0:
        json_target = Path(f"outputs/colab/grounding_eval_{arm}.json")
        print(f"[Launcher] Downloading {json_target.name} from Colab...", flush=True)
        dl = subprocess.run(
            [
                "colab",
                "download",
                "-s",
                session_id,
                f"/content/grounding_eval_{arm}.json",
                str(json_target),
            ],
            capture_output=True,
            text=True,
        )
        print(f"[Launcher] Download result: {dl.stdout} {dl.stderr}", flush=True)

    sys.exit(ret)


if __name__ == "__main__":
    main()
