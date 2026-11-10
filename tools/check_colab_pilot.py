"""Check live status of the detached BifocalGemma pilot on Colab."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else "pilot-a100"

    print(f"[Status] Checking Colab session '{session_id}'...", flush=True)
    sessions = subprocess.run(["colab", "sessions"], capture_output=True, text=True)
    if session_id not in sessions.stdout:
        print(
            f"[Status] Colab session '{session_id}' is NOT active (completed & released, or stopped).",
            flush=True,
        )
        return

    check_code = """import json, os, subprocess
from pathlib import Path

pid_file = Path('/content/pilot_daemon.pid')
pid = int(pid_file.read_text(encoding='utf-8').strip()) if pid_file.is_file() else None
alive = False
if pid:
    try:
        os.kill(pid, 0)
        alive = True
    except (ValueError, OSError):
        alive = False

gpu_info = ""
try:
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
except Exception:
    gpu_info = "N/A"

log_lines = []
log_file = Path('/content/pilot_daemon.log')
if log_file.is_file():
    with log_file.open('r', encoding='utf-8', errors='replace') as f:
        log_lines = f.readlines()[-25:]

events = []
evt_file = Path('/content/bifocalgemma-test/events.jsonl')
if evt_file.is_file():
    with evt_file.open('r', encoding='utf-8', errors='replace') as f:
        events = [line.strip() for line in f.readlines()[-5:] if line.strip()]

print(json.dumps({
    "pid": pid,
    "alive": alive,
    "gpu": gpu_info,
    "last_events": events,
    "tail_log": "".join(log_lines),
}))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(check_code)
        temp_file = tf.name

    try:
        res = subprocess.run(
            ["colab", "exec", "-s", session_id, "--timeout", "30", "-f", temp_file],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(f"[Status] Error querying session: {res.stderr}")
            return
        data = json.loads(res.stdout.strip().splitlines()[-1])
        status_str = "RUNNING" if data["alive"] else "STOPPED/FINISHED"
        print(
            f"[Status] Process: PID={data['pid']} | State={status_str} | GPU Utilization/VRAM={data['gpu']}"
        )
        if data["last_events"]:
            print("[Status] Latest Events:")
            for ev in data["last_events"]:
                print(f"  {ev}")
        print("\n[Status] Recent Daemon Log Output:")
        print("-" * 60)
        print(data["tail_log"].strip())
        print("-" * 60)
    finally:
        Path(temp_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
