"""Generate a standalone repository study notebook with a deterministic source bundle."""

import argparse
import shutil
import subprocess
from pathlib import Path

from notebook_bundle import source_bundle

ROOT = Path(__file__).resolve().parents[1]


def render():
    literal, bundle_sha = source_bundle(ROOT)
    text = (ROOT / "tools/templates/repository_notebook.py.txt").read_text(encoding="utf-8")
    text = text.replace("{{BUNDLE_PAYLOAD}}", literal).replace("{{BUNDLE_SHA}}", bundle_sha)
    ruff = shutil.which("ruff")
    if ruff is None:
        raise RuntimeError("Ruff is required to render the notebook.")
    for args in (["check", "--select", "I", "--fix"], ["format"]):
        text = subprocess.run(
            [ruff, *args, "--stdin-filename", "notebooks/bifocalgemma_repository_study.py", "-"],
            input=text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        ).stdout
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "notebooks/bifocalgemma_repository_study.py"
    text = render()
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            raise SystemExit("Repository notebook bundle is stale.")
    else:
        target.write_text(text, encoding="utf-8")
    print("Repository notebook source bundle synchronized.")


if __name__ == "__main__":
    main()
