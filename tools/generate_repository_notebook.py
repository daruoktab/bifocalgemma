"""Generate a standalone repository study notebook with a deterministic source bundle."""

import argparse
import base64
import hashlib
import io
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    buffer = io.BytesIO()
    paths = sorted(
        [
            p
            for directory in (ROOT / "src", ROOT / "configs", ROOT / "examples/repository_study")
            for p in directory.rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".jsonl", ".md"}
        ]
    )
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths + [ROOT / "LICENSE"]:
            info = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    encoded = base64.b85encode(payload).decode()
    literal = (
        "(\n"
        + "\n".join("        " + repr(encoded[i : i + 100]) for i in range(0, len(encoded), 100))
        + "\n    )"
    )
    text = (ROOT / "tools/templates/repository_notebook.py.txt").read_text(encoding="utf-8")
    text = text.replace("{{BUNDLE_PAYLOAD}}", literal).replace(
        "{{BUNDLE_SHA}}", hashlib.sha256(payload).hexdigest()
    )
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
