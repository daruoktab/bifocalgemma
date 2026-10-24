"""Deterministic source bundle shared by standalone Molab notebooks."""

import base64
import hashlib
import io
import zipfile
from pathlib import Path


def source_bundle_bytes(root: Path):
    paths = sorted(
        p
        for directory in (root / "src", root / "configs", root / "examples/repository_study")
        for p in directory.rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".jsonl", ".md"}
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths + [
            root / "LICENSE",
            root / "tools/notebook_dependencies.py",
            root / "tools/notebook_launcher.py",
        ]:
            item = zipfile.ZipInfo(
                path.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, path.read_bytes())
    payload = buffer.getvalue()
    return payload


def source_bundle(root: Path):
    payload = source_bundle_bytes(root)
    encoded = base64.b85encode(payload).decode()
    literal = (
        "(\n"
        + "\n".join("        " + repr(encoded[i : i + 100]) for i in range(0, len(encoded), 100))
        + "\n    )"
    )
    return literal, hashlib.sha256(payload).hexdigest()
