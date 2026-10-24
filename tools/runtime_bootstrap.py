"""Small stdlib loader embedded in the six cloud notebooks."""


def load_runtime(repo_id, filename, revision, expected_sha, notebook_file=None):
    import hashlib
    import io
    import urllib.request
    import zipfile
    from pathlib import Path
    from uuid import uuid4

    root = (Path("outputs/corpus_runtime") / expected_sha[:16]).resolve()
    local = Path(notebook_file).resolve().parent / "runtime" / filename if notebook_file else None
    cached = root / "runtime.zip"
    source = local if local is not None and local.is_file() else cached
    data = source.read_bytes() if source.is_file() else b""
    if hashlib.sha256(data).hexdigest() != expected_sha:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(64 * 2**20 + 1)
    if len(data) > 64 * 2**20 or hashlib.sha256(data).hexdigest() != expected_sha:
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
    return root
