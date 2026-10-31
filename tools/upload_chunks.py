"""Upload large binary file to Colab session in safe chunks."""

import base64
import subprocess
import sys
from pathlib import Path


def upload_in_chunks(
    session_id: str, local_path: Path, remote_path: str, chunk_size_bytes: int = 4 * 1024 * 1024
) -> None:
    local_path = Path(local_path)
    data = local_path.read_bytes()
    total_bytes = len(data)
    print(
        f"[Uploader] Uploading {local_path.name} ({total_bytes} bytes) to {remote_path} on session {session_id}..."
    )

    # Clear remote file first
    init_code = f"open('{remote_path}', 'wb').close(); print('Initialized {remote_path}')\n"
    subprocess.run(["colab", "exec", "-s", session_id], input=init_code, text=True, check=True)

    offset = 0
    chunk_idx = 0
    while offset < total_bytes:
        chunk = data[offset : offset + chunk_size_bytes]
        b64 = base64.b64encode(chunk).decode("ascii")
        code = f"""
import base64
with open('{remote_path}', 'ab') as f:
    f.write(base64.b64decode('{b64}'))
print('Appended chunk {chunk_idx}:', {len(chunk)}, 'bytes. Current total:', open('{remote_path}', 'rb').seek(0, 2))
"""
        proc = subprocess.run(
            ["colab", "exec", "-s", session_id], input=code, text=True, capture_output=True
        )
        if proc.returncode != 0:
            print(f"[Uploader] Error uploading chunk {chunk_idx}:", proc.stderr)
            sys.exit(1)
        offset += len(chunk)
        print(
            f"[Uploader] Chunk {chunk_idx + 1} ({offset}/{total_bytes} bytes, {offset / total_bytes:.1%}) uploaded."
        )
        chunk_idx += 1

    print(f"[Uploader] Successfully uploaded {total_bytes} bytes to {remote_path}!")


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "eval-a100"
    upload_in_chunks(
        session,
        Path("outputs/colab/adapter_itenc.safetensors.gz"),
        "/content/adapter_itenc.safetensors.gz",
    )
    # Decompress on remote VM
    decompress_code = """
import gzip, shutil
with gzip.open('/content/adapter_itenc.safetensors.gz', 'rb') as f_in:
    with open('/content/stage_adapters.safetensors', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
import os
print('Decompressed /content/stage_adapters.safetensors:', os.path.getsize('/content/stage_adapters.safetensors'))
"""
    p = subprocess.run(
        ["colab", "exec", "-s", session], input=decompress_code, text=True, capture_output=True
    )
    print("[Uploader] Decompression output:", p.stdout, p.stderr)
