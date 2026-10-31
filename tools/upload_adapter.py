"""Split and upload adapter to Colab using native colab upload."""

import math
import subprocess
from pathlib import Path


def main() -> None:
    session = "eval-a100"
    source = Path("outputs/colab/adapter_itenc.safetensors.gz")
    parts_dir = Path("outputs/colab/tmp_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    data = source.read_bytes()
    chunk_size = 8 * 1024 * 1024
    num_chunks = math.ceil(len(data) / chunk_size)
    print(f"Splitting {len(data)} bytes into {num_chunks} chunks of 8MB...")

    part_paths = []
    for i in range(num_chunks):
        part_path = parts_dir / f"part_{i:02d}.bin"
        part_data = data[i * chunk_size : (i + 1) * chunk_size]
        part_path.write_bytes(part_data)
        part_paths.append(part_path)

    for i, p in enumerate(part_paths):
        print(f"Uploading part {i + 1}/{num_chunks} ({p.stat().st_size} bytes)...", flush=True)
        res = subprocess.run(
            ["colab", "upload", "-s", session, str(p), f"/content/tmp_part_{i:02d}.bin"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(" ", res.stdout.strip())

    print("Reassembling on remote VM...", flush=True)
    reassemble_script = Path("outputs/colab/tmp_reassemble.py")
    reassemble_script.write_text(
        f"""
import gzip, os, shutil
with open('/content/adapter_itenc.safetensors.gz', 'wb') as out_f:
    for i in range({num_chunks}):
        part_file = f'/content/tmp_part_{{i:02d}}.bin'
        with open(part_file, 'rb') as in_f:
            shutil.copyfileobj(in_f, out_f)
        os.remove(part_file)

print('Reassembled .gz size:', os.path.getsize('/content/adapter_itenc.safetensors.gz'))

with gzip.open('/content/adapter_itenc.safetensors.gz', 'rb') as f_in:
    with open('/content/stage_adapters.safetensors', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print('Decompressed safetensors size:', os.path.getsize('/content/stage_adapters.safetensors'))
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["colab", "exec", "-s", session, "-f", str(reassemble_script)],
        capture_output=True,
        text=True,
        check=True,
    )
    print("Reassemble output:\n", proc.stdout)
    print("Done!")


if __name__ == "__main__":
    main()
