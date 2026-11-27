"""Verify the generated standalone repository study notebook stays synchronized."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_notebook_bundle_is_synchronized():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/generate_repository_notebook.py"), "--check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
