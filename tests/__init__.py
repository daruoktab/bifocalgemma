import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct test discovery across all runners
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
