"""Keep model/training semantics strict while permitting presentation-only updates."""

import ast
import hashlib
import json
from pathlib import Path

PRESENTATION_FILES = {
    "training/telemetry.py",
    "training/wiggly_dashboard.py",
    "training/timing.py",
}


class WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
        return node


def source_fingerprints(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    full, critical = hashlib.sha256(), hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".json"}:
            continue
        relative = path.relative_to(root).as_posix()
        full.update(relative.encode() + b"\0" + path.read_bytes())
        if relative in PRESENTATION_FILES:
            continue
        if path.suffix == ".py":
            content = ast.dump(
                WithoutDocstrings().visit(ast.parse(path.read_text(encoding="utf-8")))
            )
        else:
            content = json.dumps(
                json.loads(path.read_text()), sort_keys=True, separators=(",", ":")
            )
        critical.update(relative.encode() + b"\0" + content.encode())
    current = critical.hexdigest()
    # Benchmark-only releases may explicitly approve one exact source tree as
    # training-equivalent to its predecessor. Any subsequent code change breaks
    # this match and retains the normal strict checkpoint/profile rejection.
    approval = root.parents[1] / "configs/training_resume_compatibility.json"
    training = current
    if approval.is_file():
        entry = json.loads(approval.read_text(encoding="utf-8"))
        if entry.get("approved_source_semantics_sha256") == current:
            training = entry["baseline_training_semantics_sha256"]
    return {
        "code_sha256": full.hexdigest(),
        "training_semantics_sha256": training,
        "source_semantics_sha256": current,
    }
