"""Render matched standalone notebooks; only THREE control defaults differ."""

import argparse
import ast
import shutil
import subprocess
from pathlib import Path

from notebook_bundle import source_bundle

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "symmetric": ("BifocalGemma-E2B-E2B", "outputs/training_symmetric"),
    "asymmetric": ("BifocalGemma-E2B-E4B", "outputs/training_asymmetric"),
    "baseline": ("Gemma-4-E2B-IT", "outputs/training_baseline"),
}


def module_definitions(path):
    code = path.read_text(encoding="utf-8")
    parts = []
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(code, node)
            if segment is None:
                raise ValueError(f"Cannot read source definition from {path}")
            parts.append(segment)
    return "\n\n".join(parts)


def render(family):
    template = (ROOT / "tools/templates/training_notebook.py.txt").read_text(encoding="utf-8")
    for marker, path in {
        "{{DATA_HELPERS}}": ROOT / "src/bifocalgemma/data/offline_sft.py",
        "{{TRAIN_HELPERS}}": ROOT / "src/bifocalgemma/training/offline_sft.py",
        "{{POLICY_HELPERS}}": ROOT / "src/bifocalgemma/training/policy.py",
    }.items():
        definitions = module_definitions(path)
        # Marker is indented inside a cell; subsequent lines need the same indentation.
        template = template.replace(marker, definitions.replace("\n", "\n    "))
    payload, bundle_sha = source_bundle(ROOT)
    template = template.replace("{{BUNDLE_PAYLOAD}}", payload).replace("{{BUNDLE_SHA}}", bundle_sha)
    experiment, output = VARIANTS[family]
    policy_family = "decoder_only" if family == "baseline" else family
    template = template.replace('FAMILY = "symmetric"', f'FAMILY = "{policy_family}"', 1)
    template = template.replace(
        'EXPERIMENT = "BifocalGemma-E2B-E2B"', f'EXPERIMENT = "{experiment}"', 1
    )
    template = template.replace(
        'OUTPUT_ROOT = "outputs/training_symmetric"', f'OUTPUT_ROOT = "{output}"', 1
    )
    ast.parse(template)
    formatter = shutil.which("ruff")
    if formatter is None:
        raise RuntimeError("Install the project dev dependency ruff before generation.")
    filename = f"notebooks/bifocalgemma_train_{family}.py"
    template = subprocess.run(
        [formatter, "check", "--select", "I", "--fix", "--stdin-filename", filename, "-"],
        input=template,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout
    return subprocess.run(
        [formatter, "format", "--stdin-filename", filename, "-"],
        input=template,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for family in VARIANTS:
        path = ROOT / f"notebooks/bifocalgemma_train_{family}.py"
        code = render(family)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != code:
                raise SystemExit(f"Notebook out of sync: {path}")
        else:
            path.write_text(code, encoding="utf-8")
    print("Training notebooks synchronized; only control-center family/experiment/output differ.")


if __name__ == "__main__":
    main()
