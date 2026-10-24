"""Render six standalone v2 notebooks from one template and identical source bundle."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from notebook_bundle import source_bundle_bytes

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "e2b-e2b": "Encoder E2B / Decoder E2B IT",
    "e4b-e2b": "Encoder E4B / Decoder E2B IT",
    "e2b-e4b": "Encoder E2B / Decoder E4B IT",
    "e4b-e4b": "Encoder E4B / Decoder E4B IT",
    "e2b-it": "Baseline E2B IT",
    "e4b-it": "Baseline E4B IT",
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


def use_embedded_imports(code):
    """Keep marimo's package scanner from requesting our bundled source on PyPI.

    Rewrite only project import statements, preserving comments/cell structure.
    Imports inside embedded ZIP source remain ordinary Python imports.
    """
    lines = code.splitlines(keepends=True)
    imports = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and (node.module == "bifocalgemma" or node.module.startswith("bifocalgemma."))
    ]
    for node in sorted(imports, key=lambda item: item.lineno, reverse=True):
        indent = " " * node.col_offset
        replacements = []
        for alias in node.names:
            if alias.name == "*":
                raise ValueError("Bundled notebook imports must name explicit symbols.")
            replacements.append(
                f"{indent}{alias.asname or alias.name} = __import__('importlib').import_module({node.module!r}).{alias.name}\n"
            )
        lines[node.lineno - 1 : node.end_lineno] = replacements
    return "".join(lines)


def render(family, manifest=None):
    template = (ROOT / "tools/templates/training_v2.py.txt").read_text(encoding="utf-8")
    for marker, path in {
        "{{RUNTIME_LOADER}}": ROOT / "tools/runtime_bootstrap.py",
    }.items():
        definitions = module_definitions(path)
        # Marker is indented inside a cell; subsequent lines need the same indentation.
        template = template.replace(marker, definitions.replace("\n", "\n    "))
    if manifest is None:
        manifest = json.loads((ROOT / "notebooks/runtime-manifest.json").read_text())
    if hashlib.sha256(source_bundle_bytes(ROOT)).hexdigest() != manifest["sha256"]:
        raise RuntimeError(
            "Runtime changed. Publish the new immutable bundle with --publish before generating notebooks."
        )
    template = template.replace("{{RUNTIME_MANIFEST}}", repr(manifest))
    template = template.replace("{{MODEL_KEY}}", family).replace("{{TITLE}}", VARIANTS[family])
    template = use_embedded_imports(template)
    formatter = shutil.which("ruff")
    if formatter is None:
        raise RuntimeError("Install the project dev dependency ruff before generation.")
    filename = f"notebooks/train_{family.replace('-', '_')}.py"
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
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Prepare a reviewable local runtime archive without publishing.",
    )
    args = parser.parse_args()
    if args.stage:
        payload = source_bundle_bytes(ROOT)
        digest = hashlib.sha256(payload).hexdigest()
        root = ROOT / "outputs/runtime-release"
        root.mkdir(parents=True, exist_ok=True)
        filename = "runtime-" + digest[:16] + ".zip"
        (root / filename).write_bytes(payload)
        candidate = {
            "repo_id": "daruokta/bifocalgemma-v2-assets",
            "filename": filename,
            "sha256": digest,
            "bytes": len(payload),
            "path": str(root / filename),
        }
        (root / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
        preview = root / "preview"
        (preview / "runtime").mkdir(parents=True, exist_ok=True)
        (preview / "runtime" / filename).write_bytes(payload)
        pointer = dict(
            repo_id=candidate["repo_id"], filename=filename, revision="LOCAL-PREVIEW", sha256=digest
        )
        (preview / "runtime-manifest.json").write_text(json.dumps(pointer, indent=2) + "\n")
        for family in VARIANTS:
            (preview / f"train_{family.replace('-', '_')}.py").write_text(
                render(family, pointer), encoding="utf-8"
            )
        print(json.dumps(candidate))
        return
    if args.publish:
        from huggingface_hub import HfApi, hf_hub_download

        payload = source_bundle_bytes(ROOT)
        digest = hashlib.sha256(payload).hexdigest()
        filename = "runtime-" + digest[:16] + ".zip"
        repo_id = "daruokta/bifocalgemma-v2-assets"
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        commit = api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=filename,
            path_or_fileobj=payload,
            commit_message="Publish immutable BifocalGemma notebook runtime",
        )
        remote = hf_hub_download(repo_id, filename, repo_type="dataset", revision=commit.oid)
        if hashlib.sha256(Path(remote).read_bytes()).hexdigest() != digest:
            raise ValueError("Published runtime checksum mismatch.")
        manifest = dict(repo_id=repo_id, filename=filename, revision=commit.oid, sha256=digest)
        target = ROOT / "notebooks/runtime"
        target.mkdir(exist_ok=True)
        (target / filename).write_bytes(payload)
        (ROOT / "notebooks/runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for family in VARIANTS:
        path = ROOT / f"notebooks/train_{family.replace('-', '_')}.py"
        code = render(family)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != code:
                raise SystemExit(f"Notebook out of sync: {path}")
        else:
            path.write_text(code, encoding="utf-8")
    print("Six standalone v2 notebooks synchronized; one fixed model and one recipe each.")


if __name__ == "__main__":
    main()
