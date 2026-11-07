"""Prepare standalone Colab script to evaluate grounding and constrained decoding on trained checkpoints."""

import ast
import json
from pathlib import Path


def render_eval_script(
    root: Path | None = None,
    humaneval_limit: int = 16,
) -> str:
    if root is None:
        root = Path(__file__).resolve().parents[1]

    manifest = json.loads((root / "notebooks/runtime-manifest.json").read_text(encoding="utf-8"))
    loader = (root / "tools/runtime_bootstrap.py").read_text(encoding="utf-8")

    worker_code = """
import difflib
import json
import keyword
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import runpy
root = Path(sys.argv[1])
runpy.run_path(str(root / 'tools/notebook_launcher.py'))['prepare_runtime'](root)

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from bifocalgemma.config import get_v2_experiment
from bifocalgemma.data.patches import extract_unified_diff
from bifocalgemma.data.releases import fetch_collection
from bifocalgemma.data.tokenization import decode_text, load_verified_tokenizer
from bifocalgemma.evaluation.context import prompt_ids, score_context_case
from bifocalgemma.inference.generation import greedy_decode
from bifocalgemma.models.memory_efficient import configure_memory_efficient_training
from bifocalgemma.repository_study import snapshot_from_json
from bifocalgemma.training.policy import configure_policy_model
from bifocalgemma.training.recipes import unified_recipe
from bifocalgemma.training.stages import atomic_json
from bifocalgemma.training.v2 import DATASET_REPO, DATASET_REVISION, download_model

PYTHON_BUILTINS = [
    'Exception', 'ValueError', 'abs', 'all', 'any', 'bool', 'cls', 'dict', 'enumerate',
    'filter', 'float', 'int', 'isinstance', 'len', 'list', 'map', 'max', 'min', 'next',
    'object', 'open', 'print', 'range', 'reversed', 'round', 'self', 'set', 'slice',
    'sorted', 'str', 'sum', 'super', 'tuple', 'type', 'zip'
]


def normalize_and_recount_patch(completion_text: str, snapshot: Any, default_path: str = "solution.py") -> str | None:
    raw = completion_text.strip()
    try:
        patch = extract_unified_diff(raw)
    except ValueError:
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\\n|\\n```$", "", raw.strip()).strip()
        if "@@ " not in stripped:
            return None
        if not stripped.startswith("diff --git "):
            if not stripped.startswith("--- "):
                stripped = f"--- a/{default_path}\\n+++ b/{default_path}\\n" + stripped
            patch = f"diff --git a/{default_path} b/{default_path}\\n" + stripped
        else:
            patch = stripped
    if not patch.endswith("\\n"):
        patch += "\\n"
    source_files = {f.path: f.content for f in snapshot.files}
    if default_path not in source_files:
        return None
    with tempfile.TemporaryDirectory(prefix="pilot-recount-") as temp_dir:
        work = Path(temp_dir)
        for rel_path, content in source_files.items():
            target = work / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        proc = subprocess.run(
            ["git", "apply", "--recount", "--whitespace=nowarn", "-"],
            input=patch.encode("utf-8"),
            cwd=work,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        changed = (work / default_path).read_text(encoding="utf-8")
        source = source_files[default_path]
        if changed == source:
            return None
        return "diff --git a/solution.py b/solution.py\\n" + "".join(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                changed.splitlines(keepends=True),
                fromfile="a/solution.py",
                tofile="b/solution.py",
            )
        )


def evaluate_grounding(
    root: Path,
    *,
    adapter_zip: Path,
    arm_name: str = "it_encoder_frozen_then_joint",
    humaneval_limit: int = 16,
) -> dict[str, Any]:
    print(f"=== Starting Grounding Evaluation for {arm_name} ===", flush=True)
    cache = root / "cache"
    token = os.environ.get("HF_TOKEN")
    base_spec = get_v2_experiment("BifocalGemma-E2B-E2B")
    spec = replace(base_spec, encoder=base_spec.decoder) if "it_encoder" in arm_name else base_spec

    # Locate or unpack adapter
    direct_adapter = Path("/content/stage_adapters.safetensors")
    if direct_adapter.is_file():
        adapter_file = direct_adapter
        print(f"Using direct adapter from {adapter_file} ({adapter_file.stat().st_size} bytes)", flush=True)
    else:
        extract_dir = root / "extracted_adapters" / arm_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(adapter_zip) as z:
            for name in z.namelist():
                if "stage2_joint/stage_adapters.safetensors" in name:
                    target_path = extract_dir / "stage_adapters.safetensors"
                    target_path.write_bytes(z.read(name))
                    print(f"Extracted adapter to {target_path}", flush=True)
                    break

        adapter_file = extract_dir / "stage_adapters.safetensors"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"stage_adapters.safetensors not found in {adapter_zip}")

    print("Downloading and initializing backbone model...", flush=True)
    model, _ = download_model(spec, cache, token, lambda e: None)
    policy = unified_recipe("symmetric")
    configure_policy_model(model, policy)
    memory_options = {
        "attention_backend": "tiled_eager_fp32",
        "query_chunk_size": 2048,
        "loss_chunk_size": 128,
        "gradient_checkpointing": True,
    }
    configure_memory_efficient_training(model, memory_options)

    print("Loading adapter weights...", flush=True)
    adapters = load_file(str(adapter_file))
    model.load_state_dict(adapters, strict=False)
    print(f"Successfully loaded {len(adapters)} adapter tensors.", flush=True)

    tok_cache = cache / "tokenizers"
    dec_tok, _ = load_verified_tokenizer(spec.decoder, tok_cache)
    assert spec.encoder is not None
    enc_tok, _ = load_verified_tokenizer(spec.encoder, tok_cache)

    humaneval_dir = cache / "benchmarks/humanevalfix_python_v2"
    if not (humaneval_dir / "eval_private/specs.jsonl").is_file():
        import shutil
        shutil.rmtree(humaneval_dir, ignore_errors=True)
        print(f"Fetching evaluation collection with gold to {humaneval_dir}...", flush=True)
        fetch_collection(
            DATASET_REPO,
            DATASET_REVISION,
            humaneval_dir.name,
            humaneval_dir,
            expected_role="eval",
            include_gold=True,
            token=token,
        )
    directory = humaneval_dir
    rows = [
        json.loads(line)
        for line in (directory / "eval_inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = {
        r["instance_id"]: r
        for r in (
            json.loads(line)
            for line in (directory / "eval_private/specs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    priority_ids = ["Python/0", "Python/1", "Python/39"]
    priority = [r for r in rows if r["instance_id"] in priority_ids]
    others = [r for r in rows if r["instance_id"] not in priority_ids]
    selected = (priority + others)[:humaneval_limit]

    model.eval()
    results: dict[str, Any] = {
        "arm_name": arm_name,
        "evaluated_cases": len(selected),
        "conditions": {},
    }

    conditions = [
        "unconstrained_baseline",
        "instruction_file_grounded",
        "prefix_grounded_header",
        "prefix_grounded_with_recount",
    ]

    for cond in conditions:
        print(f"\\n--- Evaluating Condition: {cond} ---", flush=True)
        cases = []
        for index, row in enumerate(selected):
            identifier = row["instance_id"]
            gold_row = gold[identifier]
            snapshot = snapshot_from_json(directory / gold_row["snapshot_file"])
            target_file = snapshot.files[0].path if snapshot.files else "solution.py"

            enc_ids, orig_prefix_ids = prompt_ids(
                row, spec, enc_tok, dec_tok, encoder_limit=32768
            )

            # Determine prefix and instruction based on condition
            if cond == "unconstrained_baseline":
                curr_prefix_ids = orig_prefix_ids
                seed_header = ""
            elif cond == "instruction_file_grounded":
                grounded_prompt = (
                    f"Return only a unified diff modifying '{target_file}' fulfilling the instruction.\n"
                    + row["instruction"]
                )
                text = dec_tok.apply_chat_template(
                    [{"role": "user", "content": grounded_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                curr_prefix_ids = tuple(dec_tok.encode(text, add_special_tokens=False))
                seed_header = ""
            elif cond in ("prefix_grounded_header", "prefix_grounded_with_recount"):
                # Seed the assistant message with diff header
                seed_header = f"diff --git a/{target_file} b/{target_file}\\n--- a/{target_file}\\n+++ b/{target_file}\\n"
                prompt_text = (
                    f"Return only a unified diff modifying '{target_file}' fulfilling the instruction.\n"
                    + row["instruction"]
                )
                messages = [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": seed_header},
                ]
                text = dec_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                curr_prefix_ids = tuple(dec_tok.encode(text.rstrip(), add_special_tokens=False))

            with torch.inference_mode():
                generated = greedy_decode(
                    model,
                    enc_ids,
                    curr_prefix_ids,
                    max_new_tokens=384,
                    max_total_decoder_tokens=41472,
                    use_cache=True,
                )

            tokens = generated.token_ids
            while tokens and tokens[-1] in (1, 106, 50):
                tokens = tokens[:-1]
            gen_text = decode_text(dec_tok, tokens, skip_special_tokens=False)
            if seed_header and not gen_text.startswith(seed_header):
                full_text = seed_header + gen_text
            else:
                full_text = gen_text

            strict_score = score_context_case(
                row,
                gold_row,
                full_text,
                directory=directory,
                python_executable=sys.executable,
                allow_execution=True,
            )

            recounted_score = None
            if cond == "prefix_grounded_with_recount" and not strict_score.get("patch_applied"):
                canonical = normalize_and_recount_patch(full_text, snapshot, default_path=target_file)
                if canonical is not None:
                    recounted_score = score_context_case(
                        row,
                        gold_row,
                        canonical,
                        directory=directory,
                        python_executable=sys.executable,
                        allow_execution=True,
                    )

            effective_score = (
                recounted_score if (recounted_score and recounted_score.get("patch_applied")) else strict_score
            )
            app = bool(effective_score.get("patch_applied"))
            fn_pass = bool(effective_score.get("functional_pass"))
            print(f"[{cond}] Case {index+1}/{len(selected)} ({identifier}): Applied={app}, Pass={fn_pass}, Target={target_file}", flush=True)
            cases.append(
                {
                    "instance_id": identifier,
                    "target_file": target_file,
                    "patch_applied": app,
                    "functional_pass": fn_pass,
                    "reason": effective_score.get("reason"),
                    "preview": full_text[:200],
                }
            )

        applied_count = sum(1 for c in cases if c["patch_applied"])
        pass_count = sum(1 for c in cases if c["functional_pass"])
        results["conditions"][cond] = {
            "applied_count": applied_count,
            "applied_rate": applied_count / len(cases),
            "pass_count": pass_count,
            "pass_rate": pass_count / len(cases),
            "cases": cases,
        }
        print(f"Condition '{cond}' Result: Applied = {applied_count}/{len(cases)} ({applied_count/len(cases):.1%}), Pass = {pass_count}/{len(cases)} ({pass_count/len(cases):.1%})", flush=True)

    out_file = Path(f"/content/grounding_eval_{arm_name}.json")
    atomic_json(out_file, results)
    print(f"Saved complete evaluation report to {out_file}", flush=True)
    return results


if __name__ == '__main__':
    arm = sys.argv[2] if len(sys.argv) > 2 else "it_encoder_frozen_then_joint"
    zip_path = Path(f"/content/{arm}_results.zip")
    evaluate_grounding(root, adapter_zip=zip_path, arm_name=arm, humaneval_limit=16)
"""

    child = "import os, runpy, sys\nfrom pathlib import Path\n" + worker_code
    ast.parse(child)

    runner = (
        '"""Generated Colab A100 grounding evaluator for BifocalGemma."""\n'
        + loader
        + "\nimport subprocess\nimport sys\nfrom pathlib import Path\n"
        + f"manifest = {manifest!r}\n"
        + "root = load_runtime(manifest['repo_id'], manifest['filename'], manifest['revision'], manifest['sha256'])\n"
        + "child = Path('/content/eval_worker.py')\n"
        + f"child.write_text({child!r}, encoding='utf-8')\n"
        + "arm = sys.argv[1] if len(sys.argv) > 1 else 'it_encoder_frozen_then_joint'\n"
        + "proc = subprocess.run([sys.executable, '-u', str(child), str(root), arm])\n"
        + "sys.exit(proc.returncode)\n"
    )
    ast.parse(runner)
    return runner


def main() -> None:
    content = render_eval_script()
    out = Path("outputs/colab/eval_grounding.py")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"Generated {out} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
