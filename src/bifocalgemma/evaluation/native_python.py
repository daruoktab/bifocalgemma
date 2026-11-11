"""Native Python benchmark scoring, explicit code execution, fixed denominators.

A subprocess/venv is not a security sandbox. Tests run only when requested.
"""

import ast
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from bifocalgemma.contracts import RepositorySnapshot
from bifocalgemma.data.context import canonical_snapshot_hash
from bifocalgemma.data.patches import apply_patch_to_snapshot, extract_unified_diff, safe_repo_path


@dataclass(frozen=True)
class NativeTaskResult:
    instance_id: str
    status: str
    patch_applied: bool
    functional_pass: bool | None
    reason: str = ""
    stdout: str = ""
    stderr: str = ""


def score_python_patch(
    instance_id: str,
    snapshot: RepositorySnapshot,
    prediction: str,
    test_script: str,
    *,
    python_executable: str,
    expected_snapshot_sha256: str,
    allow_code_execution: bool = False,
    timeout_seconds: float = 30,
    expected_python_version: str | None = None,
) -> NativeTaskResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_must_be_positive")
    if canonical_snapshot_hash(snapshot) != expected_snapshot_sha256:
        raise ValueError("evaluation_snapshot_hash_mismatch")
    try:
        fixed = apply_patch_to_snapshot(snapshot, extract_unified_diff(prediction))
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        return NativeTaskResult(instance_id, "PATCH_FAILED", False, False, str(error)[:500])
    if not allow_code_execution:
        return NativeTaskResult(
            instance_id, "STATIC_ONLY", True, None, "Native execution disabled."
        )
    if not test_script.strip():
        return NativeTaskResult(instance_id, "INFRA_ERROR", True, None, "No Python test script.")
    # Gold patch, hidden tests and credentials never enter the model input.
    with tempfile.TemporaryDirectory(prefix="bifocal-eval-") as name:
        root = Path(name).resolve()
        for file in fixed.files:
            path = (root / safe_repo_path(file.path)).resolve()
            if not path.is_relative_to(root):
                raise ValueError("evaluation_path_escape")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(file.content.encode("utf-8"))
        sentinel = "BIFOCAL_EVAL_FINISHED_" + uuid.uuid4().hex
        script = root / ("_bifocal_evaluator_" + uuid.uuid4().hex + ".py")
        # Keep the final marker unreachable after an early exit in model code.
        # unittest.main(exit=False) is wrapped so failing/empty suites still fail.
        try:
            tree = ast.parse(test_script)
        except SyntaxError as error:
            return NativeTaskResult(
                instance_id, "INFRA_ERROR", True, None, "Invalid evaluator script: " + str(error)
            )
        unittest_names = {"unittest"}
        main_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                unittest_names.update(
                    item.asname or item.name for item in node.names if item.name == "unittest"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "unittest":
                main_names.update(
                    item.asname or item.name for item in node.names if item.name == "main"
                )

        class UnittestMain(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                is_main = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "main"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in unittest_names
                ) or (isinstance(node.func, ast.Name) and node.func.id in main_names)
                if is_main:
                    node.keywords = [k for k in node.keywords if k.arg != "exit"] + [
                        ast.keyword(arg="exit", value=ast.Constant(False))
                    ]
                    return ast.copy_location(
                        ast.Call(
                            func=ast.Name(id="_bifocal_check_suite", ctx=ast.Load()),
                            args=[node],
                            keywords=[],
                        ),
                        node,
                    )
                return node

        transformed = ast.unparse(ast.fix_missing_locations(UnittestMain().visit(tree)))
        harness = (
            "import sys\n"
            f"sys.path.insert(0, {str(root)!r})\n"
            "def _bifocal_check_suite(program):\n"
            "    result = program.result\n"
            "    if result is None or result.testsRun == 0 or not result.wasSuccessful():\n"
            "        raise SystemExit(1)\n"
            "    return program\n"
            "_scope = globals()\n"
            f"exec(compile({transformed!r}, '<hidden-benchmark-tests>', 'exec'), _scope)\n"
            f"print({sentinel!r})\n"
        )
        script.write_text(harness, encoding="utf-8")
        # Minimal environment; no HF/GitHub/API credentials or inherited PYTHONPATH.
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LD_LIBRARY_PATH"}
        env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        env.update(
            HOME=str(root),
            USERPROFILE=str(root),
            TMP=str(root),
            TEMP=str(root),
            TMPDIR=str(root),
            PYTHONNOUSERSITE="1",
            PYTHONHASHSEED="0",
            PYTHONIOENCODING="utf-8",
        )
        if expected_python_version:
            expected = re.search(r"\d+\.\d+(?:\.\d+)?", expected_python_version)
            if expected is None:
                return NativeTaskResult(
                    instance_id, "INFRA_ERROR", True, None, "Invalid expected Python version."
                )
            try:
                version = subprocess.run(
                    [
                        python_executable,
                        "-I",
                        "-c",
                        "import platform;print(platform.python_version())",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                return NativeTaskResult(instance_id, "INFRA_ERROR", True, None, str(error)[:500])
            actual = version.stdout.strip()
            if version.returncode or not (
                actual == expected.group() or actual.startswith(expected.group() + ".")
            ):
                return NativeTaskResult(
                    instance_id,
                    "INFRA_ERROR",
                    True,
                    None,
                    f"Python runtime mismatch: expected {expected.group()}, actual {actual}.",
                )
        try:
            result = subprocess.run(
                [python_executable, "-I", str(script)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return NativeTaskResult(instance_id, "TIMEOUT", True, False, "Native test timeout.")
        except OSError as error:
            return NativeTaskResult(instance_id, "INFRA_ERROR", True, None, str(error)[:500])
        passed = result.returncode == 0 and sentinel in result.stdout
        status = "PASS" if passed else "FAIL"
        if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
            status = "INFRA_ERROR"
        if root != Path(name).resolve():
            raise ValueError("temporary_evaluation_root_changed")
        return NativeTaskResult(
            instance_id,
            status,
            True,
            True if passed else (None if status == "INFRA_ERROR" else False),
            "",
            result.stdout.replace(sentinel, "<finished>")[-4000:],
            result.stderr[-4000:],
        )


def summarize_native_results(results: list[NativeTaskResult]) -> dict:
    if len({item.instance_id for item in results}) != len(results):
        raise ValueError("duplicate_evaluation_instance_ids")
    attempted = len(results)
    passed = sum(result.functional_pass is True for result in results)
    scored = sum(result.functional_pass is not None for result in results)
    return {
        "attempted": attempted,
        "functional_scored": scored,
        "passed": passed,
        "functional_pass_rate": passed / attempted if attempted and scored == attempted else None,
        "conservative_pass_rate": passed / attempted if attempted and scored else None,
        "fully_evaluated": attempted > 0 and scored == attempted,
        "patch_apply_rate": sum(r.patch_applied for r in results) / attempted
        if attempted
        else None,
        "statuses": {
            status: sum(r.status == status for r in results)
            for status in sorted({r.status for r in results})
        },
        "results": [asdict(result) for result in results],
    }
