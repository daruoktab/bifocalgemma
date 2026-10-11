"""Exercise generated HF-only loading and completed-run upload without real network/GPU."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    ROOT / "notebooks" / f"bifocalgemma_train_{family}.py" for family in ("symmetric", "asymmetric")
]
MO = SimpleNamespace(
    md=lambda text: text,
    vstack=lambda items: None,
    ui=SimpleNamespace(table=lambda rows, **kw: rows),
)


def cell_function(path, output):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(isinstance(v, ast.Name) and v.id == output for v in ast.walk(n))
    )
    node = copy.deepcopy(node)
    node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def test_defaults_and_no_curation_or_sample_controls():
    for path in PATHS + [ROOT / "notebooks/bifocalgemma_repository_study.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        controls = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        assert controls["RUN_TRAINING"] is True
        assert controls["MAX_STEPS"] == 0 and controls["EPOCHS"] == 1
        assert "PREPARE_FULL_POOL" not in controls
        assert "DATA_SOURCE" not in controls and "TRAINING_DATA_MODE" not in controls
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert not calls & {"prepare_full_corpus", "prepare_repository_corpus", "dataset_rows"}
        if path in PATHS:
            assert controls["HF_UPLOAD_ENABLED"] is True
            assert not {"RAW_LIMIT", "TRAIN_LIMIT", "DEV_LIMIT", "DEV_FRACTION"} & controls.keys()


@pytest.mark.parametrize("exists", [False, True])
def test_loader_always_verifies_pinned_collection_even_for_warm_cache(tmp_path, exists):
    folder = tmp_path / "dataset"
    if exists:
        folder.mkdir()
    calls = []
    limits = {"encoder": 8192, "pair_decoder": 2048, "baseline_decoder": 10752}
    manifest = {
        "report": {"tokens": {"limits": limits}, "training_plan": {"updates_full_epoch": 1}},
        "record_ids": {"train": ["t"], "dev": ["v"]},
        "fingerprint": "fixture",
    }
    modules = {
        "bifocalgemma.data.releases": SimpleNamespace(
            fetch_collection=lambda *args, **kw: calls.append((args, kw))
        ),
        "bifocalgemma.data.offline_sft": SimpleNamespace(
            load_curated_pool=lambda path: ({}, manifest),
            validate_curated_pool_limits=lambda *args, **kw: limits,
        ),
    }
    function = cell_function(PATHS[0], "corpus_pool_status")
    result = function(
        True,
        "owner/data",
        "a" * 40,
        "frozen",
        str(folder),
        str(tmp_path / "tokens"),
        "instruction",
        8192,
        2048,
        Path,
        SimpleNamespace(import_module=lambda name: modules[name]),
        SimpleNamespace(fetch_tokenizer_assets=lambda *args, **kw: None),
        SimpleNamespace(get_experiment=lambda name: name),
        "test-read-credential",
        MO,
    )
    assert len(calls) == 1
    assert calls[0][0][:3] == ("owner/data", "a" * 40, "frozen")
    assert calls[0][1]["token"] == "test-read-credential"
    assert result[0]["status"] == "READY"


@pytest.mark.parametrize("status", ["DISABLED", "FAILED", "SKIPPED"])
def test_upload_enabled_does_not_require_credentials_or_touch_hub_before_completion(
    tmp_path, status
):
    def forbidden_api(**kwargs):
        raise AssertionError("An incomplete run must not touch Hub")

    function = cell_function(PATHS[0], "upload_result")
    result = function(
        "BifocalGemma-E2B-E2B",
        "symmetric",
        "owner/model",
        True,
        forbidden_api,
        "run",
        {"role": "none"},
        False,
        MO,
        tmp_path,
        {"status": status},
    )
    assert result[0]["status"] == "SKIPPED"


def test_completed_run_upload_uses_family_experiment_run_path(tmp_path):
    (tmp_path / "training_report.json").write_text("{}", encoding="utf-8")
    calls = []

    class Api:
        def __init__(self, **kwargs):
            assert kwargs["token"] == "test-write-credential"

        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            return "https://example.invalid/commit/fixture"

    function = cell_function(PATHS[0], "upload_result")
    result = function(
        "BifocalGemma-E2B-E4B",
        "asymmetric",
        "owner/model",
        True,
        Api,
        "run",
        {"role": "write"},
        "test-write-credential",
        MO,
        tmp_path,
        {"status": "COMPLETED"},
    )
    assert result[0]["status"] == "COMPLETED"
    assert calls[1][1]["path_in_repo"] == "asymmetric/BifocalGemma-E2B-E4B/run"
    assert calls[1][1]["repo_type"] == "model"
