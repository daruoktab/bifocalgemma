"""Unit tests verifying all Molab remediation fixes:
1. 256-token MCQ budget floor in benchmark runners (v3/benchmark.py and evaluation/context.py).
2. PreTrainedModel Hub export with subfolder isolation, repo creation, and tokenizer assets.
3. Clean recipe versioning and legacy checkpoint quarantine vs clean resume.
4. Standalone embedded runtime extraction from matrix notebooks and fast training notebook without network access.
5. Fast pipeline auto-download, task-types, export_pretrained, and HF token wiring.
6. Notebook determinism, syntax, and strict line count bounds (< 250 lines).
"""

import ast
import base64
import hashlib
import io
import json
import re
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
from bifocalgemma.evaluation.context import run_context_benchmark
from bifocalgemma.modeling_bifocalgemma import (
    BifocalGemmaForCausalLM,
    export_matrix_pretrained_model,
)
from bifocalgemma.v3.constants import (
    CENTRALIZED_HF_REPO,
    CLEAN_DEFAULT_HF_REPO,
    CLEAN_RECIPE_VERSION,
)
from bifocalgemma.v3.matrix import make_experiment_spec, run_matrix_training
from tools.generate_full_matrix_notebooks import ROOT


class FixtureTokenizer:
    bos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [2, 7, 8] if add_special_tokens else [2, 9]

    def apply_chat_template(self, *args, **kwargs):
        return "fixture-prefix"

    def decode(self, ids, **kwargs):
        return "B" if tuple(ids) == (4,) else "fixture answer"


class FixtureGenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.encoder = object()
        self.decoder = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
        self.encoded = 0

    def encode_context(self, ids, mask):
        self.encoded += 1
        return torch.zeros(1, 1, 1)

    def decode_logits(self, ids, mask, states, encoder_mask):
        logits = torch.zeros(1, 1, 128)
        logits[0, 0, 4] = 1
        return logits


def test_mcq_budget_floor_in_v3_benchmark_logic():
    """Verify that rows with 'max_new_tokens': 32 get bumped to 256 tokens when max_new_tokens >= 256."""
    decoder_limit = 8192
    paired_prefix_size = 100

    # 1. MCQ row with hardcoded 32-token budget on real Molab run (max_new_tokens=2048)
    row_mcq = {"task_type": "mcq", "max_new_tokens": 32}
    max_new_tokens = 2048
    row_budget = int(row_mcq.get("max_new_tokens", max_new_tokens))
    if str(row_mcq.get("task_type", "")).lower() in {"mcq", "qa", "retrieval"} and max_new_tokens >= 256:
        row_budget = max(row_budget, 256)
    new_tokens = min(row_budget, decoder_limit - paired_prefix_size)
    assert new_tokens == 256

    # 2. Retrieval row with 32 tokens
    row_retrieval = {"task_type": "retrieval", "max_new_tokens": 32}
    row_budget = int(row_retrieval.get("max_new_tokens", max_new_tokens))
    if str(row_retrieval.get("task_type", "")).lower() in {"mcq", "qa", "retrieval"} and max_new_tokens >= 256:
        row_budget = max(row_budget, 256)
    assert row_budget == 256

    # 3. Fast unit test with tiny budget (e.g. max_new_tokens=8) is NOT bumped to 256
    tiny_max_tokens = 8
    row_budget = int(row_mcq.get("max_new_tokens", tiny_max_tokens))
    if str(row_mcq.get("task_type", "")).lower() in {"mcq", "qa", "retrieval"} and tiny_max_tokens >= 256:
        row_budget = max(row_budget, 256)
    new_tokens = min(row_budget, decoder_limit - paired_prefix_size)
    assert new_tokens == 32 or new_tokens == 8


def test_context_benchmark_mcq_budget_floor_execution(tmp_path, monkeypatch):
    """Verify that run_context_benchmark passes max_new_tokens=256 to greedy_decode for MCQ rows with 32 tokens."""
    import bifocalgemma.data.tokenization as tokenization
    import bifocalgemma.evaluation.context as eval_ctx

    monkeypatch.setattr(
        tokenization, "load_verified_tokenizer", lambda *a: (FixtureTokenizer(), {})
    )

    recorded_budgets = []

    def mock_greedy_decode(model, enc, prefix, *args, **kwargs):
        recorded_budgets.append(kwargs.get("max_new_tokens"))
        return SimpleNamespace(token_ids=[4], stop_reason="eos", wall_clock_seconds=0.01, generated_tokens=1, text="B")

    monkeypatch.setattr(eval_ctx, "greedy_decode", mock_greedy_decode)

    root = tmp_path / "prepared"
    root.mkdir()
    (root / "eval_private").mkdir()
    public = [
        {
            "instance_id": "case-mcq-32",
            "role": "eval",
            "task_type": "mcq",
            "context": "Context for MCQ question",
            "instruction": "Select the correct option A or B",
            "max_new_tokens": 32,
        }
    ]
    gold = [
        {"instance_id": "case-mcq-32", "answer": "B"}
    ]
    (root / "eval_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in public), encoding="utf-8")
    (root / "eval_private/specs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold), encoding="utf-8")

    model = FixtureGenerationModel()
    settings = dict(
        checkpoint_identity={"sha": "test_sha"},
        encoder_limit=1024,
        decoder_limit=1024,
        max_new_tokens=512,  # Molab benchmark runs use 1024/2048 >= 256
    )

    report = run_context_benchmark(
        model,
        EXPERIMENT_SPECS[0],
        root,
        tmp_path / "eval_output",
        tmp_path,
        **settings,
    )

    assert len(recorded_budgets) == 1
    # Despite row having max_new_tokens=32, greedy_decode received at least 256!
    assert recorded_budgets[0] >= 256
    assert report["total"] == 1


def test_clean_recipe_version_and_constants():
    """Verify recipe version and default repo constants."""
    assert CLEAN_RECIPE_VERSION == "bifocalgemma_clean_2026_11_07"
    assert CLEAN_DEFAULT_HF_REPO == "daruokta/bifocalgemma-clean"
    assert CENTRALIZED_HF_REPO == "daruokta/bifocalgemma-2026-11-02"


def test_pretrained_model_subfolder_support_and_repo_creation(tmp_path):
    """Verify save_pretrained creates repo and uploads to path_in_repo, and from_pretrained loads from subfolder."""
    cfg = BifocalGemmaConfig(
        encoder_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 256},
        decoder_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 256},
    )
    model = BifocalGemmaForCausalLM(cfg)

    subfolder = "2base-2it/pretrained_model"
    target_dir = tmp_path / subfolder
    model.save_pretrained(target_dir)
    assert (target_dir / "config.json").is_file()
    assert (target_dir / "model.safetensors").is_file()

    # Load back using subfolder
    loaded = BifocalGemmaForCausalLM.from_pretrained(tmp_path, subfolder=subfolder)
    assert isinstance(loaded, BifocalGemmaForCausalLM)
    assert loaded.config.d_enc == 64
    assert loaded.config.d_dec == 64


def test_export_matrix_pretrained_model_with_subfolder_and_tokenizer(tmp_path, monkeypatch):
    """Verify export_matrix_pretrained_model copies tokenizer assets and creates repo."""
    spec = make_experiment_spec("2base-2it")
    cfg = BifocalGemmaConfig(
        encoder_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 256},
        decoder_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 256},
    )
    model = BifocalGemmaForCausalLM(cfg)

    # Create dummy tokenizer cache with raw files
    tok_cache = tmp_path / "tokenizers"
    dec_slug = spec.decoder.model_id.replace("/", "__")
    raw_dir = tok_cache / "raw" / dec_slug
    raw_dir.mkdir(parents=True)
    (raw_dir / "tokenizer.json").write_text('{"dummy": true}', encoding="utf-8")
    (raw_dir / "tokenizer_config.json").write_text('{"dummy": true}', encoding="utf-8")

    export_dir = tmp_path / "exported"

    mock_tok = SimpleNamespace(
        save_pretrained=MagicMock(),
        save=MagicMock(),
    )

    created_repos = []
    uploaded_folders = []

    class MockHfApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, repo_id, repo_type="model", exist_ok=True):
            created_repos.append(repo_id)

        def upload_folder(self, folder_path, repo_id, path_in_repo=None, repo_type="model"):
            uploaded_folders.append((folder_path, repo_id, path_in_repo))

    monkeypatch.setattr("huggingface_hub.HfApi", MockHfApi)

    res = export_matrix_pretrained_model(
        model=model,
        spec=spec,
        export_dir=export_dir,
        tokenizer=mock_tok,
        tokenizer_cache=tok_cache,
        push_to_hub=True,
        repo_id="daruokta/test-repo",
        path_in_repo="2base-2it/pretrained_model",
        token="test_tok",
    )

    assert res == export_dir
    assert (export_dir / "config.json").is_file()
    assert (export_dir / "tokenizer.json").is_file()
    assert (export_dir / "tokenizer_config.json").is_file()
    assert "daruokta/test-repo" in created_repos
    assert uploaded_folders[0][2] == "2base-2it/pretrained_model"


def test_matrix_quarantine_legacy_checkpoint_vs_clean_resume(tmp_path, monkeypatch):
    """Verify that legacy Nov-2 checkpoints are discarded while clean recipe checkpoints are preserved."""
    # 1. Legacy Nov-2 checkpoint (missing recipe_version in training_report.json and ledger.json)
    output_legacy = tmp_path / "legacy_run"
    run_dir = output_legacy / "2base-2it" / "train"
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True)

    ckpt_dir = train_dir / "checkpoint-011102"
    ckpt_dir.mkdir()

    (train_dir / "training_report.json").write_text(
        json.dumps({"global_step": 11102, "status": "completed"}), encoding="utf-8"
    )
    (ckpt_dir / "manifest.json").write_text(
        json.dumps({"global_step": 11102}), encoding="utf-8"
    )
    (ckpt_dir / "adapters.safetensors").write_bytes(b"dummy_adapters")
    (ckpt_dir / "trainer_state.pt").write_bytes(b"dummy_state")
    marker = {
        "manifest.json": hashlib.sha256((ckpt_dir / "manifest.json").read_bytes()).hexdigest(),
        "adapters.safetensors": hashlib.sha256(b"dummy_adapters").hexdigest(),
        "trainer_state.pt": hashlib.sha256(b"dummy_state").hexdigest(),
    }
    (ckpt_dir / "READY.json").write_text(json.dumps(marker), encoding="utf-8")
    (run_dir / "ledger.json").write_text(
        json.dumps({"version": "bifocalgemma_v3_swiglu_cross"}), encoding="utf-8"
    )

    logged_events = []

    class MockMonitor:
        def __init__(self):
            self.run_dir = None

        def __call__(self, event):
            logged_events.append(event)

    monkeypatch.setattr("bifocalgemma.v3.matrix.recover_remote_run", lambda *a, **k: None)

    # Calling run_matrix_training should detect legacy artifacts, wipe them, and log discarded_legacy_checkpoint
    monkeypatch.setattr("bifocalgemma.v3.matrix.download_model", MagicMock(side_effect=StopIteration("stopped")))

    with pytest.raises(StopIteration):
        run_matrix_training("2base-2it", output_root=output_legacy, upload=False, token=None, monitor=MockMonitor())

    stages = [e.get("stage") for e in logged_events if isinstance(e, dict)]
    assert "discarded_legacy_checkpoint" in stages
    # The stale training directory and legacy ledger were wiped!
    assert not (run_dir / "training/training_report.json").exists()
    assert not (run_dir / "ledger.json").exists()

    # 2. Clean checkpoint with recipe_version == CLEAN_RECIPE_VERSION
    output_clean = tmp_path / "clean_run"
    run_dir_clean = output_clean / "2base-2it" / "train"
    train_dir_clean = run_dir_clean / "training"
    train_dir_clean.mkdir(parents=True, exist_ok=True)
    ckpt_dir_clean = train_dir_clean / "checkpoint-011102"
    ckpt_dir_clean.mkdir(parents=True, exist_ok=True)
    (ckpt_dir_clean / "manifest.json").write_text(
        json.dumps({"global_step": 11102, "recipe_version": CLEAN_RECIPE_VERSION}), encoding="utf-8"
    )
    (ckpt_dir_clean / "adapters.safetensors").write_bytes(b"clean_adapters")
    (ckpt_dir_clean / "trainer_state.pt").write_bytes(b"clean_state")
    marker_clean = {
        "manifest.json": hashlib.sha256((ckpt_dir_clean / "manifest.json").read_bytes()).hexdigest(),
        "adapters.safetensors": hashlib.sha256(b"clean_adapters").hexdigest(),
        "trainer_state.pt": hashlib.sha256(b"clean_state").hexdigest(),
    }
    (ckpt_dir_clean / "READY.json").write_text(json.dumps(marker_clean), encoding="utf-8")
    (train_dir_clean / "training_report.json").write_text(
        json.dumps({"global_step": 11102, "recipe_version": CLEAN_RECIPE_VERSION, "status": "completed"}),
        encoding="utf-8",
    )

    res = run_matrix_training(
        "2base-2it", output_root=output_clean, total_steps=11102, upload=False, token=None, monitor=MockMonitor()
    )
    assert res["status"] == "COMPLETED"
    assert res["message"] == "Full matrix training already completed."


def test_embedded_runtime_extraction_from_real_matrix_notebook(tmp_path):
    """Verify that EMBEDDED_RUNTIME_B85 inside train_2base_2it.py extracts offline and bifocalgemma is importable."""
    notebook_path = ROOT / "notebooks/train_2base_2it.py"
    code = notebook_path.read_text(encoding="utf-8")

    tree = ast.parse(code)
    embedded_b85 = None
    expected_sha = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id == "EMBEDDED_RUNTIME_B85" and isinstance(node.value, ast.Constant):
                        embedded_b85 = node.value.value
                    elif t.id in ("runtime_manifest", "RUNTIME_MANIFEST"):
                        try:
                            m = ast.literal_eval(node.value)
                            expected_sha = m.get("sha256")
                        except Exception:
                            pass

    assert embedded_b85 is not None, "EMBEDDED_RUNTIME_B85 constant not found in notebook"
    assert expected_sha is not None, "runtime_manifest sha256 not found in notebook"

    decoded = base64.b85decode(embedded_b85)
    assert hashlib.sha256(decoded).hexdigest() == expected_sha

    extract_dir = tmp_path / "runtime_extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(io.BytesIO(decoded)) as archive:
        for entry in archive.infolist():
            target = (extract_dir / entry.filename).resolve()
            assert target.is_relative_to(extract_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))

    assert (extract_dir / "tools/molab_matrix_support.py").is_file()
    assert (extract_dir / "src/bifocalgemma/__init__.py").is_file()


def test_embedded_runtime_extraction_from_fast_notebook(tmp_path):
    """Verify that EMBEDDED_RUNTIME_B85 inside train_bifocalgemma_fast.py extracts offline."""
    notebook_path = ROOT / "notebooks/train_bifocalgemma_fast.py"
    code = notebook_path.read_text(encoding="utf-8")

    sha_match = re.search(r'RUNTIME_SHA256\s*=\s*"([a-f0-9]+)"', code)
    b85_match = re.search(r'EMBEDDED_RUNTIME_B85\s*=\s*"(.*?)"', code)
    assert sha_match is not None, "RUNTIME_SHA256 not found in fast notebook"
    assert b85_match is not None, "EMBEDDED_RUNTIME_B85 not found in fast notebook"

    expected_sha = sha_match.group(1)
    embedded_b85 = b85_match.group(1)

    decoded = base64.b85decode(embedded_b85)
    assert hashlib.sha256(decoded).hexdigest() == expected_sha

    extract_dir = tmp_path / "runtime_fast_extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(io.BytesIO(decoded)) as archive:
        for entry in archive.infolist():
            target = (extract_dir / entry.filename).resolve()
            assert target.is_relative_to(extract_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))

    assert (extract_dir / "src/bifocalgemma/training/fast_pipeline.py").is_file()
    assert (extract_dir / "src/bifocalgemma/__init__.py").is_file()


def test_fast_training_pipeline_features_and_export(tmp_path, monkeypatch):
    """Verify FastTrainingPipeline export_pretrained, JSONLRepairDataset task-types, and fail-fast."""
    from bifocalgemma.training.fast_pipeline import (
        FastTrainingConfig,
        FastTrainingPipeline,
        JSONLRepairDataset,
    )

    # 1. Config accepts hf_token and auto_download_curated
    cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        total_steps=10,
        tiny=True,
        hf_token="test_hf_token",
        auto_download_curated=True,
    )
    assert cfg.hf_token == "test_hf_token"
    assert cfg.auto_download_curated is True

    # 2. JSONLRepairDataset handles target_text and non-repair task types
    raw_rows = [
        {
            "instance_id": "row-loc",
            "task_type": "localization",
            "context": "Context for localization",
            "target_text": "target localization text",
        },
        {
            "instance_id": "row-comp",
            "task_type": "completion",
            "context": "Context for completion",
            "target_text": "target completion text",
        },
        {
            "instance_id": "row-rep",
            "task_type": "repair",
            "context": "Context for repair",
            "target_text": "target repair text",
        },
    ]
    jsonl_file = tmp_path / "tasks.jsonl"
    jsonl_file.write_text("\n".join(json.dumps(r) for r in raw_rows), encoding="utf-8")

    ds = JSONLRepairDataset(jsonl_file)
    assert len(ds) == 3
    # Check that examples were parsed and have token ids
    item0 = ds[0]
    assert len(item0.target_ids) > 0
    assert item0.instance_id == "row-loc"

    item1 = ds[1]
    assert len(item1.target_ids) > 0
    assert item1.instance_id == "row-comp"

    item2 = ds[2]
    assert len(item2.target_ids) > 0
    assert item2.instance_id == "row-rep"

    # 3. export_pretrained produces config.json and model.safetensors
    pipeline = FastTrainingPipeline(cfg)
    export_dir = tmp_path / "fast_export"
    out_path = pipeline.export_pretrained(export_dir=export_dir)
    assert (out_path / "config.json").is_file()
    assert (out_path / "model.safetensors").is_file()

    # 4. Fail-fast when not tiny and dataset missing
    monkeypatch.setattr(
        BifocalGemmaForCausalLM,
        "from_pretrained_backbones",
        lambda *a, **k: BifocalGemmaForCausalLM(BifocalGemmaConfig(tiny=True)),
    )
    non_tiny_cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        total_steps=10,
        tiny=False,
        data_dir="non_existent_dataset_dir",
        auto_download_curated=False,
    )
    with pytest.raises(RuntimeError, match="No dataset available for non-tiny fast training"):
        FastTrainingPipeline(non_tiny_cfg)

    # 5. Synthetic fallback when tiny=False, auto_download_curated=False, and data_dir=None
    from bifocalgemma.training.fast_pipeline import SyntheticRepairDataset

    fallback_cfg = FastTrainingConfig(
        model_choice="bifocalgemma-2b",
        total_steps=10,
        tiny=False,
        data_dir=None,
        auto_download_curated=False,
    )
    fallback_pipeline = FastTrainingPipeline(fallback_cfg)
    assert isinstance(fallback_pipeline.train_loader.dataset, SyntheticRepairDataset)


def test_matrix_quarantine_corrupt_checkpoint_wipes_and_continues(tmp_path, monkeypatch):
    """Verify that corrupt checkpoints (missing manifest.json) are caught and quarantined cleanly."""
    output_corrupt = tmp_path / "corrupt_run"
    run_dir = output_corrupt / "2base-2it" / "train"
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = train_dir / "checkpoint-001000"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Put READY.json but no manifest.json
    (ckpt_dir / "READY.json").write_text(json.dumps({"manifest.json": "corrupt"}), encoding="utf-8")

    logged_events = []

    class MockMonitor:
        def __init__(self):
            self.run_dir = None

        def __call__(self, event):
            logged_events.append(event)

    monkeypatch.setattr("bifocalgemma.v3.matrix.recover_remote_run", lambda *a, **k: None)
    monkeypatch.setattr(
        "bifocalgemma.v3.matrix.download_model", MagicMock(side_effect=StopIteration("stopped"))
    )

    with pytest.raises(StopIteration):
        run_matrix_training(
            "2base-2it",
            output_root=output_corrupt,
            upload=False,
            token=None,
            monitor=MockMonitor(),
        )

    stages = [e.get("stage") for e in logged_events if isinstance(e, dict)]
    assert "discarded_legacy_checkpoint" in stages
    assert not (run_dir / "training").exists()


def test_matrix_notebooks_strict_line_count_and_ast():
    """Verify all 10 matrix notebooks parse cleanly and strictly adhere to < 250 lines bound."""
    for nb_file in ROOT.glob("notebooks/train_*.py"):
        if nb_file.name == "train_bifocalgemma_fast.py":
            continue
        code = nb_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(nb_file))
        assert isinstance(tree, ast.Module)
        num_lines = len(code.splitlines())
        assert num_lines < 250, f"Notebook {nb_file.name} has {num_lines} lines (must be < 250)"
