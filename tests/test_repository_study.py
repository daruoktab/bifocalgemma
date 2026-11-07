"""Repository preparation, leakage boundaries, greedy decode and native scoring."""

import copy
import difflib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
import torch

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.context import canonical_snapshot_hash, select_repository_context
from bifocalgemma.data.offline_sft import editing_record, split_editing_records
from bifocalgemma.data.patches import apply_patch_to_snapshot, extract_unified_diff, patch_paths
from bifocalgemma.data.repository import (
    RepositoryTask,
    prepare_repository_record,
    repository_task_from_row,
)
from bifocalgemma.evaluation.native_python import score_python_patch, summarize_native_results
from bifocalgemma.inference.generation import greedy_decode
from bifocalgemma.models.v2.native import build_native_tiny_model
from bifocalgemma.models.v2.pretrained import load_sft_adapters_for_inference
from bifocalgemma.training.offline_sft import save_sft_checkpoint
from bifocalgemma.training.policy import (
    build_training_groups,
    configure_policy_model,
    default_training_profiles,
)


class ScoringOptions(TypedDict):
    python_executable: str
    expected_snapshot_sha256: str


class ExecutingScoringOptions(ScoringOptions):
    allow_code_execution: bool


def diff(path, before, after):
    return (
        "diff --git a/"
        + path
        + " b/"
        + path
        + "\n"
        + "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="a/" + path,
                tofile="b/" + path,
            )
        )
    )


def fixture():
    before = "def add_one(value):\n    return value\n"
    after = "def add_one(value):\n    return value + 1\n"
    snapshot = RepositorySnapshot(
        "fixture/project",
        "a" * 40,
        (
            RepositoryFile("helpers.py", before),
            RepositoryFile(
                "calculator.py",
                "from helpers import add_one\ndef calculate(value):\n    return add_one(value)\n",
            ),
        ),
    )
    return snapshot, diff("helpers.py", before, after)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [2] + list(text.encode()) if add_special_tokens else list(text.encode())


def test_multifile_add_delete_unicode_and_path_guards():
    snapshot = RepositorySnapshot(
        "fixture/project",
        "a" * 40,
        (
            RepositoryFile("old.py", 'x = "ä½ å¥½"\n'),
            RepositoryFile("remove.txt", "old\n"),
        ),
    )
    patch = diff("old.py", 'x = "ä½ å¥½"\n', 'x = "Ã©"\n')
    patch += "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+y = 2\n"
    patch += "diff --git a/remove.txt b/remove.txt\ndeleted file mode 100644\n--- a/remove.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    fixed = apply_patch_to_snapshot(snapshot, patch)
    assert {f.path: f.content for f in fixed.files} == {"new.py": "y = 2\n", "old.py": 'x = "Ã©"\n'}
    for unsafe in ("../outside.py", ".git/config", "C:/bad.py"):
        with pytest.raises(ValueError):
            patch_paths(diff(unsafe, "x\n", "y\n"))
    with pytest.raises(ValueError, match="unsupported_patch_operation"):
        patch_paths(patch.replace("new file mode 100644", "new file mode 120000"))
    # Preserve the trailing blank context line; stripping spaces corrupts this diff.
    last_context = diff("a.py", "x = 1\n\n", "x = 2\n\n")
    assert extract_unified_diff(last_context) == last_context


def test_source_whitelist_and_gold_separation():
    snapshot, patch = fixture()
    raw = dict(
        instance_id="x",
        repo=snapshot.repo,
        base_commit=snapshot.base_commit,
        problem_statement="Fix calculator.calculate using add_one.",
        patch=patch,
        hints_text="GOLD SECRET",
        test_patch="HIDDEN SECRET",
        pr_description="ANSWER SECRET",
    )
    task = repository_task_from_row("swe_fixer", raw)
    row, hidden = prepare_repository_record(
        task, snapshot, CharacterTokenizer(), blocked_repos=set(), max_context_tokens=1000
    )
    assert hidden is None and row["target_patch"] == patch
    assert not any(
        secret in row["context"] for secret in ("GOLD SECRET", "HIDDEN SECRET", "ANSWER SECRET")
    )
    changed = {**raw, "patch": patch.replace("+    return value + 1", "+    return value + 2")}
    row2, _ = prepare_repository_record(
        repository_task_from_row("swe_fixer", changed),
        snapshot,
        CharacterTokenizer(),
        blocked_repos=set(),
        max_context_tokens=1000,
    )
    assert row2["context"] == row["context"]
    eval_task = RepositoryTask(
        "eval",
        "Qurrent/RES-Q",
        "rev",
        "eval",
        snapshot.repo,
        snapshot.base_commit,
        task.issue,
        patch,
        "unknown",
        "assert True",
        "python3.13",
        "",
    )
    public, private = prepare_repository_record(
        eval_task, snapshot, CharacterTokenizer(), blocked_repos=set(), max_context_tokens=1000
    )
    assert private is not None
    assert (
        "target_patch" not in public
        and "test_script" not in public
        and private["gold_patch"] == patch
    )
    with pytest.raises(ValueError, match="evaluation_record_cannot_enter_training"):
        editing_record(public, source="editpackft")


def test_quarantine_fork_alias_and_grouped_ingestion():
    snapshot, patch = fixture()
    task = RepositoryTask(
        "x",
        "test/train",
        "rev",
        "train",
        snapshot.repo,
        snapshot.base_commit,
        "Fix add_one.",
        patch,
        "unknown",
    )
    with pytest.raises(ValueError, match="benchmark_repository_quarantine"):
        prepare_repository_record(
            task,
            snapshot,
            CharacterTokenizer(),
            blocked_repos={"origin/project"},
            aliases={"fixture/project": "origin/project"},
            max_context_tokens=1000,
        )
    records = []
    for index in range(3):
        row, _ = prepare_repository_record(
            task, snapshot, CharacterTokenizer(), blocked_repos=set(), max_context_tokens=1000
        )
        row.update(instance_id=str(index), group="repo/" + str(index), repos="repo/" + str(index))
        records.append(editing_record(row, source="editpackft"))
    train, dev = split_editing_records(records, train_limit=2, dev_limit=1)
    assert not ({r["group"] for r in train} & {r["group"] for r in dev})
    malicious = {**row, "context_selection": {"gold_used": True}}
    with pytest.raises(ValueError, match="not_target_free"):
        editing_record(malicious, source="editpackft")


def test_context_budget_and_hidden_tests_are_not_rendered():
    snapshot, _ = fixture()
    snapshot = RepositorySnapshot(
        snapshot.repo,
        snapshot.base_commit,
        snapshot.files + (RepositoryFile("tests/hidden.py", "GOLD ANSWER"),),
    )
    context, meta = select_repository_context(
        snapshot, "Fix calculate and add_one.", CharacterTokenizer(), max_tokens=500
    )
    assert len(CharacterTokenizer().encode(context)) <= 500 and not meta["gold_used"]
    assert "GOLD ANSWER" not in context
    with pytest.raises(ValueError, match="issue_exceeds"):
        select_repository_context(snapshot, "a" * 100, CharacterTokenizer(), max_tokens=10)


def test_native_score_pass_fail_disabled_and_runtime(monkeypatch):
    snapshot, patch = fixture()
    kwargs: ScoringOptions = dict(
        python_executable=sys.executable, expected_snapshot_sha256=canonical_snapshot_hash(snapshot)
    )
    test = 'from calculator import calculate\nassert calculate(3) == 4\nimport os\nassert "HF_TOKEN" not in os.environ\n'
    monkeypatch.setenv("HF_TOKEN", "fixture-only-not-a-real-token")
    static = score_python_patch("static", snapshot, patch, test, **kwargs)
    assert static.status == "STATIC_ONLY" and static.functional_pass is None
    passed = score_python_patch("pass", snapshot, patch, test, allow_code_execution=True, **kwargs)
    assert passed.status == "PASS"
    failed = score_python_patch(
        "fail",
        snapshot,
        patch.replace("+    return value + 1", "+    return value + 2"),
        test,
        allow_code_execution=True,
        **kwargs,
    )
    assert failed.status == "FAIL"
    mismatch = score_python_patch(
        "runtime",
        snapshot,
        patch,
        test,
        allow_code_execution=True,
        expected_python_version="python2.7",
        **kwargs,
    )
    assert mismatch.status == "INFRA_ERROR"
    report = summarize_native_results([static, passed, failed, mismatch])
    assert (
        report["attempted"] == 4
        and report["functional_pass_rate"] is None
        and report["conservative_pass_rate"] == 0.25
    )


def test_unittest_results_and_early_exit_do_not_false_pass():
    snapshot, patch = fixture()
    kwargs: ExecutingScoringOptions = dict(
        python_executable=sys.executable,
        expected_snapshot_sha256=canonical_snapshot_hash(snapshot),
        allow_code_execution=True,
    )
    script = "import unittest\nfrom calculator import calculate\nclass Checks(unittest.TestCase):\n    def test_fix(self):\n        self.assertEqual(calculate(3), 4)\nunittest.main()\n"
    assert score_python_patch("good", snapshot, patch, script, **kwargs).status == "PASS"
    assert (
        score_python_patch(
            "bad", snapshot, patch, script.replace("), 4)", "), 5)"), **kwargs
        ).status
        == "FAIL"
    )
    empty = "import unittest\nunittest.main()\n"
    assert score_python_patch("empty", snapshot, patch, empty, **kwargs).status == "FAIL"
    early = patch.replace("+    return value + 1", "+    raise SystemExit(0)")
    assert score_python_patch("exit", snapshot, early, script, **kwargs).status == "FAIL"


class ScriptedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.encoder = object()
        self.decoder = SimpleNamespace(config=SimpleNamespace(vocab_size=128))
        self.encodes = 0
        self.steps = 0

    def encode_context(self, ids, mask):
        self.encodes += 1
        return torch.zeros(1, ids.shape[1], 1)

    def decode_logits(self, ids, mask, states, enc_mask):
        self.steps += 1
        logits = torch.zeros(1, ids.shape[1], 128)
        logits[..., 18 if self.steps == 1 else 106] = 1
        return logits


def test_greedy_encoder_once_stop_mode_and_vocab_guards():
    model = ScriptedModel()
    result = greedy_decode(model, (2, 3), (2, 105), max_new_tokens=4)
    assert result.token_ids == (18, 106) and result.stop_reason == "eos"
    assert model.encodes == 1 and model.training
    events = []
    monitored = greedy_decode(ScriptedModel(), (2, 3), (2, 105), max_new_tokens=4,
                              progress=events.append)
    assert monitored == result
    assert [e["generated_tokens"] for e in events if e["stage"] == "benchmark_decoding"] == [0, 1, 2]
    with pytest.raises(ValueError, match="vocabulary"):
        greedy_decode(model, (262143,), (2,), max_new_tokens=1)
    with pytest.raises(ValueError, match="budget"):
        greedy_decode(model, (), (2,), max_new_tokens=5, max_total_decoder_tokens=4)


@pytest.mark.parametrize("index,family", [(0, "symmetric"), (2, "asymmetric"), (4, "decoder_only")])
def test_inference_v2_adapter_roundtrip_and_baseline_group(index, family):
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        policy = default_training_profiles()[family]
        policy["rank"] = 2
        policy["alpha"] = 4
        model = build_native_tiny_model(EXPERIMENT_SPECS[index])
        fresh = copy.deepcopy(model)
        configure_policy_model(model, policy)
        groups = build_training_groups(model, policy)
        if family == "decoder_only":
            assert [g["name"] for g in groups] == ["decoder"]
        optimizer = torch.optim.AdamW(groups)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data.fill_(0.01)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = save_sft_checkpoint(
                model,
                optimizer,
                {"global_step": 0},
                Path(temp),
                {"experiment": model.spec.name, "training_policy": policy},
                checkpoint_format="bifocalgemma_notebook_sft_v2",
            )
            restored = load_sft_adapters_for_inference(fresh, checkpoint)
            assert restored["format"].endswith("v2")
            expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert all(
                torch.equal(dict(fresh.named_parameters())[n], p) for n, p in expected.items()
            )
            bad = Path(checkpoint) / "adapters.safetensors"
            bad.write_bytes(bad.read_bytes() + b"bad")
            with pytest.raises(ValueError, match="checksum"):
                load_sft_adapters_for_inference(build_native_tiny_model(model.spec), checkpoint)
    finally:
        torch.set_num_threads(previous)


def test_diagnostic_preparation_separates_reference_and_denominator():
    from bifocalgemma.data.diagnostics import (
        prepare_crosscode_tasks,
        prepare_repoqa_tasks,
        score_diagnostics,
    )

    source = "def one():\n    return 1\n\ndef two():\n    return 2\n"
    needle = {
        "name": "one",
        "path": "a.py",
        "start_byte": 0,
        "end_byte": len("def one():\n    return 1\n"),
        "description": "Return the integer one.",
    }
    public, hidden = prepare_repoqa_tasks(
        {"python": [{"repo": "fixture/r", "content": {"a.py": source}, "needles": [needle]}]},
        CharacterTokenizer(),
        max_context_tokens=200,
    )
    assert "reference" not in public[0] and "def one" in public[0]["context"]
    assert public[0]["official_protocol"] is False
    result = score_diagnostics(hidden, [])
    assert (
        result["attempted"] == 1
        and result["missing_predictions"] == 1
        and result["exact_match_rate"] == 0
    )
    assert (
        score_diagnostics(
            hidden, [{"instance_id": hidden[0]["instance_id"], "text": hidden[0]["reference"]}]
        )["exact_match_rate"]
        == 1
    )
    rows = [
        {
            "metadata": {"task_id": "c1"},
            "prompt": "value = ",
            "crossfile_context": {"text": "public context"},
            "groundtruth": "one()",
        }
    ]
    inputs, private = prepare_crosscode_tasks(rows, CharacterTokenizer(), max_context_tokens=200)
    assert "groundtruth" not in inputs[0] and private[0]["reference"] == "one()"


def test_last_token_logits_match_full_head():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        model = build_native_tiny_model(EXPERIMENT_SPECS[4]).eval()
        ids = torch.tensor([[2, 3, 4]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            full = model.decode_logits(ids, mask)
            last = model.decode_next_logits(ids, mask)
        assert last.shape == (1, 1, 128)
        torch.testing.assert_close(last, full[:, -1:])
    finally:
        torch.set_num_threads(previous)


def test_local_loader_preflights_and_preserves_shared_words(tmp_path):
    from safetensors.torch import save_file

    from bifocalgemma.models.v2.pretrained import load_local_native_model

    original = build_native_tiny_model(EXPERIMENT_SPECS[0])
    assert original.encoder is not None
    decoder = tmp_path / "decoder.safetensors"
    encoder = tmp_path / "encoder.safetensors"
    for component, path in ((original.decoder, decoder), (original.encoder, encoder)):
        save_file(
            {
                "model.language_model." + key: value.clone()
                for key, value in component.state_dict().items()
            },
            path,
        )
    loaded, report = load_local_native_model(
        original.spec, decoder, encoder, device="cpu", dtype=torch.float32, tiny=True
    )
    assert report["weights_loaded"] and not report["source_identity_verified"]
    assert loaded.encoder is not None
    assert loaded.encoder.embed_tokens.weight is loaded.decoder.embed_tokens.weight
    assert (
        loaded.encoder.embed_tokens_per_layer.weight
        is not loaded.decoder.embed_tokens_per_layer.weight
    )
    for name, value in original.decoder.state_dict().items():
        torch.testing.assert_close(loaded.decoder.state_dict()[name], value)
    save_file({"model.language_model.unexpected.weight": torch.zeros(1)}, encoder)
    with pytest.raises(ValueError):
        load_local_native_model(
            original.spec, decoder, encoder, device="cpu", dtype=torch.float32, tiny=True
        )


def test_preparation_reuses_one_verified_snapshot_without_changing_export(monkeypatch):
    import bifocalgemma.data.repository as repository

    snapshot, patch = fixture()
    task = repository_task_from_row(
        "swe_fixer",
        dict(
            instance_id="reuse",
            repo=snapshot.repo,
            base_commit=snapshot.base_commit,
            problem_statement="Fix calculator.calculate using add_one.",
            patch=patch,
        ),
    )
    original = repository.prepare_repository_record(
        task, snapshot, CharacterTokenizer(), blocked_repos=set(), max_context_tokens=1000
    )
    apply = repository.apply_patch_to_snapshot
    calls = []

    def counted(before, target):
        calls.append((before, target))
        return apply(before, target)

    monkeypatch.setattr(repository, "apply_patch_to_snapshot", counted)
    row, hidden, fixed = repository.prepare_repository_record_with_snapshot(
        task, snapshot, CharacterTokenizer(), blocked_repos=set(), max_context_tokens=1000
    )
    assert (row, hidden) == original and len(calls) == 1
    assert canonical_snapshot_hash(fixed) == row["fixed_snapshot_sha256"]
    assert fixed == apply(snapshot, patch)
    assert dict((file.path, file.content) for file in snapshot.files) != dict(
        (file.path, file.content) for file in fixed.files
    )
