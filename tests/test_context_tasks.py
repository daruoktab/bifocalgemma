"""Task boundaries and objective construction must not leak targets or benchmark gold."""

import importlib.util
from pathlib import Path

import pytest

from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.data.context_tasks import (
    normalize_context_task,
    repair_preimage_coverage,
    tokenize_context_task,
)
from bifocalgemma.evaluation.context import (
    completion_metrics,
    normalize_completion_prediction,
    prompt_ids,
    strict_letter,
)


def task():
    return {
        "record_type": "context_sft_v1",
        "role": "train",
        "task_type": "localization",
        "instance_id": "fixture",
        "group": "fixture/repo",
        "instruction": "Find bug",
        "context": "### File: a.py\nx=0",
        "target_text": '["a.py"]',
        "source": "nebius/SWE-rebench-V2",
        "source_key": "swe_rebench_v2",
        "source_revision": "a" * 40,
        "snapshot_sha256": "b" * 64,
        "commit": "c" * 40,
        "repos": "fixture/repo",
        "context_selection": {"gold_used": False, "windows": [{"path": "a.py"}]},
    }


def test_eval_role_and_invisible_localization_target_rejected():
    assert normalize_context_task(task())["target_text"] == '["a.py"]'
    with pytest.raises(ValueError, match="not_training"):
        normalize_context_task({**task(), "role": "eval"})
    with pytest.raises(ValueError, match="not_visible"):
        normalize_context_task({**task(), "target_text": '["hidden.py"]'})
    with pytest.raises(ValueError, match="benchmark_source"):
        normalize_context_task({**task(), "benchmark": "SWE-QA"})


def test_short_answer_scoring_rejects_unrequested_prose():
    assert strict_letter(" B\n") == "B"
    assert strict_letter("B)") == "B"
    assert strict_letter("The answer is B") is None
    assert strict_letter("A or B") is None
    assert completion_metrics("return a + b", "return a+b")["exact_match"]
    assert not completion_metrics("return a-b", "return a+b")["exact_match"]


def test_repair_requires_visible_preimage_without_refilling_context():
    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -10,3 +10,3 @@\n a\n-b\n+c\n d\n"
    selection = {
        "gold_used": False,
        "windows": [{"path": "a.py", "start_line": 10, "end_line": 12}],
    }
    assert repair_preimage_coverage(patch, selection)
    selection["windows"][0]["end_line"] = 11
    assert not repair_preimage_coverage(patch, selection)
    selection["windows"][0]["path"] = "unrelated.py"
    assert not repair_preimage_coverage(patch, selection)


def test_repair_benchmark_requests_same_diff_format_for_pair_and_baseline():
    from types import SimpleNamespace

    from bifocalgemma.config import EXPERIMENT_SPECS

    prompts = []
    decoder = SimpleNamespace(
        bos_token_id=2,
        encode=lambda text, **kwargs: [2, 7],
        apply_chat_template=lambda messages, **kwargs: (
            prompts.append(messages[0]["content"]) or "formatted"
        ),
    )
    row = {
        "role": "eval",
        "record_type": "repository_patch_v1",
        "instruction": "Fix the function.",
        "context": "x=0",
    }
    for spec in (EXPERIMENT_SPECS[0], EXPERIMENT_SPECS[4]):
        prompt_ids(row, spec, decoder if spec.encoder else None, decoder, encoder_limit=100)
    assert all(text.startswith("Return only a unified diff") for text in prompts)
    assert row["instruction"] == "Fix the function."


def test_dependency_completion_removes_target_and_retains_real_helper():
    path = Path(__file__).resolve().parents[1] / "tools/prepare_context_data.py"
    spec = importlib.util.spec_from_file_location("context_preparer_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = "from .helper import combine\n\ndef run(x):\n    result = combine(x, 1)\n    adjusted = result * 2 + result * 3\n    return adjusted + result\n"
    snapshot = RepositorySnapshot(
        "fixture/repo",
        "a" * 40,
        (
            RepositoryFile("pkg/owner.py", owner),
            RepositoryFile("pkg/helper.py", "def combine(a, b):\n    return a + b\n"),
            RepositoryFile("broken.py", "x=0\n"),
        ),
    )
    result = module.completion_from_snapshot(
        {
            "target_patch": "diff --git a/broken.py b/broken.py\n--- a/broken.py\n+++ b/broken.py\n@@ -1 +1 @@\n-x=0\n+x=1\n"
        },
        snapshot,
    )
    assert result is not None
    instruction, context, target, selection = result
    assert "pkg/helper.py" in context and "# <MISSING_CODE>" in context
    assert target not in context and target in owner
    assert "combine" in target and selection["gold_used"] is False
    assert "run" in instruction


def test_tokenize_context_task_preserves_target_indentation_and_newlines():
    from bifocalgemma.config import EXPERIMENT_SPECS

    class TrimmingTokenizer:
        bos_token_id = 2
        eos_token_id = 1
        unk_token_id = 3

        def __init__(self):
            self.vocab = {"<turn|>": 106}
            self.next_id = 200

        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, 3)

        def apply_chat_template(
            self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=False
        ):
            assert add_generation_prompt is True, "Target text must never pass through | trim chat template"
            return "<bos><|turn>user\n" + messages[0]["content"] + "<turn|>\n<|turn>model\n"

        def encode(self, text, add_special_tokens=False):
            if text.startswith("<bos>"):
                ids = [2]
                text = text.removeprefix("<bos>")
            else:
                ids = [2] if add_special_tokens else []
            for ch in text:
                if ch not in self.vocab:
                    self.vocab[ch] = self.next_id
                    self.next_id += 1
                ids.append(self.vocab[ch])
            return ids

        def decode(self, token_ids, skip_special_tokens=False):
            inv = {v: k for k, v in self.vocab.items()}
            return "".join(
                inv[i] for i in token_ids if not (skip_special_tokens and i in (1, 2, 106))
            )

    target = "\n    first()\nsecond()\n"
    row = {**task(), "task_type": "completion", "target_text": target}
    tok = TrimmingTokenizer()
    collator = RepairCollator(pad_token_id=0, ignore_index=-100)

    def assert_roundtrip_and_collator(ex, decode_fn, *, expected_bos=2):
        if ex.encoder_ids:
            assert ex.encoder_ids[0] == expected_bos
        assert ex.target_ids[-1] == 106
        assert decode_fn(list(ex.target_ids)) == target
        batch = collator([ex])
        prefix_len = len(ex.decoder_prefix_ids)
        assert batch.labels.shape == (1, prefix_len + len(ex.target_ids) - 1)
        assert all(v == -100 for v in batch.labels[0, : prefix_len - 1].tolist())
        supervised = tuple(batch.labels[0, prefix_len - 1 :].tolist())
        assert supervised == ex.target_ids
        assert supervised[-1] == 106
        assert decode_fn(list(supervised)) == target

    for spec in (EXPERIMENT_SPECS[0], EXPERIMENT_SPECS[4]):
        ex = tokenize_context_task(
            row, spec, tok if spec.encoder else None, tok, encoder_limit=1024, decoder_limit=1024
        )
        assert_roundtrip_and_collator(
            ex, lambda ids: tok.decode(ids, skip_special_tokens=True), expected_bos=2
        )

    raw_root = Path(__file__).resolve().parents[1] / "outputs/context_tokenizers/raw"
    cached_base = raw_root / "google__gemma-4-E2B"
    cached_it = raw_root / "google__gemma-4-E2B-it"
    if cached_base.is_dir() and cached_it.is_dir():
        from transformers import AutoTokenizer

        real_base = AutoTokenizer.from_pretrained(cached_base, local_files_only=True)
        real_it = AutoTokenizer.from_pretrained(cached_it, local_files_only=True)
        assert real_base is not None and real_it is not None
        for spec in (EXPERIMENT_SPECS[0], EXPERIMENT_SPECS[4]):
            ex = tokenize_context_task(
                row,
                spec,
                real_base if spec.encoder else None,
                real_it,
                encoder_limit=1024,
                decoder_limit=1024,
            )
            assert_roundtrip_and_collator(
                ex, lambda ids: real_it.decode(ids, skip_special_tokens=True), expected_bos=2
            )


def test_patch_matrix_tokenization_preserves_target_whitespace_and_collator_alignment(
    monkeypatch,
):
    import bifocalgemma.data.context_tasks as ct_mod
    import bifocalgemma.data.tokenization as tok_mod
    import bifocalgemma.evaluation.context as eval_ctx_mod
    import bifocalgemma.training.curated_sft as cs_mod
    import bifocalgemma.training.repository_sft as rs_mod
    from bifocalgemma.config import EXPERIMENT_SPECS

    monkeypatch.setattr(tok_mod, "load_verified_tokenizer", tok_mod.load_verified_tokenizer)
    monkeypatch.setattr(cs_mod, "load_verified_tokenizer", cs_mod.load_verified_tokenizer)
    monkeypatch.setattr(ct_mod, "tokenize_context_task", ct_mod.tokenize_context_task)
    if hasattr(rs_mod, "tokenize_context_task"):
        monkeypatch.setattr(
            rs_mod, "tokenize_context_task", getattr(rs_mod, "tokenize_context_task")
        )
    monkeypatch.setattr(eval_ctx_mod, "prompt_ids", eval_ctx_mod.prompt_ids)
    monkeypatch.setattr(
        tok_mod, "_matrix_bos_patched", getattr(tok_mod, "_matrix_bos_patched", False), raising=False
    )
    monkeypatch.setattr(
        ct_mod, "_matrix_bos_patched", getattr(ct_mod, "_matrix_bos_patched", False), raising=False
    )
    monkeypatch.setattr(
        eval_ctx_mod,
        "_matrix_bos_patched",
        getattr(eval_ctx_mod, "_matrix_bos_patched", False),
        raising=False,
    )

    from bifocalgemma.training.runtime import patch_matrix_tokenization

    setattr(tok_mod, "_matrix_bos_patched", False)
    setattr(ct_mod, "_matrix_bos_patched", False)
    setattr(eval_ctx_mod, "_matrix_bos_patched", False)
    patch_matrix_tokenization()

    class ItEncoderTokenizer:
        bos_token_id = 2
        eos_token_id = 1
        unk_token_id = 3

        def __init__(self):
            self.vocab = {"<turn|>": 106}
            self.next_id = 200

        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, 3)

        def apply_chat_template(
            self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=False
        ):
            assert add_generation_prompt is True
            return "<bos><|turn>user\n" + messages[0]["content"] + "<turn|>\n<|turn>model\n"

        def encode(self, text, add_special_tokens=False):
            # Intentionally omit BOS even when add_special_tokens=True (matches Gemma-4-IT behavior).
            if text.startswith("<bos>"):
                ids = [2]
                text = text.removeprefix("<bos>")
            else:
                ids = []
            for ch in text:
                if ch not in self.vocab:
                    self.vocab[ch] = self.next_id
                    self.next_id += 1
                ids.append(self.vocab[ch])
            return ids

        def decode(self, token_ids, skip_special_tokens=False):
            inv = {v: k for k, v in self.vocab.items()}
            return "".join(
                inv[i] for i in token_ids if not (skip_special_tokens and i in (1, 2, 106))
            )

    target = "\n    first()\nsecond()\n"
    row = {**task(), "task_type": "completion", "target_text": target}
    tok = ItEncoderTokenizer()
    collator = RepairCollator(pad_token_id=0, ignore_index=-100)
    for spec in (EXPERIMENT_SPECS[1], EXPERIMENT_SPECS[4]):
        ex = ct_mod.tokenize_context_task(
            row, spec, tok if spec.encoder else None, tok, encoder_limit=1024, decoder_limit=1024
        )
        if spec.encoder:
            assert ex.encoder_ids[0] == 2
        assert ex.target_ids[-1] == 106
        assert tok.decode(ex.target_ids, skip_special_tokens=True) == target
        batch = collator([ex])
        supervised = tuple(batch.labels[0, len(ex.decoder_prefix_ids) - 1 :].tolist())
        assert supervised == ex.target_ids
        assert tok.decode(supervised, skip_special_tokens=True) == target

    cached_it = (
        Path(__file__).resolve().parents[1] / "outputs/context_tokenizers/raw/google__gemma-4-E2B-it"
    )
    decoder = tok
    if cached_it.is_dir():
        from transformers import AutoTokenizer

        real_it = AutoTokenizer.from_pretrained(cached_it, local_files_only=True)
        assert real_it is not None
        decoder = real_it
        for spec in (EXPERIMENT_SPECS[1], EXPERIMENT_SPECS[4]):
            ex = ct_mod.tokenize_context_task(
                row,
                spec,
                real_it if spec.encoder else None,
                real_it,
                encoder_limit=1024,
                decoder_limit=1024,
            )
            if spec.encoder:
                assert ex.encoder_ids[0] == real_it.bos_token_id
            assert ex.target_ids[-1] == 106
            assert real_it.decode(list(ex.target_ids), skip_special_tokens=True) == target
            batch = collator([ex])
            supervised = tuple(batch.labels[0, len(ex.decoder_prefix_ids) - 1 :].tolist())
            assert supervised == ex.target_ids
            assert real_it.decode(list(supervised), skip_special_tokens=True) == target

    raw_patch = "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    raw_row = {
        "record_type": "repository_patch_v1",
        "role": "train",
        "instance_id": "raw-patch",
        "context": "a = 1\n",
        "instruction": "Set a to two",
        "target_patch": raw_patch,
    }
    for spec in (EXPERIMENT_SPECS[1], EXPERIMENT_SPECS[4]):
        ex = rs_mod.tokenize_repository_record(
            raw_row, spec, decoder if spec.encoder else None, decoder
        )
        batch = collator([ex])
        supervised = batch.labels[0, len(ex.decoder_prefix_ids) - 1 :].tolist()
        assert supervised[-1] == 106
        assert decoder.decode(supervised[:-1], skip_special_tokens=True) == raw_patch


def test_prepare_sweqa_instance_id_ignores_correct_answer(tmp_path, monkeypatch):
    import json

    import huggingface_hub

    from bifocalgemma.data import context_benchmarks

    def make_rows(correct_letter: str):
        return [
            {
                "question": f"Which helper validates path {i}?",
                "options": {"A": "safe_a", "B": "safe_b", "C": "safe_c", "D": "safe_d"},
                "correct_answer": correct_letter,
                "repo": "pytest",
                "category": "understanding",
                "code": "def safe_a(): pass",
                "chunks": [{"path": "src/path.py"}],
            }
            for i in range(9072)
        ]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "mock.parquet")

    monkeypatch.setattr(context_benchmarks, "read_source_rows", lambda *a, **k: make_rows("A"))
    out_a = tmp_path / "out_a"
    context_benchmarks.prepare_sweqa(tmp_path / "cache_a", out_a)

    monkeypatch.setattr(context_benchmarks, "read_source_rows", lambda *a, **k: make_rows("D"))
    out_d = tmp_path / "out_d"
    context_benchmarks.prepare_sweqa(tmp_path / "cache_d", out_d)

    for cond in ("oracle", "noisy_oracle"):
        public_a = [
            json.loads(line)
            for line in (out_a / f"sweqa_{cond}_v1/eval_inputs.jsonl").read_text().splitlines()
        ]
        public_d = [
            json.loads(line)
            for line in (out_d / f"sweqa_{cond}_v1/eval_inputs.jsonl").read_text().splitlines()
        ]
        gold_a = [
            json.loads(line)["answer"]
            for line in (out_a / f"sweqa_{cond}_v1/eval_private/specs.jsonl")
            .read_text()
            .splitlines()
        ]
        gold_d = [
            json.loads(line)["answer"]
            for line in (out_d / f"sweqa_{cond}_v1/eval_private/specs.jsonl")
            .read_text()
            .splitlines()
        ]
        assert public_a == public_d and len(public_a) == 9072
        assert gold_a == ["A"] * 9072 and gold_d == ["D"] * 9072


def test_prompt_ids_instance_id_isolation_and_crosscodelong_normalization():
    from bifocalgemma.config import EXPERIMENT_SPECS

    seen_texts = []

    class SpyTokenizer:
        bos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            text = messages[0]["content"]
            seen_texts.append(text)
            return text

        def encode(self, text, add_special_tokens=False):
            seen_texts.append(text)
            return [2, 10, 11]

    row = {
        "role": "eval",
        "instance_id": "SECRET_INSTANCE_ID_999",
        "task_type": "completion",
        "instruction": "Complete function",
        "context": "def f(): pass",
    }
    spy = SpyTokenizer()
    prompt_ids(row, EXPERIMENT_SPECS[0], spy, spy, encoder_limit=256)
    assert all("SECRET_INSTANCE_ID_999" not in t for t in seen_texts)

    pred = "Here is the code:\n```python\n    return a + b\n```\n### SUFFIX\n    return c"
    assert normalize_completion_prediction(pred, suffix="\n    return c") == "    return a + b"
    assert completion_metrics(pred, "    return a + b", suffix="\n    return c")[
        "normalized_exact_match"
    ]
