"""Offline tests for v4 evaluation parity fixes, 3-way controls, and external baseline runner."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bifocalgemma import config
from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.context import canonical_snapshot_hash
from bifocalgemma.evaluation import eval_code, external_runner, gsm8k, protocol
from bifocalgemma.evaluation.benchmark import infer_benchmark_weight_tier
from bifocalgemma.evaluation.context import (
    _prompt_ids_impl,
    decoder_prompt,
    score_context_case,
    split_context_for_mode,
)


def test_v4_config_constants_and_external_registry():
    assert config.GATE_INIT_TANH == pytest.approx(0.15)
    assert config.GATE_INIT_ATANH == pytest.approx(math.atanh(0.15))

    assert config.V4_EFFECTIVE_BATCH == 16
    assert config.V4_ACCUMULATION == 8
    assert config.V4_BATCH_SIZE * config.V4_ACCUMULATION == config.V4_EFFECTIVE_BATCH
    assert config.V4_STAGE1_FRACTION == pytest.approx(0.06)
    assert config.V4_NUM_EPOCHS == pytest.approx(1.5)
    assert config.V4_COLLECTION == "mixed_context_enc32k_dec8k_v4"
    assert config.V4_HEAVY_BENCHMARK_SAMPLE_LIMIT == 500

    # Legacy v3 constants remain intact for resume compatibility.
    assert config.TOTAL_STEPS == 11102
    assert config.HEAVY_BENCHMARK_SAMPLE_LIMIT == 100
    assert len(config.BENCHMARKS) == 8

    expected_revs = {
        "gemma-4-e2b-it": ("google/gemma-4-E2B-it", "3e22461f"),
        "gemma-4-e4b-it": ("google/gemma-4-E4B-it", "ee0ef602"),
        "qwen3-4b-instruct-2507": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c0"),
        "qwen2.5-coder-3b-instruct": ("Qwen/Qwen2.5-Coder-3B-Instruct", "488639f1ff80"),
        "qwen2.5-coder-7b-instruct": ("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207"),
    }
    assert set(config.EXTERNAL_COMPARISON_MODELS) == set(expected_revs)
    for key, (model_id, rev_prefix) in expected_revs.items():
        entry = config.EXTERNAL_COMPARISON_MODELS[key]
        assert entry["model_id"] == model_id
        assert entry["revision"].startswith(rev_prefix)

    v4_by_col = {item["collection"]: item for item in config.V4_BENCHMARKS}
    assert "reponiah_gemma32k_dec8k_v1" in v4_by_col
    assert v4_by_col["reponiah_gemma32k_dec8k_v1"]["depths"] == (4096, 8192, 16384, 32768)
    assert "humanevalfix_python_full_v1" in v4_by_col
    assert v4_by_col["humanevalfix_python_full_v1"]["repair_mode"] == "full_function"
    assert v4_by_col["crosscodelong_chunk_gemma32k_dec8k_v1"]["sample_limit"] == 500
    assert v4_by_col["crosscodelong_function_gemma32k_dec8k_v1"]["sample_limit"] == 500
    assert infer_benchmark_weight_tier({"collection": "reponiah_gemma32k_dec8k_v1"}) == "medium_long"


def test_gsm8k_no_space_hash_fix_and_resilient_accuracy_surfacing():
    assert gsm8k.extract_release_answer("Reasoning...\n####18") == "18"
    assert gsm8k.extract_release_answer("steps\n####7.5") == "7.5"
    assert gsm8k.extract_release_answer("steps\n####   -1,234.50") == "-1234.50"
    assert gsm8k.extract_release_answer("####$18") is None

    rows = [
        {"question": "Q1", "answer": "#### 18"},
        {"question": "Q2", "answer": "#### 42"},
        {"question": "Q3", "answer": "#### 9"},
    ]
    public, gold, _ = gsm8k.make_records(rows, [], shots=0)
    preds = [
        {"instance_id": public[0]["instance_id"], "completion": "####18"},
        {"instance_id": public[1]["instance_id"], "completion": r"So \boxed{42}"},
        {"instance_id": public[2]["instance_id"], "completion": "No numeric answer"},
    ]
    report = gsm8k.score_gsm8k(public, gold, preds)
    assert report["strict_correct"] == 1
    assert report["resilient_correct"] == 2
    assert report["strict_accuracy"] == pytest.approx(1 / 3)
    assert report["resilient_accuracy"] == pytest.approx(2 / 3)
    assert report["resilient_recovery_delta"] == pytest.approx(1 / 3)
    assert report["accuracy_summary"] == {
        "strict_accuracy": pytest.approx(1 / 3),
        "resilient_accuracy": pytest.approx(2 / 3),
    }

    scored_cases = [
        score_context_case(
            {"benchmark": "gsm8k"},
            {"answer": "18"},
            "####18",
            directory=None,
            python_executable=None,
            allow_execution=False,
        ),
        score_context_case(
            {"benchmark": "gsm8k"},
            {"answer": "42"},
            r"\boxed{42}",
            directory=None,
            python_executable=None,
            allow_execution=False,
        ),
    ]
    proto_metrics = protocol.gsm8k_report_metrics(scored_cases)
    assert proto_metrics["strict_accuracy"] == pytest.approx(0.5)
    assert proto_metrics["resilient_accuracy"] == pytest.approx(1.0)
    assert proto_metrics["resilient_recovery_delta"] == pytest.approx(0.5)
    assert proto_metrics["accuracy_summary"]["resilient_accuracy"] == pytest.approx(1.0)


def _write_repair_fixture(tmp_path: Path):
    before = (
        "from typing import List\n"
        "\n"
        "def helper_zero() -> int:\n"
        "    return 0\n"
        "\n"
        "def add_one(xs: List[int]) -> List[int]:\n"
        "    return [x for x in xs]\n"
    )
    snapshot = RepositorySnapshot(
        "benchmark/humanevalfix-python",
        hashlib.sha1(before.encode()).hexdigest(),
        (RepositoryFile("solution.py", before),),
    )
    snap_rel = "snapshots/fixture.json"
    snap_path = tmp_path / snap_rel
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(
        json.dumps(
            {
                "repo": snapshot.repo,
                "base_commit": snapshot.base_commit,
                "files": [{"path": "solution.py", "content": before}],
            }
        ),
        encoding="utf-8",
    )
    row = {
        "instance_id": "hefix-1",
        "role": "eval",
        "task_type": "repair",
        "instruction": "Fix add_one so each element is incremented by 1.",
        "context": f"### Issue\nFix add_one.\n\n### File: solution.py\n{before}",
        "context_selection": {"gold_used": False, "windows": [{"path": "solution.py"}]},
    }
    gold = {
        "instance_id": "hefix-1",
        "snapshot_file": snap_rel,
        "snapshot_sha256": canonical_snapshot_hash(snapshot),
        "test_script": "from solution import add_one\nassert add_one([1, 2]) == [2, 3]\n",
    }
    return row, gold


def test_humanevalfix_full_function_pass_at_1_and_recount_diagnostic(tmp_path: Path):
    row, gold = _write_repair_fixture(tmp_path)

    # 1. Full-function pass@1 with fenced single-function output (spliced cleanly with imports/helpers)
    single_fn_fix = "```python\ndef add_one(xs: List[int]) -> List[int]:\n    return [x + 1 for x in xs]\n```"
    res_pass = score_context_case(
        row,
        gold,
        single_fn_fix,
        directory=tmp_path,
        python_executable=sys.executable,
        allow_execution=True,
        repair_mode="full_function",
    )
    assert res_pass["repair_mode"] == "full_function"
    assert res_pass["pass_at_1"] is True
    assert res_pass["correct"] is True
    assert res_pass["syntax_valid"] is True

    # 2. Full-function buggy output fails tests
    res_fail = score_context_case(
        row,
        gold,
        "def add_one(xs: List[int]) -> List[int]:\n    return [x + 2 for x in xs]\n",
        directory=tmp_path,
        python_executable=sys.executable,
        allow_execution=True,
        repair_mode="full_function",
    )
    assert res_fail["pass_at_1"] is False
    assert res_fail["status"] == "FAIL"

    # 3. Full-function syntax error is caught cleanly
    res_syn = score_context_case(
        row,
        gold,
        "def add_one(xs:\n    return xs",
        directory=tmp_path,
        python_executable=sys.executable,
        allow_execution=True,
        repair_mode="full_function",
    )
    assert res_syn["pass_at_1"] is False
    assert res_syn["status"] == "SYNTAX_ERROR"
    assert res_syn["syntax_valid"] is False

    # 4. Unified diff with corrupted hunk count fails strict apply, passes --recount diagnostic!
    bad_count_diff = (
        "diff --git a/solution.py b/solution.py\n"
        "--- a/solution.py\n"
        "+++ b/solution.py\n"
        "@@ -6,99 +6,99 @@\n"
        " def add_one(xs: List[int]) -> List[int]:\n"
        "-    return [x for x in xs]\n"
        "+    return [x + 1 for x in xs]\n"
    )
    res_diff = score_context_case(
        row,
        gold,
        bad_count_diff,
        directory=tmp_path,
        python_executable=sys.executable,
        allow_execution=True,
        repair_mode="unified_diff",
    )
    assert res_diff["repair_mode"] == "unified_diff"
    assert res_diff["patch_applied"] is False
    assert res_diff["functional_pass"] is False
    assert res_diff["recount_patch_applied"] is True
    assert res_diff["recount_functional_pass"] is True

    agg_diff = protocol.repair_report_metrics([res_diff])
    assert agg_diff["patch_apply_rate"] == 0.0
    assert agg_diff["recount_patch_apply_rate"] == 1.0
    assert agg_diff["recount_functional_pass_rate"] == 1.0

    agg_full = protocol.repair_report_metrics([res_pass, res_fail])
    assert agg_full["repair_mode"] == "full_function"
    assert agg_full["pass_at_1"] == pytest.approx(0.5)


def test_three_way_control_modes_longcodeqa_dedup_and_fim_marker():
    # 1. CrossCodeLong FIM 3-way split & # <MISSING_CODE> unification
    prefix_lines = [f"pref_line_{i} = {i}" for i in range(1, 61)]
    suffix_lines = [f"suff_line_{i} = {i}" for i in range(1, 61)]
    ccl_row = {
        "instance_id": "ccl-1",
        "role": "eval",
        "task_type": "completion",
        "benchmark": "CrossCodeLongEval",
        "instruction": "Complete the missing Python code.",
        "context": (
            "### CROSS-FILE CONTEXT\n"
            "def cross_helper():\n    return 99\n\n"
            "### PREFIX\n"
            + "\n".join(prefix_lines)
            + "\n### MISSING CODE\n\n"
            + "### SUFFIX\n"
            + "\n".join(suffix_lines)
        ),
    }

    dec_full, enc_full = split_context_for_mode(ccl_row, context_mode="full")
    dec_disjoint, enc_disjoint = split_context_for_mode(ccl_row, context_mode="disjoint")
    dec_local, enc_local = split_context_for_mode(ccl_row, context_mode="local_only")

    assert enc_full == ""
    assert enc_local == ""
    assert dec_local == dec_disjoint  # Exact decoder parity for control!
    assert "# <MISSING_CODE>" in dec_full
    assert "### MISSING CODE" not in dec_full
    assert "# <MISSING_CODE>" in dec_disjoint
    assert "cross_helper" in enc_disjoint
    assert "cross_helper" not in dec_disjoint
    assert "cross_helper" not in dec_local
    # Strict disjointness: lines 1..12 of prefix are only in encoder, lines 13..60 only in decoder.
    assert "13: pref_line_13 = 13" in dec_disjoint
    assert "pref_line_13" not in enc_disjoint
    assert "1: pref_line_1 = 1" in enc_disjoint
    assert "pref_line_1 =" not in dec_disjoint
    assert "1: suff_line_1 = 1" in dec_disjoint
    assert "49: suff_line_49 = 49" in enc_disjoint
    assert "suff_line_49" not in dec_disjoint

    # 2. Multi-file repair 3-way split
    repair_row = {
        "instance_id": "rep-1",
        "role": "eval",
        "task_type": "repair",
        "instruction": "Fix bug in target.py.",
        "context": (
            "### Issue\nFix bug in target.py.\n\n"
            "### File: target.py\nx = 1\ny = 2\n\n"
            "### File: sibling.py\ndef cross_dep():\n    return 42\n"
        ),
        "context_selection": {
            "gold_used": False,
            "windows": [{"path": "target.py", "start_line": 1, "end_line": 2}],
        },
    }
    r_dec_disjoint, r_enc_disjoint = split_context_for_mode(repair_row, context_mode="disjoint")
    r_dec_local, r_enc_local = split_context_for_mode(repair_row, context_mode="local_only")
    assert r_dec_disjoint == r_dec_local
    assert r_enc_local == ""
    assert "1: x = 1" in r_dec_disjoint
    assert "sibling.py" in r_enc_disjoint
    assert "sibling.py" not in r_dec_disjoint
    assert "x = 1" not in r_enc_disjoint

    r_dec_full_fn = decoder_prompt(
        repair_row, paired=False, context_mode="full", repair_mode="full_function"
    )
    assert r_dec_full_fn.startswith("Return only the complete fixed Python code")

    # 3. LongCodeQA double-header deduplication
    lcqa_row = {
        "instance_id": "lcqa-1",
        "role": "eval",
        "task_type": "mcq",
        "benchmark": "LongCodeQA",
        "instruction": "Repository: Repository: acme/app\nQuestion: Question: What is X?\nA) 1\nB) 2",
        "context": "Repository: Repository: acme/app\n\ndef f(): return 1\n\nQuestion:\nQuestion: What is X?\nA) 1\nB) 2",
    }
    lcqa_full = decoder_prompt(lcqa_row, paired=False, context_mode="full")
    assert "Repository: Repository:" not in lcqa_full
    assert "Question: Question:" not in lcqa_full
    assert "Question:\nQuestion:" not in lcqa_full
    assert lcqa_full.count("Repository:") == 1
    assert lcqa_full.count("Question:") == 1

    # 4. prompt_ids context_mode contract
    class SpyTok:
        bos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, add_special_tokens=False):
            return [2] + [7] * max(1, len(text.split()))

    paired_spec = config.make_experiment_spec("2it-2it")
    baseline_spec = config.make_experiment_spec("e2b-it")
    spy = SpyTok()
    enc_d, pref_d = _prompt_ids_impl(
        ccl_row, paired_spec, spy, spy, encoder_limit=4096, context_mode="disjoint"
    )
    enc_l, pref_l = _prompt_ids_impl(
        ccl_row, paired_spec, spy, spy, encoder_limit=4096, context_mode="local_only"
    )
    assert enc_d and enc_l == ()
    assert pref_d == pref_l
    with pytest.raises(ValueError, match="disjoint_context_mode_requires_encoder"):
        _prompt_ids_impl(
            ccl_row, baseline_spec, None, spy, encoder_limit=4096, context_mode="disjoint"
        )


class MockQwenTokenizer:
    """Mock Qwen BPE tokenizer where bos_token_id is None and token ID 1 is '\"'."""

    bos_token_id = None
    eos_token_id = 151645  # <|im_end|>
    pad_token_id = 151643  # <|endoftext|>
    unk_token_id = None
    chat_template = "qwen-chat"
    all_special_tokens = ("<|im_end|>", "<|endoftext|>")

    def __init__(self, *, support_enable_thinking: bool = True, completion_text: str = '####18'):
        self.support_enable_thinking = support_enable_thinking
        self.completion_text = completion_text
        self.chat_calls: list[dict] = []
        self._vocab = {
            '"': 1,
            "<|endoftext|>": 151643,
            "<|im_end|>": 151645,
        }

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tok: str):
        return self._vocab.get(tok, 999)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if not self.support_enable_thinking and "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.chat_calls.append(kwargs)
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text: str, add_special_tokens: bool = False):
        # Intentionally include token ID 1 ('"') to verify no Gemma BOS/EOS assumptions.
        return [1, 10, 11, len(text) % 50 + 20]

    def decode(self, ids, skip_special_tokens: bool = True):
        assert 151645 not in ids and 151643 not in ids
        # Token 1 ('"') must NOT have been stripped as a Gemma <eos> token!
        if 1 in ids:
            return f'"{self.completion_text}"'
        return self.completion_text


class MockCausalLM:
    def __init__(self, generated_suffix: list[int]):
        self.generated_suffix = generated_suffix
        self.training = False
        self.config = SimpleNamespace(eos_token_id=[151645, 151643])
        self.generation_config = SimpleNamespace(eos_token_id=[151645, 151643])
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["do_sample"] is False
        assert 1 not in kwargs["eos_token_id"]
        bsz = input_ids.shape[0]
        suffix = torch.tensor(self.generated_suffix, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        return torch.cat([input_ids, suffix], dim=1)


def test_external_runner_qwen_bos_none_and_eos_handling(tmp_path: Path):
    qwen_tok = MockQwenTokenizer(support_enable_thinking=True, completion_text="####18")
    # Token 1 is '"' in Qwen; 151645 is <|im_end|>; 151643 is <|endoftext|>
    mock_model = MockCausalLM(generated_suffix=[42, 1, 43, 151645, 151643])

    eos_ids = external_runner.resolve_external_eos_ids(qwen_tok, mock_model)
    assert eos_ids == (151645, 151643)
    assert 1 not in eos_ids

    kept, stop_reason = external_runner.extract_generated_tokens([42, 1, 43, 151645, 151643], eos_ids)
    assert kept == (42, 1, 43)
    assert stop_reason == "eos"

    # Tokenizer without enable_thinking support falls back cleanly.
    fallback_tok = MockQwenTokenizer(support_enable_thinking=False)
    _, ids = external_runner.format_external_prompt(
        {"role": "eval", "task_type": "mcq", "instruction": "Pick A.", "context": "def f(): pass"},
        fallback_tok,
        context_mode="full",
        enable_thinking=False,
    )
    assert ids[0] == 1  # bos_token_id=None must not raise!

    # End-to-end mock GSM8K run through external_runner using exact scorer
    bench_dir = tmp_path / "gsm8k_zero_shot_v2"
    bench_dir.mkdir(parents=True)
    public = [
        {
            "instance_id": "gsm-1",
            "role": "eval",
            "benchmark": "gsm8k",
            "question": "9 + 9?",
            "instruction": gsm8k.INSTRUCTION,
            "context": "Question: 9 + 9?\nAnswer:",
        }
    ]
    gold = [{"instance_id": "gsm-1", "answer": "18"}]
    (bench_dir / "inputs.jsonl").write_text(json.dumps(public[0]) + "\n", encoding="utf-8")
    (bench_dir / "hidden_gold.jsonl").write_text(json.dumps(gold[0]) + "\n", encoding="utf-8")

    # Model emits [42, 151645] -> decodes to "####18" (no space) -> fixed strict + resilient both pass!
    clean_model = MockCausalLM(generated_suffix=[42, 151645])
    out_dir = tmp_path / "out_gsm"
    report = external_runner.run_external_benchmark(
        "qwen3-4b-instruct-2507",
        config.V4_BENCHMARKS[6],  # gsm8k_zero_shot_v2
        tmp_path,
        out_dir,
        context_mode="full",
        model=clean_model,
        tokenizer=qwen_tok,
    )
    assert report["status"] == "COMPLETED"
    assert report["strict_accuracy"] == 1.0
    assert report["resilient_accuracy"] == 1.0
    assert report["eos_ids"] == [151645, 151643]
    assert qwen_tok.chat_calls[-1]["enable_thinking"] is False
    assert (out_dir / "predictions.jsonl").is_file()
    assert (out_dir / "benchmark_report.json").is_file()


def test_eval_external_baselines_cli_dry_run(capsys):
    from tools import eval_external_baselines

    rc = eval_external_baselines.main(
        [
            "--models",
            "qwen3-4b-instruct-2507,qwen2.5-coder-3b-instruct",
            "--benchmarks",
            "reponiah,humanevalfix_python_full_v1",
            "--context-mode",
            "local_only",
            "--dry-run",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["context_mode"] == "local_only"
    assert payload["models"] == ["qwen3-4b-instruct-2507", "qwen2.5-coder-3b-instruct"]
    assert payload["benchmarks"] == ["reponiah_gemma32k_dec8k_v1", "humanevalfix_python_full_v1"]
    assert payload["results"]["qwen3-4b-instruct-2507"]["reponiah_gemma32k_dec8k_v1"]["planned_cases"] == 100

    # Also verify eval_code v4 table helper runs cleanly without mutating REFERENCE_BASELINES
    assert set(eval_code.REFERENCE_BASELINES) == {"Gemma-4-E2B-IT", "Gemma-4-E4B-IT"}
    eval_code.print_v4_parity_comparison_table(
        {"qwen3-4b-instruct-2507 (full)": {"longcodeqa": 75.0, "gsm8k_resilient": 88.0}}
    )
