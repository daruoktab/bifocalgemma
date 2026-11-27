"""Offline tests for v4 disjoint context splitting, multi-function FIM, curation gates, and multi-epoch schedules."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

from bifocalgemma.config import EXPERIMENT_SPECS
from bifocalgemma.contracts import RepositoryFile, RepositorySnapshot
from bifocalgemma.data.context_tasks import (
    format_numbered_code,
    split_disjoint_context,
    task_prompt,
    tokenize_context_task,
)
from bifocalgemma.data.curriculum import V4_DEFAULT_MIX, build_v4_curriculum, classify_v4_bucket
from bifocalgemma.data.offline_sft import load_curated_pool
from bifocalgemma.data.repository_corpus import (
    format_numbered_code as repo_format_numbered_code,
)
from bifocalgemma.data.repository_corpus import (
    split_disjoint_context as repo_split_disjoint_context,
)
from bifocalgemma.data.repository_corpus import (
    v4_repository_task_from_row,
)
from bifocalgemma.training.curated_sft import prepare_curated_training
from bifocalgemma.training.schedule import (
    build_shared_schedule,
    canonical_schedule_records,
    persist_shared_schedule,
)


def _load_prepare_module():
    path = Path(__file__).resolve().parents[1] / "tools/prepare_context_data.py"
    spec = importlib.util.spec_from_file_location("prepare_context_data_v4_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CharTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    unk_token_id = 3

    def __init__(self):
        self.vocab: dict[str, int] = {"<eos>": 1, "<bos>": 2, "<turn|>": 106}
        self.next_id = 200

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        assert tokenize is False and add_generation_prompt is True
        return "<bos><|turn>user\n" + messages[0]["content"] + "<turn|>\n<|turn>model\n"

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        if text.startswith("<bos>"):
            ids = [self.bos_token_id]
            text = text.removeprefix("<bos>")
        else:
            ids = [self.bos_token_id] if add_special_tokens else []
        for ch in text:
            if ch not in self.vocab:
                self.vocab[ch] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[ch])
        return ids

    def decode(self, token_ids, skip_special_tokens=False) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in token_ids if not (skip_special_tokens and i in (1, 2, 106)))


def _numbered_line_numbers(text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"^(\d+):\s", text, re.M)]


def test_format_numbered_code_and_exports():
    assert repo_format_numbered_code is format_numbered_code
    assert repo_split_disjoint_context is split_disjoint_context
    assert format_numbered_code("") == ""
    assert format_numbered_code("a = 1\nb = 2") == "1: a = 1\n2: b = 2"
    assert format_numbered_code("x = 10\ny = 20", start_line=42) == "42: x = 10\n43: y = 20"
    already = "10: def f():\n11:     return 1"
    assert format_numbered_code(already, skip_if_numbered=True) == already
    with pytest.raises(ValueError):
        format_numbered_code("x = 1", start_line=0)


def test_disjoint_split_completion_zero_overlap_and_three_way_parity():
    main_lines = [f"def fn_{i}(): return {i}" for i in range(1, 141)]
    main_lines[74] = "    # <MISSING_CODE>"
    row = {
        "record_type": "context_sft_v1",
        "context_format": "v4",
        "disjoint_context": True,
        "role": "train",
        "task_type": "completion",
        "instance_id": "comp-1",
        "group": "owner/repo",
        "instruction": "Complete fn_75 using helper_sum.",
        "context": (
            "### Instruction\nComplete fn_75 using helper_sum.\n\n"
            "### File: pkg/helper.py\n"
            "def helper_sum(a, b):\n    return a + b\n\n"
            "### File: pkg/main.py\n" + "\n".join(main_lines)
        ),
        "target_text": "    return helper_sum(74, 1)",
        "source": "nebius/SWE-rebench-V2",
        "source_key": "swe_rebench_v2",
        "source_revision": "a" * 40,
        "snapshot_sha256": "b" * 64,
        "commit": "c" * 40,
        "repos": "owner/repo",
        "context_selection": {
            "gold_used": False,
            "windows": [{"path": "pkg/helper.py"}, {"path": "pkg/main.py"}],
        },
    }

    split = split_disjoint_context(row, local_window_lines=32)
    assert split.cursor_path == "pkg/main.py"
    # Cursor is at line 75; ±32 => lines 43..107 in decoder local window
    assert split.local_line_range == (43, 107)
    assert "### File: pkg/helper.py (lines 1-2)\n1: def helper_sum" in split.encoder_context
    assert "def helper_sum" not in split.decoder_local_context
    assert "# <MISSING_CODE>" in split.decoder_local_context
    assert "# <MISSING_CODE>" not in split.encoder_context

    # Zero line-number overlap within pkg/main.py between encoder and decoder
    main_enc_nums: set[int] = set()
    for block in re.split(r"(?=^### File: )", split.encoder_context, flags=re.M):
        if block.startswith("### File: pkg/main.py"):
            main_enc_nums.update(_numbered_line_numbers(block))
    main_dec_nums = set(_numbered_line_numbers(split.decoder_local_context))
    assert main_enc_nums == set(range(1, 43)) | set(range(108, 141))
    assert main_dec_nums == set(range(43, 108))
    assert main_enc_nums.isdisjoint(main_dec_nums)

    # 100% text parity vs paired=False and exact 3-way control with decoder_local_only=True
    p_bifocal = task_prompt(row, paired=True, disjoint_context=True, local_window_lines=32)
    p_baseline = task_prompt(row, paired=False, disjoint_context=True, local_window_lines=32)
    p_local_only = task_prompt(
        row, paired=True, disjoint_context=True, decoder_local_only=True, local_window_lines=32
    )
    p_local_only_from_base = task_prompt(
        row, paired=False, disjoint_context=True, decoder_local_only=True, local_window_lines=32
    )
    assert p_bifocal == p_local_only == p_local_only_from_base
    assert split.encoder_context in p_baseline
    assert split.decoder_local_context in p_baseline
    assert split.encoder_context not in p_bifocal

    # Legacy compatibility when disjoint_context=False
    p_legacy = task_prompt(row, paired=True, disjoint_context=False)
    assert "### Local Cursor" not in p_legacy
    assert p_legacy.endswith("Complete fn_75 using helper_sum.")

    # Tokenize all 3 modes cleanly
    tok = _CharTokenizer()
    ex_bifocal = tokenize_context_task(
        row,
        EXPERIMENT_SPECS[0],
        tok,
        tok,
        encoder_limit=8192,
        decoder_limit=4096,
        local_window_lines=32,
    )
    ex_baseline = tokenize_context_task(
        row,
        EXPERIMENT_SPECS[4],
        None,
        tok,
        encoder_limit=8192,
        decoder_limit=16384,
        local_window_lines=32,
    )
    ex_local_only = tokenize_context_task(
        row,
        EXPERIMENT_SPECS[0],
        tok,
        tok,
        encoder_limit=8192,
        decoder_limit=4096,
        decoder_local_only=True,
        local_window_lines=32,
    )
    assert ex_bifocal.decoder_prefix_ids == ex_local_only.decoder_prefix_ids
    assert ex_local_only.encoder_ids == ()
    assert ex_baseline.encoder_ids == ()
    assert "43: def fn_43" in tok.decode(ex_bifocal.decoder_prefix_ids)
    assert "1: def helper_sum" in tok.decode(ex_bifocal.encoder_ids)
    assert "43: def fn_43" not in tok.decode(ex_bifocal.encoder_ids)
    assert "1: def helper_sum" in tok.decode(ex_baseline.decoder_prefix_ids)
    assert "43: def fn_43" in tok.decode(ex_baseline.decoder_prefix_ids)


def test_disjoint_split_crosscodelong_repair_and_qa_layouts():
    # CrossCodeLong FIM layout
    ccl_row = {
        "task_type": "completion",
        "instruction": "Complete the missing Python code between PREFIX and SUFFIX. Return only that code.",
        "context": (
            "Complete the missing Python code between PREFIX and SUFFIX. Return only that code.\n\n"
            "### CROSS-FILE CONTEXT\ndef remote_api():\n    return 1\n\n"
            "### PREFIX\n"
            + "\n".join(f"pre_{i} = {i}" for i in range(1, 80))
            + "\n### MISSING CODE\n"
            + "\n### SUFFIX\n"
            + "\n".join(f"suf_{i} = {i}" for i in range(1, 80))
        ),
    }
    ccl_split = split_disjoint_context(ccl_row, local_window_lines=32)
    assert "remote_api" in ccl_split.encoder_context
    assert (
        "pre_1 = 1" in ccl_split.encoder_context
        and "pre_1 = 1" not in ccl_split.decoder_local_context
    )
    assert (
        "pre_79 = 79" in ccl_split.decoder_local_context
        and "pre_79 = 79" not in ccl_split.encoder_context
    )
    assert (
        "suf_1 = 1" in ccl_split.decoder_local_context
        and "suf_1 = 1" not in ccl_split.encoder_context
    )
    assert (
        "suf_79 = 79" in ccl_split.encoder_context
        and "suf_79 = 79" not in ccl_split.decoder_local_context
    )

    # Multi-file repair with explicit line window offset
    rep_row = {
        "task_type": "repair",
        "instruction": "Fix target_calc off-by-one in app/calc.py",
        "context": (
            "### Issue\nFix target_calc off-by-one in app/calc.py\n\n"
            "### File: app/utils.py (lines 1-5)\n"
            + "\n".join(f"u_{i} = {i}" for i in range(1, 6))
            + "\n\n"
            "### File: app/calc.py (lines 10-70)\n"
            + "\n".join(
                "def target_calc(x):" if i == 30 else f"line_{i} = {i}" for i in range(10, 71)
            )
        ),
    }
    rep_split = split_disjoint_context(rep_row, local_window_lines=10)
    assert rep_split.cursor_path == "app/calc.py"
    assert "def target_calc(x):" in rep_split.decoder_local_context
    assert "30: def target_calc(x):" in rep_split.decoder_local_context
    assert "app/utils.py" in rep_split.encoder_context
    assert "app/utils.py" not in rep_split.decoder_local_context
    assert "30: def target_calc(x):" not in rep_split.encoder_context

    # QA / MCQ keeps decoder free of context
    qa_row = {
        "task_type": "long_context_qa",
        "instruction": "What does alpha return?",
        "context": "### File: mod.py\ndef alpha():\n    return 99\n",
    }
    qa_split = split_disjoint_context(qa_row)
    assert qa_split.decoder_local_context == ""
    assert "1: def alpha():" in qa_split.encoder_context
    assert qa_split.full_context == qa_split.encoder_context


def test_multifunction_completion_extraction_removes_single_return_cap():
    mod = _load_prepare_module()
    owner_code = (
        "from .helper import combine\n\n"
        "def func_one(x):\n"
        "    def nested(z):\n"
        "        return combine(z, 1) + combine(z, 2) + combine(z, 3) + combine(z, 4)\n"
        "    a = combine(x, 1)\n"
        "    b = combine(a, 2)\n"
        "    return nested(b) + a + b\n\n"
        "def func_two(y):\n"
        "    first = combine(y, 10)\n"
        "    second = combine(first, 20)\n"
        "    return first * second + combine(first, second)\n\n"
        "def func_three(k):\n"
        "    v1 = combine(k, 100)\n"
        "    v2 = combine(v1, 200)\n"
        "    return v1 + v2 + combine(k, v2)\n"
    )
    snapshot = RepositorySnapshot(
        "owner/repo",
        "a" * 40,
        (
            RepositoryFile("pkg/owner.py", owner_code),
            RepositoryFile("pkg/helper.py", "def combine(a, b):\n    return a + b\n"),
            RepositoryFile("broken.py", "x = 0\n"),
        ),
    )
    row = {
        "target_patch": "diff --git a/broken.py b/broken.py\n--- a/broken.py\n+++ b/broken.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"
    }
    many = mod.completions_from_snapshot(row, snapshot)
    assert len(many) == 3
    names = [item[0] for item in many]
    assert any("func_one" in n for n in names)
    assert any("func_two" in n for n in names)
    assert any("func_three" in n for n in names)
    # Nested function inside func_one must NOT be extracted separately (non-overlapping rule)
    assert not any("nested" in n for n in names)
    for instr, ctx, tgt, sel in many:
        assert ctx.count("# <MISSING_CODE>") == 1
        assert tgt not in ctx
        assert sel["gold_used"] is False
    # Backwards-compatible single-tuple helper returns the first function
    single = mod.completion_from_snapshot(row, snapshot)
    assert single == many[0]


def test_v4_quarantine_licenses_dedup_and_nextcoder_rejection():
    mod = _load_prepare_module()
    blocked_repos = {"blocked/bench-repo"}
    blocked_slugs = {"cross-slug"}
    sigs = mod.code_fragment_signatures(
        "def leaked_bench():\n    x = 1\n    y = 2\n    return x + y\n" * 4
    )
    gsm8k_fps = mod.build_gsm8k_fingerprints(["Janet has 16 eggs per day. How much does she make?"])

    # 1. NextCoderDataset explicitly dropped/blocked
    with pytest.raises(ValueError, match="blocked_v4_dataset"):
        mod.adapt_v4_row("nextcoder", {"instruction": "x", "context": "y", "target_text": "z"})
    with pytest.raises(ValueError, match="blocked_v4_dataset"):
        mod.adapt_v4_row(
            "opc_sft_stage2",
            {
                "source": "microsoft/NextCoderDataset",
                "instruction": "x",
                "context": "y",
                "target_text": "z",
            },
        )

    # 2. Non-commercial / GPL / unknown licenses rejected on FIM-Midtraining-400K
    for bad_lic in ("cc-by-nc-4.0", "cc-by-nc-nd-4.0", "gpl-3.0", "unknown"):
        with pytest.raises(ValueError, match="non_permissive_license"):
            mod.adapt_v4_row(
                "fim_midtraining",
                {
                    "prefix": "def f(x):\n",
                    "suffix": "return y",
                    "middle": "    y = x + 1\n",
                    "metadata": {
                        "repository_url": "https://github.com/ok/repo",
                        "license": bad_lic,
                        "file_path": "a.py",
                    },
                },
            )

    # Permissive MIT license accepted on FIM-Midtraining-400K
    ok_fim = mod.adapt_v4_row(
        "fim_midtraining",
        {
            "prefix": "def f(x):\n",
            "suffix": "    return y",
            "middle": "    y = x + 1\n",
            "metadata": {
                "repository_url": "https://github.com/ok/repo",
                "license": "MIT",
                "file_path": "a.py",
            },
        },
    )
    assert ok_fim["task_type"] == "completion"
    assert ok_fim["disjoint_context"] is True

    # 3. Benchmark repo & code overlap rejected
    with pytest.raises(ValueError, match="benchmark_repository_quarantine"):
        mod.adapt_v4_row(
            "swe_smith_py",
            {
                "instance_id": "sw-1",
                "repo": "blocked/bench-repo",
                "base_commit": "a" * 40,
                "problem_statement": "Fix bug",
                "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x=1\n+x=2\n",
                "context": "### File: a.py\nx=1\n",
            },
            blocked_repos=blocked_repos,
            blocked_slugs=blocked_slugs,
        )
    with pytest.raises(ValueError, match="benchmark_code_overlap"):
        mod.adapt_v4_row(
            "opc_sft_stage2",
            {
                "instruction": "Implement helper",
                "context": "### File: a.py\n"
                + ("def leaked_bench():\n    x = 1\n    y = 2\n    return x + y\n" * 4),
                "target_text": "pass",
            },
            code_signatures=sigs,
        )

    # 4. OpenCodeInstruct requires average_test_score >= 0.9
    with pytest.raises(ValueError, match="opencodeinstruct_low_test_score"):
        mod.adapt_v4_row(
            "opencodeinstruct",
            {"instruction": "Write f", "output": "def f(): return 1", "average_test_score": 0.85},
        )
    ok_oci = mod.adapt_v4_row(
        "opencodeinstruct",
        {"instruction": "Write f", "output": "def f(): return 1", "average_test_score": 0.95},
    )
    assert ok_oci["v4_bucket"] == "repair_loc"

    # 5. LongAlign <= 32k token filter & GSM8K dedup
    with pytest.raises(ValueError, match="v4_context_exceeds_max_tokens"):
        mod.adapt_v4_row(
            "longalign",
            {
                "instruction": "Summarize",
                "context": "x " * 100,
                "output": "ok",
                "encoder_tokens": 40000,
            },
        )
    with pytest.raises(ValueError, match="gsm8k_test_overlap"):
        mod.adapt_v4_row(
            "smoltalk_metamathqa",
            {
                "instruction": "Janet has 16 eggs per day. How much does she make?",
                "output": "#### 18",
            },
            gsm8k_fingerprints=gsm8k_fps,
        )

    # 6. v4 repository task normalization
    task = v4_repository_task_from_row(
        "swe_smith_py",
        {
            "instance_id": "smith-1",
            "repo": "owner/smith-proj",
            "base_commit": "d" * 40,
            "problem_statement": "Fix off by one",
            "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x=1\n+x=2\n",
        },
    )
    assert task.source == "SWE-bench/SWE-smith-py"


def test_mixed_pool_v4_curriculum_grounded_and_curated_sft(tmp_path):
    mod = _load_prepare_module()
    long_body = "\n".join(f"val_{k} = {k}" for k in range(40))
    long_code = f"### File: src/mod.py\n{long_body}\n"

    # Build synthetic v4 pool across all 4 buckets, mixing <8k and 8k-32k rows
    v4_sources: dict[str, list[dict[str, Any]]] = {
        "swe_fixer": [
            {
                "instance_id": f"fixer-{i}",
                "repo": f"org/repo_{i % 6}",
                "base_commit": f"{i:040x}",
                "problem_statement": f"Fix issue {i} in mod.py",
                "context": long_code + f"# issue {i}\n",
                "patch": f"diff --git a/src/mod.py b/src/mod.py\n@@ -1 +1 @@\n-val_0 = 0\n+val_0 = {i + 1}\n",
                "encoder_tokens": 12000 if i % 2 == 0 else 1500,
            }
            for i in range(12)
        ],
        "fim_midtraining": [
            {
                "prefix": f"def comp_{i}(x):\n",
                "suffix": "    return z\n",
                "middle": f"    z = x + {i}\n",
                "encoder_tokens": 10000 if i % 2 == 0 else 900,
                "metadata": {
                    "repository_url": f"https://github.com/org/fim_{i % 6}",
                    "license": "mit",
                    "file_path": "pkg/f.py",
                    "cross_file_context": f"def helper_{i}(): return {i}\n",
                },
            }
            for i in range(10)
        ],
        "longalign": [
            {
                "instance_id": f"la-{i}",
                "instruction": f"Explain section {i}",
                "context": f"Document {i}\n" + ("alpha beta gamma delta\n" * 10),
                "output": f"Section {i} computes alpha.",
                "encoder_tokens": 16000 if i < 6 else 3000,
            }
            for i in range(8)
        ],
        "smoltalk_constraints": [
            {
                "instance_id": f"ret-{i}",
                "instruction": f"Return JSON with key id={i}",
                "output": json.dumps({"id": i}),
                "encoder_tokens": 256,
            }
            for i in range(6)
        ],
    }

    pool_dir = tmp_path / "mixed_context_enc32k_dec8k_v4"
    manifest = mod.mixed_pool_v4(
        v4_sources,
        None,
        pool_dir,
        blocked_repos={"org/blocked"},
        min_long_context_ratio=0.40,
        grounded_max_parents=4,
        seed=17,
    )
    assert manifest["report"]["pool_recipe"] == "mixed_context_enc32k_dec8k_v4"
    assert manifest["report"]["disjoint_context"] is True
    assert "microsoft/NextCoderDataset" in manifest["report"]["dropped_sources"]
    assert manifest["report"]["grounded_tasks_count"] > 0
    assert manifest["report"]["curriculum_stats"]["long_context_ratio"] >= 0.40

    rows, loaded_manifest = load_curated_pool(pool_dir)
    assert len(rows["train"]) > 0 and len(rows["dev"]) > 0
    assert all(r["source"] != "microsoft/NextCoderDataset" for r in rows["train"] + rows["dev"])
    # Verify grounded tasks generated and placed cleanly in their parent's split
    assert any(
        r.get("record_type") == "grounded_context_sft_v1" for r in rows["train"] + rows["dev"]
    )
    assert set(loaded_manifest["report"]["bucket_counts"]) <= set(V4_DEFAULT_MIX)

    # Check curriculum directly
    selected, c_stats = build_v4_curriculum(
        rows["train"], target_count=16, min_long_context_ratio=0.40
    )
    assert len(selected) == 16
    assert c_stats["long_context_ratio"] >= 0.40
    assert {classify_v4_bucket(r) for r in selected} == set(V4_DEFAULT_MIX)

    # Verify curated_sft.prepare_curated_training supports v4 pool disjoint tokenization offline
    tok_cache = tmp_path / "tok_cache"
    tok = _CharTokenizer()
    from bifocalgemma.training import curated_sft as cs_mod

    orig_loader = cs_mod.load_verified_tokenizer
    try:
        setattr(
            cs_mod,
            "load_verified_tokenizer",
            lambda checkpoint, tokenizer_cache: (
                tok,
                {"model_id": checkpoint.model_id, "files": {"tokenizer.json": "a" * 64}},
            ),
        )
        tr_pair, dev_pair, m_pair = prepare_curated_training(
            EXPERIMENT_SPECS[0], pool_dir, tok_cache, local_window_lines=16
        )
        tr_local, _, _ = prepare_curated_training(
            EXPERIMENT_SPECS[0], pool_dir, tok_cache, decoder_local_only=True, local_window_lines=16
        )
        assert len(tr_pair) == len(rows["train"]) and len(dev_pair) == len(rows["dev"])
        assert tr_pair[0].decoder_prefix_ids == tr_local[0].decoder_prefix_ids
        assert tr_local[0].encoder_ids == ()
        assert m_pair["pool_fingerprint"] == loaded_manifest["fingerprint"]
    finally:
        cs_mod.load_verified_tokenizer = orig_loader


def test_multi_epoch_and_stratified_balanced_schedule(tmp_path):
    # Skewed raw cohort: mostly short repair, few long completion/qa
    raw = []
    for i in range(20):
        task = "repair" if i < 16 else ("completion" if i < 18 else "long_context_qa")
        enc_len = 1000 if i < 16 else 12000
        raw.append(
            {
                "instance_id": f"row-{i:04d}",
                "source": f"src-{i % 2}",
                "task_type": task,
                "instruction": f"Task {i}",
                "context": f"def fn_{i}(): return {i}",
                "target_text": f"return {i}",
                "token_lengths": {"encoder": enc_len, "target": 5},
            }
        )
    records = canonical_schedule_records(raw)
    identity = {"dataset": "v4", "split": "train"}

    s1 = build_shared_schedule(records, dataset_identity=identity, num_epochs=1)
    s15 = build_shared_schedule(records, dataset_identity=identity, num_epochs=1.5)
    s2 = build_shared_schedule(records, dataset_identity=identity, num_epochs=2)

    assert s1.total_examples == 20
    assert s15.total_examples == 30
    assert s2.total_examples == 40
    assert len({s1.signature, s15.signature, s2.signature}) == 3
    # Epoch 1 prefix of 1.5 and 2 matches 1-epoch schedule
    assert s15.ordered_ids[:20] == s1.ordered_ids
    assert s2.ordered_ids[:20] == s1.ordered_ids
    # Epoch 2 order is a distinct deterministic permutation of all 20 IDs
    assert sorted(s2.ordered_ids[20:]) == sorted(s1.ordered_ids)
    assert s2.ordered_ids[20:] != s1.ordered_ids

    # Input-order invariance on multi-epoch & balanced schedules
    s15_rev = build_shared_schedule(
        list(reversed(records)), dataset_identity=identity, num_epochs=1.5
    )
    assert s15_rev.ordered_ids == s15.ordered_ids
    assert s15_rev.signature == s15.signature
    assert [records[i]["instance_id"] for i in s15.ordered_indices] == list(s15.ordered_ids)
    assert [list(reversed(records))[i]["instance_id"] for i in s15_rev.ordered_indices] == list(
        s15_rev.ordered_ids
    )

    # Stratified balancing across (task, length_bin) equalizes representation
    bal = build_shared_schedule(
        records, dataset_identity=identity, num_epochs=1.5, balance_strata=True
    )
    bal_rev = build_shared_schedule(
        list(reversed(records)), dataset_identity=identity, num_epochs=1.5, balance_strata=True
    )
    assert bal.ordered_ids == bal_rev.ordered_ids
    assert bal.signature == bal_rev.signature
    task_counts = {}
    for rec in bal.records[:20]:
        task_counts[rec.task] = task_counts.get(rec.task, 0) + 1
    assert task_counts["completion"] >= 6 and task_counts["long_context_qa"] >= 6

    # Manifest persistence & validation
    manifest_path = tmp_path / "schedule_v4.json"
    persist_shared_schedule(manifest_path, s15)
    persist_shared_schedule(manifest_path, s15)
    with pytest.raises(ValueError, match="does not match"):
        persist_shared_schedule(manifest_path, s2)

    for bad_epochs in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="num_epochs"):
            build_shared_schedule(records, dataset_identity=identity, num_epochs=bad_epochs)


def test_compound_spdx_license_and_crosscodelong_no_duplicate_missing_code():
    from bifocalgemma.data.offline_sft import validate_context_license_record

    mod = _load_prepare_module()
    for spdx in ("MIT OR Apache-2.0", "MIT AND BSD-3-Clause", "(MIT OR Apache-2.0)", "Apache_2.0"):
        rec = mod.adapt_v4_row(
            "fim_midtraining",
            {
                "prefix": "def f(x):\n",
                "suffix": "    return y",
                "middle": "    y = x + 1\n",
                "metadata": {
                    "repository_url": "https://github.com/ok/repo",
                    "license": spdx,
                    "file_path": "a.py",
                },
            },
        )
        validate_context_license_record(rec)

    with pytest.raises(ValueError, match="non_permissive_license"):
        mod.adapt_v4_row(
            "fim_midtraining",
            {
                "prefix": "def f(x):\n",
                "suffix": "    return y",
                "middle": "    y = x + 1\n",
                "metadata": {
                    "repository_url": "https://github.com/ok/repo",
                    "license": "MIT OR GPL-3.0",
                    "file_path": "a.py",
                },
            },
        )

    # Bug #6a: already-normalized # <MISSING_CODE> must not be emitted twice
    for marker in ("# <MISSING_CODE>", "5: # <MISSING_CODE>", "### MISSING CODE"):
        ccl_row = {
            "task_type": "completion",
            "instruction": "Complete code.",
            "context": (
                "### CROSS-FILE CONTEXT\ndef helper(): return 1\n\n"
                "### PREFIX\n"
                + "\n".join(f"pre_{i} = {i}" for i in range(1, 5))
                + f"\n{marker}\n"
                + "### SUFFIX\n"
                + "\n".join(f"suf_{i} = {i}" for i in range(1, 5))
            ),
        }
        split = split_disjoint_context(ccl_row, local_window_lines=32)
        assert split.decoder_local_context.count("# <MISSING_CODE>") == 1
        assert split.full_context.count("# <MISSING_CODE>") == 1
        assert "# <MISSING_CODE>" not in split.encoder_context


def _write_mock_curated_pool(directory: Path, rows: list[dict[str, Any]]) -> None:
    mod = _load_prepare_module()
    normalized_rows = []
    for r in rows:
        before_val = r.get("before", r.get("context", ""))
        item = {
            "path": r.get("path", "repository_context"),
            "before": before_val,
            "after": r.get("after", ""),
            "role": "train",
            "source_revision": r.get("source_revision", "a" * 40),
            "before_signature": r.get("before_signature", mod.code_signature(before_val)),
            "normalized_pair_hash": r.get("normalized_pair_hash", "0" * 64),
            **r,
        }
        normalized_rows.append(item)
    mod.export_pool(normalized_rows, directory, {"benchmark_quarantine": {}})


def test_mixed_pool_v4_admits_foundation_before_patch_and_quarantines_context_pool(
    tmp_path: Path, monkeypatch
):
    mod = _load_prepare_module()
    leaked_fn = "def leaked_eval_helper(x):\n    alpha = x + 10\n    beta = alpha * 2\n    return beta - 3\n"
    leaked_gsm8k = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."

    # 1. Foundation pool with EditPackFT + CommitPackFT (using before & target_patch) + NextCoder (must drop)
    foundation_dir = tmp_path / "foundation"
    foundation_rows = [
        {
            "instance_id": f"ed-{i}",
            "group": f"grp-ed-{i}",
            "path": "pkg/mod.py",
            "before": f"def add_{i}(x):\n    return x\n",
            "after": f"def add_{i}(x):\n    return x + 1\n",
            "instruction": f"Increment return value in add_{i}.",
            "target_patch": f"--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,2 +1,2 @@\n def add_{i}(x):\n-    return x\n+    return x + 1\n",
            "source": "nuprl/EditPackFT" if i < 2 else "bigcode/commitpackft",
            "commit": f"{i:040x}",
            "repos": f"ok/found_{i}",
            "license": "MIT OR Apache-2.0",
            "token_lengths": {"encoder": 64, "target": 16},
            "pool_split": "train" if i < 3 else "dev",
        }
        for i in range(4)
    ]
    foundation_rows.append(
        {
            "instance_id": "nc-drop",
            "group": "grp-nc",
            "path": "a.py",
            "before": "x = 1\n",
            "after": "x = 2\n",
            "instruction": "Drop me",
            "target_patch": "-x=1\n+x=2\n",
            "source": "microsoft/NextCoderDataset",
            "commit": "f" * 40,
            "repos": "ok/nc",
            "license": "mit",
            "pool_split": "train",
        }
    )
    _write_mock_curated_pool(foundation_dir, foundation_rows)

    # 2. Context pool with clean rows + blocked repo + leaked benchmark code
    context_dir = tmp_path / "context"
    lic_sha = "a" * 64
    context_rows = [
        {
            "instance_id": "ctx-ok-1",
            "record_type": "context_sft_v1",
            "role": "train",
            "task_type": "repair",
            "group": "ok/ctx_repo_1",
            "canonical_repos": ["ok/ctx_repo_1"],
            "repos": "ok/ctx_repo_1",
            "instruction": "Fix ok helper",
            "context": "### File: ok.py\ndef ok_fn():\n    return 1\n",
            "target_text": "return 2",
            "source": "nebius/SWE-rebench-V2",
            "source_key": "swe_rebench_v2",
            "source_revision": "a" * 40,
            "snapshot_sha256": "1" * 64,
            "commit": "2" * 40,
            "license": "mit",
            "context_selection": {"gold_used": False, "windows": [{"path": "ok.py"}]},
            "context_source_code_license_evidence": [
                {"path": "ok.py", "licenses": [{"license": "mit", "sha256": lic_sha}]}
            ],
            "token_lengths": {"encoder": 9000, "target": 16},
            "pool_split": "train",
        },
        {
            "instance_id": "ctx-ok-dev",
            "record_type": "context_sft_v1",
            "role": "train",
            "task_type": "repair",
            "group": "ok/ctx_repo_dev",
            "canonical_repos": ["ok/ctx_repo_dev"],
            "repos": "ok/ctx_repo_dev",
            "instruction": "Fix ok dev helper",
            "context": "### File: ok_dev.py\ndef ok_dev():\n    return 3\n",
            "target_text": "return 4",
            "source": "nebius/SWE-rebench-V2",
            "source_key": "swe_rebench_v2",
            "source_revision": "a" * 40,
            "snapshot_sha256": "1" * 64,
            "commit": "2" * 40,
            "license": "mit",
            "context_selection": {"gold_used": False, "windows": [{"path": "ok_dev.py"}]},
            "context_source_code_license_evidence": [
                {"path": "ok_dev.py", "licenses": [{"license": "mit", "sha256": lic_sha}]}
            ],
            "token_lengths": {"encoder": 9000, "target": 16},
            "pool_split": "dev",
        },
        {
            "instance_id": "ctx-blocked-repo",
            "record_type": "context_sft_v1",
            "role": "train",
            "task_type": "repair",
            "group": "blocked/bench-repo",
            "canonical_repos": ["blocked/bench-repo"],
            "repos": "blocked/bench-repo",
            "instruction": "Blocked repo",
            "context": "### File: br.py\ndef br(): return 1\n",
            "target_text": "return 2",
            "source": "nebius/SWE-rebench-V2",
            "source_key": "swe_rebench_v2",
            "source_revision": "a" * 40,
            "snapshot_sha256": "1" * 64,
            "commit": "2" * 40,
            "license": "mit",
            "context_selection": {"gold_used": False, "windows": [{"path": "br.py"}]},
            "context_source_code_license_evidence": [
                {"path": "br.py", "licenses": [{"license": "mit", "sha256": lic_sha}]}
            ],
            "token_lengths": {"encoder": 512, "target": 16},
            "pool_split": "train",
        },
        {
            "instance_id": "ctx-leaked-code",
            "record_type": "context_sft_v1",
            "role": "train",
            "task_type": "repair",
            "group": "ok/ctx_repo_3",
            "canonical_repos": ["ok/ctx_repo_3"],
            "repos": "ok/ctx_repo_3",
            "instruction": "Leaked code",
            "context": f"### Instruction\nLeaked code\n\n### File: leak.py\n{leaked_fn}",
            "target_text": "return 0",
            "source": "nebius/SWE-rebench-V2",
            "source_key": "swe_rebench_v2",
            "source_revision": "a" * 40,
            "snapshot_sha256": "1" * 64,
            "commit": "2" * 40,
            "license": "mit",
            "context_selection": {"gold_used": False, "windows": [{"path": "leak.py"}]},
            "context_source_code_license_evidence": [
                {"path": "leak.py", "licenses": [{"license": "mit", "sha256": lic_sha}]}
            ],
            "token_lengths": {"encoder": 512, "target": 16},
            "pool_split": "train",
        },
    ]
    _write_mock_curated_pool(context_dir, context_rows)

    # 3. Synthetic benchmark_root with quarantine JSON, CrossCodeLong specs, and GSM8K inputs
    bench_root = tmp_path / "benchmarks"
    (bench_root / "crosscodelong_chunk_source_v1/eval_private").mkdir(parents=True)
    (bench_root / "crosscodelong_chunk_source_v1/eval_private/specs.jsonl").write_text(
        json.dumps({"instance_id": "ccl-1", "reference": leaked_fn}) + "\n", encoding="utf-8"
    )
    (bench_root / "gsm8k_zero_shot_v2").mkdir(parents=True)
    (bench_root / "gsm8k_zero_shot_v2/eval_inputs.jsonl").write_text(
        json.dumps({"instance_id": "gsm-1", "benchmark": "gsm8k", "question": leaked_gsm8k}) + "\n",
        encoding="utf-8",
    )
    (bench_root / "benchmark_quarantine.json").write_text(
        json.dumps({"repositories": ["blocked/bench-repo"], "slugs": []}),
        encoding="utf-8",
    )

    auto_sigs, auto_gsm = mod._collect_benchmark_quarantine_assets(bench_root)
    assert mod.code_signature(leaked_fn) in auto_sigs
    assert leaked_gsm8k in auto_gsm

    v4_sources = {
        "smoltalk_metamathqa": [
            {"instance_id": "mm-leak", "instruction": leaked_gsm8k, "output": "#### 24"},
            {"instance_id": "mm-ok", "instruction": "Compute 10 + 15.", "output": "#### 25"},
        ],
        "longalign": [
            {
                "instance_id": "la-ok",
                "instruction": "Summarize module",
                "context": "Document\n" + ("alpha beta\n" * 10),
                "output": "Summary.",
                "encoder_tokens": 12000,
            }
        ],
    }

    out_dir = tmp_path / "mixed_v4_out"
    manifest = mod.mixed_pool_v4(
        v4_sources,
        bench_root,
        out_dir,
        foundation=foundation_dir,
        context=context_dir,
        code_signatures=auto_sigs,
        gsm8k_questions=auto_gsm,
        grounded_max_parents=0,
        min_long_context_ratio=0.0,
        seed=17,
    )
    loaded, _ = load_curated_pool(out_dir)
    all_ids = {r["instance_id"] for r in loaded["train"] + loaded["dev"]}
    # All 4 foundation EditPackFT/CommitPackFT rows admitted!
    for i in range(4):
        assert f"ed-{i}" in all_ids
    assert "nc-drop" not in all_ids
    # Clean context rows admitted; quarantined context rows dropped!
    assert "ctx-ok-1" in all_ids
    assert "ctx-ok-dev" in all_ids
    assert "ctx-blocked-repo" not in all_ids
    assert "ctx-leaked-code" not in all_ids
    # GSM8K overlap rejected; non-overlapping row kept
    assert "mm-leak" not in all_ids
    assert "mm-ok" in all_ids
    assert manifest["report"]["rejections"]["benchmark_repository_quarantine"] >= 1
    assert manifest["report"]["rejections"]["benchmark_code_overlap"] >= 1
    assert manifest["report"]["rejections"]["gsm8k_test_overlap"] >= 1

    # Also verify BLOCKED_V4_DATASETS enforcement on context pool rows directly
    orig_loader = mod.load_curated_pool
    blocked_ds_row = {
        **context_rows[0],
        "instance_id": "ctx-blocked-ds",
        "source": "KodCode/KodCode-V1",
        "source_key": "kodcode",
    }
    monkeypatch.setattr(
        mod,
        "load_curated_pool",
        lambda d: (
            ({"train": [context_rows[0], blocked_ds_row], "dev": [context_rows[1]]}, {})
            if Path(d) == context_dir
            else orig_loader(d)
        ),
    )
    out_dir_2 = tmp_path / "mixed_v4_out_2"
    manifest_2 = mod.mixed_pool_v4(
        v4_sources,
        bench_root,
        out_dir_2,
        context=context_dir,
        grounded_max_parents=0,
        min_long_context_ratio=0.0,
        seed=17,
    )
    loaded_2, _ = orig_loader(out_dir_2)
    assert "ctx-blocked-ds" not in {r["instance_id"] for r in loaded_2["train"] + loaded_2["dev"]}
    assert manifest_2["report"]["rejections"]["blocked_v4_dataset"] >= 1
