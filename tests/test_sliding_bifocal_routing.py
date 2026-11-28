"""Regression tests for Dynamic Sliding Bifocal routing and anti-shortcut training."""

from __future__ import annotations

import re

from bifocalgemma.config import (
    MATRIX_MODELS,
    V4_LAMBDA_PTR_STAGE1,
    V4_LAMBDA_PTR_STAGE2,
    V4_TOTAL_STEPS,
    make_experiment_spec,
)
from bifocalgemma.data.context_tasks import (
    dynamic_sliding_bifocal_split,
    task_prompt,
    tokenize_context_task,
)
from bifocalgemma.evaluation.context import _prompt_ids_with_metadata, build_repo_niah_case
from bifocalgemma.training.matrix import _resolve_profile_settings
from bifocalgemma.training.objectives import streaming_pointer_auxiliary_loss


class _SimpleTokenizer:
    """Deterministic whitespace/char-level test tokenizer with Gemma-style special tokens."""

    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0

    def convert_tokens_to_ids(self, tok: str) -> int:
        if tok == "<turn|>":
            return 107
        return 3

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        content = messages[-1]["content"]
        if add_generation_prompt:
            return f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
        return f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<turn|>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return [self.bos_token_id] if add_special_tokens else []
        tokens = [10 + (sum(ord(c) for c in word) % 200) for word in re.findall(r"\S+|\n", text)]
        if not tokens:
            tokens = [11]
        if add_special_tokens:
            return [self.bos_token_id, *tokens]
        return [self.bos_token_id, *tokens] if text.startswith("<bos>") else tokens

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(str(i) for i in ids)


def _line_numbers(rendered: str) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(r"(?m)^\s*(\d+)\s*:", rendered)}


def test_short_context_bypass_for_repair_qa_and_mcq():
    tok = _SimpleTokenizer()
    spec = make_experiment_spec("2it-2it")

    rows = [
        {
            "instance_id": "short-repair",
            "source": "synthetic",
            "disjoint_context": True,
            "task_type": "repair",
            "instruction": "Fix off-by-one bug",
            "context": "### File: app.py\ndef add_one(x):\n    return x + 2\n",
            "target_text": "diff --git a/app.py b/app.py\n-    return x + 2\n+    return x + 1\n",
        },
        {
            "instance_id": "short-qa",
            "source": "synthetic",
            "disjoint_context": True,
            "task_type": "qa",
            "instruction": "What does add_one return?",
            "context": "### File: app.py\ndef add_one(x):\n    return x + 1\n",
            "target_text": "x + 1",
        },
        {
            "instance_id": "short-mcq",
            "source": "synthetic",
            "disjoint_context": True,
            "task_type": "mcq",
            "instruction": "Which value is returned?\nA) x\nB) x + 1",
            "context": "### File: app.py\ndef add_one(x):\n    return x + 1\n",
            "target_text": "B",
        },
    ]

    for row in rows:
        tokenized = tokenize_context_task(
            row,
            spec,
            tok,
            tok,
            encoder_limit=4096,
            decoder_limit=8192,
        )
        assert tokenized.encoder_ids == ()

        eval_row = {
            "role": "eval",
            "instance_id": row["instance_id"],
            "task_type": row["task_type"],
            "instruction": row["instruction"],
            "context": row["context"],
        }
        enc_ids, _, _ = _prompt_ids_with_metadata(
            eval_row,
            spec,
            tok,
            tok,
            encoder_limit=4096,
            decoder_limit=8192,
        )
        assert enc_ids == ()

        split = dynamic_sliding_bifocal_split(
            row,
            decoder_context_token_budget=4096,
            encoder_token_budget=4096,
            decoder_tokenizer=tok,
            encoder_tokenizer=tok,
        )
        assert split.encoder_context == ""
        assert "### Issue" not in split.encoder_context
        assert "add_one" in split.decoder_local_context


def test_long_context_dynamic_split_disjoint_and_zero_gold_leakage():
    tok = _SimpleTokenizer()
    spec = make_experiment_spec("2it-2it")

    # > 9k chars across 320 numbered lines with distinct needle identifiers
    code_lines = [
        f"def helper_{i}(val): return val + {i}  # filler token_{i}" for i in range(1, 321)
    ]
    code_lines[180] = "def target_needle_fn(val): return val * 42  # SECRET_NEEDLE_ANCHOR"
    long_code = "\n".join(code_lines)
    assert len(long_code) > 9000

    for task_type, instruction, target_field, target_a, target_b in [
        (
            "qa",
            "What does target_needle_fn return?",
            "target_text",
            "val * 42",
            "COMPLETELY_DIFFERENT_GOLD_ANSWER_XYZ",
        ),
        (
            "completion",
            "Complete target_needle_fn",
            "target_text",
            "return val * 42",
            "return val - 999",
        ),
        (
            "repair",
            "Fix target_needle_fn",
            "target_patch",
            "diff --git a/a.py b/a.py\n-return 0\n+return val * 42\n",
            "diff --git a/a.py b/a.py\n-return 0\n+return 123456\n",
        ),
    ]:
        row_a = {
            "instance_id": f"long-{task_type}",
            "source": "synthetic",
            "disjoint_context": True,
            "task_type": task_type,
            "instruction": instruction,
            "context": f"### File: mod.py\n{long_code}\n",
            target_field: target_a,
            "correct_answer": target_a,
        }
        row_b = {
            **row_a,
            target_field: target_b,
            "correct_answer": target_b,
        }

        split_a = dynamic_sliding_bifocal_split(
            row_a,
            decoder_context_token_budget=1000,
            encoder_token_budget=8192,
            decoder_tokenizer=tok,
            encoder_tokenizer=tok,
        )
        split_b = dynamic_sliding_bifocal_split(
            row_b,
            decoder_context_token_budget=1000,
            encoder_token_budget=8192,
            decoder_tokenizer=tok,
            encoder_tokenizer=tok,
        )

        assert len(split_a.decoder_local_context) > 0
        assert len(split_a.encoder_context) > 0
        assert split_a == split_b

        dec_lines = _line_numbers(split_a.decoder_local_context)
        enc_lines = _line_numbers(split_a.encoder_context)
        assert dec_lines and enc_lines
        assert dec_lines.isdisjoint(enc_lines)

        tok_a = tokenize_context_task(
            row_a,
            spec,
            tok,
            tok,
            encoder_limit=8192,
            decoder_limit=1024,
        )
        tok_b = tokenize_context_task(
            row_b,
            spec,
            tok,
            tok,
            encoder_limit=8192,
            decoder_limit=1024,
        )
        assert tok_a.encoder_ids == tok_b.encoder_ids
        assert tok_a.decoder_prefix_ids == tok_b.decoder_prefix_ids
        assert len(tok_a.encoder_ids) > 0


def test_training_anti_shortcut_contracts_foveal_budget_deterministically():
    tok = _SimpleTokenizer()
    spec = make_experiment_spec("2it-2it")

    # Build a long context (>1024 tokens) that fits in decoder_limit=4096 without anti-shortcut
    # so that anti-shortcut contraction (budget<=768) moves lines into encoder_ids.
    code_lines = [f"def fn_{i}(x): return x + {i}" for i in range(1, 240)]
    context = "### File: main.py\n" + "\n".join(code_lines) + "\n"

    contracted_ids: list[str] = []
    for idx in range(30):
        iid = f"anti-shortcut-{idx}"
        row = {
            "instance_id": iid,
            "source": "synthetic",
            "disjoint_context": True,
            "task_type": "repair",
            "instruction": "Fix fn_10",
            "context": context,
            "target_text": "diff --git a/main.py b/main.py\n+return x + 10\n",
        }
        base_1 = tokenize_context_task(
            row,
            spec,
            tok,
            tok,
            encoder_limit=8192,
            decoder_limit=4096,
            training_anti_shortcut=False,
        )
        anti_1 = tokenize_context_task(
            row,
            spec,
            tok,
            tok,
            encoder_limit=8192,
            decoder_limit=4096,
            training_anti_shortcut=True,
        )
        anti_2 = tokenize_context_task(
            row,
            spec,
            tok,
            tok,
            encoder_limit=8192,
            decoder_limit=4096,
            training_anti_shortcut=True,
        )
        assert anti_1.encoder_ids == anti_2.encoder_ids
        assert anti_1.decoder_prefix_ids == anti_2.decoder_prefix_ids
        if len(anti_1.encoder_ids) > len(base_1.encoder_ids):
            contracted_ids.append(iid)

    assert len(contracted_ids) > 0


def test_paired_false_baseline_and_v4_namespace_invariance():
    row = {
        "instance_id": "baseline-1",
        "task_type": "repair",
        "instruction": "Fix bug",
        "context": "### File: a.py\ndef foo():\n    return 1\n",
    }
    prompt = task_prompt(row, paired=False)
    assert "### File: a.py" in prompt
    assert "def foo():" in prompt

    assert MATRIX_MODELS["e2b-it"]["subfolder"] == "latest"
    assert MATRIX_MODELS["e4b-it"]["subfolder"] == "latest"
    assert (
        _resolve_profile_settings("v4", family="decoder_only")["run_namespace"]
        == "versionless-repair-v4"
    )
    assert (
        _resolve_profile_settings("v4", family="symmetric")["run_namespace"] == "sliding-bifocal-v4"
    )
    assert V4_TOTAL_STEPS == 4494
    assert V4_LAMBDA_PTR_STAGE1 == 0.10
    assert V4_LAMBDA_PTR_STAGE2 == 0.03


class _RecordingTokenizer(_SimpleTokenizer):
    """Captures exact rendered chat prompts and encoder bodies passed into encode()."""

    def __init__(self) -> None:
        self.chat_prompts: list[str] = []
        self.encoded_texts: list[str] = []

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        self.chat_prompts.append(messages[-1]["content"])
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encoded_texts.append(text)
        return super().encode(text, add_special_tokens=add_special_tokens)


def test_gsm8k_paired_eval_matches_clean_prose_without_line_numbers_or_duplication():
    base_spec = make_experiment_spec("e2b-it")
    pair_spec = make_experiment_spec("2it-2it")
    row = {
        "instance_id": "gsm8k:test:0000",
        "role": "eval",
        "benchmark": "gsm8k",
        "question": "Janet has 16 eggs per day. How much does she make?",
        "instruction": "Solve the math problem step by step. End with #### followed by the final numeric answer.",
        "context": "Question: Ex1\nAnswer: 1\n\nQuestion: Janet has 16 eggs per day. How much does she make?\nAnswer:",
    }

    tok_base = _RecordingTokenizer()
    enc_base, pref_base, _ = _prompt_ids_with_metadata(
        row, base_spec, None, tok_base, encoder_limit=32768, decoder_limit=8192
    )
    tok_pair = _RecordingTokenizer()
    enc_pair, pref_pair, _ = _prompt_ids_with_metadata(
        row, pair_spec, tok_pair, tok_pair, encoder_limit=32768, decoder_limit=8192
    )

    assert enc_base == ()
    assert enc_pair == ()
    assert pref_pair == pref_base
    assert tok_pair.chat_prompts == tok_base.chat_prompts
    rendered_prompt = tok_pair.chat_prompts[-1]
    assert "### Context" not in rendered_prompt
    assert "1: Question:" not in rendered_prompt
    assert rendered_prompt.count("Janet has 16 eggs per day.") == 1


def test_reponiah_disables_lexical_routing_so_distant_needle_stays_in_encoder():
    tok = _RecordingTokenizer()
    pair_spec = make_experiment_spec("2it-2it")
    pub, gold = build_repo_niah_case(4200, 0.75, seed=19, tokenizer=tok)

    enc, pref, _ = _prompt_ids_with_metadata(
        pub,
        pair_spec,
        tok,
        tok,
        encoder_limit=32768,
        decoder_limit=1600,
        max_new_tokens_reserve=128,
    )

    assert len(enc) > 0
    assert len(pref) + 128 <= 1600
    assert gold["needle"] not in tok.chat_prompts[-1]
    assert "### Context" not in tok.chat_prompts[-1]
    assert any(gold["needle"] in body for body in tok.encoded_texts if body != tok.chat_prompts[-1])


def test_unnumbered_fallback_triggers_before_any_block_is_dropped_on_long_code():
    tok = _RecordingTokenizer()
    pair_spec = make_experiment_spec("2it-2it")
    # 120 numbered code lines: with `1: ` line numbers each line adds 1 extra token (+120 tokens),
    # overflowing encoder_limit=380 + decoder_limit=220 unless number_lines=False activates.
    code_lines = [f"x_{i} = y_{i}" for i in range(1, 121)]
    code_lines[-1] = "FINAL_TAIL_SENTINEL = 999"
    row = {
        "instance_id": "overflow-unnumbered-fallback",
        "role": "eval",
        "task_type": "qa",
        "instruction": "Inspect module",
        "context": "### File: tail_mod.py\n" + "\n".join(code_lines),
    }

    enc, pref, used_num = _prompt_ids_with_metadata(
        row,
        pair_spec,
        tok,
        tok,
        encoder_limit=340,
        decoder_limit=180,
        max_new_tokens_reserve=64,
    )
    assert used_num is False
    assert len(enc) <= 340
    assert len(pref) + 64 <= 180
    assert any("FINAL_TAIL_SENTINEL = 999" in t for t in tok.encoded_texts[-3:])


def test_oversized_single_line_prose_splits_cleanly_into_fovea_and_encoder():
    tok = _SimpleTokenizer()
    sentences = [
        f"Section {i} discusses project governance and climate adaptation metrics in detail."
        for i in range(1, 160)
    ]
    sentences[95] = "CRITICAL_GOVERNANCE_NEEDLE describes the exact board oversight responsibility."
    giant_two_line_doc = "Document Title\n" + " ".join(sentences)

    row = {
        "instance_id": "longalign-two-line",
        "task_type": "long_context_qa",
        "instruction": "What is the CRITICAL_GOVERNANCE_NEEDLE oversight responsibility?",
        "context": giant_two_line_doc,
    }
    split = dynamic_sliding_bifocal_split(
        row,
        decoder_context_token_budget=400,
        encoder_token_budget=4096,
        decoder_tokenizer=tok,
        encoder_tokenizer=tok,
    )
    assert "CRITICAL_GOVERNANCE_NEEDLE" in split.decoder_local_context
    assert len(split.encoder_context) > 0
    assert "### Context" not in split.decoder_local_context
    assert "1: " not in split.decoder_local_context


def test_grounded_line_lookup_stays_in_encoder_and_reconstruction_targets_exact_line():
    tok = _SimpleTokenizer()
    numbered_lines = [f"{i}: def helper_{i}(): return {i}" for i in range(1, 80)]
    numbered_lines[10] = "11: # <MISSING_CODE>"  # pre-existing parent marker
    numbered_lines[64] = "65: # <MISSING_CODE>"  # actual reconstruction target line
    recon_ctx = "### Numbered source\n" + "\n".join(numbered_lines)

    recon_row = {
        "instance_id": "recon-65",
        "task_type": "source_reconstruction",
        "instruction": "Reconstruct source line 65 replacing # <MISSING_CODE>. Return only that line, preserving indentation.",
        "context": recon_ctx,
    }
    recon_split = dynamic_sliding_bifocal_split(
        recon_row,
        decoder_context_token_budget=320,
        encoder_token_budget=4096,
        decoder_tokenizer=tok,
        encoder_tokenizer=tok,
    )
    assert "65: # <MISSING_CODE>" in recon_split.decoder_local_context
    assert "11: # <MISSING_CODE>" in recon_split.encoder_context

    mcq_lines = [f"{i}: value_{i} = {i * 10}" for i in range(1, 80)]
    mcq_row = {
        "instance_id": "mcq-55",
        "task_type": "source_mcq",
        "instruction": (
            "Which option is the exact text at numbered line 55? Return only its letter.\n"
            "A. value_12 = 120\nB. value_55 = 550\nC. value_30 = 300\nD. value_70 = 700"
        ),
        "context": "### Numbered source\n" + "\n".join(mcq_lines),
    }
    mcq_split = dynamic_sliding_bifocal_split(
        mcq_row,
        decoder_context_token_budget=2000,
        encoder_token_budget=4096,
        decoder_tokenizer=tok,
        encoder_tokenizer=tok,
    )
    assert mcq_split.decoder_local_context == ""
    assert "55: value_55 = 550" in mcq_split.encoder_context


def test_eval_decoder_overflow_backoff_keeps_prefix_within_tight_budget():
    tok = _RecordingTokenizer()
    pair_spec = make_experiment_spec("2it-2it")
    long_instr = " ".join(f"ident_{i}" for i in range(60))
    code_lines = [f"x_{i} = {i} + ident_{i % 60}" for i in range(1, 180)]
    row = {
        "instance_id": "backoff-eval-1",
        "role": "eval",
        "task_type": "completion",
        "instruction": long_instr,
        "context": "### File: mod.py\n" + "\n".join(code_lines),
    }

    enc, pref, _ = _prompt_ids_with_metadata(
        row,
        pair_spec,
        tok,
        tok,
        encoder_limit=8192,
        decoder_limit=1024,
        max_new_tokens_reserve=512,
    )
    assert len(pref) + 512 <= 1024
    assert len(enc) <= 8192


def test_streaming_pointer_loss_is_dimension_invariant_and_calibrated():
    import torch

    def _run_loss(d_model: int) -> torch.Tensor:
        torch.manual_seed(11)
        h_dec = torch.randn(1, 2, d_model, requires_grad=True)
        z_enc = torch.randn(1, 4, d_model, requires_grad=True)
        with torch.no_grad():
            h_dec[0, 0] = z_enc[0, 2]
            h_dec[0, 1] = -z_enc[0, 0]
        enc_ids = torch.tensor([[10, 20, 30, 40]])
        tgt_ids = torch.tensor([[30, 50]])
        enc_mask = torch.ones_like(enc_ids, dtype=torch.bool)
        loss = streaming_pointer_auxiliary_loss(
            h_dec, z_enc, enc_ids, tgt_ids, enc_mask, use_checkpoint=False
        )
        loss.backward()
        assert h_dec.grad is not None and z_enc.grad is not None
        assert torch.isfinite(h_dec.grad).all() and torch.isfinite(z_enc.grad).all()
        return loss.detach()

    loss_64 = _run_loss(64)
    loss_256 = _run_loss(256)
    assert torch.isfinite(loss_64) and torch.isfinite(loss_256)
    # Orthogonal distractors have ~0 cosine while exact match has cosine +1.0 scaled by 10.0
    assert float(loss_64.item()) < 0.05
    assert abs(float(loss_64.item()) - float(loss_256.item())) < 0.05


def test_tokenize_context_task_long_target_overflow_shifts_to_encoder_without_error():
    tok = _RecordingTokenizer()
    spec = make_experiment_spec("2it-2it")
    # Completion reserved_out is min(2304, decoder_limit // 2) = 512 when decoder_limit = 1024,
    # leaving ~450 tokens of initial dec_ctx_budget. Put ~320 tokens of context (smaller than 450,
    # triggering the old slack trap!) and a 760-token target (> 512) so total would be ~1100 > 1024.
    code_lines = [f"var_{i} = func_{i}(arg_{i})" for i in range(1, 90)]
    code_lines[40] = "# <MISSING_CODE>"
    long_target = "\n".join(f"out_{i} = compute_{i}(val_{i}, extra_{i})" for i in range(1, 150))
    row = {
        "instance_id": "long-target-completion-1",
        "task_type": "completion",
        "source": "nebius/SWE-rebench-V2",
        "instruction": "Fill in # <MISSING_CODE>.",
        "context": "### File: big.py\n" + "\n".join(code_lines),
        "target_text": long_target,
        "disjoint_context": True,
    }

    ex = tokenize_context_task(
        row,
        spec,
        tok,
        tok,
        encoder_limit=4096,
        decoder_limit=1024,
        disjoint_context=True,
    )
    assert len(ex.decoder_prefix_ids) + len(ex.target_ids) - 1 <= 1024
    assert 0 < len(ex.encoder_ids) <= 4096


def test_tokenize_context_task_encoder_overflow_falls_back_to_unnumbered():
    tok = _RecordingTokenizer()
    spec = make_experiment_spec("2it-2it")
    code_lines = [f"x_{i} = y_{i}" for i in range(1, 121)]
    code_lines[-1] = "TAIL_TOKEN_SENTINEL = 777"
    row = {
        "instance_id": "train-enc-unnumbered-1",
        "task_type": "qa",
        "source": "nebius/SWE-rebench-V2",
        "instruction": "Inspect module",
        "context": "### File: tail_mod.py\n" + "\n".join(code_lines),
        "target_text": "TAIL_TOKEN_SENTINEL = 777",
        "disjoint_context": True,
    }

    ex = tokenize_context_task(
        row,
        spec,
        tok,
        tok,
        encoder_limit=340,
        decoder_limit=180,
        disjoint_context=True,
    )
    assert len(ex.decoder_prefix_ids) + len(ex.target_ids) - 1 <= 180
    assert len(ex.encoder_ids) <= 340
    assert any("TAIL_TOKEN_SENTINEL = 777" in t for t in tok.encoded_texts[-4:])
