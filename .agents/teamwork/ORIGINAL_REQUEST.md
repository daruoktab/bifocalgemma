# Original User Request

## Parameters

[skip-interview]
How this run was launched; not part of the task.

## Request — 2026-11-02T08:54:35Z

kerjakan mereka dengan sub-sub agent yang banyak yang spesifik

Rincian konteks tugas yang harus dikerjakan:
Terdapat 4 perbaikan teknis algoritma penting pada arsitektur dan pipeline pelatihan BifocalGemma V3 di `tools/molab_matrix_support.py`:

1. [P1] Replay Prompt Mismatch (tools/molab_matrix_support.py:1634):
Saat ini `synthesize_comprehension_examples` mengganti target dengan potongan token acak, tetapi mempertahankan `decoder_prefix_ids` dari instruksi repair asli. Ini mengajarkan model menjawab prompt repair bug repository dengan potongan token acak.
Solusi: Buat prompt instruksi yang sesuai dengan tugasnya:
- Untuk `span_copy`: Prompt instruksi seperti "Extract and copy the exact target identifier or code span from the context."
- Untuk `code_qa_mcq`: Prompt instruksi seperti "Identify the correct symbol or answer based on the context."
Tokenize prompt instruksi ini menggunakan `decoder.apply_chat_template(..., tokenize=False, add_generation_prompt=True)` lalu encode untuk mendapatkan `decoder_prefix_ids` yang benar dan selaras dengan target.

2. [P2] Pointer Loss Backward Autograd Memory Overhead (tools/molab_matrix_support.py:711-714):
Akumulasi `total_loss = total_loss + valid_nll.sum()` mempertahankan graph autograd semua chunk hingga backward, memakan ~4.5 GiB VRAM ekstra pada konteks 32K enc x 8K dec.
Solusi: Gunakan `torch.utils.checkpoint.checkpoint` per chunk (atau streaming chunk computation) agar tensor pairwise float32 dan mask tidak ditahan serentak di memory. Selain itu, filter token trivial/tanda baca/whitespace/stopwords dari `match_mask` agar pointer loss secara spesifik menargetkan identifier dan kode bermakna, bukan spasi atau koma sembarangan.

3. [P2] Differential Diff Token Weighting Inactive (tools/molab_matrix_support.py:882):
Saat ini `compute_diff_token_weights(batch.labels)` dipanggil tanpa `add_token_ids`, `del_token_ids`, `hdr_token_ids`. Semua label valid mendapat bobot sama lalu dinormalisasi menjadi 1.0 (loss kembali ke uniform CE biasa).
Solusi: Ambil token IDs penanda diff (`+`, `-`, `diff`, `---`, `+++`, `@@`) dari tokenizer decoder dan teruskan ke pemanggilan `compute_diff_token_weights(batch.labels, ...)`.

4. [P2] Parameter-Name-Matched Optimizer State Transfer (tools/molab_matrix_support.py:2486-2491):
Saat transisi dari Stage 1 ke Stage 2, jumlah parameter group berubah sehingga `optimizer.load_state_dict` melempar exception yang di-`pass` diam-diam oleh `try...except`. Akibatnya momentum dan varians AdamW dari Stage 1 hilang saat masuk Stage 2.
Solusi: Petakan dan pulihkan tensor state optimizer (`exp_avg`, `exp_avg_sq`, `step`) berdasarkan nama parameter (`param_names`) yang cocok secara eksplisit antar-checkpoint dan optimizer aktif.

5. Verifikasi & Regenerasi:
- Jalankan `tools/generate_full_matrix_notebooks.py` untuk mengupdate ke-10 notebook di `notebooks/train_*.py`.
- Pastikan semua verifikasi lulus: `pyrefly check`, `ty check`, `C:\Users\daru\miniconda3\envs\ai\python.exe -m ruff check --fix`, `C:\Users\daru\miniconda3\envs\ai\python.exe -m pytest tests/test_molab_matrix_notebooks.py -q`, dan `C:\Users\daru\miniconda3\envs\ai\python.exe -m marimo check notebooks/train_*.py`.


## Follow-up — 2026-11-02T09:06:37Z

Direct steering update from Lead Architect to teamwork orchestrator:

1. **Dataset Insights**:
   - `outputs/dataset_release_context32k_v3/collections/mixed_context_enc32k_dec8k_v3` (`pool.zip/train.jsonl`) contains **44,408 total rows**:
     - 41,040 `repair` records (92.4%)
     - 2,241 `completion` records (5.0%, prompt: "Return only the missing Python code, preserving its indentation.")
     - 1,127 `localization` records (2.5%, prompt: "Return only a JSON list of repository file paths relevant to fixing the issue.")
   - **Key Guideline**: `build_anti_forgetting_curriculum` should first harvest and prioritize these 3,368 authentic `completion` and `localization` records as natural replay examples. For any additional synthesized comprehension examples, do NOT keep the repair prompt prefix; construct an explicit prompt (e.g. "Extract and copy the exact target identifier or code span from the context.") and tokenize it via `decoder.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)` so that prompt instructions match the target spans.

2. **Gemma-4 Diff Token IDs for Weighting**:
   - Verified token IDs:
     - `add_token_ids`: `(236862, 900)` (`+`, ` +`)
     - `del_token_ids`: `(236772, 753)` (`-`, ` -`)
     - `hdr_token_ids`: `(16417, 7243, 24523, 10379, 15338, 2617, 13705)` (`diff`, `---`, `+++`, `@@`, ` @@`, `--`, `git`)
     - Newline: `107`
   - Wire these token IDs from decoder tokenizer into `compute_diff_token_weights(batch.labels, ...)`.

3. **Pointer Loss Autograd Checkpoint & Token Filtering**:
   - Use `torch.utils.checkpoint.checkpoint` per chunk to avoid the 4.5 GiB autograd graph retention across all chunks.
   - Filter trivial punctuation/whitespace tokens (token `\n`: 107, space ` `: 236758, etc.) from `match_mask` so pointer loss specifically grounds identifiers and meaningful code tokens.

4. **Optimizer State Transfer Across Stages**:
   - PyTorch `AdamW.load_state_dict` fails when param groups change from Stage 1 (3 groups) to Stage 2 (4 groups).
   - In `restore_matrix_checkpoint`, map optimizer state tensor by parameter names: iterate through `model.named_parameters()`, locate parameter state in `saved_state_dict["state"]`, and copy `exp_avg`, `exp_avg_sq`, and `step` directly into the active optimizer state tensor for matching parameters.

5. **Safety Constraint**:
   - Do NOT edit files under `src/bifocalgemma/` directly (keep AST fingerprint intact for `tests/test_benchmark_training_compatibility.py`). All logic belongs in `tools/molab_matrix_support.py`.

Incorporate these findings directly into Milestone 1 & 2 implementation.


## Follow-up — 2026-11-02T10:23:22Z

The 2 untracked challenger scratch scripts (`tests/test_challenger_molab_2_stress.py` and `tests/test_pointer_loss_empirical_stress.py`) have been moved to their respective workspace directories in `.agents/teamwork/challenger_molab_2/` and `.agents/teamwork/challenger_molab_1/`.

Immediate verification results on the clean workspace:
- `pyrefly check`: INFO 0 errors (Exit code 0)
- `ty check`: All checks passed! (Exit code 0)
- `ruff check --fix`: All checks passed! (Exit code 0)
- `pytest tests/test_benchmark_training_compatibility.py`: 1 passed (Exit code 0)
- `pytest tests/test_molab_matrix_notebooks.py -q`: 54 passed (Exit code 0)
- `marimo check`: All 10 notebooks clean (Exit code 0)

Please trigger the clean re-audit now.
