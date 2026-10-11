# Dataset and benchmark inventory

Current defaults use instruction_enc8k_dec2k_v2 from the private HF repository at the commit in [the lock](../configs/dataset_release.lock.json). **45,230 auxiliary single-file editing records: 40,706 train /4,524 dev**. Three credited sources are EditPackFT, CommitPackFT Python and NextCoder Python. Encoder cap8,192 /paired decoder2,048 /baseline decoder10,752; fresh real-tokenizer profiling and static compile/patch checks are documented in [the v2 audit](DATASET_REMEDIATION_V2.md). Historical sections below describe their named earlier builds and source research; they are not current release counts.

The training cell consumes the selected frozen collection only. An existing local directory must match the pinned collection fingerprint. MAX_STEPS=0 completes configured epochs; v2 requires a new run, not optimizer resume from v1. HumanEvalFix164 and GSM8K1,319 unique questions in two protocols are the published evaluation artifacts. Other catalog entries and the repository corpus remain pending.

## Instruction foundation: implemented, complete local historical manifest

| Source | Raw release used | Format / input and target | Retained after all historical gates | Raw cache size |
| --- | ---: | --- | ---: | ---: |
| nuprl/EditPackFT | 22,602 Python edits | Parquet; old/new file contents plus editing instruction | 18,086 train +2,043 dev =20,129 | 48,124,127 bytes |
| bigcode/commitpackft (Python) | 56,025 | JSONL; old/new code, commit subject/message, repository and license metadata | 14,803 train +1,704 dev =16,507 | 135,858,935 bytes |
| microsoft/NextCoderDataset | 381,123 mixed-language rows; 45,837 Python | Six Arrow shards; prompt/completion fenced-code adapter, synthetic instruction editing | 11,444 train +1,179 dev =12,623 | 2,627,226,320 bytes, all six shards |

Historical final pool: **49,259 examples =44,333 train /4,926 dev**, 16,513,084
supervised train tokens, 11,084 optimizer updates for one epoch at batch1/accum4.
The export occupies 479,291,101 bytes including train/dev and manifest; indexes
and source caches are additional. See [frozen receipt](data/full_instruction_pool_20261009.json).
These counts describe that immutable build, not the running Molab build or a
new syntax-diagnostics build before its manifest exists.

All three adapters produce unified-diff targets. Paired models receive the old
code/instruction context in the encoder; baselines receive the same visible
information in their decoder prefix. These are primarily file editing tasks,
not complete repository snapshots or exclusively bug fixes. NextCoder has no
verified original-repository lineage; synthetic metadata is not a substitute
for source-history evidence. Per-record code license metadata is checked for
EditPackFT/CommitPackFT, while NextCoder's declared synthetic dataset license
is retained with its provenance limitation.

## Repository corpus: planned primary study and optional secondary source

| Source | Actual raw table | Candidates before snapshot gates | Role and implementation |
| --- | --- | --- | --- |
| nebius/SWE-rebench-V2 | 32,079 rows; 7,243 Python | 7,192 after 46 benchmark-repo and 5 lineage exclusions; 685 repos | Retained source for the planned repository SFT stage; static snapshot/patch/context/licensing builder implemented, final production pool pending |
| internlm/SWE-Fixer-Train-110K | 115,406 rows under the source adapter's Python convention | 114,352 after table gates; 846 repos | Optional source supported by the builder, not automatically mixed into the default pool |

SWE-rebench V2 raw Parquet is 428,839,266 bytes. SWE-Fixer raw JSONL is
1,471,204,778 bytes. Before-repository archives are additional and dominate the
storage budget. Dataset metadata is CC-BY-4.0 for V2 /MIT for SWE-Fixer in our
provenance, but **source-code licenses are checked independently**. All SWE-Fixer
table rows lacked source-code license metadata; its language field is absent in
the inspected schema, so the adapter convention is not per-file Python proof.

Input: released public issue plus BM25-selected windows from the pinned before
snapshot. Hints, test patches, gold file identities and repair targets do not
select the training context. Target: the released repair unified diff. The
builder verifies static patch application, repaired Python AST, source-code
licenses for patched and all selected context files, and benchmark overlap.
No repository code or tests execute during preparation. Static validation does
not establish a functional repair. Missing binaries, links and non-UTF8 assets
remain explicit limits of text-only archives.

Resource defaults: at most 10,000 accepted tasks, 50/repo, 20 GiB archive cache;
individual archives <=100 MB and text snapshots <=50 MB /5,000 files. V2 has
only 7,192 table candidates before stricter gates, so 10,000 is an upper budget,
not an achievable-count promise. Train/dev uses connected repository/commit/
before-seed groups with an approximately 10% dev target; final counts are unknown.

Current local notebook caps (user update, 10 October): encoder8,192 /paired
decoder2,048 /baseline decoder10,752. Historical 49,259-example counts above
remain tied to the older 2k/2k/4,608 frozen pool; the superseded v1 8k release had 44,366 train /4,930 dev; current v2 has 40,706 train /4,524 dev. Source
profiles show 8,192 paired-decoder coverage of 96.01% for V2 and 98.63% for
SWE-Fixer. An 8k paired-decoder budget remains a separate candidate requiring
original-model VRAM testing. The requested encoder8k default does not imply
that decoder8k has been selected or that training memory has been validated. New caps require a new named frozen pool.

## Benchmarks: evaluation roles and current implementation status

All evaluation data stays outside the training pools. Five code benchmarks are
already mandatory quarantine inputs; that alone does not mean all five have a
production evaluation runner.

| Benchmark | Released/selected task count | Purpose | Actual implementation / limit |
| --- | ---: | --- | --- |
| HumanEvalFix Python, bigcode/humanevalpack | 164 | Function-level bug repair | Unified-diff adapter and native Python test scorer implemented; 164 gold repairs passed the local scorer check. Original-model scores and official runtime parity remain pending. |
| CanItEdit | 105 | Instruction-following code editing | Actual before/after data is quarantined. Dedicated production prepare/scoring adapter is not implemented in repository_study.py yet; proposed repair/editing evaluation, not an already enabled runner. |
| RES-Q | 100 released, 95 Python /5 JS | Repository-level repair/editing | Source adapter exists; native Python execution is opt-in and needs task-specific environments/dependencies. Complete real gold/harness validation is pending; JS is not covered by the Python runner. |
| RepoQA Python | 100 | Retrieve a function described in natural language | Adapted diagnostic implemented. Current shortened windows are centered on the known function span and intentionally contain the answer. This is an oracle-window reader diagnostic, not full-repository long-context search or the official RepoQA protocol. |
| CrossCodeEval Python | 2,665 across 471 encoded repo slugs | Cross-file next-statement completion | Non-reference precomputed cross-file context adapter; adapted exact code match. Official identifier/parser metrics are not implemented. |
| GSM8K | 1,319 test, 7,473 train for optional demos | Arithmetic reasoning retention before/after coding SFT | Pinned prepare/generate/score CLI implemented; zero-shot and 8-shot use identical frozen test IDs. First-eight train examples supply demos only. Release-style first #### numeric-string extraction, comma removal, fixed denominator; no calculator/RL/VM. Model accuracy remains unmeasured. |

RepoQA and CrossCodeEval scores are adapted_exact_code_match with
 official_benchmark_score=False. Do not compare them directly with published
upstream scores. RepoQA's answer-centered crop is a material protocol limitation,
not evidence of long-context dependency understanding. HumanEvalFix's gold check
validates the scorer only. RES-Q is low infrastructure only for tasks whose native
environments actually work; it is not yet certified VM-free for every task.

GSM8K measured maxima under the declared protocol: 222 zero-shot prompt tokens,
1,280 eight-shot prompt tokens, 301 cleaned reference-answer tokens. Generation
may be longer; its candidate budget is 2k input /2k new tokens, total decoder4k.
It uses a math prompt/scorer, never the repair prompt. See [GSM8K guide](GSM8K_CONTEXT.md).

## Excluded, conditional and registry-only names

InstructCoder and github-codereview are not in SOURCE_REFS, so they are not in the
active full instruction pool. Code-Feedback and SWE-smith-py are excluded from
this recipe. SWE-Gym, R2E-Gym, RepoExec, DevEval and SWE-bench appear in the
research/provenance registry but are not active training-source dispatch here.
T5Gemma/T5Gemma2 are prior art, not benchmark baselines in the six-model matrix.

The matrix remains E2B-base/E2B-IT, E4B-base/E4B-IT, E2B-base/E4B-IT,
E4B-base/E2B-IT, plus E2B-IT and E4B-IT decoder-only baselines. Same frozen
cohort per experiment, batch1, accumulation4, attention LoRA rank16/alpha32,
ordinary words/PLE frozen, default LR2e-5 and warmup5. Symmetric pairs share
ordinary embeddings; asymmetric pairs train one alignment projection. One model
per run; no automatic concurrent training, RL, distillation or teacher model.


## Research background and limitations retained from earlier planning

Earlier rationale and native dependency checks remain in
[Molab data/evaluation plan](MOLAB_DATA_EVALUATION_PLAN.md),
[repository-data research](RESEARCH_REPOSITORY_DATA_AND_BENCHMARKS.md), and
[SWE-smith direction contract](NATIVE_LOCAL_VALIDATION.md). Their historical
"planned" or "pilot" statements must be read alongside the current dispatch
inventory above. CanItEdit's earlier 83-case non-stdlib-import screen is only
a dependency heuristic; it is not a validated native functional subset.

SWE-smith supports SFT and does not inherently require RL; its executable
environment/harness is the additional infrastructure excluded from this recipe.
Native Python execution is opt-in and is not a security sandbox. Functional
pass rate cannot be inferred from patch application, AST parsing or held-out CE.
Exact match can reject a different but correct repair. Pretraining contamination
is not certified away by repository/code-overlap filters.

Direct base-encoder/IT-decoder SFT remains a research hypothesis, not a result
implied by T5Gemma2's UL2-adapted architecture. Real pretrained memory and repair
quality must be measured independently of data-preparation/scorer checks.
