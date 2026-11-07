"""BifocalGemma V3: the architecture and training stack used by the notebooks.

Module map:
- `constants`   repositories, step budget, ten-model matrix, eight benchmarks.
- `runtime`     allocator guards, `beta_N`, query-injected context, tokenizer patch.
- `bridge`      `SwiGLUDualGateBridge`.
- `attention`   decoupled full-rank cross-attention + inference cache.
- `objectives`  diff token weighting, streaming pointer loss, optimizer groups.
- `stage`       two-stage freezing, parameter groups, schedule.
- `curriculum`  anti-forgetting comprehension replay.
- `benchmark`   batched decode and the benchmark suite.
- `checkpoint`  save/resume/prune and Hugging Face synchronization.
- `matrix`      the ten-model entry point.

`tools/molab_matrix_support.py` is kept as a compatibility facade for the
pinned notebook runtime.
"""

from bifocalgemma.v3 import _environment as _environment  # noqa: F401  (side effects first)
from bifocalgemma.v3.attention import (
    Bf16FullLinear,
    FullCrossCachedAttention,
    FullCrossTiledAttention,
    _compute_per_row_cross_query,  # noqa: F401  (test-facing helper)
    configure_full_rank_cross_attention,
)
from bifocalgemma.v3.benchmark import (
    batched_greedy_decode,
    infer_benchmark_weight_tier,
    prepare_sampled_benchmark_collection,
    resolve_optimal_eval_batch_size,
    run_batched_context_benchmark,
)
from bifocalgemma.v3.bridge import SwiGLUDualGateBridge
from bifocalgemma.v3.checkpoint import (
    load_matrix_adapters_for_inference,
    prune_matrix_checkpoints,
    restore_matrix_checkpoint,
    restore_optimizer_state_by_name,
    save_matrix_checkpoint,
    synchronize_matrix_v3,
)
from bifocalgemma.v3.constants import (
    ACCUMULATION,
    ARCHIVE_HF_REPO,
    BATCH_SIZE,
    BENCHMARK_SAMPLE_SEED,
    BENCHMARK_TIER_CONFIG,
    BENCHMARKS,
    CENTRALIZED_HF_REPO,
    COLLECTION,
    COMPREHENSION_MIX_RATIO,
    DATASET_REPO,
    DATASET_REVISION,
    EFFECTIVE_BATCH,
    GATE_INIT_ATANH,
    GATE_INIT_TANH,
    HEAVY_BENCHMARK_SAMPLE_LIMIT,
    MATRIX_MODELS,
    MODEL_EVAL_FOOTPRINT,
    STAGE1_STEPS,
    STAGE2_STEPS,
    TOTAL_EXAMPLES,
    TOTAL_STEPS,
    WARMUP_RATIO,
)
from bifocalgemma.v3.curriculum import (
    COMPREHENSION_TASK_TYPES,
    build_anti_forgetting_curriculum,
    encode_comprehension_prefix,
    is_comprehension_example,
    synthesize_comprehension_examples,
)
from bifocalgemma.v3.matrix import (
    make_experiment_spec,
    run_matrix_benchmark,
    run_matrix_stage,
    run_matrix_training,
    run_matrix_training_and_benchmark,
)
from bifocalgemma.v3.objectives import (
    DEFAULT_DIFF_ADD_TOKEN_IDS,
    DEFAULT_DIFF_DEL_TOKEN_IDS,
    DEFAULT_DIFF_HDR_TOKEN_IDS,
    DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS,
    StreamingPointerLoss,
    build_bifocalgemma_v3_optimizer,
    chunked_differential_cross_entropy,
    compute_bifocalgemma_v3_loss,
    compute_diff_token_weights,
    compute_v3_step_loss,
    differential_diff_cross_entropy,
    extract_trivial_token_ids,
    is_cross_bridge_parameter,
    is_zero_weight_decay_bridge_parameter,
    resolve_diff_token_ids,
    streaming_pointer_auxiliary_loss,
)
from bifocalgemma.v3.runtime import (
    clamped_triton_kernel_options,
    critical_attention_beta,
    ensure_hardware_memory_guards,
    format_query_injected_context,
    patch_matrix_tokenization,
)
from bifocalgemma.v3.stage import configure_matrix_stage

synchronize_v2 = synchronize_matrix_v3

__all__ = [
    "ACCUMULATION",
    "ARCHIVE_HF_REPO",
    "BATCH_SIZE",
    "BENCHMARKS",
    "BENCHMARK_SAMPLE_SEED",
    "BENCHMARK_TIER_CONFIG",
    "CENTRALIZED_HF_REPO",
    "COLLECTION",
    "COMPREHENSION_MIX_RATIO",
    "COMPREHENSION_TASK_TYPES",
    "DEFAULT_DIFF_ADD_TOKEN_IDS",
    "DEFAULT_DIFF_DEL_TOKEN_IDS",
    "DEFAULT_DIFF_HDR_TOKEN_IDS",
    "DEFAULT_GEMMA_TRIVIAL_TOKEN_IDS",
    "DATASET_REPO",
    "DATASET_REVISION",
    "EFFECTIVE_BATCH",
    "FullCrossCachedAttention",
    "FullCrossTiledAttention",
    "GATE_INIT_ATANH",
    "GATE_INIT_TANH",
    "HEAVY_BENCHMARK_SAMPLE_LIMIT",
    "MATRIX_MODELS",
    "MODEL_EVAL_FOOTPRINT",
    "STAGE1_STEPS",
    "STAGE2_STEPS",
    "StreamingPointerLoss",
    "SwiGLUDualGateBridge",
    "TOTAL_EXAMPLES",
    "TOTAL_STEPS",
    "WARMUP_RATIO",
    "Bf16FullLinear",
    "batched_greedy_decode",
    "build_anti_forgetting_curriculum",
    "build_bifocalgemma_v3_optimizer",
    "chunked_differential_cross_entropy",
    "clamped_triton_kernel_options",
    "compute_diff_token_weights",
    "compute_bifocalgemma_v3_loss",
    "compute_v3_step_loss",
    "configure_full_rank_cross_attention",
    "configure_matrix_stage",
    "critical_attention_beta",
    "differential_diff_cross_entropy",
    "encode_comprehension_prefix",
    "ensure_hardware_memory_guards",
    "extract_trivial_token_ids",
    "format_query_injected_context",
    "infer_benchmark_weight_tier",
    "is_comprehension_example",
    "is_cross_bridge_parameter",
    "is_zero_weight_decay_bridge_parameter",
    "load_matrix_adapters_for_inference",
    "make_experiment_spec",
    "patch_matrix_tokenization",
    "prepare_sampled_benchmark_collection",
    "prune_matrix_checkpoints",
    "resolve_diff_token_ids",
    "resolve_optimal_eval_batch_size",
    "restore_matrix_checkpoint",
    "restore_optimizer_state_by_name",
    "run_batched_context_benchmark",
    "run_matrix_benchmark",
    "run_matrix_stage",
    "run_matrix_training",
    "run_matrix_training_and_benchmark",
    "save_matrix_checkpoint",
    "streaming_pointer_auxiliary_loss",
    "synchronize_matrix_v3",
    "synchronize_v2",
    "synthesize_comprehension_examples",
]
