"""Named construction blocks and explicit integration boundaries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildBlock:
    id: str
    status: str
    source: str
    purpose: str


BUILD_BLOCKS = (
    BuildBlock(
        "C01", "implemented", "config.py", "Six-model specification and checkpoint provenance"
    ),
    BuildBlock(
        "D01", "implemented", "contracts.py", "Repository, example, batch and prediction contracts"
    ),
    BuildBlock("D02", "implemented", "data/collator.py", "Pretokenized teacher-forcing batches"),
    BuildBlock(
        "D03", "implemented", "data/tokenization.py", "Pinned base/IT tokenizers and formatting"
    ),
    BuildBlock(
        "D04", "placeholder", "data/adapters.py", "Training dataset adapters and overlap audit"
    ),
    BuildBlock(
        "D05",
        "placeholder",
        "data/ast.py",
        "Tree-sitter features aligned to original token offsets",
    ),
    BuildBlock(
        "D06", "implemented", "data/swesmith.py", "Local SWE-smith patch-direction contract"
    ),
    BuildBlock("M07", "native_tiny", "models/native.py", "Native random-weight merged integration"),
    BuildBlock(
        "M01",
        "reference",
        "models/merged_attention.py",
        "Joint softmax, GQA and asymmetric projection",
    ),
    BuildBlock("M02", "implemented", "models/embeddings.py", "Symmetric word parameter sharing"),
    BuildBlock("M03", "reference", "models/reference.py", "Tiny offline assembly and backward"),
    BuildBlock(
        "M04", "placeholder", "models/initialization.py", "Base/IT weight mapping into Gemma layers"
    ),
    BuildBlock(
        "M05", "placeholder", "models/model.py", "Full Gemma encoder/merged decoder integration"
    ),
    BuildBlock(
        "M06", "placeholder", "models/cache.py", "Production self/cross KV and repository cache"
    ),
    BuildBlock(
        "T01", "implemented", "training/objectives.py", "Patch-only supervised cross entropy"
    ),
    BuildBlock(
        "T02", "placeholder", "training/trainer.py", "Profiled PEFT optimizer and training loop"
    ),
    BuildBlock(
        "T03", "placeholder", "training/profiling.py", "Actual memory and latency measurements"
    ),
    BuildBlock(
        "T04", "optional_placeholder", "training/distillation.py", "Optional later teacher losses"
    ),
    BuildBlock(
        "I01",
        "reference",
        "inference/generation.py",
        "Target-free greedy native decoding; temporal cache pending",
    ),
    BuildBlock(
        "D07",
        "implemented",
        "data/repository.py",
        "Static pinned-source snapshot exports and quarantine",
    ),
    BuildBlock("D08", "implemented", "data/context.py", "Target-free lexical context windows"),
    BuildBlock(
        "E03",
        "reference",
        "evaluation/native_python.py",
        "Opt-in native Python scorer; real benchmark gold validation pending",
    ),
    BuildBlock(
        "T05",
        "implemented",
        "training/repository_sft.py",
        "Matched-cohort repository SFT including decoder-only baselines",
    ),
    BuildBlock(
        "A01",
        "optional_placeholder",
        "agent/loop.py",
        "Tool parsing and bounded developer-agent loop",
    ),
    BuildBlock(
        "E01", "optional_placeholder", "evaluation/harness.py", "Real Docker repair evaluation"
    ),
    BuildBlock(
        "E02",
        "implemented",
        "evaluation/predictions.py",
        "SWE-bench prediction JSONL serialization",
    ),
    BuildBlock("S01", "placeholder", "checkpointing.py", "Alias-safe checkpoint save/load"),
)
BLOCKS_BY_ID = {block.id: block for block in BUILD_BLOCKS}


class FeatureNotReadyError(NotImplementedError):
    """An intentional, documented boundary, not a fabricated successful run."""

    def __init__(self, block_id: str, detail: str = ""):
        block = BLOCKS_BY_ID[block_id]
        self.block_id = block_id
        self.documentation = f"docs/BUILD_GUIDE.md#{block_id.lower()}"
        super().__init__(
            f"{block_id}: {block.purpose} is not integrated. {detail} "
            f"Next steps: {self.documentation}"
        )
