"""D03: pinned base/IT formatting. See docs/DATA_PIPELINE.md."""

from pathlib import Path

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.contracts import RepairExample, TokenizedRepairExample
from bifocalgemma.scaffold import FeatureNotReadyError


def format_encoder_context(example: RepairExample) -> str:
    """Only issue and pre-fix files; never append target_patch or evaluator artifacts."""
    files = "\n\n".join(
        f"### File: {file.path}\n{file.content}"
        for file in sorted(example.snapshot.files, key=lambda file: file.path)
    )
    return f"### Issue\n{example.issue}\n\n{files}"


def tokenize_repair_example(
    example: RepairExample, spec: ExperimentSpec, tokenizer_cache: Path
) -> TokenizedRepairExample:
    """Future: base BOS once, IT Jinja prefix, target patch + correct closing token.

    Must pin tokenizers, preserve original offsets for AST, truncate by a recorded
    context policy, and keep gold patch/test artifacts out of encoder/prefix input.
    No remote calls are made by this placeholder.
    """
    raise FeatureNotReadyError(
        "D03", "Implement pinned tokenizer loading and prefix/target formatting."
    )
