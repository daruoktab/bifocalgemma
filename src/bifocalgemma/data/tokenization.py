"""D03: verified local tokenizer assets, official IT template, patch-only targets."""

import hashlib
import json
from pathlib import Path

from transformers.models.gemma.tokenization_gemma import GemmaTokenizer

from bifocalgemma.assets import load_asset_json
from bifocalgemma.config import CheckpointRef, ExperimentSpec
from bifocalgemma.contracts import RepairExample, TokenizedRepairExample


def format_encoder_context(example: RepairExample) -> str:
    rendered = "\n\n".join(
        f"### File: {file.path}\n{file.content}"
        for file in sorted(example.snapshot.files, key=lambda file: file.path)
    )
    return f"### Issue\n{example.issue}\n\n{rendered}"


def load_verified_tokenizer(checkpoint: CheckpointRef, tokenizer_cache: Path):
    """Cache root contains raw/google__gemma-4-...; verify bytes before loading.

    Revision is bound to checked-in asset hashes, not inferred from directory name.
    Network fetching and remote tokenizer code are never enabled.
    """
    manifest = load_asset_json("tokenizer_manifest.json")
    expected = manifest[checkpoint.model_id]
    if expected["revision"] != checkpoint.revision:
        raise ValueError("Tokenizer revision is not present in the audited manifest.")
    directory = tokenizer_cache / "raw" / checkpoint.model_id.replace("/", "__")
    for name, digest in expected["files"].items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing offline tokenizer asset: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Tokenizer hash mismatch: {path}")
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    if config.get("tokenizer_class") != "GemmaTokenizer":
        raise ValueError("Audited tokenizer class is not the supported native GemmaTokenizer.")
    # Instantiate the official class declared by the verified config. AutoTokenizer's
    # model-config dispatch can load tensor/generation modules even for static audits.
    tokenizer = GemmaTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    return tokenizer, expected


def decode_text(tokenizer, token_ids, *, skip_special_tokens=False) -> str:
    """Decode one token sequence without accepting a batch-shaped return value."""
    text = tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
    if not isinstance(text, str):
        raise ValueError("Single token sequence must decode to text.")
    return text


def tokenize_repair_example(
    example: RepairExample,
    spec: ExperimentSpec,
    tokenizer_cache: Path,
    *,
    max_encoder_tokens: int = 4096,
    max_decoder_tokens: int = 4096,
) -> TokenizedRepairExample:
    """Tokenize SFT examples; limits reject overflow instead of silently truncating.

    Real tokenizer IDs are not compatible with the native tiny vocabulary of 128.
    Target-free generation remains I01, not a dummy training target here.
    """
    if not 0 < max_encoder_tokens <= 131072 or not 0 < max_decoder_tokens <= 131072:
        raise ValueError("Token limits must be within 1..131072.")
    if not example.target_patch or not example.target_patch.strip():
        raise ValueError("SFT tokenization requires a nonempty target patch.")
    decoder, decoder_manifest = load_verified_tokenizer(spec.decoder, tokenizer_cache)
    context = format_encoder_context(example)
    encoder_ids: tuple[int, ...] = ()
    encoder_manifest = None
    if spec.encoder:
        encoder, encoder_manifest = load_verified_tokenizer(spec.encoder, tokenizer_cache)
        encoder_ids = tuple(encoder.encode(context, add_special_tokens=True))
        if encoder_ids[0] != encoder.bos_token_id:
            raise ValueError("Audited base tokenizer did not prepend BOS.")
        prompt = f"Return only a unified diff that fixes this issue.\n\n{example.issue}"
    else:
        prompt = f"Return only a unified diff that fixes the issue.\n\n{context}"
    messages = [{"role": "user", "content": prompt}]
    prefix = decoder.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    complete = decoder.apply_chat_template(
        messages + [{"role": "assistant", "content": example.target_patch}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(prefix, str) or not isinstance(complete, str):
        raise ValueError("Single-conversation IT template must return text.")
    if not complete.startswith(prefix):
        raise ValueError("IT template does not preserve the generation prefix.")
    prefix_ids = tuple(decoder.encode(prefix, add_special_tokens=False))
    # The canonical template appends a newline after the closing turn. Generation
    # stops at the turn token; do not train a token after the stopping boundary.
    full_ids = tuple(decoder.encode(complete.rstrip(), add_special_tokens=False))
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("BPE token crosses the prefix/target boundary.")
    target_ids = full_ids[len(prefix_ids) :]
    if not target_ids or target_ids[-1] != decoder.convert_tokens_to_ids("<turn|>"):
        raise ValueError("IT response must finish at the canonical closing turn token.")
    if prefix_ids[0] != decoder.bos_token_id:
        raise ValueError("IT template must supply BOS.")
    if len(encoder_ids) > max_encoder_tokens or len(full_ids) > max_decoder_tokens:
        raise ValueError("Token budget exceeded; no implicit truncation is performed.")
    return TokenizedRepairExample(
        example.instance_id,
        encoder_ids,
        prefix_ids,
        target_ids,
        metadata={
            "encoder_assets": encoder_manifest,
            "decoder_assets": decoder_manifest,
            "enable_thinking": False,
            "tools": False,
            "truncated": False,
            "context_policy": "all_files_sorted_by_path",
            "encoder_tokens": len(encoder_ids),
            "decoder_tokens": len(full_ids),
            "prefix_tokens": len(prefix_ids),
            "target_tokens": len(target_ids),
            "max_encoder_tokens": max_encoder_tokens,
            "max_decoder_tokens": max_decoder_tokens,
        },
    )
