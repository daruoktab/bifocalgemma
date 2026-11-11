"""I01 short greedy reference generation; no temporal KV cache or fake scores."""

from dataclasses import dataclass
from typing import Any

import torch

from bifocalgemma.config import ExperimentSpec
from bifocalgemma.contracts import RepairExample, RepairPrediction
from bifocalgemma.data.tokenization import (
    decode_text,
    format_encoder_context,
    load_verified_tokenizer,
)
from bifocalgemma.scaffold import FeatureNotReadyError


@dataclass(frozen=True)
class GreedyGeneration:
    token_ids: tuple[int, ...]
    stop_reason: str
    encoder_tokens: int
    decoder_prefix_tokens: int


def greedy_decode(
    model,
    encoder_ids: tuple[int, ...],
    prefix_ids: tuple[int, ...],
    *,
    max_new_tokens: int,
    eos_ids: tuple[int, ...] = (1, 106, 50),
    max_total_decoder_tokens: int = 4096,
    progress=None,
    use_cache=False,
) -> GreedyGeneration:
    """Encode once and recompute causal decoder per step; suitable for correctness pilots."""
    if (
        max_new_tokens < 1
        or not prefix_ids
        or len(prefix_ids) + max_new_tokens > max_total_decoder_tokens
    ):
        raise ValueError("generation_budget_exceeded")
    vocabulary = model.decoder.config.vocab_size
    if min((*encoder_ids, *prefix_ids)) < 0 or max((*encoder_ids, *prefix_ids)) >= vocabulary:
        raise ValueError("tokenizer_ids_do_not_fit_model_vocabulary")
    if model.encoder is None and encoder_ids:
        raise ValueError("baseline_cannot_receive_encoder_tokens")
    device = next(model.parameters()).device
    if device.type == "meta":
        raise ValueError("generation_requires_materialized_weights")
    previous_mode = model.training
    model.eval()
    output = []
    reason = "length_limit"
    try:
        with torch.inference_mode():
            if progress:
                progress({"stage": "benchmark_encoding", "generated_tokens": 0})
            enc = torch.tensor([encoder_ids], dtype=torch.long, device=device)
            enc_mask: torch.Tensor | None = torch.ones_like(enc, dtype=torch.bool)
            states = model.encode_context(enc, enc_mask) if model.encoder is not None else None
            if model.encoder is None:
                enc_mask = None
            sequence = list(prefix_ids)
            from bifocalgemma.models.inference_cache import InferenceCache, cached_logits

            cache = InferenceCache() if use_cache else None
            if progress:
                progress({"stage": "benchmark_decoding", "generated_tokens": 0})
            for _ in range(max_new_tokens):
                ids = torch.tensor(
                    [sequence if cache is None or cache.offset == 0 else sequence[-1:]],
                    dtype=torch.long,
                    device=device,
                )
                mask = torch.ones_like(ids, dtype=torch.bool)
                decode = getattr(model, "decode_next_logits", model.decode_logits)
                logits = (
                    cached_logits(model, ids, cache, states, enc_mask)
                    if cache is not None
                    else decode(ids, mask, states, enc_mask)
                )[:, -1]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite_generation_logits")
                token = int(logits.argmax(-1).item())
                output.append(token)
                if progress:
                    progress({"stage": "benchmark_decoding", "generated_tokens": len(output)})
                if token in eos_ids:
                    reason = "tool_boundary" if token == 50 else "eos"
                    break
                sequence.append(token)
    finally:
        model.train(previous_mode)
    return GreedyGeneration(tuple(output), reason, len(encoder_ids), len(prefix_ids))


def _safe_apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def repair_prompt_ids(
    spec, example, tokenizer_cache, *, context: str | None = None, max_encoder_tokens=4096
):
    """Never reads target_patch; input may carry a separately audited retrieval context."""
    decoder, _ = load_verified_tokenizer(spec.decoder, tokenizer_cache)
    rendered = format_encoder_context(example) if context is None else context
    encoder_ids = ()
    if spec.encoder:
        encoder, _ = load_verified_tokenizer(spec.encoder, tokenizer_cache)
        encoder_ids = tuple(encoder.encode(rendered, add_special_tokens=True))
        if len(encoder_ids) > max_encoder_tokens:
            raise ValueError("encoder_generation_budget_exceeded")
        prompt = (
            "Return only a unified diff fulfilling the instruction.\n"
            if context is not None
            else "Return only a unified diff that fixes this issue.\n"
        ) + example.issue
    else:
        prompt = (
            "Return only a unified diff fulfilling the instruction.\n\n"
            if context is not None
            else "Return only a unified diff that fixes the issue.\n\n"
        ) + rendered
    text = _safe_apply_chat_template(
        decoder,
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix = tuple(decoder.encode(text, add_special_tokens=False))
    if not prefix or prefix[0] != decoder.bos_token_id:
        raise ValueError("generation_bos_policy")
    return encoder_ids, prefix, decoder


def generate_patch(
    spec: ExperimentSpec,
    example: RepairExample,
    max_new_tokens: int,
    *,
    model=None,
    tokenizer_cache=None,
    context: str | None = None,
    max_encoder_tokens: int = 4096,
    max_total_decoder_tokens: int = 8192,
) -> RepairPrediction:
    if model is None or tokenizer_cache is None:
        raise FeatureNotReadyError(
            "I01", "Supply a loaded native model and verified tokenizer cache."
        )
    if model.spec != spec:
        raise ValueError("generation_model_spec_mismatch")
    enc, prefix, decoder = repair_prompt_ids(
        spec,
        example,
        tokenizer_cache,
        context=context,
        max_encoder_tokens=max_encoder_tokens,
    )
    generated = greedy_decode(
        model,
        enc,
        prefix,
        max_new_tokens=max_new_tokens,
        max_total_decoder_tokens=max_total_decoder_tokens,
    )
    ids = generated.token_ids
    if ids and ids[-1] in (1, 106, 50):
        ids = ids[:-1]
    patch = decode_text(decoder, ids, skip_special_tokens=False)
    return RepairPrediction(example.instance_id, spec.name, patch)


def generate_diagnostic_text(
    spec,
    model,
    instruction,
    context,
    tokenizer_cache,
    *,
    max_new_tokens=1024,
    max_encoder_tokens=4096,
    max_total_decoder_tokens=8192,
):
    decoder, _ = load_verified_tokenizer(spec.decoder, tokenizer_cache)
    enc = ()
    if spec.encoder:
        encoder, _ = load_verified_tokenizer(spec.encoder, tokenizer_cache)
        enc = tuple(encoder.encode(context, add_special_tokens=True))
        if len(enc) > max_encoder_tokens:
            raise ValueError("diagnostic_input_overflow")
        prompt = instruction
    else:
        prompt = instruction + "\n\n" + context
    text = _safe_apply_chat_template(
        decoder,
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix = tuple(decoder.encode(text, add_special_tokens=False))
    generated = greedy_decode(
        model,
        enc,
        prefix,
        max_new_tokens=max_new_tokens,
        max_total_decoder_tokens=max_total_decoder_tokens,
    )
    ids = (
        generated.token_ids[:-1]
        if generated.token_ids and generated.token_ids[-1] in (1, 106, 50)
        else generated.token_ids
    )
    return decode_text(decoder, ids, skip_special_tokens=False), generated.stop_reason
