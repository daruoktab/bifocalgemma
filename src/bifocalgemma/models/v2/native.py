"""Native Gemma 4 tiny integration: random weights, eager attention, no temporal cache.

The upstream text stack owns PLE, residuals, MLP and RoPE. Only decoder attention
is replaced. Same-forward shared KV includes both self and cross branches.
"""

import json
from importlib.metadata import version
from importlib.resources import files
from typing import Callable, cast

import torch
import torch.nn as nn
from transformers import Gemma4TextConfig, Gemma4TextModel
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv,
)

from bifocalgemma.config import ExperimentSpec, ModelSize
from bifocalgemma.contracts import ModelOutput, Seq2SeqBatch, TokenizedRepairExample
from bifocalgemma.data.collator import RepairCollator
from bifocalgemma.models.v2.alignment import NormalizedAlignment
from bifocalgemma.models.v2.merged_attention import build_joint_visibility
from bifocalgemma.training.objectives import chunked_linear_cross_entropy, patch_cross_entropy


def native_config(size: ModelSize, *, tiny: bool = True, encoder: bool = False):
    """Full dimensions are for meta inspection only; factory always uses tiny."""
    if version("transformers") not in {"5.19.0", "5.20.2", "5.22.1"}:
        raise RuntimeError(
            "Native integration requires the validated transformers 5.19.0/5.20.2 interface."
        )
    asset = files("bifocalgemma.assets").joinpath(f"gemma4_{size.lower()}.json")
    values = json.loads(asset.read_text(encoding="utf-8"))["text_config"]
    if tiny:
        width = 32 if size == "E2B" else 48
        values.update(
            vocab_size=128,
            vocab_size_per_layer_input=128,
            hidden_size=width,
            intermediate_size=4 * width,
            hidden_size_per_layer_input=4,
            head_dim=8,
            global_head_dim=16,
        )
    values.update(use_cache=False, use_bidirectional_attention="all" if encoder else None)
    config = Gemma4TextConfig.from_dict(values)
    config._attn_implementation = "eager"
    return config


class NativeMergedAttention(Gemma4TextAttention):
    """Preserve native parameter names and KV owners; concatenate after self RoPE."""

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        past_key_values=None,
        *,
        encoder_hidden_states=None,
        encoder_valid=None,
        decoder_valid=None,
        **kwargs,
    ):
        if past_key_values is not None:
            raise ValueError("Native tiny merged attention does not support temporal cache.")
        if encoder_hidden_states is None or encoder_valid is None or decoder_valid is None:
            raise ValueError(
                "Merged attention requires explicit encoder states and validity masks."
            )
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        if self.is_kv_shared_layer:
            key, value = shared_kv_states[self.layer_type]
        else:
            assert self.v_proj is not None
            key = self.k_norm(self.k_proj(hidden_states).view(shape))
            key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
            cross_shape = (
                batch,
                encoder_hidden_states.shape[1],
                self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                self.head_dim,
            )
            cross_key = self.k_norm(self.k_proj(encoder_hidden_states).view(cross_shape))
            cross_value = self.v_norm(self.v_proj(encoder_hidden_states).view(cross_shape))
            key = torch.cat((key, cross_key.transpose(1, 2)), dim=2)
            value = torch.cat((value, cross_value.transpose(1, 2)), dim=2)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (key, value)
        allowed = build_joint_visibility(encoder_valid, decoder_valid, self.sliding_window)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        probabilities = probabilities.masked_fill(~allowed, 0)
        probabilities = nn.functional.dropout(
            probabilities, p=float(self.attention_dropout or 0.0), training=self.training
        )
        result = (probabilities @ value).transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(result), None


class NativeTinyModel(nn.Module):
    """Native text stacks; tiny by default, full dimensions only with explicit opt-in."""

    def __init__(self, spec: ExperimentSpec, *, tiny: bool = True):
        super().__init__()
        self.spec = spec
        self.loss_chunk_size = None
        self._stage_callback: Callable | None = None
        self.decoder = Gemma4TextModel(native_config(spec.decoder.size, tiny=tiny))
        self.encoder = (
            Gemma4TextModel(native_config(spec.encoder.size, tiny=tiny, encoder=True))
            if spec.encoder
            else None
        )
        width = self.decoder.config.hidden_size
        self.encoder_projection = None
        if self.encoder is not None:
            enc_width = self.encoder.config.hidden_size
            self.encoder_projection = (
                NormalizedAlignment(enc_width, width)
                if spec.architecture_version == "normalized_merged_v2"
                else nn.Identity()
                if enc_width == width
                else nn.Linear(enc_width, width, bias=False)
            )
            for index, layer in enumerate(self.decoder.layers):
                layer = cast(Gemma4TextDecoderLayer, layer)
                attention = NativeMergedAttention(self.decoder.config, index)
                attention.load_state_dict(layer.self_attn.state_dict(), strict=True)
                layer.self_attn = attention
        self.lm_head = nn.Linear(width, self.decoder.config.vocab_size, bias=False)
        self._tie_words()

    def _tie_words(self):
        self.lm_head.weight = self.decoder.embed_tokens.weight
        if self.encoder is not None and self.spec.tie_encoder_decoder_words:
            self.encoder.embed_tokens.weight = self.decoder.embed_tokens.weight

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._tie_words()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._tie_words()
        return result

    @staticmethod
    def _positions(mask):
        return (mask.long().cumsum(-1) - 1).clamp_min(0)

    def encode_context(self, input_ids, attention_mask):
        if self.encoder is None or self.encoder_projection is None:
            raise ValueError("Decoder-only baseline has no encoder.")
        if input_ids.shape[1] == 0:
            return self.decoder.embed_tokens.weight.new_empty(
                input_ids.shape[0], 0, self.decoder.config.hidden_size
            )
        if self._stage_callback is not None:
            self._stage_callback({"stage": "encoder_forward"})
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=(
                {"full_attention": None, "sliding_attention": None}
                if hasattr(self, "memory_policy")
                and self.spec.architecture_version == "normalized_merged_v2"
                else attention_mask
            ),
            self_valid=attention_mask,
            position_ids=self._positions(attention_mask),
            use_cache=False,
        ).last_hidden_state
        # No padding states enter K/V, including batches with wholly masked context.
        return self.encoder_projection(hidden).masked_fill(~attention_mask.bool()[..., None], 0)

    def _decode_hidden(
        self,
        input_ids,
        decoder_mask,
        encoder_states=None,
        encoder_mask=None,
        *,
        last_token_only=False,
    ):
        kwargs = {}
        if self.encoder is not None:
            if encoder_states is None:
                encoder_states = self.decoder.embed_tokens.weight.new_empty(
                    input_ids.shape[0], 0, self.decoder.config.hidden_size
                )
                encoder_mask = decoder_mask.new_zeros((input_ids.shape[0], 0))
            if encoder_mask is None or encoder_mask.shape != encoder_states.shape[:2]:
                raise ValueError("Encoder mask must match encoder states.")
            kwargs = dict(
                encoder_hidden_states=encoder_states,
                encoder_valid=encoder_mask,
                decoder_valid=decoder_mask,
            )
        elif encoder_states is not None or encoder_mask is not None:
            raise ValueError("Decoder-only baseline cannot receive encoder states.")
        if self._stage_callback is not None:
            self._stage_callback({"stage": "decoder_forward"})
        hidden = self.decoder(
            input_ids=input_ids,
            attention_mask=(
                {"full_attention": None, "sliding_attention": None}
                if hasattr(self, "memory_policy")
                and self.spec.architecture_version == "normalized_merged_v2"
                else decoder_mask
            ),
            self_valid=decoder_mask,
            position_ids=self._positions(decoder_mask),
            use_cache=False,
            **kwargs,
        ).last_hidden_state
        if last_token_only:
            hidden = hidden[:, -1:]
        return hidden

    def decode_logits(
        self,
        input_ids,
        decoder_mask,
        encoder_states=None,
        encoder_mask=None,
        *,
        last_token_only=False,
    ):
        hidden = self._decode_hidden(
            input_ids, decoder_mask, encoder_states, encoder_mask, last_token_only=last_token_only
        )
        logits = self.lm_head(hidden)
        cap = self.decoder.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def decode_next_logits(self, input_ids, decoder_mask, encoder_states=None, encoder_mask=None):
        return self.decode_logits(
            input_ids, decoder_mask, encoder_states, encoder_mask, last_token_only=True
        )

    def forward(self, batch: Seq2SeqBatch, *, return_logits=True) -> ModelOutput:  # ty: ignore[invalid-mutable-override]  # nn.Module.forward is annotated Callable.
        if self.encoder is None:
            if batch.encoder_input_ids.shape[1] != 0:
                raise ValueError("Baseline requires an empty encoder sequence.")
            hidden, mask = None, None
        else:
            hidden = self.encode_context(batch.encoder_input_ids, batch.encoder_attention_mask)
            mask = batch.encoder_attention_mask
        if not return_logits and self.loss_chunk_size is not None:
            decoder_hidden = self._decode_hidden(
                batch.decoder_input_ids, batch.decoder_attention_mask, hidden, mask
            )
            if self._stage_callback is not None:
                self._stage_callback({"stage": "loss_forward"})
            loss = chunked_linear_cross_entropy(
                decoder_hidden,
                self.lm_head.weight,
                batch.labels,
                chunk_size=self.loss_chunk_size,
                softcap=self.decoder.config.final_logit_softcapping,
            )
            return ModelOutput(None, loss)
        logits = self.decode_logits(
            batch.decoder_input_ids, batch.decoder_attention_mask, hidden, mask
        )
        return ModelOutput(logits, patch_cross_entropy(logits, batch.labels))

    def forward_loss(self, batch: Seq2SeqBatch) -> ModelOutput:
        return self.forward(batch, return_logits=False)


def build_native_tiny_model(spec: ExperimentSpec) -> NativeTinyModel:
    return NativeTinyModel(spec)


def run_native_smoke(spec: ExperimentSpec, device: str = "cpu") -> dict:
    torch.manual_seed(17)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model = build_native_tiny_model(spec).to(device)
    contexts = [(2, 12, 13, 14), (2, 15)] if spec.encoder else [(), ()]
    batch = RepairCollator()(
        [
            TokenizedRepairExample("synthetic-1", contexts[0], (2, 105, 7), (18, 19, 106)),
            TokenizedRepairExample("synthetic-2", contexts[1], (2, 105, 8), (20, 106)),
        ]
    ).to(device)
    output = model(batch)
    assert output.loss is not None
    output.loss.backward()
    parameters = list(model.parameters())
    gradients = [p.grad for p in parameters if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    nonzero = sum(bool(g.abs().sum() > 0) for g in gradients)
    if not torch.isfinite(output.logits).all() or not finite or not nonzero:
        raise RuntimeError("Native smoke produced invalid logits or gradients.")
    return dict(
        experiment=spec.name,
        mode="native_tiny_random",
        device=device,
        encoder_width=model.encoder.config.hidden_size if model.encoder else None,
        decoder_width=model.decoder.config.hidden_size,
        encoder_layers=len(model.encoder.layers) if model.encoder else 0,
        decoder_layers=len(model.decoder.layers),
        logits_shape=list(output.logits.shape),
        parameters=sum(p.numel() for p in parameters),
        synthetic_loss=output.loss.item(),
        gradients_finite=finite,
        parameters_with_nonzero_gradient=nonzero,
        parameter_bytes=sum(p.numel() * p.element_size() for p in parameters),
        gradient_bytes=sum(g.numel() * g.element_size() for g in gradients),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None,
        word_embeddings_shared=(
            model.encoder is not None
            and model.encoder.embed_tokens.weight is model.decoder.embed_tokens.weight
        ),
        decoder_lm_head_shared=model.lm_head.weight is model.decoder.embed_tokens.weight,
        pretrained_weights_loaded=False,
        benchmark_executed=False,
    )


def build_native_model(spec: ExperimentSpec, *, tiny=True, device="cpu", dtype=torch.float32):
    """Construct directly on destination; preserve native nonpersistent buffers."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            return NativeTinyModel(spec, tiny=tiny)
    finally:
        torch.set_default_dtype(previous)
