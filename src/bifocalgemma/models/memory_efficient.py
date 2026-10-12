"""Bounded-memory exact attention: query tiles, FP32 softmax, joint normalization.

No quantization, context truncation, independent cross softmax or detached encoder.
Eager remains the reference. Recompute trades throughput for activation memory.
"""

import copy

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, apply_rotary_pos_emb

DEFAULT_MEMORY_POLICY = {
    "version": "tiled_recompute_v1",
    "attention_backend": "tiled_eager_fp32",
    "query_chunk_size": 128,
    "loss_chunk_size": 128,
    "gradient_checkpointing": True,
    "ple_offload": False,
    "kv_gradient_accumulation": "float32",
}


def joint_visibility_tile(encoder_valid, decoder_valid, start, stop, window):
    positions = torch.arange(decoder_valid.shape[1], device=decoder_valid.device)
    queries = positions[start:stop, None]
    visible = positions[None, :] <= queries
    if window is not None:
        visible &= positions[None, :] > queries - window
    self_visible = visible[None] & decoder_valid[:, None, :].bool()
    cross_visible = encoder_valid[:, None, :].bool().expand(-1, stop - start, -1)
    return (
        torch.cat((self_visible, cross_visible), -1)[:, None]
        & decoder_valid[:, None, start:stop, None].bool()
    )


def attention_tile(query, key, value, mask, scaling, dropout, boolean_mask):
    batch, heads, length, width = query.shape
    kv_heads = key.shape[1]
    if heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads.")
    groups = heads // kv_heads
    # Preserve native head/group layout. Backend matmul workspace is transient.
    key = key.to(query.dtype)
    value = value.to(query.dtype)
    grouped_query = query.reshape(batch, kv_heads, groups, length, width)
    scores = torch.matmul(grouped_query, key[:, :, None].transpose(-1, -2))
    scores = scores.reshape(batch, heads, length, key.shape[2]) * scaling
    if mask is not None:
        if boolean_mask:
            scores = scores.masked_fill(~mask, float("-inf"))
            scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        else:
            scores = scores + mask
    probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
    if mask is not None and boolean_mask:
        probabilities = probabilities.masked_fill(~mask, 0)
    probabilities = torch.nn.functional.dropout(probabilities, p=dropout, training=dropout > 0)
    grouped_probabilities = probabilities.reshape(batch, kv_heads, groups, length, key.shape[2])
    attended = torch.matmul(grouped_probabilities, value[:, :, None])
    return attended.reshape(batch, heads, length, value.shape[-1])


class _TiledAttention(torch.autograd.Function):
    """First-order tiled eager attention; FP32 KV accumulation before one cast.

    Forward keeps native score/probability dtypes. Backward recomputes softmax
    tiles; it does not sum BF16 partial KV gradients from separate graph nodes.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attention_mask,
        encoder_valid,
        decoder_valid,
        chunk_size,
        scaling,
        window,
    ):
        ctx.save_for_backward(query, key, value)
        ctx.attention_mask, ctx.encoder_valid, ctx.decoder_valid = (
            attention_mask,
            encoder_valid,
            decoder_valid,
        )
        ctx.chunk_size, ctx.scaling, ctx.window = chunk_size, scaling, window
        pieces = []
        for start in range(0, query.shape[2], chunk_size):
            stop = min(start + chunk_size, query.shape[2])
            mask = (
                joint_visibility_tile(encoder_valid, decoder_valid, start, stop, window)
                if encoder_valid is not None
                else attention_mask[..., start:stop, :]
                if attention_mask is not None
                else None
            )
            pieces.append(
                attention_tile(
                    query[:, :, start:stop],
                    key,
                    value,
                    mask,
                    scaling,
                    0.0,
                    encoder_valid is not None,
                )
            )
        return torch.cat(pieces, dim=2)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        query, key, value = ctx.saved_tensors
        batch, heads, length, width = query.shape
        groups = heads // key.shape[1]
        # Match native repeat_kv head order, including its final head-group reduction.
        repeated_key = key.repeat_interleave(groups, dim=1)
        repeated_value = value.repeat_interleave(groups, dim=1)
        accumulation_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        dk = torch.zeros(repeated_key.shape, dtype=accumulation_dtype, device=key.device)
        dv = torch.zeros(repeated_value.shape, dtype=accumulation_dtype, device=value.device)
        dq = torch.zeros_like(query)
        merged = ctx.encoder_valid is not None
        for start in range(0, length, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, length)
            q = query[:, :, start:stop]
            grad = grad_output[:, :, start:stop]
            mask = (
                joint_visibility_tile(ctx.encoder_valid, ctx.decoder_valid, start, stop, ctx.window)
                if merged
                else ctx.attention_mask[..., start:stop, :]
                if ctx.attention_mask is not None
                else None
            )
            scores = torch.matmul(q, repeated_key.transpose(-1, -2)) * ctx.scaling
            if mask is not None:
                if merged:
                    scores = scores.masked_fill(~mask, float("-inf"))
                    scores = torch.where(
                        mask.any(-1, keepdim=True), scores, torch.zeros_like(scores)
                    )
                else:
                    scores = scores + mask
            raw_probabilities = scores.softmax(-1, dtype=torch.float32)
            probabilities = raw_probabilities.to(query.dtype)
            if mask is not None and merged:
                probabilities = probabilities.masked_fill(~mask, 0)
            # dP has native matmul dtype; softmax backward uses native FP32 precision.
            dp = torch.matmul(grad, repeated_value.transpose(-1, -2))
            if mask is not None and merged:
                dp = dp.masked_fill(~mask, 0)
            ds = torch.ops.aten._softmax_backward_data(
                dp.float(), raw_probabilities, -1, torch.float32
            )
            ds = ds.to(query.dtype)
            if mask is not None and merged:
                ds = ds.masked_fill(~mask, 0)
            ds = ds * ctx.scaling
            dq[:, :, start:stop] = torch.matmul(ds, repeated_key)
            # Sum partial query reductions in FP32 instead of repeated BF16 adds.
            dk.add_(
                torch.matmul(ds.to(accumulation_dtype).transpose(-1, -2), q.to(accumulation_dtype))
            )
            dv.add_(
                torch.matmul(
                    probabilities.to(accumulation_dtype).transpose(-1, -2),
                    grad.to(accumulation_dtype),
                )
            )
        dk = dk.to(key.dtype).reshape(batch, key.shape[1], groups, key.shape[2], width).sum(dim=2)
        dv = (
            dv.to(value.dtype)
            .reshape(batch, value.shape[1], groups, value.shape[2], value.shape[-1])
            .sum(dim=2)
        )
        return dq, dk, dv, None, None, None, None, None, None


def tiled_attention(
    query,
    key,
    value,
    *,
    chunk_size,
    scaling=1.0,
    dropout=0.0,
    attention_mask=None,
    encoder_valid=None,
    decoder_valid=None,
    window=None,
):
    if chunk_size < 1 or query.shape[2] < 1:
        raise ValueError("Attention chunk size and query length must be positive.")
    if dropout != 0.0:
        raise ValueError("Conservative tiled attention is validated for native Gemma dropout=0.")
    if encoder_valid is not None and decoder_valid is None:
        raise ValueError("Merged attention requires decoder validity.")
    return _TiledAttention.apply(
        query, key, value, attention_mask, encoder_valid, decoder_valid, chunk_size, scaling, window
    )


class TiledEncoderAttention(Gemma4TextAttention):
    """Native self attention, including RoPE/KV owners, with bounded score tiles."""

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            raise ValueError("Tiled training attention does not support temporal cache.")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        if self.is_kv_shared_layer:
            key, value = shared_kv_states[self.layer_type]
            key, value = key.to(query.device), value.to(query.device)
        else:
            key = self.k_proj(hidden_states).view(shape)
            value = self.v_proj(hidden_states).view(shape) if self.v_proj is not None else key
            key = self.k_norm(key)
            key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value = self.v_norm(value).transpose(1, 2)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (key, value)
        result = tiled_attention(
            query,
            key,
            value,
            chunk_size=self.query_chunk_size,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            attention_mask=attention_mask,
        )
        result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(result), None


class TiledMergedAttention(TiledEncoderAttention):
    """A single joint softmax over causal/windowed self keys and all encoder keys."""

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
            raise ValueError("Tiled training attention does not support temporal cache.")
        if encoder_hidden_states is None or encoder_valid is None or decoder_valid is None:
            raise ValueError("Merged attention requires encoder states and validity masks.")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        if self.is_kv_shared_layer:
            key, value = shared_kv_states[self.layer_type]
        else:
            if self.v_proj is None:
                raise ValueError("Merged branch requires separate value projection.")
            key = self.k_norm(self.k_proj(hidden_states).view(shape))
            key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value = self.v_norm(self.v_proj(hidden_states).view(shape)).transpose(1, 2)
            cross_shape = (
                batch,
                encoder_hidden_states.shape[1],
                self.config.per_layer_config[self.layer_idx].num_key_value_heads,
                self.head_dim,
            )
            cross_key = self.k_norm(self.k_proj(encoder_hidden_states).view(cross_shape)).transpose(
                1, 2
            )
            cross_value = self.v_norm(
                self.v_proj(encoder_hidden_states).view(cross_shape)
            ).transpose(1, 2)
            key = torch.cat((key, cross_key), dim=2)
            value = torch.cat((value, cross_value), dim=2)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = (key, value)
        result = tiled_attention(
            query,
            key,
            value,
            chunk_size=self.query_chunk_size,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            encoder_valid=encoder_valid,
            decoder_valid=decoder_valid,
            window=self.sliding_window,
        )
        result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(result), None


def configure_memory_efficient_training(model, policy=None):
    options = {**DEFAULT_MEMORY_POLICY, **(policy or {})}
    if set(options) != set(DEFAULT_MEMORY_POLICY) or options["version"] != "tiled_recompute_v1":
        raise ValueError("Unknown memory policy.")
    if options["attention_backend"] != "tiled_eager_fp32" or options["ple_offload"] is not False:
        raise ValueError(
            "Only conservative tiled FP32 attention with GPU-resident PLE is supported."
        )
    if options["kv_gradient_accumulation"] != "float32":
        raise ValueError("KV gradients must accumulate in FP32 for tiled training.")
    for name in ("query_chunk_size", "loss_chunk_size"):
        value = options[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Invalid {name}.")
    if type(options["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean.")
    for role in ("encoder", "decoder"):
        stack = getattr(model, role, None)
        if stack is None:
            continue
        for layer in stack.layers:
            source = layer.self_attn
            target = copy.copy(source)
            target.__class__ = (
                TiledMergedAttention
                if role == "decoder" and model.encoder is not None
                else TiledEncoderAttention
            )
            target.query_chunk_size = options["query_chunk_size"]
            layer.self_attn = target
        if options["gradient_checkpointing"]:
            stack.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            stack.gradient_checkpointing_disable()
        if stack.config.attention_dropout != 0.0:
            raise ValueError("Tiled memory policy requires native attention dropout=0.")
        stack.config.use_cache = False
    model.loss_chunk_size = options["loss_chunk_size"]
    model.memory_policy = options
    return copy.deepcopy(options)


def run_training_memory_preflight(model, examples, *, progress=None):
    """Probe actual maxima plus a combined shape stress, without any optimizer update.

    Restores RNG/mode and discards probe gradients. Combined context/target is a
    shape-only stress fixture, never inserted into the training cohort.
    """
    from bifocalgemma.contracts import TokenizedRepairExample
    from bifocalgemma.data.collator import RepairCollator
    from bifocalgemma.training.telemetry import resource_snapshot

    if not examples or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Preflight requires fresh parameters and nonempty examples.")

    def cost(item):
        e = len(item.encoder_ids)
        d = len(item.decoder_prefix_ids) + len(item.target_ids) - 1
        return e * e + d * (e + d)

    if hasattr(examples, "extreme_examples"):
        longest_encoder, longest_decoder, hardest_joint = examples.extreme_examples()
    else:
        longest_encoder = max(examples, key=lambda x: len(x.encoder_ids))
        longest_decoder = max(examples, key=lambda x: len(x.decoder_prefix_ids) + len(x.target_ids))
        hardest_joint = max(examples, key=cost)
    items = {x.instance_id: x for x in (longest_encoder, longest_decoder, hardest_joint)}
    combined = TokenizedRepairExample(
        "shape-only-combined-maxima",
        longest_encoder.encoder_ids,
        longest_decoder.decoder_prefix_ids,
        longest_decoder.target_ids,
        {"source": "shape_stress_only"},
    )
    probes = [*items.values(), combined]
    device = next(model.parameters()).device
    old_mode, old_rng = model.training, torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    results = []
    try:
        model.train()
        for item in probes:
            batch = RepairCollator()([item]).to(device)
            row = {
                "stage": "memory_preflight_started",
                "instance_ids": [item.instance_id],
                "encoder_tokens": batch.encoder_input_ids.shape[1],
                "decoder_tokens": batch.decoder_input_ids.shape[1],
                "batch_target_tokens": int((batch.labels != -100).sum()),
                "shape_only": item.instance_id == combined.instance_id,
            }
            if progress:
                progress({**row, **resource_snapshot(device)})
            model.zero_grad(set_to_none=True)
            output = model.forward_loss(batch)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError("Nonfinite memory-preflight loss.")
            output.loss.backward()
            gradients = [
                p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
            ]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError("Invalid memory-preflight gradients.")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row.update(
                stage="memory_preflight_complete",
                loss=output.loss.detach().item(),
                **resource_snapshot(device),
            )
            results.append(row)
            if progress:
                progress(row)
            del output, batch, gradients
            model.zero_grad(set_to_none=True)
        return {
            "status": "PASS",
            "scope": "actual_length_maxima_and_combined_shape_not_quality",
            "optimizer_updates": 0,
            "rows": results,
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(old_mode)
        torch.set_rng_state(old_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
