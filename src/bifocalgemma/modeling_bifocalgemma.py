# mypy: disable-error-code="override"
"""Hugging Face standard PreTrainedModel implementation for BifocalGemma."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    Gemma4TextConfig,
    Gemma4TextModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from bifocalgemma.configuration_bifocalgemma import BifocalGemmaConfig
from bifocalgemma.contracts import ModelOutput, Seq2SeqBatch
from bifocalgemma.models.checkpoint_transfer import transfer_text_checkpoint
from bifocalgemma.training.memory_probe import ProbeLoRALinear
from bifocalgemma.v3.attention import configure_full_rank_cross_attention
from bifocalgemma.v3.bridge import SwiGLUDualGateBridge
from bifocalgemma.v3.objectives import is_cross_bridge_parameter

logger = logging.getLogger(__name__)


def _resolve_backbone_weights(
    path_or_id: str | Path,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    p = Path(path_or_id)
    if p.is_file():
        return p
    if p.is_dir():
        cand = p / "model.safetensors"
        if cand.is_file():
            return cand
        safes = sorted(p.glob("*.safetensors"))
        if safes:
            return safes[0]
        bins = sorted(p.glob("pytorch_model*.bin"))
        if bins:
            return bins[0]

    # Try Hugging Face Hub download or cache
    try:
        from huggingface_hub import hf_hub_download

        resolved = hf_hub_download(
            repo_id=str(path_or_id),
            filename="model.safetensors",
            cache_dir=cache_dir,
            token=token,
        )
        return Path(resolved)
    except Exception as exc:
        import os

        hub_cache = (
            Path(cache_dir) if cache_dir else Path(os.path.expanduser("~/.cache/huggingface/hub"))
        )
        repo_dir = hub_cache / f"models--{str(path_or_id).replace('/', '--')}"
        if repo_dir.is_dir():
            safes = sorted(repo_dir.glob("**/model.safetensors"))
            if safes:
                return safes[0]
        raise FileNotFoundError(
            f"Could not resolve pretrained backbone weights for {path_or_id!r}: {exc}"
        ) from exc


class BifocalGemmaPreTrainedModel(PreTrainedModel):
    """Base class for all BifocalGemma PreTrainedModel architectures."""

    config_class = BifocalGemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Gemma4TextDecoderLayer",
        "SwiGLUDualGateBridge",
        "FullCrossTiledAttention",
    ]
    _tied_weights_keys = {"lm_head.weight": "decoder.embed_tokens.weight"}

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class BifocalGemmaForCausalLM(BifocalGemmaPreTrainedModel):
    """Full-featured standalone Hugging Face BifocalGemma Causal LM.

    Combines a bidirectional long-context encoder (32k), non-linear SwiGLU dual-gate bridge,
    and decoupled cross-attention causal decoder (8k) into an intact, unified PreTrainedModel.
    Provides standard save_pretrained() and from_pretrained() with automatic LoRA consolidation
    (merge_and_unload) into clean model.safetensors artifacts.
    """

    def __init__(self, config: BifocalGemmaConfig) -> None:
        super().__init__(config)
        self.config = config

        # 1. Instantiate decoder text backbone
        dec_cfg = (
            Gemma4TextConfig.from_dict(config.decoder_config)
            if isinstance(config.decoder_config, dict)
            else config.decoder_config
        )
        dec_cfg._attn_implementation = "eager"
        self.decoder = Gemma4TextModel(dec_cfg)

        # 2. Instantiate encoder text backbone if configured
        if config.encoder_config is not None and config.encoder_size != "none":
            enc_cfg = (
                Gemma4TextConfig.from_dict(config.encoder_config)
                if isinstance(config.encoder_config, dict)
                else config.encoder_config
            )
            enc_cfg._attn_implementation = "eager"
            self.encoder: Gemma4TextModel | None = Gemma4TextModel(enc_cfg)
            self.encoder_projection: nn.Module | None = SwiGLUDualGateBridge(
                d_enc=config.d_enc,
                d_dec=config.d_dec,
                d_mid=config.d_mid,
                init_rms_scale=config.init_rms_scale,
            )
        else:
            self.encoder = None
            self.encoder_projection = None
            self._tied_weights_keys = {"lm_head.weight": "decoder.embed_tokens.weight"}

        # 3. LM head
        self.config.vocab_size = dec_cfg.vocab_size
        self.lm_head = nn.Linear(config.d_dec, dec_cfg.vocab_size, bias=False)
        # Transformers caches this mapping at the start of post_init(), before
        # calling tie_weights(recompute_mapping=False). Set the complete instance
        # policy now so a default class mapping cannot alias an independent encoder.
        self._tied_weights_keys = {}
        if config.tie_word_embeddings:
            self._tied_weights_keys["lm_head.weight"] = "decoder.embed_tokens.weight"
        if self.encoder is not None and config.tie_encoder_decoder_words and config.d_enc == config.d_dec:
            self._tied_weights_keys["encoder.embed_tokens.weight"] = "decoder.embed_tokens.weight"

        # 4. Configure full-rank decoupled cross attention on decoder layers
        configure_full_rank_cross_attention(self)

        # 5. Tie embeddings
        self._tie_words()
        self.loss_chunk_size = 128
        self.post_init()

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        """Tie weights cleanly, excluding encoder unless explicitly configured and dimensions match."""
        tie_enc_dec = (
            getattr(self, "encoder", None) is not None
            and getattr(self.config, "tie_encoder_decoder_words", False)
            and getattr(self.config, "d_enc", None) == getattr(self.config, "d_dec", None)
        )
        tied: dict[str, str] = {}
        if getattr(self.config, "tie_word_embeddings", True):
            tied["lm_head.weight"] = "decoder.embed_tokens.weight"
        if tie_enc_dec:
            tied["encoder.embed_tokens.weight"] = "decoder.embed_tokens.weight"
        self._tied_weights_keys = tied
        super().tie_weights(*args, **kwargs)

    def _tie_words(self) -> None:
        """Tie decoder word embeddings to lm_head and optionally encoder embeddings."""
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.decoder.embed_tokens.weight
        if (
            self.encoder is not None
            and getattr(self.config, "tie_encoder_decoder_words", False)
            and getattr(self.config, "d_enc", None) == getattr(self.config, "d_dec", None)
            and hasattr(self.encoder, "embed_tokens")
        ):
            self.encoder.embed_tokens.weight = self.decoder.embed_tokens.weight

    def _apply(self, fn: Callable[..., Any], recurse: bool = True) -> BifocalGemmaForCausalLM:
        super()._apply(fn, recurse=recurse)
        for name, param in self.named_parameters():
            if (
                is_cross_bridge_parameter(name) or name.endswith((".lora_A", ".lora_B"))
            ) and param.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                param.data = param.data.float()
                if param.grad is not None and param.grad.dtype in (torch.bfloat16, torch.float16):
                    param.grad.data = param.grad.data.float()
        self._tie_words()
        return self

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._tie_words()
        return result

    def merge_and_unload(self) -> BifocalGemmaForCausalLM:
        """Merge all active LoRA parameters (ProbeLoRALinear or PEFT) directly into base weights.

        Consolidates the adapter weights into pristine base parameter tensors and unwraps the
        modules back into standard nn.Linear layers.
        """
        # Merge ProbeLoRALinear modules
        for name, module in list(self.named_modules()):
            if isinstance(module, ProbeLoRALinear):
                if "." in name:
                    parent_name, attr = name.rsplit(".", 1)
                    parent = self.get_submodule(parent_name)
                else:
                    parent = self
                    attr = name
                with torch.no_grad():
                    delta_w = (
                        module.lora_B.data.float() @ module.lora_A.data.float()
                    ) * module.scaling
                    module.base.weight.data.add_(delta_w.to(module.base.weight.dtype))
                setattr(parent, attr, module.base)

        # Check for PEFT merge_and_unload
        base_model = getattr(self, "base_model", None)
        if base_model is not None and hasattr(base_model, "merge_and_unload"):
            try:
                cast(Any, base_model).merge_and_unload()
            except Exception as exc:
                logger.debug("PEFT merge_and_unload skipped: %s", exc)

        self._tie_words()
        return self

    def enable_memory_efficient_training(
        self,
        query_chunk_size: int = 128,
        loss_chunk_size: int = 128,
        gradient_checkpointing: bool = True,
    ) -> BifocalGemmaForCausalLM:
        """Configure tiled attention, gradient checkpointing, and chunked loss for memory efficiency."""
        import copy

        from bifocalgemma.models.memory_efficient import TiledEncoderAttention

        if self.encoder is not None:
            for layer in self.encoder.layers:
                attn = layer.self_attn
                if not isinstance(attn, TiledEncoderAttention):
                    tiled = copy.copy(attn)
                    object.__setattr__(tiled, "__class__", TiledEncoderAttention)
                    object.__setattr__(tiled, "query_chunk_size", query_chunk_size)
                    object.__setattr__(tiled, "attention_backend", "tiled_eager_fp32")
                    object.__setattr__(tiled, "structured_masks", True)
                    layer.self_attn = cast(Any, tiled)
                else:
                    object.__setattr__(attn, "query_chunk_size", query_chunk_size)
                    object.__setattr__(attn, "structured_masks", True)

        if self.decoder is not None:
            for layer in self.decoder.layers:
                attn = layer.self_attn
                if self.encoder is None:
                    if not isinstance(attn, TiledEncoderAttention):
                        tiled = copy.copy(attn)
                        object.__setattr__(tiled, "__class__", TiledEncoderAttention)
                        object.__setattr__(tiled, "query_chunk_size", query_chunk_size)
                        object.__setattr__(tiled, "attention_backend", "tiled_eager_fp32")
                        object.__setattr__(tiled, "structured_masks", True)
                        layer.self_attn = cast(Any, tiled)
                    else:
                        object.__setattr__(attn, "query_chunk_size", query_chunk_size)
                        object.__setattr__(attn, "structured_masks", True)
                else:
                    # Paired decoders already use FullCrossTiledAttention
                    object.__setattr__(attn, "query_chunk_size", query_chunk_size)

        for stack in (self.encoder, self.decoder):
            if stack is not None:
                if gradient_checkpointing:
                    stack.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                else:
                    stack.gradient_checkpointing_disable()
                stack.config.use_cache = False

        self.loss_chunk_size = loss_chunk_size
        return self

    def encode_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode long context using the bidirectional encoder and SwiGLU manifold bridge."""
        if self.encoder is None or self.encoder_projection is None:
            raise ValueError("Decoder-only baseline model does not have an encoder.")
        if input_ids.shape[1] == 0:
            return self.decoder.embed_tokens.weight.new_empty(
                input_ids.shape[0], 0, self.decoder.config.hidden_size
            )
        eff_mask = attention_mask if attention_mask is not None else encoder_attention_mask
        if eff_mask is None:
            eff_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        pos_ids = self._positions(eff_mask)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=eff_mask,
            self_valid=eff_mask,
            position_ids=pos_ids,
            use_cache=False,
        )
        hidden = (
            encoder_outputs.last_hidden_state
            if hasattr(encoder_outputs, "last_hidden_state")
            else encoder_outputs[0]
        )
        return self.encoder_projection(hidden).masked_fill(~eff_mask.bool()[..., None], 0)

    @staticmethod
    def _positions(mask: torch.Tensor) -> torch.Tensor:
        return (mask.long().cumsum(-1) - 1).clamp_min(0)

    def _decode_hidden(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None,
        encoder_states: torch.Tensor | None,
        encoder_mask: torch.Tensor | None,
        *,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        """Pass decoder inputs through Gemma4TextModel with cross-attention kwargs."""
        batch, length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(
                (batch, length), dtype=torch.bool, device=decoder_input_ids.device
            )

        kwargs: dict[str, Any] = {}
        if self.encoder is not None:
            if encoder_states is None:
                encoder_states = self.decoder.embed_tokens.weight.new_empty(
                    batch, 0, self.decoder.config.hidden_size
                )
                encoder_mask = decoder_attention_mask.new_zeros((batch, 0))
            if encoder_mask is None:
                encoder_mask = torch.ones(
                    (batch, encoder_states.shape[1]),
                    dtype=torch.bool,
                    device=decoder_input_ids.device,
                )
            kwargs = dict(
                encoder_hidden_states=encoder_states,
                encoder_valid=encoder_mask,
                decoder_valid=decoder_attention_mask,
            )

        pos_ids = self._positions(decoder_attention_mask)
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            self_valid=decoder_attention_mask,
            position_ids=pos_ids,
            use_cache=False,
            **kwargs,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if last_token_only:
            hidden = hidden[:, -1:]
        return hidden

    def decode_logits(
        self,
        input_ids: torch.Tensor,
        decoder_mask: torch.Tensor | None,
        encoder_states: torch.Tensor | None,
        encoder_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute vocabulary logits over decoder tokens."""
        hidden = self._decode_hidden(input_ids, decoder_mask, encoder_states, encoder_mask)
        logits = self.lm_head(hidden)
        cap = getattr(self.decoder.config, "final_logit_softcapping", None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        batch: Seq2SeqBatch | None = None,
        return_dict: bool = True,
        shift_labels: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | ModelOutput | tuple[Any, ...]:
        """Forward execution supporting Hugging Face standard kwargs and Seq2SeqBatch."""
        # Support Seq2SeqBatch input
        if batch is not None:
            input_ids = batch.decoder_input_ids
            attention_mask = batch.decoder_attention_mask
            encoder_input_ids = batch.encoder_input_ids
            encoder_attention_mask = batch.encoder_attention_mask
            labels = batch.labels
        elif isinstance(input_ids, Seq2SeqBatch):
            attention_mask = input_ids.decoder_attention_mask
            encoder_input_ids = input_ids.encoder_input_ids
            encoder_attention_mask = input_ids.encoder_attention_mask
            labels = input_ids.labels
            input_ids = input_ids.decoder_input_ids

        if input_ids is None:
            raise ValueError("input_ids or batch must be provided to forward().")

        # Context encoding
        if (
            encoder_hidden_states is None
            and encoder_input_ids is not None
            and self.encoder is not None
        ):
            encoder_hidden_states = self.encode_context(
                encoder_input_ids, encoder_attention_mask=encoder_attention_mask
            )

        logits = self.decode_logits(
            input_ids,
            decoder_mask=attention_mask,
            encoder_states=encoder_hidden_states,
            encoder_mask=encoder_attention_mask,
        )

        loss = None
        if labels is not None:
            import torch.nn.functional as F

            should_shift = (
                shift_labels
                if shift_labels is not None
                else (labels.shape[1] > 1 and labels.shape == input_ids.shape and not (labels == -100).any())
            )
            if not should_shift:
                loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                shift_log = logits[..., :-1, :].contiguous()
                shift_lab = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_log.view(-1, shift_log.size(-1)),
                    shift_lab.view(-1),
                    ignore_index=-100,
                )

        if not return_dict:
            output = (logits, encoder_hidden_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=cast(Any, loss),
            logits=cast(Any, logits),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def forward_loss(self, batch: Any) -> ModelOutput:
        """Bridge compatibility helper for legacy training loops."""
        result = self.forward(batch=batch)
        if isinstance(result, tuple):
            loss = result[0] if len(result) > 1 else None
            logits = result[1] if len(result) > 1 else result[0]
            return ModelOutput(loss=loss, logits=logits)
        return ModelOutput(loss=result.loss, logits=result.logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive greedy generation with pre-encoded context representation."""
        self.eval()
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        encoder_states = None
        if encoder_input_ids is not None and self.encoder is not None:
            encoder_states = self.encode_context(
                encoder_input_ids, encoder_attention_mask=encoder_attention_mask
            )

        current_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.decode_logits(
                current_ids,
                decoder_mask=None,
                encoder_states=encoder_states,
                encoder_mask=encoder_attention_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if eos_id is not None and bool((next_token == eos_id).all()):
                break
        return current_ids

    def save_pretrained(  # ty: ignore[incompatible-method-override]  # pyrefly: ignore [bad-override]  # type: ignore[override]
        self,
        save_directory: str | Path,
        is_main_process: bool = True,
        state_dict: dict[str, torch.Tensor] | None = None,
        save_function: Callable[..., Any] | None = None,
        push_to_hub: bool = False,
        safe_serialization: bool = True,
        token: str | bool | None = None,
        tokenizer: Any = None,
        copy_default_tokenizer: bool = True,
        distributed_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        """Save complete pre-trained model artifact after merging active LoRA adapters."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Consolidate and merge all LoRA adapters into standard parameters
        self.merge_and_unload()

        # 2. Save Hugging Face configuration
        self.config.save_pretrained(save_dir)

        # 3. Save generation configuration if present, or write default
        if hasattr(self, "generation_config") and self.generation_config is not None:
            try:
                self.generation_config.save_pretrained(save_dir)
            except Exception as _exc:
                logger.debug("Failed saving generation_config: %s", _exc)
        if not (save_dir / "generation_config.json").exists():
            try:
                from transformers import GenerationConfig

                gen_cfg = GenerationConfig(
                    bos_token_id=getattr(self.config, "bos_token_id", 2),
                    eos_token_id=getattr(self.config, "eos_token_id", 1),
                    pad_token_id=getattr(self.config, "pad_token_id", 0),
                    max_new_tokens=getattr(self.config, "max_new_tokens", 512),
                )
                gen_cfg.save_pretrained(save_dir)
            except Exception:
                gen_data = {
                    "bos_token_id": getattr(self.config, "bos_token_id", 2),
                    "eos_token_id": getattr(self.config, "eos_token_id", 1),
                    "pad_token_id": getattr(self.config, "pad_token_id", 0),
                    "max_new_tokens": 512,
                }
                (save_dir / "generation_config.json").write_text(
                    json.dumps(gen_data, indent=2), encoding="utf-8"
                )

        # 4. Save model weights as safetensors or bin
        if safe_serialization:
            weights_file = save_dir / "model.safetensors"
            to_save = state_dict if state_dict is not None else self.state_dict()
            clean_state: dict[str, torch.Tensor] = {}
            seen_ptrs: set[int] = set()
            for k, v in to_save.items():
                t = v.detach().cpu()
                if t.data_ptr() in seen_ptrs:
                    clean_state[k] = t.clone().contiguous()
                else:
                    clean_state[k] = t.contiguous()
                    seen_ptrs.add(t.data_ptr())
            save_file(clean_state, str(weights_file))
        else:
            bin_file = save_dir / "pytorch_model.bin"
            if state_dict is not None:
                torch.save(state_dict, str(bin_file))
            else:
                torch.save(self.state_dict(), str(bin_file))

        # 5. Save tokenizer artifacts
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(save_dir)
        elif copy_default_tokenizer:
            default_tok_dir = Path("outputs/context_tokenizers/raw/google__gemma-4-E2B-it")
            if default_tok_dir.is_dir():
                for tok_file in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "special_tokens_map.json",
                ):
                    src = default_tok_dir / tok_file
                    if src.is_file():
                        shutil.copy2(src, save_dir / tok_file)

        # 6. Save Python code files for remote code / zero-dependency Hub compatibility
        try:
            import inspect

            import bifocalgemma.configuration_bifocalgemma as cfg_mod
            import bifocalgemma.modeling_bifocalgemma as model_mod

            cfg_file = Path(inspect.getfile(cfg_mod))
            model_file = Path(inspect.getfile(model_mod))
            if cfg_file.is_file():
                shutil.copy2(cfg_file, save_dir / "configuration_bifocalgemma.py")
            if model_file.is_file():
                shutil.copy2(model_file, save_dir / "modeling_bifocalgemma.py")
        except Exception as _exc:
            logger.debug("Remote code script copy skipped: %s", _exc)

        # 7. Generate model card
        model_card = f"""---
language: en
tags:
- code
- bifocalgemma
- causal-lm
- long-context
pipeline_tag: text-generation
license: apache-2.0
---

# BifocalGemma Pretrained Model

BifocalGemma integrates a bidirectional encoder (32k) with a causal decoder (8k)
via a SwiGLU dual-gate manifold bridge and decoupled cross-attention.

- Model Architecture: BifocalGemmaForCausalLM
- Encoder: {self.config.encoder_size}
- Decoder: {self.config.decoder_size}
- Hidden Dimensions: d_enc={self.config.d_enc}, d_dec={self.config.d_dec}
"""
        (save_dir / "README.md").write_text(model_card, encoding="utf-8")

        if push_to_hub:
            from huggingface_hub import HfApi

            repo_id = kwargs.get("repo_id")
            path_in_repo: str | None = kwargs.get("path_in_repo")
            if repo_id:
                api = HfApi(token=token)
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=str(save_dir),
                    repo_id=repo_id,
                    repo_type="model",
                    path_in_repo=path_in_repo,
                )

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        config: BifocalGemmaConfig | None = None,
        cache_dir: str | Path | None = None,
        torch_dtype: torch.dtype | str | None = None,
        device_map: str | dict[str, Any] | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> BifocalGemmaForCausalLM:
        """Load an intact BifocalGemma pre-trained model from local path or Hub."""
        subfolder: str | None = kwargs.pop("subfolder", None)
        model_path = Path(pretrained_model_name_or_path)
        if model_path.exists():
            if subfolder:
                model_path = model_path / subfolder
        else:
            from huggingface_hub import snapshot_download

            patterns = [f"{subfolder.strip('/')}/*"] if subfolder else None
            downloaded = Path(
                snapshot_download(
                    str(pretrained_model_name_or_path),
                    cache_dir=cache_dir,
                    token=kwargs.get("token"),
                    allow_patterns=patterns,
                )
            )
            model_path = downloaded / subfolder.strip("/") if subfolder else downloaded

        if config is None:
            config = BifocalGemmaConfig.from_pretrained(model_path)

        model = cls(config)
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            state_dict = load_file(str(weights_path), device="cpu")
            load_result = model.load_state_dict(state_dict, strict=strict)
            if load_result.missing_keys:
                logger.warning("from_pretrained: missing keys when loading state_dict: %s", load_result.missing_keys)
                # Check for critical missing core layers
                critical_missing = [k for k in load_result.missing_keys if "decoder.embed_tokens" in k or "decoder.layers.0." in k]
                if critical_missing:
                    raise ValueError(f"Checkpoint in {model_path} is missing critical backbone weights: {critical_missing}")
            if load_result.unexpected_keys:
                logger.warning("from_pretrained: unexpected keys in state_dict: %s", load_result.unexpected_keys)
        else:
            bin_path = model_path / "pytorch_model.bin"
            if bin_path.exists():
                state_dict = torch.load(str(bin_path), map_location="cpu")
                load_result = model.load_state_dict(state_dict, strict=strict)
                if load_result.missing_keys:
                    logger.warning("from_pretrained: missing keys when loading state_dict: %s", load_result.missing_keys)
                    critical_missing = [k for k in load_result.missing_keys if "decoder.embed_tokens" in k or "decoder.layers.0." in k]
                    if critical_missing:
                        raise ValueError(f"Checkpoint in {model_path} is missing critical backbone weights: {critical_missing}")
                if load_result.unexpected_keys:
                    logger.warning("from_pretrained: unexpected keys in state_dict: %s", load_result.unexpected_keys)

        if torch_dtype is not None:
            if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
                torch_dtype = getattr(torch, torch_dtype)
            if isinstance(torch_dtype, torch.dtype):
                cast(Any, model).to(dtype=torch_dtype)

        if device_map is not None:
            if isinstance(device_map, (str, torch.device)):
                cast(Any, model).to(device=device_map)

        model._tie_words()
        model.eval()
        return model

    @classmethod
    def from_pretrained_backbones(
        cls,
        config: BifocalGemmaConfig | None = None,
        *,
        decoder_path_or_id: str | Path = "google/gemma-4-E2B-it",
        encoder_path_or_id: str | Path | None = None,
        pretrained_backbone_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        torch_dtype: torch.dtype | str | None = None,
        device: str | torch.device | None = None,
        device_map: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BifocalGemmaForCausalLM:
        """Load pretrained Gemma-4 backbone weights into decoder (and optional encoder),
        while cleanly initializing the SwiGLU bridge and decoupled cross-attention.
        """
        if pretrained_backbone_dir is not None:
            backbone_dir = Path(pretrained_backbone_dir)
            dec_candidate = backbone_dir / "decoder"
            enc_candidate = backbone_dir / "encoder"
            if dec_candidate.exists():
                decoder_path_or_id = dec_candidate
            elif (backbone_dir / "model.safetensors").is_file():
                decoder_path_or_id = backbone_dir / "model.safetensors"
            if enc_candidate.exists():
                encoder_path_or_id = enc_candidate

        is_baseline = encoder_path_or_id is None and (
            config is not None and config.encoder_size == "none"
        )
        if config is None:
            config = BifocalGemmaConfig(
                encoder_size="none" if is_baseline else "E2B",
                decoder_size="E2B",
                tiny=False,
                gate_init_tanh=0.05,
                init_rms_scale=1.0,
                learnable_cross_scale=True,
                tie_encoder_decoder_words=False,
            )

        model = cls(config)

        if not config.tiny:
            # 1. Resolve and transfer decoder weights
            dec_file = _resolve_backbone_weights(
                decoder_path_or_id, cache_dir=cache_dir, token=token
            )
            transfer_text_checkpoint(model.decoder, dec_file)

            # 2. Resolve and transfer encoder weights if configured
            if model.encoder is not None:
                enc_id = encoder_path_or_id or "google/gemma-4-E2B"
                enc_file = _resolve_backbone_weights(enc_id, cache_dir=cache_dir, token=token)
                transfer_text_checkpoint(
                    model.encoder,
                    enc_file,
                    skip_words=getattr(config, "tie_encoder_decoder_words", False),
                )

            # 3. Cleanly initialize the SwiGLU bridge and decoupled cross-attention
            configure_full_rank_cross_attention(model)
        else:
            # In tiny mode, allow explicit local files or directories for verification tests
            try:
                dec_file = None
                if (
                    isinstance(decoder_path_or_id, (str, Path))
                    and Path(decoder_path_or_id).exists()
                ):
                    dec_file = _resolve_backbone_weights(
                        decoder_path_or_id, cache_dir=cache_dir, token=token
                    )
                if dec_file is not None and dec_file.is_file():
                    transfer_text_checkpoint(model.decoder, dec_file)

                if model.encoder is not None and encoder_path_or_id:
                    enc_file = None
                    if (
                        isinstance(encoder_path_or_id, (str, Path))
                        and Path(encoder_path_or_id).exists()
                    ):
                        enc_file = _resolve_backbone_weights(
                            encoder_path_or_id, cache_dir=cache_dir, token=token
                        )
                    if enc_file is not None and enc_file.is_file():
                        transfer_text_checkpoint(
                            model.encoder,
                            enc_file,
                            skip_words=getattr(config, "tie_word_embeddings", True),
                        )
                configure_full_rank_cross_attention(model)
            except Exception as _exc:
                logger.debug("Tiny backbone transfer skipped: %s", _exc)

        if torch_dtype is not None:
            if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
                torch_dtype = getattr(torch, torch_dtype)
            if isinstance(torch_dtype, torch.dtype):
                cast(Any, model).to(dtype=torch_dtype)

        if device_map is not None:
            if isinstance(device_map, (str, torch.device)):
                cast(Any, model).to(device=device_map)
        elif device is not None:
            cast(Any, model).to(device=device)

        model._tie_words()
        return model


def export_matrix_pretrained_model(
    model: Any,
    spec: Any = None,
    export_dir: str | Path = "outputs/pretrained_model",
    tokenizer: Any = None,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    token: str | bool | None = None,
    path_in_repo: str | None = None,
    tokenizer_cache: str | Path | None = None,
) -> Path:
    """Export a consolidated standalone Hugging Face PreTrainedModel directory from matrix model."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    # 1. Merge all ProbeLoRALinear adapters into base weights
    for name, module in list(model.named_modules()):
        if isinstance(module, ProbeLoRALinear):
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr = name
            with torch.no_grad():
                delta_w = (module.lora_B.data.float() @ module.lora_A.data.float()) * module.scaling
                module.base.weight.data.add_(delta_w.to(module.base.weight.dtype))
            setattr(parent, attr, module.base)

    # 2. Extract or build BifocalGemmaConfig
    if hasattr(model, "config") and isinstance(model.config, BifocalGemmaConfig):
        config = model.config
    else:
        is_tiny = getattr(model.decoder.config, "vocab_size", 256000) <= 256
        enc_size = (
            spec.encoder.size
            if (spec and getattr(spec, "encoder", None))
            else ("E2B" if getattr(model, "encoder", None) is not None else "none")
        )
        dec_size = spec.decoder.size if (spec and getattr(spec, "decoder", None)) else "E2B"
        d_enc = (
            int(model.encoder.config.hidden_size)
            if getattr(model, "encoder", None) is not None
            else 0
        )
        d_dec = int(model.decoder.config.hidden_size)
        d_mid = (
            int(getattr(model.encoder_projection, "d_mid", 2048))
            if getattr(model, "encoder_projection", None) is not None
            else 2048
        )

        config = BifocalGemmaConfig(
            encoder_size=enc_size,
            decoder_size=dec_size,
            tiny=is_tiny,
            gate_init_tanh=0.05,
            init_rms_scale=1.0,
            learnable_cross_scale=True,
            tie_encoder_decoder_words=False,
            encoder_config=model.encoder.config.to_dict()
            if getattr(model, "encoder", None) is not None
            else None,
            decoder_config=model.decoder.config.to_dict(),
            d_enc=d_enc,
            d_dec=d_dec,
            d_mid=d_mid,
        )

    # 3. Save config.json
    config.save_pretrained(export_path)

    # 4. Save generation config if available or write default
    gen_config_path = export_path / "generation_config.json"
    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.save_pretrained(export_path)
        except Exception as _exc:
            logger.debug("Failed saving generation_config: %s", _exc)
    if not gen_config_path.exists():
        try:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig(
                bos_token_id=getattr(config, "bos_token_id", 2),
                eos_token_id=getattr(config, "eos_token_id", 1),
                pad_token_id=getattr(config, "pad_token_id", 0),
                max_new_tokens=getattr(config, "max_new_tokens", 512),
            )
            gen_cfg.save_pretrained(export_path)
        except Exception:
            gen_data = {
                "bos_token_id": getattr(config, "bos_token_id", 2),
                "eos_token_id": getattr(config, "eos_token_id", 1),
                "pad_token_id": getattr(config, "pad_token_id", 0),
                "max_new_tokens": 512,
            }
            gen_config_path.write_text(json.dumps(gen_data, indent=2), encoding="utf-8")

    # 5. Save model.safetensors
    weights_path = export_path / "model.safetensors"
    clean_state: dict[str, torch.Tensor] = {}
    seen_ptrs: set[int] = set()
    for name, tensor in model.state_dict().items():
        if name.endswith((".lora_A", ".lora_B")):
            continue
        t = tensor.detach().cpu()
        if t.data_ptr() in seen_ptrs:
            clean_state[name] = t.clone().contiguous()
        else:
            clean_state[name] = t.contiguous()
            seen_ptrs.add(t.data_ptr())
    save_file(clean_state, str(weights_path))

    # 6. Save Python code files for remote code / zero-dependency Hub compatibility
    try:
        import inspect

        import bifocalgemma.configuration_bifocalgemma as cfg_mod
        import bifocalgemma.modeling_bifocalgemma as model_mod

        cfg_file = Path(inspect.getfile(cfg_mod))
        model_file = Path(inspect.getfile(model_mod))
        if cfg_file.is_file():
            shutil.copy2(cfg_file, export_path / "configuration_bifocalgemma.py")
        if model_file.is_file():
            shutil.copy2(model_file, export_path / "modeling_bifocalgemma.py")
    except Exception as _exc:
        logger.debug("Remote code script copy skipped: %s", _exc)

    # 7. Generate model card
    model_card = f"""---
language: en
tags:
- code
- bifocalgemma
- causal-lm
- long-context
pipeline_tag: text-generation
license: apache-2.0
---

# BifocalGemma Pretrained Model

BifocalGemma integrates a bidirectional encoder (32k) with a causal decoder (8k)
via a SwiGLU dual-gate manifold bridge and decoupled cross-attention.

- Model Architecture: BifocalGemmaForCausalLM
- Encoder: {config.encoder_size}
- Decoder: {config.decoder_size}
- Hidden Dimensions: d_enc={config.d_enc}, d_dec={config.d_dec}
"""
    (export_path / "README.md").write_text(model_card, encoding="utf-8")

    # 8. Save tokenizer assets if available
    dec_slug = ""
    if spec is not None and getattr(spec, "decoder", None) is not None:
        dec_slug = spec.decoder.model_id.replace("/", "__")
    elif hasattr(config, "decoder_config") and isinstance(config.decoder_config, dict):
        dec_id = config.decoder_config.get("_name_or_path") or config.decoder_config.get(
            "model_id", ""
        )
        dec_slug = str(dec_id).replace("/", "__") if dec_id else ""

    tok_candidates = []
    if tokenizer_cache is not None:
        tok_base = Path(tokenizer_cache)
        if dec_slug:
            tok_candidates.append(tok_base / "raw" / dec_slug)
            tok_candidates.append(tok_base / dec_slug)
        tok_candidates.append(tok_base)
    if dec_slug:
        tok_candidates.append(Path(f"outputs/v2_cache/tokenizers/raw/{dec_slug}"))
        tok_candidates.append(Path(f"outputs/context_tokenizers/raw/{dec_slug}"))
    tok_candidates.append(Path("outputs/context_tokenizers/raw/google__gemma-4-E2B-it"))

    for tok_dir in tok_candidates:
        if tok_dir.is_dir():
            for tok_file in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "special_tokens_map.json",
            ):
                src = tok_dir / tok_file
                dest = export_path / tok_file
                if src.is_file() and not dest.exists():
                    shutil.copy2(src, dest)

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(export_path)
    elif tokenizer is not None and hasattr(tokenizer, "save"):
        if not (export_path / "tokenizer.json").exists():
            tokenizer.save(str(export_path / "tokenizer.json"))

    if push_to_hub and repo_id:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(export_path),
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type="model",
        )

    return export_path


# Direct alias
BifocalGemmaModel = BifocalGemmaForCausalLM

# Register with Hugging Face Auto classes
try:
    AutoConfig.register("bifocalgemma", BifocalGemmaConfig)
    AutoModel.register(BifocalGemmaConfig, BifocalGemmaModel)
    AutoModelForCausalLM.register(BifocalGemmaConfig, BifocalGemmaForCausalLM)
except Exception as _exc:
    logger.debug("AutoConfig / AutoModel registration skipped: %s", _exc)
