"""Hugging Face standard configuration for BifocalGemma."""

from __future__ import annotations

import copy
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class BifocalGemmaConfig(PretrainedConfig):
    """Configuration class for BifocalGemma models.

    Compatible with standard Hugging Face AutoConfig and PretrainedConfig APIs.
    Houses architecture parameters for the bidirectional encoder, non-linear
    SwiGLU dual-gate bridge, and decoupled cross-attention causal decoder.
    """

    model_type = "bifocalgemma"
    is_composition = True

    def __init__(
        self,
        encoder_config: dict[str, Any] | PretrainedConfig | None = None,
        decoder_config: dict[str, Any] | PretrainedConfig | None = None,
        encoder_size: str = "E2B",
        decoder_size: str = "E2B",
        d_enc: int | None = None,
        d_dec: int | None = None,
        d_mid: int | None = None,
        init_rms_scale: float = 1.0,
        gate_init_tanh: float = 0.05,
        learnable_cross_scale: bool = True,
        tie_word_embeddings: bool = True,
        vocab_size: int = 256000,
        max_position_embeddings: int = 32768,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        tiny: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.encoder_size = str(encoder_size)
        self.decoder_size = str(decoder_size)
        self.tiny = bool(tiny)

        # Resolve decoder subconfig
        if decoder_config is None:
            decoder_config = self._load_default_text_config(self.decoder_size, tiny=self.tiny)
        elif isinstance(decoder_config, PretrainedConfig):
            decoder_config = decoder_config.to_dict()
        self.decoder_config = decoder_config

        # Resolve encoder subconfig
        if encoder_config is None and self.encoder_size != "none":
            encoder_config = self._load_default_text_config(
                self.encoder_size, tiny=self.tiny, encoder=True
            )
        elif isinstance(encoder_config, PretrainedConfig):
            encoder_config = encoder_config.to_dict()
        self.encoder_config = encoder_config

        # Hidden dimensions
        dec_hidden = int(
            self.decoder_config.get("hidden_size", 1536 if not self.tiny else 32)
            if isinstance(self.decoder_config, dict)
            else getattr(self.decoder_config, "hidden_size", 1536)
        )
        enc_hidden = (
            int(
                self.encoder_config.get("hidden_size", 1536 if not self.tiny else 32)
                if isinstance(self.encoder_config, dict)
                else getattr(self.encoder_config, "hidden_size", 1536)
            )
            if self.encoder_config is not None
            else dec_hidden
        )

        self.d_enc = int(d_enc) if d_enc is not None else enc_hidden
        self.d_dec = int(d_dec) if d_dec is not None else dec_hidden
        self.d_mid = int(d_mid) if d_mid is not None else max(self.d_enc, self.d_dec)

        self.init_rms_scale = float(init_rms_scale)
        self.gate_init_tanh = float(gate_init_tanh)
        self.learnable_cross_scale = bool(learnable_cross_scale)
        dec_vocab = (
            self.decoder_config.get("vocab_size", 262144 if not self.tiny else 128)
            if isinstance(self.decoder_config, dict)
            else getattr(self.decoder_config, "vocab_size", 262144 if not self.tiny else 128)
        )
        self.vocab_size = int(dec_vocab)
        self.max_position_embeddings = int(max_position_embeddings)
        self.architectures = ["BifocalGemmaForCausalLM"]
        self.auto_map = {
            "AutoConfig": "configuration_bifocalgemma.BifocalGemmaConfig",
            "AutoModel": "modeling_bifocalgemma.BifocalGemmaModel",
            "AutoModelForCausalLM": "modeling_bifocalgemma.BifocalGemmaForCausalLM",
        }

    @staticmethod
    def _load_default_text_config(
        size: str, *, tiny: bool = False, encoder: bool = False
    ) -> dict[str, Any]:
        """Load text configuration using native_config."""
        size_clean = size.upper()
        if not size_clean.startswith("E"):
            size_clean = f"E{size_clean}"
        if size_clean not in ("E2B", "E4B"):
            size_clean = "E2B"
        from bifocalgemma.models.v2.native import native_config
        return native_config(size_clean, tiny=tiny, encoder=encoder).to_dict()

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(super().to_dict())
        output["model_type"] = self.model_type
        output["encoder_config"] = self.encoder_config
        output["decoder_config"] = self.decoder_config
        output["d_enc"] = self.d_enc
        output["d_dec"] = self.d_dec
        output["d_mid"] = self.d_mid
        output["init_rms_scale"] = self.init_rms_scale
        output["gate_init_tanh"] = self.gate_init_tanh
        output["learnable_cross_scale"] = self.learnable_cross_scale
        output["architectures"] = self.architectures
        output["auto_map"] = self.auto_map
        output["vocab_size"] = self.vocab_size
        return output
