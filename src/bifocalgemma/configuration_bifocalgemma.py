"""Hugging Face standard configuration for BifocalGemma."""

from __future__ import annotations

import copy
from typing import Any, cast

from transformers.configuration_utils import PretrainedConfig


def _check_transformers_version() -> None:
    """Verify transformers satisfies >=5.19.0,<6.0.0 without hardcoding patch releases."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("transformers")
    except PackageNotFoundError:
        import transformers

        raw = getattr(transformers, "__version__", "0.0.0")

    try:
        from packaging.version import Version

        parsed = Version(raw)
        if not (Version("5.19.0") <= parsed < Version("6.0.0")):
            raise RuntimeError(
                f"Native integration requires transformers >=5.19.0,<6.0.0 (found {raw!r})."
            )
    except ImportError:
        parts: list[int] = []
        for token in raw.split(".")[:3]:
            digits = "".join(ch for ch in token if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        if not ((5, 19, 0) <= tuple(parts[:3]) < (6, 0, 0)):
            raise RuntimeError(
                f"Native integration requires transformers >=5.19.0,<6.0.0 (found {raw!r})."
            )


def native_config(size: str, *, tiny: bool = True, encoder: bool = False) -> Any:
    """Build a Gemma4TextConfig from pinned local assets; tiny by default."""
    _check_transformers_version()
    from transformers import Gemma4TextConfig

    from bifocalgemma.assets import load_asset_json

    size_clean = size.upper()
    if size_clean not in ("E2B", "E4B"):
        raise ValueError(f"Unsupported Gemma4 size {size!r}; expected 'E2B' or 'E4B'.")
    values = dict(load_asset_json(f"gemma4_{size_clean.lower()}.json")["text_config"])
    if tiny:
        width = 32 if size_clean == "E2B" else 48
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
        gate_init_tanh: float = 0.15,
        learnable_cross_scale: bool = False,
        decouple_full_attn_cross_kv: bool = True,
        tie_word_embeddings: bool = True,
        tie_encoder_decoder_words: bool = False,
        vocab_size: int = 256000,
        max_position_embeddings: int = 32768,
        bos_token_id: int = 2,
        eos_token_id: int | list[int] = 1,
        pad_token_id: int = 0,
        tiny: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **cast(
                Any,
                {
                    "bos_token_id": bos_token_id,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": pad_token_id,
                    "tie_word_embeddings": tie_word_embeddings,
                    "tie_encoder_decoder_words": tie_encoder_decoder_words,
                    **kwargs,
                },
            )
        )

        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.tiny = tiny
        self.tie_encoder_decoder_words = tie_encoder_decoder_words

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

        self.d_enc = d_enc if d_enc is not None else enc_hidden
        self.d_dec = d_dec if d_dec is not None else dec_hidden
        self.d_mid = d_mid if d_mid is not None else max(self.d_enc, self.d_dec)

        self.init_rms_scale = float(init_rms_scale)
        self.gate_init_tanh = float(gate_init_tanh)
        self.learnable_cross_scale = learnable_cross_scale
        self.decouple_full_attn_cross_kv = decouple_full_attn_cross_kv
        dec_vocab = (
            self.decoder_config.get("vocab_size", 262144 if not self.tiny else 128)
            if isinstance(self.decoder_config, dict)
            else getattr(self.decoder_config, "vocab_size", 262144 if not self.tiny else 128)
        )
        self.vocab_size = int(dec_vocab)
        self.max_position_embeddings = max_position_embeddings
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
        """Load text configuration using canonical native_config."""
        size_clean = size.upper()
        if not size_clean.startswith("E"):
            size_clean = f"E{size_clean}"
        if size_clean not in ("E2B", "E4B"):
            size_clean = "E2B"
        return native_config(size_clean, tiny=tiny, encoder=encoder).to_dict()

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(super().to_dict())
        output["model_type"] = self.model_type
        output["encoder_config"] = self.encoder_config
        output["decoder_config"] = self.decoder_config
        output["encoder_size"] = self.encoder_size
        output["decoder_size"] = self.decoder_size
        output["tiny"] = self.tiny
        output["tie_encoder_decoder_words"] = self.tie_encoder_decoder_words
        output["d_enc"] = self.d_enc
        output["d_dec"] = self.d_dec
        output["d_mid"] = self.d_mid
        output["init_rms_scale"] = self.init_rms_scale
        output["gate_init_tanh"] = self.gate_init_tanh
        output["learnable_cross_scale"] = self.learnable_cross_scale
        output["decouple_full_attn_cross_kv"] = self.decouple_full_attn_cross_kv
        output["architectures"] = self.architectures
        output["auto_map"] = self.auto_map
        output["vocab_size"] = self.vocab_size
        return output
