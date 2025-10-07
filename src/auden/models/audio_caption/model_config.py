from __future__ import annotations

from typing import Any

from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig


class AudioCaptionConfig(BaseConfig):
    """Configuration for Audio Caption model (audio encoder + custom Transformer decoder).

    Fields
    -----
    - audio_encoder_config: dict | BaseConfig | None
        Encoder configuration for audio backbone. When dict, passed to AutoConfig.for_model.
        Defaults to ZipformerConfig when None.
    - d_model: int | None
        Decoder model dimension. If None, defaults to max(encoder_dim).
    - decoder_nhead: int
        Number of attention heads in decoder.
    - num_decoder_layers: int
        Number of TransformerDecoder layers.
    - dim_feedforward: int
        FFN inner dimension in decoder layers.
    - decoder_dropout: float
        Dropout rate in decoder layers.
    - decoder_activation: str
        Activation function in decoder FFN.
    - decoder_norm_first: bool
        Whether to apply LayerNorm before attention/FFN blocks.
    - decoder_bias: bool
        Whether to use bias in attention/FFN projections.
    - decoder_shared_emb: bool
        Share output projection weight with token embedding table.
    - label_smoothing: float
        CE label smoothing.
    """

    model_type: str = "audio-caption"

    def __init__(
        self,
        *,
        audio_encoder_config: dict | BaseConfig | None = None,
        d_model: int = 768,
        decoder_nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int | None = None,
        decoder_dropout: float = 0.1,
        decoder_activation: str = "gelu",
        decoder_norm_first: bool = False,
        decoder_bias: bool = False,
        decoder_shared_emb: bool = False,
        label_smoothing: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = AutoConfig.for_model(**audio_encoder_config)
        elif audio_encoder_config is None:
            from ...models.zipformer.model_config import ZipformerConfig

            audio_encoder_config = ZipformerConfig()
        self.audio_encoder_config = audio_encoder_config

        # Decoder hyperparameters for the self-implemented Transformer decoder
        self.d_model = d_model
        self.decoder_nhead = int(decoder_nhead)
        self.num_decoder_layers = int(num_decoder_layers)
        self.dim_feedforward = (
            int(dim_feedforward) if dim_feedforward is not None else 4 * d_model
        )
        self.decoder_dropout = float(decoder_dropout)
        self.decoder_activation = str(decoder_activation)
        self.decoder_norm_first = bool(decoder_norm_first)
        self.decoder_bias = bool(decoder_bias)
        self.decoder_shared_emb = bool(decoder_shared_emb)
        self.label_smoothing = float(label_smoothing)

        super().__init__(**kwargs)
