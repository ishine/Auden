from __future__ import annotations

from transformers import AutoConfig as HFConfig

from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig


class TtaConfig(BaseConfig):
    model_type = "tta"

    def __init__(
        self,
        speech_encoder_config=None,
        text_encoder_config=None,
        special_tokens=None,
        decoder_dim: int = 512,
        context_size: int = 2,
        joiner_dim: int = 512,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        ignore_id: int = -1,
        label_smoothing: float = 0.1,
        **kwargs,
    ):
        # speech encoder config
        if speech_encoder_config is None:
            speech_encoder_config = AutoConfig.from_pretrained("zipformer-large")
        elif isinstance(speech_encoder_config, dict):
            speech_encoder_config = AutoConfig.for_model(**speech_encoder_config)
        self.speech_encoder_config = speech_encoder_config

        # text encoder config
        if text_encoder_config is None:
            text_encoder_config = HFConfig.from_pretrained(
                "bert-base-multilingual-uncased"
            )
        else:
            if isinstance(text_encoder_config, dict):
                text_encoder_config = HFConfig.for_model(**text_encoder_config)
        self.text_encoder_config = text_encoder_config

        self.special_tokens = special_tokens
        self.decoder_dim = decoder_dim
        self.context_size = context_size
        self.joiner_dim = joiner_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.ignore_id = ignore_id
        self.label_smoothing = label_smoothing

        super().__init__(**kwargs)
