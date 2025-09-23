from __future__ import annotations

from transformers import AutoConfig as HFConfig
from transformers import PretrainedConfig

from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig


class ClapConfig(BaseConfig):
    """Configuration for CLAP-style audio-text model."""

    model_type: str = "clap"

    def __init__(
        self,
        *,
        audio_encoder_config: dict | BaseConfig | None = None,
        text_encoder_config: dict | PretrainedConfig | None = None,
        shared_emb_dim: int = 1024,
        temp: float = 0.07,
        loss: str = "info-nce",
        embed_reg: bool = True,
        **kwargs,
    ):
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = AutoConfig.for_model(**audio_encoder_config)
        elif audio_encoder_config is None:
            from ...models.zipformer.model_config import ZipformerConfig

            audio_encoder_config = ZipformerConfig()
        self.audio_encoder_config = audio_encoder_config

        if isinstance(text_encoder_config, dict):
            text_encoder_config = HFConfig.for_model(**text_encoder_config)
        elif text_encoder_config is None:
            text_encoder_config = HFConfig.from_pretrained("bert-base-uncased")
        self.text_encoder_config = text_encoder_config

        self.loss = loss
        self.shared_emb_dim = shared_emb_dim
        self.temp = temp
        self.embed_reg = embed_reg

        super().__init__(**kwargs)
