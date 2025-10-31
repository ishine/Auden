from __future__ import annotations

from transformers import AutoConfig as HFConfig

from omegaconf import DictConfig
from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig

def load_encoder_config(config):
    if config.model_type is None:
        return None
    if 'whisper' in config.model_type:
        config = HFConfig.from_pretrained(config.model_path)
    else:
        config = AutoConfig.from_pretrained(config.model_path).encoder_config
    return config


class AzerosConfig(BaseConfig):
    model_type = "azeros"
    use_flash_attn: bool = False
    exclude_from_checkpoint: list = None

    def __init__(
        self,
        llm,
        speech_encoder,
        paraling_encoder,
        **kwargs,
    ):
        self.speech_encoder = DictConfig(speech_encoder)
        self.paraling_encoder = DictConfig(paraling_encoder)
        self.speech_encoder_config = load_encoder_config(self.speech_encoder)
        self.paraling_encoder_config = load_encoder_config(self.paraling_encoder)

        self.llm = llm
        self.llm_config = HFConfig.from_pretrained(llm)

        super().__init__(**kwargs)
