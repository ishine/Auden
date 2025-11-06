from __future__ import annotations

from transformers import AutoConfig as HFConfig

from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig

def load_encoder_config(config):
    if config['model_type'] is None:
        return None
    if 'whisper' in config['model_type']:
        encoder_config = HFConfig.from_pretrained(config['model_path'])
    else:
        encoder_config = AutoConfig.from_pretrained(config['model_path'])
    return encoder_config


class AzerosConfig(BaseConfig):
    model_type = "azeros"
    use_flash_attn: bool = False
    exclude_from_checkpoint: list = None

    def __init__(
        self,
        llm: str,
        speech_encoder,
        speech_encoder_projector,
        paraling_encoder,
        paraling_encoder_projector,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.speech_encoder = dict(**speech_encoder)
        self.speech_encoder_projector = dict(**speech_encoder_projector)

        self.paraling_encoder = dict(**paraling_encoder)
        self.paraling_encoder_projector = dict(**paraling_encoder_projector)

        self.llm = llm
        self.llm_config = HFConfig.from_pretrained(llm)

        self.speech_encoder_config = load_encoder_config(self.speech_encoder)
        self.paraling_encoder_config = load_encoder_config(self.paraling_encoder)
