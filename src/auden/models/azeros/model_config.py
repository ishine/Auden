from __future__ import annotations

from transformers import PretrainedConfig
from transformers import AutoConfig as HFConfig

from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig

def load_config(config):
    if config is None:
        return None
    if isinstance(config, dict):
        try:
            config = AutoConfig.for_model(**config)
        except Exception as e:
            config = HFConfig.for_model(**config)
    return config


class AzerosConfig(BaseConfig):
    model_type = "azeros"
    use_flash_attn: bool = False
    exclude_from_checkpoint: list = None
    paraling_encoder_projector_ds_rate: int = 4
    speech_encoder_projector_ds_rate: int = 4

    def __init__(
        self,
        llm_config: dict | PretrainedConfig | None = None,
        speech_encoder_config: dict | BaseConfig | PretrainedConfig | None = None,
        paraling_encoder_config: dict | BaseConfig | PretrainedConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.speech_encoder_config = load_config(speech_encoder_config)
        self.paraling_encoder_config = load_config(paraling_encoder_config)
        self.llm_config = load_config(llm_config)
