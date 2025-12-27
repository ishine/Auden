from ...auto.auto_config import AutoConfig
from ..base.model_config import BaseConfig


class AudioTagConfig(BaseConfig):
    model_type = "audio-tag"
    loss = "ce"

    def __init__(self, encoder_config=None, **kwargs):
        if encoder_config is None:
            encoder_config = AutoConfig.for_model("zipformer")

        if isinstance(encoder_config, dict):
            encoder_config = AutoConfig.for_model(**encoder_config)

        self.encoder_config = encoder_config
        super().__init__(**kwargs)
