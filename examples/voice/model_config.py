"""Configuration for voice multitask model."""

from auden.models.base.model_config import BaseConfig
from auden.auto.auto_config import AutoConfig


class VoiceMultitaskConfig(BaseConfig):
    """
    Configuration for voice multitask model with 4 classification heads:
    - Speaker ID
    - Emotion
    - Gender
    - Age
    """

    model_type = "voice-multitask"
    fuse_encoder = False

    def __init__(self, encoder_config=None, **kwargs):
        if encoder_config is None:
            encoder_config = AutoConfig.for_model("zipformer")

        if isinstance(encoder_config, dict):
            encoder_config = AutoConfig.for_model(**encoder_config)

        self.encoder_config = encoder_config
        super().__init__(**kwargs)

