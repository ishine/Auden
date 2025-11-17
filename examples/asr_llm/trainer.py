"""Audio-LLM training loop built on BaseTrainer.

This trainer wires the Audio-LLM model with the Lhotse-based datamodule and logs
token-level loss and accuracy on validation. See examples/asr_llm/configs for details.
"""

import logging
import random

import torch

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class AsrLLMTrainer(BaseTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        prompt_file = cfg.prompt_file
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_list = [line.strip() for line in f if line.strip()]

    def _forward_one_batch(self, batch: dict, is_training: bool):
        device = self.device
        feature = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        texts = supervisions["text"]
        audio_token = self.cfg.model.audio_token
        batch_size = len(texts)
        messages = []
        for text in texts:
            prompt = random.choice(self.prompt_list)
            message = [
                {"role": "user", "content": f"{audio_token} {prompt}"},
                {"role": "assistant", "content": text},
            ]
            messages.append(message)
        with torch.set_grad_enabled(is_training):
            model_outputs, acc = self.model(
                x=feature,
                x_lens=feature_lens,
                messages=messages,
                pack_sequences=False,
            )
            loss = model_outputs.loss

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = sum(len(text) for text in messages)
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", batch_size, normalization="sum")
        info.set_value("loss", loss.detach().cpu().item(), normalization="frame_avg")
        info.set_value("acc", acc, normalization="sample_avg")

        return loss, info

    def validate(self, epoch: int):
        """
        Validation is provided by BaseTrainer.

        Override in a subclass if you need ASR-LLM specific validation logic
        (e.g., generation-based metrics like WER/CER).
        """
        return super().validate(epoch)
