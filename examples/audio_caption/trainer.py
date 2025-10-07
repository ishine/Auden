"""Audio Caption training loop built on BaseTrainer.

This trainer wires the audio-caption model with the Lhotse-based datamodule.
It reports captioning loss during training.
"""

import logging

import torch

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class AudioCaptionTrainer(BaseTrainer):
    def _forward_one_batch(self, batch: dict, is_training: bool):
        device = self.device
        feature = batch["inputs"]  # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        text = supervisions["caption"]
        batch_size = len(text)
        with torch.set_grad_enabled(is_training):
            # The audio-caption model should return cross-entropy (or similar) loss
            loss = self.model(
                x=feature,
                x_lens=feature_lens,
                text=text,
                parallel_decoding_prob=(
                    self.cfg.trainer.parallel_decoding_prob if is_training else 0.0
                ),
                max_length=self.cfg.trainer.max_length,
            )

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = (feature_lens // 4).sum().item()
        num_samples = batch_size
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", num_samples, normalization="sum")
        info.set_value(
            "caption_loss",
            loss.detach().cpu().item(),
            normalization="sample_avg",
        )
        return loss, info

    def validate(self, epoch: int):
        """
        Validation is provided by BaseTrainer.

        Override in a subclass if you need audio-caption specific validation logic
        (e.g., decoding and computing BLEU/CIDEr or other captioning metrics).
        """
        return super().validate(epoch)
