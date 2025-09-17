import torch

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class AsrTrainer(BaseTrainer):
    def _forward_one_batch(self, batch: dict, is_training: bool, return_emb=False):
        device = self.device
        feature = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        batch_idx_train = self.global_step
        warm_step = self.cfg.trainer.rnnt_warm_step

        texts = batch["supervisions"]["text"]
        batch_size = len(texts)

        with torch.set_grad_enabled(is_training):
            simple_loss, pruned_loss, ctc_loss = self.model(
                x=feature,
                x_lens=feature_lens,
                texts=texts,
                prune_range=self.cfg.trainer.prune_range,
                am_scale=self.cfg.trainer.am_scale,
                lm_scale=self.cfg.trainer.lm_scale,
            )

            loss = 0.0

            if simple_loss:
                s = self.cfg.trainer.simple_loss_scale
                # take down the scale on the simple loss from 1.0 at the start
                # to simple_loss scale by warm_step.
                simple_loss_scale = (
                    s
                    if batch_idx_train >= warm_step
                    else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
                )
                pruned_loss_scale = (
                    1.0
                    if batch_idx_train >= warm_step
                    else 0.1 + 0.9 * (batch_idx_train / warm_step)
                )
                loss += (
                    simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
                )

            if ctc_loss:
                loss += self.cfg.trainer.ctc_loss_scale * ctc_loss

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = (feature_lens // 4).sum().item()
        num_samples = batch_size
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", num_samples, normalization="sum")

        # Note: We use reduction=sum while computing the loss.
        info.set_value(
            "loss", loss.detach().cpu().item() / num_frames, normalization="frame_avg"
        )
        if simple_loss:
            info.set_value(
                "simple_loss",
                simple_loss.detach().cpu().item() / num_frames,
                normalization="frame_avg",
            )
            info.set_value(
                "pruned_loss",
                pruned_loss.detach().cpu().item() / num_frames,
                normalization="frame_avg",
            )
        if ctc_loss:
            info.set_value(
                "ctc_loss",
                ctc_loss.detach().cpu().item() / num_frames,
                normalization="frame_avg",
            )

        return loss, info

    def validate(self, epoch: int):
        """
        Validation is provided by BaseTrainer.

        Override in a subclass if you need ASR-specific validation logic
        (e.g., decoding to compute WER/CER, beam search, or task-specific metrics).
        """
        return super().validate(epoch)
