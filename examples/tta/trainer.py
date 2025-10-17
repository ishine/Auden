import torch

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class TtaTrainer(BaseTrainer):
    def _forward_one_batch(self, batch: dict, is_training: bool):
        device = self.device
        feature = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        # Extract required texts and optional languages from supervisions
        source_texts = supervisions.get("source_text")
        target_texts = supervisions.get("target_text")
        source_language = supervisions.get("source_language", None)
        target_language = supervisions.get("target_language", None)

        batch_idx_train = self.global_step

        num_frames = (feature_lens // 4).sum().item()
        # For TTA, training may include multiple objectives depending on config
        # We expect the underlying model to return a tuple of losses when applicable:
        #   simple_loss, pruned_loss, attention_loss, s2t_align_loss
        forward_attention_decoder = False
        forward_s2t_alignment = False
        if batch_idx_train > self.cfg.trainer.forward_attention_decoder_step:
            forward_attention_decoder = True
        if batch_idx_train > self.cfg.trainer.forward_s2t_alignment_step:
            forward_s2t_alignment = True

        with torch.set_grad_enabled(is_training):
            simple_loss, pruned_loss, attention_loss, s2t_align_loss = self.model(
                x=feature,
                x_lens=feature_lens,
                source_texts=source_texts,
                target_texts=target_texts,
                source_language=source_language,
                target_language=target_language,
                prune_range=self.cfg.trainer.prune_range,
                am_scale=self.cfg.trainer.am_scale,
                lm_scale=self.cfg.trainer.lm_scale,
                forward_attention_decoder=forward_attention_decoder,
                forward_s2t_alignment=forward_s2t_alignment,
            )

            loss = 0.0

            s = self.cfg.trainer.simple_loss_scale
            warm_step = self.cfg.trainer.rnnt_warm_step
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
            loss += simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

            if attention_loss:
                loss += self.cfg.trainer.attention_loss_scale * attention_loss

            if s2t_align_loss:
                loss += self.cfg.trainer.s2t_align_loss_scale * s2t_align_loss

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", len(feature_lens), normalization="sum")
        info.set_value("loss", loss.detach().cpu().item(), normalization="frame_avg")
        info.set_value(
            "simple_loss", simple_loss.detach().cpu().item(), normalization="frame_avg"
        )
        info.set_value(
            "pruned_loss", pruned_loss.detach().cpu().item(), normalization="frame_avg"
        )
        if attention_loss:
            info.set_value(
                "attention_loss",
                attention_loss.detach().cpu().item(),
                normalization="frame_avg",
            )
        if s2t_align_loss:
            info.set_value(
                "s2t_align_loss",
                s2t_align_loss.detach().cpu().item(),
                normalization="sample_avg",
            )
        return loss, info

    def validate(self, epoch: int):
        return super().validate(epoch)
