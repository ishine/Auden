"""CLAP training loop built on BaseTrainer.

This trainer wires the CLAP model with the Lhotse-based datamodule and logs
batch-wise contrastive loss as well as retrieval metrics on validation.
See examples/clap/configs for configuration details.
"""

import logging

import torch

from auden.models.clap.utils import a2t_metric, t2a_metric
from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.dist_utils import ddp_all_gather_variable_tensor_to_rank0
from auden.utils.metric_tracker import MetricsTracker


class ClapTrainer(BaseTrainer):
    def _forward_one_batch(self, batch: dict, is_training: bool, return_emb=False):
        device = self.device
        feature = batch["inputs"]  # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        text = supervisions["caption"]
        batch_size = len(text)

        with torch.set_grad_enabled(is_training):
            loss, audio_embed, text_embed = self.model(
                x=feature,
                x_lens=feature_lens,
                text=text,
                gather_embeddings=(
                    self.cfg.trainer.get("gather_embeddings", False)
                    if is_training
                    else False
                ),
            )

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = (feature_lens // 4).sum().item()
        num_samples = batch_size
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", num_samples, normalization="sum")
        info.set_value(
            "clap_loss",
            loss.detach().cpu().item() / num_samples,
            normalization="sample_avg",
        )

        if not return_emb:
            (
                r1,
                _,
                _,
                _,
                _,
                _,
            ) = a2t_metric(
                audio_embed, text_embed
            )  # r1 within this batch
            info.set_value(
                "batch_a2t_r1", r1, normalization="sample_avg"
            )  # batch-wise R1
            return loss, info
        else:
            return loss, info, audio_embed.detach(), text_embed.detach()

    def validate(self, epoch):
        """Run the validation process."""
        self.model.eval()
        with torch.no_grad():
            for valid_name, valid_dl in zip(
                self.data_module.valid_names, self.data_module.valid_dls
            ):
                audio_embeds = []
                text_embeds = []
                tot_info = MetricsTracker()
                for batch in valid_dl:
                    loss, info, audio_embed, text_embed = self._forward_one_batch(
                        batch=batch, is_training=False, return_emb=True
                    )

                    assert loss.requires_grad is False
                    tot_info.update(info)
                    audio_embeds.append(audio_embed)
                    text_embeds.append(text_embed)

                if self.world_size > 1:
                    tot_info.reduce(loss.device)

                audio_embeds_all = torch.cat(audio_embeds, dim=0)
                text_embeds_all = torch.cat(text_embeds, dim=0)
                audio_embeds_all = ddp_all_gather_variable_tensor_to_rank0(
                    audio_embeds_all
                )
                text_embeds_all = ddp_all_gather_variable_tensor_to_rank0(
                    text_embeds_all
                )

                if self.rank == 0:
                    a2t_r1, a2t_r5, a2t_r10, _, _, _ = a2t_metric(
                        audio_embeds_all, text_embeds_all
                    )
                    tot_info.set_value("a2t_r1", a2t_r1, normalization="sample_avg")
                    tot_info.set_value("a2t_r5", a2t_r5, normalization="sample_avg")
                    tot_info.set_value("a2t_r10", a2t_r10, normalization="sample_avg")

                    t2a_r1, t2a_r5, t2a_r10, _, _, _ = t2a_metric(
                        text_embeds_all, audio_embeds_all
                    )
                    tot_info.set_value("t2a_r1", t2a_r1, normalization="sample_avg")
                    tot_info.set_value("t2a_r5", t2a_r5, normalization="sample_avg")
                    tot_info.set_value("t2a_r10", t2a_r10, normalization="sample_avg")

                if self.rank == 0:
                    logging.info(
                        f"Epoch {epoch}, global batch {self.global_step}, validation: {tot_info}"
                    )
                    if self.tb_writer is not None:
                        tot_info.write_summary(
                            self.tb_writer,
                            f"train/valid_{valid_name}_",
                            self.global_step,
                        )

        self.model.train()
