"""Trainer for voice multitask model."""

import logging

import torch

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class VoiceTrainer(BaseTrainer):
    """
    Trainer for voice multitask model with 4 classification heads:
    - Speaker ID
    - Emotion
    - Gender
    - Age
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Task weights for loss combination
        self.task_weights = {
            "id": cfg.trainer.get("weight_id", 1.0),
            "emotion": cfg.trainer.get("weight_emotion", 2.5),
            "gender": cfg.trainer.get("weight_gender", 3.0),
            "age": cfg.trainer.get("weight_age", 3.0),
        }
        logging.info(f"Task weights: {self.task_weights}")

    def _forward_one_batch(self, batch: dict, is_training: bool, return_logits=False):
        """
        Forward pass for one batch.

        Args:
            batch: Dictionary containing inputs and supervisions
            is_training: Whether in training mode
            return_logits: Whether to return logits (for validation)

        Returns:
            loss: Total weighted loss
            info: MetricsTracker with loss and accuracy metrics
            logits_list (optional): List of logits for each task
        """
        device = self.device
        feature = batch["inputs"]
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        batch_size = feature.size(0)

        # Extract labels for all tasks
        tags = {
            "id": supervisions["id"],
            "emotion": supervisions["emotion"],
            "gender": supervisions["gender"],
            "age": supervisions["age"],
        }

        with torch.set_grad_enabled(is_training):
            outputs = self.model(
                x=feature,
                x_lens=feature_lens,
                tags=tags,
                return_dict=True,
            )
            loss_list = outputs["loss_list"]
            logits_list = outputs["logits_list"]
            top1_acc_list = outputs["top1_acc_list"]
            top5_acc_list = outputs["top5_acc_list"]

            # Combine losses with weights
            weighted_loss = 0.0
            task_names = ["id", "emotion", "gender", "age"]
            for task, loss in zip(task_names, loss_list):
                weighted_loss += self.task_weights[task] * loss

            total_loss = weighted_loss

        assert total_loss.requires_grad == is_training

        info = MetricsTracker()
        num_frames = (feature_lens // 4).sum().item()
        num_samples = batch_size
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", num_samples, normalization="sum")
        info.set_value(
            "loss",
            total_loss.detach().cpu().item() / num_samples,
            normalization="sample_avg",
        )

        # Log individual task losses and accuracies
        for i, task in enumerate(task_names):
            loss_val = loss_list[i].detach().cpu().item() / num_samples
            info.set_value(f"loss_{task}", loss_val, normalization="sample_avg")
            info.set_value(
                f"top1_acc_{task}", top1_acc_list[i], normalization="sample_avg"
            )
            if i == 0:  # Only speaker ID has top5 accuracy
                info.set_value(
                    f"top5_acc_{task}", top5_acc_list[i], normalization="sample_avg"
                )

        if not return_logits:
            return total_loss, info
        else:
            return total_loss, info, logits_list

    def validate(self, epoch):
        """Run validation on all validation sets."""
        self.model.eval()
        with torch.no_grad():
            for valid_name, valid_dl in zip(
                self.data_module.valid_names, self.data_module.valid_dls
            ):
                tot_info = MetricsTracker()
                for batch_idx, batch in enumerate(valid_dl):
                    loss, info, logits_list = self._forward_one_batch(
                        batch=batch, is_training=False, return_logits=True
                    )

                    assert loss.requires_grad is False
                    tot_info.update(info)

                if self.world_size > 1:
                    tot_info.reduce(loss.device)

                if self.rank == 0:
                    logging.info(
                        f"Epoch {epoch}, global batch {self.global_step}, validation {valid_name}: {tot_info}"
                    )
                    if self.tb_writer is not None:
                        tot_info.write_summary(
                            self.tb_writer,
                            f"train/valid_{valid_name}_",
                            self.global_step,
                        )

        self.model.train()
