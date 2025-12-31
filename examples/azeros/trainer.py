import torch

from auden.trainer.ddp_trainer import BaseTrainer
from auden.utils.metric_tracker import MetricsTracker


class AzerosTrainer(BaseTrainer):
    def _forward_one_batch(self, batch: dict, is_training: bool):
        device = self.device
        feature = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        # Extract required texts and optional languages from supervisions
        responses = supervisions.get("response")
        instructions = supervisions.get("instruction")
        audio_token_wrapped = getattr(self.model, 'module', self.model).audio_token_wrapped

        messages = []
        for instruction, response in zip(instructions, responses):
            msg = [
                {"role": "user",
                 "content": f"{audio_token_wrapped} {instruction}".strip()},
                {"role": "assistant", "content": f"{response}"},
            ]
            messages.append(msg)

        num_frames = (feature_lens // 4).sum().item()

        with torch.set_grad_enabled(is_training):
            model_outputs, acc = self.model(
                x=feature,
                x_lens=feature_lens,
                messages=messages,
            )
            loss = model_outputs.loss

        assert loss.requires_grad == is_training

        info = MetricsTracker()
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", len(feature_lens), normalization="sum")
        info.set_value("loss", loss.detach().cpu().item(), normalization="frame_avg")
        info.set_value("acc", acc, normalization="sample_avg")
        return loss, info

    def validate(self, epoch: int):
        return super().validate(epoch)
