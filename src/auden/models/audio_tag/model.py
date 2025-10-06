from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...auto.auto_config import AutoConfig
from ...auto.auto_model import MODEL_MAPPING, AutoModel
from ..zipformer.utils.padding import make_pad_mask


def load_id2label(path: str | os.PathLike) -> dict:
    import json

    with open(path, "r") as f:
        return json.load(f)


class AudioTagModel(nn.Module):
    """Audio tagging model with a pluggable encoder and linear classifier.

    Typical usage:
        - Build config and id2label mapping, then initialize: ``model = AudioTagModel(config, id2label)``
        - Use ``forward`` for training (returns loss/logits/top1/top5); use ``generate`` for inference.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ):
        """Load an AudioTag model from a directory or weight file.

        Resolves the directory, loads ``config.json`` and ``id2label.json``,
        instantiates the model, then loads weights from ``pretrained.pt``/``model.pt``
        or the given ``.pt`` file.
        """
        # Resolve model_dir and weight path
        if os.path.isdir(model_path):
            model_dir = model_path
            weight_path = None
            for ext in (".safetensors", ".pt"):
                for name in ("pretrained", "model"):
                    p = os.path.join(model_dir, f"{name}{ext}")
                    if os.path.exists(p):
                        weight_path = p
                        break
                if weight_path is not None:
                    break
            if weight_path is None:
                raise FileNotFoundError(
                    f"Expected one of ['pretrained.safetensors','model.safetensors','pretrained.pt','model.pt'] under {model_dir}"
                )
        else:
            weight_path = model_path
            model_dir, _ = os.path.split(model_path)

        config = AutoConfig.from_pretrained(model_dir)
        id2label_json = Path(model_dir) / "id2label.json"
        id2label = load_id2label(id2label_json)
        model = cls(config, id2label)

        ext = os.path.splitext(weight_path)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file as safe_load_file

            device_arg = (
                str(map_location)
                if isinstance(map_location, torch.device)
                else map_location
            )
            state_obj = safe_load_file(weight_path, device=device_arg)
        else:
            state_obj = torch.load(weight_path, map_location=map_location)
        state_dict = (
            state_obj["state_dict"]
            if isinstance(state_obj, dict) and "state_dict" in state_obj
            else state_obj
        )
        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model

    def __init__(self, config, id2label):
        """Construct the audio tagging model.

        Args:
            config: ``AudioTagConfig`` with an ``encoder_config``.
            id2label: Dict mapping class index (str) to label string.
        """
        super().__init__()
        self.config = config
        self.id2label = id2label
        self.label2id = {label: int(idx) for idx, label in id2label.items()}
        self.num_classes = len(self.id2label)
        self.loss_type = config.loss

        self.encoder = AutoModel.from_config(self.config.encoder_config)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(max(config.encoder_config.encoder_dim), self.num_classes),
        )

        if self.loss_type == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
        elif self.loss_type == "ce":
            self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if getattr(self.config, "fuse_encoder", False):
            self.encoder_fusion_weights = nn.Parameter(
                torch.zeros(len(self.config.num_encoder_layers))
            )
        else:
            self.encoder_fusion_weights = None

    def tag2multihot(self, tag_strings: List[str]) -> torch.Tensor:
        multihot = torch.zeros(
            (len(tag_strings), self.num_classes), dtype=torch.float32
        )
        for i, tag_str in enumerate(tag_strings):
            tags = tag_str.split(";")
            for tag in tags:
                multihot[i, int(self.label2id[tag])] = 1.0
        return multihot

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor, tags: List[str]):
        """Compute classification loss and logits.

        Args:
            x: Input features of shape (N, T, C).
            x_lens: Number of valid frames per sample, shape (N,).
            tags: List of tag strings per sample; multilabel separated by ';'.

        Returns:
            loss, logits, top1_acc, top5_acc
        """
        targets = self.tag2multihot(tags).to(x.device)
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        encoder_output = self.encoder(x, x_lens)

        if self.encoder_fusion_weights is not None:
            fusion_weights = F.softmax(self.encoder_fusion_weights, dim=0).view(
                -1, 1, 1, 1
            )
            encoder_out = (encoder_output.encoder_out_full * fusion_weights).sum(dim=0)
        else:
            encoder_out = encoder_output.encoder_out

        logits = self.forward_classifier(
            encoder_out=encoder_out, encoder_out_lens=encoder_output.encoder_out_lens
        )

        loss = self.criterion(logits, targets)

        top1_acc, top5_acc = compute_acc(logits, targets)

        return loss, logits, top1_acc, top5_acc

    def forward_classifier(
        self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor
    ):
        """Average-pool logits over time with padding mask, return (N, num_classes)."""
        logits = self.classifier(encoder_out)  # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
        return logits

    def generate(
        self,
        input,
        return_full_logits: bool = False,
        threshold: float = 0.0,
        topk: int = 1,
    ):
        """Inference helper.

        Accepts either (features, feature_lens) or raw inputs (wav paths / mono waveforms).
        If ``return_full_logits`` is True, returns full logits; else returns top-k labels/logits/probs.
        """
        if isinstance(input, tuple) and len(input) == 2:
            x, x_lens = input
        else:
            x, x_lens = self.encoder.extract_feature(input)

        device = next(self.parameters()).device
        x = x.to(device)
        x_lens = x_lens.to(device)
        encoder_output = self.encoder(x, x_lens)
        if self.encoder_fusion_weights is not None:
            fusion_weights = F.softmax(self.encoder_fusion_weights, dim=0).view(
                -1, 1, 1, 1
            )
            encoder_out = (encoder_output.encoder_out_full * fusion_weights).sum(dim=0)
        else:
            encoder_out = encoder_output.encoder_out

        logits_full = self.forward_classifier(
            encoder_out, encoder_output.encoder_out_lens
        )

        if return_full_logits:
            return logits_full

        if self.loss_type == "bce":
            probs_full = torch.sigmoid(logits_full)
            topk_probs, topk_indices = torch.topk(
                probs_full, k=min(topk, probs_full.size(-1)), dim=-1
            )
            topk_logits = torch.gather(logits_full, dim=1, index=topk_indices)

            labels = [
                [
                    self.id2label[str(idx.item())]
                    for idx, prob in zip(indices, probs)
                    if prob.item() > threshold
                ]
                for indices, probs in zip(topk_indices, topk_probs)
            ]
            return labels, topk_logits, topk_probs
        else:
            probs_full = torch.softmax(logits_full, dim=-1)
            topk_probs, topk_indices = torch.topk(
                probs_full, k=min(topk, probs_full.size(-1)), dim=-1
            )
            topk_logits = torch.gather(logits_full, dim=1, index=topk_indices)

            labels = [
                [self.id2label[str(idx.item())] for idx in indices]
                for indices in topk_indices
            ]
            return labels, topk_logits, topk_probs


def compute_acc(logits: torch.Tensor, targets: torch.Tensor):
    """Compute top-1/top-5 accuracy for single- and multi-label cases."""
    with torch.no_grad():
        if targets.ndim == 2 and targets.sum(dim=1).max() > 1:  # multi-label
            probs = torch.sigmoid(logits)
            top1 = probs.argmax(dim=-1)
            eq1 = targets[torch.arange(targets.size(0)), top1] > 0
            top1_acc = eq1.float().mean().item()

            top5_probs, top5_idx = torch.topk(probs, k=min(5, probs.size(-1)), dim=-1)
            eq5 = torch.gather(targets, 1, top5_idx).max(dim=1).values > 0
            top5_acc = eq5.float().mean().item()
        else:  # single-label
            probs = torch.softmax(logits, dim=-1)
            topk = min(5, probs.size(-1))
            top1 = probs.argmax(dim=-1)
            eq1 = targets.argmax(dim=-1) == top1
            top1_acc = eq1.float().mean().item()

            top5 = torch.topk(probs, k=topk, dim=-1).indices
            eq5 = (targets.argmax(dim=-1).unsqueeze(1) == top5).any(dim=1)
            top5_acc = eq5.float().mean().item()
    return top1_acc, top5_acc
