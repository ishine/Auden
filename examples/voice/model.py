"""Voice multitask model with 4 classification heads."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_config import VoiceMultitaskConfig

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel
from auden.models.audio_tag.model import compute_acc
from auden.models.audio_tag.utils import load_id2label
from auden.models.zipformer.utils.padding import make_pad_mask


class VoiceMultitaskModel(nn.Module):
    """
    Voice multitask model with 4 classification heads:
    - Speaker ID classification
    - Emotion classification
    - Gender classification
    - Age classification
    """

    @classmethod
    def from_pretrained(cls, model_path):
        """Load model from pretrained checkpoint."""
        if os.path.isdir(model_path):
            model_dir = model_path
            model_path = os.path.join(model_dir, "pretrained.pt")
        else:
            model_dir, _ = os.path.split(model_path)

        # Load config directly from saved config.json without AutoConfig registration
        import json

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Create VoiceMultitaskConfig directly from dict
        config = VoiceMultitaskConfig(**config_dict)

        # Load all id2label mappings
        id2label_id = load_id2label(Path(model_dir) / "id2label_id.json")
        id2label_emotion = load_id2label(Path(model_dir) / "id2label_emotion.json")
        id2label_gender = load_id2label(Path(model_dir) / "id2label_gender.json")
        id2label_age = load_id2label(Path(model_dir) / "id2label_age.json")

        model = cls(
            config, id2label_id, id2label_emotion, id2label_gender, id2label_age
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        return model

    def __init__(
        self,
        config,
        id2label_id,
        id2label_emotion,
        id2label_gender,
        id2label_age,
        pretrained_modules=None,
    ):
        super().__init__()
        self.config = config

        # Store all label mappings
        self.id2label_id = id2label_id
        self.id2label_emotion = id2label_emotion
        self.id2label_gender = id2label_gender
        self.id2label_age = id2label_age

        # Create reverse mappings
        self.label2id_id = {label: int(idx) for idx, label in id2label_id.items()}
        self.label2id_emotion = {
            label: int(idx) for idx, label in id2label_emotion.items()
        }
        self.label2id_gender = {
            label: int(idx) for idx, label in id2label_gender.items()
        }
        self.label2id_age = {label: int(idx) for idx, label in id2label_age.items()}

        # Number of classes for each task
        self.num_classes_id = len(self.id2label_id)
        self.num_classes_emotion = len(self.id2label_emotion)
        self.num_classes_gender = len(self.id2label_gender)
        self.num_classes_age = len(self.id2label_age)

        # Shared encoder
        self.encoder = AutoModel.from_config(self.config.encoder_config)

        # Get encoder output dimension
        encoder_dim = max(config.encoder_config.encoder_dim)

        # Four classification heads
        self.classifier_id = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, self.num_classes_id),
        )

        self.classifier_emotion = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, self.num_classes_emotion),
        )

        self.classifier_gender = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, self.num_classes_gender),
        )

        self.classifier_age = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, self.num_classes_age),
        )

        # Loss functions for each task (all single-label classification)
        self.criterion_id = torch.nn.CrossEntropyLoss(reduction="sum")
        self.criterion_emotion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.criterion_gender = torch.nn.CrossEntropyLoss(reduction="sum")
        self.criterion_age = torch.nn.CrossEntropyLoss(reduction="sum")

        if pretrained_modules:
            self.load_pretrained_modules(pretrained_modules)

    def load_pretrained_modules(self, pretrained_modules: dict):
        """Load pretrained weights for specified modules."""
        for module_name, path_or_name in pretrained_modules.items():
            if path_or_name is None:
                continue
            if module_name == "encoder":
                encoder = AutoModel.from_pretrained(path_or_name)
                self.encoder.load_state_dict(encoder.state_dict(), strict=True)
                logging.info(f"Loaded encoder from {path_or_name}")

    def tag2id(self, tag_strings, task):
        """Convert tag strings to class IDs for a specific task."""
        if task == "id":
            label2id = self.label2id_id
        elif task == "emotion":
            label2id = self.label2id_emotion
        elif task == "gender":
            label2id = self.label2id_gender
        elif task == "age":
            label2id = self.label2id_age
        else:
            raise ValueError(f"Unknown task: {task}")

        ids = []
        for tag_str in tag_strings:
            if tag_str in label2id:
                ids.append(label2id[tag_str])
            else:
                logging.warning(f"Unknown label '{tag_str}' for task '{task}', using 0")
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, x, x_lens, tags, return_dict: bool = True):
        """
        Forward pass for training.

        Args:
            x: A 3-D tensor of shape (N, T, C).
            x_lens: A 1-D tensor of shape (N,). Number of frames before padding.
            tags: Dict containing labels for each task, e.g.:
                  {'id': ['spk1', 'spk2'], 'emotion': ['happy', 'sad'],
                   'gender': ['male', 'female'], 'age': ['young', 'old']}

        Returns:
            Tuple of (loss_list, logits_list, top1_acc_list, top5_acc_list)
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        # Compute encoder outputs
        encoder_output = self.encoder(x, x_lens)
        encoder_out = encoder_output["encoder_out"]
        encoder_out_lens = encoder_output["encoder_out_lens"]

        # Forward all classifiers
        logits_id = self.forward_classifier(encoder_out, encoder_out_lens, "id")
        logits_emotion = self.forward_classifier(
            encoder_out, encoder_out_lens, "emotion"
        )
        logits_gender = self.forward_classifier(encoder_out, encoder_out_lens, "gender")
        logits_age = self.forward_classifier(encoder_out, encoder_out_lens, "age")

        # Compute losses and accuracies for each task
        loss_list = []
        logits_list = [logits_id, logits_emotion, logits_gender, logits_age]
        top1_acc_list = []
        top5_acc_list = []

        tasks = ["id", "emotion", "gender", "age"]
        criterions = [
            self.criterion_id,
            self.criterion_emotion,
            self.criterion_gender,
            self.criterion_age,
        ]

        for i, (task, criterion, logits) in enumerate(
            zip(tasks, criterions, logits_list)
        ):
            if task in tags:
                # Filter out missing labels, only compute loss on samples with labels
                valid_indices = []
                valid_labels = []
                for j, label in enumerate(tags[task]):
                    if label is not None and label != "None" and label != "Null":
                        valid_indices.append(j)
                        valid_labels.append(label)

                if valid_labels:  # If there are valid labels
                    valid_indices = torch.tensor(valid_indices, device=x.device)
                    valid_logits = logits[valid_indices]
                    targets = self.tag2id(valid_labels, task).to(x.device)
                    loss = criterion(valid_logits, targets)

                    # Convert targets to one-hot for accuracy computation
                    targets_onehot = F.one_hot(
                        targets, num_classes=logits.size(-1)
                    ).float()
                    top1_acc, top5_acc = compute_acc(valid_logits, targets_onehot)
                else:
                    # If no valid labels, use zero loss
                    loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                    top1_acc = 0.0
                    top5_acc = 0.0
            else:
                # If task labels not provided, use zero loss
                loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                top1_acc = 0.0
                top5_acc = 0.0

            loss_list.append(loss)
            top1_acc_list.append(top1_acc)
            top5_acc_list.append(top5_acc)

        if return_dict:
            return {
                "loss_list": loss_list,
                "logits_list": logits_list,
                "top1_acc_list": top1_acc_list,
                "top5_acc_list": top5_acc_list,
            }
        else:
            return loss_list, logits_list, top1_acc_list, top5_acc_list

    def forward_classifier(self, encoder_out, encoder_out_lens, task):
        """
        Forward pass through a specific classifier head.

        Args:
            encoder_out: A 3-D tensor of shape (N, T, C).
            encoder_out_lens: A 1-D tensor of shape (N,). Number of frames before padding.
            task: Task name ("id", "emotion", "gender", "age")

        Returns:
            A 2-D tensor of shape (N, num_classes).
        """
        if task == "id":
            classifier = self.classifier_id
        elif task == "emotion":
            classifier = self.classifier_emotion
        elif task == "gender":
            classifier = self.classifier_gender
        elif task == "age":
            classifier = self.classifier_age
        else:
            raise ValueError(f"Unknown task: {task}")

        logits = classifier(encoder_out)  # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens)
        logits[padding_mask] = 0  # mask the padding frames

        # Average pooling on the logits
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
        return logits

    def generate(self, input, task="id", topk=1):
        """
        Generate predictions for a specific task.

        Args:
            input: (x, x_lens) or list of file paths or raw waveforms
            task: Task name ("id", "emotion", "gender", "age")
            topk: Number of top predictions to return

        Returns:
            - labels (List[List[str]])
            - logits (Tensor): (N, topk)
            - probs  (Tensor): (N, topk)
        """
        # Print informative message for speaker identification task
        if task == "id":
            print(
                "⚠️  Speaker identification is a closed-set task. "
                "The model will output the most likely speaker seen in training data. "
                "If you wish to do speaker verification (open-set), "
                "directly use the encoder to compare embedding distance/similarity."
            )

        # Handle flexible input
        if isinstance(input, tuple) and len(input) == 2:
            x, x_lens = input
        else:
            x, x_lens = self.encoder.extract_feature(input)

        device = next(self.parameters()).device
        x = x.to(device)
        x_lens = x_lens.to(device)

        # Forward encoder
        encoder_output = self.encoder(x, x_lens)
        encoder_out = encoder_output["encoder_out"]
        encoder_out_lens = encoder_output["encoder_out_lens"]

        # Forward specific classifier
        logits_full = self.forward_classifier(
            encoder_out, encoder_out_lens, task
        )  # (N, num_classes)

        # Get id2label mapping for the task
        if task == "id":
            id2label = self.id2label_id
        elif task == "emotion":
            id2label = self.id2label_emotion
        elif task == "gender":
            id2label = self.id2label_gender
        elif task == "age":
            id2label = self.id2label_age
        else:
            raise ValueError(f"Unknown task: {task}")

        # Single-label classification
        probs_full = torch.softmax(logits_full, dim=-1)
        topk_probs, topk_indices = torch.topk(
            probs_full, k=min(topk, probs_full.size(-1)), dim=-1
        )
        topk_logits = torch.gather(logits_full, dim=1, index=topk_indices)

        labels = [
            [id2label[str(idx.item())] for idx in indices] for indices in topk_indices
        ]
        return labels, topk_logits, topk_probs
