from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel as HFModel
from transformers import AutoTokenizer as HFTokenizer

from ...auto.auto_model import AutoConfig, AutoModel
from ...models.zipformer.utils.padding import make_pad_mask
from ...utils.dist_utils import all_gather_embeddings
from .loss import InfoNCELoss, SigLIPLoss
from .model_config import ClapConfig


class ClapModel(nn.Module):
    """CLAP-style audio–text model.

    Components:
      - Audio encoder (built via AutoModel from ``config.audio_encoder_config``)
      - Text encoder (HF model built from ``config.text_encoder_config``)
      - Two projection heads into a shared embedding space (``shared_emb_dim``)
      - Contrastive loss (InfoNCE or SigLIP) controlled by ``config.loss``

    Public API:
      - ``encode_audio(x, x_lens) -> (B, D)``: pooled and normalized audio embeddings
      - ``encode_text(text: List[str]) -> (B, D)``: normalized text embeddings
      - ``forward(x, x_lens, text, gather_embeddings=False)``: training step returning
        loss and per-modality embeddings
      - ``generate(input, text)``: return similarity matrices for inference/eval
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ):
        """Load CLAP model from directory or weight file, supports .pt and .safetensors."""

        # Support HF Hub repo IDs
        if not os.path.exists(model_path):
            model_path = AutoModel._download_from_hub(model_path)
        # Resolve model_dir and candidate weight file
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
        else:
            weight_path = model_path
            model_dir, _ = os.path.split(model_path)

        # Load config and construct model
        config = AutoConfig.from_pretrained(model_dir)
        tokenizer = HFTokenizer.from_pretrained(model_dir)
        model = cls(config, tokenizer)

        # Load weights if present
        if weight_path and os.path.exists(weight_path):
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

    def __init__(self, config: ClapConfig, tokenizer):
        """Initialize the CLAP model from a configuration and tokenizer.

        Args:
            config: ``ClapConfig`` containing encoder configs and projection sizes.
            tokenizer: HF tokenizer compatible with the text encoder.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.text_encoder = HFModel.from_config(
            self.config.text_encoder_config, add_pooling_layer=False
        )
        self.text_encoder_dim = self.config.text_encoder_config.hidden_size

        self.audio_encoder = AutoModel.from_config(config.audio_encoder_config)
        self.audio_encoder_dim = self.audio_encoder.encoder_out_dim

        # clap initialization
        self.shared_emb_dim = config.shared_emb_dim
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder_dim, self.shared_emb_dim),
            nn.ReLU(),
            nn.Linear(self.shared_emb_dim, self.shared_emb_dim),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_encoder_dim, self.shared_emb_dim),
            nn.ReLU(),
            nn.Linear(self.shared_emb_dim, self.shared_emb_dim),
        )

        self.temp = config.temp
        self.embed_reg = config.embed_reg

        # Loss selection based on config.loss
        loss_name = str(getattr(config, "loss", "info-nce")).lower()
        if loss_name in {"info-nce", "infonce", "nce"}:
            self.criterion = InfoNCELoss()
        elif loss_name in {"siglip", "sigmoid"}:
            self.criterion = SigLIPLoss()
        else:
            self.criterion = InfoNCELoss()

    def encode_audio(self, x, x_lens):
        """Encode audio features to a single embedding per utterance.

        Args:
            x: Tensor of shape (B, T, C) - features (e.g., fbank)
            x_lens: Tensor of shape (B,) - valid frame counts per sample

        Returns:
            L2-normalized audio embeddings of shape (B, shared_emb_dim)
        """
        output = self.audio_encoder(x, x_lens)
        audio_feats = self.audio_proj(output["encoder_out"])
        padding_mask = make_pad_mask(
            output["encoder_out_lens"], max_len=audio_feats.size(1)
        ).to(audio_feats.device)
        audio_feats = audio_feats.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        frame_counts = (~padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        audio_embeds = audio_feats.sum(dim=1) / frame_counts
        return F.normalize(audio_embeds, dim=-1)

    def encode_text(self, text):
        """
        Args:
            text: List[str]
        Returns:
            text_embeds: Tensor [B, embed_dim]
        """
        device = next(self.parameters()).device  # infer from model
        encoded = self.tokenizer(
            text, padding="longest", truncation=True, max_length=30, return_tensors="pt"
        ).to(device)
        text_output = self.text_encoder(
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask
        )[0]
        text_embeds = text_output[:, 0, :]  # [cls] token
        return F.normalize(self.text_proj(text_embeds), dim=-1)

    def forward(
        self, x, x_lens, text, gather_embeddings=False, return_dict: bool = True
    ):
        """Compute contrastive loss and return embeddings for both modalities.

        Args:
            x: (B, T, C) audio features
            x_lens: (B,) valid frame counts
            text: List[str] tokenized by the internal tokenizer
            gather_embeddings: if True and DDP is initialized, gather embeddings across ranks

        Returns:
            loss: scalar training loss
            audio_embeds: (B, D) normalized audio embeddings
            text_embeds: (B, D) normalized text embeddings
        """
        audio_embeds = self.encode_audio(x, x_lens)
        text_embeds = self.encode_text(text)

        # Gather embeddings across all processes if distributed
        if self.training and dist.is_initialized() and gather_embeddings:
            audio_embeds_all = all_gather_embeddings(audio_embeds)
            text_embeds_all = all_gather_embeddings(text_embeds)
        else:
            audio_embeds_all = audio_embeds
            text_embeds_all = text_embeds

        # Construct similarity and target
        sim_targets = torch.eye(
            audio_embeds_all.size(0), device=audio_embeds_all.device
        )

        sim_a2t = audio_embeds_all @ text_embeds_all.T / self.temp
        sim_t2a = text_embeds_all @ audio_embeds_all.T / self.temp

        loss = self.criterion(sim_a2t, sim_t2a, sim_targets)
        if self.embed_reg:
            loss += torch.mean(torch.abs(audio_embeds_all)) / torch.sqrt(
                torch.sum(audio_embeds_all**2)
            ) + torch.mean(torch.abs(text_embeds_all)) / torch.sqrt(
                torch.sum(text_embeds_all**2)
            )
        if return_dict:
            return {
                "loss": loss,
                "audio_embeds": audio_embeds,
                "text_embeds": text_embeds,
            }
        else:
            return loss, audio_embeds, text_embeds

    def generate(self, input, text):
        """Generate cross-modal similarities for inference/evaluation.

        Args:
            input: either (features, feature_lens) or raw input consumable by the audio encoder
            text: List[str]

        Returns:
            sim_a2t: (B, M) audio-to-text similarity matrix
            sim_t2a: (M, B) text-to-audio similarity matrix
        """
        # Handle flexible input
        if isinstance(input, tuple) and len(input) == 2:
            x, x_lens = input
        else:
            x, x_lens = self.audio_encoder.extract_feature(input)
        # Encode audio and text inputs
        audio_embeds = self.encode_audio(x, x_lens)
        text_embeds = self.encode_text(text)

        # Compute similarity scores (audio-to-text and text-to-audio)
        sim_a2t = audio_embeds @ text_embeds.T
        sim_t2a = text_embeds @ audio_embeds.T

        return sim_a2t, sim_t2a
