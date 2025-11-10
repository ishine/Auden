"""Voice multitask training entry script using Hydra and torch.distributed.

This script trains a voice multitask model with 4 classification heads:
speaker ID, emotion, gender, and age.

"""

import logging
import os

import hydra
import torch
import torch.distributed as dist

# Import local modules directly
from data_module import VoiceDatamodule
from model import VoiceMultitaskModel
from model_config import VoiceMultitaskConfig
from omegaconf import DictConfig, OmegaConf
from trainer import VoiceTrainer

from auden.auto.auto_config import AutoConfig  # Only for encoder config

# Import utility functions
from auden.auto.auto_model import AutoModel
from auden.models.audio_tag.utils import load_id2label, save_id2label


def load_pretrained_encoder(cfg: DictConfig):
    """load pretrained encoder config and module.
    If pretrained_encoder is provided, load config from there.
    Otherwise, build an empty encoder config from the encoder type.

    Args:
        cfg: The model section, e.g. ``cfg.model`` with fields:
             - encoder: string identifier (e.g., "zipformer")
             - pretrained_encoder: optional path/identifier

    Returns:
        (encoder_config, pretrained_encoder)
    """
    if cfg.get("pretrained_encoder") is not None:
        try:
            pretrained_encoder = AutoModel.from_pretrained(cfg.pretrained_encoder)
            encoder_config = pretrained_encoder.config
            logging.info(f"Loaded pretrained encoder from {cfg.pretrained_encoder}")
            return encoder_config, pretrained_encoder
        except Exception:
            raise ValueError(f"Failed to load encoder from {cfg.pretrained_encoder}")
    else:
        try:
            encoder_config = AutoConfig.for_model(cfg.encoder)
            logging.info(f"Built encoder config for {cfg.encoder}")
        except Exception:
            raise ValueError(f"Failed to build encoder config for {cfg.encoder}")
        pretrained_encoder = None
        return encoder_config, pretrained_encoder


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):

    # Log full config for reproducibility (align style with tta)
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # DDP environment setup
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # Create experiment directory
    if cfg.get("exp_dir"):
        os.makedirs(cfg.exp_dir, exist_ok=True)

    encoder_config, pretrained_encoder = load_pretrained_encoder(cfg.model)

    # Load all id2label mappings for 4 tasks
    id2label_id = load_id2label(cfg.model.id2label_json_id)
    id2label_emotion = load_id2label(cfg.model.id2label_json_emotion)
    id2label_gender = load_id2label(cfg.model.id2label_json_gender)
    id2label_age = load_id2label(cfg.model.id2label_json_age)

    # Create model config and model directly
    config = VoiceMultitaskConfig(encoder_config=encoder_config)
    model = VoiceMultitaskModel(
        config=config,
        id2label_id=id2label_id,
        id2label_emotion=id2label_emotion,
        id2label_gender=id2label_gender,
        id2label_age=id2label_age,
    )

    # Load pretrained encoder weights if provided
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)
        logging.info(
            f"Loaded pretrained encoder weights from {cfg.model.pretrained_encoder}"
        )

    # Optionally freeze encoder
    if cfg.model.get("freeze_encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False
        num_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        logging.info(f"[voice.train] Froze encoder weights ({num_params:.2f} M params)")

    # Save config and id2label
    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        save_id2label(id2label_id, cfg.exp_dir, name="id2label_id.json")
        save_id2label(id2label_emotion, cfg.exp_dir, name="id2label_emotion.json")
        save_id2label(id2label_gender, cfg.exp_dir, name="id2label_gender.json")
        save_id2label(id2label_age, cfg.exp_dir, name="id2label_age.json")

    # Datamodule
    data_module = VoiceDatamodule(cfg.data)

    # Trainer
    trainer = VoiceTrainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
