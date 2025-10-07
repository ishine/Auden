"""Audio Caption training entry script using Hydra and torch.distributed.

This mirrors the CLAP example structure but targets audio-to-text captioning.
It wires model, datamodule and trainer, and supports single-GPU and DDP via torchrun.

Usage:
    # single GPU
    python examples/audio_caption/train.py exp_dir=./exp/your_exp_dir

    # multi-GPU (DDP)
    torchrun --nproc_per_node=8 examples/audio_caption/train.py exp_dir=./exp/your_exp_dir

Config highlights (see examples/audio_caption/configs/train.yaml):
- cfg.model.model_type: "audio_caption"
- cfg.model.audio_encoder: { model_type, pretrained_model?, frozen }
- cfg.model.tokenizer: HF tokenizer id (e.g., "facebook/bart-base")
- cfg.data: Lhotse-based datamodule configs (see examples/audio_caption/configs/)
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AudioCaptionDatamodule
from omegaconf import DictConfig, OmegaConf
from trainer import AudioCaptionTrainer as Trainer
from transformers import AutoTokenizer as HFTokenizer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel


def load_pretrained_audio_encoder(cfg: DictConfig):
    """Build audio encoder config and optionally load a pretrained module.

    Args:
        cfg: The audio encoder section, e.g. ``cfg.model.audio_encoder`` with fields:
             - model_type: string identifier (e.g., "zipformer")
             - pretrained_model: optional path/identifier

    Returns:
        (encoder_config, encoder_module_or_None)
    """
    if cfg.get("model_type") == "zipformer":
        from auden.models.zipformer.model import ZipformerEncoderModel
        from auden.models.zipformer.model_config import ZipformerConfig

        pretrained = cfg.get("pretrained_model")
        if pretrained:
            encoder_model = ZipformerEncoderModel.from_pretrained(pretrained)
            encoder_config = encoder_model.config
        else:
            encoder_config = ZipformerConfig()
            encoder_model = None
    else:
        raise ValueError(f"Unsupported encoder model type: {cfg.get('model_type')}")
    return encoder_config, encoder_model


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # DDP env
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # exp dir
    if cfg.get("exp_dir"):
        os.makedirs(cfg.exp_dir, exist_ok=True)

    # load pretrained audio encoder
    audio_encoder_config, pretrained_audio_encoder = load_pretrained_audio_encoder(
        cfg.model.audio_encoder
    )

    # Assemble AudioCaption config and model
    config = AutoConfig.for_model(
        cfg.model.model_type,
        audio_encoder_config=audio_encoder_config,
    )
    # Build tokenizer for the custom decoder
    tokenizer_name_or_path = cfg.get("tokenizer", "facebook/bart-base")
    tokenizer = HFTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModel.from_config(config, tokenizer=tokenizer)

    # load pretrained encoder weights
    if pretrained_audio_encoder is not None:
        model.audio_encoder.load_state_dict(
            pretrained_audio_encoder.state_dict(), strict=True
        )
        if rank == 0:
            src = cfg.model.audio_encoder.get("pretrained_model")
            num_params = sum(p.numel() for p in model.audio_encoder.parameters()) / 1e6
            logging.info(
                f"[audio_caption.train] Loaded audio encoder weights from {src} (strict=True); params={num_params} M"
            )

    # freeze encoder weights
    if cfg.model.audio_encoder.get("frozen"):
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        if rank == 0:
            num_params = sum(p.numel() for p in model.audio_encoder.parameters()) / 1e6
            logging.info(
                f"[audio_caption.train] Froze audio encoder weights ({num_params} M params)"
            )

    # Save config for reproducibility
    if rank == 0 and cfg.get("exp_dir"):
        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)
        logging.info(
            f"[audio_caption.train] Saved config and tokenizer to {cfg.exp_dir}"
        )

    # Datamodule & Trainer
    data_module = AudioCaptionDatamodule(cfg.data)
    trainer = Trainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
