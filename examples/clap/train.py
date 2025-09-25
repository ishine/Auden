"""CLAP training entry script using Hydra and torch.distributed.

This mirrors the audio_tag example structure while adapting the trainer/datamodule
to audio–text contrastive learning. It wires model, datamodule and trainer, and
supports single-GPU and DDP via torchrun.

Usage:
    # single GPU
    python examples/clap/train.py exp_dir=./exp/your_exp_dir

    # multi-GPU (DDP)
    torchrun --nproc_per_node=8 examples/clap/train.py exp_dir=./exp/your_exp_dir

Config highlights (see examples/clap/configs/train.yaml):
- cfg.model.model_type: "clap"
- cfg.model.audio_encoder: { model_type, pretrained_model?, frozen }
- cfg.model.text_encoder:  { model_type or pretrained_model?, frozen }
- cfg.data: Lhotse-based datamodule configs (see examples/clap/configs/)
- cfg.trainer: optimizer/scheduler, intervals, optional gather_embeddings, etc.

Notes:
- If pretrained encoders are provided, their weights are loaded and can be frozen
  according to cfg.model.*.frozen, with logging of sources and parameter counts.
- For the text encoder: if "pretrained_model" is set it is used; otherwise
  "model_type" should be a valid HF identifier (e.g., "bert-base-uncased",
  "roberta-base").
- The script saves the model config and tokenizer to exp_dir for reproducibility.
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AudioCaptionDatamodule
from omegaconf import DictConfig, OmegaConf
from trainer import ClapTrainer as Trainer
from transformers import AutoConfig as HFConfig
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


def load_pretrained_text_encoder(cfg: DictConfig):
    """Build text encoder config and optionally load a pretrained HF module.

    Supported modes:
    1) Pretrained: set ``pretrained_model`` to an HF/local identifier to load
       weights, config and tokenizer from that source.
    2) Empty-by-type: provide ``model_type`` and optionally ``tokenizer_name``.
       We will construct a config without weights using either:
         - AutoConfig.from_pretrained(model_type) if it looks like a valid
           identifier (e.g., "bert-base-uncased", "roberta-base"); or
         - AutoConfig.for_model(model_type) when ``model_type`` matches a family
           key (e.g., "bert", "roberta", "distilbert", ...).
       The tokenizer is loaded from ``tokenizer_name`` or a sensible default
       for that family.

    Args:
        cfg: The text encoder section, e.g. ``cfg.model.text_encoder`` with fields:
             - pretrained_model: optional HF/local path/name for weights
             - model_type: HF identifier or family when not using pretrained weights
             - tokenizer_name: optional tokenizer id (fallback to a default per family)

    Returns:
        (hf_config, hf_model_or_None, tokenizer_or_None)
    """
    from transformers import AutoConfig as HFConfig
    from transformers import AutoModel as HFModel

    pretrained = cfg.get("pretrained_model")
    model_type = cfg.get("model_type", "bert")

    # 1) Pretrained weights path/name provided
    if pretrained:
        encoder_model = HFModel.from_pretrained(pretrained, add_pooling_layer=False)
        encoder_config = encoder_model.config
        tokenizer = HFTokenizer.from_pretrained(pretrained)
        return encoder_config, encoder_model, tokenizer

    # 2) Empty-by-type via AutoConfig.from_pretrained when model_type is an identifier
    encoder_model = None
    # Treat model_type as a full identifier for AutoConfig/tokenizer
    encoder_config = HFConfig.from_pretrained(model_type, add_pooling_layer=False)
    tokenizer = HFTokenizer.from_pretrained(model_type)
    return encoder_config, encoder_model, tokenizer


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

    # load pretrained encoders
    audio_encoder_config, pretrained_audio_encoder = load_pretrained_audio_encoder(
        cfg.model.audio_encoder
    )
    text_encoder_config, pretrained_text_encoder, tokenizer = (
        load_pretrained_text_encoder(cfg.model.text_encoder)
    )

    # Assemble CLAP config and model
    config = AutoConfig.for_model(
        cfg.model.model_type,
        audio_encoder_config=audio_encoder_config,
        text_encoder_config=text_encoder_config,
    )
    model = AutoModel.from_config(config, tokenizer=tokenizer)

    # load pretrained encoder weights
    if pretrained_audio_encoder is not None:
        model.audio_encoder.load_state_dict(
            pretrained_audio_encoder.state_dict(), strict=True
        )
        # from omegaconf import open_dict

        # with open_dict(cfg.trainer):
        #     cfg.trainer.init_batch_count = 100000
        # logging.info(
        #     "Loaded pretrained audio encoder; setting trainer.init_batch_count=100000 to saturate scheduledfloat"
        # )
        if rank == 0:
            src = cfg.model.audio_encoder.get("pretrained_model")
            num_params = sum(p.numel() for p in model.audio_encoder.parameters()) / 1e6
            logging.info(
                f"[clap.train] Loaded audio encoder weights from {src} (strict=True); params={num_params} M"
            )

    if pretrained_text_encoder is not None:
        model.text_encoder.load_state_dict(
            pretrained_text_encoder.state_dict(), strict=True
        )
        if rank == 0:
            src_txt = cfg.model.text_encoder.get("pretrained_model")
            num_params_txt = (
                sum(p.numel() for p in model.text_encoder.parameters()) / 1e6
            )
            logging.info(
                f"[clap.train] Loaded text encoder weights from {src_txt} (strict=True); params={num_params_txt} M"
            )

    # freeze encoder weights
    if cfg.model.audio_encoder.get("frozen"):
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        if rank == 0:
            num_params = sum(p.numel() for p in model.audio_encoder.parameters()) / 1e6
            logging.info(
                f"[clap.train] Froze audio encoder weights ({num_params} M params)"
            )
    if cfg.model.text_encoder.get("frozen"):
        for p in model.text_encoder.parameters():
            p.requires_grad = False
        if rank == 0:
            num_params_txt = (
                sum(p.numel() for p in model.text_encoder.parameters()) / 1e6
            )
            logging.info(
                f"[clap.train] Froze text encoder weights ({num_params_txt} M params)"
            )

    # Save config for reproducibility
    if rank == 0 and cfg.get("exp_dir"):
        config.save_pretrained(cfg.exp_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(cfg.exp_dir)
        logging.info(f"[clap.train] Saved config and tokenizer to {cfg.exp_dir}")

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
