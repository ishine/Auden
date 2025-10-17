"""TTA training entry script using Hydra and torch.distributed.

This script wires together the model, tokenizer(s), data module and trainer,
supporting single- or multi-GPU training via torchrun.

Compared to the plain ASR example, this variant optionally initializes a
separate text-encoder tokenizer and passes TTA-specific flags/special tokens
into the model configuration.
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import TtaDatamodule
from omegaconf import DictConfig, OmegaConf
from trainer import TtaTrainer as Trainer
from transformers import AutoConfig as HFConfig
from transformers import AutoTokenizer as HFTokenizer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel
from auden.auto.auto_tokenizer import AutoTokenizer


def load_pretrained_speech_encoder(cfg: DictConfig):
    """Build audio encoder config and optionally load a pretrained module.

    Args:
        cfg: The speech encoder section, e.g. ``cfg.model.speech_encoder`` with fields:
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
        encoder_model = HFModel.from_pretrained(pretrained)
        encoder_config = encoder_model.config
        tokenizer = HFTokenizer.from_pretrained(pretrained)
        return encoder_config, encoder_model, tokenizer

    # 2) Empty-by-type via AutoConfig.from_pretrained when model_type is an identifier
    encoder_model = None
    # Treat model_type as a full identifier for AutoConfig/tokenizer
    encoder_config = HFConfig.from_pretrained(model_type)
    tokenizer = HFTokenizer.from_pretrained(model_type)
    return encoder_config, encoder_model, tokenizer


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # 1) Gather torchrun environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 2) Initialize process group if multi-GPU
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # 3) Create experiment directory
    if cfg.get("exp_dir"):
        os.makedirs(cfg.exp_dir, exist_ok=True)

    speech_encoder_config, pretrained_speech_encoder = load_pretrained_speech_encoder(
        cfg.model.speech_encoder
    )
    text_encoder_config, pretrained_text_encoder, text_tokenizer = (
        load_pretrained_text_encoder(cfg.model.text_encoder)
    )

    special_tokens = [str(x) for x in cfg.model.get("special_tokens", [])]
    if special_tokens:
        logging.info(f"Using {len(special_tokens)} special tokens: {special_tokens}")

    config = AutoConfig.for_model(
        cfg.model.model_type,
        speech_encoder_config=speech_encoder_config,
        text_encoder_config=text_encoder_config,
        special_tokens=special_tokens,
    )

    asr_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    model = AutoModel.from_config(
        config=config,
        asr_tokenizer=asr_tokenizer,
        text_tokenizer=text_tokenizer,
    )

    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        asr_tokenizer.save_pretrained(os.path.join(cfg.exp_dir, "asr_tokenizer"))
        text_tokenizer.save_pretrained(os.path.join(cfg.exp_dir, "text_tokenizer"))

    # load pretrained encoder weights
    if pretrained_speech_encoder is not None:
        model.speech_encoder.load_state_dict(
            pretrained_speech_encoder.state_dict(), strict=True
        )
        if rank == 0:
            src = cfg.model.speech_encoder.get("pretrained_model")
            num_params = sum(p.numel() for p in model.speech_encoder.parameters()) / 1e6
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
    if cfg.model.speech_encoder.get("frozen"):
        for p in model.speech_encoder.parameters():
            p.requires_grad = False
        if rank == 0:
            num_params = sum(p.numel() for p in model.speech_encoder.parameters()) / 1e6
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

    # 7) Initialize data module and trainer
    data_module = TtaDatamodule(cfg.data)
    trainer = Trainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    # 8) Destroy process group if used
    if world_size > 1:
        dist.destroy_process_group()

    logging.info("Training finished successfully.")


if __name__ == "__main__":
    main()
