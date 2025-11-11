"""AZeroS training entry script using Hydra and torch.distributed.

This script wires together the encoder, projector, LLM, data module and trainer,
supporting single- or multi-GPU training via torchrun.
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AzerosDatamodule
from omegaconf import DictConfig, OmegaConf
from trainer import AzerosTrainer as Trainer
from transformers import AutoConfig as HFConfig
from transformers import AutoTokenizer as HFTokenizer
from transformers import AutoModelForCausalLM as HFCausalLM

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel


def load_pretrained_encoder(cfg):
    """Build encoder config and optionally load a pretrained module.

    Supports:
    - zipformer (Auden native)
    - whisper (HF; loads WhisperEncoder config/weights)

    Args:
        cfg: cfg.model.audio_encoder with fields:
             - model_type: e.g., "zipformer" or "whisper-large-v3"
             - pretrained_model: optional path/identifier

    Returns:
        (encoder_config, encoder_module_or_None)
    """
    model_type = cfg.get("model_type")
    pretrained = cfg.get("pretrained_model")

    if model_type is None:
        return None, None

    if model_type == "zipformer":
        from auden.models.zipformer.model import ZipformerEncoderModel
        from auden.models.zipformer.model_config import ZipformerConfig

        if pretrained:
            encoder_model = ZipformerEncoderModel.from_pretrained(pretrained)
            encoder_config = encoder_model.config
        else:
            encoder_config = ZipformerConfig(
                encoder_dim=[192, 256, 512, 768, 512, 256],
                feedforward_dim=[576, 768, 1536, 2304, 1536, 768],
                num_encoder_layers=[2, 2, 4, 5, 4, 2],
            )
            encoder_model = None
        return encoder_config, encoder_model

    # Treat any value containing "whisper" as HF Whisper family
    if "whisper" in str(model_type).lower():
        from transformers.models.whisper.configuration_whisper import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperModel

        if pretrained:
            full = WhisperModel.from_pretrained(pretrained)
            encoder_model = full.encoder
            encoder_config = full.config
        else:
            encoder_config = WhisperConfig()
            encoder_model = None
        return encoder_config, encoder_model

    logging.info(f"Unsupported audio encoder model_type: {model_type}")


def load_pretrained_llm(cfg):
    """Build LLM config and optionally load a pretrained HF module and tokenizer.

    Modes:
    - Pretrained: cfg.pretrained_model -> load weights/config/tokenizer
    - Empty-by-type: cfg.model_type -> build config via HF without weights; tokenizer from model_type
    """
    pretrained = cfg.get("pretrained_model")
    model_type = cfg.get("model_type", "qwen2")

    if pretrained:
        llm = HFCausalLM.from_pretrained(pretrained, torch_dtype=torch.float16)
        llm_config = llm.config
        tokenizer = HFTokenizer.from_pretrained(pretrained)

        return llm_config, llm, tokenizer

    # Empty-by-type
    try:
        llm_config = HFConfig.from_pretrained(model_type)
    except Exception:
        llm_config = HFConfig.for_model(model_type)
    tokenizer = HFTokenizer.from_pretrained(model_type)
    return llm_config, None, tokenizer


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

    # 4) model & config setup
    speech_encoder_config, pretrained_speech_encoder = load_pretrained_encoder(
        cfg.model.speech_encoder)
    paraling_encoder_config, pretrained_paraling_encoder = load_pretrained_encoder(
        cfg.model.paraling_encoder)
    llm_config, pretrained_llm, tokenizer = load_pretrained_llm(cfg.model.llm)

    if cfg.model.use_flash_attn:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    frozen_modules = []
    if cfg.model.llm.frozen:
        frozen_modules.append('llm')
    if cfg.model.speech_encoder.frozen:
        frozen_modules.append('speech_encoder')
    if cfg.model.speech_encoder_projector.frozen:
        frozen_modules.append('speech_encoder_projector')
    if cfg.model.paraling_encoder.frozen:
        frozen_modules.append('paraling_encoder')
    if cfg.model.paraling_encoder_projector.frozen:
        frozen_modules.append('paraling_encoder_projector')

    exclude_from_checkpoint = frozen_modules
    logging.info(f"Modules to be excluded from checkpoints: {exclude_from_checkpoint}")

    config = AutoConfig.for_model(
        cfg.model.model_type,
        llm_config=llm_config,
        speech_encoder_config=speech_encoder_config,
        paraling_encoder_config=paraling_encoder_config,
        speech_encoder_projector_ds_rate=cfg.model.speech_encoder_projector.ds_rate,
        paraling_encoder_projector_ds_rate=cfg.model.paraling_encoder_projector.ds_rate,
        exclude_from_checkpoint=exclude_from_checkpoint
    )

    model = AutoModel.from_config(
        config=config,
        tokenizer=tokenizer,
    )

    # 5) Load pretrained components
    if pretrained_llm is not None:
        model.llm.load_state_dict(pretrained_llm.state_dict(), strict=False)
        src = cfg.model.llm.get("pretrained_model")
        num_params_txt = sum(p.numel() for p in model.llm.parameters()) / 1e6
        logging.info(f"Load {num_params_txt:.2f}M weights from {src} for LLM")
    if pretrained_speech_encoder is not None:
        model.speech_encoder.load_state_dict(
            pretrained_speech_encoder.state_dict(), strict=True
        )
        src = cfg.model.speech_encoder.get("pretrained_model")
        num_params = sum(p.numel() for p in model.speech_encoder.parameters()) / 1e6
        logging.info(f"Load {num_params:.2f}M weights from {src} for speech_encoder")
    if pretrained_paraling_encoder is not None:
        model.paraling_encoder.load_state_dict(
            pretrained_paraling_encoder.state_dict(), strict=True
        )
        src = cfg.model.paraling_encoder.get("pretrained_model")
        num_params = sum(p.numel() for p in model.paraling_encoder.parameters()) / 1e6
        logging.info(f"Load {num_params:.2f}M weights from {src} for paraling_encoder")

    # 6) save & freeze modules
    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)
        for module in exclude_from_checkpoint:
            if getattr(model, module) is None:
                continue
            save_path = os.path.join(cfg.exp_dir, module)
            try:
                getattr(model, module).save_pretrained(save_path)
                logging.info(f"Saved {module} to {save_path}")
            except:
                logging.info(f"Failed to save {module}")


    # freeze modules
    for module in frozen_modules:
        if getattr(model, module) is not None:
            logging.info(f"Freeze params in {module}")
            for p in getattr(model, module).parameters():
                p.requires_grad = False

    # 7) Initialize data module and trainer
    data_module = AzerosDatamodule(cfg.data)
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
