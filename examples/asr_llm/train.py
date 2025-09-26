import json
import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AsrDatamodule
from lhotse.utils import fix_random_seed
from omegaconf import DictConfig, OmegaConf
from trainer import AsrLLMTrainer as Trainer
from transformers import AutoConfig as HFConfig
from transformers import AutoModelForCausalLM as HFCausalLM
from transformers import AutoTokenizer as HFTokenizer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel


def load_pretrained_audio_encoder(cfg):
    """Build audio encoder config and optionally load a pretrained module.

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

    raise ValueError(f"Unsupported audio encoder model_type: {model_type}")


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

    # Seed
    fix_random_seed(114514)

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

    # 1) Build sub-configs and optionally load submodules
    audio_encoder_config, pretrained_audio_encoder = load_pretrained_audio_encoder(
        cfg.model.audio_encoder
    )
    llm_config, pretrained_llm, tokenizer = load_pretrained_llm(cfg.model.llm)
    # 2) Tokenizer with audio token
    DEFAULT_AUDIO_TOKEN = cfg.model.get("audio_token", "<|AUDIO|>")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEFAULT_AUDIO_TOKEN]},
        replace_additional_special_tokens=False,
    )
    tokenizer.padding_side = (
        "left" if cfg.model.get("use_flash_attn", False) else "right"
    )

    # 4) Assemble Audio-LLM config and model
    config = AutoConfig.for_model(
        cfg.model.model_type,
        audio_encoder_config=audio_encoder_config,
        llm_config=llm_config,
        audio_encoder_projector_ds_rate=cfg.model.get(
            "audio_encoder_projector_ds_rate", 8
        ),
        use_flash_attn=cfg.model.get("use_flash_attn", False),
        tag_audio_boundary=cfg.model.get("tag_audio_boundary", False),
        exclude_from_checkpoint=list(cfg.model.get("exclude_from_checkpoint", None)),
        audio_token=DEFAULT_AUDIO_TOKEN,
    )

    model = AutoModel.from_config(config, tokenizer=tokenizer)

    # 5) Load pretrained weights (if provided)
    if pretrained_audio_encoder is not None:
        model.audio_encoder.load_state_dict(
            pretrained_audio_encoder.state_dict(), strict=True
        )
        src = cfg.model.audio_encoder.get("pretrained_model")
        num_params = sum(p.numel() for p in model.audio_encoder.parameters()) / 1e6
        logging.info(
            f"[asr_llm.train] Loaded audio encoder from {src}; params={num_params:.2f} M"
        )
    if pretrained_llm is not None:
        model.llm.load_state_dict(pretrained_llm.state_dict(), strict=False)
        src_txt = cfg.model.llm.get("pretrained_model")
        num_params_txt = sum(p.numel() for p in model.llm.parameters()) / 1e6
        logging.info(
            f"[asr_llm.train] Loaded LLM weights from {src_txt}; params={num_params_txt:.2f} M"
        )

    # 6) Freeze modules if requested
    if cfg.model.audio_encoder.get("frozen"):
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        logging.info(f"[asr_llm.train] Froze audio encoder")
    if cfg.model.llm.get("frozen"):
        for p in model.llm.parameters():
            p.requires_grad = False
        logging.info(f"[asr_llm.train] Froze LLM")

    # 7) Save excluded modules (if any) and config/tokenizer
    if rank == 0 and cfg.get("exp_dir"):
        if getattr(config, "exclude_from_checkpoint", None):
            if "audio_encoder" in config.exclude_from_checkpoint:
                audio_encoder_path = os.path.join(cfg.exp_dir, "audio_encoder")
                logging.info(
                    f"[asr_llm.train] Saved audio encoder to {audio_encoder_path}"
                )
                model.audio_encoder.save_pretrained(audio_encoder_path)
            if "llm" in config.exclude_from_checkpoint:
                llm_path = os.path.join(cfg.exp_dir, "llm")
                logging.info(f"[asr_llm.train] Saved LLM to {llm_path}")
                model.llm.save_pretrained(llm_path)

        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)
        logging.info(f"[asr_llm.train] Saved config/tokenizer to {cfg.exp_dir}")

    # 8) Data & Trainer
    data_module = AsrDatamodule(cfg.data)
    trainer = Trainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
