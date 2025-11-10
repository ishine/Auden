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

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel
from auden.auto.auto_tokenizer import AutoTokenizer


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

    # 4) Load pretrained components
    tokenizer = HFTokenizer.from_pretrained(cfg.model.llm)
    if cfg.model.use_flash_attn:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    frozen_modules = ["speech_encoder", "llm", "paraling_encoder"]
    exclude_from_checkpoint = frozen_modules

    # 5) model setup
    model_type = cfg.model.model_type
    del cfg.model.model_type
    config = AutoConfig.for_model(
        model_type,
        **cfg.model,
        exclude_from_checkpoint=exclude_from_checkpoint
    )

    model = AutoModel.from_config(
        config=config,
        tokenizer=tokenizer,
    )

    # 6) save configs & tokenizer, freeze weights
    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)

    # freeze LLM and other optional modules
    for p in model.llm.parameters():
        p.requires_grad = False
    if cfg.model.speech_encoder.get("frozen"):
        for p in model.speech_encoder.parameters():
            p.requires_grad = False
    if cfg.model.speech_encoder_projector.get("frozen"):
        for p in model.speech_encoder_projector.parameters():
            p.requires_grad = False
    if hasattr(cfg.model, 'paraling_encoder') and cfg.model.paraling_encoder.get("frozen"):
        for p in model.paraling_encoder.parameters():
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
