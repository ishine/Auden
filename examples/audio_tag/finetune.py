"""Audio Tagging finetuning entry script using Hydra and torch.distributed.

This mirrors the ASR example structure while adapting the trainer/datamodule to
audio tagging. It wires model, datamodule and trainer, and supports DDP via torchrun.

Usage:
    torchrun --nproc_per_node=8 examples/audio_tag/finetune.py

Config highlights:
- cfg.model.encoder.model_type: encoder architecture key (e.g., 'zipformer')
- cfg.model.encoder.pretrained_model: path or repo id for the pretrained encoder source
- cfg.model.encoder.freeze_encoder: whether to freeze the encoder (linear probing)
- cfg.model.id2label_json: path to an id2label.json mapping
- cfg.data: Lhotse-based datamodule configs (see configs/)
"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AudioTagDatamodule
from omegaconf import DictConfig, OmegaConf
from trainer import AudioTagTrainer

from auden.auto.auto_config import AutoConfig
from auden.auto.auto_model import AutoModel
from auden.models.audio_tag.utils import load_id2label, save_id2label


def load_pretrained_encoder(cfg: DictConfig):
    """Construct and load a pretrained encoder module, returning (config, module).

    Expected cfg structure (subset):
        cfg.model.encoder.model_type: str
        cfg.model.encoder.pretrained_model: Optional[str] (path or repo id)

    Returns
    -------
    (encoder_config, encoder_model)
        encoder_config: AutoConfig instance for the encoder
        encoder_model: nn.Module instance loaded with pretrained weights

    Notes
    -----
    - Currently supports 'zipformer'. Extend with additional branches for new encoders.
    - The caller is responsible for attaching this encoder to the task model
      (we copy the state_dict into ``model.encoder``).
    - If you plan to linear-probe, set ``cfg.model.encoder.freeze_encoder = True``.
    """
    if cfg.model.encoder.model_type == "zipformer":
        from auden.models.zipformer.model import ZipformerEncoderModel

        encoder_model = ZipformerEncoderModel.from_pretrained(
            cfg.model.encoder.pretrained_model
        )
        encoder_config = encoder_model.config
    else:
        raise ValueError(
            f"Unsupported encoder model type: {cfg.model.encoder.model_type}"
        )
    return encoder_config, encoder_model


@hydra.main(version_base=None, config_path="configs", config_name="finetune")
def main(cfg: DictConfig):
    """Hydra entrypoint for finetuning an audio tagging model with a pretrained encoder.

    Steps
    -----
    1) Load a pretrained encoder (config + weights).
    2) Build the task model using the loaded encoder config.
    3) Copy encoder weights into the task model's ``model.encoder``.
    4) Optionally freeze the encoder for linear probing.
    5) Initialize the datamodule and start training.
    """
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

    pretrained_encoder_config, pretrained_encoder = load_pretrained_encoder(cfg)
    # AudioTag config and model (respect loss if provided)
    config = AutoConfig.for_model(
        cfg.model.model_type,
        encoder_config=pretrained_encoder_config,
        loss=cfg.model.loss,
    )
    id2label = load_id2label(cfg.model.id2label_json)
    model = AutoModel.from_config(config, id2label=id2label)

    # load pretrained encoder weights
    model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)

    # freeze encoder weights
    if cfg.model.encoder.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        logging.info("Froze encoder weights")

    # save config and id2label
    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        save_id2label(id2label, cfg.exp_dir)

    # Datamodule
    data_module = AudioTagDatamodule(cfg.data)

    # Trainer
    trainer = AudioTagTrainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
