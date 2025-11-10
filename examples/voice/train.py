"""Voice multitask training entry script using Hydra and torch.distributed.

This script trains a voice multitask model with 4 classification heads:
speaker ID, emotion, gender, and age.

"""

import logging
import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

# Import local modules directly
from data_module import VoiceDatamodule
from trainer import VoiceTrainer
from model import VoiceMultitaskModel
from model_config import VoiceMultitaskConfig

# Import utility functions
from auden.models.audio_tag.utils import load_id2label
from auden.auto.auto_config import AutoConfig  # Only for encoder config


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):

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

    # Load encoder config
    # Priority 1: If pretrained_encoder is provided, load config from there
    if hasattr(cfg.model, 'pretrained_encoder') and cfg.model.pretrained_encoder:
        encoder_config = AutoConfig.from_pretrained(cfg.model.pretrained_encoder)
        if rank == 0:
            logging.info(f"Loading encoder config from pretrained: {cfg.model.pretrained_encoder}")
    # Priority 2: If encoder is provided, try to load from path or model type
    elif hasattr(cfg.model, 'encoder') and cfg.model.encoder:
        try:
            encoder_config = AutoConfig.from_pretrained(cfg.model.encoder)
        except Exception:
            # If path doesn't work, treat as model type
            encoder_config = AutoConfig.for_model(cfg.model.encoder)
    # Priority 3: Default to small zipformer
    else:
        encoder_config = AutoConfig.for_model("zipformer")
        if rank == 0:
            logging.info("Using default small zipformer encoder (emb_dim=512)")

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
    if hasattr(cfg.model, 'pretrained_encoder') and cfg.model.pretrained_encoder:
        pretrained_path = cfg.model.pretrained_encoder
        
        # If directory, find the weight file
        if os.path.isdir(pretrained_path):
            # Try common weight file names in order of preference
            candidate_files = [
                "pretrained.pt",
                "model.pt", 
                "pytorch_model.bin",
                "model.safetensors",
                "pretrained.safetensors"
            ]
            
            weight_file = None
            for fname in candidate_files:
                fpath = os.path.join(pretrained_path, fname)
                if os.path.exists(fpath):
                    weight_file = fpath
                    break
            
            if weight_file is None:
                raise FileNotFoundError(
                    f"Could not find pretrained weights in {pretrained_path}. "
                    f"Expected one of: {candidate_files}"
                )
            pretrained_path = weight_file
        
        if rank == 0:
            logging.info(f"Loading pretrained encoder from: {pretrained_path}")
        
        # Load weights based on file extension
        if pretrained_path.endswith('.safetensors'):
            # Load safetensors format
            try:
                from safetensors.torch import load_file
                pretrained_state = load_file(pretrained_path)
            except ImportError:
                raise ImportError(
                    "safetensors is required to load .safetensors files. "
                    "Install with: pip install safetensors"
                )
        else:
            # Load PyTorch format (.pt, .bin, etc.)
            pretrained_state = torch.load(pretrained_path, map_location="cpu")
        
        # Filter encoder weights only (keys starting with "encoder.")
        encoder_state = {
            k.replace("encoder.", "", 1): v 
            for k, v in pretrained_state.items() 
            if k.startswith("encoder.")
        }
        
        # If no "encoder." prefix found, assume all weights are encoder weights
        if not encoder_state:
            encoder_state = pretrained_state
        
        # Load into model's encoder
        missing_keys, unexpected_keys = model.encoder.load_state_dict(
            encoder_state, strict=False
        )
        
        if rank == 0:
            if missing_keys:
                logging.warning(f"Missing keys when loading encoder: {missing_keys[:5]}...")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading encoder: {unexpected_keys[:5]}...")
            logging.info("Pretrained encoder loaded successfully!")

    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        
        # Save each id2label to separate files
        import json
        from pathlib import Path
        
        exp_dir = Path(cfg.exp_dir)
        json.dump(id2label_id, open(exp_dir / "id2label_id.json", "w"), ensure_ascii=False, indent=2)
        json.dump(id2label_emotion, open(exp_dir / "id2label_emotion.json", "w"), ensure_ascii=False, indent=2)
        json.dump(id2label_gender, open(exp_dir / "id2label_gender.json", "w"), ensure_ascii=False, indent=2)
        json.dump(id2label_age, open(exp_dir / "id2label_age.json", "w"), ensure_ascii=False, indent=2)

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

