#!/bin/bash
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=1 --nproc_per_node=4 \
    train.py \
        exp_dir=exp/stage2 \
        model.pretrained_model=exp/stage1/pretrained.pt \
        model.speech_encoder_projector.frozen=true \
        data.train_data_config=configs/train_stage2.yaml \
        data.valid_data_config=configs/valid_data_config.yaml \
        trainer.num_steps=1000000 
