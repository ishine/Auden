#!/bin/bash
source scripts/path.sh

torchrun --nnodes=1 --nproc_per_node=4 \
    train.py \
        exp_dir=exp/stage2 \
        model.pretrained_model=$(realpath exp/stage1/pretrained.pt) \
        model.speech_encoder_projector.frozen=true \
        data.train_data_config=myfolder/configs/train_stage2.yaml \
        data.valid_data_config=myfolder/configs/valid.yaml \
        trainer.num_steps=1000000 
