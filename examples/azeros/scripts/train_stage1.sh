#!/bin/bash
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=1 --nproc_per_node=4 \
    train.py \
        exp_dir=exp/stage1 \
        data.train_data_config=configs/train_stage1.yaml \
        data.valid_data_config=configs/valid_data_config.yaml \
        model.paraling_encoder.model_type=null \
        trainer.num_steps=5000000
