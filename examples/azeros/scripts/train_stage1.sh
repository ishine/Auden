#!/bin/bash
source scripts/path.sh

torchrun --nnodes=1 --nproc_per_node=4 \
    train.py \
        exp_dir=myfolder/exp/stage1 \
        data.train_data_config=myfolder/configs/train_stage1.yaml \
        data.valid_data_config=myfolder/configs/valid.yaml \
        model.paraling_encoder.model_type=null \
        trainer.num_steps=5000000
