#!/bin/bash
source scripts/path.sh

# torchrun --nnodes=1 --nproc_per_node=4 \
python \
    train.py \
        exp_dir=exp/stage2 \
        data.train_data_config=myfolder/configs/train_stage2.yaml \
        data.valid_data_config=myfolder/configs/valid.yaml \
        trainer.save_every_n=1 \
        trainer.num_steps=1000000 
