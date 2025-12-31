#!/bin/bash
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=1 --nproc_per_node=4 \
    train.py \
        exp_dir=exp/whisper \
        model.speech_encoder.model_type=whisper \
        model.speech_encoder.pretrained_model=myfolder/whisper-large-v2 \
        model.paraling_encoder.model_type=null \
        data.train_data_config=myfolder/configs/train_stage1.yaml \
        data.valid_data_config=myfolder/configs/valid.yaml \
        trainer.num_steps=5000000 
