#!/bin/bash
exp_dir=your_exp_dir
python evaluate.py \
    exp_dir=$exp_dir \
    data.test_data_config=configs/audioset/test_data_config_audioset.yaml \
    checkpoint.filename=pretrained.pt \
    # checkpoint.iter=12000 \
    # checkpoint.avg=1