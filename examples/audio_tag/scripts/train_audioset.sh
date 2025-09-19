#!/bin/bash
# ----------------train from scratch------------------
exp_dir=exp/audioset_pt
torchrun --nproc_per_node=4 \
        --master-port=29501 \
        train.py \
        exp_dir=$exp_dir \
        trainer.optimizer.lr=0.045 \
        trainer.scheduler.lr_steps_per_epoch=10000 \
        data.train_data_config=configs/audioset/train_data_config_audioset.yaml \
        data.valid_data_config=configs/audioset/test_data_config_audioset.yaml \
        data.sampler.max_duration=800 \
        data.use_infinite_dataset=true \
        model.id2label_json=configs/audioset/id2label_audioset.json \
        model.loss=bce


# # # # ----------------finetune audioset with a pretrained encoder------------------
# pretrained_model=your_path_to_pretrained_model
# exp_dir=exp/audioset_ft
# torchrun --nproc_per_node=4 \
#         --master-port=29501 \
#         finetune.py \
#         exp_dir=$exp_dir \
#         trainer.optimizer.lr=0.015 \
#         trainer.scheduler.lr_steps_per_epoch=10000 \
#         data.train_data_config=configs/audioset/train_data_config_audioset.yaml \
#         data.valid_data_config=configs/audioset/test_data_config_audioset.yaml \
#         data.sampler.max_duration=800 \
#         data.use_infinite_dataset=true \
#         model.id2label_json=configs/audioset/id2label_audioset.json \
#         model.loss=bce \
#         model.encoder.model_type=zipformer \
#         model.encoder.pretrained_model=$pretrained_model \
#         model.encoder.freeze_encoder=false \
#         # trainer.start_batch=52000 # if you want to resume training