#!/bin/bash
# ----------------linear probing------------------
pretrained_model=exp/audioset_pt/pretrained.pt
model_type=zipformer
exp_dir=exp/linear_prob_audioset

# -----------------esc50-------------------- 1gpu,22mins
num_steps=4000
torchrun --nproc_per_node=1 \
         --master_port=29502 \
        finetune.py \
        exp_dir=$exp_dir/esc50 \
        trainer.optimizer.lr=0.0045 \
        trainer.valid_interval=100 \
        trainer.num_steps=$num_steps \
        data.train_data_config=configs/esc50/train_data_config_esc50.yaml \
        data.valid_data_config=configs/esc50/test_data_config_esc50.yaml \
        data.sampler.max_duration=800 \
        data.use_infinite_dataset=true \
        model.id2label_json=configs/esc50/id2label_esc50.json \
        model.encoder.model_type=$model_type \
        model.encoder.pretrained_model=$pretrained_model \
        model.encoder.freeze_encoder=true
        

python evaluate.py \
    exp_dir=$exp_dir/esc50 \
    data.test_data_config=configs/esc50/test_data_config_esc50.yaml \
    checkpoint.iter=$num_steps \
    checkpoint.avg=1

# ---------------us8k------------------ 1gpu, 14mins
num_steps=2400
torchrun --nproc_per_node=1 \
         --master_port=29502 \
        finetune.py \
        exp_dir=$exp_dir/urbansound \
        trainer.optimizer.lr=0.0045 \
        trainer.valid_interval=100 \
        trainer.num_steps=$num_steps \
        data.train_data_config=configs/urbansound/train_data_config_urbansound.yaml \
        data.valid_data_config=configs/urbansound/test_data_config_urbansound.yaml \
        data.sampler.max_duration=800 \
        data.use_infinite_dataset=true \
        model.id2label_json=configs/urbansound/id2label_urbansound.json \
        model.encoder.model_type=$model_type \
        model.encoder.pretrained_model=$pretrained_model \
        model.encoder.freeze_encoder=true

python evaluate.py \
    exp_dir=$exp_dir/urbansound \
    data.test_data_config=configs/urbansound/test_data_config_urbansound.yaml \
    checkpoint.iter=$num_steps \
    checkpoint.avg=1

# --------------vggsound------------------ 4gpus, 6hours
num_steps=32000
torchrun --nproc_per_node=4 \
        --master_port=29502 \
        finetune.py \
        exp_dir=$exp_dir/vggsound \
        trainer.optimizer.lr=0.0045 \
        trainer.num_steps=$num_steps \
        data.train_data_config=configs/vggsound/train_data_config_vggsound.yaml \
        data.valid_data_config=configs/vggsound/test_data_config_vggsound.yaml \
        data.sampler.max_duration=800 \
        data.use_infinite_dataset=true \
        model.id2label_json=configs/vggsound/id2label_vggsound.json \
        model.encoder.model_type=$model_type \
        model.encoder.pretrained_model=$pretrained_model \
        model.encoder.freeze_encoder=true

python evaluate.py \
    exp_dir=$exp_dir/vggsound \
    data.test_data_config=configs/vggsound/test_data_config_vggsound.yaml \
    checkpoint.iter=$num_steps \
    checkpoint.avg=1
