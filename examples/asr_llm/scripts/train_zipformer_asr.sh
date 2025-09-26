#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 #,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/apdcephfs_cq10/share_1603164/user/yiwenyshao/lhotse:$PYTHONPATH
export PYTHONPATH=/apdcephfs_cq10/share_1603164/user/yiwenyshao/auden_open/src:$PYTHONPATH

num_nodes=1
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

audio_encoder_model_type=zipformer
audio_encoder=/apdcephfs_cq10/share_1603164/user/yiwenyshao/independent/auden/egs/asr/exp/asr_zh_pt_new
llm_model_type=qwen2
llm=/apdcephfs_cq12/share_302080740/model/Qwen2.5-7B-Instruct

torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --master_port=29502 \
    train.py \
        exp_dir=exp/zipformer_asr_${num_gpus}gpus \
        model.audio_encoder.model_type=$audio_encoder_model_type \
        model.audio_encoder.pretrained_model=$audio_encoder \
        model.llm.model_type=$llm_model_type \
        model.llm.pretrained_model=$llm \
        model.llm.use_lora=false \
        model.llm.use_flash_attn=false \
        model.encoder_projector_ds_rate=4 \
        model.tag_audio_boundary=false \
        model.exclude_from_checkpoint="["llm"]" \
        trainer.save_every_n=1 \
        data.feature=fbank \
        data.pad_to_30s=false \
        data.train_data_config=configs/aishell2/train_data_config.yaml \
        data.valid_data_config=configs/aishell2/valid_data_config.yaml