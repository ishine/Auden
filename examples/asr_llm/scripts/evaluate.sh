#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 #,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/apdcephfs_cq10/share_1603164/user/yiwenyshao/lhotse:$PYTHONPATH
export PYTHONPATH=/apdcephfs_cq10/share_1603164/user/yiwenyshao/auden_open/src:$PYTHONPATH

# exp_dir=exp/zipformer_asr_1gpus
exp_dir=exp/whisper_asr_1gpus
python evaluate.py \
  data.max_duration=100 \
  exp_dir=$exp_dir \
  checkpoint.iter=6000 \
  checkpoint.avg=1 \
  data.test_data_config=configs/aishell2/valid_data_config.yaml \
  data.whisper_fbank=true \
  data.pad_to_30s=true