#!/bin/bash

# Voice multitask training script for development/testing
# Usage: 
#   cd Auden/examples/voice
#   bash scripts/train.sh

# Experiment directory
EXP_DIR="./exp/auden_voice_test"

# Number of GPUs
NUM_GPUS=1
  
# Optional: pretrained encoder path (comment out if training from scratch)
# PRETRAINED_ENCODER="/path/to/pretrained/encoder"

TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS} train.py \
  exp_dir=${EXP_DIR} \
  model.id2label_json_id=configs/id2label_id.json \
  model.id2label_json_emotion=configs/id2label_emotion.json \
  model.id2label_json_gender=configs/id2label_gender.json \
  model.id2label_json_age=configs/id2label_age.json \
  data.train_data_config=configs/data_configs/train_data_config.yaml \
  data.valid_data_config=configs/data_configs/valid_data_config.yaml \
  trainer.save_every_n=2"

# Run training
eval ${TRAIN_CMD} 

echo "Training completed. Results saved in: ${EXP_DIR}"

