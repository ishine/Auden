#!/bin/bash
# Usage: 
#   cd Auden/examples/voice
#   bash scripts/evaluate.sh

# Run evaluation
python evaluate.py \
    exp_dir=./exp/voice_test \
    checkpoint.avg=5 \
    checkpoint.iter=20000

