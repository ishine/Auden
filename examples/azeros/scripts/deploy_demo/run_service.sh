#!/bin/bash

# Add CosyVoice environments
export PYTHONPATH=$PYTHONPATH:$PWD/myfolder/CosyVoice:$PWD/myfolder/Matcha-TTS

# run service
python scripts/deploy_demo/run_service.py \
    --model-path exp/stage2/pretrained.pt \
    --cosyvoice-path myfolder/CosyVoice2-0.5B \
    --zeroshot-prompt assets/zero_shot_prompt.wav
