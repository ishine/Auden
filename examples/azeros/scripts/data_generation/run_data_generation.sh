#!/bin/bash

# example script to run data processing of one mode='sift_sp'
# with 8 gpus and 8 shards on a single machine. 
NGPUS=8
NShards=8
offset=0
mode=sift_sp
outdir=./myfolder/manifests/$mode


for i in $(seq 0 $(expr $NGPUS - 1)); do
    if [ $i == 0 ]; then
        verbose='--verbose '
    else
        verbose=' '
    fi
    CUDA_VISIBLE_DEVICES=$i \
    python scripts/run_data_generation.py \
        --mode $mode \
        --output-dir $outdir \
        --nshards $NShards \
        --shard $(expr $i + $offset) \
        ${verbose} \
        --batch-size 200 &
done

wait

echo 'All Done'

# then for each datasets, use 
# `lhotse combine $outdir/xxx/${mode}_*.jsonl.gz xxx.jsonl.gz` to combine all shards
