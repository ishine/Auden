#!/bin/bash
source scripts/path.sh

NGPUS=8
NShards=8
offset=0
mode=simt_sp_2
outdir=./myfolder/manifests/$mode

# ps aux | grep run_data_generation | awk '{print $2}' | xargs kill -9

for i in $(seq 0 $(expr $NGPUS - 1)); do
    CUDA_VISIBLE_DEVICES=$i \
    python scripts/run_data_generation.py \
        --mode $mode \
        --output-dir $outdir \
        --nshards $NShards \
        --shard $(expr $i + $offset) &
done

wait

# use `lhotse combine $outdir/xxx/${mode}_*.jsonl.gz xxx.jsonl.gz` to combine all shards
