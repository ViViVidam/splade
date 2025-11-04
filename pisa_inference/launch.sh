#!/bin/bash
RANK=0
MODEL=/projects/bfcj/yzound/checkpoint/splade_v3_qat/
#CHECKPOINT=""
OUTPUT=/work/hdd/bfcj/yzound/index/beir/spladev3/
WORKERS=4
export RANK
#export CHECKPOINT
export OUTPUT
export MODEL
for i in $( seq 0 $((WORKERS - 1)))
do
    RANK=$i
    echo "$RANK"
    sbatch run_index.sh
    sleep 5
done