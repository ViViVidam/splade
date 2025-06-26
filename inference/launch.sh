#!/bin/bash
#conda activate ML
RANK=0
WORLD_SIZE=4
OUTPUT="/expanse/lustre/projects/csb175/yzound/index/sparse/naver/splade-cocondenser-ensembledistil"
export RANK
export WORLD_SIZE
export OUTPUT
for i in $( seq 0 $((WORLD_SIZE - 1)))
do
    RANK=$i
    echo "$RANK $WORLD_SIZE $OUTPUT"
    sbatch inference_msmarco_doc.sh
    sleep 5
    #/bin/bash -c  "python main.py -id=$RANK -begin=$BEGIN -end=$END"
done
