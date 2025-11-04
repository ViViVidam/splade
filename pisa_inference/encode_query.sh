#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="encode query splade V3"
#SBATCH --output="./output/encode-query/spladev3-%j.out"
#SBATCH --error="./output/encode-query/spladev3-%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH -t 0:30:00

### conda activation
__conda_setup="$('/u/yzound/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/yzound/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/u/yzound/miniconda3/profile.d/conda.sh"
    else
        export PATH="/u/yzound/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate ML

MODEL=/projects/bfcj/yzound/checkpoint/splade_v3_qat/
#MODEL=naver/splade-v3

python -m inference_q_SPLADE --model $MODEL --output /work/hdd/bfcj/yzound/index/beir/spladev3/