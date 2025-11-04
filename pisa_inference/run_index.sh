#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="CL+KD margine MSE splade V3"
#SBATCH --output="./output/finetune/spladev3-%j.out"
#SBATCH --error="./output/finetune/spladev3-%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH -t 10:00:00

### conda activation
__conda_setup="$('/home/yzound/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/yzound/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/yzound/miniconda3/profile.d/conda.sh"
    else
        export PATH="/home/yzound/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate ML

/bin/bash -c "python -m inference_SPLADE --model $MODEL --output $OUTPUT --id $RANK"
#/bin/bash -c "python -m inference_SPLADE --model Luyu/co-condenser-marco --checkpoint /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/40000/model_state_dict.pt --output /expanse/lustre/projects/csb176/yzound/index/sparse/splade/ --id 1"
#/bin/bash -c "python -m inference_SPLADE --model Luyu/co-condenser-marco --checkpoint /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/40000/model_state_dict.pt --output /expanse/lustre/projects/csb176/yzound/index/sparse/splade/ --id 2"
#/bin/bash -c "python -m inference_SPLADE --model Luyu/co-condenser-marco --checkpoint /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/40000/model_state_dict.pt --output /expanse/lustre/projects/csb176/yzound/index/sparse/splade/ --id 3"
#/bin/bash -c "python -m inference_q_SPLADE --model $MODEL --output /expanse/lustre/projects/csb176/yzound/index/sparse/splade/"