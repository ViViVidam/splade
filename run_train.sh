#!/bin/bash
#SBATCH -A csb185
#SBATCH --job-name="splade ce training"
#SBATCH --output="./output/fusion/spladeCE-%j.out"
#SBATCH --error="./output/fusion/spladeCE-%j.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=96G
#SBATCH -t 48:00:00

# load conda env variables
# loading the profile of user yzound to inject require functions
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

#/bin/bash -c "python -m mainKL --docModel sentence-transformers/msmarco-bert-co-condensor --queryModel sentence-transformers/msmarco-bert-co-condensor --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/tie/"
#/bin/bash -c "python -m mainKL --docModel sentence-transformers/msmarco-bert-co-condensor --queryModel sentence-transformers/msmarco-bert-co-condensor --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/untie/ --use_untie 1"
#/bin/bash -c "python -m trainAvgPool --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/avgPool/"
#/bin/bash -c "python -m train_splade --batchsize 128 --epoch 10 --docModel sentence-transformers/msmarco-bert-co-condensor --queryModel sentence-transformers/msmarco-bert-co-condensor --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/"
#/bin/bash -c "python -m train_splade_ce --batchsize 128 --epoch 10 --docModel Luyu/co-condenser-marco --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/"
#accelerate launch --multi_gpu train_splade_ce.py --batchsize 128 --epoch 10 --docModel /expanse/lustre/projects/csb185/thess/splade/splade_training_ht/checkpoints/warmup_Splade_0_MLMTransformer --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/ --gradient_accum 4 --use_untie 1
accelerate launch --multi_gpu train_splade_ce.py --batchsize 64 --epoch 10 --docModel intfloat/simlm-base-msmarco-finetuned --gradient_accum 4 --use_untie 1 --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/