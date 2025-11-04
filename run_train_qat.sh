#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="KD margine MSE splade V3"
#SBATCH --output="./output/finetune/spladev3-%j.out"
#SBATCH --error="./output/finetune/spladev3-%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH -t 16:00:00

module load cuda/12.4.0

# load conda env variables
# loading the profile of user zwang48 to inject require functions
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

#/bin/bash -c "python -m mainKL --docModel sentence-transformers/msmarco-bert-co-condensor --queryModel sentence-transformers/msmarco-bert-co-condensor --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/tie/"
#/bin/bash -c "python -m mainKL --docModel sentence-transformers/msmarco-bert-co-condensor --queryModel sentence-transformers/msmarco-bert-co-condensor --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/untie/ --use_untie 1"
#/bin/bash -c "python -m trainAvgPool --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/avgPool/"
#/bin/bash -c "python -m train_splade --batchsize 128 --epoch 10 --docModel sentence-transformers/msmarco-bert-co-condensor --queryModel sentence-transformers/msmarco-bert-co-condensor --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/"
#/bin/bash -c "python -m train_splade_ce --batchsize 128 --epoch 10 --docModel Luyu/co-condenser-marco --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/"
#accelerate launch --multi_gpu train_splade_ce.py --batchsize 128 --epoch 10 --docModel /expanse/lustre/projects/csb185/thess/splade/splade_training_ht/checkpoints/warmup_Splade_0_MLMTransformer --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade/crossEntrophy/ --gradient_accum 4 --use_untie 1
accelerate launch --multi_gpu train_splade_qat.py --per_device_batch_size 64 --epoch 1 --docModel naver/splade-cocondenser-ensembledistil --gradient_accum 1 --save_path /work/nvme/bcgk/yzound/checkpoint/splade-cocondenser-ensembledistil-qat/