#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="inference splade V3"
#SBATCH --output="./output/inference/spladev3-%j.out"
#SBATCH --error="./output/inference/spladev3-%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=196G
#SBATCH -t 30:00:00

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

python indexing.py --model naver/splade-cocondenser-selfdistil \
    --index_dir /projects/bfcj/yzound/index/splade_selfdistil \
    --corpus_path /projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv 

python evaluate.py --model naver/splade-cocondenser-selfdistil \
    --index_dir /projects/bfcj/yzound/index/splade_selfdistil \
    --query_path /projects/bcgk/yzound/datasets/msmarco/queries.train.tsv