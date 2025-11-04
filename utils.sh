#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="splade utils task"
#SBATCH --output="./output/splade_utils-%j.out"
#SBATCH --error="./output/splade_utils-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -t 12:00:00

# load conda env variables
# loading the profile of user zwang48 to inject require functions

conda activate ML

nvidia-smi

python utils/collect_passage_length.py --threshold 0.01