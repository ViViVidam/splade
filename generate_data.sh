#!/bin/bash
#SBATCH -A bcgk-delta-cpu
#SBATCH --job-name="inference splade V3"
#SBATCH --output="./output/mining/spladev3-%j.out"
#SBATCH --error="./output/mining/spladev3-%j.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
#SBATCH --mem=140G
#SBATCH -t 8:00:00

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

python generate_datasets.py