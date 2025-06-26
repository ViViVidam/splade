#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="splade test training"
#SBATCH --output="./output/fusion/splade_test-%j.out"
#SBATCH --error="./output/fusion/splade_test-%j.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=96G
#SBATCH -t 48:00:00

module purge
module load gpu
module load slurm

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

# /bin/bash -c "python -m main --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/tie/ --steps 9000"
/bin/bash -c "python -m test"
# /bin/bash -c "python -m trainAvgPool --save_path /expanse/lustre/projects/csb176/yzound/checkpoint/splade/splade2model/avgPool/"