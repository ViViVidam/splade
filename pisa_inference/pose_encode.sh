#!/bin/bash
#SBATCH -A bcgk-delta-cpu
#SBATCH --job-name="post-encode splade V3"
#SBATCH --output="./output/post-encode/spladev3-%j.out"
#SBATCH --error="./output/post-encode/spladev3-%j.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --mem=96G
#SBATCH -t 16:00:00

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

INDEX=/work/hdd/bfcj/yzound/index/beir/spladev3/
python index2pisa.py --collection $INDEX/file_ --numbers 4 --output $INDEX/index
python queries2pisa.py $INDEX/index $INDEX/queries.dev.tsv $INDEX/index.queries 0