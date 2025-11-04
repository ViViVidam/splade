#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="bmp splade V3"
#SBATCH --output="./output/encode/spladev3-%j.out"
#SBATCH --error="./output/encode/spladev3-%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH -t 24:00:00

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

MODEL=/work/nvme/bcgk/yzound/checkpoint/splade-cocondenser-ensembledistil-qat/
#MODEL=naver/splade-v3
#"arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity" "fever" "hotpotqa" "nq"
for subset in "climate-fever"; do 
    echo $subset
    python -m inference_all_beir_doc_bmp --model $MODEL --output /work/nvme/bcgk/yzound/index/beir/splade-cocondenser-ensembledistil/ --beirname $subset
    python -m inference_all_beir_queries_bmp --model $MODEL --output /work/nvme/bcgk/yzound/index/beir/splade-cocondenser-ensembledistil --beirname $subset
done