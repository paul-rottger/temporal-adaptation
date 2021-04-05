#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --gres=gpu:v100:1

# reset modules
module purge

# load python module
module load python/anaconda3/2019.03

# activate the right conda environment
source activate $DATA/conda-envs/gab-language-change

# Useful job diagnostics
#
nvidia-smi
#

# Executing the finetuning script with set options
for modelpath in $DATA/gab-language-change/finetuned-models/reddit/total-models/bert-rand_10m-*/; do

    python run_test.py \
        --model_name_or_path $modelpath \
        --test_file $DATA/gab-language-change/0_data/clean/labelled_reddit/total/test_rand_10k.csv \
        --per_device_eval_batch_size 256 \
        --output_dir $DATA/gab-language-change/eval-results/reddit/classification/total-models \
        --output_name $(basename $modelpath)-test_rand_10k \
        --overwrite_output_dir \
        --max_seq_length 128 \
        --use_special_tokens

done