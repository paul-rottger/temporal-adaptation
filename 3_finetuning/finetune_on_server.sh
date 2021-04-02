#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=10m-rand-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=10m-rand-finetune.out
#SBATCH --error=10m-rand-finetune.err
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
for modelpath in $DATA/gab-language-change/adapted-models/reddit/total-models/bert-rand_10m/; do
    for trainpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/total/train_*.csv; do
        python run_finetuning.py \
            --model_name_or_path $modelpath \
            --train_file $trainpath \
            --validation_file $trainpath \
            --do_train \
            --per_device_train_batch_size 32 \
            --save_steps 100000 \
            --output_dir $DATA/gab-language-change/finetuned-models/reddit/total-models/$(basename $modelpath)-$(basename $trainpath .csv) \
            --overwrite_output_dir \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --use_special_tokens
    done
done