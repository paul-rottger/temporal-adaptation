#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=m17-ra-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=m17-ra-finetune.out
#SBATCH --error=m17-ra-finetune.err
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
for modelpath in $DATA/gab-language-change/adapted-models/reddit/month-models/bert-2017*/; do
    for trainpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/total/train_rand_20k.csv; do
        python run_finetuning.py \
            --model_name_or_path $modelpath \
            --train_file $trainpath \
            --validation_file $trainpath \
            --do_train \
            --per_device_train_batch_size 32 \
            --save_steps 100000 \
            --output_dir $DATA/gab-language-change/finetuned-models/reddit/month-models/train-rand/$(basename $modelpath)-train_rand_20k \
            --overwrite_output_dir \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --use_special_tokens
    done
done