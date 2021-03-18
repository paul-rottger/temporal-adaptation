#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=finetune.out
#SBATCH --error=finetune.err
#SBATCH --gres=gpu:k80:1

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
#for i in 01 02 03 04 05 06 07 08 09 10; do
for modelpath in $DATA/gab-language-change/adapted-models/month-models/*/; do

    python run_finetuning.py \
        --model_name_or_path $modelpath \
        --train_file $DATA/gab-language-change/0_data/clean/labelled_gq/train_random.csv \
        --validation_file $DATA/gab-language-change/0_data/clean/labelled_gq/eval_random.csv \
        --do_train \
        --per_device_train_batch_size 32 \
        --do_eval \
        --per_device_eval_batch_size 128 \
        --output_dir $DATA/gab-language-change/finetuned-models/gq/$(basename $modelpath)-train-gq \
        --overwrite_output_dir \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --use_special_tokens

done