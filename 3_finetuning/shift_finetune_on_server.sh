#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=match-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=match-finetune.out
#SBATCH --error=match-finetune.err
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


for modelpath in $DATA/gab-language-change/adapted-models/reddit/month-models/bert*/; do
    for trainpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train*_20k.csv; do
        
        if [[ $(basename $modelpath | cut -c6-9) = $(basename $trainpath | cut -c7-10) ]] && \
        [[ $(( 10#$(basename $modelpath | cut -c11-12) +0 )) = $(( 10#$(basename $trainpath | cut -c12-13) +0 )) ]]
        then
            python run_finetuning.py \
                --model_name_or_path $modelpath \
                --train_file $trainpath \
                --validation_file $trainpath \
                --do_train \
                --per_device_train_batch_size 32 \
                --output_dir $DATA/gab-language-change/finetuned-models/reddit/month-models/match/$(basename $modelpath)-$(basename $trainpath .csv) \
                --overwrite_output_dir \
                --num_train_epochs 3 \
                --max_seq_length 128 \
                --use_special_tokens
        fi

    done
done