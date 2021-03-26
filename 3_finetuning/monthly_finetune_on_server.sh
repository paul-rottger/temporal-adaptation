#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=m17-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=finetune-monthly17.out
#SBATCH --error=finetune-monthly17.err
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


# Finetune the base model on each month-set of training data
for filename in $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_2017*_20k.csv; do
    
    python run_finetuning.py \
        --model_name_or_path $DATA/gab-language-change/default-models/bert-base-uncased \
        --train_file $filename \
        --validation_file $filename \
        --do_train \
        --per_device_train_batch_size 32 \
        --output_dir $DATA/gab-language-change/finetuned-models/reddit/month-models/bert-base-$(basename $filename .csv) \
        --overwrite_output_dir \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --use_special_tokens

done