#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=pretrain
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=mlm.out
#SBATCH --error=mlm.err
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

python run_mlm.py \
    --model_name_or_path $DATA/gab-language-change/default-models/bert-base-uncased \
    --train_file $DATA/0_data/clean/unlabelled_pushshift/train_random_10m.txt \
    --validation_file $DATA/0_data/clean/unlabelled_pushshift/eval_random_10k.txt \
    --save_steps 25000 \
    --use_special_tokens \
    --line_by_line \
    --do_train \
    --per_device_train_batch_size 128 \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --evaluation_strategy epoch \
    --output_dir $DATA/gab-language-change/adapted-models/bert-random-10m \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 128
