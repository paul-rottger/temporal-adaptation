#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=10:00:00
#SBATCH --job-name=PR-test-030321
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=mlm.out
#SBATCH --error=mlm.err
#SBATCH --gres=gpu:1

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
    --model_name_or_path $DATA/bert-base-uncased \
    --train_file ../0_data/clean/train.txt \
    --validation_file ../0_data/clean/eval.txt \
    --use_special_tokens \
    --line_by_line \
    --do_train \
    --do_eval \
    --output_dir $DATA/adapted-models/test-mlm \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --max_seq_length 64