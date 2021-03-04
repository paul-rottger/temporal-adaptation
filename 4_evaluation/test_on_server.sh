#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=test.out
#SBATCH --error=test.err
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

# Executing the finetuning script with set options
python run_test.py \
    --model_name_or_path $DATA/finetuned-models/bert-rand-1m-3ep-rand \
    --test_file $DATA/0_data/clean/labelled_ghc/eval_random.csv \
    --per_device_eval_batch_size 512 \
    --output_dir ./results \
    --output_name test123 \
    --overwrite_output_dir \
    --max_seq_length 128 \
    --use_special_tokens