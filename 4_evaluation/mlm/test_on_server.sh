#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=mlm-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=mlm-test.out
#SBATCH --error=mlm-test.err
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

for modelpath in $DATA/gab-language-change/adapted-models/total_models_models/bert-rand-10m/; do
    for testpath in $DATA/gab-language-change/0_data/clean/unlabelled_pushshift/month_splits/test_*.txt; do

        echo $(basename $modelpath) $(basename $testpath)

        python test_mlm.py \
            --model_name_or_path $modelpath \
            --validation_file $testpath \
            --use_special_tokens \
            --line_by_line \
            --do_eval \
            --per_device_eval_batch_size 256 \
            --output_dir $DATA/gab-language-change/eval-results/predictions/mlm/pseudo-perplexity \
            --output_name $(basename $modelpath)-$(basename $testpath .txt | cut -c11-) \
            --overwrite_output_dir \
            --max_seq_length 128

    done
done