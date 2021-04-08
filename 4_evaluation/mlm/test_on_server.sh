#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=m-mlm-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=m-mlm-test.out
#SBATCH --error=m-mlm-test.err
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

for modelpath in $DATA/gab-language-change/adapted-models/reddit/month-models/bert-*/; do
    for testpath in $DATA/gab-language-change/0_data/clean/unlabelled_reddit/total/test_rand_10k.txt; do

        echo $(basename $modelpath) $(basename $testpath)

        python test_mlm.py \
            --model_name_or_path $modelpath \
            --validation_file $testpath \
            --use_special_tokens \
            --line_by_line \
            --do_eval \
            --per_device_eval_batch_size 256 \
            --output_dir $DATA/gab-language-change/eval-results/mlm/reddit/test-on-rand \
            --output_name $(basename $modelpath)-$(basename $testpath .txt) \
            --overwrite_output_dir \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --max_seq_length 128

    done
done