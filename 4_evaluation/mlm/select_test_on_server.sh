#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=1704-pmlm-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=1704-pmlm-test.out
#SBATCH --error=1704-pmlm-test.err
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


# select model for testing
model_path="$DATA/gab-language-change/adapted-models/reddit/month-models/bert-2017_04_1m/"


# select test sets for testing
for date in "2017 04" "2017 08" "2017 12" "2018 04" "2018 08" "2018 12" "2019 04" "2019 08" "2019 12"; do
    
    arr=($date)
    test_year=$((10#${arr[0]}))
    test_month=$((10#${arr[1]}))

    test_path="$DATA/gab-language-change/0_data/clean/unlabelled_reddit/politics_test/test_${arr[0]}_${arr[1]}_5k.txt"

    echo $(basename $model_path) $(basename $test_path)

    python test_mlm.py \
        --model_name_or_path $model_path \
        --validation_file $test_path \
        --use_special_tokens \
        --line_by_line \
        --do_eval \
        --per_device_eval_batch_size 256 \
        --output_dir $DATA/gab-language-change/eval-results/mlm/reddit/politics-test \
        --output_name $(basename $model_path)-$(basename $test_path .txt) \
        --overwrite_output_dir \
        --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
        --max_seq_length 128

done