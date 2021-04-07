#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=2k17-cumu-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=2k17-cumu-test.out
#SBATCH --error=2k17-cumu-test.err
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

for modelpath in $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert*2017*_2k/; do
    for testpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/test*_5k.csv; do
    
    echo $(basename $modelpath)-$(basename $testpath .csv)
    
    python run_test.py \
        --model_name_or_path $modelpath \
        --test_file $testpath \
        --per_device_eval_batch_size 256 \
        --output_dir $DATA/gab-language-change/eval-results/classification/reddit/cumu-models/2k \
        --output_name $(basename $modelpath)-$(basename $testpath .csv) \
        --overwrite_output_dir \
        --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
        --max_seq_length 128 \
        --use_special_tokens

    done
done