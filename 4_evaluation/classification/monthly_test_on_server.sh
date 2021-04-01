#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=m17-adarand-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=m17-adarand-test.out
#SBATCH --error=m17-adarand-test.err
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


# Executing the finetuning script with set options

for modelpath in $DATA/gab-language-change/finetuned-models/reddit/month-models/ada-rand/bert*2017*/; do
    for testpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/test*5k.csv; do
    
    echo $(basename $modelpath)-$(basename $testpath .csv)
    
    python run_test.py \
        --model_name_or_path $modelpath \
        --test_file $testpath \
        --per_device_eval_batch_size 256 \
        --output_dir $DATA/gab-language-change/eval-results/reddit/classification/month-models/ada-rand \
        --output_name $(basename $modelpath)-$(basename $testpath .csv) \
        --overwrite_output_dir \
        --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
        --max_seq_length 128 \
        --use_special_tokens

    done
done