#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=2k-cumu-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=2k-cumu-finetune.out
#SBATCH --error=2k-cumu-finetune.err
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

# adapt/train data size
data_size=2

# transition from 2017 to 2018

for modelpath in $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_2017_12_${data_size}k/; do
    for trainpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_2018_01_${data_size}k.csv; do
        python run_finetuning.py \
            --model_name_or_path $modelpath \
            --train_file $trainpath \
            --validation_file $trainpath \
            --do_train \
            --per_device_train_batch_size 32 \
            --save_steps 100000 \
            --output_dir $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_2018_01_${data_size}k \
            --overwrite_output_dir \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --use_special_tokens
    done
done

# set year for which to adapt models (run separately and subsequently for subsequent years)
year="2018"

# set month to load initial model from
load_month="01"

for train_month in "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12"; do
    
    echo "loading model from:" $load_month $year
    echo "finetuning it with data from:" $train_month $year

    python run_finetuning.py \
        --model_name_or_path $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_${year}_${load_month}_${data_size}k \
        --train_file $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_${year}_${train_month}_${data_size}k.csv \
        --validation_file $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_${year}_${train_month}_2k.csv \
        --do_train \
        --per_device_train_batch_size 32 \
        --output_dir $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_${year}_${train_month}_${data_size}k \
        --overwrite_output_dir \
        --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --use_special_tokens

    # set load_month for next iteration to be adapt_month from this iteration
    load_month=$train_month
    
    # sleep just to make sure all the model saving from the adaptation script is done (not sure if necessary)
    sleep 15s

done

# transition from 2018 to 2019

for modelpath in $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_2018_12_${data_size}k/; do
    for trainpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_2019_01_${data_size}k.csv; do
        python run_finetuning.py \
            --model_name_or_path $modelpath \
            --train_file $trainpath \
            --validation_file $trainpath \
            --do_train \
            --per_device_train_batch_size 32 \
            --save_steps 100000 \
            --output_dir $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_2019_01_${data_size}k \
            --overwrite_output_dir \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --use_special_tokens
    done
done

# set year for which to adapt models (run separately and subsequently for subsequent years)
year="2019"

# set month to load initial model from
load_month="01"

for train_month in "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12"; do
    
    echo "loading model from:" $load_month $year
    echo "finetuning it with data from:" $train_month $year

    python run_finetuning.py \
        --model_name_or_path $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_${year}_${load_month}_${data_size}k \
        --train_file $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_${year}_${train_month}_${data_size}k.csv \
        --validation_file $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_${year}_${train_month}_2k.csv \
        --do_train \
        --per_device_train_batch_size 32 \
        --output_dir $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_${year}_${train_month}_${data_size}k \
        --overwrite_output_dir \
        --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --use_special_tokens

    # set load_month for next iteration to be adapt_month from this iteration
    load_month=$train_month
    
    # sleep just to make sure all the model saving from the adaptation script is done (not sure if necessary)
    sleep 15s

done

# transition from 2019 to 2020

for modelpath in $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_2019_12_${data_size}k/; do
    for trainpath in $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_2020_01_${data_size}k.csv; do
        python run_finetuning.py \
            --model_name_or_path $modelpath \
            --train_file $trainpath \
            --validation_file $trainpath \
            --do_train \
            --per_device_train_batch_size 32 \
            --save_steps 100000 \
            --output_dir $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_2020_01_${data_size}k \
            --overwrite_output_dir \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --use_special_tokens
    done
done

# set year for which to adapt models (run separately and subsequently for subsequent years)
year="2020"

# set month to load initial model from
load_month="01"

for train_month in "02"; do
    
    echo "loading model from:" $load_month $year
    echo "finetuning it with data from:" $train_month $year

    python run_finetuning.py \
        --model_name_or_path $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_${year}_${load_month}_${data_size}k \
        --train_file $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_${year}_${train_month}_${data_size}k.csv \
        --validation_file $DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_${year}_${train_month}_2k.csv \
        --do_train \
        --per_device_train_batch_size 32 \
        --output_dir $DATA/gab-language-change/finetuned-models/reddit/cumu-models/bert-base-train_cumu_${year}_${train_month}_${data_size}k \
        --overwrite_output_dir \
        --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --use_special_tokens

    # set load_month for next iteration to be adapt_month from this iteration
    load_month=$train_month
    
    # sleep just to make sure all the model saving from the adaptation script is done (not sure if necessary)
    sleep 15s

done