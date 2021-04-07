#!/bin/sh

#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --job-name=2ks+1-finetune
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=2ks+1-finetune.out
#SBATCH --error=2ks+1-finetune.err
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

# training set size (2 or 20k for reddit)
training_size=2

# number of months to shift (ft + add = adapt)
add=1

# zero for prepending, making single digit numbers have leading zero
zero="0"

# manually adjust the below to end at max_time - add (e.g. 2020 01 if add = 1)
for date in "2017 03" "2017 04" "2017 05" "2017 06" "2017 07" "2017 08" "2017 09" "2017 10" "2017 11" "2017 12" \
"2018 01" "2018 02" "2018 03" "2018 04" "2018 05" "2018 06" "2018 07" "2018 08" "2018 09" "2018 10" "2018 11" "2018 12" \
"2019 01" "2019 02" "2019 03" "2019 04" "2019 05" "2019 06" "2019 07" "2019 08" "2019 09" "2019 10" "2019 11" "2019 12" \
"2020 01" ; do
    
    arr=($date)
    ft_year=$((10#${arr[0]}))
    ft_month=$((10#${arr[1]}))

    train_path="$DATA/gab-language-change/0_data/clean/labelled_reddit/month_splits/train_${arr[0]}_${arr[1]}_${training_size}k.csv"

    if [ $(($ft_month + $add)) -gt 12 ] # if month + add falls into next year
    then
        adapt_year=$(($ft_year + 1))
        adapt_month=$zero$(($ft_month + $add - 12))
        adapt_month="${adapt_month:(-2)}"

        model_path="$DATA/gab-language-change/adapted-models/reddit/month-models/bert-${adapt_year}_${adapt_month}_1m/"

        echo "model adapted to" $adapt_year $adapt_month "finetuned on" ${arr[0]} ${arr[1]}

        python run_finetuning.py \
            --model_name_or_path $model_path \
            --train_file $train_path \
            --validation_file $train_path \
            --do_train \
            --per_device_train_batch_size 32 \
            --output_dir $DATA/gab-language-change/finetuned-models/reddit/month-models/shift+1/$(basename $model_path)-$(basename $train_path .csv) \
            --overwrite_output_dir \
            --save_steps 100000 \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --use_special_tokens


    else # if month + add falls into same year
        adapt_year=$(($ft_year)) 
        adapt_month=$zero$(($ft_month + $add))
        adapt_month="${adapt_month:(-2)}"

        model_path="$DATA/gab-language-change/adapted-models/reddit/month-models/bert-${adapt_year}_${adapt_month}_1m/"

        echo "model adapted to" $adapt_year $adapt_month "finetuned on" ${arr[0]} ${arr[1]}

        python run_finetuning.py \
            --model_name_or_path $model_path \
            --train_file $train_path \
            --validation_file $train_path \
            --do_train \
            --per_device_train_batch_size 32 \
            --output_dir $DATA/gab-language-change/finetuned-models/reddit/month-models/shift+1/$(basename $model_path)-$(basename $train_path .csv) \
            --overwrite_output_dir \
            --save_steps 100000 \
            --dataset_cache_dir $DATA/gab-language-change/z_cache/datasets \
            --num_train_epochs 3 \
            --max_seq_length 128 \
            --use_special_tokens

    fi
done
