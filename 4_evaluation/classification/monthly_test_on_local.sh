#!/bin/sh
# Executing the finetuning script with set options

for modelpath in ../0_models/*/; do
    for testpath in ../0_data/clean/labelled_ghc/month_splits/test*.csv; do
    
    echo $(basename $modelpath)-$(basename $testpath .csv)
    
    python run_test.py \
        --model_name_or_path $modelpath \
        --test_file $testpath \
        --per_device_eval_batch_size 16 \
        --output_dir ../0_results/month-models \
        --output_name $(basename $modelpath)-$(basename $testpath .csv) \
        --overwrite_output_dir \
        --max_seq_length 128 \
        --max_steps 1 \
        --use_special_tokens

    done
done