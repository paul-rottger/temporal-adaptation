#!/bin/sh
# Executing the finetuning script with set options
for filename in ../0_data/clean/labelled_ghc/month_splits/*.csv; do
    
    python run_finetuning.py \
        --model_name_or_path ../0_models/default-model \
        --train_file $filename \
        --validation_file $filename \
        --do_train \
        --output_dir ../0_models/bert-base-$(basename $filename .csv) \
        --overwrite_output_dir \
        --max_steps 3 \
        --num_train_epochs 1 \
        --max_seq_length 128 \
        --use_special_tokens

done