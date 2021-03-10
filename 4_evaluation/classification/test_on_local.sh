#!/bin/sh
# Executing the finetuning script with set options
python run_test.py \
    --model_name_or_path ../0_models/bert-rand-1m-3ep-rand \
    --test_file ../0_data/clean/labelled_ghc/eval_random_small.csv \
    --per_device_eval_batch_size 16 \
    --output_dir ./results \
    --output_name bert-rand-1m-3ep-rand \
    --overwrite_output_dir \
    --max_seq_length 128 \
    --use_special_tokens