#!/bin/sh
# Executing the finetuning script with set options
python run_finetuning.py \
    --model_name_or_path ../0_models/default-model \
    --train_file ../0_data/clean/labelled_ghc/train_random.csv \
    --validation_file ../0_data/clean/labelled_ghc/eval_random.csv \
    --test_file ../0_data/clean/labelled_ghc/eval_random.csv \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir ../0_models/test-mlm \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 128 \
    --use_special_tokens