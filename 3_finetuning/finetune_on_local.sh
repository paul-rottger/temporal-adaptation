#!/bin/sh
# Executing the finetuning script with set options
python run_finetuning.py \
    --model_name_or_path ../0_models/bert-random-1m-3ep \
    --train_file ../0_data/clean/train.txt \
    --validation_file ../0_data/clean/eval.txt \
    --do_train \
    --do_eval \
    --output_dir ../0_models/test-mlm \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 64