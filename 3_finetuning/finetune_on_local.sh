#!/bin/sh
# Executing the pretraining script with set options
python run_mlm.py \
    --model_name_or_path ../0_models/default-model \
    --train_file ../0_data/clean/train.txt \
    --validation_file ../0_data/clean/eval.txt \
    --do_train \
    --do_eval \
    --output_dir ./test-mlm \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 64