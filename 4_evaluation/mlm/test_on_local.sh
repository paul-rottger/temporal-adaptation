#!/bin/sh
# Executing the pretraining script with set options

for modelpath in ../../0_models/*/; do
    for testpath in ../../0_data/clean/unlabelled_pushshift/month_splits/total/test_s*.txt; do

        echo $(basename $modelpath) $(basename $testpath)

        python test_mlm.py \
            --model_name_or_path $modelpath \
            --validation_file $testpath \
            --use_special_tokens \
            --line_by_line \
            --do_eval \
            --per_device_eval_batch_size 64 \
            --output_dir ./test-mlm \
            --output_name $(basename $modelpath)-$(basename $testpath .txt | cut -c6) \
            --overwrite_output_dir \
            --num_train_epochs 1 \
            --max_seq_length 128

    done
done