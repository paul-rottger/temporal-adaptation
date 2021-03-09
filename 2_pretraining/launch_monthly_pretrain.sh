#!/bin/bash

for filename in $DATA/gab-language-change/0_data/clean/unlabelled_pushshift/month_splits/train*.txt; do
    sbatch ./monthly_pretrain_on_server.sh $filename
    sleep 1
done