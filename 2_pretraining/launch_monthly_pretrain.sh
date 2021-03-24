#!/bin/bash

for filename in $DATA/gab-language-change/0_data/clean/unlabelled_reddit/month_splits/train*03_1m.txt; do
    sbatch ./monthly_pretrain_on_server.sh $filename
    sleep 1
done