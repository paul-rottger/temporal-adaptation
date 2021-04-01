#!/bin/bash

for filename in $DATA/gab-language-change/0_data/clean/unlabelled_reddit/month_splits/train*08_1m.txt; do
    echo $filename
    sbatch ./monthly_pretrain_on_server.sh $filename
    sleep 1
done