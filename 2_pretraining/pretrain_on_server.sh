#!/bin/sh
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=10:00:00
# set name of job
#SBATCH --job-name=test123
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=paul.roettger@yahoo.de
#SBATCH --output=mlm.out
#SBATCH --error=mlm.err

python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file ../0_data/clean/train.txt \
    --validation_file ../0_data/clean/eval.txt \
    --use_special_tokens \
    --cache_dir $DATA/huggingface-default-models \
    --line_by_line \
    --do_train \
    --do_eval \
    --output_dir ./test-mlm \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 64