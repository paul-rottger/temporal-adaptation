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
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk
#SBATCH --output=mlm.out
#SBATCH --error=mlm.err

# reset modules
module purge

# load python module
module load python/anaconda3/2019.03

# activate the right conda environment
source activate $DATA/conda-envs/gab-language-change

python run_mlm.py \
    --model_name_or_path $DATA/bert-base-uncased \
    --train_file ../0_data/clean/train.txt \
    --validation_file ../0_data/clean/eval.txt \
    --line_by_line \
    --do_train \
    --do_eval \
    --output_dir $DATA/adapted-models/test-mlm \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_seq_length 64