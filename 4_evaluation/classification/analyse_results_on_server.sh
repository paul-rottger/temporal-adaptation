#!/bin/sh

# reset modules
module purge

# load python module
module load python/anaconda3/2019.03

# activate the right conda environment
source activate $DATA/conda-envs/gab-language-change

# Read results and provide analysis
python analyse_results.py 
# watch out for hard-coded paths in .py script! (to do --> implement python argument input)