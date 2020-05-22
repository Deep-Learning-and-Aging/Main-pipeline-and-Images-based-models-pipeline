#!/bin/bash
#SBATCH -p gpu
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
set -e
module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.1
source ~/python_3.6.0/bin/activate
python -u ../scripts/MI09D_Plots_attentionmaps.py $1 $2 $3 $4 && echo "PYTHON SCRIPT COMPLETED"

