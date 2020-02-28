#!/bin/bash
#SBATCH -p short
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source ~/python_3.6.0/bin/activate
python -u ../scripts/MI05A_Ensembles_predictions.py $1 $2 && echo "PYTHON SCRIPT COMPLETED"
