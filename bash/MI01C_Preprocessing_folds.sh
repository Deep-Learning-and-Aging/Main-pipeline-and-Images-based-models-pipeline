#!/bin/bash
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --parsable
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
source ~/python_3.6.0/bin/activate
python -u ../scripts/MI01C_Preprocessing_folds.py $1 $2 && echo "PYTHON SCRIPT COMPLETED"

