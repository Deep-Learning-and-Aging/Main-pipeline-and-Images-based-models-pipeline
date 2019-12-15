#!/bin/bash
#SBATCH -p gpu
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source ~/python_3.6.0/bin/activate
python ../scripts/MI02_Training.py $1 $2 $3 $4 $5 $6 $7 $8 $9
