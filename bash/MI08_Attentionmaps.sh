#!/bin/bash
#SBATCH --parsable
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu
#SBATCH -x compute-g-16-175,compute-g-16-176,compute-g-16-197

set -e
module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.1
source /home/al311/python_3.6.0/bin/activate

python -u ../scripts/MI08_Attentionmaps.py $1 $2 $3 $4 && echo "PYTHON SCRIPT COMPLETED"

