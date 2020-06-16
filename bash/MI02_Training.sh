#!/bin/bash
#SBATCH -p gpu
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.1
source ~/python_3.6.0/bin/activate
python -u ../scripts/MI02_Training.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} && echo "PYTHON SCRIPT COMPLETED"

if [-f ../eo/$SLURM_JOBID]
 then
  rm ../eo/$SLURM_JOBID
  exit 0
fi

