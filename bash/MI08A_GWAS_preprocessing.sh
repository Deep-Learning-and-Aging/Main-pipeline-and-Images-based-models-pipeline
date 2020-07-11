#!/bin/bash
#SBATCH --job-name=MI08A.job
#SBATCH --output=../eo/MI08A.out
#SBATCH --error=../eo/MI08A.err
#SBATCH --mem-per-cpu=8G 
#SBATCH -c 1
#SBATCH -t 20
#SBATCH --parsable 
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
source ~/python_3.6.0/bin/activate
python -u ../scripts/MI08A_GWAS_preprocessing.py && echo "PYTHON SCRIPT COMPLETED"

