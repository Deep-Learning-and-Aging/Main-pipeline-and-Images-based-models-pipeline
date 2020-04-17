#!/bin/bash
#SBATCH -p short
#SBATCH --open-mode=truncate
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lingsheng_dong@hms.harvard.edu
set -e
module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source ~/python_3.6.0/bin/activate
python -u ../scripts/MI09_Plot_correlations.ld.py Age B  && echo "PYTHON SCRIPT COMPLETED"

