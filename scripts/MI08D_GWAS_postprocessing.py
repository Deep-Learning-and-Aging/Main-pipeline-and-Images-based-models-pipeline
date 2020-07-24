import sys
from MI_Classes import GWASPostprocessing

# Compute results
GWAS_Postprocessing = GWASPostprocessing()
GWAS_Postprocessing.processing_all_targets_and_organs()

# Exit
print('Done.')
sys.exit(0)
