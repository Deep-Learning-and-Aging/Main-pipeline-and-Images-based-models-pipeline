import sys
from MI_Classes import GWASPostprocessing

# Compute results
GWAS_Postprocessing = GWASPostprocessing()
GWAS_Postprocessing.processing_all_targets_and_organs()
GWAS_Postprocessing.parse_heritability_scores()

# Exit
print('Done.')
sys.exit(0)

remls = []
files = glob.glob('../eo/MI08C*_reml.out')
for file in files:
    remls.append(reml.split('_')[1:3])

remls = pd.DataFrame(remls, columns=['target', 'organ'])
ORGANS_REML = {}
for target in remls['target'].unique():
    remls_target = remls[remls['target'] == target]
    ORGANS_REML[target] = remls_target['organ'].unique()