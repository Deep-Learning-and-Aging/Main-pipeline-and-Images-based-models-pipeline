from MI_Libraries import *
from MI_Classes import ResidualsCorrelations

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # fold

# Options
debug_mode = True

# Compute results
Residuals_Correlations = ResidualsCorrelations(target=sys.argv[1], fold=sys.argv[2], debug_mode=debug_mode)
Residuals_Correlations.preprocessing()
Residuals_Correlations.generate_correlations()
Residuals_Correlations.save_correlations()

# Exit
print('Done.')
sys.exit(0)
