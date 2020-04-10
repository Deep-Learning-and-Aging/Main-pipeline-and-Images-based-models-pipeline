from MI_Libraries import *
from MI_Classes import PlotsCorrelations

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # fold

# Options
save_figures = True

# Compute results
Plots_Correlations = PlotsCorrelations(target=sys.argv[1], fold=sys.argv[2], save_figures=save_figures)
Plots_Correlations.preprocessing()
Plots_Correlations.generate_plots()

# Exit
print('Done.')
sys.exit(0)
