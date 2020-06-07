import sys
from MI_Classes import PlotsCorrelations

# Default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # fold
    sys.argv.append('eids')  # pred_type

# Options
save_figures = True

# Compute results
Plots_Correlations = PlotsCorrelations(target=sys.argv[1], fold=sys.argv[2], pred_type=sys.argv[3],
                                       save_figures=save_figures)
Plots_Correlations.preprocessing()
Plots_Correlations.generate_plots()

# Exit
print('Done.')
sys.exit(0)
