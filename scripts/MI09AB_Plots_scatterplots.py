import sys
from MI_Classes import PlotsScatter

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('eids')  # pred_type

# Generate results
Plots_Scatter = PlotsScatter(target=sys.argv[1], pred_type=sys.argv[2])
Plots_Scatter.generate_plots()

# Exit
print('Done.')
sys.exit(0)
