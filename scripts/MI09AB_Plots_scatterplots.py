import sys
from MI_Classes import PlotsScatter

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Generate results
Plots_Scatter = PlotsScatter(target=sys.argv[1])
Plots_Scatter.generate_plots()

# Exit
print('Done.')
sys.exit(0)
