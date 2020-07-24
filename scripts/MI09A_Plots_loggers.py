import sys
from MI_Classes import PlotsLoggers

# Options
display_learning_rate = True

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Generate results
Plots_Loggers = PlotsLoggers(target=sys.argv[1], display_learning_rate=display_learning_rate)
Plots_Loggers.generate_plots()

# Exit
print('Done.')
sys.exit(0)
