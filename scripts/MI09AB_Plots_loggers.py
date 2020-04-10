from MI_Libraries import *
from MI_Classes import PlotsLoggers

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Read parameters from command
target = sys.argv[1]

# Options
display_learning_rate = True

# Generate results
Plots_Loggers = PlotsLoggers(target=sys.argv[1], display_learning_rate=display_learning_rate)
Plots_Loggers.generate_plots()

# Exit
print('Done.')
sys.exit(0)
