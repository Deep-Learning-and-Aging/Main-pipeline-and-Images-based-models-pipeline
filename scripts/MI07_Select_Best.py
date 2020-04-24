import sys
from MI_Classes import SelectBest

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Compute results
Select_Best = SelectBest(target=sys.argv[1])
Select_Best.select_models()
Select_Best.save_data()

# Exit
print('Done.')
sys.exit(0)
