import sys
from MI_Classes import PreprocessingFolds

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Heart')  # organ

# Compute results
Preprocessing_Folds = PreprocessingFolds(target=sys.argv[1], organ=sys.argv[2])
Preprocessing_Folds.generate_folds()

# Exit
print('Done.')
sys.exit(0)
