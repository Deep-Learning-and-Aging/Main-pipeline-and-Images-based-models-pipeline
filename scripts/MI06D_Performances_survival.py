import sys
from MI_Classes import PerformancesSurvival

# Options
# Only bootstrap 3 times instead of 1000 times
debug_mode = True

# Default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # inner_fold
    sys.argv.append('eids')  # pred_type

# Compute results
Performances_Survival = PerformancesSurvival(target=sys.argv[1], fold=sys.argv[2], pred_type=sys.argv[3],
                                             debug_mode=debug_mode)
Performances_Survival.load_data()
Performances_Survival.compute_CIs_and_save_data()

# Exit
print('Done.')
sys.exit(0)
