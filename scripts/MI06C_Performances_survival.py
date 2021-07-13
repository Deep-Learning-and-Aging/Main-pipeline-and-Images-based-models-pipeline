import sys
from MI_Classes import PerformancesSurvival

# Default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # inner_fold
    sys.argv.append('instances')  # pred_type

# Options
debug_mode = True  # Only bootstrap 3 times instead of 1000 times

# Compute results
Performances_Survival = PerformancesSurvival(target=sys.argv[1], fold=sys.argv[2], pred_type=sys.argv[3],
                                             debug_mode=debug_mode)
Performances_Survival.load_data()
Performances_Survival.compute_c_index_and_save_data()
Performances_Survival.print_key_results()

# Exit
print('Done.')
sys.exit(0)
