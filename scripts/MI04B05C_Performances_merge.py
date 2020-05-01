import sys
from MI_Classes import PerformancesMerge
    
# Default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # inner_fold
    sys.argv.append('True')  # ensemble_models. Set False for MI04B and True for MI05B

# Compute results
Performances_Merge = PerformancesMerge(target=sys.argv[1], fold=sys.argv[2], ensemble_models=sys.argv[3])
Performances_Merge.merge_performances()
Performances_Merge.save_performances()

# Exit
print('Done.')
sys.exit(0)
