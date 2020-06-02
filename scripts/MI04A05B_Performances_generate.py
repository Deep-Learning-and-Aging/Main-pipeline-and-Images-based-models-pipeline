import sys
from MI_Classes import PerformancesGenerate

# Options
# Use a small number for the bootstrapping
debug_mode = True

# Default parameters
if len(sys.argv) != 11:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Liver')  # organ
    sys.argv.append('main')  # view
    sys.argv.append('raw')  # transformation
    sys.argv.append('InceptionV3')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.2')  # dropout
    sys.argv.append('test')  # fold

# Default parameters for ensemble models
# if len(sys.argv) != 11:
#     print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
#     sys.argv = ['']
#     sys.argv.append('Age')  # target
#     sys.argv.append('*')  # organ
#     sys.argv.append('*')  # view
#     sys.argv.append('*')  # transformation
#     sys.argv.append('*')  # architecture
#     sys.argv.append('*')  # optimizer
#     sys.argv.append('*')  # learning_rate
#     sys.argv.append('*')  # weight decay
#     sys.argv.append('*')  # dropout
#     sys.argv.append('val')  # fold

# Compute results
Performances_Generate = PerformancesGenerate(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3],
                                             transformation=sys.argv[4], architecture=sys.argv[5],
                                             optimizer=sys.argv[6], learning_rate=sys.argv[7], weight_decay=sys.argv[8],
                                             dropout_rate=sys.argv[9], fold=sys.argv[10], debug_mode=False)
Performances_Generate.preprocessing()
Performances_Generate.compute_performances()
Performances_Generate.save_performances()

# Exit
print('Done.')
sys.exit(0)
