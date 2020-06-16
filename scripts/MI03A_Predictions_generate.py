import sys
from MI_Classes import PredictionsGenerate

# options
# debug mode
debug_mode = False
# save predictions
save_predictions = True

# Default parameters
if len(sys.argv) != 12:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Pancreas')  # organ
    sys.argv.append('main')  # view
    sys.argv.append('raw')  # transformation
    sys.argv.append('InceptionV3')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.2')  # dropout
    sys.argv.append('0.1')  # data_augmentation_factor
    sys.argv.append('0')  # outer_fold

# Compute results
Predictions_Generate = PredictionsGenerate(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3],
                                           transformation=sys.argv[4], architecture=sys.argv[5], optimizer=sys.argv[6],
                                           learning_rate=sys.argv[7], weight_decay=sys.argv[8],
                                           dropout_rate=sys.argv[9], data_augmentation_factor=sys.argv[10],
                                           outer_fold=sys.argv[11], debug_mode=debug_mode)
Predictions_Generate.generate_predictions()
if save_predictions:
    Predictions_Generate.save_predictions()

# Exit
Predictions_Generate.clean_exit()
