import sys
from MI_Classes import PredictionsGenerate

# options
# debug mode
debug_mode = False
# save predictions
save_predictions = True

# Default parameters
if len(sys.argv) != 13:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Heart')  # organ
    sys.argv.append('4chambers')  # view
    sys.argv.append('contrast')  # transformation
    sys.argv.append('InceptionV3')  # architecture
    sys.argv.append('0')  # n_fc_layers
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.0001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.0')  # dropout_rate
    sys.argv.append('2.0')  # data_augmentation_factor
    sys.argv.append('0')  # outer_fold

# Compute results
Predictions_Generate = PredictionsGenerate(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3],
                                           transformation=sys.argv[4], architecture=sys.argv[5],
                                           n_fc_layers=sys.argv[6], optimizer=sys.argv[7], learning_rate=sys.argv[8],
                                           weight_decay=sys.argv[9], dropout_rate=sys.argv[10],
                                           data_augmentation_factor=sys.argv[11], outer_fold=sys.argv[12],
                                           debug_mode=debug_mode)
Predictions_Generate.generate_predictions()
if save_predictions:
    Predictions_Generate.save_predictions()

# Exit
Predictions_Generate.clean_exit()
