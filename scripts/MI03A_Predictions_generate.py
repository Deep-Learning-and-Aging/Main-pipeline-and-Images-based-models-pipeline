import sys
from MI_Classes import PredictionsGenerate

# options
# debug mode
debug_mode = True
# save predictions
save_predictions = True


# Default parameters
if len(sys.argv) != 10:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Heart_20208')  # organ_id, Heart_20208
    sys.argv.append('3chambers')  # view
    sys.argv.append('raw')  # transformation
    sys.argv.append('InceptionResNetV2')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.0000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.0')  # dropout

# Compute results
Predictions_Generate = PredictionsGenerate(target=sys.argv[1], organ_id=sys.argv[2], view=sys.argv[3],
                                           transformation=sys.argv[4], architecture=sys.argv[5], optimizer=sys.argv[6],
                                           learning_rate=sys.argv[7], weight_decay=sys.argv[8],
                                           dropout_rate=sys.argv[9], debug_mode=debug_mode)
Predictions_Generate.generate_predictions()
if save_predictions:
    Predictions_Generate.save_predictions()

# Exit
print('Done.')
Predictions_Generate.clean_exit()
