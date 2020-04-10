from MI_Libraries import *
from MI_Classes import PredictionsGenerate

# options
# debug mode: exclude train set
debug_mode = True
# save predictions
save_predictions = True

# Default parameters
if len(sys.argv) != 9:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Heart_20208_3chambers')  # organ_id_view, e.g PhysicalActivity_90001_main, or Heart_20208_3chambers
    sys.argv.append('raw')  # transformation
    sys.argv.append('InceptionResNetV2')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.0')  # dropout

# Compute results
Predictions_Generate = PredictionsGenerate(target=sys.argv[1], organ_id_view=sys.argv[2], transformation=sys.argv[3],
                                           architecture=sys.argv[4], optimizer=sys.argv[5], learning_rate=sys.argv[6],
                                           weight_decay=sys.argv[7], dropout_rate=sys.argv[8], debug_mode=debug_mode)
Predictions_Generate.generate_predictions()
if save_predictions:
    Predictions_Generate.save_predictions()

# Exit
print('Done.')
sys.exit(0)
