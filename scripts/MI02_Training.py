from MI_Libraries import *
from MI_Classes import Training

# Options
# Use a small subset of the data VS. run the actual full data pipeline to get accurate results
# /!\ if True, path to save weights will be automatically modified to avoid rewriting them
debug_mode = True
# Load weights from previous best training results, VS. start from scratch
continue_training = True
# Try to find a similar model among those already trained and evaluated to perform transfer learning
max_transfer_learning = False
# Compute the metrics during training on the train and val sets VS. only compute loss (faster)
display_full_metrics = False

# Default parameters
if len(sys.argv) != 10:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Heart_20208_3chambers')  # organ_id_view, e.g Heart_20208_3chambers.
    sys.argv.append('raw')  # transformation
    sys.argv.append('EfficientNetB7')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.0')  # dropout_rate
    sys.argv.append('1')  # outer_fold

# Compute results
Model_Training = Training(target=sys.argv[1], organ_id_view=sys.argv[2], transformation=sys.argv[3],
                          architecture=sys.argv[4], optimizer=sys.argv[5], learning_rate=sys.argv[6],
                          weight_decay=sys.argv[7], dropout_rate=sys.argv[8], outer_fold=sys.argv[9],
                          debug_mode=debug_mode, continue_training=continue_training,
                          max_transfer_learning=max_transfer_learning, display_full_metrics=display_full_metrics)
Model_Training.data_preprocessing()
Model_Training.build_model()
Model_Training.train_model()
Model_Training.clean_exit()
