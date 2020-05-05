import sys
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
if len(sys.argv) != 11:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Heart_20208')  # organ_id, e.g Heart_20208.  EyeFundus_210156
    sys.argv.append('2chambers')  # view
    sys.argv.append('raw')  # transformation
    sys.argv.append('EfficientNetB7')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.0')  # dropout_rate
    sys.argv.append('0')  # outer_fold

# Compute results
Model_Training = Training(target=sys.argv[1], organ_id=sys.argv[2], view=sys.argv[3], transformation=sys.argv[4],
                          architecture=sys.argv[5], optimizer=sys.argv[6], learning_rate=sys.argv[7],
                          weight_decay=sys.argv[8], dropout_rate=sys.argv[9], outer_fold=sys.argv[10],
                          debug_mode=debug_mode, continue_training=continue_training,
                          max_transfer_learning=max_transfer_learning, display_full_metrics=display_full_metrics)
Model_Training.data_preprocessing()
Model_Training.build_model()
Model_Training.train_model()
Model_Training.clean_exit()
