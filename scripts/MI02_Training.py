import sys
from MI_Classes import Training

# Options
# Use a small subset of the data VS. run the actual full data pipeline to get accurate results
# /!\ if True, path to save weights will be automatically modified to avoid rewriting them
debug_mode = False
# Load weights from previous best training results, VS. start from scratch
continue_training = True
# Try to find a similar model among those already trained and evaluated to perform transfer learning
max_transfer_learning = False
# Compute all the metrics during training VS. only compute loss and main metric (faster)
display_full_metrics = False

# Default parameters
if len(sys.argv) != 12:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Liver')  # organ
    sys.argv.append('main')  # view
    sys.argv.append('contrast')  # transformation
    sys.argv.append('InceptionResNetV2')  # architecture
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.000001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.0')  # dropout_rate
    sys.argv.append('0.1')  # data_augmentation_factor
    sys.argv.append('0')  # outer_fold

# Compute results
Model_Training = Training(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3], transformation=sys.argv[4],
                          architecture=sys.argv[5], optimizer=sys.argv[6], learning_rate=sys.argv[7],
                          weight_decay=sys.argv[8], dropout_rate=sys.argv[9], data_augmentation_factor=sys.argv[10],
                          outer_fold=sys.argv[11], debug_mode=debug_mode, continue_training=continue_training,
                          max_transfer_learning=max_transfer_learning, display_full_metrics=display_full_metrics)
Model_Training.data_preprocessing()
Model_Training.build_model()
Model_Training.train_model()
Model_Training.clean_exit()

main_metric_name = self.dict_main_metrics_names[self.target]
main_metric_mode = self.main_metrics_modes[main_metric_name]
Perf_col_name = main_metric_name + '_all'
for model in self.models:
    Performances_model = self.Performances[self.Performances['model'] == model]
    Performances_model.sort_values([Perf_col_name, 'learning_rate', 'dropout_rate', 'weight_decay',
                                    'data_augmentation_factor'],
                                   ascending=[main_metric_mode == 'min', False, False, False, True], inplace=True)
    best_version = Performances_model['version'][
        Performances_model[Perf_col_name] == Performances_model[Perf_col_name].max()].values[0]
    versions_to_drop = [version for version in Performances_model['version'].values if
                        not version == best_version]
    # define columns from predictions to drop
    cols_to_drop = ['pred_' + version for version in versions_to_drop] + ['outer_fold_' + version for version in
                                                                          versions_to_drop]
    for fold in self.folds:
        self.PERFORMANCES[fold].drop(versions_to_drop, inplace=True)
        self.PREDICTIONS[fold].drop(cols_to_drop, inplace=True, axis=1)

# drop 'model' column
self.Performances.drop(['model'], inplace=True, axis=1)

# Display results
for fold in self.folds:
    print('The tuned ' + fold + ' performances are:')
    print(self.PERFORMANCES[fold])