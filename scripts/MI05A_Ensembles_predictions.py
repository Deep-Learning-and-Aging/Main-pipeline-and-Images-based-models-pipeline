import sys
from MI_Classes import EnsemblesPredictions

# Options
regenerate_models = False  # False = Only compute ensemble model if it was not already computed

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('eids')  # pred_type

# Compute results
Ensembles_Predictions = EnsemblesPredictions(target=sys.argv[1], pred_type=sys.argv[2],
                                             regenerate_models=regenerate_models)
Ensembles_Predictions.load_data()
Ensembles_Predictions.generate_ensemble_predictions()
Ensembles_Predictions.save_predictions()

# Exit
print('Done.')
sys.exit(0)



self = Ensembles_Predictions
import pandas as pd
import re
from multiprocessing import Pool

version = 'Age_*_*_*_*_*_*_*_*_*_*_*'
print('Building the ensemble model ' + version)
Predictions = self.PREDICTIONS['val']
# Select the outerfolds columns for the model
ensemble_outerfolds_cols = [col for col in Predictions.columns.values if
                            bool(re.compile('outer_fold_' + version).match(col))][:-1]
ensemble_preds_cols = [col.replace('outer_fold_', 'pred_') for col in ensemble_outerfolds_cols]
# Select the rows for the model
Ensemble_preds = Predictions[ensemble_preds_cols]
Ensemble_outerfolds = Predictions[ensemble_outerfolds_cols]
Ensemble_outerfolds = Ensemble_outerfolds[~Ensemble_preds.isna().all(1)]

PREDICTIONS_OUTERFOLDS = {}
ENSEMBLE_INPUTS = {}
outer_fold = '0'
# take the subset of the rows that correspond to the outer_fold
PREDICTIONS_OUTERFOLDS[outer_fold] = {}
XS_outer_fold = {}
YS_outer_fold = {}
fold = 'val'
Ensemble_outerfolds_fold = self.PREDICTIONS[fold][ensemble_outerfolds_cols]
self.PREDICTIONS[fold]['outer_fold_' + version] = Ensemble_outerfolds_fold.mean(axis=1, skipna=False)
PREDICTIONS_OUTERFOLDS[outer_fold][fold] = self.PREDICTIONS[fold][
    self.PREDICTIONS[fold]['outer_fold_' + version] == float(outer_fold)]
X = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['id', 'eid', 'instance'] + ensemble_preds_cols]
X.set_index('id', inplace=True)
XS_outer_fold[fold] = X
y = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['id', 'eid', self.target]]
y.set_index('id', inplace=True)
YS_outer_fold[fold] = y
ENSEMBLE_INPUTS[outer_fold] = [XS_outer_fold['val'], YS_outer_fold['val']]
# Build ensemble model
pool = Pool(self.N_ensemble_CV_split)

# Useful for EnsemblesPredictions. This function needs to be global to allow pool to pickle it.
def compute_ensemble_folds(ensemble_inputs):
    if len(ensemble_inputs[1]) < 100:
        print('small sample size:' + str(len(ensemble_inputs[1])))
        print(ensemble_inputs[1])
        n_inner_splits = 2
        n_iter = 500
    else:
        n_inner_splits = 10
        n_iter = 30
    cv = InnerCV(models=['ElasticNet'], inner_splits=n_inner_splits, n_iter=n_iter) #, 'LightGBM', 'NeuralNetwork']
    model = cv.optimize_hyperparameters(ensemble_inputs[0], ensemble_inputs[1], scoring='r2')
    return model

MODELS = pool.map(compute_ensemble_folds, list(ENSEMBLE_INPUTS.values()))
pool.close()
pool.join()