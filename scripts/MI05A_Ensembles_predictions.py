import sys
from MI_Classes import EnsemblesPredictions

# Options
regenerate_models = True  # False = Only compute ensemble model if it was not already computed

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
#Ensembles_Predictions.save_predictions()

self = Ensembles_Predictions
for fold in ['val', 'test']:
    df = self.PREDICTIONS_outerfold[fold].drop(columns=['outer_fold_Age_Eyes_AllBiomarkers_Raw_*_*_*_*_*_*_*_*',
                                                        'pred_Age_Eyes_AllBiomarkers_Raw_*_*_*_*_*_*_*_*'])
    df.to_csv('../for_Samuel/' + fold + '.csv', index=False)


# Exit
print('Done.')
sys.exit(0)

pool = Pool(self.N_ensemble_CV_split)
MODELS = pool.map(self._compute_ensemble_fold, ENSEMBLE_INPUTS)
pool.close()
pool.join()

PREDICTIONS_ENSEMBLE = {}
ENSEMBLE_INPUTS = {}
for outer_fold in self.outer_folds:
    # take the subset of the rows that correspond to the outer_fold
    col_outer_fold = ensemble_outerfolds_cols[0]
    PREDICTIONS_outerfold = {}
    XS_outer_fold = {}
    YS_outer_fold = {}
    for fold in self.folds:
        self.PREDICTIONS[fold]['outer_fold_' + version] = self.PREDICTIONS[fold][col_outer_fold]
        PREDICTIONS_outerfold[fold] = self.PREDICTIONS[fold][
            self.PREDICTIONS[fold]['outer_fold_' + version] == float(outer_fold)]
        X = PREDICTIONS_outerfold[fold][['id', 'eid', 'instance'] + ensemble_preds_cols]
        X.set_index('id', inplace=True)
        XS_outer_fold[fold] = X
        y = PREDICTIONS_outerfold[fold][['id', 'eid', self.target]]
        y.set_index('id', inplace=True)
        YS_outer_fold[fold] = y
    ENSEMBLE_INPUTS[outer_fold] = [X, y]
    # Build ensemble model
    cv = InnerCV(model='ElasticNet', inner_splits=self.N_ensemble_CV_split,
                 n_iter=self.N_ensemble_iterations)
    model = cv.optimize_hyperparameters(XS_outer_fold['val'], YS_outer_fold['val'], scoring='r2')
    for fold in self.folds:
        X = XS_outer_fold[fold].drop(columns=['eid', 'instance'])
        PREDICTIONS_outerfold[fold]['pred_' + version] = model.predict(X)

def compute_ensemble_folds(ensemble_inputs):
    cv = InnerCV(model='ElasticNet', inner_splits=self.N_ensemble_CV_split, n_iter=self.N_ensemble_iterations)
    model = cv.optimize_hyperparameters(ensemble_inputs[0], ensemble_inputs[1], scoring='r2')
    return model
    



def _compute_ensemble_folds(ensemble_inputs):
    print('entered cef')
    print(ensemble_inputs)
    cv = InnerCV(model='ElasticNet', inner_splits=self.N_ensemble_CV_split, n_iter=self.N_ensemble_iterations)
    print('B\n\n')
    model = cv.optimize_hyperparameters(ensemble_inputs[0], ensemble_inputs[1], scoring='r2')
    print('C')
    return model