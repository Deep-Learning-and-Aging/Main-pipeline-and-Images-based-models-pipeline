import sys
from MI_Classes import EnsemblesPredictions

# Options
regenerate_models = False  # False = Only compute ensemble model if it was not already computed

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('instances')  # pred_type

# Compute results
Ensembles_Predictions = EnsemblesPredictions(target=sys.argv[1], pred_type=sys.argv[2],
                                             regenerate_models=regenerate_models)
Ensembles_Predictions.load_data()
Ensembles_Predictions.generate_ensemble_predictions()
Ensembles_Predictions.save_predictions()

# Exit
print('Done.')
sys.exit(0)

# Prepare template to which each model will be appended
Predictions = self.Predictions[['eid'] + self.demographic_vars]
Predictions = Predictions.groupby('eid', as_index=True).mean()
Predictions.index.name = 'column_names'
Predictions['eid'] = Predictions.index.values
Predictions['instance'] = '*'
Predictions['id'] = Predictions['eid'].astype(str) + '_*'
self.Predictions_eids = Predictions.copy()
self.Predictions_eids['outer_fold'] = -1
for i in range(self.n_CV_outer_folds):
    Predictions_i = Predictions.copy()
    Predictions_i['outer_fold'] = i
    self.Predictions_eids = self.Predictions_eids.append(Predictions_i)

# Append each model one by one because the folds are different
print(str(len(self.pred_versions)) + ' models to compute.')
for pred_version in self.pred_versions:
    print("Computing results for version " + str(pred_version.replace('pred_', '')))
    of_version = pred_version.replace('pred_', 'outer_fold_')
    Predictions_version = self.Predictions[['eid', pred_version, of_version]]
    Predictions_version.rename(columns={of_version: 'outer_fold'}, inplace=True)
    # Use placeholder for NaN in outer_folds
    Predictions_version['outer_fold'][Predictions_version['outer_fold'].isna()] = -1
    Predictions_version_eids = Predictions_version.groupby(['eid', 'outer_fold'], as_index=False).mean()
    self.Predictions_eids = self.Predictions_eids.merge(Predictions_version_eids, on=['eid', 'outer_fold'])
    self.Predictions_eids[of_version] = self.Predictions_eids['outer_fold']
    self.Predictions_eids[of_version][self.Predictions_eids[of_version] == -1] = np.nan
    del Predictions_version
    _ = gc.collect
    
print("Computing results for version " + str(pred_version.replace('pred_', '')))
of_version = pred_version.replace('pred_', 'outer_fold_')
Predictions_version = self.Predictions[['eid', pred_version, of_version]]
Predictions_version.rename(columns={of_version: 'outer_fold'}, inplace=True)
# Use placeholder for NaN in outer_folds
Predictions_version['outer_fold'][Predictions_version['outer_fold'].isna()] = -1
Predictions_version_eids = Predictions_version.groupby(['eid', 'outer_fold'], as_index=False).mean()
self.Predictions_eids = self.Predictions_eids.merge(Predictions_version_eids, on=['eid', 'outer_fold'], how='outer')
self.Predictions_eids[of_version] = self.Predictions_eids['outer_fold']
self.Predictions_eids[of_version][self.Predictions_eids[of_version] == -1] = np.nan
del Predictions_version
_ = gc.collect