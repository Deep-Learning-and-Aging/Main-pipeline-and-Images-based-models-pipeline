import sys
from MI_Classes import PredictionsEids

# Options
# Only compute the results for the first 1000 eids
debug_mode = False

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # fold

# Compute results
Predictions_Eids = PredictionsEids(target=sys.argv[1], fold=sys.argv[2], debug_mode=debug_mode)
Predictions_Eids.preprocessing()
Predictions_Eids.processing()
Predictions_Eids.postprocessing()
Predictions_Eids.save_predictions()

# Exit
print('Done.')
sys.exit(0)

self = Predictions_Eids
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load predictions
self.Predictions = pd.read_csv(
    self.path_data + 'PREDICTIONS_withoutEnsembles_instances_' + self.target + '_' + self.fold + '.csv')
self.Predictions.drop(columns=['id'], inplace=True)
self.Predictions['eid'] = self.Predictions['eid'].astype(str)
self.Predictions.index.name = 'column_names'
self.pred_versions = [col for col in self.Predictions.columns.values if 'pred_' in col]

# Prepare target values on instance 0 as a reference
target_0s = pd.read_csv(self.path_data + 'data-features_eids.csv', usecols=['eid', self.target])
target_0s['eid'] = target_0s['eid'].astype(str)
target_0s.set_index('eid', inplace=True)
target_0s = target_0s[self.target]
target_0s.name = 'target_0'
target_0s = target_0s[self.Predictions['eid'].unique()]
self.Predictions = self.Predictions.merge(target_0s, on='eid')

# Compute biological ages reported to target_0
correction = self.Predictions['target_0'] - self.Predictions[self.target]
pred = self.pred_versions[0]

# TODO Compute the biais of the predictions as a function of age
print('Generating residuals for model ' + pred.replace('pred_', ''))
df_model = self.Predictions[['Age', pred]] #change here
no_na_indices = [not b for b in df_model[pred].isna()]
df_model.dropna(inplace=True)
#if (len(df_model.index)) > 0:
age = df_model.loc[:, ['Age']]
res = df_model['Age'] - df_model[pred] #change here
regr = LinearRegression()
regr.fit(age, res)
self.Predictions[pred.replace('pred_', 'correction_')] = regr.predict(self.Predictions[['Age']])
self.Predictions['target_0_correction'] = regr.predict(self.Predictions[['target_0']])

#res_corrected = res - res_correction
#self.Residuals.loc[no_na_indices, 'pred_' + model] = res_corrected