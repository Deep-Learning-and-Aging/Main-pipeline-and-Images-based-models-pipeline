import sys
from MI_Classes import PerformancesSurvival

# Default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('test')  # inner_fold
    sys.argv.append('eids')  # pred_type

# Compute results
self = PerformancesSurvival(target=sys.argv[1], fold=sys.argv[2], pred_type=sys.argv[3], debug_mode=True)
self.compute_CIs_for_modelstypes()

self.

# Exit
print('Done.')
sys.exit(0)

from MI_Classes import Metrics
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
from lifelines.utils import concordance_index
from sklearn.utils import resample




target = 'Age'
fold = 'test'
pred_type = 'instances'
PERFORMANCES = pd.read_csv('../data/PERFORMANCES_bestmodels_alphabetical_' + pred_type + '_' + target + '_' + fold +
                           '.csv')
PERFORMANCES.set_index('version', drop=False, inplace=True)
for inner_fold in ['all'] + [str(i) for i in range(10)]:
    for metric in ['C-Index', 'C-Index-difference']:
        for mode in self.modes:
            PERFORMANCES[metric + mode + '_' + inner_fold] = np.nan

Residuals = pd.read_csv('../data/RESIDUALS_instances_' + target + '_' + fold + '.csv')
data_surv = pd.read_csv('../data/data_survival.csv')
survival = pd.merge(data_surv[['id', 'FollowUpTime', 'Death']], Residuals, on='id')

n_bootstrap_iterations = 3

def _bootstrap_CI(data):
    results = []
    for i in range(n_bootstrap_iterations):
        data_i = resample(data, replace=True, n_samples=len(data.index))
        if len(data_i['Death'].unique()) == 2:
            results.append(concordance_index(data_i['Age'], -data_i['pred'], data_i['Death']))
        if len(results) > 0:
            results_mean = np.mean(results)
            results_std = np.std(results)
        else:
            results_mean = np.nan
            results_std = np.nan
    return results_mean, results_std

data_folds = pd.read_csv('../data/data-features_eids.csv', usecols=['eid', 'outer_fold'])
IDs = {}
SURV = {}
for i in range(10):
    IDs[i] = data_folds['eid'][data_folds['outer_fold'] == i].values
    SURV[i] = survival[survival['eid'].isin(IDs[i])]

models = [col.replace('res_', '') for col in Residuals.columns if 'res_' in col]
for k, model in enumerate(models):
    if k % 30 == 0:
        print('Computing CI for the ' + str(k) + 'th model out of ' + str(len(models)) + ' models.')
    # Load Performances dataframes
    PERFS = {}
    for mode in self.modes:
        PERFS[mode] = pd.read_csv('../data/Performances_instances_' + model + '_test' + mode + '.csv')
        PERFS[mode].set_index('outer_fold', drop=False, inplace=True)
        PERFS[mode]['C-Index'] = np.nan
        PERFS[mode]['C-Index-difference'] = np.nan
    df_model = survival[['FollowUpTime', 'Death', 'Age', 'res_' + model]].dropna()
    df_model.rename(columns={'res_' + model: 'pred'}, inplace=True)
    # Compute CI over all samples
    if len(df_model['Death'].unique()) == 2:
        ci_model = concordance_index(df_model['FollowUpTime'], -(df_model['Age'] - df_model['pred']), df_model['Death'])
        ci_age = concordance_index(df_model['FollowUpTime'], -df_model['Age'], df_model['Death'])
        ci_diff = ci_model - ci_age
        PERFS[''].loc['all', 'C-Index'] = ci_model
        PERFS[''].loc['all', 'C-Index-difference'] = ci_diff
        PERFORMANCES.loc[model, 'C-Index_all'] = ci_model
        PERFORMANCES.loc[model, 'C-Index-difference_all'] = ci_model
        _, ci_sd = _bootstrap_CI(df_model)
        PERFS['_sd'].loc['all', 'C-Index'] = ci_sd
        PERFS['_sd'].loc['all', 'C-Index-difference'] = ci_sd
        PERFORMANCES.loc[model, 'C-Index_sd_all'] = ci_sd
        PERFORMANCES.loc[model, 'C-Index-difference_sd_all'] = ci_sd
    # Compute CI over each fold
    for i in range(10):
        df_model_i = SURV[i][['FollowUpTime', 'Death', 'Age', 'res_' + model]].dropna()
        df_model_i.rename(columns={'res_' + model: 'pred'}, inplace=True)
        if len(df_model_i['Death'].unique()) == 2:
            ci_model_i = concordance_index(df_model_i['FollowUpTime'], -(df_model_i['Age'] - df_model_i['pred']),
                                           df_model_i['Death'])
            ci_age_i = concordance_index(df_model_i['FollowUpTime'], -df_model_i['Age'], df_model_i['Death'])
            ci_diff_i = ci_model_i - ci_age_i
            PERFS[''].loc[str(i), 'C-Index'] = ci_model_i
            PERFS[''].loc[str(i), 'C-Index-difference'] = ci_diff_i
            PERFORMANCES.loc[model, 'C-Index_' + str(i)] = ci_model_i
            PERFORMANCES.loc[model, 'C-Index-difference_' + str(i)] = ci_diff_i
            _, ci_i_sd = _bootstrap_CI(df_model_i)
            PERFS['_sd'].loc[str(i), 'C-Index'] = ci_i_sd
            PERFS['_sd'].loc[str(i), 'C-Index-difference'] = ci_i_sd
            PERFORMANCES.loc[model, 'C-Index_sd_' + str(i)] = ci_i_sd
            PERFORMANCES.loc[model, 'C-Index-difference_sd_' + str(i)] = ci_i_sd
    # Compute sd using all folds
    ci_str = round(PERFS[''][['C-Index', 'C-Index-difference']], 3).astype(str) + '+-' + \
             round(PERFS['_sd'][['C-Index', 'C-Index-difference']], 3).astype(str)
    PERFS['_str'][['C-Index', 'C-Index-difference']] = ci_str
    for col in ['C-Index', 'C-Index-difference']:
        cols = [col + '_str_' + str(i) for i in range(10)]
        # Fill model's performance matrix
        ci_std_lst = PERFS['_str'].loc['all', col].split('+-')
        ci_std_lst.insert(1, str(round(PERFS[''][col].iloc[1:].std(), 3)))
        ci_std_str = '+-'.join(ci_std_lst)
        PERFS['_str'].loc['all', col] = ci_std_str
        # Fill global performances matrix
        PERFORMANCES.loc[model, cols] = ci_str[col].values[1:]
        PERFORMANCES.loc[model, col + '_str_all'] = ci_std_str
    print(PERFS['_str'])
    # Save new performances
    for mode in self.modes:
        PERFS[mode].to_csv('../data/Performances_instances_withCI_' + model + '_test' + mode + '.csv')

# Save PERFORMANCES dataframes
PERFORMANCES.to_csv('../data/PERFORMANCES_withCI_bestmodels_alphabetical_' + pred_type + '_' + target + '_' + fold +
                    '.csv')
