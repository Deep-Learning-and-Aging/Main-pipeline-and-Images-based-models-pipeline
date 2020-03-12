#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:00:53 2020

@author: Alan
"""

#load libraries, import functions and import parameters (nested import in the line below)
from MI_helpers import *

#default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age') #target
    sys.argv.append('B') #id_set

#read parameters from command
target = sys.argv[1]
id_set = sys.argv[2]

#set other parameters
main_metric_name = dict_main_metrics_names[target]
Perf_col_name = main_metric_name + '_all'

#compute the new prediction for each inner fold, and save them in the Predictions dataframes
#prepare variables
PERFORMANCES = {}
PREDICTIONS = {}
for fold in folds:
    PERFORMANCES[fold] = pd.read_csv(path_store + 'PERFORMANCES_withoutEnsembles_ranked_' + target + '_' + fold + '_' + id_set + '.csv').set_index('version', drop=False)
    PERFORMANCES[fold]['field_id'] = PERFORMANCES[fold]['field_id'].astype(str)
    PERFORMANCES[fold].index.name = 'columns_names'
    PREDICTIONS[fold] = pd.read_csv(path_store + 'PREDICTIONS_withoutEnsembles_' + target + '_' + fold + '_' + id_set + '.csv')

#Get list of distinct models without taking into account hyperparameters tuning
Performances = PERFORMANCES['val']
Performances['model'] = Performances['organ'] + '_' + Performances['field_id'] + '_' + Performances['view'] + '_' + Performances['transformation'] + '_' + Performances['architecture']
models = Performances['model'].unique()

#For each model, only keep the best performing hyperparameters
for model in models:
    Performances_model = Performances[Performances['model'] == model]
    best_version = Performances_model['version'][Performances_model[Perf_col_name] == Performances_model[Perf_col_name].max()].values[0]
    versions_to_drop = [version for version in Performances_model['version'].values if not version == best_version]
    #define columns from predictions to drop
    cols_to_drop = ['pred_' + version for version in versions_to_drop] + ['outer_fold_' + version for version in versions_to_drop]
    for fold in folds:
        PERFORMANCES[fold].drop(versions_to_drop, inplace=True)
        PREDICTIONS[fold].drop(cols_to_drop, inplace= True, axis=1)

#Save the files
for fold in folds:
    PREDICTIONS[fold].to_csv(path_store + 'PREDICTIONS_tuned_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
    PERFORMANCES[fold].to_csv(path_store + 'PERFORMANCES_tuned_ranked_' + target + '_' + fold + '_' + id_set + '.csv', index=False)
    Performances_alphabetical = PERFORMANCES[fold].sort_values(by='version')
    Performances_alphabetical.to_csv(path_store + 'PERFORMANCES_tuned_alphabetical_' + target + '_' + fold + '_' + id_set + '.csv', index=False)    

print('Done.')
sys.exit(0)
